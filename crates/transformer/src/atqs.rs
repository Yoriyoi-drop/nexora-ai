/// ATQS-compressed weight storage.
/// Always compiled. AWQ 4-bit compression/decompression always available.
#[derive(Debug, Clone)]
pub struct WeightsAtqs {
    pub token_embedding: Option<RawCompressedWeight>,
    pub lm_head: Option<RawCompressedWeight>,
    pub blocks: Vec<BlockCompressedWeights>,
}

#[derive(Debug, Clone)]
pub struct BlockCompressedWeights {
    pub wq: RawCompressedWeight,
    pub wk: RawCompressedWeight,
    pub wv: RawCompressedWeight,
    pub wo: RawCompressedWeight,
    pub w1: RawCompressedWeight,
    pub w2: RawCompressedWeight,
    pub w3: RawCompressedWeight,
    pub experts: Option<Vec<ExpertCompressedWeights>>,
}

#[derive(Debug, Clone)]
pub struct ExpertCompressedWeights {
    pub w1: RawCompressedWeight,
    pub w2: RawCompressedWeight,
    pub w3: RawCompressedWeight,
}

/// Mirror of `AWQQuantizedTensor` fields without the ATQS dependency.
#[derive(Debug, Clone)]
pub struct RawCompressedWeight {
    pub qdata: Vec<u8>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
    pub group_size: usize,
    pub bits: u8,
    pub original_rows: usize,
    pub original_cols: usize,
}

impl RawCompressedWeight {
    pub fn compress_ratio(&self) -> f32 {
        let original_bits = (self.original_rows * self.original_cols) as f32 * 32.0;
        let compressed_bits = self.qdata.len() as f32 * 8.0
            + self.scales.len() as f32 * 32.0
            + self.zero_points.len() as f32 * 32.0;
        original_bits / compressed_bits
    }
}

// ── Conversion from/to AWQQuantizedTensor ──

use ndarray::Array2;

use crate::{TransformerError, TransformerResult};
use crate::block::TransformerBlock;

impl From<nexora_atqs::awq::AWQQuantizedTensor> for RawCompressedWeight {
    fn from(t: nexora_atqs::awq::AWQQuantizedTensor) -> Self {
        Self {
            qdata: t.qdata,
            scales: t.scales,
            zero_points: t.zero_points,
            group_size: t.group_size,
            bits: t.bits,
            original_rows: t.original_shape[0],
            original_cols: t.original_shape[1],
        }
    }
}

impl From<RawCompressedWeight> for nexora_atqs::awq::AWQQuantizedTensor {
    fn from(r: RawCompressedWeight) -> Self {
        Self {
            qdata: r.qdata,
            scales: r.scales,
            zero_points: r.zero_points,
            group_size: r.group_size,
            bits: r.bits,
            original_shape: vec![r.original_rows, r.original_cols],
        }
    }
}

// ── Compression ──

fn compress_weight(
    engine: &nexora_atqs::awq::AWQEngine,
    weight: &Array2<f32>,
) -> RawCompressedWeight {
    let act_proxy = weight.map_axis(ndarray::Axis(0), |col| {
        col.iter().map(|v| v.abs()).sum::<f32>()
    });
    let act_proxy = act_proxy.mapv(|v| v.max(1.0));
    let saliency = engine.compute_saliency(weight, &act_proxy);
    let (scales, zps) = engine.find_optimal_scales(weight, &saliency);
    let q = engine.quantize(weight, &scales, &zps);
    RawCompressedWeight::from(q)
}

fn decompress_weight(raw: &RawCompressedWeight) -> Array2<f32> {
    let engine = nexora_atqs::awq::AWQEngine::new(
        nexora_atqs::awq::AWQConfig {
            group_size: raw.group_size,
            bits: raw.bits,
            ..Default::default()
        }
    );
    let q: nexora_atqs::awq::AWQQuantizedTensor = raw.clone().into();
    engine.dequantize(&q)
}

impl WeightsAtqs {
    /// Compress all f32 weights from a CausalLM into this cache.
    /// Uses AWQ 4-bit group quantization with column-norm activation proxy.
    pub fn compress(causal_lm: &super::CausalLM) -> TransformerResult<Self> {
        use nexora_atqs::awq::{AWQConfig, AWQEngine};
        let engine = AWQEngine::new(AWQConfig::w4a16());

        let token_embedding = causal_lm.token_embedding.as_ref()
            .map(|w| compress_weight(&engine, w));
        let lm_head = causal_lm.lm_head.as_ref()
            .map(|w| compress_weight(&engine, w));

        let blocks: TransformerResult<Vec<BlockCompressedWeights>> = causal_lm.blocks.iter().map(|block| {
            compress_block(&engine, block)
        }).collect();
        let blocks = blocks?;

        Ok(Self { token_embedding, lm_head, blocks })
    }

    /// Decompress all weights back into a CausalLM's weight fields.
    /// Sets w1/w2/w3/wq/wk/wv/wo fields from compressed data.
    pub fn restore(&self, causal_lm: &mut super::CausalLM) -> TransformerResult<()> {
        if let Some(ref te) = self.token_embedding {
            causal_lm.token_embedding = Some(decompress_weight(te));
        }
        if let Some(ref lm) = self.lm_head {
            causal_lm.lm_head = Some(decompress_weight(lm));
        }

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let b = &mut causal_lm.blocks[block_idx];
            b.attention.wq = Some(decompress_weight(&block.wq));
            b.attention.wk = Some(decompress_weight(&block.wk));
            b.attention.wv = Some(decompress_weight(&block.wv));
            b.attention.wo = Some(decompress_weight(&block.wo));
            b.ffn.w1 = Some(decompress_weight(&block.w1));
            b.ffn.w2 = Some(decompress_weight(&block.w2));
            b.ffn.w3 = Some(decompress_weight(&block.w3));

            if let Some(ref experts) = block.experts {
                if let Some(ref mut b_experts) = b.experts {
                    for (expert_idx, expert) in experts.iter().enumerate() {
                        if expert_idx < b_experts.len() {
                            b_experts[expert_idx].w1 = Some(decompress_weight(&expert.w1));
                            b_experts[expert_idx].w2 = Some(decompress_weight(&expert.w2));
                            b_experts[expert_idx].w3 = Some(decompress_weight(&expert.w3));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

fn compress_block(
    engine: &nexora_atqs::awq::AWQEngine,
    block: &TransformerBlock,
) -> TransformerResult<BlockCompressedWeights> {
    let wq = compress_weight(engine, block.attention.wq.as_ref().ok_or_else(||
        TransformerError::Implementation("missing wq".into())
    )?);
    let wk = compress_weight(engine, block.attention.wk.as_ref().ok_or_else(||
        TransformerError::Implementation("missing wk".into())
    )?);
    let wv = compress_weight(engine, block.attention.wv.as_ref().ok_or_else(||
        TransformerError::Implementation("missing wv".into())
    )?);
    let wo = compress_weight(engine, block.attention.wo.as_ref().ok_or_else(||
        TransformerError::Implementation("missing wo".into())
    )?);
    let w1 = compress_weight(engine, block.ffn.w1.as_ref().ok_or_else(||
        TransformerError::Implementation("missing w1".into())
    )?);
    let w2 = compress_weight(engine, block.ffn.w2.as_ref().ok_or_else(||
        TransformerError::Implementation("missing w2".into())
    )?);
    let w3 = compress_weight(engine, block.ffn.w3.as_ref().ok_or_else(||
        TransformerError::Implementation("missing w3".into())
    )?);

    let experts = block.experts.as_ref().map(|experts| {
        experts.iter().map(|e| {
            Ok(ExpertCompressedWeights {
                w1: compress_weight(engine, e.w1.as_ref().ok_or_else(||
                    TransformerError::Implementation("missing expert w1".into())
                )?),
                w2: compress_weight(engine, e.w2.as_ref().ok_or_else(||
                    TransformerError::Implementation("missing expert w2".into())
                )?),
                w3: compress_weight(engine, e.w3.as_ref().ok_or_else(||
                    TransformerError::Implementation("missing expert w3".into())
                )?),
            })
        }).collect::<TransformerResult<Vec<_>>>()
    }).transpose()?;

    Ok(BlockCompressedWeights { wq, wk, wv, wo, w1, w2, w3, experts })
}
