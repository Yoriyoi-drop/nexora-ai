use ndarray::Array2;

use crate::TransformerResult;

/// Group-wise dequantization entry for a single weight matrix.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized data (bit-packed per group)
    pub qdata: Vec<u8>,
    /// Per-group scale factors
    pub scales: Vec<f32>,
    /// Per-group zero points
    pub zero_points: Vec<i32>,
    /// Number of columns per group
    pub group_size: usize,
    /// Number of bits per element (4 for W4A16, 3 for W3A16)
    pub bits: u8,
    /// Original shape [rows, cols]
    pub original_shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Dequantize this tensor back to f32.
    pub fn dequantize(&self) -> Array2<f32> {
        let rows = self.original_shape[0];
        let cols = self.original_shape[1];
        let mask = (1 << self.bits) - 1;
        let entries_per_byte = 8 / self.bits as usize;
        let num_groups = self.scales.len();

        let mut result = Array2::zeros((rows, cols));
        let mut idx = 0;

        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(cols);
            let s = self.scales[g];
            let zp = self.zero_points[g] as f32;

            for i in 0..rows {
                for j in start..end {
                    let byte = self.qdata[idx / entries_per_byte];
                    let bit_offset = (idx % entries_per_byte) * self.bits as usize;
                    let q_val = ((byte >> bit_offset) & mask) as f32;
                    result[[i, j]] = (q_val - zp) * s;
                    idx += 1;
                }
            }
        }

        result
    }
}

/// Quantized weights for the entire model.
/// Stores all weight matrices as `QuantizedTensor` and dequantizes on-the-fly
/// during the forward pass (CPU) or pre-dequantized (GPU).
#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    pub token_embedding: Option<QuantizedTensor>,
    pub lm_head: Option<QuantizedTensor>,
    pub norm_weight: Option<QuantizedTensor>,
    pub blocks: Vec<BlockQuantizedWeights>,
}

#[derive(Debug, Clone)]
pub struct BlockQuantizedWeights {
    pub attention_norm_weight: Option<QuantizedTensor>,
    pub ffn_norm_weight: Option<QuantizedTensor>,
    pub wq: Option<QuantizedTensor>,
    pub wk: Option<QuantizedTensor>,
    pub wv: Option<QuantizedTensor>,
    pub wo: Option<QuantizedTensor>,
    pub w1: Option<QuantizedTensor>,
    pub w2: Option<QuantizedTensor>,
    pub w3: Option<QuantizedTensor>,
}

impl BlockQuantizedWeights {
    pub fn new() -> Self {
        Self {
            attention_norm_weight: None,
            ffn_norm_weight: None,
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            w1: None,
            w2: None,
            w3: None,
        }
    }
}

impl Default for BlockQuantizedWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-dequantized weights stored as f32 arrays.
/// Populated once from `QuantizedWeights` and reused for subsequent forward passes.
#[derive(Debug, Clone)]
pub struct DequantizedWeights {
    pub token_embedding: Array2<f32>,
    pub lm_head: Array2<f32>,
    pub blocks: Vec<DequantizedBlockWeights>,
}

#[derive(Debug, Clone)]
pub struct DequantizedBlockWeights {
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
}

impl QuantizedWeights {
    /// Dequantize all weights into f32 arrays for CPU usage.
    pub fn dequantize_all(
        &self,
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> TransformerResult<DequantizedWeights> {
        let token_embedding = self
            .token_embedding
            .as_ref()
            .map(|t| t.dequantize())
            .unwrap_or_else(|| Array2::zeros((vocab_size, hidden_size)));

        let lm_head = self
            .lm_head
            .as_ref()
            .map(|t| t.dequantize())
            .unwrap_or_else(|| Array2::zeros((vocab_size, hidden_size)));

        let mut blocks = Vec::with_capacity(num_layers);
        for bq in &self.blocks {
            blocks.push(DequantizedBlockWeights {
                wq: bq.wq.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
                wk: bq.wk.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
                wv: bq.wv.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
                wo: bq.wo.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
                w1: bq.w1.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
                w2: bq.w2.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
                w3: bq.w3.as_ref().map(|t| t.dequantize()).unwrap_or_else(|| {
                    Array2::zeros((0, 0))
                }),
            });
        }

        Ok(DequantizedWeights {
            token_embedding,
            lm_head,
            blocks,
        })
    }
}
