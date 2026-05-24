use std::sync::Mutex;
use std::time::Duration;

use ndarray::{Array1, Array2};
use rand::Rng;
use tracing::{info, warn};

use super::block::TransformerBlock;
use super::config::TransformerConfig;
use super::gqa::{KVCacheEntry, KVCacheProvider, CpuKVCache, PagedCacheReader};
use super::rope::RoPE;
use crate::{TransformerError, TransformerResult};

/// Hook for injecting processing between transformer layers.
pub trait LayerInjector: std::fmt::Debug + Send {
    fn after_layer(
        &mut self,
        layer_idx: usize,
        h: &mut Array2<f32>,
        pos: usize,
    ) -> TransformerResult<()>;
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_shape_fn(logits.len(), |i| {
        if sum > 0.0 {
            exps[i] / sum
        } else {
            1.0 / logits.len() as f32
        }
    })
}

#[cfg(feature = "gpu")]
fn sample_token_gpu(logits: &Array1<f32>, temperature: f32, top_k: usize, top_p: f32, seed: u64) -> u32 {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    let fallback = || sample_token(logits, temperature, top_k);
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(_) => return fallback(),
    };
    let shape = vec![1, logits.len()];
    let cpu = match ndarray::ArrayD::from_shape_vec(shape.clone(), logits.to_vec()) {
        Ok(c) => c,
        Err(_) => return fallback(),
    };
    let gpu = match GpuTensor::from_cpu(&cpu) {
        Ok(g) => g,
        Err(_) => return fallback(),
    };
    match ctx.gpu_sample(&gpu, temperature, u32::try_from(top_k).unwrap_or(u32::MAX), top_p, seed) {
        Ok(result) => {
            let raw = result.to_cpu_raw_bytes();
            u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]])
        }
        Err(_) => fallback(),
    }
}

pub fn sample_token(logits: &Array1<f32>, temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return u32::try_from(best).unwrap_or(u32::MAX);
    }
    let scaled: Array1<f32> = logits.mapv(|v| v / temperature);
    let probs = softmax(&scaled);

    if top_k > 0 && top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = top_k.min(probs.len());
        let sum_k: f32 = indices[..k].iter().map(|&i| probs[i]).sum();
        if sum_k <= 0.0 {
            return indices[0] as u32;
        }
        let r: f32 = rand::random::<f32>() * sum_k;
        let mut cum = 0.0;
        for &i in &indices[..k] {
            cum += probs[i];
            if cum >= r {
                return i as u32;
            }
        }
        return indices[k - 1] as u32;
    }

    let r: f32 = rand::random();
    let mut cum = 0.0;
    for i in 0..probs.len() {
        cum += probs[i];
        if cum >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub(crate) struct GpuWeights {
    pub token_embedding: nexora_autograd::gpu::GpuTensor,
    pub lm_head_t: nexora_autograd::gpu::GpuTensor,
    pub norm_weight: nexora_autograd::gpu::GpuTensor,
    pub block_weights: Vec<BlockGpuWeights>,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub(crate) struct BlockGpuWeights {
    pub attention_norm_weight: nexora_autograd::gpu::GpuTensor,
    pub ffn_norm_weight: nexora_autograd::gpu::GpuTensor,
    pub wq_t: nexora_autograd::gpu::GpuTensor,
    pub wk_t: nexora_autograd::gpu::GpuTensor,
    pub wv_t: nexora_autograd::gpu::GpuTensor,
    pub wo_t: nexora_autograd::gpu::GpuTensor,
    pub w1_t: nexora_autograd::gpu::GpuTensor,
    pub w2_t: nexora_autograd::gpu::GpuTensor,
    pub w3_t: nexora_autograd::gpu::GpuTensor,
}

#[derive(Debug)]
pub struct CausalLM {
    pub config: TransformerConfig,
    pub token_embedding: Array2<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub norm: super::rms_norm::RMSNorm,
    pub lm_head: Array2<f32>,
    pub rope: RoPE,
    pub precomputed_cos: Array1<f32>,
    pub precomputed_sin: Array1<f32>,
    /// Optional injectors that run after specified layers.
    /// Key = layer index, value = the injector.
    pub injectors: Vec<(usize, Mutex<Box<dyn LayerInjector>>)>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: parking_lot::Mutex<Option<GpuWeights>>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_cache: parking_lot::Mutex<Option<Vec<super::gqa::GpuKVCacheEntry>>>,
}

#[cfg(not(feature = "gpu"))]
impl Clone for CausalLM {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            blocks: self.blocks.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rope: self.rope.clone(),
            precomputed_cos: self.precomputed_cos.clone(),
            precomputed_sin: self.precomputed_sin.clone(),
            injectors: Vec::new(),
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for CausalLM {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            blocks: self.blocks.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rope: self.rope.clone(),
            precomputed_cos: self.precomputed_cos.clone(),
            precomputed_sin: self.precomputed_sin.clone(),
            injectors: Vec::new(),
            gpu_weights: parking_lot::Mutex::new(None),
            gpu_cache: parking_lot::Mutex::new(None),
        }
    }
}

impl CausalLM {
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (config.hidden_size as f32).sqrt().recip();

        let token_embedding =
            Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            });

        let blocks = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.hidden_size,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim(),
                    config.intermediate_size,
                    config.norm_eps,
                )
            })
            .collect();

        let norm = super::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps);

        let lm_head = Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });

        let rope = RoPE::new(config.head_dim(), config.max_seq_len, config.rope_theta);
        let (cos_full, sin_full) = rope.precompute_freqs_cis();
        let half = config.head_dim() / 2;
        let precomputed_cos = cos_full
            .into_shape(config.max_seq_len * half)
            .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));
        let precomputed_sin = sin_full
            .into_shape(config.max_seq_len * half)
            .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));

        Self {
            config,
            token_embedding,
            blocks,
            norm,
            lm_head,
            rope,
            precomputed_cos,
            precomputed_sin,
            injectors: Vec::new(),
            #[cfg(feature = "gpu")]
            gpu_weights: parking_lot::Mutex::new(None),
            #[cfg(feature = "gpu")]
            gpu_cache: parking_lot::Mutex::new(None),
        }
    }

    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
    ) -> TransformerResult<Array1<f32>> {
        #[cfg(feature = "gpu")]
        if let Ok(logits) = self.forward_gpu_with_cache_provider(input_ids, kv_cache) {
            return Ok(logits);
        }
        #[cfg(feature = "gpu")]
        tracing::warn!("GPU forward pass failed, falling back to CPU");

        // CPU fallback: extract inner Vec<KVCacheEntry> from CpuKVCache
        let entries = kv_cache
            .as_cpu_entries()
            .ok_or_else(|| TransformerError::Implementation("CPU forward requires CpuKVCache".to_string()))?;
        self.forward_cpu_impl(input_ids, entries)
    }

    /// Internal CPU forward implementation operating on `Vec<KVCacheEntry>`.
    fn forward_cpu_impl(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> TransformerResult<Array1<f32>> {
        let batch_size = 1;

        let mut h = Array2::zeros((batch_size, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = token_id as usize;
            if tid >= self.config.vocab_size {
                return Err(TransformerError::Implementation(
                    format!("Token ID {} out of range [0, {})", tid, self.config.vocab_size)
                ));
            }
            h.row_mut(0).assign(&self.token_embedding.row(tid));
        }

        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, kv_cache, layer_idx, &cos_slice, &sin_slice);

            // Run any registered injectors after this layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        TransformerError::Implementation(format!("Injector lock: {}", e))
                    })?;
                    guard.after_layer(layer_idx, &mut h, pos)?;
                }
            }
        }

        h = self.norm.forward(&h);

        Ok(h.row(0).dot(&self.lm_head.t()))
    }

    pub fn reset_cache(&self) -> CpuKVCache {
        CpuKVCache::new(self.config.num_layers)
    }

    /// Forward pass using a paged KV cache.
    /// `token_pos` is the current position being generated.
    pub fn forward_paged(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn PagedCacheReader,
        seq_id: u64,
    ) -> TransformerResult<Array1<f32>> {
        let batch_size = 1;
        let mut h = Array2::zeros((batch_size, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = token_id as usize;
            if tid >= self.config.vocab_size {
                return Err(TransformerError::Implementation(
                    format!("Token ID {} out of range [0, {})", tid, self.config.vocab_size)
                ));
            }
            h.row_mut(0).assign(&self.token_embedding.row(tid));
        }

        let token_pos = kv_cache.num_tokens(seq_id).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice = if token_pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![token_pos * half..(token_pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice = if token_pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![token_pos * half..(token_pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_paged(
                &h, kv_cache, seq_id, layer_idx, token_pos, &cos_slice, &sin_slice,
            );
        }

        h = self.norm.forward(&h);

        Ok(h.row(0).dot(&self.lm_head.t()))
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.token_embedding.len();
        count += self.lm_head.len();
        for block in &self.blocks {
            count += block.attention.wq.len();
            count += block.attention.wk.len();
            count += block.attention.wv.len();
            count += block.attention.wo.len();
            count += block.ffn.w1.len();
            count += block.ffn.w2.len();
            count += block.ffn.w3.len();
            count += block.attention_norm.weight.len();
            count += block.ffn_norm.weight.len();
        }
        count += self.norm.weight.len();
        count
    }

    /// Collect all 2D weight matrices for SEDC compression.
    /// Returns (weights, names, fusion_pairs).
    pub fn collect_weights_for_sedc(&self) -> (Vec<Array2<f32>>, Vec<String>, Vec<(usize, usize)>) {
        let mut weights: Vec<Array2<f32>> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        weights.push(self.token_embedding.clone());
        names.push("token_embedding".to_string());

        weights.push(self.lm_head.clone());
        names.push("lm_head".to_string());

        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{}", i);
            weights.push(block.attention.wq.clone());
            names.push(format!("{}.attention.wq", prefix));
            weights.push(block.attention.wk.clone());
            names.push(format!("{}.attention.wk", prefix));
            weights.push(block.attention.wv.clone());
            names.push(format!("{}.attention.wv", prefix));
            weights.push(block.attention.wo.clone());
            names.push(format!("{}.attention.wo", prefix));
            weights.push(block.ffn.w1.clone());
            names.push(format!("{}.ffn.w1", prefix));
            weights.push(block.ffn.w2.clone());
            names.push(format!("{}.ffn.w2", prefix));
            weights.push(block.ffn.w3.clone());
            names.push(format!("{}.ffn.w3", prefix));
        }

        // Fuse consecutive FFN layers in each block: w2 (intermediate→hidden) with next block's wq (hidden→head)
        let num_fuse = self.blocks.len().saturating_sub(1);
        let mut fuse_pairs = Vec::with_capacity(num_fuse);
        // Indices into weights vec: ffn.w2 at offset 2 + i*7, attention.wq at offset 2 + (i+1)*7
        let per_block = 7;
        let base = 2; // token_embedding + lm_head
        for i in 0..num_fuse {
            let w2_idx = base + i * per_block + 6; // ffn.w2 is 7th matrix in block (0-indexed: wq=0, wk=1, wv=2, wo=3, w1=4, w2=5, w3=6)
            let wq_idx = base + (i + 1) * per_block + 0;
            if w2_idx < weights.len() && wq_idx < weights.len() {
                fuse_pairs.push((w2_idx, wq_idx));
            }
        }

        (weights, names, fuse_pairs)
    }

    /// Run SEDC compression on all model weights with default config.
    /// Returns compression report as JSON if GPU is available.
    pub fn compress_sedc_default(&self) -> TransformerResult<Option<serde_json::Value>> {
        #[cfg(feature = "gpu")]
        {
            let config = nexora_autograd::gpu_sedc::SedcConfig::default();
            self.compress_sedc_json(&config)
        }
        #[cfg(not(feature = "gpu"))]
        {
            info!("SEDC requires GPU feature — disabled");
            Ok(None)
        }
    }

    /// Run SEDC compression on all model weights with custom config.
    /// Returns compression report as JSON if GPU is available.
    #[cfg(feature = "gpu")]
    pub fn compress_sedc_json(
        &self,
        config: &nexora_autograd::gpu_sedc::SedcConfig,
    ) -> TransformerResult<Option<serde_json::Value>> {
        use nexora_autograd::gpu_sedc::SedcCompressor;

        let (weights, names, fuse_pairs) = self.collect_weights_for_sedc();

        if nexora_autograd::gpu::GpuContext::global().is_err() {
            info!("No GPU context available — SEDC compression skipped");
            return Ok(None);
        }

        let compressor = SedcCompressor::new(config.clone());
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let fused = compressor
            .compress_model(&weights, &name_refs, &fuse_pairs)
            .map_err(|e| TransformerError::Implementation(format!("SEDC compress: {}", e)))?;

        let report = fused.report;
        info!(
            "SEDC compression complete: {}→{} params ({:.2}% ratio), {:.4} error",
            report.total_original,
            report.total_compressed,
            report.total_ratio * 100.0,
            report.mean_relative_error,
        );

        let layers_json: Vec<serde_json::Value> = report
            .layers
            .iter()
            .map(|l| {
                serde_json::json!({
                    "layer": l.layer,
                    "name": l.name,
                    "original_params": l.original_params,
                    "compressed_params": l.compressed_params,
                    "compression_ratio": l.compression_ratio,
                    "rank": l.rank,
                    "spectral_entropy": l.spectral_entropy,
                    "relative_error": l.relative_error,
                    "sparsity": l.sparsity,
                })
            })
            .collect();

        Ok(Some(serde_json::json!({
            "total_original": report.total_original,
            "total_compressed": report.total_compressed,
            "total_ratio": report.total_ratio,
            "mean_relative_error": report.mean_relative_error,
            "mean_sparsity": report.mean_sparsity,
            "layers": layers_json,
        })))
    }

    pub fn from_checkpoint(
        config: TransformerConfig,
        path: &str,
    ) -> crate::TransformerResult<Self> {
        let mut model = Self::new(config);
        let loaded = crate::safetensors::load_safetensors(path)?;
        let get_arr = |name: &str| -> crate::TransformerResult<ndarray::ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::TransformerError::Implementation(format!("Missing tensor: {}", name))
            })
        };
        fn to_fixed<D: ndarray::Dimension>(
            arr: ndarray::ArrayD<f32>,
            name: &str,
        ) -> crate::TransformerResult<ndarray::Array<f32, D>> {
            arr.into_dimensionality::<D>().map_err(|e| {
                crate::TransformerError::Implementation(format!(
                    "Shape mismatch for {}: {}",
                    name, e
                ))
            })
        }
        model.token_embedding =
            to_fixed::<ndarray::Ix2>(get_arr("token_embedding")?, "token_embedding")?.to_owned();
        model.lm_head = to_fixed::<ndarray::Ix2>(get_arr("lm_head")?, "lm_head")?.to_owned();
        model.norm.weight =
            to_fixed::<ndarray::Ix1>(get_arr("norm.weight")?, "norm.weight")?.to_owned();
        for (i, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{}.", i);
            let name = prefix.clone() + "attention_norm.weight";
            block.attention_norm.weight =
                to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn_norm.weight";
            block.ffn_norm.weight = to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wq";
            block.attention.wq = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wk";
            block.attention.wk = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wv";
            block.attention.wv = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "attention.wo";
            block.attention.wo = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn.w1";
            block.ffn.w1 = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn.w2";
            block.ffn.w2 = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
            let name = prefix.clone() + "ffn.w3";
            block.ffn.w3 = to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned();
        }
        model.injectors = Vec::new();
        Ok(model)
    }

    /// Pre-upload ALL model weights to GPU persistent buffer.
    /// Token embedding, lm_head (pre-transposed), norm weight, and per-block
    /// attention/FFN weights (all pre-transposed) are uploaded once and reused.
    /// Called once at the start of any GPU forward pass.
    #[cfg(feature = "gpu")]
    pub fn preupload_weights_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor, GpuError};
        use ndarray::ArrayD;

        let mut guard = self.gpu_weights.lock();
        if guard.is_some() {
            return Ok(());
        }

        let ctx = GpuContext::global()?;

        let mk_gpu = |arr: &Array2<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr.as_slice().ok_or_else(|| {
                GpuError::Unsupported("non-contiguous weight".into())
            })?.to_vec();
            Ok(GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(shape, data).map_err(|e| GpuError::Unsupported(e.to_string()))?
            )?)
        };

        let mk_gpu_1d = |arr: &Array1<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.len()];
            let data = arr.to_vec();
            Ok(GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(shape, data).map_err(|e| GpuError::Unsupported(e.to_string()))?
            )?)
        };

        let token_embedding = mk_gpu(&self.token_embedding)?;

        let lm_head_gpu = mk_gpu(&self.lm_head)?;
        let lm_head_t = ctx.transpose(&lm_head_gpu)?;

        let norm_weight = mk_gpu_1d(&self.norm.weight)?;

        let mut block_weights = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let attention_norm_weight = mk_gpu_1d(&block.attention_norm.weight)?;
            let ffn_norm_weight = mk_gpu_1d(&block.ffn_norm.weight)?;

            let wq = mk_gpu(&block.attention.wq)?;
            let wk = mk_gpu(&block.attention.wk)?;
            let wv = mk_gpu(&block.attention.wv)?;
            let wo = mk_gpu(&block.attention.wo)?;
            let w1 = mk_gpu(&block.ffn.w1)?;
            let w2 = mk_gpu(&block.ffn.w2)?;
            let w3 = mk_gpu(&block.ffn.w3)?;

            block_weights.push(BlockGpuWeights {
                attention_norm_weight,
                ffn_norm_weight,
                wq_t: ctx.transpose(&wq)?,
                wk_t: ctx.transpose(&wk)?,
                wv_t: ctx.transpose(&wv)?,
                wo_t: ctx.transpose(&wo)?,
                w1_t: ctx.transpose(&w1)?,
                w2_t: ctx.transpose(&w2)?,
                w3_t: ctx.transpose(&w3)?,
            });
        }

        *guard = Some(GpuWeights {
            token_embedding,
            lm_head_t,
            norm_weight,
            block_weights,
        });

        Ok(())
    }

    /// GPU forward with pre-uploaded cos/sin GPU tensors.
    /// cos/sin diupload SEKALI per step — tidak per-layer.
    /// cos_gpu, sin_gpu: [1, half] GPU tensors for the current position.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_precomputed_rope(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
        cos_gpu: &nexora_autograd::gpu::GpuTensor,
        sin_gpu: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gpu_weights = self.gpu_weights.lock();
        let gw = gpu_weights.as_ref().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("GPU weights not initialized after preupload".into())
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(gw.token_embedding.buffer(), offset, h.buffer(), 0, row_bytes);
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_rope_gpu(&h, kv_cache, layer_idx, cos_gpu, sin_gpu)?;
        }

        h = self.norm.forward_gpu(&h)?;

        let logits_gpu = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_cpu = logits_gpu.to_cpu();
        let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// GPU forward pass using GPU-resident KV cache.
    /// No CPU round-trip for K/V operations — eliminates O(n²) data transfer.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache(
        &self,
        input_ids: &[u32],
        cache: &mut [super::gqa::GpuKVCacheEntry],
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gpu_weights = self.gpu_weights.lock();
        let gw = gpu_weights.as_ref().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("GPU weights not initialized after preupload".into())
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(gw.token_embedding.buffer(), offset, h.buffer(), 0, row_bytes);
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        let pos = cache.first().map(|e| e.seq_len).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice: Array1<f32> = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice: Array1<f32> = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        // Upload cos/sin to GPU ONCE — reused across all layers
        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], cos_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?
        )?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope(&h, cache, layer_idx, &cos_gpu, &sin_gpu)?;
        }

        h = self.norm.forward_gpu(&h)?;

        let logits_gpu = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_cpu = logits_gpu.to_cpu();
        let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// GPU forward via [`KVCacheProvider`].
    /// Routes to `GpuKVCache` for true zero-copy GPU→GPU cache,
    /// or auto-converts `CpuKVCache` to GPU cache to avoid O(n²) round-trip.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache_provider(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global()?;

        // Pure GPU path — no CPU round-trip for K/V
        if let Some(gpu_entries) = kv_cache.as_gpu_entries() {
            return self.forward_gpu_with_cache(input_ids, gpu_entries);
        }

        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq = self.config.max_seq_len;

        // Get or create GPU-side cache mirroring the CpuKVCache
        let cpu_entries = kv_cache
            .as_cpu_entries()
            .ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported(
                "forward_gpu_with_cache_provider needs CpuKVCache or GpuKVCache".into(),
            ))?;

        let mut cache_guard = self.gpu_cache.lock();
        if cache_guard.is_none() {
            let mut gpu_entries: Vec<super::gqa::GpuKVCacheEntry> =
                (0..num_layers)
                    .map(|_| super::gqa::GpuKVCacheEntry::new(n_kv_heads, head_dim, max_seq))
                    .collect::<Result<Vec<_>, _>>()?;

            // Upload existing CPU K/V to GPU (one-time O(seq_len) cost)
            for layer_idx in 0..num_layers {
                if layer_idx < cpu_entries.len() && cpu_entries[layer_idx].seq_len() > 0 {
                    let ce = &cpu_entries[layer_idx];
                    let seq = ce.seq_len();
                    let k_cpu = ArrayD::from_shape_vec(
                        vec![seq, n_kv_heads, head_dim],
                        ce.k.clone(),
                    )
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
                    let v_cpu = ArrayD::from_shape_vec(
                        vec![seq, n_kv_heads, head_dim],
                        ce.v.clone(),
                    )
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;

                    let k_gpu = GpuTensor::from_cpu(&k_cpu)?;
                    let v_gpu = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * n_kv_heads * head_dim * 4) as u64;

                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(
                            k_gpu.buffer(),
                            0,
                            gpu_entries[layer_idx].k.buffer(),
                            0,
                            bytes,
                        );
                        enc.copy_buffer_to_buffer(
                            v_gpu.buffer(),
                            0,
                            gpu_entries[layer_idx].v.buffer(),
                            0,
                            bytes,
                        );
                        Ok(())
                    })?;

                    gpu_entries[layer_idx].seq_len = seq;
                }
            }

            *cache_guard = Some(gpu_entries);
        }

        let gpu_entries = cache_guard.as_mut().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("GPU cache not initialized".into())
        })?;

        // GPU forward with GPU-resident cache — no K/V round-trip
        let result = self.forward_gpu_with_cache(input_ids, gpu_entries)?;

        // Sync back: read only the NEW token's K/V from GPU → CPU cache (O(1) per step)
        let kv_elems = n_kv_heads * head_dim;
        let token_bytes = (kv_elems * 4) as u64;

        for layer_idx in 0..num_layers {
            let gpu_entry = &gpu_entries[layer_idx];
            let last_idx = gpu_entry.seq_len.wrapping_sub(1);
            let byte_off = (last_idx * kv_elems * 4) as u64;

            let staging_k = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv_sync_back_k"),
                size: token_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let staging_v = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv_sync_back_v"),
                size: token_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            ctx.batch_dispatch(|enc| {
                enc.copy_buffer_to_buffer(
                    gpu_entry.k.buffer(),
                    byte_off,
                    &staging_k,
                    0,
                    token_bytes,
                );
                enc.copy_buffer_to_buffer(
                    gpu_entry.v.buffer(),
                    byte_off,
                    &staging_v,
                    0,
                    token_bytes,
                );
                Ok(())
            })?;

            ctx.sync();

            let read_back = |staging: &wgpu::Buffer| -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
                let slice = staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(
                    wgpu::MapMode::Read,
                    move |r| {
                        let _ = tx.send(r);
                    },
                );
                ctx.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(30)),
                });
                rx.recv_timeout(Duration::from_secs(30))
                    .map_err(|_| nexora_autograd::gpu::GpuError::Timeout("KV cache readback timed out after 30s".into()))?
                    .map_err(|e| nexora_autograd::gpu::GpuError::Device(format!("map_async: {e:?}")))?;
                let mapped = slice.get_mapped_range();
                let out: Vec<f32> = mapped
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                staging.unmap();
                Ok(out)
            };

            let new_k = read_back(&staging_k)?;
            let new_v = read_back(&staging_v)?;

            // Append to CPU cache
            if layer_idx < cpu_entries.len() {
                let cpu_entry = &mut cpu_entries[layer_idx];
                cpu_entry.k.extend_from_slice(&new_k);
                cpu_entry.v.extend_from_slice(&new_v);
            }
        }

        Ok(result)
    }

    /// Generate using GPU with pre-uploaded cos/sin.
    /// cos/sin diupload SEKALI ke GPU di awal, lalu slice_tensor GPU-side copy per step.
    /// Eliminates N_layers × 2 CPU→GPU uploads per step → 0 uploads.
    #[cfg(feature = "gpu")]
    fn generate_gpu_impl(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;

        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => return self.generate(prompt_ids, max_tokens, temperature, top_k),
        };

        // Create GPU-resident cache for all layers (zero CPU round-trip)
        let mut gpu_cache: Vec<super::gqa::GpuKVCacheEntry> = match (0..self.config.num_layers)
            .map(|_| {
                super::gqa::GpuKVCacheEntry::new(
                    self.config.num_kv_heads,
                    self.config.head_dim(),
                    self.config.max_seq_len,
                )
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(cache) => cache,
            Err(_) => return self.generate(prompt_ids, max_tokens, temperature, top_k),
        };

        ctx.flush();

        // Prefill prompt tokens
        for &token_id in prompt_ids {
            if self.forward_gpu_with_cache(&[token_id], &mut gpu_cache).is_err() {
                let (tokens, _) = self.generate(prompt_ids, max_tokens, temperature, top_k);
                return (tokens, Vec::new());
            }
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            match self.forward_gpu_with_cache(&[last_id], &mut gpu_cache) {
                Ok(logits) => {
                    let next_id = sample_token_gpu(&logits, temperature, top_k, 0.0, 12345);
                    output.push(next_id);
                    if next_id == 0 { break; }
                    last_id = next_id;
                }
                Err(_) => {
                    let logits = self.forward(&[last_id], &mut self.reset_cache())
                        .unwrap_or_else(|_| Array1::zeros(self.config.vocab_size));
                    let next_id = sample_token(&logits, temperature, top_k);
                    output.push(next_id);
                    if next_id == 0 { break; }
                    last_id = next_id;
                }
            }
        }

        // One-time GPU readback: convert GpuKVCacheEntry -> KVCacheEntry
        let cpu_cache: Vec<KVCacheEntry> = gpu_cache
            .iter()
            .map(|entry| {
                let k_cpu: ndarray::ArrayD<f32> = entry.k_view().to_cpu();
                let v_cpu: ndarray::ArrayD<f32> = entry.v_view().to_cpu();
                let kv_dim = entry.kv_heads * entry.head_dim;
                KVCacheEntry {
                    k: k_cpu.into_iter().collect(),
                    v: v_cpu.into_iter().collect(),
                    kv_dim,
                }
            })
            .collect();

        (output, cpu_cache)
    }

    pub fn memory_bytes(&self) -> usize {
        self.parameter_count() * 4
    }

    /// GPU forward pass: returns logits on CPU.
    /// Handles token embedding, all transformer blocks, final norm + lm_head on GPU.
    /// Falls back to CPU if GPU is unavailable.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gpu_weights = self.gpu_weights.lock();
        let gw = gpu_weights.as_ref().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("GPU weights not initialized after preupload".into())
        })?;

        // 1. Token embedding: buffer-to-buffer copy of ONE row from pre-uploaded embedding tensor.
        //    No CPU→GPU upload — the entire embedding table lives on GPU permanently.
        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(gw.token_embedding.buffer(), offset, h.buffer(), 0, row_bytes);
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

        // 2. Get RoPE cos/sin for current position
        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice: Array1<f32> = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };
        let sin_slice: Array1<f32> = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            Array1::zeros(half)
        };

        // 3. Forward through all transformer blocks on GPU
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu(&h, kv_cache, layer_idx, &cos_slice, &sin_slice)?;
        }

        // 4. Final RMSNorm on GPU
        h = self.norm.forward_gpu(&h)?;

        // 5. LM head: matmul(h, lm_head^T) on GPU with pre-transposed weight.
        //    No lazy init — lm_head_t is pre-uploaded once.
        let logits_gpu = ctx.matmul(&h, &gw.lm_head_t)?;

        // 6. Download logits to CPU
        let logits_cpu = logits_gpu.to_cpu();
        let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// Batched GPU forward pass: processes multiple sequences in a single batch.
    /// All GPU dispatches are coalesced into one queue.submit() via batch mode.
    /// Returns one logit vector per sequence.
    /// Uses GPU-resident KV caches per sequence — no CPU round-trip for K/V.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched(
        &self,
        batch_tokens: &[u32],
        kv_caches: &mut [Vec<KVCacheEntry>],
    ) -> Result<Vec<Array1<f32>>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        use ndarray::ArrayD;

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq = self.config.max_seq_len;
        let gpu_weights = self.gpu_weights.lock();
        let gw = gpu_weights.as_ref().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("GPU weights not initialized after preupload".into())
        })?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        // Create per-sequence GPU KV caches (one Vec<GpuKVCacheEntry> per sequence)
        use super::gqa::GpuKVCacheEntry;
        let mut gpu_caches: Vec<Vec<GpuKVCacheEntry>> = (0..batch_size)
            .map(|_| {
                (0..num_layers)
                    .map(|_| GpuKVCacheEntry::new(n_kv_heads, head_dim, max_seq))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Sync existing CPU data to GPU caches (one-time O(seq) per sequence)
        for seq_idx in 0..batch_size {
            let cpu_cache = &kv_caches[seq_idx];
            for layer_idx in 0..num_layers {
                if layer_idx < cpu_cache.len() && cpu_cache[layer_idx].seq_len() > 0 {
                    let ce = &cpu_cache[layer_idx];
                    let seq = ce.seq_len();
                    let k_cpu = ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.k.clone())
                        .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
                    let v_cpu = ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.v.clone())
                        .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
                    let k_gpu = GpuTensor::from_cpu(&k_cpu)?;
                    let v_gpu = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * n_kv_heads * head_dim * 4) as u64;
                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(k_gpu.buffer(), 0, gpu_caches[seq_idx][layer_idx].k.buffer(), 0, bytes);
                        enc.copy_buffer_to_buffer(v_gpu.buffer(), 0, gpu_caches[seq_idx][layer_idx].v.buffer(), 0, bytes);
                        Ok(())
                    })?;
                    gpu_caches[seq_idx][layer_idx].seq_len = seq;
                }
            }
        }

        // Process all sequences using GPU-resident KV cache
        let mut all_logits = Vec::with_capacity(batch_size);
        for seq_idx in 0..batch_size {
            let token_id = batch_tokens[seq_idx];

            // Embedding: buffer-to-buffer copy of ONE row
            let tid = token_id as usize;
            let row_bytes = (hidden_size * 4) as u64;
            let offset = (tid * hidden_size * 4) as u64;
            let mut h = GpuTensor::zeros(&[1, hidden_size])?;
            ctx.batch_dispatch(|enc| {
                enc.copy_buffer_to_buffer(gw.token_embedding.buffer(), offset, h.buffer(), 0, row_bytes);
                Ok(())
            })?;

            // RoPE cos/sin for this sequence's current position
            let pos = gpu_caches[seq_idx].first().map(|e| e.seq_len).unwrap_or(0);
            let half = self.config.head_dim() / 2;
            let cos_slice = if pos * half < self.precomputed_cos.len() {
                self.precomputed_cos
                    .slice(ndarray::s![pos * half..(pos + 1) * half])
                    .to_owned()
            } else {
                Array1::zeros(half)
            };
            let sin_slice = if pos * half < self.precomputed_sin.len() {
                self.precomputed_sin
                    .slice(ndarray::s![pos * half..(pos + 1) * half])
                    .to_owned()
            } else {
                Array1::zeros(half)
            };

            // Forward through all blocks using GPU KV cache
            for (layer_idx, block) in self.blocks.iter().enumerate() {
                h = block.forward_gpu_with_cache(&h, &mut gpu_caches[seq_idx], layer_idx, &cos_slice, &sin_slice)?;
            }

            h = self.norm.forward_gpu(&h)?;

            let logits_gpu = ctx.matmul(&h, &gw.lm_head_t)?;

            let logits_cpu = logits_gpu.to_cpu();
            let logits_flat: Vec<f32> = logits_cpu.iter().copied().collect();
            all_logits.push(Array1::from_vec(logits_flat));
        }

        // Sync back new token's K/V from GPU cache to each CPU cache (O(1) per seq per layer)
        let kv_elems = n_kv_heads * head_dim;
        for seq_idx in 0..batch_size {
            let cpu_cache = &mut kv_caches[seq_idx];
            for layer_idx in 0..num_layers {
                let gpu_entry = &gpu_caches[seq_idx][layer_idx];
                if gpu_entry.seq_len == 0 {
                    continue;
                }
                let last_idx = gpu_entry.seq_len - 1;
                let byte_off = (last_idx * kv_elems * 4) as u64;
                let token_bytes = (kv_elems * 4) as u64;

                let staging_k = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("batched_sync_k"),
                    size: token_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let staging_v = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("batched_sync_v"),
                    size: token_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(gpu_entry.k.buffer(), byte_off, &staging_k, 0, token_bytes);
                    enc.copy_buffer_to_buffer(gpu_entry.v.buffer(), byte_off, &staging_v, 0, token_bytes);
                    Ok(())
                })?;

                ctx.sync();

                let read_back = |staging: &wgpu::Buffer| -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
                    let slice = staging.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
                    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(10)) });
                    rx.recv()
                        .map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("readback channel closed".into()))?
                        .map_err(|e| nexora_autograd::gpu::GpuError::Device(format!("map_async: {e:?}")))?;
                    let mapped = slice.get_mapped_range();
                    let out: Vec<f32> = mapped.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect();
                    staging.unmap();
                    Ok(out)
                };

                let new_k = read_back(&staging_k)?;
                let new_v = read_back(&staging_v)?;

                if layer_idx < cpu_cache.len() {
                    let entry = &mut cpu_cache[layer_idx];
                    entry.k.extend_from_slice(&new_k);
                    entry.v.extend_from_slice(&new_v);
                } else {
                    cpu_cache.push(KVCacheEntry {
                        k: new_k,
                        v: new_v,
                        kv_dim: kv_elems,
                    });
                }
            }
        }

        ctx.end_batch_mode();
        Ok(all_logits)
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, true)
    }

    pub fn generate_with_gpu(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        use_gpu: bool,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        let mut cache = self.reset_cache();

        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                return self.generate_gpu_impl(prompt_ids, max_tokens, temperature, top_k);
            }
            #[cfg(not(feature = "gpu"))]
            {
                warn!("use_gpu=true but GPU feature is not enabled — falling back to CPU");
            }
        }

        for &token_id in prompt_ids {
            if self.forward(&[token_id], &mut cache).is_err() {
                warn!("Forward failed during prefill for token {}", token_id);
            }
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            let logits = self.forward(&[last_id], &mut cache)
                .unwrap_or_else(|e| {
                    warn!("Forward failed during generation: {e}");
                    Array1::zeros(self.config.vocab_size)
                });
            let next_id = if use_gpu {
                #[cfg(feature = "gpu")]
                {
                    sample_token_gpu(&logits, temperature, top_k, 0.0, 12345)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    sample_token(&logits, temperature, top_k)
                }
            } else {
                sample_token(&logits, temperature, top_k)
            };
            output.push(next_id);
            if next_id == 0 {
                break;
            }
            last_id = next_id;
        }

        (output, cache.entries)
    }
}
