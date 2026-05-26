use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Duration;

use ndarray::{Array1, Array2};
use rand::Rng;
use tracing::{info, warn};

pub static GPU_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

use super::block::TransformerBlock;
use super::config::TransformerConfig;
use super::gqa::{CpuKVCache, KVCacheEntry, KVCacheProvider, PagedCacheReader};
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

    /// GPU-native injector: receives a mutable `GpuTensor` instead of CPU array.
    /// Default implementation falls back to CPU: reads GPU tensor, calls `after_layer`,
    /// and writes the result back.
    /// Override this for zero-copy GPU-native injection.
    #[cfg(feature = "gpu")]
    fn after_layer_gpu(
        &mut self,
        layer_idx: usize,
        h: &mut nexora_autograd::gpu::GpuTensor,
        pos: usize,
        ctx: &nexora_autograd::gpu::GpuContext,
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        let h_cpu = h.to_cpu()?;
        let dims = h_cpu.shape().to_vec();
        let mut h_2d = h_cpu.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(
                format!("injector reshape [{}x{}]: {}", dims[0], dims[1], e)
            ))?;
        self.after_layer(layer_idx, &mut h_2d, pos)
            .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(
                format!("injector after_layer: {}", e)
            ))?;
        let h_dyn = h_2d.into_dyn();
        *h = nexora_autograd::gpu::GpuTensor::from_cpu(&h_dyn)?;
        Ok(())
    }
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
fn sample_token_gpu(
    logits: &Array1<f32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> u32 {
    use nexora_autograd::gpu::{GpuContext, GpuTensor};
    let fallback = || sample_token(logits, temperature, top_k);
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("GPU context unavailable, falling back to CPU: {}", e);
            return fallback();
        }
    };
    let shape = vec![1, logits.len()];
    let cpu = match ndarray::ArrayD::from_shape_vec(shape.clone(), logits.to_vec()) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Failed to upload tensor to GPU: {}", e);
            return fallback();
        }
    };
    let gpu = match GpuTensor::from_cpu(&cpu) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!("Failed to upload tensor to GPU: {}", e);
            return fallback();
        }
    };
    match ctx.gpu_sample(
        &gpu,
        temperature,
        u32::try_from(top_k).unwrap_or(u32::MAX),
        top_p,
        seed,
    ) {
        Ok(result) => {
            let raw = match result.to_cpu_raw_bytes() {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("Failed to download GPU tensor to CPU: {}", e);
                    return fallback();
                }
            };
            u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]])
        }
        Err(e) => {
            tracing::warn!("GPU sampling failed: {}", e);
            fallback()
        }
    }
}

/// Sample a token from GPU-resident logits — zero CPU round-trip for logits.
/// Takes logits that are ALREADY on GPU, returns sampled token as GPU tensor (still on GPU).
#[cfg(feature = "gpu")]
fn sample_token_gpu_keep_gpu(
    logits_gpu: &nexora_autograd::gpu::GpuTensor,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
    use nexora_autograd::gpu::GpuContext;

    let ctx = GpuContext::global()?;
    ctx.gpu_sample(
        logits_gpu,
        temperature,
        u32::try_from(top_k).unwrap_or(u32::MAX),
        top_p,
        seed,
    )
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
    pub lm_head_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub lm_head_scale: f32,
    pub lm_head_f16: Option<nexora_autograd::gpu::GpuTensor>,
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
    pub wq_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub wk_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub wv_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub wo_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub w1_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub w2_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub w3_i8: Option<nexora_autograd::gpu::GpuTensor>,
    pub scale_wq: f32,
    pub scale_wk: f32,
    pub scale_wv: f32,
    pub scale_wo: f32,
    pub scale_w1: f32,
    pub scale_w2: f32,
    pub scale_w3: f32,
    pub wq_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub wk_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub wv_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub wo_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub w1_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub w2_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub w3_f16: Option<nexora_autograd::gpu::GpuTensor>,
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
    /// If true, keep intermediates on GPU to avoid repeated CPU readbacks.
    /// When set, `forward()` will use the GPU-resident path and only read back
    /// the final logits, eliminating ~4×N_layers synchronous sync transfers.
    pub keep_on_gpu: bool,
    /// If true, upload quantized (int8) weights for GPU matmul acceleration.
    /// Weight matrices are quantized during `preupload_weights_gpu()` using
    /// symmetric per-tensor int8 quantization. The int8 kernels dequantize
    /// on-the-fly in the WGSL shader, reducing GPU memory bandwidth ~4×.
    pub quantize_weights: bool,
    /// If true, store weights as packed F16 (2 f16 per u32, 2× memory savings).
    /// At the start of each forward pass, F16 weights are bulk-upconverted to
    /// F32 temp buffers for matmul computation. Ignored when `quantize_weights`
    /// is true (int8 takes priority).
    pub use_half_precision: bool,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<GpuWeights>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_cache: RwLock<Option<Vec<super::gqa::GpuKVCacheEntry>>>, // never held across await points
}

#[cfg(not(feature = "gpu"))]
impl Clone for CausalLM {
    fn clone(&self) -> Self {
        tracing::warn!("Cloning CausalLM — weight memory is NOT duplicated. Use Arc<CausalLM> for shared ownership.");
        Self {
            config: self.config.clone(),
            token_embedding: Array2::zeros((0, 0)),
            blocks: Vec::new(),
            norm: self.norm.clone(),
            lm_head: Array2::zeros((0, 0)),
            rope: self.rope.clone(),
            precomputed_cos: self.precomputed_cos.clone(),
            precomputed_sin: self.precomputed_sin.clone(),
            injectors: Vec::new(),
            keep_on_gpu: self.keep_on_gpu,
            quantize_weights: self.quantize_weights,
            use_half_precision: self.use_half_precision,
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
            keep_on_gpu: self.keep_on_gpu,
            quantize_weights: self.quantize_weights,
            use_half_precision: self.use_half_precision,
            gpu_weights: OnceLock::new(),
            gpu_cache: RwLock::new(None),
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
            keep_on_gpu: true,
            quantize_weights: false,
            use_half_precision: false,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_cache: RwLock::new(None),
        }
    }

    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
    ) -> TransformerResult<Array1<f32>> {
        #[cfg(feature = "gpu")]
        if self.keep_on_gpu {
            match self.forward_gpu_with_cache_provider(input_ids, kv_cache) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::error!("keep_on_gpu forward failed: {e}");
                    return Err(TransformerError::Implementation(format!(
                        "GPU forward failed: {e}"
                    )));
                }
            }
        }

        #[cfg(feature = "gpu")]
        match self.forward_gpu_with_cache_provider(input_ids, kv_cache) {
            Ok(logits) => return Ok(logits),
            Err(e) => {
                tracing::error!(error = %e, "GPU forward pass failed");
                GPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        let entries = kv_cache.as_cpu_entries().ok_or_else(|| {
            TransformerError::Implementation("CPU forward requires CpuKVCache".to_string())
        })?;
        self.forward_cpu_impl(input_ids, entries, 1)
    }

    /// Internal CPU forward implementation operating on `Vec<KVCacheEntry>`.
    /// `batch_size` controls how many stacked tokens from the end of `input_ids`
    /// are processed simultaneously. When `batch_size > 1`, all tokens are embedded
    /// into a stacked `[batch_size, hidden_size]` tensor and passed through the
    /// transformer blocks, returning logits for the first batch element.
    fn forward_cpu_impl(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
        batch_size: usize,
    ) -> TransformerResult<Array1<f32>> {
        let bs = if batch_size == 0 { 1 } else { batch_size };
        if bs > input_ids.len() {
            return Err(TransformerError::Implementation(format!(
                "batch_size {} exceeds input_ids len {}",
                bs,
                input_ids.len()
            )));
        }

        let mut h = Array2::zeros((bs, self.config.hidden_size));

        let start = input_ids.len() - bs;
        for b in 0..bs {
            let token_id = input_ids[start + b];
            let tid = token_id as usize;
            if tid >= self.config.vocab_size {
                return Err(TransformerError::Implementation(format!(
                    "Token ID {} out of range [0, {})",
                    tid, self.config.vocab_size
                )));
            }
            h.row_mut(b).assign(&self.token_embedding.row(tid));
        }

        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice: &[f32] = if pos * half < self.precomputed_cos.len() {
            &self.precomputed_cos.as_slice().unwrap_or(&[])[pos * half..(pos + 1) * half]
        } else {
            &[]
        };
        let sin_slice: &[f32] = if pos * half < self.precomputed_sin.len() {
            &self.precomputed_sin.as_slice().unwrap_or(&[])[pos * half..(pos + 1) * half]
        } else {
            &[]
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, kv_cache, layer_idx, cos_slice, sin_slice);

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

    pub fn set_keep_on_gpu(&mut self, keep: bool) {
        self.keep_on_gpu = keep;
    }

    pub fn clear_gpu_cache(&self) {
        #[cfg(feature = "gpu")]
        if let Ok(mut cache) = self.gpu_cache.write() {
            if let Some(gpu_entries) = cache.as_mut() {
                gpu_entries.clear();
            }
        }
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
                return Err(TransformerError::Implementation(format!(
                    "Token ID {} out of range [0, {})",
                    tid, self.config.vocab_size
                )));
            }
            h.row_mut(0).assign(&self.token_embedding.row(tid));
        }

        let token_pos = kv_cache.num_tokens(seq_id).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        let cos_slice: &[f32] = if token_pos * half < self.precomputed_cos.len() {
            &self.precomputed_cos.as_slice().unwrap_or(&[])
                [token_pos * half..(token_pos + 1) * half]
        } else {
            &[]
        };
        let sin_slice: &[f32] = if token_pos * half < self.precomputed_sin.len() {
            &self.precomputed_sin.as_slice().unwrap_or(&[])
                [token_pos * half..(token_pos + 1) * half]
        } else {
            &[]
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_paged(
                &h, kv_cache, seq_id, layer_idx, token_pos, cos_slice, sin_slice,
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
        model.keep_on_gpu = true;
        Ok(model)
    }

    /// Symmetric per-tensor int8 quantization.
    /// Returns (packed u32 data, scale) where each u32 holds 4 int8 values
    /// in little-endian byte order.
    #[cfg(feature = "gpu")]
    fn quantize_weight(weight: &Array2<f32>) -> (Vec<u32>, f32) {
        let numel = weight.len();
        let mut max_abs = 0.0f32;
        for &v in weight.iter() {
            max_abs = max_abs.max(v.abs());
        }
        if max_abs == 0.0 {
            return (vec![0u32; (numel + 3) / 4], 0.0);
        }
        let scale = max_abs / 127.0f32;
        let inv_scale = 1.0 / scale;

        let mut packed = Vec::with_capacity((numel + 3) / 4);
        let mut word: u32 = 0;
        for (i, &v) in weight.iter().enumerate() {
            let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i32 as i8 as u8;
            word |= (q as u32) << ((i % 4) * 8);
            if i % 4 == 3 {
                packed.push(word);
                word = 0;
            }
        }
        if numel % 4 != 0 {
            packed.push(word);
        }
        (packed, scale)
    }

    /// Pre-upload ALL model weights to GPU persistent buffer.
    /// Token embedding, lm_head (pre-transposed), norm weight, and per-block
    /// attention/FFN weights (all pre-transposed) are uploaded once and reused.
    /// Called once at the start of any GPU forward pass.
    ///
    /// If `self.quantize_weights` is true, also uploads int8 quantized variants
    /// of weight matrices. The int8 kernels dequantize on-the-fly in the WGSL
    /// shader, reducing GPU memory bandwidth ~4×.
    #[cfg(feature = "gpu")]
    pub fn preupload_weights_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuDtype, GpuError, GpuTensor};

        if self.gpu_weights.get().is_some() {
            return Ok(());
        }

        let ctx = GpuContext::global()?;

        let mk_gpu = |arr: &Array2<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr
                .as_slice()
                .ok_or_else(|| GpuError::Unsupported("non-contiguous weight".into()))?
                .to_vec();
            Ok(GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(shape, data)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?,
            )?)
        };

        let mk_gpu_1d = |arr: &Array1<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.len()];
            let data = arr.to_vec();
            Ok(GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(shape, data)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?,
            )?)
        };

        let token_embedding = mk_gpu(&self.token_embedding)?;

        let lm_head_gpu = mk_gpu(&self.lm_head)?;
        let lm_head_t = ctx.transpose(&lm_head_gpu)?;

        let (lm_head_i8, lm_head_scale) = if self.quantize_weights {
            let shape = vec![self.lm_head.shape()[0], self.lm_head.shape()[1]];
            let (packed, s) = Self::quantize_weight(&self.lm_head);
            (Some(GpuTensor::from_cpu_i8_packed(shape, &packed)?), s)
        } else {
            (None, 0.0)
        };

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

            let (wq_i8, scale_wq) = if self.quantize_weights {
                let shape = vec![block.attention.wq.shape()[0], block.attention.wq.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.attention.wq);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };
            let (wk_i8, scale_wk) = if self.quantize_weights {
                let shape = vec![block.attention.wk.shape()[0], block.attention.wk.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.attention.wk);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };
            let (wv_i8, scale_wv) = if self.quantize_weights {
                let shape = vec![block.attention.wv.shape()[0], block.attention.wv.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.attention.wv);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };
            let (wo_i8, scale_wo) = if self.quantize_weights {
                let shape = vec![block.attention.wo.shape()[0], block.attention.wo.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.attention.wo);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };
            let (w1_i8, scale_w1) = if self.quantize_weights {
                let shape = vec![block.ffn.w1.shape()[0], block.ffn.w1.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.ffn.w1);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };
            let (w2_i8, scale_w2) = if self.quantize_weights {
                let shape = vec![block.ffn.w2.shape()[0], block.ffn.w2.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.ffn.w2);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };
            let (w3_i8, scale_w3) = if self.quantize_weights {
                let shape = vec![block.ffn.w3.shape()[0], block.ffn.w3.shape()[1]];
                let (p, s) = Self::quantize_weight(&block.ffn.w3);
                (Some(GpuTensor::from_cpu_i8_packed(shape, &p)?), s)
            } else {
                (None, 0.0)
            };

            let use_f16 = self.use_half_precision && !self.quantize_weights;
            let wq_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&wq)?) } else { None };
            let wk_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&wk)?) } else { None };
            let wv_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&wv)?) } else { None };
            let wo_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&wo)?) } else { None };
            let w1_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&w1)?) } else { None };
            let w2_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&w2)?) } else { None };
            let w3_f16 = if use_f16 { Some(ctx.f32_to_f16_packed(&w3)?) } else { None };

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
                wq_i8,
                wk_i8,
                wv_i8,
                wo_i8,
                w1_i8,
                w2_i8,
                w3_i8,
                scale_wq,
                scale_wk,
                scale_wv,
                scale_wo,
                scale_w1,
                scale_w2,
                scale_w3,
                wq_f16,
                wk_f16,
                wv_f16,
                wo_f16,
                w1_f16,
                w2_f16,
                w3_f16,
            });
        }

        let lm_head_f16 = if self.use_half_precision && !self.quantize_weights {
            Some(ctx.f32_to_f16_packed(&lm_head_gpu)?)
        } else {
            None
        };

        self.gpu_weights
            .set(GpuWeights {
                token_embedding,
                lm_head_t,
                lm_head_i8,
                lm_head_scale,
                lm_head_f16,
                norm_weight,
                block_weights,
            })
            .map_err(|_| {
                nexora_autograd::gpu::GpuError::Unsupported("weights already set".into())
            })?;

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
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
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

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
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
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
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
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope(
                &h, cache, layer_idx, &cos_gpu, &sin_gpu,
            )?;

            // Run injectors (Echo-Net APSS, etc.) after each layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// GPU forward pass: returns logits as `GpuTensor` (still on GPU, zero readback).
    /// Same as `forward_gpu_with_cache` but skips `to_cpu()` on the logits.
    /// Use with `generate_gpu_keep_gpu` to eliminate PCIe round-trip per token.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_with_cache_keep_gpu(
        &self,
        input_ids: &[u32],
        cache: &mut [super::gqa::GpuKVCacheEntry],
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
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

        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], cos_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, half], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope(
                &h, cache, layer_idx, &cos_gpu, &sin_gpu,
            )?;

            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        Ok(logits)
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
        self.forward_gpu_with_cache_provider_inner(input_ids, kv_cache, false)
    }

    fn forward_gpu_with_cache_provider_inner(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
        needs_sync_back: bool,
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
        let cpu_entries = kv_cache.as_cpu_entries().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "forward_gpu_with_cache_provider needs CpuKVCache or GpuKVCache".into(),
            )
        })?;

        let mut cache_guard = self.gpu_cache.write().map_err(|e| {
            nexora_autograd::gpu::GpuError::Unsupported(format!("GPU cache lock poisoned: {}", e))
        })?;
        if cache_guard.is_none() {
            let mut gpu_entries: Vec<super::gqa::GpuKVCacheEntry> = (0..num_layers)
                .map(|_| super::gqa::GpuKVCacheEntry::new(&ctx, n_kv_heads, head_dim, max_seq, self.use_half_precision))
                .collect::<Result<Vec<_>, _>>()?;

            // Upload existing CPU K/V to GPU (one-time O(seq_len) cost)
            for layer_idx in 0..num_layers {
                if layer_idx < cpu_entries.len() && cpu_entries[layer_idx].seq_len() > 0 {
                    let ce = &cpu_entries[layer_idx];
                    let seq = ce.seq_len();
                    let k_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.k.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;
                    let v_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.v.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;

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
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU cache not initialized after write".into(),
            )
        })?;

        // GPU forward with GPU-resident cache — no K/V round-trip
        let result = self.forward_gpu_with_cache(input_ids, gpu_entries)?;

        // Sync back to CPU only when explicitly needed (CpuKVCache conversion case).
        // When cache is GpuKVCache, the caller gets entries via as_gpu_entries() and
        // the early return above prevents reaching here entirely.
        if needs_sync_back {
            let kv_elems = n_kv_heads * head_dim;

            for layer_idx in 0..num_layers {
                let gpu_entry = &gpu_entries[layer_idx];
                let last_idx = gpu_entry.seq_len.wrapping_sub(1);
                let elem_size: u64 = if gpu_entry.f16_storage { 2 } else { 4 };
                let byte_off = (last_idx * kv_elems) as u64 * elem_size;
                let token_bytes = (kv_elems as u64) * elem_size;

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

                let read_back =
                    |staging: &wgpu::Buffer, is_f16: bool| -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
                        let slice = staging.slice(..);
                        let (tx, rx) = std::sync::mpsc::channel();
                        slice.map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx.send(r);
                        });
                        // Called from spawn_blocking — sync device.poll(Wait) is acceptable here
                        ctx.device.poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(30)),
                        });
                        rx.recv_timeout(Duration::from_secs(30))
                            .map_err(|_| {
                                nexora_autograd::gpu::GpuError::Timeout(
                                    "KV cache readback timed out after 30s".into(),
                                )
                            })?
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Device(format!("map_async: {e:?}"))
                            })?;
                        let mapped = slice.get_mapped_range();
                        let out: Vec<f32> = if is_f16 {
                            mapped
                                .chunks_exact(2)
                                .map(|c| {
                                    let bits = u16::from_ne_bytes([c[0], c[1]]);
                                    half::f16::from_bits(bits).to_f32()
                                })
                                .collect()
                        } else {
                            mapped
                                .chunks_exact(4)
                                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        };
                        staging.unmap();
                        Ok(out)
                    };

                let new_k = read_back(&staging_k, gpu_entry.f16_storage)?;
                let new_v = read_back(&staging_v, gpu_entry.f16_storage)?;

                if layer_idx < cpu_entries.len() {
                    let cpu_entry = &mut cpu_entries[layer_idx];
                    cpu_entry.k.extend_from_slice(&new_k);
                    cpu_entry.v.extend_from_slice(&new_v);
                }
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
                    &ctx,
                    self.config.num_kv_heads,
                    self.config.head_dim(),
                    self.config.max_seq_len,
                    self.use_half_precision,
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
            if self
                .forward_gpu_with_cache(&[token_id], &mut gpu_cache)
                .is_err()
            {
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
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
                Err(_) => {
                    let logits = self
                        .forward(&[last_id], &mut self.reset_cache())
                        .unwrap_or_else(|_| Array1::zeros(self.config.vocab_size));
                    let next_id = sample_token(&logits, temperature, top_k);
                    output.push(next_id);
                    if next_id == 0 {
                        break;
                    }
                    last_id = next_id;
                }
            }
        }

        // One-time GPU readback: convert GpuKVCacheEntry -> KVCacheEntry
        let cpu_cache: Vec<KVCacheEntry> = gpu_cache
            .iter()
            .map(|entry| {
                let k_cpu: ndarray::ArrayD<f32> = entry.k_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback k failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
                let v_cpu: ndarray::ArrayD<f32> = entry.v_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback v failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
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

    /// GPU-RESIDENT generation loop: logits NEVER leave GPU during the loop.
    /// Forward → `forward_gpu_with_cache_keep_gpu` (returns GpuTensor, no to_cpu).
    /// Sample  → `sample_token_gpu_keep_gpu` (takes GpuTensor, returns GpuTensor).
    /// Readback: only 4 bytes per token (token ID), plus one bulk readback at the end.
    /// Eliminates ~`vocab_size * 4` bytes per token PCIe traffic (e.g. 128KB/token for 32K vocab).
    #[cfg(feature = "gpu")]
    fn generate_gpu_keep_gpu_impl(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;
        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(_) => return self.generate(prompt_ids, max_tokens, temperature, top_k),
        };
        // Prefill: feed all prompt tokens through the GPU-forward path.
        // We still use the CPU forward for prefill (batched matmul isn't ready
        // for GPU-only KV cache). After prefill, switch to the GPU-resident loop.
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut last_id = *prompt_ids.last().unwrap_or(&0);
        let mut gpu_cache: Vec<super::gqa::GpuKVCacheEntry> = match (0..self.config.num_layers)
            .map(|_| {
                super::gqa::GpuKVCacheEntry::new(
                    &ctx,
                    self.config.num_kv_heads,
                    self.config.head_dim(),
                    self.config.max_seq_len,
                    self.use_half_precision,
                )
            })
            .collect()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to create GPU KV cache: {e}, falling back to CPU generation");
                return self.generate(prompt_ids, max_tokens, temperature, top_k);
            }
        };

        // Prefill prompt tokens into GPU cache (one-by-one for correctness)
        for &id in prompt_ids {
            let _ = match self.forward_gpu_with_cache_keep_gpu(&[id], &mut gpu_cache) {
                Ok(_) => (),
                Err(e) => {
                    tracing::warn!("GPU prefill failed at token {id}: {e}, falling back");
                    return self.generate(prompt_ids, max_tokens, temperature, top_k);
                }
            };
        }

        let mut rng_seed = seed;

        // GPU-resident generation loop
        for _ in 0..max_tokens {
            let logits_gpu = match self.forward_gpu_with_cache_keep_gpu(&[last_id], &mut gpu_cache)
            {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("GPU forward failed at step: {e}, falling back");
                    break;
                }
            };

            let token_gpu = match sample_token_gpu_keep_gpu(
                &logits_gpu,
                temperature,
                top_k,
                top_p,
                rng_seed,
            ) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("GPU sampling failed: {e}, falling back");
                    break;
                }
            };

            // Read back only 4 bytes per token
            let raw = match token_gpu.to_cpu_raw_bytes() {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("GPU token readback failed: {e}, falling back");
                    break;
                }
            };
            let token = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);

            output.push(token);
            last_id = token;
            rng_seed = rng_seed.wrapping_add(1);

            if token == 2 {
                break;
            }
        }

        // One-time GPU readback: convert GpuKVCacheEntry -> KVCacheEntry
        let cpu_cache: Vec<KVCacheEntry> = gpu_cache
            .iter()
            .map(|entry| {
                let k_cpu = entry.k_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback k failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
                let v_cpu = entry.v_view().to_cpu().unwrap_or_else(|e| {
                    tracing::warn!("KV cache GPU readback v failed: {e}");
                    ndarray::ArrayD::zeros(vec![0])
                });
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
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
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

            // Run injectors (Echo-Net APSS) after each layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        // 4. Final RMSNorm on GPU
        h = self.norm.forward_gpu(&h)?;

        // 5. LM head: matmul(h, lm_head^T) on GPU with pre-transposed weight.
        //    No lazy init — lm_head_t is pre-uploaded once.
        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        // 6. Download logits to CPU (only at final return point)
        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// GPU forward pass that enforces GPU residency — never falls back to CPU.
    /// If GPU fails, the error is propagated instead of silently falling back.
    /// Preferred method when `keep_on_gpu` is set.
    #[cfg(feature = "gpu")]
    pub fn forward_keep_gpu(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = 1;
        let hidden_size = self.config.hidden_size;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let mut h = match input_ids.last() {
            Some(&token_id) => {
                let tid = token_id as usize;
                if tid >= self.config.vocab_size {
                    return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "Token ID {} out of range [0, {})",
                        tid, self.config.vocab_size
                    )));
                }
                let row_bytes = (hidden_size * 4) as u64;
                let offset = (tid * hidden_size * 4) as u64;
                let h = GpuTensor::zeros(&[batch_size, hidden_size])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        gw.token_embedding.buffer(),
                        offset,
                        h.buffer(),
                        0,
                        row_bytes,
                    );
                    Ok(())
                })?;
                h
            }
            None => GpuTensor::zeros(&[batch_size, hidden_size])?,
        };

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

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu(&h, kv_cache, layer_idx, &cos_slice, &sin_slice)?;

            // Run injectors (Echo-Net APSS) after each layer
            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!("injector lock: {}", e))
                    })?;
                    guard.after_layer_gpu(layer_idx, &mut h, pos, ctx)?;
                }
            }
        }

        h = self.norm.forward_gpu(&h)?;

        let logits = ctx.matmul(&h, &gw.lm_head_t)?;

        let logits_flat: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        Ok(Array1::from_vec(logits_flat))
    }

    /// Truly batched GPU forward pass: processes ALL sequences in parallel
    /// through batched matmuls. QKV projections, output projections, FFN,
    /// final norm, and LM head are all batched [batch_size, hidden_size]
    /// matmuls. Per-sequence RoPE + attention is unavoidable due to
    /// per-sequence KV caches with independent lengths.
    ///
    /// Single LM head readback at the end — no per-sequence CPU sync.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched(
        &self,
        batch_tokens: &[u32],
        kv_caches: &mut [Vec<KVCacheEntry>],
    ) -> Result<Vec<Array1<f32>>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let max_seq = self.config.max_seq_len;
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        // ── 0a. Bulk upconvert F16 weights → f32 temps if configured ──
        let f16_temps: Option<(Vec<[GpuTensor; 7]>, Option<GpuTensor>)> = if self.use_half_precision
            && !self.quantize_weights
        {
            let mut block_temps: Vec<[GpuTensor; 7]> = Vec::with_capacity(num_layers);
            for block_gw in &gw.block_weights {
                let wq = ctx.f16_packed_to_f32(block_gw.wq_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight wq missing f16 variant".into(),
                    )
                })?)?;
                let wk = ctx.f16_packed_to_f32(block_gw.wk_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight wk missing f16 variant".into(),
                    )
                })?)?;
                let wv = ctx.f16_packed_to_f32(block_gw.wv_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight wv missing f16 variant".into(),
                    )
                })?)?;
                let wo = ctx.f16_packed_to_f32(block_gw.wo_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight wo missing f16 variant".into(),
                    )
                })?)?;
                let w1 = ctx.f16_packed_to_f32(block_gw.w1_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight w1 missing f16 variant".into(),
                    )
                })?)?;
                let w2 = ctx.f16_packed_to_f32(block_gw.w2_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight w2 missing f16 variant".into(),
                    )
                })?)?;
                let w3 = ctx.f16_packed_to_f32(block_gw.w3_f16.as_ref().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported(
                        "use_half_precision: block weight w3 missing f16 variant".into(),
                    )
                })?)?;
                block_temps.push([wq, wk, wv, wo, w1, w2, w3]);
            }
            let lm_head = gw.lm_head_f16.as_ref().map(|f16| ctx.f16_packed_to_f32(f16)).transpose()?;
            Some((block_temps, lm_head))
        } else {
            None
        };

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        // Create per-sequence GPU KV caches (one Vec<GpuKVCacheEntry> per sequence)
        use super::gqa::GpuKVCacheEntry;
        let mut gpu_caches: Vec<Vec<GpuKVCacheEntry>> = (0..batch_size)
            .map(|_| {
                (0..num_layers)
                    .map(|_| GpuKVCacheEntry::new(&ctx, n_kv_heads, head_dim, max_seq, self.use_half_precision))
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
                    let k_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.k.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;
                    let v_cpu =
                        ArrayD::from_shape_vec(vec![seq, n_kv_heads, head_dim], ce.v.clone())
                            .map_err(|e| {
                                nexora_autograd::gpu::GpuError::Unsupported(e.to_string())
                            })?;
                    let k_gpu = GpuTensor::from_cpu(&k_cpu)?;
                    let v_gpu = GpuTensor::from_cpu(&v_cpu)?;
                    let bytes = (seq * n_kv_heads * head_dim * 4) as u64;
                    ctx.batch_dispatch(|enc| {
                        enc.copy_buffer_to_buffer(
                            k_gpu.buffer(),
                            0,
                            gpu_caches[seq_idx][layer_idx].k.buffer(),
                            0,
                            bytes,
                        );
                        enc.copy_buffer_to_buffer(
                            v_gpu.buffer(),
                            0,
                            gpu_caches[seq_idx][layer_idx].v.buffer(),
                            0,
                            bytes,
                        );
                        Ok(())
                    })?;
                    gpu_caches[seq_idx][layer_idx].seq_len = seq;
                }
            }
        }

        // ── 1. Batched embedding: copy ALL token rows into one [batch_size, hidden_size] tensor ──
        let row_bytes = (hidden_size * 4) as u64;
        let mut h = GpuTensor::zeros(&[batch_size, hidden_size])?;
        for &token_id in batch_tokens.iter() {
            let tid = token_id as usize;
            if tid >= vocab_size {
                return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                    "Token ID {} out of range [0, {})",
                    tid, vocab_size
                )));
            }
        }
        ctx.batch_dispatch(|enc| {
            for (seq_idx, &token_id) in batch_tokens.iter().enumerate() {
                let offset = (token_id as usize * hidden_size * 4) as u64;
                let dst_offset = (seq_idx * hidden_size * 4) as u64;
                enc.copy_buffer_to_buffer(
                    gw.token_embedding.buffer(),
                    offset,
                    h.buffer(),
                    dst_offset,
                    row_bytes,
                );
            }
            Ok(())
        })?;

        // ── 2. Forward through all blocks (batched QKV/FFN, per-sequence attention) ──
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_gw = &gw.block_weights[layer_idx];

            // Batched RMSNorm
            let normed = block.attention_norm.forward_gpu(&h)?;

            // Batched QKV projection: 3 matmuls with [batch_size, hidden_size]
            let q_proj = if let Some(ref wq_i8) = block_gw.wq_i8 {
                ctx.matmul_int8_weight(&normed, wq_i8, block_gw.scale_wq)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&normed, &temps[layer_idx][0])?
            } else {
                ctx.matmul(&normed, &block_gw.wq_t)?
            };
            let k_proj = if let Some(ref wk_i8) = block_gw.wk_i8 {
                ctx.matmul_int8_weight(&normed, wk_i8, block_gw.scale_wk)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&normed, &temps[layer_idx][1])?
            } else {
                ctx.matmul(&normed, &block_gw.wk_t)?
            };
            let v_proj = if let Some(ref wv_i8) = block_gw.wv_i8 {
                ctx.matmul_int8_weight(&normed, wv_i8, block_gw.scale_wv)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&normed, &temps[layer_idx][2])?
            } else {
                ctx.matmul(&normed, &block_gw.wv_t)?
            };

            // Batch-upload cos/sin for ALL sequences at once: [batch_size, half]
            let mut cos_flat = Vec::with_capacity(batch_size * half);
            let mut sin_flat = Vec::with_capacity(batch_size * half);
            for seq_idx in 0..batch_size {
                let pos = gpu_caches[seq_idx].first().map(|e| e.seq_len).unwrap_or(0);
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
                cos_flat.extend_from_slice(cos_slice.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("cos_slice not contiguous".into())
                })?);
                sin_flat.extend_from_slice(sin_slice.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("sin_slice not contiguous".into())
                })?);
            }
            let cos_gpu_batch = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![batch_size, half], cos_flat)
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
            )?;
            let sin_gpu_batch = GpuTensor::from_cpu(
                &ArrayD::from_shape_vec(vec![batch_size, half], sin_flat)
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
            )?;

            let q_dim = n_heads * head_dim;
            let kv_dim_total = n_kv_heads * head_dim;
            let mut attn_rows = Vec::with_capacity(batch_size);

            for seq_idx in 0..batch_size {
                // Extract this sequence's Q/K/V rows via buffer copy
                let q_row = GpuTensor::zeros(&[1, q_dim])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        q_proj.buffer(),
                        (seq_idx * q_dim * 4) as u64,
                        q_row.buffer(),
                        0,
                        (q_dim * 4) as u64,
                    );
                    Ok(())
                })?;

                let k_row = GpuTensor::zeros(&[1, kv_dim_total])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        k_proj.buffer(),
                        (seq_idx * kv_dim_total * 4) as u64,
                        k_row.buffer(),
                        0,
                        (kv_dim_total * 4) as u64,
                    );
                    Ok(())
                })?;

                let v_row = GpuTensor::zeros(&[1, kv_dim_total])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        v_proj.buffer(),
                        (seq_idx * kv_dim_total * 4) as u64,
                        v_row.buffer(),
                        0,
                        (kv_dim_total * 4) as u64,
                    );
                    Ok(())
                })?;

                // Copy this sequence's cos/sin row from batched GPU tensor
                let row_cos = GpuTensor::zeros(&[1, half])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        cos_gpu_batch.buffer(),
                        (seq_idx * half * 4) as u64,
                        row_cos.buffer(),
                        0,
                        (half * 4) as u64,
                    );
                    Ok(())
                })?;
                let row_sin = GpuTensor::zeros(&[1, half])?;
                ctx.batch_dispatch(|enc| {
                    enc.copy_buffer_to_buffer(
                        sin_gpu_batch.buffer(),
                        (seq_idx * half * 4) as u64,
                        row_sin.buffer(),
                        0,
                        (half * 4) as u64,
                    );
                    Ok(())
                })?;

                // RoPE (per-sequence because positions differ)
                let q_rotated =
                    ctx.rotary_embedding(&q_row, &row_cos, &row_sin, head_dim as u32)?;
                let k_rotated =
                    ctx.rotary_embedding(&k_row, &row_cos, &row_sin, head_dim as u32)?;

                // Append K/V to per-sequence GPU cache
                let k_3d = k_rotated.reshape(vec![1, n_kv_heads, head_dim])?;
                let v_3d = v_row.reshape(vec![1, n_kv_heads, head_dim])?;
                gpu_caches[seq_idx][layer_idx].append(&ctx, &k_3d, &v_3d)?;

                // Fused attention with cached K/V
                let total_seq = gpu_caches[seq_idx][layer_idx].seq_len;
                let (k_rep, v_rep) =
                    gpu_caches[seq_idx][layer_idx].get_repeated_kv(&ctx, n_heads as u32)?;
                let q_4d = q_rotated.reshape(vec![1, n_heads, 1, head_dim])?;
                let k_4d = k_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;
                let v_4d = v_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;

                let attn_out = ctx.fused_attention(&q_4d, &k_4d, &v_4d, scale, false)?;
                attn_rows.push(attn_out.reshape(vec![1, n_heads * head_dim])?);
            }

            // Stack per-sequence attention outputs → [batch_size, hidden_size]
            let attn_concat = GpuTensor::zeros(&[batch_size, hidden_size])?;
            ctx.batch_dispatch(|enc| {
                for (seq_idx, row) in attn_rows.iter().enumerate() {
                    enc.copy_buffer_to_buffer(
                        row.buffer(),
                        0,
                        attn_concat.buffer(),
                        (seq_idx * hidden_size * 4) as u64,
                        (hidden_size * 4) as u64,
                    );
                }
                Ok(())
            })?;

            // Batched output projection + residual
            let attn_proj = if let Some(ref wo_i8) = block_gw.wo_i8 {
                ctx.matmul_int8_weight(&attn_concat, wo_i8, block_gw.scale_wo)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&attn_concat, &temps[layer_idx][3])?
            } else {
                ctx.matmul(&attn_concat, &block_gw.wo_t)?
            };
            h = ctx.add(&h, &attn_proj)?;

            // Batched FFN sub-block
            let normed_ffn = block.ffn_norm.forward_gpu(&h)?;
            let ffn_gate = if let Some(ref w1_i8) = block_gw.w1_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w1_i8, block_gw.scale_w1)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&normed_ffn, &temps[layer_idx][4])?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w1_t)?
            };
            let ffn_hidden = if let Some(ref w3_i8) = block_gw.w3_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w3_i8, block_gw.scale_w3)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&normed_ffn, &temps[layer_idx][6])?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w3_t)?
            };
            let ffn_silu_mul = ctx.mul(&ctx.silu(&ffn_gate)?, &ffn_hidden)?;
            let ffn_out = if let Some(ref w2_i8) = block_gw.w2_i8 {
                ctx.matmul_int8_weight(&ffn_silu_mul, w2_i8, block_gw.scale_w2)?
            } else if let Some((ref temps, _)) = f16_temps {
                ctx.matmul(&ffn_silu_mul, &temps[layer_idx][5])?
            } else {
                ctx.matmul(&ffn_silu_mul, &block_gw.w2_t)?
            };
            h = ctx.add(&h, &ffn_out)?;
        }

        // ── 3. Final norm + SINGLE LM head matmul (batched) ──
        h = self.norm.forward_gpu(&h)?;
        let logits = if let Some(ref lm_head_i8) = gw.lm_head_i8 {
            ctx.matmul_int8_weight(&h, lm_head_i8, gw.lm_head_scale)?
        } else if let Some((_, Some(ref lm_head_f32))) = f16_temps {
            ctx.matmul(&h, lm_head_f32)?
        } else {
            ctx.matmul(&h, &gw.lm_head_t)?
        };

        // ── 4. SINGLE readback — split into per-sequence logits ──
        let logits_data: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        let all_logits: Vec<Array1<f32>> = logits_data
            .chunks(vocab_size)
            .map(|chunk| Array1::from_vec(chunk.to_vec()))
            .collect();

        // ── 5. Sync back new token's K/V from GPU cache to each CPU cache ──
        let kv_elems = n_kv_heads * head_dim;
        for seq_idx in 0..batch_size {
            let cpu_cache = &mut kv_caches[seq_idx];
            for layer_idx in 0..num_layers {
                let gpu_entry = &gpu_caches[seq_idx][layer_idx];
                if gpu_entry.seq_len == 0 {
                    continue;
                }
                let last_idx = gpu_entry.seq_len - 1;
                let elem_size: u64 = if gpu_entry.f16_storage { 2 } else { 4 };
                let byte_off = (last_idx * kv_elems) as u64 * elem_size;
                let token_bytes = (kv_elems as u64) * elem_size;

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

                let read_back =
                    |staging: &wgpu::Buffer, is_f16: bool| -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
                        let slice = staging.slice(..);
                        let (tx, rx) = std::sync::mpsc::channel();
                        slice.map_async(wgpu::MapMode::Read, move |r| {
                            let _ = tx.send(r);
                        });
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(10);
                        let map_result = loop {
                            ctx.device.poll(wgpu::PollType::Wait {
                                submission_index: None,
                                timeout: Some(std::time::Duration::from_millis(100)),
                            });
                            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                                Ok(result) => break result,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    if std::time::Instant::now() > deadline {
                                        break Err(wgpu::BufferAsyncError);
                                    }
                                    continue;
                                }
                                Err(_) => break Err(wgpu::BufferAsyncError),
                            }
                        };
                        let map_result = map_result.map_err(|e| {
                            nexora_autograd::gpu::GpuError::Device(format!("map_async: {e:?}"))
                        })?;
                        let mapped = slice.get_mapped_range();
                        let out: Vec<f32> = if is_f16 {
                            mapped
                                .chunks_exact(2)
                                .map(|c| {
                                    let bits = u16::from_ne_bytes([c[0], c[1]]);
                                    half::f16::from_bits(bits).to_f32()
                                })
                                .collect()
                        } else {
                            mapped
                                .chunks_exact(4)
                                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        };
                        staging.unmap();
                        Ok(out)
                    };

                let new_k = read_back(&staging_k, gpu_entry.f16_storage)?;
                let new_v = read_back(&staging_v, gpu_entry.f16_storage)?;

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
            let logits = self.forward(&[last_id], &mut cache).unwrap_or_else(|e| {
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
            // EOS token is 2 in MiniTokenizer; also stop on padding (0)
            if next_id == 0 || next_id == 2 {
                break;
            }
            last_id = next_id;
        }

        (output, cache.entries)
    }

    /// GPU-resident generation: logits stay GPU-native in the loop.
    /// Only 4 bytes per token read back (token ID), not full vocab_size vector.
    /// Falls back to `generate_with_gpu` if GPU unavailable or on error.
    #[cfg(feature = "gpu")]
    pub fn generate_with_gpu_keep_gpu(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        use nexora_autograd::gpu::GpuContext;

        match GpuContext::global() {
            Ok(_) => self.generate_gpu_keep_gpu_impl(prompt_ids, max_tokens, temperature, top_k, top_p, seed),
            Err(_) => {
                tracing::warn!("GPU context unavailable for GPU-resident generation, falling back to generate_with_gpu");
                self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, true)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransformerConfig;

    fn small_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 64,
            intermediate_size: 64,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
        }
    }

    fn small_model() -> CausalLM {
        CausalLM::new(small_config())
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = Array1::from(vec![2.0, 1.0, 0.1, -1.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_softmax_uniform_when_all_equal() {
        let logits = Array1::from(vec![5.0; 10]);
        let probs = softmax(&logits);
        for &p in probs.iter() {
            assert!((p - 0.1).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_nan_input() {
        let logits = Array1::from(vec![f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_token_greedy() {
        let logits = Array1::from(vec![1.0, 5.0, 2.0, 0.5, 3.0]);
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 1); // highest logit at index 1
    }

    #[test]
    fn test_sample_token_greedy_last() {
        let logits = Array1::from(vec![0.1, 0.2, 0.3, 10.0]);
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_sample_token_temperature_zero_is_greedy() {
        let logits = Array1::from(vec![0.0, 0.0, 100.0, 0.0]);
        let token = sample_token(&logits, 0.0, 0);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_token_top_k() {
        let logits = Array1::from(vec![0.0, 10.0, 0.0, 9.0, 0.0]);
        // With top_k=2, should sample from {1, 3}
        for _ in 0..20 {
            let token = sample_token(&logits, 1.0, 2);
            assert!(
                token == 1 || token == 3,
                "top_k=2 should only pick from top 2, got {}",
                token
            );
        }
    }

    #[test]
    fn test_causal_lm_new_shapes() {
        let cfg = small_config();
        let model = small_model();
        assert_eq!(model.token_embedding.dim(), (100, 32));
        assert_eq!(model.lm_head.dim(), (100, 32));
        assert_eq!(model.norm.weight.len(), 32);
        assert_eq!(model.blocks.len(), 2);
        assert_eq!(
            model.precomputed_cos.len(),
            cfg.max_seq_len * cfg.head_dim() / 2
        );
        assert_eq!(
            model.precomputed_sin.len(),
            cfg.max_seq_len * cfg.head_dim() / 2
        );
    }

    #[test]
    fn test_causal_lm_forward_shape() {
        let mut model = small_model();
        model.set_keep_on_gpu(false);
        let mut cache = model.reset_cache();
        let logits = model.forward(&[0, 1], &mut cache).unwrap();
        assert_eq!(logits.len(), 100);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_causal_lm_forward_single_token() {
        let mut model = small_model();
        model.set_keep_on_gpu(false);
        let mut cache = model.reset_cache();
        let logits = model.forward(&[5], &mut cache).unwrap();
        assert_eq!(logits.len(), 100);
    }

    #[test]
    fn test_causal_lm_forward_out_of_range_token() {
        let model = small_model();
        let mut cache = model.reset_cache();
        let result = model.forward(&[999], &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_returns_tokens() {
        let model = small_model();
        let (tokens, cache) = model.generate_with_gpu(&[0, 1], 10, 1.0, 0, false);
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 10);
        if !cache.is_empty() {
            assert!(!cache[0].k.is_empty());
        }
    }

    #[test]
    fn test_generate_single_token() {
        let model = small_model();
        let (tokens, _) = model.generate_with_gpu(&[5], 5, 1.0, 0, false);
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 5);
    }

    #[test]
    fn test_parameter_count() {
        let model = small_model();
        let count = model.parameter_count();
        let cfg = small_config();
        let expected = cfg.parameter_count();
        assert_eq!(count, expected);
    }

    #[test]
    fn test_reset_cache() {
        let model = small_model();
        let cache = model.reset_cache();
        assert_eq!(cache.num_layers(), 0);
    }

    #[test]
    fn test_memory_bytes() {
        let model = small_model();
        let bytes = model.memory_bytes();
        assert_eq!(bytes, model.parameter_count() * 4);
    }

    #[test]
    fn test_keep_on_gpu_default() {
        let model = small_model();
        assert!(model.keep_on_gpu);
    }

    #[test]
    fn test_set_keep_on_gpu() {
        let mut model = small_model();
        model.set_keep_on_gpu(false);
        assert!(!model.keep_on_gpu);
    }

    #[test]
    fn test_forward_paged_shape() {
        use crate::gqa::PagedCacheReader;

        struct DummyPagedCache;
        impl PagedCacheReader for DummyPagedCache {
            fn read(
                &self,
                _seq_id: u64,
                _layer: usize,
                _token_pos: usize,
            ) -> Option<(Vec<f32>, Vec<f32>)> {
                None
            }
            fn num_tokens(&self, _seq_id: u64) -> Option<usize> {
                Some(1)
            }
            fn append(
                &mut self,
                _seq_id: u64,
                _layer: usize,
                _token_pos: usize,
                _k: &[f32],
                _v: &[f32],
            ) {
            }
        }

        let model = small_model();
        let mut cache = DummyPagedCache;
        let result = model.forward_paged(&[1, 2], &mut cache, 0);
        assert!(result.is_ok());
        if let Ok(logits) = result {
            assert_eq!(logits.len(), 100);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_forward_gpu_int8_matches_cpu() {
        use crate::GpuKVCacheEntry;
        use nexora_autograd::gpu::GpuContext;

        let _ctx = GpuContext::init().expect("GPU context init failed");

        let model = small_model();
        let mut cache = model.reset_cache();
        let cpu_logits = model.forward(&[0, 1], &mut cache).unwrap();

        let mut model_gpu = small_model();
        model_gpu.quantize_weights = true;
        let mut gpu_cache: Vec<GpuKVCacheEntry> = (0..model_gpu.config.num_layers)
            .map(|_| {
                GpuKVCacheEntry::new(
                    &GpuContext::global().unwrap(),
                    model_gpu.config.num_kv_heads,
                    model_gpu.config.head_dim(),
                    model_gpu.config.max_seq_len,
                    false,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let gpu_logits = model_gpu.forward_gpu_with_cache(&[0, 1], &mut gpu_cache).unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 1.0,
                "int8 logit mismatch at [{i}]: cpu={c} gpu={g} diff={diff}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_forward_gpu_f16_matches_cpu() {
        use crate::GpuKVCacheEntry;
        use nexora_autograd::gpu::GpuContext;

        let _ctx = GpuContext::init().expect("GPU context init failed");

        let model = small_model();
        let mut cache = model.reset_cache();
        let cpu_logits = model.forward(&[0, 1], &mut cache).unwrap();

        let mut model_gpu = small_model();
        model_gpu.use_half_precision = true;
        let mut gpu_cache: Vec<GpuKVCacheEntry> = (0..model_gpu.config.num_layers)
            .map(|_| {
                GpuKVCacheEntry::new(
                    &GpuContext::global().unwrap(),
                    model_gpu.config.num_kv_heads,
                    model_gpu.config.head_dim(),
                    model_gpu.config.max_seq_len,
                    true,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let gpu_logits = model_gpu.forward_gpu_with_cache(&[0, 1], &mut gpu_cache).unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 0.1,
                "F16 logit mismatch at [{i}]: cpu={c} gpu={g} diff={diff}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_forward_gpu_f16_kv_cache_matches_cpu() {
        use crate::GpuKVCacheEntry;
        use nexora_autograd::gpu::GpuContext;

        let _ctx = GpuContext::init().expect("GPU context init failed");

        let model = small_model();
        let mut cache = model.reset_cache();
        let cpu_logits = model.forward(&[1, 2, 3], &mut cache).unwrap();

        let mut model_gpu = small_model();
        model_gpu.use_half_precision = true;
        let mut gpu_cache: Vec<GpuKVCacheEntry> = (0..model_gpu.config.num_layers)
            .map(|_| {
                GpuKVCacheEntry::new(
                    &GpuContext::global().unwrap(),
                    model_gpu.config.num_kv_heads,
                    model_gpu.config.head_dim(),
                    model_gpu.config.max_seq_len,
                    true,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Multi-token prefill (3 tokens)
        let gpu_logits = model_gpu.forward_gpu_with_cache(&[1, 2, 3], &mut gpu_cache).unwrap();

        assert_eq!(cpu_logits.len(), gpu_logits.len());
        for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 0.1,
                "F16 KV cache logit mismatch at [{i}]: cpu={c} gpu={g} diff={diff}"
            );
        }
    }

    #[test]
    fn test_collect_weights_for_sedc() {
        let model = small_model();
        let (weights, names, fuse_pairs) = model.collect_weights_for_sedc();
        let per_block = 7;
        let expected = 2 + model.blocks.len() * per_block; // token_embedding + lm_head + blocks*7
        assert_eq!(weights.len(), expected);
        assert_eq!(names.len(), expected);
        assert_eq!(fuse_pairs.len(), model.blocks.len().saturating_sub(1));
    }
}
