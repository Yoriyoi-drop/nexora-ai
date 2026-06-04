use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Duration;

use ndarray::{Array1, Array2};
use nexora_quantization::QFormat;
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
    /// Called after a layer's forward pass.
    fn after_layer(
        &mut self,
        layer_idx: usize,
        h: &mut Array2<f32>,
        pos: usize,
    ) -> TransformerResult<()>;

    /// Reset runtime state (ring buffers, caches) for a new sequence.
    /// Default is a no-op — override if the injector maintains per-sequence state.
    fn reset(&mut self) {}

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
        let mut h_2d = h_cpu.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            nexora_autograd::gpu::GpuError::Unsupported(format!(
                "injector reshape [{}x{}]: {}",
                dims[0], dims[1], e
            ))
        })?;
        self.after_layer(layer_idx, &mut h_2d, pos).map_err(|e| {
            nexora_autograd::gpu::GpuError::Unsupported(format!("injector after_layer: {}", e))
        })?;
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
pub fn sample_token_gpu_keep_gpu(
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
    pub lm_head_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub lm_head_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
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
    pub wq_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub wk_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub wv_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub wo_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub w1_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub w2_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub w3_scales: Option<nexora_autograd::gpu::GpuTensor>,
    pub wq_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
    pub wk_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
    pub wv_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
    pub wo_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
    pub w1_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
    pub w2_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
    pub w3_zero_points: Option<nexora_autograd::gpu::GpuTensor>,
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
    pub token_embedding: Option<Array2<f32>>,
    pub blocks: Vec<TransformerBlock>,
    pub norm: super::rms_norm::RMSNorm,
    pub lm_head: Option<Array2<f32>>,
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
    /// Weight change notifier — external components register here to be
    /// notified when weights are updated (sync_to_inference, load_checkpoint).
    pub weight_notifier: super::observer::WeightNotifier,
    /// Optional lazy weight loader for on-demand block weight loading.
    /// When set, individual block weights are loaded from disk during forward
    /// and can be unloaded after to save memory.
    pub lazy_loader: Option<std::sync::Arc<super::lazy_weights::LazyWeightLoader>>,
    /// If true, unload block weights after each forward pass.
    pub unload_blocks_after_forward: bool,
    /// Collective communication for tensor parallelism.
    /// Set when `config.shard.num_shards > 1`. Used internally by `forward_cpu_impl`
    /// to all-reduce partial attention/FFN outputs across shards.
    pub collective: Option<super::sharded::ShardCollective>,

    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<GpuWeights>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_cache: RwLock<Option<Vec<super::gqa::GpuKVCacheEntry>>>, // never held across await points
}

#[cfg(not(feature = "gpu"))]
impl Clone for CausalLM {
    fn clone(&self) -> Self {
        tracing::warn!("Cloning CausalLM — creates independent weight copies (~{}GB). Use Arc<CausalLM> for shared ownership.",
            (self.config.vocab_size * self.config.hidden_size * 2 // embedding + lm_head
                + self.config.num_layers * self.config.hidden_size * self.config.hidden_size * 8 // ~8 weight matrices per block
            ) * 4 / 1_000_000_000);
        Self {
            config: self.config.clone(),
            token_embedding: self.token_embedding.clone(),
            blocks: self.blocks.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            rope: self.rope.clone(),
            precomputed_cos: self.precomputed_cos.clone(),
            precomputed_sin: self.precomputed_sin.clone(),
            injectors: self.injectors.clone(),
            keep_on_gpu: self.keep_on_gpu,
            quantize_weights: self.quantize_weights,
            use_half_precision: self.use_half_precision,
            weight_notifier: super::observer::WeightNotifier::new(),
            lazy_loader: self.lazy_loader.clone(),
            unload_blocks_after_forward: self.unload_blocks_after_forward,
            collective: self.collective.clone(),
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
            weight_notifier: super::observer::WeightNotifier::new(),
            lazy_loader: self.lazy_loader.clone(),
            unload_blocks_after_forward: self.unload_blocks_after_forward,
            collective: self.collective.clone(),
            gpu_weights: OnceLock::new(),
            gpu_cache: RwLock::new(None),
        }
    }
}

impl CausalLM {
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (config.hidden_size as f32).sqrt().recip();

        let token_embedding = Some(
            Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            }));

        let nh_local = config.num_heads_local();
        let nkv_local = config.num_kv_heads_local();
        let int_local = config.intermediate_size_local();
        let exp_int_local = config.expert_intermediate_size_local();
        let head_dim = config.head_dim();

        let blocks = (0..config.num_layers)
            .map(|_| {
                let mut block = TransformerBlock::new(
                    config.hidden_size,
                    nh_local,
                    nkv_local,
                    head_dim,
                    int_local,
                    config.norm_eps,
                    config.num_experts,
                    exp_int_local,
                );
                block.init_random(config.hidden_size, nh_local, nkv_local, head_dim, int_local, exp_int_local);
                block
            })
            .collect();
        let norm = super::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps);
        let lm_head = Some(Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
        let rope = RoPE::new(head_dim, config.max_seq_len, config.rope_theta);
        let (cos_full, sin_full) = rope.precompute_freqs_cis();
        let half = head_dim / 2;
        let precomputed_cos = cos_full
            .into_shape(config.max_seq_len * half)
            .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));
        let precomputed_sin = sin_full
            .into_shape(config.max_seq_len * half)
            .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));

        let collective = if config.is_sharded() {
            Some(super::sharded::ShardCollective::CpuLocal(
                super::sharded::CpuLocalCollective::new(
                    config.shard.num_shards,
                    config.shard.shard_rank,
                    config.hidden_size,
                ),
            ))
        } else {
            None
        };

        let qw = matches!(config.quantization, QFormat::Q8{..} | QFormat::Q6{..} | QFormat::Q5{..} | QFormat::Q4{..});
        let hp = config.use_half_precision;

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
            keep_on_gpu: {
                #[cfg(feature = "gpu")]
                {
                    nexora_autograd::gpu::GpuContext::is_available()
                }
                #[cfg(not(feature = "gpu"))]
                {
                    false
                }
            },
            quantize_weights: qw,
            use_half_precision: hp,
            weight_notifier: super::observer::WeightNotifier::new(),
            lazy_loader: None,
            unload_blocks_after_forward: false,
            collective,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_cache: RwLock::new(None),
        }
    }

    /// Create a CausalLM with no block weights loaded (weights loaded lazily).
    /// Only structural fields (rope, cos/sin) are initialized; token_embedding,
    /// lm_head, norm, and all block weights are set to None.
    /// Call `load_lazy_blocks()` or set weights manually before forward.
    pub fn new_empty(config: TransformerConfig) -> Self {
        let head_dim = config.head_dim();
        let nh_local = config.num_heads_local();
        let nkv_local = config.num_kv_heads_local();
        let int_local = config.intermediate_size_local();
        let exp_int_local = config.expert_intermediate_size_local();

        let rope = RoPE::new(head_dim, config.max_seq_len, config.rope_theta);
        let (cos_full, sin_full) = rope.precompute_freqs_cis();
        let half = head_dim / 2;
        let collective = if config.is_sharded() {
            Some(super::sharded::ShardCollective::CpuLocal(
                super::sharded::CpuLocalCollective::new(
                    config.shard.num_shards,
                    config.shard.shard_rank,
                    config.hidden_size,
                ),
            ))
        } else {
            None
        };
        let mut blocks: Vec<TransformerBlock> = (0..config.num_layers)
            .map(|_| TransformerBlock::new(
                config.hidden_size,
                nh_local,
                nkv_local,
                head_dim,
                int_local,
                config.norm_eps,
                config.num_experts,
                exp_int_local,
            ))
            .collect();
        Self {
            config: config.clone(),
            token_embedding: None,
            blocks,
            norm: super::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps),
            lm_head: None,
            rope,
            precomputed_cos: cos_full
                .into_shape(config.max_seq_len * half)
                .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half)),
            precomputed_sin: sin_full
                .into_shape(config.max_seq_len * half)
                .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half)),
            injectors: Vec::new(),
            keep_on_gpu: false,
            quantize_weights: matches!(config.quantization, QFormat::Q8{..} | QFormat::Q6{..} | QFormat::Q5{..} | QFormat::Q4{..}),
            use_half_precision: config.use_half_precision,
            weight_notifier: super::observer::WeightNotifier::new(),
            lazy_loader: None,
            unload_blocks_after_forward: false,
            collective,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_cache: RwLock::new(None),
        }
    }

    /// Load all block weights from the lazy weight loader.
    /// Must be called after setting `lazy_loader` if weights are not already loaded.
    pub fn load_lazy_blocks(&mut self) -> TransformerResult<()> {
        let loader = self.lazy_loader.as_ref().ok_or_else(|| {
            TransformerError::Implementation("lazy_loader not set".into())
        })?;
        for i in 0..self.blocks.len() {
            loader.apply_block_weights(i, &mut self.blocks[i])?;
        }
        Ok(())
    }

    /// Ensure all lazy block weights are loaded before forward.
    /// If `unload_blocks_after_forward` is true, blocks are loaded
    /// on demand during `forward_cpu_impl`.
    pub fn ensure_blocks_loaded(&mut self) -> TransformerResult<()> {
        if self.blocks.iter().all(|b| b.attention.wq.is_some()) {
            return Ok(());
        }
        let loader = self.lazy_loader.as_ref().ok_or_else(|| {
            TransformerError::Implementation("Blocks not loaded and no lazy_loader set".into())
        })?;
        for i in 0..self.blocks.len() {
            let block = &mut self.blocks[i];
            if block.attention.wq.is_none() {
                loader.apply_block_weights(i, block)?;
            }
        }
        Ok(())
    }

    /// Unload all block weights from memory, freeing CPU RAM.
    /// After this call, `forward()` will fail — call `load_lazy_blocks()`
    /// or `ensure_blocks_loaded()` before the next forward.
    /// Also notifies the `LazyWeightLoader` to evict from its LRU cache.
    pub fn unload_blocks(&mut self) {
        for block in &mut self.blocks {
            block.attention.wq = None;
            block.attention.wk = None;
            block.attention.wv = None;
            block.attention.wo = None;
            block.ffn.w1 = None;
            block.ffn.w2 = None;
            block.ffn.w3 = None;
            block.attention_norm.weight = None;
            block.ffn_norm.weight = None;
        }
        if let Some(loader) = &self.lazy_loader {
            loader.evict_all_blocks();
        }
    }

    /// Register a weight-change observer.
    /// Fires on every `sync_to_inference()` and `load_checkpoint()`.
    pub fn register_weight_observer(&self, observer: Box<dyn super::observer::WeightObserver>) {
        self.weight_notifier.add(observer);
    }

    /// Notify all registered observers (called by weight-mutating code).
    pub fn notify_weight_changed(&self) {
        self.weight_notifier.notify();
    }

    /// Enable half-precision (f16) weight storage on GPU.
    /// Weights disimpan dalam packed f16 (2× VRAM hemat),
    /// lalu di-upconvert ke f32 tiap forward pass.
    pub fn with_half_precision(mut self) -> Self {
        self.use_half_precision = true;
        for block in &mut self.blocks {
            block.set_use_half_precision();
        }
        self
    }

    fn get_token_embedding(&self) -> TransformerResult<&Array2<f32>> {
        self.token_embedding.as_ref().ok_or_else(|| {
            TransformerError::Implementation("token_embedding not available — use readback_weights() or forward_gpu()".into())
        })
    }

    fn get_lm_head(&self) -> TransformerResult<&Array2<f32>> {
        self.lm_head.as_ref().ok_or_else(|| {
            TransformerError::Implementation("lm_head not available".into())
        })
    }

    pub fn drop_cpu_weights(&mut self) {
        self.token_embedding = None;
        self.lm_head = None;
        for block in &mut self.blocks {
            block.attention.drop_cpu_weights();
            block.ffn.drop_cpu_weights();
            block.attention_norm.drop_cpu_weight();
            block.ffn_norm.drop_cpu_weight();
        }
        self.norm.drop_cpu_weight();
    }

    /// Reset GPU weight caches so the next `forward()` re-uploads from CPU.
    /// Needed after training updates CPU weights via `sync_to_inference()`.
    pub fn reset_gpu_weights(&mut self) {
        #[cfg(feature = "gpu")]
        {
            self.gpu_weights = OnceLock::new();
            self.norm.gpu_weights = OnceLock::new();
            for block in &mut self.blocks {
                block.attention.gpu_weights = OnceLock::new();
                block.ffn.gpu_weights = OnceLock::new();
                block.attention_norm.gpu_weights = OnceLock::new();
                block.ffn_norm.gpu_weights = OnceLock::new();
            }
        }
    }

    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_cache: &mut dyn KVCacheProvider,
    ) -> TransformerResult<Array1<f32>> {
        #[cfg(feature = "gpu")]
        let gpu_available = nexora_autograd::gpu::GpuContext::global().is_ok();

        #[cfg(feature = "gpu")]
        if gpu_available {
            match self.forward_gpu_with_cache_provider(input_ids, kv_cache) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    if self.keep_on_gpu {
                        tracing::warn!(error = %e, "GPU forward pass failed, falling back to CPU");
                    } else {
                        tracing::error!(error = %e, "GPU forward pass failed, falling back to CPU");
                    }
                    GPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let entries = kv_cache.as_cpu_entries().ok_or_else(|| {
            TransformerError::Implementation("CPU forward requires CpuKVCache".to_string())
        })?;
        self.forward_cpu_impl(input_ids, entries)
    }

    fn forward_cpu_impl(
        &self,
        input_ids: &[u32],
        kv_cache: &mut Vec<KVCacheEntry>,
    ) -> TransformerResult<Array1<f32>> {
        if input_ids.is_empty() {
            return Err(TransformerError::Implementation(
                "forward_cpu_impl called with empty input_ids".into()
            ));
        }

        // Check that blocks are loaded (lazy loading must happen before forward)
        if self.lazy_loader.is_some() && self.blocks.first().map_or(true, |b| b.attention.wq.is_none()) {
            return Err(TransformerError::Implementation(
                "Blocks not loaded — call load_lazy_blocks() or ensure_blocks_loaded() before forward()".into()
            ));
        }

        let token_id = input_ids.last().copied().ok_or_else(|| {
            TransformerError::Implementation("forward_cpu_impl called with empty input_ids".into())
        })? as usize;
        if token_id >= self.config.vocab_size {
            return Err(TransformerError::Implementation(format!(
                "Token ID {} out of range [0, {})",
                token_id, self.config.vocab_size
            )));
        }

        let te = self.get_token_embedding()?;
        let mut h = Array2::zeros((1, self.config.hidden_size));
        h.row_mut(0).assign(&te.row(token_id));

        let pos = kv_cache.first().map(|e| e.seq_len()).unwrap_or(0);
        let half = self.config.head_dim() / 2;
        if pos * half + half > self.precomputed_cos.len() && !self.precomputed_cos.is_empty() {
            return Err(TransformerError::Implementation(format!(
                "RoPE cos position {} out of range (len={}, half={})",
                pos,
                self.precomputed_cos.len(),
                half
            )));
        }
        let (cos_slice, sin_slice) = self.get_cos_sin_slices(pos);

        let (embed_vocab, _embed_hidden) = te.dim();
        if embed_vocab != self.config.vocab_size {
            return Err(TransformerError::Implementation(format!(
                "token_embedding vocab {} != config {}",
                embed_vocab, self.config.vocab_size
            )));
        }

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_with_collective(&h, kv_cache, layer_idx, cos_slice, sin_slice, collective)?;

            for (target_layer, injector) in &self.injectors {
                if *target_layer == layer_idx {
                    let mut guard = injector.lock().map_err(|e| {
                        TransformerError::Implementation(format!("Injector lock: {}", e))
                    })?;
                    guard.after_layer(layer_idx, &mut h, pos)?;
                }
            }
        }

        h = self.norm.forward(&h)?;

        let lh = self.get_lm_head()?;
        Ok(h.row(0).dot(&lh.t()))
    }

    pub fn reset_cache(&self) -> CpuKVCache {
        // Reset all injectors for a new sequence (fork-on-write for ring buffers)
        for (_, injector) in &self.injectors {
            if let Ok(mut guard) = injector.lock() {
                guard.reset();
            }
        }
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
        let mut h = Array2::zeros((1, self.config.hidden_size));

        if let Some(&token_id) = input_ids.last() {
            let tid = token_id as usize;
            if tid >= self.config.vocab_size {
                return Err(TransformerError::Implementation(format!(
                    "Token ID {} out of range [0, {})",
                    tid, self.config.vocab_size
                )));
            }
            let te = self.get_token_embedding()?;
            h.row_mut(0).assign(&te.row(tid));
        }

        let token_pos = kv_cache.num_tokens(seq_id).unwrap_or(0);
        let (cos_slice, sin_slice) = self.get_cos_sin_slices(token_pos);

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_paged_with_collective(
                &h, kv_cache, seq_id, layer_idx, token_pos, cos_slice, sin_slice, collective,
            )?;
        }

        h = self.norm.forward(&h)?;

        let lh = self.get_lm_head()?;
        Ok(h.row(0).dot(&lh.t()))
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.token_embedding.as_ref().map_or(0, |w| w.len());
        count += self.lm_head.as_ref().map_or(0, |w| w.len());
        for block in &self.blocks {
            count += block.attention.wq.as_ref().map_or(0, |w| w.len());
            count += block.attention.wk.as_ref().map_or(0, |w| w.len());
            count += block.attention.wv.as_ref().map_or(0, |w| w.len());
            count += block.attention.wo.as_ref().map_or(0, |w| w.len());
            count += block.ffn.w1.as_ref().map_or(0, |w| w.len());
            count += block.ffn.w2.as_ref().map_or(0, |w| w.len());
            count += block.ffn.w3.as_ref().map_or(0, |w| w.len());
            count += block.attention_norm.weight.as_ref().map_or(0, |w| w.len());
            count += block.ffn_norm.weight.as_ref().map_or(0, |w| w.len());
        }
        count += self.norm.weight.as_ref().map_or(0, |w| w.len());
        count
    }

    /// Collect all 2D weight matrices for SEDC compression.
    /// Returns (weights, names, fusion_pairs).
    pub fn collect_weights_for_sedc(&self) -> (Vec<Array2<f32>>, Vec<String>, Vec<(usize, usize)>) {
        let mut weights: Vec<Array2<f32>> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        weights.push(self.token_embedding.clone().unwrap_or(Array2::zeros((0, 0))));
        names.push("token_embedding".to_string());

        weights.push(self.lm_head.clone().unwrap_or(Array2::zeros((0, 0))));
        names.push("lm_head".to_string());

        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{}", i);
            weights.push(block.attention.wq.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wq", prefix));
            weights.push(block.attention.wk.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wk", prefix));
            weights.push(block.attention.wv.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wv", prefix));
            weights.push(block.attention.wo.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.attention.wo", prefix));
            weights.push(block.ffn.w1.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.ffn.w1", prefix));
            weights.push(block.ffn.w2.clone().unwrap_or(Array2::zeros((0, 0))));
            names.push(format!("{}.ffn.w2", prefix));
            weights.push(block.ffn.w3.clone().unwrap_or(Array2::zeros((0, 0))));
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
        mut config: TransformerConfig,
        path: &str,
    ) -> crate::TransformerResult<Self> {
        let (loaded, meta) = crate::safetensors::load_safetensors_with_meta(path)?;
        // Apply quantization from file metadata if present
        if let Some(meta) = meta {
            if let Some(qstr) = meta.get("quantization") {
                match qstr.as_str() {
                    "F16" => config.quantization = nexora_quantization::QFormat::F16,
                    "BF16" => config.quantization = nexora_quantization::QFormat::BF16,
                    other => {
                        tracing::warn!(
                            "from_checkpoint: unknown quantization '{}' in file metadata, using config default",
                            other
                        );
                    }
                }
            }
        }
        let mut model = Self::new(config);
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
        model.token_embedding = Some(
            to_fixed::<ndarray::Ix2>(get_arr("token_embedding")?, "token_embedding")?.to_owned());
        model.lm_head = Some(
            to_fixed::<ndarray::Ix2>(get_arr("lm_head")?, "lm_head")?.to_owned());
        model.norm.weight = Some(
            to_fixed::<ndarray::Ix1>(get_arr("norm.weight")?, "norm.weight")?.to_owned());
        for (i, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{}.", i);
            let name = prefix.clone() + "attention_norm.weight";
            block.attention_norm.weight = Some(
                to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "ffn_norm.weight";
            block.ffn_norm.weight = Some(
                to_fixed::<ndarray::Ix1>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "attention.wq";
            block.attention.wq = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "attention.wk";
            block.attention.wk = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "attention.wv";
            block.attention.wv = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "attention.wo";
            block.attention.wo = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "ffn.w1";
            block.ffn.w1 = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "ffn.w2";
            block.ffn.w2 = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            let name = prefix.clone() + "ffn.w3";
            block.ffn.w3 = Some(to_fixed::<ndarray::Ix2>(get_arr(&name)?, &name)?.to_owned());
            // Load expert weights if present in checkpoint
            if let Some(experts) = &mut block.experts {
                for (e_idx, expert) in experts.iter_mut().enumerate() {
                    let e_prefix = format!("{}experts.{}.", prefix, e_idx);
                    let w1_name = format!("{}w1", e_prefix);
                    if loaded.contains_key(&w1_name) {
                        expert.w1 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w1_name)?, &w1_name)?.to_owned());
                        let w2_name = format!("{}w2", e_prefix);
                        expert.w2 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w2_name)?, &w2_name)?.to_owned());
                        let w3_name = format!("{}w3", e_prefix);
                        expert.w3 = Some(to_fixed::<ndarray::Ix2>(get_arr(&w3_name)?, &w3_name)?.to_owned());
                    }
                }
            }
        }
        model.injectors = Vec::new();
        model.keep_on_gpu = {
            #[cfg(feature = "gpu")]
            {
                nexora_autograd::gpu::GpuContext::is_available()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };
        Ok(model)
    }

    /// Load checkpoint directly to GPU — no CPU weight storage.
    /// CPU weights are never populated; all weight storage is GPU-only.
    /// Requires `gpu` feature. Returns CausalLM with `keep_on_gpu = true`.
    #[cfg(feature = "gpu")]
    pub fn from_checkpoint_gpu(
        mut config: TransformerConfig,
        path: &str,
    ) -> crate::TransformerResult<Self> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let ctx = GpuContext::global()
            .map_err(|e| crate::TransformerError::Implementation(format!("GPU unavailable: {e}")))?;
        let (loaded, meta) = crate::safetensors::load_safetensors_with_meta(path)?;
        // Apply quantization from file metadata if present
        if let Some(meta) = meta {
            if let Some(qstr) = meta.get("quantization") {
                match qstr.as_str() {
                    "F16" => config.quantization = nexora_quantization::QFormat::F16,
                    "BF16" => config.quantization = nexora_quantization::QFormat::BF16,
                    other => {
                        tracing::warn!(
                            "from_checkpoint_gpu: unknown quantization '{}' in file metadata, using config default",
                            other
                        );
                    }
                }
            }
        }
        let model = Self::new(config);

        let get_arr = |name: &str| -> crate::TransformerResult<ArrayD<f32>> {
            loaded.get(name).cloned().ok_or_else(|| {
                crate::TransformerError::Implementation(format!("Missing tensor: {}", name))
            })
        };
        let to_gpu = |arr: ArrayD<f32>| -> crate::TransformerResult<GpuTensor> {
            GpuTensor::from_cpu(&arr).map_err(|e| {
                crate::TransformerError::Implementation(format!("GPU upload failed: {e}"))
            })
        };

        let te_gpu = to_gpu(get_arr("token_embedding")?)?;
        let lh_gpu = to_gpu(get_arr("lm_head")?)?;
        let lm_head_t = ctx.transpose(&lh_gpu)
            .map_err(|e| crate::TransformerError::Implementation(format!("lm_head transpose: {e}")))?;
        let nw_gpu = to_gpu(get_arr("norm.weight")?)?;
        let _ = model.norm.gpu_weights.set(super::rms_norm::RmsNormGpuWeights {
            weight: nw_gpu.clone(),
        });

        let mut block_weights = Vec::with_capacity(model.blocks.len());
        for (i, block) in model.blocks.iter().enumerate() {
            let p = format!("blocks.{}.", i);

            let an_gpu = to_gpu(get_arr(&format!("{}attention_norm.weight", p))?)?;
            let fn_gpu = to_gpu(get_arr(&format!("{}ffn_norm.weight", p))?)?;
            let _ = block.attention_norm.gpu_weights.set(
                super::rms_norm::RmsNormGpuWeights { weight: an_gpu.clone() },
            );
            let _ = block.ffn_norm.gpu_weights.set(
                super::rms_norm::RmsNormGpuWeights { weight: fn_gpu.clone() },
            );

            let wq = to_gpu(get_arr(&format!("{}attention.wq", p))?)?;
            let wk = to_gpu(get_arr(&format!("{}attention.wk", p))?)?;
            let wv = to_gpu(get_arr(&format!("{}attention.wv", p))?)?;
            let wo = to_gpu(get_arr(&format!("{}attention.wo", p))?)?;
            let wq_shape = wq.shape().to_vec();
            let wk_shape = wk.shape().to_vec();
            let wv_shape = wv.shape().to_vec();
            let wo_shape = wo.shape().to_vec();

            let use_f16 = model.use_half_precision && !model.quantize_weights;
            let (wq_t, wk_t, wv_t, wo_t, wq_f16, wk_f16, wv_f16, wo_f16) = if use_f16 {
                let wq_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wq).map_err(|e| crate::TransformerError::Implementation(format!("{}wq t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wq f16: {e}", p)))?;
                let wk_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wk).map_err(|e| crate::TransformerError::Implementation(format!("{}wk t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wk f16: {e}", p)))?;
                let wv_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wv).map_err(|e| crate::TransformerError::Implementation(format!("{}wv t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wv f16: {e}", p)))?;
                let wo_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&wo).map_err(|e| crate::TransformerError::Implementation(format!("{}wo t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}wo f16: {e}", p)))?;
                let d = GpuTensor::zeros(&[1]).map_err(|e| crate::TransformerError::Implementation(format!("dummy: {e}")))?;
                (d.clone(), d.clone(), d.clone(), d.clone(), Some(wq_f16), Some(wk_f16), Some(wv_f16), Some(wo_f16))
            } else {
                let wq_t = ctx.transpose(&wq).map_err(|e| crate::TransformerError::Implementation(format!("{}wq t: {e}", p)))?;
                let wk_t = ctx.transpose(&wk).map_err(|e| crate::TransformerError::Implementation(format!("{}wk t: {e}", p)))?;
                let wv_t = ctx.transpose(&wv).map_err(|e| crate::TransformerError::Implementation(format!("{}wv t: {e}", p)))?;
                let wo_t = ctx.transpose(&wo).map_err(|e| crate::TransformerError::Implementation(format!("{}wo t: {e}", p)))?;
                (wq_t, wk_t, wv_t, wo_t, None, None, None, None)
            };
            let att_wq_f16 = wq_f16.clone();
            let att_wk_f16 = wk_f16.clone();
            let att_wv_f16 = wv_f16.clone();
            let att_wo_f16 = wo_f16.clone();
            let _ = block.attention.gpu_weights.set(super::gqa::GqaGpuWeights {
                wq_t: Some(wq_t.clone()), wk_t: Some(wk_t.clone()), wv_t: Some(wv_t.clone()), wo_t: Some(wo_t.clone()),
                wq_f16: att_wq_f16, wk_f16: att_wk_f16, wv_f16: att_wv_f16, wo_f16: att_wo_f16,
                wq_shape, wk_shape, wv_shape, wo_shape,
                is_f16: use_f16,
            });

            let w1 = to_gpu(get_arr(&format!("{}ffn.w1", p))?)?;
            let w2 = to_gpu(get_arr(&format!("{}ffn.w2", p))?)?;
            let w3 = to_gpu(get_arr(&format!("{}ffn.w3", p))?)?;
            let (w1_t, w2_t, w3_t, w1_f16, w2_f16, w3_f16) = if use_f16 {
                let w1_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&w1).map_err(|e| crate::TransformerError::Implementation(format!("{}w1 t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}w1 f16: {e}", p)))?;
                let w2_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&w2).map_err(|e| crate::TransformerError::Implementation(format!("{}w2 t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}w2 f16: {e}", p)))?;
                let w3_f16 = ctx.f32_to_f16_packed(&ctx.transpose(&w3).map_err(|e| crate::TransformerError::Implementation(format!("{}w3 t: {e}", p)))?)
                    .map_err(|e| crate::TransformerError::Implementation(format!("{}w3 f16: {e}", p)))?;
                let d = GpuTensor::zeros(&[1]).map_err(|e| crate::TransformerError::Implementation(format!("dummy: {e}")))?;
                (d.clone(), d.clone(), d.clone(), Some(w1_f16), Some(w2_f16), Some(w3_f16))
            } else {
                let w1_t = ctx.transpose(&w1).map_err(|e| crate::TransformerError::Implementation(format!("{}w1 t: {e}", p)))?;
                let w2_t = ctx.transpose(&w2).map_err(|e| crate::TransformerError::Implementation(format!("{}w2 t: {e}", p)))?;
                let w3_t = ctx.transpose(&w3).map_err(|e| crate::TransformerError::Implementation(format!("{}w3 t: {e}", p)))?;
                (w1_t, w2_t, w3_t, None, None, None)
            };
            let ffn_w1_f16 = w1_f16.clone();
            let ffn_w2_f16 = w2_f16.clone();
            let ffn_w3_f16 = w3_f16.clone();
            let _ = block.ffn.gpu_weights.set(super::swiglu::SwigluGpuWeights {
                w1_t: w1_t.clone(), w2_t: w2_t.clone(), w3_t: w3_t.clone(),
                w1_f16: ffn_w1_f16, w2_f16: ffn_w2_f16, w3_f16: ffn_w3_f16,
            });

            block_weights.push(BlockGpuWeights {
                attention_norm_weight: an_gpu,
                ffn_norm_weight: fn_gpu,
                wq_t, wk_t, wv_t, wo_t,
                w1_t, w2_t, w3_t,
                wq_i8: None, wk_i8: None, wv_i8: None, wo_i8: None,
                w1_i8: None, w2_i8: None, w3_i8: None,
                wq_scales: None, wk_scales: None, wv_scales: None, wo_scales: None,
                w1_scales: None, w2_scales: None, w3_scales: None,
                wq_zero_points: None, wk_zero_points: None, wv_zero_points: None, wo_zero_points: None,
                w1_zero_points: None, w2_zero_points: None, w3_zero_points: None,
                wq_f16, wk_f16, wv_f16, wo_f16,
                w1_f16, w2_f16, w3_f16,
            });
        }

        let _ = model.gpu_weights.set(GpuWeights {
            token_embedding: te_gpu,
            lm_head_t,
            lm_head_i8: None,
            lm_head_scales: None,
            lm_head_zero_points: None,
            lm_head_f16: None,
            norm_weight: nw_gpu,
            block_weights,
        });

        Ok(model)
    }

    /// Per-channel asymmetric int8 quantization.
    /// Returns (packed u32 data, per-channel scales, per-channel zero_points)
    /// where:
    ///   - packed: each u32 holds 4 int8 values in little-endian byte order
    ///   - scales[f32; N]: per-output-channel scale factor
    ///   - zero_points[f32; N]: per-output-channel dequant bias (precomputed as -zp_i8 * scale)
    /// Dequant formula in shader: val = q_i8 * scales[col] + zero_points[col]
    #[cfg(feature = "gpu")]
    fn quantize_weight(weight: &Array2<f32>) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let shape = weight.shape();
        let n = shape[0];
        let k = shape[1];

        let mut packed = Vec::with_capacity((n * k + 3) / 4);
        let mut scales = Vec::with_capacity(n);
        let mut zero_points = Vec::with_capacity(n);

        for i in 0..n {
            let row = weight.row(i);
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &v in row.iter() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }

            let (scale_i, dequant_bias_i) = if max_val == min_val {
                (1.0, 0.0)
            } else {
                let s = (max_val - min_val) / 255.0;
                let inv_s = 1.0 / s;
                // Compute int8 zero_point: q = round(v / s + 128), then convert to signed
                // so that q = zero_point_i8 + round(v / s) and zero_point_i8 maps to v=0 in [-128,127] range
                // Standard: zero_point = round(-min / s), then q = round(v/s + zero_point), clamp to [-128,127]
                let zp_f32 = (-min_val * inv_s).round().clamp(-128.0, 127.0);
                let zp_i8 = zp_f32 as i32 as i8;
                // Precompute dequant bias: -zp_i8 * s
                (s, -(zp_i8 as f32) * s)
            };

            scales.push(scale_i);
            zero_points.push(dequant_bias_i);

            // Pack this row
            let inv_scale_i = 1.0 / scale_i;
            for (j, &v) in row.iter().enumerate() {
                // Asymmetric: q_i8 = round(v / s + zero_point_i8), clamped to [-128, 127]
                let q_f32 = (v * inv_scale_i).round().clamp(-128.0, 127.0);
                let q = q_f32 as i32 as i8 as u8;
                let idx = i * k + j;
                let word_idx = idx / 4;
                let byte_off = idx % 4;
                if byte_off == 0 {
                    if word_idx >= packed.len() {
                        packed.push(0);
                    }
                }
                packed[word_idx] |= (q as u32) << (byte_off * 8);
            }
        }

        (packed, scales, zero_points)
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

        let token_embedding = mk_gpu(self.token_embedding.as_ref().ok_or_else(|| {
            GpuError::Unsupported("token_embedding not available".into())
        })?)?;

        let lm_head_gpu = mk_gpu(self.lm_head.as_ref().ok_or_else(|| {
            GpuError::Unsupported("lm_head not available".into())
        })?)?;
        // F16 mode: f32 transposed weight ga dipake — cukup f16 packed doang.
        // `zeros(&[1])` adalah DUMMY older (4 bytes) buat ngisi struct field,
        // BUKAN matriks ukuran penuh. VRAM hemat: (vocab_size × hidden_size × 4) bytes.
        let lm_head_t = ctx.transpose(&lm_head_gpu)?;

        let (lm_head_i8, lm_head_scales, lm_head_zero_points) = if self.quantize_weights {
            let lh = self.lm_head.as_ref().ok_or_else(|| {
                GpuError::Unsupported("lm_head not available".into())
            })?;
            let shape = vec![lh.shape()[0], lh.shape()[1]];
            let (packed, sc, zp) = Self::quantize_weight(lh);
            let n_out = sc.len();
            let sc_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], sc)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
            let zp_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], zp)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
            (Some(GpuTensor::from_cpu_i8_packed(shape, &packed)?), Some(sc_t), Some(zp_t))
        } else {
            (None, None, None)
        };

        let norm_weight = {
            let w = self.norm.weight.as_ref().ok_or_else(|| {
                GpuError::Unsupported("norm.weight not available for GPU upload".into())
            })?;
            mk_gpu_1d(w)?
        };

        let mut block_weights = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let attention_norm_weight = {
                let w = block.attention_norm.weight.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention_norm.weight not available".into())
                })?;
                mk_gpu_1d(w)?
            };
            let ffn_norm_weight = {
                let w = block.ffn_norm.weight.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("ffn_norm.weight not available".into())
                })?;
                mk_gpu_1d(w)?
            };

            let wq = mk_gpu(block.attention.wq.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wq not available".into())
            })?)?;
            let wk = mk_gpu(block.attention.wk.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wk not available".into())
            })?)?;
            let wv = mk_gpu(block.attention.wv.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wv not available".into())
            })?)?;
            let wo = mk_gpu(block.attention.wo.as_ref().ok_or_else(|| {
                GpuError::Unsupported("attention.wo not available".into())
            })?)?;
            let w1 = mk_gpu(block.ffn.w1.as_ref().ok_or_else(|| {
                GpuError::Unsupported("ffn.w1 not available".into())
            })?)?;
            let w2 = mk_gpu(block.ffn.w2.as_ref().ok_or_else(|| {
                GpuError::Unsupported("ffn.w2 not available".into())
            })?)?;
            let w3 = mk_gpu(block.ffn.w3.as_ref().ok_or_else(|| {
                GpuError::Unsupported("ffn.w3 not available".into())
            })?)?;

            let mk_scale_zp = |arr: &Array2<f32>| -> Result<_, GpuError> {
                let (p, sc, zp) = Self::quantize_weight(arr);
                let shape = vec![arr.shape()[0], arr.shape()[1]];
                let n_out = sc.len();
                let sc_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], sc)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
                let zp_t = GpuTensor::from_cpu(&ArrayD::from_shape_vec(vec![n_out], zp)
                    .map_err(|e| GpuError::Unsupported(e.to_string()))?)?;
                let i8_t = GpuTensor::from_cpu_i8_packed(shape, &p)?;
                Ok((i8_t, sc_t, zp_t))
            };

            let (wq_i8, wq_scales, wq_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wq.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wq not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (wk_i8, wk_scales, wk_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wk.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wk not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (wv_i8, wv_scales, wv_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wv.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wv not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (wo_i8, wo_scales, wo_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.attention.wo.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("attention.wo not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (w1_i8, w1_scales, w1_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.ffn.w1.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("ffn.w1 not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (w2_i8, w2_scales, w2_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.ffn.w2.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("ffn.w2 not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };
            let (w3_i8, w3_scales, w3_zero_points) = if self.quantize_weights {
                let (t, s, z) = mk_scale_zp(block.ffn.w3.as_ref().ok_or_else(|| {
                    GpuError::Unsupported("ffn.w3 not available".into())
                })?)?;
                (Some(t), Some(s), Some(z))
            } else {
                (None, None, None)
            };

            let use_f16 = self.use_half_precision && !self.quantize_weights;
            // Fix: transpose dulu, BARU pack f16 — matmul WGSL expect transposed weights.
            // Ini benerin bug: dulu f16 di-pack tanpa transpose, hasil matmul salah.
            let wq_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wq)?)?)
            } else {
                None
            };
            let wk_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wk)?)?)
            } else {
                None
            };
            let wv_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wv)?)?)
            } else {
                None
            };
            let wo_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wo)?)?)
            } else {
                None
            };
            let w1_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w1)?)?)
            } else {
                None
            };
            let w2_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w2)?)?)
            } else {
                None
            };
            let w3_f16 = if use_f16 {
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w3)?)?)
            } else {
                None
            };

            // F16 mode: f32 transposed weights (wq_t..w3_t) ga dipake —
            // semua matmul pake f16→f32 upconvert temp per-forward.
            // `zeros(&[1])` DUMMY older (4 bytes per weight, total ~28 bytes),
            // BUKAN 7 matriks ukuran penuh. VRAM hemat ~50% per block.
            let (wq_t, wk_t, wv_t, wo_t, w1_t, w2_t, w3_t) = if use_f16 {
                let d = GpuTensor::zeros(&[1])?;
                (d.clone(), d.clone(), d.clone(), d.clone(), d.clone(), d.clone(), d)
            } else {
                (
                    ctx.transpose(&wq)?,
                    ctx.transpose(&wk)?,
                    ctx.transpose(&wv)?,
                    ctx.transpose(&wo)?,
                    ctx.transpose(&w1)?,
                    ctx.transpose(&w2)?,
                    ctx.transpose(&w3)?,
                )
            };

            block_weights.push(BlockGpuWeights {
                attention_norm_weight,
                ffn_norm_weight,
                wq_t,
                wk_t,
                wv_t,
                wo_t,
                w1_t,
                w2_t,
                w3_t,
                wq_i8,
                wk_i8,
                wv_i8,
                wo_i8,
                w1_i8,
                w2_i8,
                w3_i8,
                wq_scales,
                wk_scales,
                wv_scales,
                wo_scales,
                w1_scales,
                w2_scales,
                w3_scales,
                wq_zero_points,
                wk_zero_points,
                wv_zero_points,
                wo_zero_points,
                w1_zero_points,
                w2_zero_points,
                w3_zero_points,
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
            Some(ctx.f32_to_f16_packed(&ctx.transpose(&lm_head_gpu)?)?)
        } else {
            None
        };

        self.gpu_weights
            .set(GpuWeights {
                token_embedding,
                lm_head_t,
                lm_head_i8,
                lm_head_scales,
                lm_head_zero_points,
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

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_rope_gpu_collective(&h, kv_cache, layer_idx, cos_gpu, sin_gpu, collective)?;
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
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        // Upload cos/sin to GPU ONCE — reused across all layers
        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, cos_slice.len()], cos_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, sin_slice.len()], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope_collective(
                &h, cache, layer_idx, &cos_gpu, &sin_gpu, collective,
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
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        let cos_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, cos_slice.len()], cos_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ndarray::ArrayD::from_shape_vec(vec![1, sin_slice.len()], sin_slice.to_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_with_cache_precomputed_rope_collective(
                &h, cache, layer_idx, &cos_gpu, &sin_gpu, collective,
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
                .map(|_| {
                    super::gqa::GpuKVCacheEntry::new(
                        &ctx,
                        n_kv_heads,
                        head_dim,
                        max_seq,
                        self.use_half_precision,
                    )
                })
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
                    |staging: &wgpu::Buffer,
                     is_f16: bool|
                     -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
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
                        let out: Vec<f32> = {
                            let mapped = slice.get_mapped_range();
                            let result = if is_f16 {
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
                            result
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
            Err(_) => {
                return self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
            }
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
            Err(_) => {
                return self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
            }
        };

        ctx.flush();

        // Prefill prompt tokens
        for &token_id in prompt_ids {
            if self
                .forward_gpu_with_cache(&[token_id], &mut gpu_cache)
                .is_err()
            {
                let (tokens, _) =
                    self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false);
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
                    num_kv_heads: entry.kv_heads,
                    head_dim: entry.head_dim,
                    k_compressed: Vec::new(),
                    v_compressed: Vec::new(),
                    k_scales: Vec::new(),
                    v_scales: Vec::new(),
                    compressed_seq_len: 0,
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
                tracing::warn!(
                    "Failed to create GPU KV cache: {e}, falling back to CPU generation"
                );
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

            let token_gpu =
                match sample_token_gpu_keep_gpu(&logits_gpu, temperature, top_k, top_p, rng_seed) {
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
                    num_kv_heads: entry.kv_heads,
                    head_dim: entry.head_dim,
                    k_compressed: Vec::new(),
                    v_compressed: Vec::new(),
                    k_scales: Vec::new(),
                    v_scales: Vec::new(),
                    compressed_seq_len: 0,
                }
            })
            .collect();

        (output, cpu_cache)
    }

    pub fn memory_bytes(&self) -> usize {
        self.parameter_count() * self.config.bytes_per_param()
    }

    /// Bytes per parameter based on this model's config quantization format.
    pub fn bytes_per_param(&self) -> usize {
        self.config.bytes_per_param()
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
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        let collective = self.collective.as_ref();

        // 3. Forward through all transformer blocks on GPU
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_collective(&h, kv_cache, layer_idx, &cos_slice, &sin_slice, collective)?;

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
        let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward_gpu_collective(&h, kv_cache, layer_idx, &cos_slice, &sin_slice, collective)?;

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
    fn prepare_f16_temps(
        &self,
        _gw: &GpuWeights,
        _ctx: &nexora_autograd::gpu::GpuContext,
        _num_layers: usize,
    ) -> Result<Option<(Vec<[nexora_autograd::gpu::GpuTensor; 7]>, Option<nexora_autograd::gpu::GpuTensor>)>, nexora_autograd::gpu::GpuError> {
        // F16 matmul uses native packed F16 weights directly via auto-dispatch in
        // GpuContext::matmul(). No upconversion needed — saves 2x VRAM bandwidth
        // and eliminates per-forward f16_packed→f32 overhead.
        Ok(None)
    }

    #[cfg(feature = "gpu")]
    fn forward_gpu_single_token_core(
        &self,
        batch_tokens: &[u32],
        ctx: &nexora_autograd::gpu::GpuContext,
        gpu_caches: &mut [Vec<super::gqa::GpuKVCacheEntry>],
        f16_temps: &Option<(Vec<[nexora_autograd::gpu::GpuTensor; 7]>, Option<nexora_autograd::gpu::GpuTensor>)>,
        gw: &GpuWeights,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        let batch_size = batch_tokens.len();
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;

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
        let q_dim = n_heads * head_dim;
        let kv_dim_total = n_kv_heads * head_dim;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_gw = &gw.block_weights[layer_idx];

            // Batched RMSNorm
            let normed = block.attention_norm.forward_gpu(&h)?;

            // Batched QKV projection: 3 matmuls with [batch_size, hidden_size]
            let q_proj = if let Some(ref wq_i8) = block_gw.wq_i8 {
                ctx.matmul_int8_weight(&normed, wq_i8, block_gw.wq_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_scales required".into()))?, block_gw.wq_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_zero_points required".into()))?)?
            } else if let Some(ref wq_f16) = block_gw.wq_f16 {
                ctx.matmul(&normed, wq_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wq_t)?
            };
            let k_proj = if let Some(ref wk_i8) = block_gw.wk_i8 {
                ctx.matmul_int8_weight(&normed, wk_i8, block_gw.wk_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_scales required".into()))?, block_gw.wk_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_zero_points required".into()))?)?
            } else if let Some(ref wk_f16) = block_gw.wk_f16 {
                ctx.matmul(&normed, wk_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wk_t)?
            };
            let v_proj = if let Some(ref wv_i8) = block_gw.wv_i8 {
                ctx.matmul_int8_weight(&normed, wv_i8, block_gw.wv_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_scales required".into()))?, block_gw.wv_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_zero_points required".into()))?)?
            } else if let Some(ref wv_f16) = block_gw.wv_f16 {
                ctx.matmul(&normed, wv_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wv_t)?
            };

            // Batch-upload cos/sin for ALL sequences at once: [batch_size, half]
            let mut cos_flat = Vec::with_capacity(batch_size * half);
            let mut sin_flat = Vec::with_capacity(batch_size * half);
            for seq_idx in 0..batch_size {
                let pos = gpu_caches[seq_idx].first().map(|e| e.seq_len).unwrap_or(0);
                let (cos_slice, sin_slice) = self.get_cos_sin_arrays(pos);
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

            let mut q_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut k_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut v_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut cos_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            let mut sin_rows: Vec<GpuTensor> = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                q_rows.push(GpuTensor::zeros(&[1, q_dim])?);
                k_rows.push(GpuTensor::zeros(&[1, kv_dim_total])?);
                v_rows.push(GpuTensor::zeros(&[1, kv_dim_total])?);
                cos_rows.push(GpuTensor::zeros(&[1, half])?);
                sin_rows.push(GpuTensor::zeros(&[1, half])?);
            }

            // Single batched copy dispatch — all sequences' rows copied in one encoder pass
            ctx.batch_dispatch(|enc| {
                let qb = q_proj.buffer();
                let kb = k_proj.buffer();
                let vb = v_proj.buffer();
                let cb = cos_gpu_batch.buffer();
                let sb = sin_gpu_batch.buffer();
                for seq_idx in 0..batch_size {
                    enc.copy_buffer_to_buffer(qb, (seq_idx * q_dim * 4) as u64, q_rows[seq_idx].buffer(), 0, (q_dim * 4) as u64);
                    enc.copy_buffer_to_buffer(kb, (seq_idx * kv_dim_total * 4) as u64, k_rows[seq_idx].buffer(), 0, (kv_dim_total * 4) as u64);
                    enc.copy_buffer_to_buffer(vb, (seq_idx * kv_dim_total * 4) as u64, v_rows[seq_idx].buffer(), 0, (kv_dim_total * 4) as u64);
                    enc.copy_buffer_to_buffer(cb, (seq_idx * half * 4) as u64, cos_rows[seq_idx].buffer(), 0, (half * 4) as u64);
                    enc.copy_buffer_to_buffer(sb, (seq_idx * half * 4) as u64, sin_rows[seq_idx].buffer(), 0, (half * 4) as u64);
                }
                Ok(())
            })?;

            let mut attn_rows = Vec::with_capacity(batch_size);

            for seq_idx in 0..batch_size {
                // RoPE (per-sequence because positions differ)
                let q_rotated =
                    ctx.rotary_embedding(&q_rows[seq_idx], &cos_rows[seq_idx], &sin_rows[seq_idx], head_dim as u32)?;
                let k_rotated =
                    ctx.rotary_embedding(&k_rows[seq_idx], &cos_rows[seq_idx], &sin_rows[seq_idx], head_dim as u32)?;

                // Append K/V to per-sequence GPU cache
                let k_3d = k_rotated.reshape(vec![1, n_kv_heads, head_dim])?;
                let v_3d = v_rows[seq_idx].reshape(vec![1, n_kv_heads, head_dim])?;
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
                ctx.matmul_int8_weight(&attn_concat, wo_i8, block_gw.wo_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_scales required".into()))?, block_gw.wo_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_zero_points required".into()))?)?
            } else if let Some(ref wo_f16) = block_gw.wo_f16 {
                ctx.matmul(&attn_concat, wo_f16)?
            } else {
                ctx.matmul(&attn_concat, &block_gw.wo_t)?
            };
            h = ctx.add(&h, &attn_proj)?;
            h = super::block::collective_gpu_all_reduce(&h, collective)?;

            // Batched FFN sub-block
            let normed_ffn = block.ffn_norm.forward_gpu(&h)?;
            let ffn_gate = if let Some(ref w1_i8) = block_gw.w1_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w1_i8, block_gw.w1_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_scales required".into()))?, block_gw.w1_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_zero_points required".into()))?)?
            } else if let Some(ref w1_f16) = block_gw.w1_f16 {
                ctx.matmul(&normed_ffn, w1_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w1_t)?
            };
            let ffn_hidden = if let Some(ref w3_i8) = block_gw.w3_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w3_i8, block_gw.w3_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_scales required".into()))?, block_gw.w3_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_zero_points required".into()))?)?
            } else if let Some(ref w3_f16) = block_gw.w3_f16 {
                ctx.matmul(&normed_ffn, w3_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w3_t)?
            };
            let ffn_silu_mul = ctx.mul(&ctx.silu(&ffn_gate)?, &ffn_hidden)?;
            let ffn_out = if let Some(ref w2_i8) = block_gw.w2_i8 {
                ctx.matmul_int8_weight(&ffn_silu_mul, w2_i8, block_gw.w2_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_scales required".into()))?, block_gw.w2_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_zero_points required".into()))?)?
            } else if let Some(ref w2_f16) = block_gw.w2_f16 {
                ctx.matmul(&ffn_silu_mul, w2_f16)?
            } else {
                ctx.matmul(&ffn_silu_mul, &block_gw.w2_t)?
            };
            h = ctx.add(&h, &ffn_out)?;
            h = super::block::collective_gpu_all_reduce(&h, collective)?;
        }

        // ── 3. Final norm + SINGLE LM head matmul (batched) ──
        h = self.norm.forward_gpu(&h)?;
        let logits = if let Some(ref lm_head_i8) = gw.lm_head_i8 {
            ctx.matmul_int8_weight(&h, lm_head_i8, gw.lm_head_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_scales required".into()))?, gw.lm_head_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_zero_points required".into()))?)?
        } else if let Some(ref lm_head_f16) = gw.lm_head_f16 {
            ctx.matmul(&h, lm_head_f16)?
        } else {
            ctx.matmul(&h, &gw.lm_head_t)?
        };

        Ok(logits)
    }

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

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        // Create per-sequence GPU KV caches (one Vec<GpuKVCacheEntry> per sequence)
        use super::gqa::GpuKVCacheEntry;
        let mut gpu_caches: Vec<Vec<GpuKVCacheEntry>> = (0..batch_size)
            .map(|_| {
                (0..num_layers)
                    .map(|_| {
                        GpuKVCacheEntry::new(
                            &ctx,
                            n_kv_heads,
                            head_dim,
                            max_seq,
                            self.use_half_precision,
                        )
                    })
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

        let logits = self.forward_gpu_single_token_core(batch_tokens, &ctx, &mut gpu_caches, &f16_temps, gw)?;

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
                    |staging: &wgpu::Buffer,
                     is_f16: bool|
                     -> Result<Vec<f32>, nexora_autograd::gpu::GpuError> {
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
                        let out: Vec<f32> = {
                            let mapped = slice.get_mapped_range();
                            let result = if is_f16 {
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
                            result
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
                        num_kv_heads: n_kv_heads,
                        head_dim,
                        k_compressed: Vec::new(),
                        v_compressed: Vec::new(),
                        k_scales: Vec::new(),
                        v_scales: Vec::new(),
                        compressed_seq_len: 0,
                    });
                }
            }
        }

        ctx.end_batch_mode();
        Ok(all_logits)
    }

    /// Truly batched GPU **prefill** forward pass.
    ///
    /// Like [`forward_gpu_batched`]], but operates on **existing GPU KV caches**
    /// instead of creating fresh ones from CPU entries. Key differences:
    ///
    /// 1. Skips the "create GpuKVCacheEntry + upload CPU K/V" phase
    /// 2. Skips the "sync GPU→CPU" phase (entries stay on GPU)
    /// 3. Returns `Vec<Option<Array1<f32>>>` — only reads back logits for
    ///    sequences where `needs_logits[i] == true`
    ///
    /// At each position, ALL active sequences' tokens are processed together
    /// with batched QKV projections, FFN, and LM head matmuls. The per-sequence
    /// RoPE + attention + KV cache append is unavoidable but uses GPU entries
    /// in-place with zero CPU round-trip.
    ///
    /// Designed to be called from `forward_prefill_batched` position-by-position:
    /// each call processes one token position across all sequences still in prefill.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched_prefill(
        &self,
        batch_tokens: &[u32],
        gpu_caches: &mut [Vec<super::gqa::GpuKVCacheEntry>],
        needs_logits: &[bool],
    ) -> Result<Vec<Option<Array1<f32>>>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        let logits = self.forward_gpu_single_token_core(batch_tokens, &ctx, gpu_caches, &f16_temps, gw)?;

        // ── 4. Conditional readback: only for sequences at their last token ──
        let needs_any = needs_logits.iter().any(|&n| n);
        let all_logits: Vec<Option<Array1<f32>>> = if needs_any {
            let logits_data: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
            logits_data
                .chunks(vocab_size)
                .zip(needs_logits.iter())
                .map(|(chunk, &need)| {
                    if need {
                        Some(Array1::from_vec(chunk.to_vec()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![None; batch_size]
        };

        // ── SKIP Phase 5: No GPU→CPU sync — entries stay on GPU ──

        ctx.end_batch_mode();
        Ok(all_logits)
    }

    /// Batched GPU forward + GPU sampling — processes one token per sequence on GPU
    /// and samples next tokens entirely on device. Zero CPU readback of full logits
    /// (128KB/seq). Only token IDs (4 bytes/seq) are read back.
    ///
    /// Takes existing GPU KV cache entries (no CPU→GPU upload, no GPU→CPU sync).
    /// Per-sequence sampling parameters for temperature/top_k/top_p.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched_sample(
        &self,
        batch_tokens: &[u32],
        gpu_caches: &mut [Vec<super::gqa::GpuKVCacheEntry>],
        temperatures: &[f32],
        top_ks: &[u32],
        top_ps: &[f32],
        seeds: &[u64],
    ) -> Result<Vec<u32>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let batch_size = batch_tokens.len();
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        ctx.begin_batch_mode();

        let logits = self.forward_gpu_single_token_core(batch_tokens, &ctx, gpu_caches, &f16_temps, gw)?;

        // ── 4. GPU sampling (zero logit readback) — batched when params are uniform ──
        let token_ids = if batch_size > 1
            && temperatures.iter().all(|&t| (t - temperatures[0]).abs() < 1e-6)
            && top_ks.iter().all(|&k| k == top_ks[0])
            && top_ps.iter().all(|&p| (p - top_ps[0]).abs() < 1e-6)
        {
            // ── Fast batched path: single gpu_sample call + single readback ──
            // WGSL per-row seed mixing (seed ^ row * 0x9E3779B9) already
            // gives each row unique random values — one seed is sufficient.
            let tok = ctx.gpu_sample(
                &logits,
                temperatures[0],
                top_ks[0],
                top_ps[0],
                seeds[0],
            )?;
            let raw = tok.to_cpu_raw_bytes()?;
            raw.chunks_exact(4)
                .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            // ── Per-seq path: different params per sequence ──
            let mut ids = Vec::with_capacity(batch_size);
            for seq_idx in 0..batch_size {
                let logits_row = ctx.slice_tensor(&logits, seq_idx as u32, 1)?;
                let tok = ctx.gpu_sample(
                    &logits_row,
                    temperatures[seq_idx],
                    top_ks[seq_idx],
                    top_ps[seq_idx],
                    seeds[seq_idx],
                )?;
                let raw = tok.to_cpu_raw_bytes()?;
                ids.push(u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]));
            }
            ids
        };

        // ── SKIP Phase 5: No GPU→CPU sync — entries stay on GPU ──

        ctx.end_batch_mode();
        Ok(token_ids)
    }

    /// True GPU padded-batch prefill — processes ALL remaining tokens across ALL
    /// sequences in ONE forward pass (not position-by-position).
    ///
    /// Key differences from `forward_gpu_batched_prefill`:
    /// - Input is `&[Vec<u32>]` (per-seq remaining tokens) not `&[u32]` (one token each)
    /// - QKV/FFN/LM-head matmuls on `[total_tokens, hidden]` (not `[N, hidden]`)
    /// - Batched RoPE with per-token position IDs
    /// - Bulk KV append per sequence (one GPU copy, not N copies)
    /// - Multi-token attention per sequence with `causal=true`
    #[cfg(feature = "gpu")]
    pub fn forward_gpu_batched_prefill_full(
        &self,
        batch_inputs: &[&[u32]],
        gpu_caches: &mut [Vec<super::gqa::GpuKVCacheEntry>],
    ) -> Result<Vec<Array1<f32>>, nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuTensor};

        self.preupload_weights_gpu()?;

        let ctx = GpuContext::global()?;
        let n = batch_inputs.len();
        if n == 0 || n != gpu_caches.len() {
            return Ok(Vec::new());
        }
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let n_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let vocab_size = self.config.vocab_size;
        let n_heads = self.config.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported(
                "GPU weights not initialized after preupload".into(),
            )
        })?;

        let f16_temps = self.prepare_f16_temps(gw, &ctx, num_layers)?;

        // ── 0b. Build flat token array + per-sequence offsets + position IDs ──
        let mut flat_tokens: Vec<u32> = Vec::new();
        let mut seq_offsets: Vec<(usize, usize)> = Vec::with_capacity(n);
        for seq_tokens in batch_inputs {
            let start = flat_tokens.len();
            flat_tokens.extend_from_slice(seq_tokens);
            seq_offsets.push((start, seq_tokens.len()));
        }
        let total_tokens = flat_tokens.len();
        if total_tokens == 0 {
            return Ok(vec![Array1::zeros(1); n]);
        }

        // Validate token IDs
        for &token_id in &flat_tokens {
            let tid = token_id as usize;
            if tid >= vocab_size {
                return Err(nexora_autograd::gpu::GpuError::Unsupported(format!(
                    "Token ID {} out of range [0, {})",
                    tid, vocab_size
                )));
            }
        }

        ctx.begin_batch_mode();

        // ── 1. Batched embedding ──
        let row_bytes = (hidden_size * 4) as u64;
        let mut h = GpuTensor::zeros(&[total_tokens, hidden_size])?;
        ctx.batch_dispatch(|enc| {
            for (flat_idx, &token_id) in flat_tokens.iter().enumerate() {
                let offset = (token_id as usize * hidden_size * 4) as u64;
                let dst_offset = (flat_idx * hidden_size * 4) as u64;
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

        // ── 1b. Build position-aware cos/sin for ALL tokens ──
        let mut cos_flat = Vec::with_capacity(total_tokens * half);
        let mut sin_flat = Vec::with_capacity(total_tokens * half);
        for (i, seq_tokens) in batch_inputs.iter().enumerate() {
            let start_pos = gpu_caches[i].first().map(|e| e.seq_len).unwrap_or(0);
            for j in 0..seq_tokens.len() {
                let pos = start_pos + j;
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
        }
        let cos_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![total_tokens, half], cos_flat)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![total_tokens, half], sin_flat)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?,
        )?;

        // ── 2. Forward through all blocks ──
        let q_dim = n_heads * head_dim;
        let kv_dim_total = n_kv_heads * head_dim;

        let collective = self.collective.as_ref();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let block_gw = &gw.block_weights[layer_idx];

            // ── 2a. Batched RMSNorm + QKV ──
            let normed = block.attention_norm.forward_gpu(&h)?;

            let q_proj = if let Some(ref wq_i8) = block_gw.wq_i8 {
                ctx.matmul_int8_weight(&normed, wq_i8, block_gw.wq_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_scales required".into()))?, block_gw.wq_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wq_zero_points required".into()))?)?
            } else if let Some(ref wq_f16) = block_gw.wq_f16 {
                ctx.matmul(&normed, wq_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wq_t)?
            };
            let k_proj = if let Some(ref wk_i8) = block_gw.wk_i8 {
                ctx.matmul_int8_weight(&normed, wk_i8, block_gw.wk_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_scales required".into()))?, block_gw.wk_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wk_zero_points required".into()))?)?
            } else if let Some(ref wk_f16) = block_gw.wk_f16 {
                ctx.matmul(&normed, wk_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wk_t)?
            };
            let v_proj = if let Some(ref wv_i8) = block_gw.wv_i8 {
                ctx.matmul_int8_weight(&normed, wv_i8, block_gw.wv_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_scales required".into()))?, block_gw.wv_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wv_zero_points required".into()))?)?
            } else if let Some(ref wv_f16) = block_gw.wv_f16 {
                ctx.matmul(&normed, wv_f16)?
            } else {
                ctx.matmul(&normed, &block_gw.wv_t)?
            };

            // ── 2b. Batched RoPE (per-token position) ──
            let q_rotated = ctx.rotary_embedding(&q_proj, &cos_gpu, &sin_gpu, head_dim as u32)?;
            let k_rotated = ctx.rotary_embedding(&k_proj, &cos_gpu, &sin_gpu, head_dim as u32)?;

            // ── 2c. Per-sequence: slice → bulk KV append → attention (causal=true) ──
            let mut attn_rows = Vec::with_capacity(n);
            for seq_idx in 0..n {
                let (start, remaining) = seq_offsets[seq_idx];

                let q_seq = ctx.slice_tensor(&q_rotated, start as u32, remaining as u32)?;
                let k_seq = ctx.slice_tensor(&k_rotated, start as u32, remaining as u32)?;
                let v_seq = ctx.slice_tensor(&v_proj, start as u32, remaining as u32)?;

                let k_3d = k_seq.reshape(vec![remaining, n_kv_heads, head_dim])?;
                let v_3d = v_seq.reshape(vec![remaining, n_kv_heads, head_dim])?;

                gpu_caches[seq_idx][layer_idx].bulk_append(&ctx, &k_3d, &v_3d, remaining)?;

                let total_seq = gpu_caches[seq_idx][layer_idx].seq_len;
                let (k_rep, v_rep) =
                    gpu_caches[seq_idx][layer_idx].get_repeated_kv(&ctx, n_heads as u32)?;

                let q_4d = q_seq.reshape(vec![1, n_heads, remaining, head_dim])?;
                let k_4d = k_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;
                let v_4d = v_rep.reshape(vec![1, n_heads, total_seq, head_dim])?;

                let attn_out = ctx.fused_attention(&q_4d, &k_4d, &v_4d, scale, true)?;
                attn_rows.push(attn_out.reshape(vec![remaining, n_heads * head_dim])?);
            }

            // ── 2d. Gather attention outputs ──
            let attn_concat = GpuTensor::zeros(&[total_tokens, hidden_size])?;
            ctx.batch_dispatch(|enc| {
                for (seq_idx, row) in attn_rows.iter().enumerate() {
                    let (start, remaining) = seq_offsets[seq_idx];
                    enc.copy_buffer_to_buffer(
                        row.buffer(),
                        0,
                        attn_concat.buffer(),
                        (start * hidden_size * 4) as u64,
                        (remaining * hidden_size * 4) as u64,
                    );
                }
                Ok(())
            })?;

            // ── 2e. Batched Wo + residual ──
            let attn_proj = if let Some(ref wo_i8) = block_gw.wo_i8 {
                ctx.matmul_int8_weight(&attn_concat, wo_i8, block_gw.wo_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_scales required".into()))?, block_gw.wo_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: wo_zero_points required".into()))?)?
            } else if let Some(ref wo_f16) = block_gw.wo_f16 {
                ctx.matmul(&attn_concat, wo_f16)?
            } else {
                ctx.matmul(&attn_concat, &block_gw.wo_t)?
            };
            h = ctx.add(&h, &attn_proj)?;
            h = super::block::collective_gpu_all_reduce(&h, collective)?;

            // ── 2f. Batched FFN ──
            let normed_ffn = block.ffn_norm.forward_gpu(&h)?;
            let ffn_gate = if let Some(ref w1_i8) = block_gw.w1_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w1_i8, block_gw.w1_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_scales required".into()))?, block_gw.w1_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w1_zero_points required".into()))?)?
            } else if let Some(ref w1_f16) = block_gw.w1_f16 {
                ctx.matmul(&normed_ffn, w1_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w1_t)?
            };
            let ffn_hidden = if let Some(ref w3_i8) = block_gw.w3_i8 {
                ctx.matmul_int8_weight(&normed_ffn, w3_i8, block_gw.w3_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_scales required".into()))?, block_gw.w3_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w3_zero_points required".into()))?)?
            } else if let Some(ref w3_f16) = block_gw.w3_f16 {
                ctx.matmul(&normed_ffn, w3_f16)?
            } else {
                ctx.matmul(&normed_ffn, &block_gw.w3_t)?
            };
            let ffn_silu_mul = ctx.mul(&ctx.silu(&ffn_gate)?, &ffn_hidden)?;
            let ffn_out = if let Some(ref w2_i8) = block_gw.w2_i8 {
                ctx.matmul_int8_weight(&ffn_silu_mul, w2_i8, block_gw.w2_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_scales required".into()))?, block_gw.w2_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: w2_zero_points required".into()))?)?
            } else if let Some(ref w2_f16) = block_gw.w2_f16 {
                ctx.matmul(&ffn_silu_mul, w2_f16)?
            } else {
                ctx.matmul(&ffn_silu_mul, &block_gw.w2_t)?
            };
            h = ctx.add(&h, &ffn_out)?;
            h = super::block::collective_gpu_all_reduce(&h, collective)?;
        }

        // ── 3. Final norm + batched LM head ──
        h = self.norm.forward_gpu(&h)?;
        let logits = if let Some(ref lm_head_i8) = gw.lm_head_i8 {
            ctx.matmul_int8_weight(&h, lm_head_i8, gw.lm_head_scales.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_scales required".into()))?, gw.lm_head_zero_points.as_ref().ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("int8: lm_head_zero_points required".into()))?)?
        } else if let Some(ref lm_head_f16) = gw.lm_head_f16 {
            ctx.matmul(&h, lm_head_f16)?
        } else {
            ctx.matmul(&h, &gw.lm_head_t)?
        };

        // ── 4. Readback: extract last-token logits per sequence ──
        let logits_data: Vec<f32> = logits.to_cpu()?.iter().copied().collect();
        let mut results = Vec::with_capacity(n);
        for (start, remaining) in &seq_offsets {
            let last_idx = start + remaining - 1;
            let offset = last_idx * vocab_size;
            let chunk = &logits_data[offset..offset + vocab_size];
            results.push(Array1::from_vec(chunk.to_vec()));
        }

        ctx.end_batch_mode();
        Ok(results)
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
            Ok(_) => self.generate_gpu_keep_gpu_impl(
                prompt_ids,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
            ),
            Err(_) => {
                tracing::warn!("GPU context unavailable for GPU-resident generation, falling back to CPU generation");
                self.generate_with_gpu(prompt_ids, max_tokens, temperature, top_k, false)
            }
        }
    }

    /// Readback ALL weights from CPU (or GPU if CPU weights dropped).
    /// Returns (name, ArrayD) pairs for safetensors save or training init.
    pub fn readback_weights(&self) -> crate::TransformerResult<Vec<(String, ndarray::ArrayD<f32>)>> {
        use ndarray::{Array1, Array2, ArrayD};
        let mut tensors: Vec<(String, ArrayD<f32>)> =
            Vec::with_capacity(3 + 9 * self.blocks.len() + 3 * self.config.num_experts * self.blocks.len());

        // token_embedding
        if let Some(te) = &self.token_embedding {
            tensors.push(("token_embedding".to_owned(), te.clone().into_dyn()));
        } else {
            #[cfg(feature = "gpu")]
            {
                let gw = self.gpu_weights.get().ok_or_else(||
                    crate::TransformerError::Implementation("token_embedding unavailable — no CPU or GPU copy".into()))?;
                let cpu = gw.token_embedding.to_cpu().map_err(|e|
                    crate::TransformerError::Implementation(format!("token_embedding GPU readback: {e}")))?;
                tensors.push(("token_embedding".to_owned(), cpu));
            }
            #[cfg(not(feature = "gpu"))]
            return Err(crate::TransformerError::Implementation(
                "token_embedding unavailable — no CPU copy and GPU feature disabled".into()));
        }

        // lm_head
        if let Some(lh) = &self.lm_head {
            tensors.push(("lm_head".to_owned(), lh.clone().into_dyn()));
        } else {
            #[cfg(feature = "gpu")]
            {
                let gw = self.gpu_weights.get().ok_or_else(||
                    crate::TransformerError::Implementation("lm_head unavailable — no CPU or GPU copy".into()))?;
                // lm_head_t is [hidden, vocab] transposed; re-transpose to get [vocab, hidden]
                let cpu = gw.lm_head_t.to_cpu().map_err(|e|
                    crate::TransformerError::Implementation(format!("lm_head GPU readback: {e}")))?;
                let cpu_2d = cpu.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| crate::TransformerError::Implementation(format!("lm_head shape: {e}")))?;
                let re_t = cpu_2d.reversed_axes();
                tensors.push(("lm_head".to_owned(), re_t.into_dyn()));
            }
            #[cfg(not(feature = "gpu"))]
            return Err(crate::TransformerError::Implementation(
                "lm_head unavailable — no CPU copy and GPU feature disabled".into()));
        }

        // norm.weight (1D)
        if let Some(nw) = &self.norm.weight {
            tensors.push(("norm.weight".to_owned(), nw.clone().into_dyn()));
        } else {
            let cpu = self.norm.readback_weight().map_err(|e|
                crate::TransformerError::Implementation(format!("norm.weight readback failed: {e}")))?;
            tensors.push(("norm.weight".to_owned(), cpu.into_dyn()));
        }

        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{}.", i);

            // attention_norm.weight
            if let Some(w) = &block.attention_norm.weight {
                tensors.push((format!("{}attention_norm.weight", p), w.clone().into_dyn()));
            } else {
                let cpu = block.attention_norm.readback_weight().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}attention_norm.weight readback: {e}", p)))?;
                tensors.push((format!("{}attention_norm.weight", p), cpu.into_dyn()));
            }

            // ffn_norm.weight
            if let Some(w) = &block.ffn_norm.weight {
                tensors.push((format!("{}ffn_norm.weight", p), w.clone().into_dyn()));
            } else {
                let cpu = block.ffn_norm.readback_weight().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}ffn_norm.weight readback: {e}", p)))?;
                tensors.push((format!("{}ffn_norm.weight", p), cpu.into_dyn()));
            }

            // GQA weights — readback from GPU if no CPU copy, else use CPU weights
            if let Some(w) = &block.attention.wq {
                tensors.push((format!("{}attention.wq", p), w.clone().into_dyn()));
                if let Some(wk) = &block.attention.wk {
                    tensors.push((format!("{}attention.wk", p), wk.clone().into_dyn()));
                }
                if let Some(wv) = &block.attention.wv {
                    tensors.push((format!("{}attention.wv", p), wv.clone().into_dyn()));
                }
                if let Some(wo) = &block.attention.wo {
                    tensors.push((format!("{}attention.wo", p), wo.clone().into_dyn()));
                }
            } else {
                let g = block.attention.readback_weights().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}attention readback: {:?}", p, e)))?;
                tensors.push((format!("{}attention.wq", p), g.0.into_dyn()));
                tensors.push((format!("{}attention.wk", p), g.1.into_dyn()));
                tensors.push((format!("{}attention.wv", p), g.2.into_dyn()));
                tensors.push((format!("{}attention.wo", p), g.3.into_dyn()));
            }

            // SwiGLU weights — same pattern
            if let Some(w) = &block.ffn.w1 {
                tensors.push((format!("{}ffn.w1", p), w.clone().into_dyn()));
                if let Some(w2) = &block.ffn.w2 {
                    tensors.push((format!("{}ffn.w2", p), w2.clone().into_dyn()));
                }
                if let Some(w3) = &block.ffn.w3 {
                    tensors.push((format!("{}ffn.w3", p), w3.clone().into_dyn()));
                }
            } else {
                let g = block.ffn.readback_weights().map_err(|e|
                    crate::TransformerError::Implementation(format!("{}ffn readback: {:?}", p, e)))?;
                tensors.push((format!("{}ffn.w1", p), g.0.into_dyn()));
                tensors.push((format!("{}ffn.w2", p), g.1.into_dyn()));
                tensors.push((format!("{}ffn.w3", p), g.2.into_dyn()));
            }

            // Expert weights
            if let Some(experts) = &block.experts {
                for (e_idx, expert) in experts.iter().enumerate() {
                    let e_prefix = format!("{}experts.{}.", p, e_idx);
                    if let Some(w) = &expert.w1 {
                        tensors.push((format!("{}w1", e_prefix), w.clone().into_dyn()));
                    }
                    if let Some(w2) = &expert.w2 {
                        tensors.push((format!("{}w2", e_prefix), w2.clone().into_dyn()));
                    }
                    if let Some(w3) = &expert.w3 {
                        tensors.push((format!("{}w3", e_prefix), w3.clone().into_dyn()));
                    }
                }
            }
        }

        Ok(tensors)
    }

    /// Save checkpoint by reading back from GPU (or using resident CPU weights).
    pub fn save_checkpoint(&self, path: &str) -> crate::TransformerResult<()> {
        let tensors = self.readback_weights()?;
        let refs: Vec<(&str, ndarray::ArrayD<f32>)> = tensors
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.clone()))
            .collect();
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            "quantization".to_string(),
            self.config.quantization.dtype_name().to_string(),
        );
        crate::safetensors::save_safetensors_with_meta(
            path,
            &refs,
            crate::safetensors::SaveDtype::F32,
            Some(meta),
        )
    }

    fn get_cos_sin_slices(&self, pos: usize) -> (&[f32], &[f32]) {
        let half = self.config.head_dim() / 2;
        let cos = if pos * half < self.precomputed_cos.len() {
            &self.precomputed_cos.as_slice().unwrap_or(&[])[pos * half..(pos + 1) * half]
        } else {
            &[]
        };
        let sin = if pos * half < self.precomputed_sin.len() {
            &self.precomputed_sin.as_slice().unwrap_or(&[])[pos * half..(pos + 1) * half]
        } else {
            &[]
        };
        (cos, sin)
    }

    fn get_cos_sin_arrays(&self, pos: usize) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
        let half = self.config.head_dim() / 2;
        let cos = if pos * half < self.precomputed_cos.len() {
            self.precomputed_cos
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            ndarray::Array1::zeros(half)
        };
        let sin = if pos * half < self.precomputed_sin.len() {
            self.precomputed_sin
                .slice(ndarray::s![pos * half..(pos + 1) * half])
                .to_owned()
        } else {
            ndarray::Array1::zeros(half)
        };
        (cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransformerConfig;
    use nexora_quantization::QFormat;

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
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            quantization: QFormat::F16,
            use_half_precision: true,
            shard: Default::default(),
            shared_expert: 0,
            use_domain_experts: false,
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
        assert_eq!(model.token_embedding.as_ref().unwrap().dim(), (100, 32));
        assert_eq!(model.lm_head.as_ref().unwrap().dim(), (100, 32));
        assert_eq!(model.norm.weight.as_ref().unwrap().len(), 32);
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
        assert_eq!(bytes, model.parameter_count() * model.bytes_per_param());
    }

    #[test]
    fn test_keep_on_gpu_default() {
        let model = small_model();
        #[cfg(feature = "gpu")]
        assert_eq!(model.keep_on_gpu, nexora_autograd::gpu::GpuContext::is_available());
        #[cfg(not(feature = "gpu"))]
        assert!(!model.keep_on_gpu);
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
        let gpu_logits = model_gpu
            .forward_gpu_with_cache(&[0, 1], &mut gpu_cache)
            .unwrap();

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
        let gpu_logits = model_gpu
            .forward_gpu_with_cache(&[0, 1], &mut gpu_cache)
            .unwrap();

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
        let gpu_logits = model_gpu
            .forward_gpu_with_cache(&[1, 2, 3], &mut gpu_cache)
            .unwrap();

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
