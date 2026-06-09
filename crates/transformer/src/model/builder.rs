use std::sync::{Arc, Mutex, OnceLock, RwLock};
use ndarray::{Array1, Array2};
use nexora_quantization::QFormat;
use crate::block::TransformerBlock;
use crate::embedding_registry;
use crate::lora::LayerLoRA;
use crate::rope::RoPE;
use crate::{LayerInjector, TransformerConfig, TransformerError, TransformerResult};

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
    pub norm: crate::rms_norm::RMSNorm,
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
    pub weight_notifier: crate::observer::WeightNotifier,
    /// Optional lazy weight loader for on-demand block weight loading.
    /// When set, individual block weights are loaded from disk during forward
    /// and can be unloaded after to save memory.
    pub lazy_loader: Option<std::sync::Arc<crate::lazy_weights::LazyWeightLoader>>,
    /// If true, unload block weights after each forward pass.
    pub unload_blocks_after_forward: bool,
    /// Collective communication for tensor parallelism.
    /// Set when `config.shard.num_shards > 1`. Used internally by `forward_cpu_impl`
    /// to all-reduce partial attention/FFN outputs across shards.
    pub collective: Option<crate::sharded::ShardCollective>,

    /// Weight tying: lm_head is the same as token_embedding.
    /// When true, `lm_head` field is `None` and all lm_head operations
    /// use `token_embedding` directly. Saves ~(vocab_size × hidden_size × 4) bytes.
    pub weight_tied: bool,

    /// LoRA PEFT adapters for runtime inference.
    /// Each entry corresponds to a transformer block layer.
    /// Applied on-the-fly in forward pass: Wx + A·B·x·scaling.
    pub lora_adapters: Option<Arc<Vec<LayerLoRA>>>,

    /// ATQS compressed weight cache (AWQ 4-bit).
    /// When `use_atqs` is true, weights are stored in this cache
    /// and can be freed from f32 memory. Call `compress_atqs()`
    /// to populate, then `free_weights()` to drop f32 copies.
    /// Call `restore_weights()` before forward when freed.
    pub atqs_compressed: Option<crate::atqs::WeightsAtqs>,
    /// If true, use ATQS compressed weights. Requires manual
    /// `compress_atqs()` + `free_weights()` / `restore_weights()`.
    pub use_atqs: bool,

    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<GpuWeights>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_cache: RwLock<Option<Vec<crate::gqa::GpuKVCacheEntry>>>, // never held across await points
    #[cfg(feature = "gpu")]
    pub(crate) rope_cos_gpu: OnceLock<nexora_autograd::gpu::GpuTensor>,
    #[cfg(feature = "gpu")]
    pub(crate) rope_sin_gpu: OnceLock<nexora_autograd::gpu::GpuTensor>,
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
            weight_notifier: crate::observer::WeightNotifier::new(),
            lazy_loader: self.lazy_loader.clone(),
            unload_blocks_after_forward: self.unload_blocks_after_forward,
            collective: self.collective.clone(),
            weight_tied: self.weight_tied,
            lora_adapters: self.lora_adapters.clone(),
            atqs_compressed: self.atqs_compressed.clone(),
            use_atqs: self.use_atqs,
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
            weight_notifier: crate::observer::WeightNotifier::new(),
            lazy_loader: self.lazy_loader.clone(),
            unload_blocks_after_forward: self.unload_blocks_after_forward,
            collective: self.collective.clone(),
            weight_tied: self.weight_tied,
            lora_adapters: self.lora_adapters.clone(),
            atqs_compressed: self.atqs_compressed.clone(),
            use_atqs: self.use_atqs,
            gpu_weights: OnceLock::new(),
            gpu_cache: RwLock::new(None),
            rope_cos_gpu: OnceLock::new(),
            rope_sin_gpu: OnceLock::new(),
        }
    }
}

impl CausalLM {
    /// Enable weight tying: lm_head reuses token_embedding data.
    /// Drops lm_head allocation, all lm_head operations use token_embedding.
    /// Returns self for chaining.
    pub fn with_weight_tying(mut self) -> Self {
        self.weight_tied = true;
        self.lm_head = None;
        self
    }

    /// Attach LoRA PEFT adapters for runtime inference.
    /// `adapters` should have one entry per transformer block layer.
    /// Returns self for chaining.
    pub fn attach_lora(mut self, adapters: Vec<LayerLoRA>) -> Self {
        self.lora_adapters = Some(Arc::new(adapters));
        self
    }

    /// Get LoRA adapter for a specific layer, if attached.
    pub fn get_lora(&self, layer_idx: usize) -> Option<&LayerLoRA> {
        self.lora_adapters.as_ref().and_then(|adapters| {
            if layer_idx < adapters.len() {
                Some(&adapters[layer_idx])
            } else {
                None
            }
        })
    }

    pub fn new(config: TransformerConfig) -> Self {
        let embedding_seed = 42;
        let shared_embed = embedding_registry::resolve_embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_seed,
        );
        let token_embedding = Some((*shared_embed).clone());

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
        let norm = crate::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps);
        // Weight tying: lm_head = token_embedding by default.
        // Set `lm_head = None` and `weight_tied = true` to save memory.
        let lm_head = Some((*shared_embed).clone());
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
            Some(crate::sharded::ShardCollective::CpuLocal(
                crate::sharded::CpuLocalCollective::new(
                    config.shard.num_shards,
                    config.shard.shard_rank,
                    config.hidden_size,
                ),
            ))
        } else {
            None
        };

        let qw = matches!(config.quantization, QFormat::Q8{..} | QFormat::Q6{..} | QFormat::Q5{..} | QFormat::Q4{..} | QFormat::Q2{..});
        let hp = config.use_half_precision;
        let use_atqs = matches!(config.quantization, QFormat::Q4 {..});

        let mut model = Self {
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
            weight_notifier: crate::observer::WeightNotifier::new(),
            lazy_loader: None,
            unload_blocks_after_forward: false,
            collective,
            weight_tied: false,
            lora_adapters: None,
            atqs_compressed: None,
            use_atqs,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_cache: RwLock::new(None),
            #[cfg(feature = "gpu")]
            rope_cos_gpu: OnceLock::new(),
            #[cfg(feature = "gpu")]
            rope_sin_gpu: OnceLock::new(),
        };

        // Auto-enable ATQS AWQ 4-bit compression when Q4 is configured.
        if use_atqs {
            if let Err(e) = model.enable_atqs() {
                tracing::warn!("ATQS auto-compression failed: {e}");
            }
        }

        model
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
            Some(crate::sharded::ShardCollective::CpuLocal(
                crate::sharded::CpuLocalCollective::new(
                    config.shard.num_shards,
                    config.shard.shard_rank,
                    config.hidden_size,
                ),
            ))
        } else {
            None
        };
        let blocks: Vec<TransformerBlock> = (0..config.num_layers)
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
            norm: crate::rms_norm::RMSNorm::new(config.hidden_size, config.norm_eps),
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
            quantize_weights: matches!(config.quantization, QFormat::Q8{..} | QFormat::Q6{..} | QFormat::Q5{..} | QFormat::Q4{..} | QFormat::Q2{..}),
            use_half_precision: config.use_half_precision,
            weight_notifier: crate::observer::WeightNotifier::new(),
            lazy_loader: None,
            unload_blocks_after_forward: false,
            collective,
            weight_tied: false,
            lora_adapters: None,
            atqs_compressed: None,
            use_atqs: false,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_cache: RwLock::new(None),
            #[cfg(feature = "gpu")]
            rope_cos_gpu: OnceLock::new(),
            #[cfg(feature = "gpu")]
            rope_sin_gpu: OnceLock::new(),
        }
    }

    /// Compress all f32 weights using ATQS AWQ 4-bit quantization.
    /// Stores compressed weights in `atqs_compressed` cache.
    pub fn compress_atqs(&mut self) -> TransformerResult<()> {
        use crate::atqs::WeightsAtqs;
        let start = std::time::Instant::now();
        let weights = WeightsAtqs::compress(self)?;
        let elapsed = start.elapsed();
        let num_weights = 2 + self.blocks.len() * 7
            + self.blocks.iter().filter_map(|b| b.experts.as_ref()).map(|e| e.len() * 3).sum::<usize>();
        tracing::info!(
            "ATQS: compressed {} weight matrices in {:.2}s",
            num_weights,
            elapsed.as_secs_f32()
        );
        self.atqs_compressed = Some(weights);
        Ok(())
    }

    /// Drop all f32 weight arrays (token_embedding, lm_head, block weights).
    /// Only safe if `atqs_compressed` is populated (call `compress_atqs()` first).
    /// Reduces memory usage by ~75% for weights.
    pub fn free_weights(&mut self) {
        self.token_embedding = None;
        self.lm_head = None;
        for block in &mut self.blocks {
            block.attention.wq = None;
            block.attention.wk = None;
            block.attention.wv = None;
            block.attention.wo = None;
            block.ffn.w1 = None;
            block.ffn.w2 = None;
            block.ffn.w3 = None;
            if let Some(ref mut experts) = block.experts {
                for expert in experts {
                    expert.w1 = None;
                    expert.w2 = None;
                    expert.w3 = None;
                }
            }
        }
        tracing::info!("ATQS: freed f32 weight memory");
    }

    /// Restore all f32 weights from the ATQS compressed cache.
    /// Requires `atqs_compressed` to be populated.
    pub fn restore_weights(&mut self) -> TransformerResult<()> {
        use crate::atqs::WeightsAtqs;
        let cache = self.atqs_compressed.take().ok_or_else(|| {
            TransformerError::Implementation(
                "ATQS: restore_weights called but atqs_compressed is None. Call compress_atqs() first.".into()
            )
        })?;
        let start = std::time::Instant::now();
        let num_weights = 2 + self.blocks.len() * 7
            + self.blocks.iter().filter_map(|b| b.experts.as_ref()).map(|e| e.len() * 3).sum::<usize>();
        cache.restore(self)?;
        self.atqs_compressed = Some(cache);
        let elapsed = start.elapsed();
        tracing::info!(
            "ATQS: restored {} f32 weight matrices in {:.2}s",
            num_weights,
            elapsed.as_secs_f32()
        );
        Ok(())
    }

    /// One-shot: compress, free, and flag for ATQS usage.
    /// After this call, f32 weights are freed and only compressed cache remains.
    /// Call `restore_weights()` before any forward pass.
    pub fn enable_atqs(&mut self) -> TransformerResult<()> {
        self.compress_atqs()?;
        self.free_weights();
        self.use_atqs = true;
        tracing::info!("ATQS: enabled — f32 weights freed, compressed cache active");
        Ok(())
    }
}
