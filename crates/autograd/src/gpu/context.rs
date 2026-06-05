use std::collections::HashMap;
use std::sync::Mutex;

use crate::gpu_caps::GpuCapabilities;

use super::gpu_types::*;
#[cfg(feature = "cuda")]
use super::cuda::CudaTensor;
/// Maximum workgroups per dimension for wgpu/WebGPU (65535).
/// Any dispatch exceeding this must be chunked across multiple dispatches
/// using buffer-slice bindings with per-chunk offsets.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Default workgroup size used by most Nexora WGSL shaders.
const DEFAULT_WORKGROUP_SIZE: u32 = 256;

impl GpuContext {
    // ── Initialization ────────────────────────────────────────────────────────

    fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        Self::from_adapter(adapter).await
    }

    /// Create a GpuContext from an already-retrieved wgpu Adapter.
    /// Used by both the single-device [`Self::new`] and multi-device [`GpuDeviceManager`].
    pub(crate) async fn from_adapter(adapter: wgpu::Adapter) -> Result<Self, GpuError> {
        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        }
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        // Enable FP16/BF16 if supported
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if adapter_features.contains(wgpu::Features::SHADER_F64) {
            required_features |= wgpu::Features::SHADER_F64;
        }
        if adapter_features.contains(wgpu::Features::PIPELINE_CACHE) {
            required_features |= wgpu::Features::PIPELINE_CACHE;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Nexora GPU Device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let caps = GpuCapabilities::detect(&device, &adapter);
        let memory_pool = Mutex::new(crate::gpu_memory::GpuMemoryPool::new(&device));

        // ── Persistent pipeline cache setup ──
        let adapter_info = adapter.get_info();

        // Unified memory coordinator — estimate VRAM from device type
        let vram_total: u64 = match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu => 24_000_000_000,
            wgpu::DeviceType::IntegratedGpu => 16_000_000_000,
            _ => 8_000_000_000,
        };
        let memory_coordinator = Mutex::new(crate::gpu_memory::MemoryCoordinator::new(vram_total));
        // Include driver version in cache key so driver updates invalidate the cache
        let driver_version = &adapter_info.driver_info;
        let cache_key = {
            use std::hash::{Hash, Hasher};
            match wgpu::util::pipeline_cache_key(&adapter_info) {
                Some(key_str) => {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    key_str.hash(&mut hasher);
                    driver_version.hash(&mut hasher);
                    const WGPU_CACHE_VERSION: u64 = 2;
                    WGPU_CACHE_VERSION.hash(&mut hasher);
                    hasher.finish()
                }
                None => {
                    // Non-Vulkan backend: hash name + backend + driver as fallback key
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    adapter_info.name.hash(&mut hasher);
                    adapter_info.backend.hash(&mut hasher);
                    driver_version.hash(&mut hasher);
                    const FALLBACK_CACHE_VERSION: u64 = 2;
                    FALLBACK_CACHE_VERSION.hash(&mut hasher);
                    hasher.finish()
                }
            }
        };
        let disk_cache = crate::persistent_cache::PipelineDiskCache::new();
        let cached_data = disk_cache.load(cache_key);
        let wgpu_cache = if device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            if let Some(ref data) = cached_data {
                Some(unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("nexora_persistent_cache"),
                        data: Some(data),
                        fallback: false,
                    })
                })
            } else {
                Some(unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("nexora_persistent_cache"),
                        data: None,
                        fallback: false,
                    })
                })
            }
        } else {
            None
        };

        // Create larger query set for ring buffer (e.g., 1000 timestamp pairs = 2000 queries)
        let query_pool_size: usize = 1000;
        let profiling_query_set = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profiling_query_set_ring_buffer"),
                count: (query_pool_size * 2) as u32, // 2 queries per operation (start, end)
                ty: wgpu::QueryType::Timestamp,
            }))
        } else {
            None
        };

        let auto_flush_ops = if caps.device_type == wgpu::DeviceType::DiscreteGpu {
            256 // NVIDIA/AMD: batch more before flush
        } else {
            64 // Integrated: avoid TDR
        };

        let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nexora_reusable_encoder"),
        });

        // Prefer CUDA backend if feature is enabled and the CUDA device can be initialized.
        #[cfg(feature = "cuda")]
        let cuda_runtime = crate::gpu::cuda::CudaRuntime::try_init();
        #[cfg(feature = "cuda")]
        let backend = if cuda_runtime.is_some() {
            GpuBackend::Cuda
        } else {
            GpuBackend::Wgpu
        };
        #[cfg(not(feature = "cuda"))]
        let backend = GpuBackend::Wgpu;

        let mut ctx = Self {
            device,
            queue,
            caps,
            pipelines: HashMap::new(),
            shader_cache: HashMap::new(),
            bind_group_layout_cache: HashMap::new(),
            bind_group_cache_mutex: Mutex::new(HashMap::new()),
            memory_pool,
            memory_coordinator,
            profiling_query_set,
            query_pool_size,
            query_index: std::sync::atomic::AtomicUsize::new(0),
            query_resolve_buf: None,
            query_readback_buf: None,
            current_encoder: Mutex::new(Some(enc)),
            ops_since_flush: std::sync::atomic::AtomicUsize::new(0),
            auto_flush_ops,
            batch_depth: std::sync::atomic::AtomicUsize::new(1), // batch mode ON by default
            wgpu_cache,
            disk_cache: Some(disk_cache),
            cache_key,
            backend,
            readback_limiter: ReadbackLimiter::new(MAX_PENDING_READBACKS),
            rebuild_lock: Mutex::new(()),
            #[cfg(feature = "cuda")]
            cuda: cuda_runtime,
        };

        ctx.compile_all_pipelines()?;

        // Save persistent pipeline cache to disk after all pipelines are compiled.
        // If the cache was loaded from disk and reused, get_data() returns the same data
        // (no driver compilation happened). If it was a cache miss, the new data is saved.
        if let Some(ref cache) = ctx.wgpu_cache {
            if let Some(data) = cache.get_data() {
                if let Some(ref disk) = ctx.disk_cache {
                    disk.save(ctx.cache_key, &data);
                }
            }
        }

        Ok(ctx)
    }

    pub(crate) fn compile_all_pipelines(&mut self) -> Result<(), GpuError> {
        let tile = self.caps.optimal_tile_size();
        self.compile_matmul_tiled(tile)?;
        self.compile_matmul_int8_tiled(tile)?;
        self.compile_matmul_int8_weight(tile)?;
        self.compile_matmul_int4_weight(tile)?;
        self.compile_matmul_f16_tiled(tile)?;
        self.compile_elementwise()?;
        self.compile_reduce(ReduceOp::Sum)?;
        self.compile_reduce(ReduceOp::Max)?;
        self.compile_reduce(ReduceOp::Min)?;
        self.compile_softmax()?;
        self.compile_rms_norm()?;
        self.compile_rms_norm_backward()?;
        self.compile_layer_norm_backward()?;
        self.compile_cross_entropy()?;
        self.compile_cross_entropy_backward()?;
        self.compile_embedding()?;
        self.compile_embedding_backward()?;
        self.compile_layer_norm()?;
        self.compile_transpose()?;
        self.compile_fused_attention()?;
        self.compile_fused_attention_backward()?;
        self.compile_fill_zero()?;
        self.compile_fill_constant()?;
        self.compile_fill_zero_u32()?;
        self.compile_scale_inplace()?;
        self.compile_gradient_allreduce()?;
        self.compile_elementwise_inplace()?;
        self.compile_causal_softmax()?;
        self.compile_l2_norm()?;
        self.compile_gradient_clip()?;
        self.compile_temperature_scale()?;
        self.compile_top_k_mask()?;
        self.compile_top_p_mask()?;
        self.compile_multinomial_sample()?;
        self.compile_dropout_mask()?;
        // Mixed precision converters
        use crate::gpu_mixed::{
            compile_f16_packed_to_f32_pipeline, compile_f32_to_f16_packed_pipeline,
        };
        compile_f32_to_f16_packed_pipeline(self)?;
        compile_f16_packed_to_f32_pipeline(self)?;
        // Phase 2: Transformer ops (RoPE, repeat_heads)
        self.compile_rotary_embedding()?;
        self.compile_repeat_heads()?;
        // MoE scatter-add (wgpu fused expert path)
        self.compile_moe_scatter_add()?;
        // Priority 1: Fused ops (matmul+bias+activation, online softmax)
        self.compile_fused_ops()?;
        // Phase 4: Training pipelines
        self.compile_adam_step()?;
        // Phase 5: SEDC compression pipelines
        self.compile_sedc_pipelines()?;
        Ok(())
    }


    pub fn init() -> Result<&'static Self, GpuError> {
        GPU_CTX.get_or_try_init(Self::new_blocking)
    }

    pub fn global() -> Result<&'static Self, GpuError> {
        GPU_CTX.get().ok_or(GpuError::NotInitialized)
    }

    pub fn is_available() -> bool {
        GPU_CTX.get().is_some()
    }

    pub fn adapter_info(&self) -> GpuAdapterInfo {
        GpuAdapterInfo {
            name: self.caps.device_name.clone(),
            backend: self.caps.backend,
            compute_backend: self.backend,
        }
    }

    /// Flush the reusable encoder: submit all accumulated dispatches at once.
    /// Call this before any readback to ensure GPU has finished.
    pub fn flush(&self) {
        let mut guard = self.current_encoder.lock().unwrap_or_else(|e| {
            tracing::warn!("Lock poisoned: {}", e);
            e.into_inner()
        });
        if let Some(enc) = guard.take() {
            self.queue.submit(Some(enc.finish()));
            *guard = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("nexora_reusable_encoder"),
                    }),
            );
        }
        self.ops_since_flush
            .store(0, std::sync::atomic::Ordering::Release);
    }

}
