use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use crate::gpu_caps::GpuCapabilities;

use super::gpu_tensor::{readback_with_timeout, GpuDtype, GpuTensor};
use super::gpu_types::*;

/// Maximum workgroups per dimension for wgpu/WebGPU (65535).
/// Any dispatch exceeding this must be chunked across multiple dispatches
/// using buffer-slice bindings with per-chunk offsets.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Default workgroup size used by most Nexora WGSL shaders.
const DEFAULT_WORKGROUP_SIZE: u32 = 256;

impl GpuContext {
    // ── Memory pool helpers ─────────────────────────────────────────────────────

    /// Allocate a buffer from the memory pool with automatic bucket sizing
    pub fn alloc_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> crate::gpu_memory::PooledBuffer {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .alloc(size, usage)
    }

    /// Allocate a buffer through the memory pool, falling back to
    /// `device.create_buffer` if the pool is unavailable.
    pub fn alloc_or_create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.alloc_buffer(size, usage).buffer
    }

    /// Return a buffer to the memory pool for reuse
    pub fn dealloc_buffer(&self, buf: crate::gpu_memory::PooledBuffer) {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .dealloc(buf);
    }

    /// Get memory pool statistics
    pub fn memory_stats(&self) -> crate::gpu_memory::PoolStats {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .stats()
            .clone()
    }

    /// Clear the memory pool (free all cached buffers)
    pub fn clear_memory_pool(&self) {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .clear();
    }

    /// Batch dispatch multiple operations with single sync at the end
    /// Returns total GPU execution time if timestamp queries are available
    pub fn batch_dispatch<F>(&self, ops: F) -> Result<Option<std::time::Duration>, GpuError>
    where
        F: FnOnce(&mut wgpu::CommandEncoder) -> Result<(), GpuError>,
    {
        if let Some(query_set) = &self.profiling_query_set {
            let (start_idx, end_idx) = self.next_query_index();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_dispatch_encoder"),
                });

            encoder.write_timestamp(query_set, start_idx);
            ops(&mut encoder)?;
            encoder.write_timestamp(query_set, end_idx);

            let query_count = 2;
            let buf_size = ((query_count as u64 * wgpu::QUERY_SIZE as u64 + 255) / 256) * 256;
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_timestamp_query_buffer"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            encoder.resolve_query_set(query_set, start_idx..end_idx + 1, &query_buffer, 0);

            let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_timestamp_readback"),
                size: buf_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            encoder.copy_buffer_to_buffer(&query_buffer, 0, &readback_buffer, 0, buf_size);

            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = readback_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                })
                .map_err(|_| GpuError::Timeout("GPU poll timed out after 5s".into()))?;

            let result = self.read_timestamps_from_buffer(&buffer_slice, &readback_buffer);
            readback_buffer.destroy();

            let gpu_duration = result.map(|(start, end)| {
                let timestamp_period = self.queue.get_timestamp_period();
                std::time::Duration::from_secs_f64(
                    (end - start) / 1_000_000_000.0 * timestamp_period as f64,
                )
            });

            Ok(gpu_duration)
        } else {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_dispatch_encoder"),
                });

            ops(&mut encoder)?;
            self.queue.submit(Some(encoder.finish()));

            Ok(None)
        }
    }

    // ── Cache helpers ─────────────────────────────────────────────────────────

    fn get_or_create_shader_module(
        &mut self,
        wgsl: std::borrow::Cow<'static, str>,
        label: &str,
    ) -> wgpu::ShaderModule {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        wgsl.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(shader) = self.shader_cache.get(&hash) {
            return shader.clone();
        }

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl),
            });

        self.shader_cache.insert(hash, shader.clone());
        shader
    }

    fn get_or_create_bind_group_layout(
        &mut self,
        entries: &[wgpu::BindGroupLayoutEntry],
        label: &str,
    ) -> wgpu::BindGroupLayout {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for entry in entries {
            entry.binding.hash(&mut hasher);
            entry.visibility.hash(&mut hasher);
            if let wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } = entry.ty
            {
                match ty {
                    wgpu::BufferBindingType::Uniform => {}
                    wgpu::BufferBindingType::Storage { read_only } => {
                        read_only.hash(&mut hasher);
                    }
                }
                has_dynamic_offset.hash(&mut hasher);
                if let Some(size) = min_binding_size {
                    size.get().hash(&mut hasher);
                }
            }
        }
        let hash = hasher.finish();

        if let Some(layout) = self.bind_group_layout_cache.get(&hash) {
            return layout.clone();
        }

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });

        self.bind_group_layout_cache.insert(hash, layout.clone());
        layout
    }

    // ── Pipeline compilation helper ────────────────────────────────────────────

    /// Public API untuk compile pipeline dengan cache
    pub fn compile_pipeline_cached(
        &mut self,
        name: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        wgsl: std::borrow::Cow<'static, str>,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, GpuError> {
        // Check if pipeline already exists in cache
        if let Some(cached) = self.pipelines.get(name) {
            return Ok(cached.pipeline.clone());
        }

        let layout = self.get_or_create_bind_group_layout(entries, &format!("{}_layout", name));
        let shader = self.get_or_create_shader_module(wgsl, &format!("{}_shader", name));

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline_layout", name)),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: self.wgpu_cache.as_ref(),
            });

        self.pipelines.insert(
            name.into(),
            CompiledPipeline {
                pipeline: pipeline.clone(),
                bind_group_layout: layout,
            },
        );

        Ok(pipeline)
    }

    pub(crate) fn compile_pipeline(
        &mut self,
        name: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        wgsl: std::borrow::Cow<'static, str>,
        entry_point: &str,
    ) -> Result<(), GpuError> {
        let layout = self.get_or_create_bind_group_layout(entries, &format!("{}_layout", name));
        let shader = self.get_or_create_shader_module(wgsl, &format!("{}_shader", name));

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline_layout", name)),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: self.wgpu_cache.as_ref(),
            });

        self.pipelines.insert(
            name.into(),
            CompiledPipeline {
                pipeline,
                bind_group_layout: layout,
            },
        );
        Ok(())
    }

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
            64 // NVIDIA/AMD: batch more before flush
        } else {
            16 // Integrated: avoid TDR
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
        // Mixed precision converters
        use crate::gpu_mixed::{
            compile_f16_packed_to_f32_pipeline, compile_f32_to_f16_packed_pipeline,
        };
        compile_f32_to_f16_packed_pipeline(self)?;
        compile_f16_packed_to_f32_pipeline(self)?;
        // Phase 2: Transformer ops (RoPE, repeat_heads)
        self.compile_rotary_embedding()?;
        self.compile_repeat_heads()?;
        // Priority 1: Fused ops (matmul+bias+activation, online softmax)
        self.compile_fused_ops()?;
        // Phase 4: Training pipelines
        self.compile_adam_step()?;
        // Phase 5: SEDC compression pipelines
        self.compile_sedc_pipelines()?;
        Ok(())
    }

    // ── Sampler / utility helpers ──────────────────────────────────────────────

    fn compile_fill_zero(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fill_zero",
            &[storage_binding(0, false)],
            std::borrow::Cow::Borrowed(FILL_ZERO_WGSL),
            "fill_zero_main",
        )
    }

    fn compile_fill_constant(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fill_constant",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(FILL_CONSTANT_WGSL),
            "fill_constant_main",
        )
    }

    fn compile_fill_zero_u32(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fill_zero_u32",
            &[storage_binding(0, false)],
            std::borrow::Cow::Borrowed(FILL_ZERO_U32_WGSL),
            "fill_zero_u32_main",
        )
    }

    fn compile_scale_inplace(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "scale_inplace",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(SCALE_INPLACE_WGSL),
            "scale_inplace_main",
        )
    }

    fn compile_gradient_allreduce(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "gradient_allreduce",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(GRADIENT_ALLREDUCE_WGSL),
            "gradient_allreduce_main",
        )
    }

    /// Average gradients across N model replicas.
    /// `grad_buffers` must be a flat buffer containing [replica0_grads | replica1_grads | ...]
    /// where each replica's gradients are `numel` f32 values.
    /// `out` must be sized for `numel` f32 values.
    pub fn gradient_allreduce(
        &self,
        grad_buffers: &GpuTensor,
        out: &GpuTensor,
        num_replicas: u32,
    ) -> Result<(), GpuError> {
        let numel = out.numel() as u32;
        let pipeline = self
            .pipelines
            .get("gradient_allreduce")
            .ok_or_else(|| GpuError::Pipeline("gradient_allreduce not compiled".into()))?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gradient_allreduce_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, num_replicas, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, grad_buffers.buffer()), (1, out.buffer())],
            &[(2, &cfg_buf)],
            numel, 256, 4,
        );
        Ok(())
    }

    fn compile_causal_softmax(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "causal_softmax",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(CAUSAL_SOFTMAX_WGSL),
            "causal_softmax_main",
        )
    }

    fn compile_l2_norm(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "l2_norm",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(L2_NORM_WGSL),
            "l2_norm_main",
        )
    }

    fn compile_gradient_clip(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "gradient_clip",
            &[
                storage_binding(0, false),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(GRADIENT_CLIP_WGSL),
            "main",
        )
    }

    fn compile_temperature_scale(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "temperature_scale",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(TEMPERATURE_SCALE_WGSL),
            "temperature_scale_main",
        )
    }

    fn compile_top_k_mask(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "top_k_mask",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(TOP_K_MASK_WGSL),
            "top_k_mask_main",
        )
    }

    fn compile_top_p_mask(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "top_p_mask",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(TOP_P_MASK_WGSL),
            "top_p_mask_main",
        )
    }

    fn compile_multinomial_sample(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "multinomial_sample",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(MULTINOMIAL_SAMPLE_WGSL),
            "multinomial_main",
        )
    }

    /// Zero out a GPU tensor buffer in-place.
    pub fn fill_zero(&self, t: &GpuTensor) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_zero")
            .ok_or_else(|| GpuError::Pipeline("fill_zero not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[], numel, 256, 4);
        Ok(())
    }

    /// Zero out a GPU buffer using u32 writes — works for any dtype.
    /// Dispatches ceil(byte_size / 4 / 256) workgroups, each writes 0u per u32.
    pub fn fill_zero_u32(&self, t: &GpuTensor) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_zero_u32")
            .ok_or_else(|| GpuError::Pipeline("fill_zero_u32 not compiled".into()))?;
        let buf_size = t.buffer().size();
        let u32_count = u32::try_from(buf_size / 4).unwrap_or(u32::MAX);
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[], u32_count, 256, 4);
        Ok(())
    }

    /// Fill a GPU tensor buffer with a constant value in-place.
    pub fn fill_constant(&self, t: &GpuTensor, value: f32) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_constant")
            .ok_or_else(|| GpuError::Pipeline("fill_constant not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fill_const_cfg"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 2] = [numel, f32::to_bits(value)];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[(1, &cfg_buf)], numel, 256, 4);
        Ok(())
    }

    /// Scale a GPU tensor buffer in-place (multiply each element by `scale`).
    pub fn scale_inplace(&self, t: &GpuTensor, scale: f32) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("scale_inplace")
            .ok_or_else(|| GpuError::Pipeline("scale_inplace not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scale_cfg"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 2] = [numel, f32::to_bits(scale)];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        self.dispatch_1d_chunked(pipeline, &[(0, t.buffer())], &[(1, &cfg_buf)], numel, 256, 4);
        Ok(())
    }

    /// Compute L2 norm (sum of squares) for a GPU tensor. Returns scalar tensor.
    pub fn l2_norm(&self, t: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let pipeline = self
            .pipelines
            .get("l2_norm")
            .ok_or_else(|| GpuError::Pipeline("l2_norm not compiled".into()))?;
        let numel = u32::try_from(t.numel()).unwrap_or(u32::MAX);
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("l2_norm_out"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("l2_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [numel, 0, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: t.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch(pipeline, &bg, (1, 1, 1));
        Ok(GpuTensor {
            shape: vec![1],
            buffer: out,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Causal softmax: softmax over last dim with causal mask.
    pub fn causal_softmax(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        let batch = u32::try_from(shape[0]).unwrap_or(u32::MAX);
        let dim = u32::try_from(shape[1]).unwrap_or(u32::MAX);
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("causal_softmax_out"),
            size: (input.numel() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cs_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [batch, dim, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));
        let pipeline = self
            .pipelines
            .get("causal_softmax")
            .ok_or_else(|| GpuError::Pipeline("causal_softmax not compiled".into()))?;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cs_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch(pipeline, &bg, (batch, 1, 1));
        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// GPU sampling: temperature → softmax → top-k → top-p → multinomial
    pub fn gpu_sample(
        &self,
        logits: &GpuTensor,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<GpuTensor, GpuError> {
        let shape = logits.shape();
        let batch = shape[0];
        let vocab = u32::try_from(shape[1]).unwrap_or(u32::MAX);
        let numel = logits.numel();

        // Working copy of logits (pool-allocated)
        let mut work_buf = {
            let pooled = self.alloc_buffer(
                (numel * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            self.with_encoder(|enc| {
                enc.copy_buffer_to_buffer(
                    logits.buffer(),
                    0,
                    &pooled.buffer,
                    0,
                    (numel * 4) as u64,
                );
            });
            pooled.buffer
        };

        // Step 1: temperature scale (in-place on work_buf)
        if (temperature - 1.0).abs() > 1e-6 && temperature > 0.0 {
            let pipeline = self
                .pipelines
                .get("temperature_scale")
                .ok_or_else(|| GpuError::Pipeline("temperature_scale not compiled".into()))?;
            let temp_buf = self.alloc_buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let temp_cfg: [u32; 4] = [
                f32::to_bits(temperature),
                u32::try_from(numel).unwrap_or(u32::MAX),
                0,
                0,
            ];
            self.queue
                .write_buffer(&temp_buf.buffer, 0, bytemuck::cast_slice(&temp_cfg));
            let workgroups = u32::try_from(numel).unwrap_or(u32::MAX);
            self.dispatch_1d_chunked(
                pipeline,
                &[(0, &work_buf)],
                &[(1, &temp_buf.buffer)],
                workgroups, 256, 4,
            );
        }

        // Step 2: softmax (via ctx.softmax — uses reusable encoder internally)
        let logits_gpu = GpuTensor {
            shape: shape.clone(),
            buffer: work_buf,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let probs = self.softmax(&logits_gpu)?;

        // Copy probs → fresh work buffer for top-k mask (avoids mutating softmax output)
        let mut masked_buf = {
            let pooled = self.alloc_buffer(
                (numel * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            self.with_encoder(|enc| {
                enc.copy_buffer_to_buffer(probs.buffer(), 0, &pooled.buffer, 0, (numel * 4) as u64);
            });
            pooled.buffer
        };

        // Step 3: top-k mask (in-place on masked_buf)
        let masked = if top_k > 0 && top_k < vocab {
            let pipeline = self
                .pipelines
                .get("top_k_mask")
                .ok_or_else(|| GpuError::Pipeline("top_k_mask not compiled".into()))?;
            let cfg_buf = self.alloc_buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let cfg: [u32; 4] = [vocab, top_k, 0, 0];
            self.queue
                .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("topk_bg"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: masked_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cfg_buf.buffer.as_entire_binding(),
                    },
                ],
            });
            self.dispatch(pipeline, &bg, (batch as u32, 1, 1));
            GpuTensor {
                shape: shape.clone(),
                buffer: masked_buf,
                dtype: GpuDtype::F32,
                device_id: 0,
            }
        } else {
            GpuTensor {
                shape: shape.clone(),
                buffer: masked_buf,
                dtype: GpuDtype::F32,
                device_id: 0,
            }
        };

        // Step 3b: top-p mask (in-place on masked buffer, after top-k)
        if top_p > 0.0 && top_p < 1.0 {
            let pipeline = self
                .pipelines
                .get("top_p_mask")
                .ok_or_else(|| GpuError::Pipeline("top_p_mask not compiled".into()))?;
            let cfg_buf = self.alloc_buffer(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let cfg: [u32; 4] = [vocab, f32::to_bits(top_p), 0, 0];
            self.queue
                .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("topp_bg"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: masked.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cfg_buf.buffer.as_entire_binding(),
                    },
                ],
            });
            self.dispatch(pipeline, &bg, (batch as u32, 1, 1));
        }

        // Step 4: multinomial sample
        let pipeline = self
            .pipelines
            .get("multinomial_sample")
            .ok_or_else(|| GpuError::Pipeline("multinomial_sample not compiled".into()))?;
        let out = self.alloc_buffer(
            (batch * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let cfg_buf = self.alloc_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg: [u32; 4] = [vocab, seed as u32, (seed >> 32) as u32, 0];
        self.queue
            .write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sample_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: masked.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.buffer.as_entire_binding(),
                },
            ],
        });
        self.dispatch(pipeline, &bg, (batch as u32, 1, 1));

        Ok(GpuTensor {
            shape: vec![batch],
            buffer: out.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// GPU sampling from F16 input: converts F16→F32 on GPU, then runs standard
    /// f32 sampling pipeline. Avoids CPU-side F16→F32 conversion and re-upload.
    pub fn gpu_sample_f16(
        &self,
        logits_f16: &GpuTensor,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<GpuTensor, GpuError> {
        let logits_f32 = self.f16_packed_to_f32(logits_f16)?;
        self.gpu_sample(&logits_f32, temperature, top_k, top_p, seed)
    }

    /// Batched GPU sampling: stack B logit vectors into [B, vocab] and sample
    /// once on GPU via F16→F32 conversion. Returns one token ID per sequence.
    /// All computation stays on GPU — only token IDs are read back.
    pub fn gpu_sample_batched_f16(
        &self,
        batch_logits: &[&GpuTensor],
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<GpuTensor, GpuError> {
        if batch_logits.is_empty() {
            return Ok(GpuTensor {
                shape: vec![0],
                buffer: self.alloc_buffer(0, wgpu::BufferUsages::COPY_SRC).buffer,
                dtype: GpuDtype::F32,
                device_id: 0,
            });
        }
        // Convert each F16 tensor to F32 on GPU
        let f32_tensors: Vec<GpuTensor> = batch_logits
            .iter()
            .map(|t| self.f16_packed_to_f32(t))
            .collect::<Result<Vec<_>, _>>()?;

        let batch = f32_tensors.len();
        let vocab = f32_tensors[0].numel();
        let total_elems = batch * vocab;

        // Allocate contiguous [B, vocab] F32 buffer
        let flat_buf = self.alloc_buffer(
            (total_elems * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        self.with_encoder(|enc| {
            let mut offset: u64 = 0;
            for t in &f32_tensors {
                enc.copy_buffer_to_buffer(
                    t.buffer(),
                    0,
                    &flat_buf.buffer,
                    offset,
                    (vocab * 4) as u64,
                );
                offset += (vocab * 4) as u64;
            }
        });

        let stacked = GpuTensor {
            shape: vec![batch, vocab],
            buffer: flat_buf.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.gpu_sample(&stacked, temperature, top_k, top_p, seed)
    }

    // ── Singleton access ──────────────────────────────────────────────────────

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

    /// Sync GPU and CPU (blocking wait with 30s timeout).
    /// Call after `flush()` before reading results.
    ///
    /// ## ⚠️ Blocking
    /// Blocks the current thread for up to 30s. Must be called from
    /// `spawn_blocking` if used in async context.
    pub fn sync(&self) {
        let start = std::time::Instant::now();
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        });
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        crate::gpu::gpu_observability::GPU_BUSY_NS
            .fetch_add(elapsed_ns, std::sync::atomic::Ordering::Relaxed);
    }

    /// Non-blocking poll — proses callback async tanpa blocking
    pub fn poll_device(&self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Blocking wait with 30s timeout — tunggu semua GPU work selesai.
    ///
    /// ## ⚠️ Blocking
    /// Blocks the current thread for up to 30s. Must be called from
    /// `spawn_blocking` if used in async context.
    pub fn wait_device(&self) {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        });
    }

    /// Register callback ketika semua submitted work selesai
    pub fn on_submitted_work_done(&self, callback: impl FnOnce() + Send + 'static) {
        self.queue.on_submitted_work_done(callback);
    }

    pub(crate) fn with_encoder<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut wgpu::CommandEncoder) -> R,
    {
        let mut guard = self.current_encoder.lock().unwrap_or_else(|e| {
            tracing::warn!("Lock poisoned: {}", e);
            e.into_inner()
        });
        let enc = guard.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nexora_reusable_encoder"),
                })
        });
        let result = f(enc);

        if self.batch_depth.load(std::sync::atomic::Ordering::Acquire) == 0 {
            let ops = self
                .ops_since_flush
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1;
            if ops >= self.auto_flush_ops {
                if let Some(enc_to_submit) = guard.take() {
                    self.queue.submit(Some(enc_to_submit.finish()));
                    self.ops_since_flush
                        .store(0, std::sync::atomic::Ordering::Release);
                    *guard = Some(self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("nexora_reusable_encoder"),
                        },
                    ));
                }
            }
        }
        result
    }

    /// Begin batch mode: suppress auto-flush so all subsequent dispatches
    /// accumulate into the current encoder. Call end_batch_mode() to submit.
    pub fn begin_batch_mode(&self) {
        self.batch_depth
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// End batch mode and flush accumulated dispatches.
    /// Safe to call nested — only the outermost decrement triggers flush.
    pub fn end_batch_mode(&self) {
        let prev = self
            .batch_depth
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        if prev == 1 {
            self.flush();
        }
    }

    /// Dispatch a compute shader using the reusable command encoder.
    /// Accumulates dispatches in the reusable encoder instead of creating a new encoder per op.
    /// Automatically flushes (submits + recreates) after `auto_flush_ops` dispatches.
    ///
    /// # Panics
    /// Panics with a clear diagnostic if any workgroup dimension exceeds
    /// `MAX_WORKGROUPS_PER_DIM` (65535). Use [`dispatch_1d_chunked`] for
    /// element-wise ops on large tensors — it automatically splits into
    /// multiple dispatches with buffer-slice bindings.
    pub(crate) fn dispatch(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let (gx, gy, gz) = workgroups;
        let gx = gx.max(1);
        let gy = gy.max(1);
        let gz = gz.max(1);

        assert!(
            gx <= MAX_WORKGROUPS_PER_DIM
                && gy <= MAX_WORKGROUPS_PER_DIM
                && gz <= MAX_WORKGROUPS_PER_DIM,
            "GPU dispatch overflow: workgroups ({}, {}, {}) exceeds limit {} per dimension. \
             This is caused by dispatching {} workgroups in X (with workgroup_size={}, \
             numel≈{}). Use dispatch_1d_chunked() for element-wise ops or reduce tensor size.",
            gx, gy, gz,
            MAX_WORKGROUPS_PER_DIM,
            gx, DEFAULT_WORKGROUP_SIZE, gx as u64 * DEFAULT_WORKGROUP_SIZE as u64,
        );

        self.dispatch_impl(pipeline, bind_group, (gx, gy, gz));
    }

    /// Raw dispatch without validation — called by `dispatch` and `dispatch_1d_chunked`.
    fn dispatch_impl(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        self.with_encoder(|enc| {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        });
    }

    /// Safe 1D element-wise dispatch with automatic chunking.
    ///
    /// When the total workgroup count exceeds `MAX_WORKGROUPS_PER_DIM` (65535),
    /// automatically splits into multiple dispatches. Each chunk uses a
    /// buffer-slice [`wgpu::BufferBinding`] with the appropriate byte offset,
    /// so shaders using `@builtin(global_invocation_id).x` to index into
    /// storage buffers process a different slice per chunk — no shader changes
    /// are required.
    ///
    /// # Arguments
    /// * `storage_bindings` — `(binding_index, &buffer)` for every storage buffer
    ///   that must be sliced per chunk (e.g. input, output, or in-place data).
    ///   All sliced buffers must use the same `element_size`.
    /// * `uniform_bindings` — `(binding_index, &buffer)` for uniform/config buffers
    ///   that are shared across all chunks (no slicing).
    /// * `numel` — total number of logical elements to process.
    /// * `workgroup_size` — the WGSL `@workgroup_size(...)` value (typically 256).
    /// * `element_size` — bytes per element (4 for f32/u32, 2 for f16, 1 for i8).
    pub(crate) fn dispatch_1d_chunked(
        &self,
        pipeline: &CompiledPipeline,
        storage_bindings: &[(u32, &wgpu::Buffer)],
        uniform_bindings: &[(u32, &wgpu::Buffer)],
        numel: u32,
        workgroup_size: u32,
        element_size: u64,
    ) {
        let wg_size = workgroup_size.max(1);
        let total_groups = (numel + wg_size - 1) / wg_size;

        if total_groups <= MAX_WORKGROUPS_PER_DIM {
            let mut entries = Vec::with_capacity(storage_bindings.len() + uniform_bindings.len());
            for &(binding, buf) in storage_bindings {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buf.as_entire_binding(),
                });
            }
            for &(binding, buf) in uniform_bindings {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buf.as_entire_binding(),
                });
            }
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dispatch_1d"),
                layout: &pipeline.bind_group_layout,
                entries: &entries,
            });
            self.dispatch_impl(pipeline, &bg, (total_groups, 1, 1));
            return;
        }

        // Chunked path: split into MAX_WORKGROUPS_PER_DIM workgroups per chunk
        let elements_per_chunk = MAX_WORKGROUPS_PER_DIM * wg_size;
        let mut remaining = numel as i64;
        let mut byte_offset: u64 = 0;

        while remaining > 0 {
            let chunk_elems = (remaining as u32).min(elements_per_chunk);
            let chunk_groups = (chunk_elems + wg_size - 1) / wg_size;
            let chunk_bytes = (chunk_elems as u64) * element_size;

            let mut entries = Vec::with_capacity(storage_bindings.len() + uniform_bindings.len());
            for &(binding, buf) in storage_bindings {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: byte_offset,
                        size: Some(wgpu::BufferSize::new(chunk_bytes).unwrap()),
                    }),
                });
            }
            for &(binding, buf) in uniform_bindings {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buf.as_entire_binding(),
                });
            }

            let chunk_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dispatch_1d_chunk"),
                layout: &pipeline.bind_group_layout,
                entries: &entries,
            });

            self.dispatch_impl(pipeline, &chunk_bg, (chunk_groups, 1, 1));

            remaining -= chunk_elems as i64;
            byte_offset += chunk_bytes;
        }
    }

    /// Slice tensor GPU — mengambil sub-view [pos..pos+len] dari dimension 0
    /// GPU-side copy via buffer-to-buffer (no CPU round-trip).
    pub fn slice_tensor(
        &self,
        tensor: &GpuTensor,
        pos: u32,
        len: u32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = tensor.shape();
        let dim0 = shape[0];
        let rest: usize = shape[1..].iter().product();
        let slice_numel = (len as usize) * rest;
        let element_size = match tensor.dtype {
            GpuDtype::F32 => 4,
            GpuDtype::F16 | GpuDtype::Bf16 => 2,
            GpuDtype::I8 => 1,
        };
        let byte_offset = (pos as usize) * dim0 /* actually per-row */;
        // Actual: each row has `rest` elements
        let row_bytes = (rest * element_size) as u64;
        let byte_offset = (pos as u64) * row_bytes;
        let byte_size = (len as u64) * row_bytes;

        let pooled = self.alloc_buffer(
            byte_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        self.with_encoder(|enc| {
            enc.copy_buffer_to_buffer(tensor.buffer(), byte_offset, &pooled.buffer, 0, byte_size);
        });

        let mut new_shape = shape.clone();
        new_shape[0] = len as usize;

        Ok(GpuTensor {
            shape: new_shape,
            buffer: pooled.buffer,
            dtype: tensor.dtype,
            device_id: tensor.device_id,
        })
    }

    /// Get next query index from ring buffer (wraps around at query_pool_size)
    fn next_query_index(&self) -> (u32, u32) {
        let idx = self
            .query_index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            % self.query_pool_size;
        let start_idx = (idx * 2) as u32;
        let end_idx = (idx * 2 + 1) as u32;
        (start_idx, end_idx)
    }

    /// Get or create bind group from cache
    /// Key is computed from buffer properties to identify unique bind group configurations
    /// Thread-safe bind group cache for &self access.
    /// Hashes buffer (id, offset) pairs for reuse across dispatches.
    /// Designed for hot-path ops where buffers are stable (pre-uploaded weights,
    /// persistent KV cache, reusable memory pool buffers).
    pub(crate) fn get_or_create_bind_group_shared(
        &self,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
        label: &str,
    ) -> wgpu::BindGroup {
        let mut hash: u64 = 0;
        for entry in entries {
            if let wgpu::BindingResource::Buffer(buffer_binding) = &entry.resource {
                hash = hash.wrapping_mul(31).wrapping_add(buffer_binding.offset);
                if let Some(size) = buffer_binding.size {
                    hash = hash.wrapping_mul(31).wrapping_add(size.get());
                }
            }
        }
        // Mix in label hash to distinguish different pipeline layouts
        {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            label.hash(&mut h);
            hash = hash.wrapping_mul(31).wrapping_add(h.finish());
        }

        let mut cache = self.bind_group_cache_mutex.lock().unwrap_or_else(|e| {
            tracing::warn!("Lock poisoned: {}", e);
            e.into_inner()
        });
        if let Some(cached) = cache.get(&hash) {
            return cached.clone();
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        });

        // Evict oldest entries when cache exceeds max size
        const MAX_CACHE_SIZE: usize = 1024;
        if cache.len() >= MAX_CACHE_SIZE {
            cache.clear();
        }
        cache.insert(hash, bind_group.clone());
        bind_group
    }

    /// Safely read timestamp values from a mapped GPU readback buffer.
    /// Returns None if the buffer is too small or mapping fails.
    fn read_timestamps_from_buffer(
        &self,
        slice: &wgpu::BufferSlice,
        readback_buffer: &wgpu::Buffer,
    ) -> Option<(f64, f64)> {
        let mapped = slice.get_mapped_range();
        if mapped.len() < 16 {
            drop(mapped);
            readback_buffer.unmap();
            return None;
        }
        let first = u64::from_le_bytes(mapped[0..8].try_into().ok()?);
        let second = u64::from_le_bytes(mapped[8..16].try_into().ok()?);
        drop(mapped);
        readback_buffer.unmap();
        Some((first as f64, second as f64))
    }

    pub(crate) fn dispatch_profiled(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) -> std::time::Duration {
        if let Some(query_set) = &self.profiling_query_set {
            let (start_idx, end_idx) = self.next_query_index();
            let query_count = 2;
            let buf_size = ((query_count as u64 * wgpu::QUERY_SIZE as u64 + 255) / 256) * 256;
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_buffer"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.write_timestamp(query_set, start_idx);
                cpass.set_pipeline(&pipeline.pipeline);
                cpass.set_bind_group(0, bind_group, &[]);
                cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
                cpass.write_timestamp(query_set, end_idx);
            }
            encoder.resolve_query_set(query_set, start_idx..(end_idx + 1), &query_buffer, 0);
            self.queue.submit(Some(encoder.finish()));

            // Readback timestamp query result (separate from kernel execution)
            let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_readback"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut copy_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            copy_encoder.copy_buffer_to_buffer(&query_buffer, 0, &readback_buffer, 0, buf_size);
            self.queue.submit(Some(copy_encoder.finish()));

            let slice = readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            // Safe readback: on failure, fall through to CPU timing
            if readback_with_timeout(&self.device, &rx).is_ok() {
                let period_ns = self.queue.get_timestamp_period() as f64;
                if let Some((start_ts, end_ts)) =
                    self.read_timestamps_from_buffer(&slice, &readback_buffer)
                {
                    let elapsed_ns = if end_ts >= start_ts {
                        (end_ts - start_ts) * period_ns
                    } else {
                        0.0
                    };
                    return std::time::Duration::from_nanos(elapsed_ns.round() as u64);
                }
            }
        }

        let start = std::time::Instant::now();
        self.dispatch(pipeline, bind_group, workgroups);
        start.elapsed()
    }

    /// Profiled dispatch with detailed timing breakdown per stage
    /// Returns DetailedTiming with encode, submit, gpu_execute, map_async, readback, and total time
    pub fn dispatch_profiled_detailed(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) -> DetailedTiming {
        let mut timing = DetailedTiming::default();
        let total_start = Instant::now();

        // Stage 1: Encode
        let encode_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline.pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        timing.encode = encode_start.elapsed();

        // Stage 2: Submit
        let submit_start = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        timing.submit = submit_start.elapsed();

        // Stage 3: GPU execute (using poll to wait for completion)
        let gpu_start = Instant::now();
        if self
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .is_err()
        {
            tracing::warn!("GPU poll timed out after 5s in dispatch_profiled_detailed");
        }
        timing.gpu_execute = gpu_start.elapsed();

        timing.end_to_end = total_start.elapsed();

        timing
    }

    /// Batch dispatch N times with single timestamp query around all dispatches
    /// Returns total GPU execution time for all N dispatches
    pub fn dispatch_batch_profiled(
        &self,
        pipeline: &CompiledPipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
        n: usize,
    ) -> std::time::Duration {
        if let Some(query_set) = &self.profiling_query_set {
            let (start_idx, end_idx) = self.next_query_index();
            let query_count = 2;
            let buf_size = ((query_count as u64 * wgpu::QUERY_SIZE as u64 + 255) / 256) * 256;
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_buffer_batch"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.write_timestamp(query_set, start_idx); // Start timestamp
                for _ in 0..n {
                    cpass.set_pipeline(&pipeline.pipeline);
                    cpass.set_bind_group(0, bind_group, &[]);
                    cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
                }
                cpass.write_timestamp(query_set, end_idx); // End timestamp
            }
            encoder.resolve_query_set(query_set, start_idx..(end_idx + 1), &query_buffer, 0);

            // Readback timestamp query result (separate from kernel execution)
            let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_readback_batch"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Combine copy into main encoder to avoid extra submit
            encoder.copy_buffer_to_buffer(&query_buffer, 0, &readback_buffer, 0, buf_size);
            self.queue.submit(Some(encoder.finish()));

            let slice = readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            // Safe readback with error fallback: fall through to CPU timing on failure
            if readback_with_timeout(&self.device, &rx).is_ok() {
                if let Some((start_ts, end_ts)) =
                    self.read_timestamps_from_buffer(&slice, &readback_buffer)
                {
                    let period_ns = self.queue.get_timestamp_period() as f64;
                    let elapsed_ns = if end_ts >= start_ts {
                        (end_ts - start_ts) * period_ns
                    } else {
                        0.0
                    };
                    return std::time::Duration::from_nanos(elapsed_ns.round() as u64);
                }
            }
            readback_buffer.unmap();
        }

        let start = std::time::Instant::now();
        for _ in 0..n {
            self.dispatch(pipeline, bind_group, workgroups);
        }
        start.elapsed()
    }

    /// Dispatch multiple different operations in a single command encoder
    /// Each operation is a tuple of (pipeline, bind_group, workgroups)
    /// Returns total GPU execution time for all operations
    pub fn dispatch_multi_ops(
        &self,
        ops: &[(&CompiledPipeline, &wgpu::BindGroup, (u32, u32, u32))],
    ) -> std::time::Duration {
        if let Some(query_set) = &self.profiling_query_set {
            let (start_idx, end_idx) = self.next_query_index();
            let query_count = 2;
            let buf_size = ((query_count as u64 * wgpu::QUERY_SIZE as u64 + 255) / 256) * 256;
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_buffer_multi"),
                size: buf_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.write_timestamp(query_set, start_idx); // Start timestamp
                for (pipeline, bind_group, workgroups) in ops {
                    cpass.set_pipeline(&pipeline.pipeline);
                    cpass.set_bind_group(0, *bind_group, &[]);
                    cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
                }
                cpass.write_timestamp(query_set, end_idx); // End timestamp
            }
            encoder.resolve_query_set(query_set, start_idx..(end_idx + 1), &query_buffer, 0);
            self.queue.submit(Some(encoder.finish()));

            // Readback timestamp query result (separate from kernel execution)
            let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_timestamp_query_readback_multi"),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut copy_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            copy_encoder.copy_buffer_to_buffer(&query_buffer, 0, &readback_buffer, 0, buf_size);
            self.queue.submit(Some(copy_encoder.finish()));

            let slice = readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            if self
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(5)),
                })
                .is_err()
            {
                tracing::warn!("GPU poll timed out after 5s in dispatch_multi_ops, falling through to polling loop");
            }

            loop {
                match rx.try_recv() {
                    Ok(Ok(())) => break,
                    Ok(Err(e)) => {
                        tracing::warn!("Query buffer mapping failed: {e}");
                        readback_buffer.unmap();
                        return std::time::Duration::ZERO;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        self.device.poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_micros(1000)),
                        });
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("Query channel disconnected: {e}");
                        readback_buffer.unmap();
                        return std::time::Duration::ZERO;
                    }
                }
            }

            let (start_ts, end_ts) = {
                let mapped = slice.get_mapped_range();
                let bytes: &[u8] = &mapped;
                let first = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or_else(|_| {
                    tracing::warn!("Query timestamp readback underflow, returning 0");
                    [0u8; 8]
                }));
                let second = u64::from_le_bytes(bytes[8..16].try_into().unwrap_or_else(|_| {
                    tracing::warn!("Query timestamp readback underflow (second), returning 0");
                    [0u8; 8]
                }));
                (first as f64, second as f64)
            };
            readback_buffer.unmap();
            let period_ns = self.queue.get_timestamp_period() as f64;
            let elapsed_ns = if end_ts >= start_ts {
                (end_ts - start_ts) * period_ns
            } else {
                0.0
            };
            return std::time::Duration::from_nanos(elapsed_ns.round() as u64);
        }

        let start = std::time::Instant::now();
        for (pipeline, bind_group, workgroups) in ops {
            self.dispatch(pipeline, bind_group, *workgroups);
        }
        start.elapsed()
    }

    /// Matmul with GPU timestamp query for accurate kernel timing
    /// Returns (result tensor, GPU execution time)
    pub fn matmul_with_timestamp(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Matmul requires 2D tensors".into()));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(GpuError::ShapeMismatch(
                format!(
                    "Matmul dimension mismatch: {}x{} @ {}x{}",
                    m, k, b_shape[0], n
                )
                .into(),
            ));
        }

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_output"),
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [m as u32, k as u32, n as u32, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("matmul_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        let tile_size = self.caps.adaptive_tile_size(m, n, k) as usize;
        let wg_m = ((m + tile_size - 1) / tile_size) as u32;
        let wg_n = ((n + tile_size - 1) / tile_size) as u32;

        let gpu_time = self.dispatch_profiled(pipeline, &bind_group, (wg_m, wg_n, 1));

        Ok((
            GpuTensor {
                shape: vec![m, n],
                buffer: out_buffer,
                dtype: GpuDtype::F32,
                device_id: 0,
            },
            gpu_time,
        ))
    }

    /// Batch matmul with GPU timestamp query for accurate kernel timing
    /// Performs N matmul operations with single timestamp query around all dispatches
    /// Returns (result tensor, total GPU execution time for all N dispatches)
    pub fn matmul_batch_with_timestamp(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        n: usize,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Matmul requires 2D tensors".into()));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n_dim = b_shape[1];

        if k != b_shape[0] {
            return Err(GpuError::ShapeMismatch(
                format!(
                    "Matmul dimension mismatch: {}x{} @ {}x{}",
                    m, k, b_shape[0], n_dim
                )
                .into(),
            ));
        }

        // Use memory pool for output buffer instead of creating new buffer each time
        let mut pool = self
            .memory_pool
            .lock()
            .map_err(|e| GpuError::LockError(e.to_string()))?;
        let out_buffer = pool.alloc(
            (m * n_dim * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        drop(pool);

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_cfg_batch"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [m as u32, k as u32, n_dim as u32, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("matmul_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_tiled not compiled".into()))?;

        // Create bind group directly
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group_batch"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        let tile_size = self.caps.adaptive_tile_size(m, n_dim, k) as usize;
        let wg_m = ((m + tile_size - 1) / tile_size) as u32;
        let wg_n = ((n_dim + tile_size - 1) / tile_size) as u32;

        let gpu_time = self.dispatch_batch_profiled(pipeline, &bind_group, (wg_m, wg_n, 1), n);

        Ok((
            GpuTensor {
                shape: vec![m, n_dim],
                buffer: out_buffer.buffer,
                dtype: GpuDtype::F32,
                device_id: 0,
            },
            gpu_time,
        ))
    }

    /// Dynamic kernel routing matmul: uses simple kernel for small matrices, fused for large
    /// Threshold: 512x512
    /// Returns (result tensor, GPU execution time)
    pub fn matmul_dynamic(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("Matmul requires 2D tensors".into()));
        }

        let max_dim = a_shape[0].max(a_shape[1]).max(b_shape[0]).max(b_shape[1]);

        // Use simple matmul for small matrices (fusion overhead dominates)
        if max_dim < 512 {
            self.matmul(a, b)
                .map(|tensor| (tensor, std::time::Duration::from_secs(0)))
        } else {
            // Use matmul_with_timestamp for large matrices (fusion beneficial)
            self.matmul_with_timestamp(a, b)
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.1: TILED MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_matmul_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_tiled") {
            return Ok(());
        }
        let wgsl =
            std::borrow::Cow::Owned(MATMUL_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()));
        self.compile_pipeline(
            "matmul_tiled",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            wgsl,
            "matmul_tiled_main",
        )
    }
    fn compile_rotary_embedding(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "rotary_embedding",
            &[
                storage_binding(0, false),
                storage_binding(1, true),
                storage_binding(2, true),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(ROTARY_EMBEDDING_WGSL),
            "rotary_main",
        )
    }

    fn compile_adam_step(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "adam_step",
            &[
                storage_binding(0, false),
                storage_binding(1, true),
                storage_binding(2, false),
                storage_binding(3, false),
                uniform_binding(4),
            ],
            std::borrow::Cow::Borrowed(ADAM_WGSL),
            "main",
        )
    }

    fn compile_repeat_heads(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "repeat_heads",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(REPEAT_HEADS_WGSL),
            "repeat_heads_main",
        )
    }

    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        if a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        // Auto-dispatch to F16 matmul kernel when weight tensor is packed F16.
        // Activation (a) stays f32 — no upconversion needed.
        if a.dtype == GpuDtype::F32 && b.dtype == GpuDtype::F16 {
            return self.matmul_f16(a, b);
        }
        // Both f32: standard path
        if a.dtype != GpuDtype::F32 || b.dtype != GpuDtype::F32 {
            return Err(GpuError::Unsupported(format!(
                "matmul: unsupported dtypes a={:?} b={:?}",
                a.dtype, b.dtype
            )));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        self.queue
            .write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_tiled_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wgx = (m_u32 + tile_u32 - 1) / tile_u32;
        let wgy = (n_u32 + tile_u32 - 1) / tile_u32;
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.1b: F16 PACKED WEIGHT MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_matmul_f16_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_f16_tiled") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_F16_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_f16_tiled",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            wgsl,
            "matmul_f16_main",
        )
    }

    /// F16 packed weight matmul: activation (f32) × weight (packed F16).
    /// Weight is stored as packed F16 (2 f16 per u32), halving VRAM reads.
    /// Accumulation stays in f32 for precision.
    /// Output is f32 with shape [a.shape[0], b_packed.shape[1]].
    pub fn matmul_f16(&self, a: &GpuTensor, b_packed: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b_packed.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        // b is packed F16: [K, N] in f32 terms, but stored as u32 with 2 f16 per u32
        // a_flat = M*K f32 elements, b_flat = K*N/2 u32 elements
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if b_packed.dtype != GpuDtype::F16 {
            return Err(GpuError::Unsupported(
                "matmul_f16: b_packed must have dtype F16".into(),
            ));
        }
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        self.queue
            .write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_f16_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_f16_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_f16_tiled_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_packed.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wgx = (m_u32 + tile_u32 - 1) / tile_u32;
        let wgy = (n_u32 + tile_u32 - 1) / tile_u32;
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![m, n],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2: INT8 QUANTIZED MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_matmul_int8_tiled(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_int8_tiled") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_INT8_TILED_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_int8_tiled",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            wgsl,
            "matmul_int8_main",
        )
    }

    /// Quantized matmul: A (int8 packed) × B (f32) → C (f32).
    ///
    /// `a` must have dtype `GpuDtype::I8`. Internally the int8 values are
    /// unpacked, dequantized with `scale`, and accumulated in f32.
    /// `b` must be f32.
    pub fn matmul_int8(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        scale: f32,
    ) -> Result<GpuTensor, GpuError> {
        if a.dtype() != GpuDtype::I8 {
            return Err(GpuError::Dtype(format!(
                "matmul_int8 expects I8 for A, got {:?}",
                a.dtype()
            )));
        }
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        if a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            20,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        let mut uniform_bytes = Vec::with_capacity(20);
        uniform_bytes.extend_from_slice(bytemuck::cast_slice(&dims_data));
        uniform_bytes.extend_from_slice(bytemuck::bytes_of(&scale));
        self.queue.write_buffer(&dims_buf, 0, &uniform_bytes);

        let pipeline = self
            .pipelines
            .get("matmul_int8_tiled")
            .ok_or_else(|| GpuError::Pipeline("matmul_int8_tiled not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_int8_tiled_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let wgx = (m_u32 + tile_u32 - 1) / tile_u32;
        let wgy = (n_u32 + tile_u32 - 1) / tile_u32;
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2b: INT8 WEIGHT MATMUL
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_matmul_int8_weight(&mut self, tile: u32) -> Result<(), GpuError> {
        if self.pipelines.contains_key("matmul_int8_weight") {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            MATMUL_INT8_WEIGHT_WGSL.replace("{{TILE_SIZE}}", &tile.to_string()),
        );
        self.compile_pipeline(
            "matmul_int8_weight",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
                storage_binding(4, true),
                storage_binding(5, true),
            ],
            wgsl,
            "matmul_int8_weight_main",
        )
    }

    /// Quantized matmul: A (f32 activations) × B (packed int8 weights) → C (f32).
    ///
    /// `b` must have dtype `GpuDtype::I8` and shape [N, K] (ORIGINAL orientation —
    /// NOT the conventional transposed layout). The shader reads B[j][k] and
    /// computes C[i][j] = sum_k A[i][k] × dequantize(B[j][k]).
    ///
    /// This is designed for the model inference pattern `activation @ weight^T`
    /// where the original weight matrix W[output_dim, hidden] is stored as int8
    /// and the transpose is handled on-the-fly in the shader.
    pub fn matmul_int8_weight(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        scales: &GpuTensor,
        zero_points: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        if b.dtype() != GpuDtype::I8 {
            return Err(GpuError::Dtype(format!(
                "matmul_int8_weight expects I8 for B (weight), got {:?}",
                b.dtype()
            )));
        }
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }
        // A[M, K] × B[N, K] → Not standard. Check K matches: a_shape[1] == b_shape[1]
        if a_shape[1] != b_shape[1] {
            return Err(GpuError::MatMulShape(a_shape, b_shape));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[0];
        let m_u32 = u32::try_from(m)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let k_u32 = u32::try_from(k)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let tile = self.caps.adaptive_tile_size(m, n, k);
        let tile_u32 = u32::try_from(tile)
            .map_err(|_| GpuError::MatMulShape(a_shape.clone(), b_shape.clone()))?;

        let c_size = (m_u32 as u64) * (n_u32 as u64) * 4;
        let c_buffer = self.alloc_or_create_buffer(
            c_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let dims_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dims_data: [u32; 4] = [m_u32, k_u32, n_u32, tile_u32];
        self.queue.write_buffer(&dims_buf, 0, bytemuck::cast_slice(&dims_data));

        let pipeline = self
            .pipelines
            .get("matmul_int8_weight")
            .ok_or_else(|| GpuError::Pipeline("matmul_int8_weight not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_int8_weight_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scales.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: zero_points.buffer().as_entire_binding(),
                },
            ],
        });

        let wgx = (m_u32 + tile_u32 - 1) / tile_u32;
        let wgy = (n_u32 + tile_u32 - 1) / tile_u32;
        self.dispatch(pipeline, &bind_group, (wgx, wgy, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[0]],
            buffer: c_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PACKED F16 CONVERSION HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Convert f32 tensor to packed F16 (2 f16 per u32). Returns tensor with GpuDtype::F16.
    pub fn f32_to_f16_packed(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        use crate::gpu_mixed::dispatch_f32_to_f16_packed;
        let pipeline = self
            .pipelines
            .get("f32_to_f16_packed")
            .ok_or_else(|| GpuError::Pipeline("f32_to_f16_packed not compiled".into()))?;
        dispatch_f32_to_f16_packed(self, &pipeline.pipeline, input)
    }

    /// Convert packed F16 tensor back to f32. Input must have dtype GpuDtype::F16.
    pub fn f16_packed_to_f32(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        use crate::gpu_mixed::dispatch_f16_packed_to_f32;
        if input.dtype() != GpuDtype::F16 {
            return Err(GpuError::Dtype(format!(
                "f16_packed_to_f32 expects F16 input, got {:?}",
                input.dtype()
            )));
        }
        let pipeline = self
            .pipelines
            .get("f16_packed_to_f32")
            .ok_or_else(|| GpuError::Pipeline("f16_packed_to_f32 not compiled".into()))?;
        dispatch_f16_packed_to_f32(self, &pipeline.pipeline, input)
    }

    fn compile_elementwise(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "elementwise",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(ELEMENTWISE_WGSL),
            "elementwise_main",
        )
    }

    fn compile_elementwise_inplace(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "elementwise_inplace",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(ELEMENTWISE_INPLACE_WGSL),
            "elementwise_inplace_main",
        )
    }

    /// Dispatch element-wise/unary op: output = op(a)
    pub fn elementwise_unary(&self, a: &GpuTensor, op: ElemOp) -> Result<GpuTensor, GpuError> {
        let shape = a.shape();
        let numel = a.numel();

        let out_buffer = self.alloc_or_create_buffer(
            (numel * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let cfg_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise")
            .ok_or_else(|| GpuError::Pipeline("elementwise not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, a.buffer()), (1, a.buffer()), (2, &out_buffer)],
            &[(3, &cfg_buf)],
            numel as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    pub fn elementwise_unary_inplace(
        &self,
        tensor: &mut GpuTensor,
        op: ElemOp,
    ) -> Result<(), GpuError> {
        let numel = tensor.numel();
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_inplace_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise_inplace")
            .ok_or_else(|| GpuError::Pipeline("elementwise_inplace not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, tensor.buffer())],
            &[(1, &cfg_buf)],
            numel as u32, 256, 4,
        );
        Ok(())
    }

    pub fn gelu_inplace(&self, tensor: &mut GpuTensor) -> Result<(), GpuError> {
        self.elementwise_unary_inplace(tensor, ElemOp::Gelu)
    }

    /// Broadcast a GPU tensor to a target shape.
    /// Efficient scalar path (1 element → N) avoids full CPU round-trip.
    /// General path reads tensor to CPU, broadcasts via ndarray, re-uploads.
    pub fn broadcast_tensor(
        &self,
        tensor: &GpuTensor,
        target: &[usize],
    ) -> Result<GpuTensor, GpuError> {
        let shape = tensor.shape();
        let target_numel: usize = target.iter().product();

        if shape == target {
            return Ok(tensor.clone());
        }

        if tensor.numel() == 1 {
            let scalar = tensor.to_cpu_first_element()?;
            let data = vec![scalar; target_numel];
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("broadcast_scalar"),
                size: (target_numel * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&data));

            crate::gpu::gpu_observability::PCIE_WRITE_BYTES.fetch_add(
                (target_numel * 4) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );

            return Ok(GpuTensor {
                shape: target.to_vec(),
                buffer,
                dtype: tensor.dtype,
                device_id: tensor.device_id,
            });
        }

        let cpu_data = tensor.to_cpu()?;
        let bc_data = match cpu_data.broadcast(target) {
            Some(view) => view.to_owned(),
            None => {
                return Err(GpuError::ShapeMismatch(format!(
                    "Cannot broadcast {:?} to {:?}",
                    shape, target
                )));
            }
        };
        GpuTensor::from_cpu(&bc_data)
    }

    /// Dispatch element-wise/binary op: output = op(a, b).
    /// Automatically broadcasts shapes (including scalar→tensor) before dispatch.
    pub fn elementwise_binary(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        op: ElemOp,
    ) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape != b_shape {
            let target = crate::broadcast::broadcast_shapes(&a_shape, &b_shape);
            let a_bc = if a_shape != target {
                self.broadcast_tensor(a, &target)?
            } else {
                a.clone()
            };
            let b_bc = if b_shape != target {
                self.broadcast_tensor(b, &target)?
            } else {
                b.clone()
            };
            return self.elementwise_binary(&a_bc, &b_bc, op);
        }

        let numel = a.numel();

        let out_buffer = self.alloc_or_create_buffer(
            (numel * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        let cfg_buf = self.alloc_or_create_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg_data: [u32; 4] = [numel as u32, op as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise")
            .ok_or_else(|| GpuError::Pipeline("elementwise not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, a.buffer()), (1, b.buffer()), (2, &out_buffer)],
            &[(3, &cfg_buf)],
            numel as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: a_shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Convenience: element-wise add on GPU
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Add)
    }

    /// GPU-side in-place add: a = a + b
    /// Uses buffer swap instead of copy-back to avoid extra allocation+copy.
    pub fn add_inplace(&self, a: &mut GpuTensor, b: &GpuTensor) -> Result<(), GpuError> {
        let result = self.add(&a.view_as(a.shape()), b)?;
        a.buffer = result.buffer;
        Ok(())
    }

    /// Convenience: element-wise sub on GPU
    pub fn sub(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Sub)
    }

    /// Convenience: element-wise mul on GPU
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Mul)
    }

    /// Convenience: element-wise div on GPU
    pub fn div(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_binary(a, b, ElemOp::Div)
    }

    /// Convenience: sqrt on GPU
    pub fn sqrt(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Sqrt)
    }

    /// Convenience: exp on GPU
    pub fn exp(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Exp)
    }

    /// Convenience: relu on GPU
    pub fn relu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Relu)
    }

    /// Convenience: sigmoid on GPU
    pub fn sigmoid(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Sigmoid)
    }

    /// Convenience: silu on GPU
    pub fn silu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Silu)
    }

    /// Convenience: gelu on GPU
    pub fn gelu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Gelu)
    }

    /// Convenience: tanh on GPU
    pub fn tanh(&self, a: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.elementwise_unary(a, ElemOp::Tanh)
    }

    /// LeakyReLU in-place on GPU: x = x > 0 ? x : x * negative_slope
    /// Uses elementwise_inplace pipeline with custom cfg for slope parameter.
    pub fn leaky_relu_inplace(
        &self,
        tensor: &mut GpuTensor,
        negative_slope: f32,
    ) -> Result<(), GpuError> {
        let numel = tensor.numel();
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("leaky_relu_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [
            numel as u32,
            ElemOp::LeakyRelu as u32,
            f32::to_bits(negative_slope),
            0,
        ];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("elementwise_inplace")
            .ok_or_else(|| GpuError::Pipeline("elementwise_inplace not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, tensor.buffer())],
            &[(1, &cfg_buf)],
            numel as u32, 256, 4,
        );
        Ok(())
    }

    /// LeakyReLU on GPU: returns new tensor
    pub fn leaky_relu(
        &self,
        input: &GpuTensor,
        negative_slope: f32,
    ) -> Result<GpuTensor, GpuError> {
        let mut result = input.clone();
        self.leaky_relu_inplace(&mut result, negative_slope)?;
        Ok(result)
    }
    /// Rotary embedding (RoPE) in-place on GPU.
    /// Applies rotary position embeddings to [total_rows, dim] tensor.
    /// cos/sin: [half] arrays for current position (precomputed via precompute_freqs_cis).
    pub fn rotary_embedding(
        &self,
        x: &GpuTensor,
        cos: &GpuTensor,
        sin: &GpuTensor,
        head_dim: u32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "RoPE GPU: x must be 2D [total_rows, dim]".into(),
            ));
        }
        let total_rows = shape[0] as u32;
        let dim = shape[1] as u32;
        if dim != head_dim {
            return Err(GpuError::ShapeMismatch(
                format!("RoPE GPU: dim {} != head_dim {}", dim, head_dim).into(),
            ));
        }
        let half = head_dim / 2;

        // Create output buffer
        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotary_output"),
            size: (shape[0] * shape[1] * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy input to output first (in-place kernel writes to output)
        self.with_encoder(|enc| {
            enc.copy_buffer_to_buffer(
                x.buffer(),
                0,
                &out_buffer,
                0,
                (shape[0] * shape[1] * 4) as u64,
            );
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotary_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [total_rows, dim, half, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("rotary_embedding")
            .ok_or_else(|| GpuError::Pipeline("rotary_embedding not compiled".into()))?;

        let num_pairs = total_rows * half;
        self.dispatch_1d_chunked(
            pipeline,
            &[(0, &out_buffer), (1, cos.buffer()), (2, sin.buffer())],
            &[(3, &cfg_buf)],
            num_pairs, 256, 4,
        );

        Ok(GpuTensor {
            shape: shape.clone(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Repeat KV heads to match Q heads for GQA.
    /// src: [seq, kv_heads, dim] → dst: [q_heads, seq, dim]
    /// Output is HEAD-MAJOR (compatible with fused_attention's [B, H, S, D] layout).
    /// The WGSL shader writes in head-major layout: element (qh, s, d) at qh*seq*dim + s*dim + d.
    pub fn repeat_heads(
        &self,
        src: &GpuTensor,
        kv_heads: u32,
        q_heads: u32,
        dim: u32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = src.shape();
        if shape.len() != 3 {
            return Err(GpuError::ShapeMismatch(
                "repeat_heads: src must be 3D [seq, kv_heads, dim]".into(),
            ));
        }
        let seq = shape[0] as u32;

        let numel_dst = (seq * q_heads * dim) as usize;
        let dst_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repeat_heads_output"),
            size: (numel_dst * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repeat_heads_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [seq, kv_heads, q_heads, dim];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("repeat_heads")
            .ok_or_else(|| GpuError::Pipeline("repeat_heads not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, src.buffer()), (1, &dst_buffer)],
            &[(2, &cfg_buf)],
            numel_dst as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![q_heads as usize, seq as usize, dim as usize],
            buffer: dst_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1.2: REDUCE KERNELS
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_reduce(&mut self, op: ReduceOp) -> Result<(), GpuError> {
        let key = match op {
            ReduceOp::Sum => "reduce_sum",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
        };
        if self.pipelines.contains_key(key) {
            return Ok(());
        }
        let wgsl = std::borrow::Cow::Owned(
            REDUCE_WGSL_TEMPLATE.replace("{{OP}}", &(op as u32).to_string()),
        );
        self.compile_pipeline(
            key,
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            wgsl,
            "reduce_main",
        )
    }

    /// Reduce 1D tensor to scalar (sum / max / min)
    pub fn reduce(&self, input: &GpuTensor, op: ReduceOp) -> Result<GpuTensor, GpuError> {
        let numel = input.numel();
        if numel == 0 {
            return Ok(GpuTensor {
                shape: vec![1],
                buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("reduce_output"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                dtype: GpuDtype::F32,
                device_id: 0,
            });
        }

        let key = match op {
            ReduceOp::Sum => "reduce_sum",
            ReduceOp::Max => "reduce_max",
            ReduceOp::Min => "reduce_min",
        };

        let pipeline = self
            .pipelines
            .get(key)
            .ok_or_else(|| GpuError::Pipeline(format!("{} not compiled", key)))?;

        // Multi-pass: each workgroup reduces 256 elements → partial results
        // Then reduce partials until scalar remains
        let mut current_buffer = input.buffer().clone();
        let mut current_numel = numel;

        loop {
            let num_groups = (current_numel + 255) / 256;
            let num_groups_u32 = num_groups as u32;

            let out_size = (num_groups * 4) as u64;
            let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce_partial"),
                size: out_size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let needs_chunking = num_groups_u32 > MAX_WORKGROUPS_PER_DIM;

            if needs_chunking {
                let elems_per_full_chunk = MAX_WORKGROUPS_PER_DIM as u64 * 256;
                let mut input_offset: u64 = 0;
                let mut output_offset: u64 = 0;
                let mut remaining = current_numel as u64;

                while remaining > 0 {
                    let chunk_elems = remaining.min(elems_per_full_chunk);
                    let chunk_groups = ((chunk_elems + 255) / 256) as u32;
                    let chunk_bytes = chunk_elems * 4;
                    let chunk_out_bytes = chunk_groups as u64 * 4;

                    let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("reduce_cfg_chunk"),
                        size: 8,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let cfg_data: [u32; 2] = [chunk_elems as u32, chunk_groups];
                    self.queue
                        .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

                    let bind_group = self.device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            label: Some(&format!("{}_bind_group_chunk", key)),
                            layout: &pipeline.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(
                                        wgpu::BufferBinding {
                                            buffer: &current_buffer,
                                            offset: input_offset,
                                            size: Some(
                                                wgpu::BufferSize::new(chunk_bytes).unwrap(),
                                            ),
                                        },
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(
                                        wgpu::BufferBinding {
                                            buffer: &out_buffer,
                                            offset: output_offset,
                                            size: Some(
                                                wgpu::BufferSize::new(chunk_out_bytes).unwrap(),
                                            ),
                                        },
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: cfg_buf.as_entire_binding(),
                                },
                            ],
                        },
                    );

                    self.dispatch_impl(pipeline, &bind_group, (chunk_groups, 1, 1));

                    input_offset += chunk_bytes;
                    output_offset += chunk_out_bytes;
                    remaining -= chunk_elems;
                }
            } else {
                let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("reduce_cfg"),
                    size: 8,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let cfg_data: [u32; 2] = [current_numel as u32, num_groups as u32];
                self.queue
                    .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

                let bind_group = self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some(&format!("{}_bind_group", key)),
                        layout: &pipeline.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: current_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: out_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cfg_buf.as_entire_binding(),
                            },
                        ],
                    },
                );

                self.dispatch(pipeline, &bind_group, (num_groups as u32, 1, 1));
            }

            if num_groups == 1 {
                return Ok(GpuTensor {
                    shape: vec![1],
                    buffer: out_buffer,
                    dtype: GpuDtype::F32,
                    device_id: 0,
                });
            }

            current_buffer = out_buffer;
            current_numel = num_groups;
        }
    }

    /// Convenience: reduce sum to scalar
    pub fn sum(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, ReduceOp::Sum)
    }

    /// Convenience: reduce max to scalar
    pub fn max(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, ReduceOp::Max)
    }

    /// Convenience: reduce min to scalar
    pub fn min(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        self.reduce(input, ReduceOp::Min)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.1: SOFTMAX
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_softmax(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "softmax",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(SOFTMAX_WGSL),
            "softmax_main",
        )
    }

    /// Softmax along last axis. Input shape: [rows, dim].
    pub fn softmax(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "Softmax GPU: input must be 2D".into(),
            ));
        }
        let rows = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = input.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [rows, dim, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("softmax")
            .ok_or_else(|| GpuError::Pipeline("softmax not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (rows, 1, 1));

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.2: RMSNORM
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_rms_norm(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "rms_norm",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(RMSNORM_WGSL),
            "rms_norm_main",
        )
    }

    /// RMSNorm: out = x / rms(x) * weight. Shapes: x[batch, dim], weight[dim].
    pub fn rms_norm(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        eps: f32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch("RMSNorm GPU: x must be 2D".into()));
        }
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = x.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("rms_norm")
            .ok_or_else(|| GpuError::Pipeline("rms_norm not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    fn compile_rms_norm_backward(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "rms_norm_backward",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                storage_binding(4, false),
                uniform_binding(5),
            ],
            std::borrow::Cow::Borrowed(RMSNORM_BACKWARD_WGSL),
            "rms_norm_bwd_main",
        )
    }

    pub fn rms_norm_backward(
        &self,
        input: &GpuTensor,
        weight: &GpuTensor,
        grad: &GpuTensor,
        eps: f32,
    ) -> Result<(GpuTensor, GpuTensor), GpuError> {
        let shape = input.shape();
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = input.numel();

        let dx_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_dx"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dw_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_dw_atomic"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Zero-init dw
        let dw_tmp = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&dw_tmp)?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("rms_norm_backward")
            .ok_or_else(|| GpuError::Pipeline("rms_norm_backward not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_bwd_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dw_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        let dx = GpuTensor {
            shape: shape.to_vec(),
            buffer: dx_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dw = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        Ok((dx, dw))
    }

    // ── LayerNorm Backward ──────────────────────────────────────────────

    fn compile_layer_norm_backward(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "layer_norm_backward",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, true),
                storage_binding(4, false),
                storage_binding(5, false),
                storage_binding(6, false),
                uniform_binding(7),
            ],
            std::borrow::Cow::Borrowed(LAYERNORM_BACKWARD_WGSL),
            "layer_norm_bwd_main",
        )
    }

    pub fn layer_norm_backward(
        &self,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        grad: &GpuTensor,
        eps: f32,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor), GpuError> {
        let shape = input.shape();
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = input.numel();

        let dx_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_dx"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dw_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_dw_atomic"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let db_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_db_atomic"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dw_tmp = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let db_tmp = GpuTensor {
            shape: vec![dim as usize],
            buffer: db_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&dw_tmp)?;
        self.fill_zero_u32(&db_tmp)?;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("layer_norm_backward")
            .ok_or_else(|| GpuError::Pipeline("layer_norm_backward not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bwd_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dw_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: db_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        let dx = GpuTensor {
            shape: shape.to_vec(),
            buffer: dx_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dw = GpuTensor {
            shape: vec![dim as usize],
            buffer: dw_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let db = GpuTensor {
            shape: vec![dim as usize],
            buffer: db_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        Ok((dx, dw, db))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.3: CROSS-ENTROPY LOSS
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_cross_entropy(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "cross_entropy",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(CROSS_ENTROPY_WGSL),
            "cross_entropy_main",
        )
    }

    /// Cross-entropy loss. logits: [batch, classes], targets: [batch] (u32 as f32).
    /// Returns per-sample loss: [batch].
    pub fn cross_entropy(
        &self,
        logits: &GpuTensor,
        targets: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "CrossEntropy GPU: logits must be 2D".into(),
            ));
        }
        let batch = shape[0] as u32;
        let classes = shape[1] as u32;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_output"),
            size: (batch as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [batch, classes, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("cross_entropy")
            .ok_or_else(|| GpuError::Pipeline("cross_entropy not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cross_entropy_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor {
            shape: vec![batch as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    fn compile_cross_entropy_backward(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "cross_entropy_backward",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
            ],
            std::borrow::Cow::Borrowed(CROSS_ENTROPY_BACKWARD_WGSL),
            "cross_entropy_bwd_main",
        )
    }

    /// Cross-entropy backward: d_logits = grad * (softmax - one_hot(targets)).
    /// softmax/grad/d_logits: [batch, classes]. targets: [batch] (u32).
    pub fn cross_entropy_backward(
        &self,
        softmax: &GpuTensor,
        grad: &GpuTensor,
        targets: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let shape = softmax.shape();
        let total = softmax.numel() as u32;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_bwd_out"),
            size: (total as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cross_entropy_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [shape[0] as u32, shape[1] as u32, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("cross_entropy_backward")
            .ok_or_else(|| GpuError::Pipeline("cross_entropy_backward not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, softmax.buffer()), (1, grad.buffer()), (2, targets.buffer()), (3, &out_buffer)],
            &[(4, &cfg_buf)],
            total, 256, 4,
        );

        Ok(GpuTensor {
            shape: shape.clone(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.4: EMBEDDING
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_embedding(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "embedding",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(EMBEDDING_WGSL),
            "embedding_main",
        )
    }

    fn compile_embedding_backward(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "embedding_backward",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
            std::borrow::Cow::Borrowed(EMBEDDING_BACKWARD_WGSL),
            "embedding_backward_main",
        )
    }

    /// Embedding backward: scatter-add grad into d_weight.
    /// ids: [N] u32, grad: [N, D] f32, output: d_weight [V, D] f32 (must be zero-initialized).
    pub fn embedding_backward(
        &self,
        ids: &GpuTensor,
        grad: &GpuTensor,
        vocab_size: usize,
    ) -> Result<GpuTensor, GpuError> {
        let grad_shape = grad.shape();
        let dim = grad_shape[grad_shape.len() - 1];
        let num_ids = ids.numel();
        let d_weight = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_d_weight"),
            size: (vocab_size * dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Zero-init d_weight
        let d_weight_tmp = GpuTensor {
            shape: vec![vocab_size, dim],
            buffer: d_weight,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&d_weight_tmp)?;
        let d_weight_buf = d_weight_tmp.buffer;

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_bwd_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [vocab_size as u32, dim as u32, num_ids as u32, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("embedding_backward")
            .ok_or_else(|| GpuError::Pipeline("embedding_backward not compiled".into()))?;

        let total_threads = (num_ids * dim) as u32;
        self.dispatch_1d_chunked(
            pipeline,
            &[(0, ids.buffer()), (1, grad.buffer()), (2, &d_weight_buf)],
            &[(3, &cfg_buf)],
            total_threads, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![vocab_size, dim],
            buffer: d_weight_buf,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Embedding lookup: out[i * dim + d] = weight[tokens[i] * dim + d].
    /// ids: [seq_len], weight: [vocab, dim].
    pub fn embedding(&self, ids: &GpuTensor, weight: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "Embedding GPU: weight must be 2D [vocab, dim]".into(),
            ));
        }
        let seq_len = ids.numel() as u32;
        let dim = w_shape[1] as u32;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_output"),
            size: (seq_len as u64) * (dim as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vocab_size = w_shape[0] as u32;
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [vocab_size, dim, seq_len, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("embedding")
            .ok_or_else(|| GpuError::Pipeline("embedding not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, ids.buffer()), (1, weight.buffer()), (2, &out_buffer)],
            &[(3, &cfg_buf)],
            seq_len * dim, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![seq_len as usize, dim as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.5: LAYERNORM
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_layer_norm(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "layer_norm",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
            ],
            std::borrow::Cow::Borrowed(LAYERNORM_WGSL),
            "layer_norm_main",
        )
    }

    /// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
    /// Shapes: x[batch, dim], weight[dim], bias[dim]
    pub fn layer_norm(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        eps: f32,
    ) -> Result<GpuTensor, GpuError> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "LayerNorm GPU: x must be 2D".into(),
            ));
        }
        let batch = shape[0] as u32;
        let dim = shape[1] as u32;
        let numel = x.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer_norm_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [batch, dim, f32::to_bits(eps), 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("layer_norm")
            .ok_or_else(|| GpuError::Pipeline("layer_norm not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(pipeline, &bind_group, (batch, 1, 1));

        Ok(GpuTensor {
            shape: vec![batch as usize, dim as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 2.6: TRANSPOSE + MATMUL BACKWARD
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_transpose(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "transpose",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(TRANSPOSE_WGSL),
            "transpose_main",
        )
    }

    /// Transpose a 2D tensor [rows, cols] → [cols, rows]
    pub fn transpose(&self, input: &GpuTensor) -> Result<GpuTensor, GpuError> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(GpuError::ShapeMismatch(
                "Transpose GPU: input must be 2D".into(),
            ));
        }
        let rows = shape[0] as u32;
        let cols = shape[1] as u32;
        let numel = input.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transpose_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_data: [u32; 4] = [rows, cols, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg_data));

        let pipeline = self
            .pipelines
            .get("transpose")
            .ok_or_else(|| GpuError::Pipeline("transpose not compiled".into()))?;

        self.dispatch_1d_chunked(
            pipeline,
            &[(0, input.buffer()), (1, &out_buffer)],
            &[(2, &cfg_buf)],
            numel as u32, 256, 4,
        );

        Ok(GpuTensor {
            shape: vec![cols as usize, rows as usize],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Matmul backward: grad_a = grad_out @ b^T, grad_b = a^T @ grad_out
    pub fn matmul_backward(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        grad_out: &GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor), GpuError> {
        let b_t = self.transpose(b)?;
        let grad_a = self.matmul(&grad_out, &b_t)?;

        let a_t = self.transpose(a)?;
        let grad_b = self.matmul(&a_t, &grad_out)?;

        Ok((grad_a, grad_b))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 3: FUSED ATTENTION (Flash Attention-style)
    // ═══════════════════════════════════════════════════════════════════════════

    fn compile_fused_attention(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fused_attention",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
                uniform_binding(5),
            ],
            std::borrow::Cow::Borrowed(FUSED_ATTENTION_WGSL),
            "fused_attention_main",
        )
    }

    /// Fused attention: O = softmax(Q @ K^T / scale) @ V
    /// Shapes: Q/K/V = [batch, heads, seq, dim], output = [batch, heads, seq, dim]
    /// Uses online softmax (Flash Attention-style) with causal masking.
    pub fn fused_attention(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        scale: f32,
        causal: bool,
    ) -> Result<GpuTensor, GpuError> {
        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(GpuError::ShapeMismatch(
                "Fused Attention: Q must be 4D [B,H,S,D]".into(),
            ));
        }
        let batch = q_shape[0] as u32;
        let heads = q_shape[1] as u32;
        let seq_len = q_shape[2] as u32;
        let dim = q_shape[3] as u32;
        let numel = q.numel();

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_output"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cfg1_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_cfg1"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg1: [u32; 4] = [batch, heads, seq_len, dim];
        self.queue
            .write_buffer(&cfg1_buf, 0, bytemuck::cast_slice(&cfg1));

        let causal_flag: u32 = if causal { 1 } else { 0 };
        let cfg2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_cfg2"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg2: [u32; 4] = [f32::to_bits(scale), causal_flag, 0, 0];
        self.queue
            .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2));

        let pipeline = self
            .pipelines
            .get("fused_attention")
            .ok_or_else(|| GpuError::Pipeline("fused_attention not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_attention_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cfg2_buf.as_entire_binding(),
                },
            ],
        });

        let num_wg = batch * heads * seq_len;
        let mut cfg2_mut: [u32; 4] = [f32::to_bits(scale), causal_flag, 0, 0];
        if num_wg > MAX_WORKGROUPS_PER_DIM {
            let mut remaining = num_wg;
            let mut base: u32 = 0;
            while remaining > 0 {
                let chunk = remaining.min(MAX_WORKGROUPS_PER_DIM);
                cfg2_mut[2] = base;
                self.queue
                    .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2_mut));
                self.dispatch_impl(pipeline, &bind_group, (chunk, 1, 1));
                base += chunk;
                remaining -= chunk;
            }
        } else {
            self.dispatch(pipeline, &bind_group, (num_wg, 1, 1));
        }

        Ok(GpuTensor {
            shape: q_shape.clone(),
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    // ── Fused Attention Backward ─────────────────────────────────────
    // Pure GPU kernel for causal_attention backward.
    // Input: Q/K/V/dO as [B,H,S,D], Output: dQ/dK/dV as [B,H,S,D].
    // Allocates dQ/dK/dV internally (caller must ensure zero-init if needed).

    fn compile_fused_attention_backward(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fused_attention_backward",
            &[
                storage_binding(0, true),  // q
                storage_binding(1, true),  // k
                storage_binding(2, true),  // v
                storage_binding(3, true),  // dO
                storage_binding(4, false), // dq
                storage_binding(5, false), // dk (atomic<u32>)
                storage_binding(6, false), // dv (atomic<u32>)
                uniform_binding(7),        // cfg1
                uniform_binding(8),        // cfg2
            ],
            std::borrow::Cow::Borrowed(FUSED_ATTENTION_BACKWARD_WGSL),
            "fused_attn_backward_main",
        )
    }

    /// Fused attention backward: compute dQ, dK, dV from Q, K, V, dO.
    /// All shapes: [batch, heads, seq, dim]. Uses causal masking.
    /// dK/dV are atomically accumulated (must be zero-initialized).
    pub fn fused_attention_backward(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        grad: &GpuTensor,
        scale: f32,
        causal: bool,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor), GpuError> {
        let shape = q.shape();
        if shape.len() != 4 {
            return Err(GpuError::ShapeMismatch(
                "Fused Attention backward: Q must be 4D [B,H,S,D]".into(),
            ));
        }
        let batch = shape[0] as u32;
        let heads = shape[1] as u32;
        let seq_len = shape[2] as u32;
        let dim = shape[3] as u32;
        let numel = q.numel();
        let byte_size = (numel * 4) as u64;

        // Allocate output buffers
        let dq_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_dq"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dk_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_dk_atomic"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dv_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attention_dv_atomic"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Zero-init dK and dV (atomic buffers start non-zero from pool)
        let dk_tmp = GpuTensor {
            shape: shape.clone(),
            buffer: dk_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dv_tmp = GpuTensor {
            shape: shape.clone(),
            buffer: dv_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        self.fill_zero_u32(&dk_tmp)?;
        self.fill_zero_u32(&dv_tmp)?;

        let cfg1_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attn_bwd_cfg1"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // batch_heads = batch * heads (heads treated as flat batch dim in shader)
        let cfg1: [u32; 4] = [batch * heads, seq_len, dim, f32::to_bits(scale)];
        self.queue
            .write_buffer(&cfg1_buf, 0, bytemuck::cast_slice(&cfg1));

        let causal_flag: u32 = if causal { 1 } else { 0 };
        let cfg2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_attn_bwd_cfg2"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cfg2_mut: [u32; 4] = [causal_flag, 0, 0, 0];
        self.queue
            .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2_mut));

        let pipeline = self
            .pipelines
            .get("fused_attention_backward")
            .ok_or_else(|| GpuError::Pipeline("fused_attention_backward not compiled".into()))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_attention_backward_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dq_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dk_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dv_tmp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cfg1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: cfg2_buf.as_entire_binding(),
                },
            ],
        });

        let num_wg = batch * heads * seq_len;
        if num_wg > MAX_WORKGROUPS_PER_DIM {
            let mut remaining = num_wg;
            let mut base: u32 = 0;
            while remaining > 0 {
                let chunk = remaining.min(MAX_WORKGROUPS_PER_DIM);
                cfg2_mut[1] = base;
                self.queue
                    .write_buffer(&cfg2_buf, 0, bytemuck::cast_slice(&cfg2_mut));
                self.dispatch_impl(pipeline, &bind_group, (chunk, 1, 1));
                base += chunk;
                remaining -= chunk;
            }
        } else {
            self.dispatch(pipeline, &bind_group, (num_wg, 1, 1));
        }

        let dq = GpuTensor {
            shape: shape.clone(),
            buffer: dq_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dk = GpuTensor {
            shape: shape.clone(),
            buffer: dk_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let dv = GpuTensor {
            shape: shape.clone(),
            buffer: dv_tmp.buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };

        Ok((dq, dk, dv))
    }

    // ── Recovery methods ───────────────────────────────────────────────────────

    /// Attempt to recover from a device lost error by rebuilding the entire
    /// GPU context (device, queue, pipelines, memory pool, shader cache).
    ///
    /// Returns `Ok(())` if recovery succeeded, `Err(GpuError)` otherwise.
    /// On success, all previously compiled pipelines are restored.
    /// GPU tensor buffers allocated before the loss are INVALID and must be
    /// re-uploaded.
    ///
    /// # Safety
    /// Caller must ensure no GPU operations are in-flight during recovery.
    /// Uses raw pointer dereference to mutate `&self` fields — only safe when
    /// exclusive access is guaranteed (all GPU ops blocked).
    pub unsafe fn try_rebuild(&self) -> Result<(), GpuError> {
        tracing::warn!("Attempting GPU context rebuild (device lost recovery)...");

        let self_ptr = self as *const GpuContext as *mut GpuContext;

        // 1. Clear memory pool
        (*self_ptr).clear_memory_pool();

        // 2. Clear caches
        (*self_ptr).shader_cache.clear();
        (*self_ptr).bind_group_layout_cache.clear();
        (*self_ptr).pipelines.clear();
        *(*self_ptr).bind_group_cache_mutex.lock().unwrap_or_else(|e| e.into_inner()) = HashMap::new();

        // 3. Recreate device + queue from new adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            },
        ))
        .map_err(|_| GpuError::DeviceLost("No adapter found after device lost".into()))?;

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
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if adapter_features.contains(wgpu::Features::PIPELINE_CACHE) {
            required_features |= wgpu::Features::PIPELINE_CACHE;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Nexora GPU Device (recovered)"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|e| GpuError::DeviceLost(format!("Device re-request failed: {}", e)))?;

        // 4. Replace device + queue
        (*self_ptr).device = device;
        (*self_ptr).queue = queue;

        // 5. Recreate memory pool
        let new_pool = crate::gpu_memory::GpuMemoryPool::new(&(*self_ptr).device);
        *(*self_ptr).memory_pool.lock().unwrap_or_else(|e| e.into_inner()) = new_pool;

        // 6. Recreate command encoder
        let enc = (*self_ptr).device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nexora_reusable_encoder_recovered"),
        });
        *(*self_ptr).current_encoder.lock().unwrap_or_else(|e| e.into_inner()) = Some(enc);

        // 7. Recompile all pipelines
        match (*self_ptr).compile_all_pipelines() {
            Ok(()) => {
                tracing::info!("GPU context rebuilt successfully after device lost");
                Ok(())
            }
            Err(e) => {
                tracing::error!("GPU context rebuild partially failed (pipelines): {:?}", e);
                Err(e)
            }
        }
    }

    /// Soft reset: clears caches and memory pool without device recreation.
    /// Use for transient recovery (timeout, temporary OOM).
    ///
    /// # Safety
    /// Caller must ensure no GPU operations are in-flight during soft reset.
    pub unsafe fn soft_reset(&self) {
        tracing::debug!("GPU soft reset: clearing caches and memory pool");
        let self_ptr = self as *const GpuContext as *mut GpuContext;
        (*self_ptr).clear_memory_pool();
        let enc = (*self_ptr).device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nexora_reusable_encoder_reset"),
        });
        *(*self_ptr).current_encoder.lock().unwrap_or_else(|e| e.into_inner()) = Some(enc);
        (*self_ptr).ops_since_flush.store(0, std::sync::atomic::Ordering::Release);
    }
}

const MATMUL_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_tiled_main(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (dims.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile of A
        let a_row = row;
        let a_col = t * TILE_SIZE + lid.y;
        if (a_row < dims.M && a_col < dims.K) {
            tile_a[lid.x][lid.y] = a[a_row * dims.K + a_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // Load tile of B
        let b_row = t * TILE_SIZE + lid.x;
        let b_col = col;
        if (b_row < dims.K && b_col < dims.N) {
            tile_b[lid.x][lid.y] = b[b_row * dims.N + b_col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // Accumulate
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.2: INT8 QUANTIZED MATMUL
// ═══════════════════════════════════════════════════════════════════════════════
//
// A is packed int8 weights (4 values per u32), B is f32 activations.
// Dequantization happens on-the-fly: int8 → f32 × scale.
// Output C is f32.
//
// The shader uses the same tiled matmul structure as MATMUL_TILED_WGSL but
// adds int8 unpacking for the A matrix. Only the weight matrix (A) is
// quantized — B (activations) stays f32 for precision during training.

const MATMUL_INT8_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A is packed as u32 — each u32 holds 4 int8 values (little-endian byte order).
// B and C remain f32.
@group(0) @binding(0) var<storage, read> a_packed: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
    scale: f32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn unpack_int8(packed: u32, byte_idx: u32) -> f32 {
    let shift = byte_idx * 8u;
    let byte = (packed >> shift) & 0xFFu;
    var val = f32(byte);
    if (byte > 127u) {
        val = val - 256.0;
    }
    return val * uniforms.scale;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_int8_main(@builtin(local_invocation_id) lid: vec3<u32>,
                    @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (packed int8 → f32) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            let a_flat_idx = a_global_row * uniforms.K + a_global_col;
            let packed_idx = a_flat_idx / 4u;
            let byte_off = a_flat_idx % 4u;
            tile_a[lid.x][lid.y] = unpack_int8(a_packed[packed_idx], byte_off);
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (f32) ──
        let b_global_row = t * TILE_SIZE + lid.x;
        let b_global_col = col;
        if (b_global_row < uniforms.K && b_global_col < uniforms.N) {
            tile_b[lid.x][lid.y] = b[b_global_row * uniforms.N + b_global_col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // ── Accumulate ──
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.2a: F16 PACKED WEIGHT MATMUL (activation f32 × packed f16 weight)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Activation (a) is f32, weight (b) is packed F16 (2 values per u32).
// Weight buffer is half the size of f32 equivalent — saves VRAM bandwidth.
// On read: unpack 2 f16 per u32 → convert to f32 → accumulate in f32.
// Output C is f32 (same precision as regular matmul).
//
// The shader follows the same tiled structure as MATMUL_TILED_WGSL but with
// packed F16 reads for the B (weight) matrix. Only the weight matrix is F16 —
// activation and output remain f32 for precision during accumulation.

const MATMUL_F16_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A is f32 activations [M, K]; B is packed f16 weights [K, N] (2 f16 per u32);
// C is f32 output [M, N].
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn f16_bits_to_f32(f16_bits: u32) -> f32 {
    let sign = (f16_bits >> 15u) & 0x1u;
    var exp = (f16_bits >> 10u) & 0x1Fu;
    let mant = f16_bits & 0x3FFu;
    var f32_bits: u32;
    if (exp == 0u) {
        f32_bits = sign << 31u;
    } else if (exp == 31u) {
        f32_bits = (u32(sign) << 31u) | (0xFFu << 23u) | (mant << 13u);
    } else {
        f32_bits = (u32(sign) << 31u) | ((exp - 15u + 127u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f32_bits);
}

fn unpack_f16(packed: u32, idx: u32) -> f32 {
    let shift = idx * 16u;
    let f16_bits = (packed >> shift) & 0xFFFFu;
    return f16_bits_to_f32(f16_bits);
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_f16_main(@builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (f32 activation) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            tile_a[lid.x][lid.y] = a[a_global_row * uniforms.K + a_global_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (packed f16 weight → f32) ──
        let b_global_row = t * TILE_SIZE + lid.x;
        let b_global_col = col;
        if (b_global_row < uniforms.K && b_global_col < uniforms.N) {
            let b_flat_idx = b_global_row * uniforms.N + b_global_col;
            let packed_idx = b_flat_idx / 2u;
            let sub_idx = b_flat_idx % 2u;
            tile_b[lid.x][lid.y] = unpack_f16(b_packed[packed_idx], sub_idx);
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // ── Accumulate in f32 ──
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.2b: INT8 WEIGHT MATMUL (activation f32 × packed int8 weight RHS)
// ═══════════════════════════════════════════════════════════════════════════════
//
// A is f32 activations [M, K], B is packed int8 weights [N, K] (ORIGINAL
// orientation — NOT transposed!). The shader computes C = A × dequant(B)^T.
//
// Weight matrix W of shape [output_dim, hidden] is stored in its original
// row-major layout. The shader reads B[j][k] = W[col][t*TILE + lid.x] and
// accumulates C[i][j] = sum_k A[i][k] * dequant(W[j][k]).
//
// This avoids transposing the int8 data and matches the inference pattern:
// activation(f32) @ weight^T.

const MATMUL_INT8_WEIGHT_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A = f32 activations [M, K]; B = packed int8 weights [N, K] (NOT transposed); C = f32 output [M, N]
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> zero_points: array<f32>;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn unpack_int8_weight(packed: u32, byte_idx: u32, col: u32) -> f32 {
    let shift = byte_idx * 8u;
    let byte = (packed >> shift) & 0xFFu;
    var val = f32(byte);
    if (byte > 127u) {
        val = val - 256.0;
    }
    return val * scales[col] + zero_points[col];
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_int8_weight_main(@builtin(local_invocation_id) lid: vec3<u32>,
                           @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (f32 activation) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            tile_a[lid.x][lid.y] = a[a_global_row * uniforms.K + a_global_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (packed int8 weight [N, K] — NOT transposed) ──
        // B[j][k] = W[col][t*TILE + lid.x] — weight matrix in original orientation
        let b_j = col;                                    // N dimension
        let b_k = t * TILE_SIZE + lid.x;                  // K dimension
        if (b_j < uniforms.N && b_k < uniforms.K) {
            let b_flat_idx = b_j * uniforms.K + b_k;
            let packed_idx = b_flat_idx / 4u;
            let byte_off = b_flat_idx % 4u;
            tile_b[lid.x][lid.y] = unpack_int8_weight(b_packed[packed_idx], byte_off, b_j);
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // ── Accumulate C[row][col] += sum_k A[row][k] * dequant(W[col][k]) ──
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

const ELEMENTWISE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Cfg {
    numel: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(3) var<uniform> cfg: Cfg;

fn gelu_f32(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn elementwise_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) {
        return;
    }

    let a_val = a[i];
    // For binary ops, b[i] is used; for unary, b[i] is same as a[i] or ignored
    let b_val = b[i];

    switch cfg.op {
        case 0: { // Add
            out[i] = a_val + b_val;
        }
        case 1: { // Sub
            out[i] = a_val - b_val;
        }
        case 2: { // Mul
            out[i] = a_val * b_val;
        }
        case 3: { // Div
            out[i] = a_val / b_val;
        }
        case 4: { // Neg
            out[i] = -a_val;
        }
        case 5: { // Exp
            out[i] = exp(a_val);
        }
        case 6: { // Ln
            out[i] = log(a_val);
        }
        case 7: { // Powf
            out[i] = pow(a_val, b_val);
        }
        case 8: { // Sqrt
            out[i] = sqrt(a_val);
        }
        case 9: { // Relu
            out[i] = max(a_val, 0.0);
        }
        case 10: { // Gelu
            out[i] = gelu_f32(a_val);
        }
        case 11: { // Sigmoid
            out[i] = sigmoid_f32(a_val);
        }
        case 12: { // Tanh
            out[i] = tanh(a_val);
        }
        case 13: { // Silu
            out[i] = a_val * sigmoid_f32(a_val);
        }
        case 14: { // LeakyRelu — b_val acts as negative_slope
            out[i] = select(a_val * b_val, a_val, a_val > 0.0);
        }
        case 15: { // BinaryCrossEntropy — a=prediction, b=target
            let p = clamp(a_val, 1.0e-7, 1.0 - 1.0e-7);
            out[i] = -(b_val * log(p) + (1.0 - b_val) * log(1.0 - p));
        }
        case 17: { // Swiglu — gate(a) * x(b)
            out[i] = (a_val * sigmoid_f32(a_val)) * b_val;
        }
        case 18: { // Step — 1 if a > 0 else 0
            out[i] = select(0.0, 1.0, a_val > 0.0);
        }
        default: {
            out[i] = a_val;
        }
    }
}
"#;

const ELEMENTWISE_INPLACE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

struct Cfg {
    numel: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(1) var<uniform> cfg: Cfg;

fn gelu_f32(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn elementwise_inplace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) {
        return;
    }

    let x = data[i];
    switch cfg.op {
        case 4: { data[i] = -x; }
        case 5: { data[i] = exp(x); }
        case 6: { data[i] = log(x); }
        case 7: { data[i] = pow(x, 0.0); }
        case 8: { data[i] = sqrt(x); }
        case 9: { data[i] = max(x, 0.0); }
        case 10: { data[i] = gelu_f32(x); }
        case 11: { data[i] = sigmoid_f32(x); }
        case 12: { data[i] = tanh(x); }
        case 13: { data[i] = x * sigmoid_f32(x); }
        case 14: { // LeakyRelu — use _pad0 reinterpreted as f32 for negative_slope
            let slope = bitcast<f32>(cfg._pad0);
            data[i] = select(x * slope, x, x > 0.0);
        }
        case 18: { // Step — 1 if x > 0 else 0
            data[i] = select(0.0, 1.0, x > 0.0);
        }
        default: { data[i] = x; }
    }
}
"#;

// ── Phase 1.1.5: Gradient Clip ────────────────────────────────────────────────

const GRADIENT_CLIP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(1) var<storage, read> norm_sq: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> wg_scale: f32;
var<workgroup> wg_norm: f32;

@compute @workgroup_size(BLOCK_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let max_norm = bitcast<f32>(cfg.y);

    if (lid.x == 0u) {
        let n = norm_sq[0];
        wg_norm = sqrt(n);
        if (wg_norm > max_norm && wg_norm > 0.0) {
            wg_scale = max_norm / wg_norm;
            output[3] = 1.0;
        } else {
            wg_scale = 1.0;
            output[3] = 0.0;
        }
        output[0] = wg_norm;
        output[1] = max_norm;
        output[2] = wg_scale;
    }
    workgroupBarrier();

    let numel = arrayLength(&gradient);
    if (idx < numel) {
        gradient[idx] = gradient[idx] * wg_scale;
    }
}
"#;

// ── Phase 1.2: Reduce ─────────────────────────────────────────────────────────

const REDUCE_WGSL_TEMPLATE: &str = r#"
const BLOCK_SIZE: u32 = 256;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Cfg {
    numel: u32,
    num_groups: u32,
};

@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

fn reduce_op(a: f32, b: f32) -> f32 {
    let op = {{OP}}u;
    // Sum=0, Max=1, Min=2
    if (op == 0u) { return a + b; }
    if (op == 1u) { if (a > b) { return a; } else { return b; } }
    return min(a, b);
}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce_main(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wg_id: vec3<u32>) {
    let group_idx = wg_id.x;
    let items_per_group = (cfg.numel + cfg.num_groups - 1) / cfg.num_groups;

    // Load
    var val: f32;
    let base = group_idx * items_per_group;
    let idx = base + lid.x;
    if (idx < cfg.numel) {
        val = input[idx];
    } else {
        val = 0.0;
    }
    scratch[lid.x] = val;
    workgroupBarrier();

    // Tree-reduce
    var stride = BLOCK_SIZE / 2;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = reduce_op(scratch[lid.x], scratch[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2;
    }

    // Write result
    if (lid.x == 0u) {
        output[group_idx] = scratch[0];
    }
}
"#;

// ── Phase 2.1: Softmax (stable, per-row) ──────────────────────────────────────

const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn softmax_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let dim = cfg.y;
    let row = wg.x;
    if (row >= batch) { return; }

    let base = row * dim;

    // === Pass 1: find row max ===
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < dim) {
        let v = input[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = mx;
    workgroupBarrier();

    // tree-reduce max
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (scratch[lid.x] < scratch[lid.x + stride]) {
                scratch[lid.x] = scratch[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0u];
    workgroupBarrier();

    // === Pass 2: exp(x - max) and sum ===
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        sum += e;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    // tree-reduce sum
    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = scratch[0u];
    workgroupBarrier();

    // === Pass 3: normalize ===
    i = lid.x;
    while (i < dim) {
        output[base + i] = output[base + i] / row_sum;
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.2: RMSNorm ───────────────────────────────────────────────────────

const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn rms_norm_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;

    let base = row * dim;

    // === Pass 1: sum(x²) ===
    var ss: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        let v = input[base + i];
        ss += v * v;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = ss;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let rms = sqrt(scratch[0u] / f32(dim) + eps);
    workgroupBarrier();

    // === Pass 2: normalize ===
    i = lid.x;
    while (i < dim) {
        output[base + i] = (input[base + i] / rms) * weight[i];
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.3: RMSNorm Backward ──────────────────────────────────────────────

const RMSNORM_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn rms_norm_bwd_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;
    let base = row * dim;

    // Pass 1: sum(x²) → rms → inv_rms
    var ssq: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        let xv = input[base + i];
        ssq += xv * xv;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = ssq;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let rms = sqrt(scratch[0u] / f32(dim) + eps);
    let inv_rms = 1.0 / rms;

    // Pass 2: sum(grad * x)
    var sum_x_g: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        sum_x_g += input[base + i] * grad[base + i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum_x_g;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let row_sum_x_g = scratch[0u];

    let rms_grad_factor = -inv_rms * inv_rms * inv_rms / f32(dim);

    // Pass 3: compute dx (per element) + accumulate dw (atomic across rows)
    i = lid.x;
    while (i < dim) {
        let xv = input[base + i];
        let gv = grad[base + i];
        let wv = weight[i];
        dx[base + i] = gv * wv * inv_rms + wv * xv * rms_grad_factor * row_sum_x_g;
        // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
        loop {
            let prev_bits = atomicLoad(&dw[i]);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + gv * xv * inv_rms);
            let res = atomicCompareExchangeWeak(&dw[i], prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.3: Cross-Entropy ─────────────────────────────────────────────────

const CROSS_ENTROPY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn cross_entropy_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let num_classes = cfg.y;
    let row = wg.x;
    if (row >= batch) { return; }

    let base = row * num_classes;
    let label = targets[row];

    // === Pass 1: find row max (stable) ===
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < num_classes) {
        let v = logits[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = mx;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (scratch[lid.x] < scratch[lid.x + stride]) {
                scratch[lid.x] = scratch[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0u];
    workgroupBarrier();

    // === Pass 2: sum(exp(x - max)) ===
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < num_classes) {
        sum += exp(logits[base + i] - row_max);
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let log_sum_exp = log(scratch[0u]) + row_max;
    workgroupBarrier();

    // === Pass 3: loss = log_sum_exp - logits[label] ===
    if (lid.x == 0u) {
        let target_logit = logits[base + label];
        losses[row] = log_sum_exp - target_logit;
    }
}
"#;

const CROSS_ENTROPY_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> softmax: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256u)
fn cross_entropy_bwd_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let batch = cfg.x;
    let classes = cfg.y;
    let total = batch * classes;
    let idx = gid.x;
    if (idx >= total) { return; }
    let b = idx / classes;
    let c = idx % classes;
    let t = targets[b];
    let p = softmax[idx];
    let g = grad[b];
    d_logits[idx] = g * (p - select(0.0, 1.0, c == t));
}
"#;

// ── Phase 2.4: Embedding (gather) ────────────────────────────────────────────

const EMBEDDING_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn embedding_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vocab_size = cfg.x;
    let dim = cfg.y;
    let seq_len = cfg.z;

    let idx = gid.x;
    if (idx >= seq_len * dim) { return; }

    let d = idx % dim;
    let s = idx / dim;
    let token_id = ids[s];
    output[idx] = weight[token_id * dim + d];
}
"#;

const EMBEDDING_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_weight: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256u)
fn embedding_backward_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vocab_size = cfg.x;
    let dim = cfg.y;
    let num_ids = cfg.z;
    let total = num_ids * dim;

    let idx = gid.x;
    if (idx >= total) { return; }

    let d = idx % dim;
    let s = idx / dim;
    let token_id = ids[s];
    if (token_id < vocab_size) {
        let g = grad[idx];
        // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
        let atom_ptr = &d_weight[token_id * dim + d];
        loop {
            let prev_bits = atomicLoad(atom_ptr);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + g);
            let res = atomicCompareExchangeWeak(atom_ptr, prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
    }
}
"#;

// ── Phase 2.5: LayerNorm ────────────────────────────────────────────────────

const LAYERNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn layer_norm_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;

    let base = row * dim;

    // === Pass 1: mean ===
    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        sum += input[base + i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = scratch[0u] / f32(dim);
    workgroupBarrier();

    // === Pass 2: variance ===
    var var_sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let diff = input[base + i] - mean;
        var_sum += diff * diff;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = var_sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let variance = scratch[0u] / f32(dim);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // === Pass 3: normalize ===
    i = lid.x;
    while (i < dim) {
        let normalized = (input[base + i] - mean) * inv_std;
        output[base + i] = normalized * weight[i] + bias[i];
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.6: Transpose ────────────────────────────────────────────────────

const TRANSPOSE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = cfg.x;
    let cols = cfg.y;
    let idx = gid.x;
    if (idx >= rows * cols) { return; }
    let r = idx / cols;
    let c = idx % cols;
    output[c * rows + r] = input[r * cols + c];
}
"#;

// ── Phase 3: Fused Attention (Flash Attention-style) ─────────────────────────

const FUSED_ATTENTION_WGSL: &str = r#"
const BLOCK_SIZE = 256u;
const TILE_SIZE = 32u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg1: vec4<u32>;
@group(0) @binding(5) var<uniform> cfg2: vec4<u32>;

var<workgroup> q_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> kv_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> score_scratch: array<f32, TILE_SIZE>;
var<workgroup> exp_scratch: array<f32, TILE_SIZE>;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn fused_attention_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch = cfg1.x;
    let heads = cfg1.y;
    let seq_len = cfg1.z;
    let dim = cfg1.w;
    let scale = bitcast<f32>(cfg2.x);
    let causal = cfg2.y;

    let tid = lid.x;

    let wg_base = cfg2.z;
    let wg_flat = wg_base + wg_id.x;
    let q_pos = wg_flat % seq_len;
    let head = (wg_flat / seq_len) % heads;
    let batch_idx = wg_flat / (seq_len * heads);

    if (batch_idx >= batch) { return; }
    if (tid >= dim) { return; }

    let head_stride = seq_len * dim;
    let batch_stride = heads * head_stride;
    let q_off = batch_idx * batch_stride + head * head_stride + q_pos * dim;
    let kv_base = batch_idx * batch_stride + head * head_stride;

    // Load Q row
    q_scratch[tid] = q[q_off + tid];
    workgroupBarrier();

    // Online softmax state
    var m: f32 = -3.402823e+38;
    var d: f32 = 0.0;
    var o: f32 = 0.0;

    // Loop over KV tiles
    var tile_start: u32 = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // === Step 1: compute scores for this tile ===
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;

            kv_scratch[tid] = k[kv_base + k_pos * dim + tid];
            workgroupBarrier();

            var partial = q_scratch[tid] * kv_scratch[tid];
            scratch[tid] = partial;
            workgroupBarrier();

            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) {
                    scratch[tid] = scratch[tid] + scratch[tid + stride];
                }
                workgroupBarrier();
                stride = stride / 2u;
            }

            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) {
                    s = -3.402823e+38;
                }
                score_scratch[ki] = s;
            }
            workgroupBarrier();
        }

        // === Step 2: max of tile scores ===
        if (tid < tile_size) {
            scratch[tid] = score_scratch[tid];
        } else {
            scratch[tid] = -3.402823e+38;
        }
        workgroupBarrier();

        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) {
                    scratch[tid] = scratch[tid + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let m_tile = scratch[0u];
        workgroupBarrier();

        // === Step 3: exp(score - m_tile) and sum ===
        if (tid < tile_size) {
            let e = exp(score_scratch[tid] - m_tile);
            exp_scratch[tid] = e;
            scratch[tid] = e;
        } else {
            scratch[tid] = 0.0;
        }
        workgroupBarrier();

        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                scratch[tid] = scratch[tid] + scratch[tid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let sum_exp_tile = scratch[0u];
        workgroupBarrier();

        // === Step 4: Online softmax update ===
        let m_new = max(m, m_tile);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(m_tile - m_new);

        d = old_scale * d + tile_scale * sum_exp_tile;
        o = o * old_scale;

        // === Step 5: Accumulate V ===
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = v[kv_base + k_pos * dim + tid];
            workgroupBarrier();

            o += tile_scale * exp_scratch[ki] * kv_scratch[tid];
            workgroupBarrier();
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // === Final: output = o / d ===
    output[q_off + tid] = o / d;
}
"#;

const FUSED_ATTENTION_BACKWARD_WGSL: &str = r#"
const BLOCK_SIZE = 256u;
const TILE_SIZE = 32u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> dO: array<f32>;
@group(0) @binding(4) var<storage, read_write> dq: array<f32>;
@group(0) @binding(5) var<storage, read_write> dk: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> dv: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> cfg1: vec4<u32>;
@group(0) @binding(8) var<uniform> cfg2: vec4<u32>;

var<workgroup> q_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> kv_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> score_scratch: array<f32, TILE_SIZE>;
var<workgroup> exp_scratch: array<f32, TILE_SIZE>;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn fused_attn_backward_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_heads = cfg1.x;
    let seq_len = cfg1.y;
    let dim = cfg1.z;
    let scale = bitcast<f32>(cfg1.w);
    let causal = cfg2.x;

    let tid = lid.x;
    let wg_base = cfg2.y;
    let wg_flat = wg_base + wg_id.x;
    let q_pos = wg_flat % seq_len;
    let head = wg_flat / seq_len;
    if (head >= batch_heads) { return; }
    if (tid >= dim) { return; }

    let head_stride = seq_len * dim;
    let q_off = head * head_stride + q_pos * dim;
    let base = head * head_stride;

    // Load Q row into workgroup memory
    q_scratch[tid] = q[q_off + tid];
    workgroupBarrier();

    // ── Pass 1: find softmax normalization constants + sum(P*dP) ──
    var m: f32 = -3.402823e+38;
    var d_norm: f32 = 0.0;
    var sum_P_dP: f32 = 0.0;

    var tile_start: u32 = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // Compute scores for this tile
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = k[base + k_pos * dim + tid];
            workgroupBarrier();
            var partial = q_scratch[tid] * kv_scratch[tid];
            scratch[tid] = partial;
            workgroupBarrier();
            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
                workgroupBarrier();
                stride = stride / 2u;
            }
            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) { s = -3.402823e+38; }
                score_scratch[ki] = s;
            }
            workgroupBarrier();
        }

        // Tile max
        if (tid < tile_size) { scratch[tid] = score_scratch[tid]; }
        else { scratch[tid] = -3.402823e+38; }
        workgroupBarrier();
        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) { scratch[tid] = scratch[tid + stride]; }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_max = scratch[0u];
        workgroupBarrier();

        // Tile exp(score - tile_max) and sum
        if (tid < tile_size) {
            let e = exp(score_scratch[tid] - tile_max);
            exp_scratch[tid] = e;
            scratch[tid] = e;
        } else { scratch[tid] = 0.0; }
        workgroupBarrier();
        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_sum = scratch[0u];
        workgroupBarrier();

        // Online softmax: d = old_scale * d + tile_scale * tile_sum
        let m_new = max(m, tile_max);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(tile_max - m_new);
        let d_prev = d_norm;
        d_norm = old_scale * d_norm + tile_scale * tile_sum;

        // Compute dP and accumulate sum(P*dP)
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = v[base + k_pos * dim + tid];
            workgroupBarrier();
            let dp_partial = dO[q_off + tid] * kv_scratch[tid];
            scratch[tid] = dp_partial;
            workgroupBarrier();
            var stride2 = BLOCK_SIZE / 2u;
            while (stride2 > 0u) {
                if (tid < stride2) { scratch[tid] = scratch[tid] + scratch[tid + stride2]; }
                workgroupBarrier();
                stride2 = stride2 / 2u;
            }
            if (tid == 0u) {
                let P_k = tile_scale * exp_scratch[ki] / d_norm;
                let dP_k = scratch[0u];
                // Rescale old contributions: old_scale * d_prev / d_norm * sum_P_dP
                // Add new: P_k * dP_k
                sum_P_dP = old_scale * d_prev / d_norm * sum_P_dP + P_k * dP_k;
            }
            workgroupBarrier();
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // ── Pass 2: recompute, compute dS, accumulate dQ + dK + dV ──
    m = -3.402823e+38;
    d_norm = 0.0;
    var dq_acc: f32 = 0.0;

    tile_start = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // Recompute scores
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = k[base + k_pos * dim + tid];
            workgroupBarrier();
            var partial = q_scratch[tid] * kv_scratch[tid];
            scratch[tid] = partial;
            workgroupBarrier();
            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
                workgroupBarrier();
                stride = stride / 2u;
            }
            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) { s = -3.402823e+38; }
                score_scratch[ki] = s;
            }
            workgroupBarrier();
        }

        // Tile max
        if (tid < tile_size) { scratch[tid] = score_scratch[tid]; }
        else { scratch[tid] = -3.402823e+38; }
        workgroupBarrier();
        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) { scratch[tid] = scratch[tid + stride]; }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_max = scratch[0u];
        workgroupBarrier();

        if (tid < tile_size) {
            let e = exp(score_scratch[tid] - tile_max);
            exp_scratch[tid] = e;
            scratch[tid] = e;
        } else { scratch[tid] = 0.0; }
        workgroupBarrier();
        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_sum = scratch[0u];
        workgroupBarrier();

        let m_new = max(m, tile_max);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(tile_max - m_new);
        d_norm = old_scale * d_norm + tile_scale * tile_sum;

        // For each key in this tile: dQ, dK, dV
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            let P_k = tile_scale * exp_scratch[ki] / d_norm;

            // dP_k = dot(dO[q_pos], V[k_pos]) — recompute
            kv_scratch[tid] = v[base + k_pos * dim + tid];
            workgroupBarrier();
            let dp_partial = dO[q_off + tid] * kv_scratch[tid];
            scratch[tid] = dp_partial;
            workgroupBarrier();
            var stride3 = BLOCK_SIZE / 2u;
            while (stride3 > 0u) {
                if (tid < stride3) { scratch[tid] = scratch[tid] + scratch[tid + stride3]; }
                workgroupBarrier();
                stride3 = stride3 / 2u;
            }
            let dP_k = scratch[0u];
            workgroupBarrier();

            // dS = P_k * (dP_k - sum_P_dP)
            let dS = P_k * (dP_k - sum_P_dP);

            // Load K for dQ + dK
            kv_scratch[tid] = k[base + k_pos * dim + tid];
            workgroupBarrier();

            // dQ[d] += dS * K[k, d] / scale
            dq_acc += dS * kv_scratch[tid] / scale;

            // dV[k, d] += P_k * dO[q, d]  (atomic)
            let dv_off = base + k_pos * dim + tid;
            let dv_val = P_k * dO[q_off + tid];
            // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
            loop {
                let prev_bits = atomicLoad(&dv[dv_off]);
                let prev_val = bitcast<f32>(prev_bits);
                let new_bits = bitcast<u32>(prev_val + dv_val);
                let res = atomicCompareExchangeWeak(&dv[dv_off], prev_bits, new_bits);
                if (res.exchanged) { break; }
            }

            // dK[k, d] += dS * Q[q, d] / scale  (atomic)
            let dk_off = base + k_pos * dim + tid;
            let dk_val = dS * q_scratch[tid] / scale;
            // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
            loop {
                let prev_bits = atomicLoad(&dk[dk_off]);
                let prev_val = bitcast<f32>(prev_bits);
                let new_bits = bitcast<u32>(prev_val + dk_val);
                let res = atomicCompareExchangeWeak(&dk[dk_off], prev_bits, new_bits);
                if (res.exchanged) { break; }
            }
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // Write dQ for this (head, q_pos, d)
    dq[q_off + tid] = dq_acc;
}
"#;

const FILL_ZERO_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256)
fn fill_zero_main(@builtin(global_invocation_id) id: vec3<u32>) {
    buf[id.x] = 0.0;
}
"#;

const FILL_CONSTANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec2<u32>;

@compute @workgroup_size(256)
fn fill_constant_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let numel = cfg.x;
    let value = bitcast<f32>(cfg.y);
    if (id.x < numel) {
        buf[id.x] = value;
    }
}
"#;

const FILL_ZERO_U32_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;

@compute @workgroup_size(256)
fn fill_zero_u32_main(@builtin(global_invocation_id) id: vec3<u32>) {
    buf[id.x] = 0u;
}
"#;

const SCALE_INPLACE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec2<u32>;

@compute @workgroup_size(256)
fn scale_inplace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let numel = cfg.x;
    let scale = bitcast<f32>(cfg.y);
    if (id.x < numel) {
        buf[id.x] = buf[id.x] * scale;
    }
}
"#;

const GRADIENT_ALLREDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> grad_buffers: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_grads: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;
// cfg.x = numel per replica (total float count per gradient set)
// cfg.y = num_replicas

@compute @workgroup_size(256)
fn gradient_allreduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numel = cfg.x;
    let num_replicas = cfg.y;
    let idx = gid.x;
    if (idx >= numel) { return; }

    var sum = 0.0;
    for (var r = 0u; r < num_replicas; r++) {
        sum += grad_buffers[r * numel + idx];
    }
    out_grads[idx] = sum / f32(num_replicas);
}
"#;

const L2_NORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn l2_norm_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = cfg.x;
    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < numel) {
        sum += input[i] * input[i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        output[0] = scratch[0u];
    }
}
"#;

const CAUSAL_SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn causal_softmax_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let dim = cfg.y;
    let row = wg.x / dim;
    let col = wg.x % dim;
    if (row >= batch) { return; }

    let base = row * dim;
    let causal_end = col + 1u;

    // Pass 1: find max over causal prefix
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < causal_end) {
        let v = input[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = mx;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (scratch[lid.x] < scratch[lid.x + stride]) {
                scratch[lid.x] = scratch[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0u];
    workgroupBarrier();

    // Pass 2: exp and sum over causal prefix
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < causal_end) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        sum += e;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = scratch[0u];
    workgroupBarrier();

    // Pass 3: normalize (only for positions <= col)
    i = lid.x;
    while (i < causal_end) {
        output[base + i] = output[base + i] / row_sum;
        i += BLOCK_SIZE;
    }
    // Zero out positions beyond causal_end
    i = lid.x + causal_end;
    while (i < dim) {
        output[base + i] = 0.0;
        i += BLOCK_SIZE;
    }
}
"#;

const ROTARY_EMBEDDING_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> x : array<f32>;
@group(0) @binding(1) var<storage, read> cos : array<f32>;
@group(0) @binding(2) var<storage, read> sin : array<f32>;
struct RotaryConfig {
    total_rows: u32,
    dim: u32,
    half: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> cfg : RotaryConfig;

@compute @workgroup_size(256)
fn rotary_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= cfg.total_rows * cfg.half {
        return;
    }
    let row = idx / cfg.half;
    let pair = idx % cfg.half;

    let i1 = row * cfg.dim + pair;
    let i2 = row * cfg.dim + pair + cfg.half;

    let v1 = x[i1];
    let v2 = x[i2];
    let c = cos[pair];
    let s = sin[pair];

    x[i1] = v1 * c - v2 * s;
    x[i2] = v1 * s + v2 * c;
}
"#;

const REPEAT_HEADS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
struct RepeatConfig {
    seq: u32,
    kv_heads: u32,
    q_heads: u32,
    dim: u32,
};
@group(0) @binding(2) var<uniform> cfg : RepeatConfig;

@compute @workgroup_size(256)
fn repeat_heads_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = cfg.seq * cfg.q_heads * cfg.dim;
    if idx >= total {
        return;
    }
    // dst layout: [q_heads, seq, dim] (head-major, compatible with fused_attention)
    let qh = idx / (cfg.seq * cfg.dim);
    let rem1 = idx % (cfg.seq * cfg.dim);
    let s = rem1 / cfg.dim;
    let d = rem1 % cfg.dim;

    // Map q_heads -> kv_heads (grouped)
    let groups = cfg.q_heads / cfg.kv_heads;
    let kvh = qh / groups;
    let src_idx = s * (cfg.kv_heads * cfg.dim) + kvh * cfg.dim + d;
    dst[idx] = src[src_idx];
}
"#;
const TEMPERATURE_SCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256)
fn temperature_scale_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let temp = bitcast<f32>(cfg.x);
    let numel = cfg.y;
    if (id.x < numel) {
        logits[id.x] = logits[id.x] / temp;
    }
}
"#;

const TOP_K_MASK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> wg_buf: array<f32, BLOCK_SIZE>;
var<workgroup> wg_idx: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn top_k_mask_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let vocab = cfg.x;
    let k = cfg.y;
    let row = wg.x;
    let base = row * vocab;

    // Find top-k threshold using local maxima
    var local_max: f32 = -1.0;
    var i = lid.x;
    while (i < vocab) {
        let p = probs[base + i];
        if (p > local_max) { local_max = p; }
        i += BLOCK_SIZE;
    }
    wg_buf[lid.x] = local_max;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (wg_buf[lid.x] < wg_buf[lid.x + stride]) {
                wg_buf[lid.x] = wg_buf[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = wg_buf[0u];
    workgroupBarrier();

    // Compute threshold as a fraction of row_max
    let threshold = row_max * 0.01;

    // Count how many exceed threshold
    var count: u32 = 0u;
    i = lid.x;
    while (i < vocab) {
        if (probs[base + i] >= threshold) {
            count = count + 1u;
        }
        i += BLOCK_SIZE;
    }
    wg_idx[lid.x] = count;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            wg_idx[lid.x] = wg_idx[lid.x] + wg_idx[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total_above = wg_idx[0u];
    workgroupBarrier();

    // Only keep top-k
    let effective_k = min(k, total_above);
    var kept: u32 = 0u;
    i = lid.x;
    while (i < vocab) {
        if (probs[base + i] >= threshold && kept < effective_k) {
            kept = kept + 1u;
        } else if (probs[base + i] < threshold && lid.x < vocab) {
            probs[base + i] = 0.0;
        }
        i += BLOCK_SIZE;
    }
}
"#;

const TOP_P_MASK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> wg_buf: array<f32, BLOCK_SIZE>;
var<workgroup> wg_idx: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn top_p_mask_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let vocab = cfg.x;
    let top_p = bitcast<f32>(cfg.y);
    if (top_p <= 0.0 || top_p >= 1.0) {
        return;
    }
    let row = wg.x;
    let base = row * vocab;

    // Iteratively accumulate largest probabilities until cum >= top_p
    var threshold: f32 = 3.402823e+38;
    var cum: f32 = 0.0;

    for (var iter: u32 = 0u; iter < vocab; iter = iter + 1u) {
        // Find max among elements below current threshold
        var mx: f32 = -3.402823e+38;
        var mx_idx: u32 = 0u;
        var i = lid.x;
        while (i < vocab) {
            let v = probs[base + i];
            if (v > mx && v < threshold - 1e-6) {
                mx = v;
                mx_idx = i;
            }
            i += BLOCK_SIZE;
        }
        wg_buf[lid.x] = mx;
        wg_idx[lid.x] = mx_idx;
        workgroupBarrier();

        // Workgroup reduction: find max and its index
        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (lid.x < stride) {
                if (wg_buf[lid.x] < wg_buf[lid.x + stride]) {
                    wg_buf[lid.x] = wg_buf[lid.x + stride];
                    wg_idx[lid.x] = wg_idx[lid.x + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        let max_val = wg_buf[0u];
        if (max_val <= 0.0) { break; }

        cum += max_val;
        threshold = max_val;

        if (cum >= top_p) { break; }
    }

    // Zero out everything below the threshold
    var i = lid.x;
    while (i < vocab) {
        if (probs[base + i] < threshold - 1e-6) {
            probs[base + i] = 0.0;
        }
        i += BLOCK_SIZE;
    }
}
"#;

const MULTINOMIAL_SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

fn xorshift64(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return x;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
    return f32(xorshift64(state) & 0x7FFFFFu) / f32(0x7FFFFFu);
}

@compute @workgroup_size(1)
fn multinomial_main(@builtin(workgroup_id) wg: vec3<u32>) {
    let vocab = cfg.x;
    let seed_lo = cfg.y;
    let seed_hi = cfg.z;
    let row = wg.x;
    let base = row * vocab;

    var rng_state = seed_lo ^ (row * 0x9E3779B9u);
    var warmup = xorshift64(&rng_state);
    let r = random_f32(&rng_state);

    var total: f32 = 0.0;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        total += probs[base + i];
    }

    var cumulative: f32 = 0.0;
    var chosen: u32 = vocab - 1u;
    let threshold = r * total;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        cumulative += probs[base + i];
        if cumulative >= threshold {
            chosen = i;
            break;
        }
    }

    output[row] = chosen;
}
"#;

// ─── ADAM OPTIMIZER WGSL ───────────────────────────────────────────────────────

const ADAM_WGSL: &str = r#"
struct Config {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    step: f32,
};

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&param);
    if (i >= n) { return; }

    let g = grad[i];
    let p = param[i];

    let m_new = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g;
    let v_new = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g * g;

    let m_hat = m_new / cfg.bias_corr1;
    let v_hat = v_new / cfg.bias_corr2;

    let update = cfg.lr * m_hat / (sqrt(v_hat) + cfg.eps);
    let decay = cfg.lr * cfg.weight_decay * p;
    param[i] = p - update - decay;

    m[i] = m_new;
    v[i] = v_new;
}
"#;

const LAYERNORM_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> dw: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> db: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn layer_norm_bwd_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;
    let base = row * dim;

    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        sum += input[base + i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = scratch[0u] / f32(dim);

    var var_sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let diff = input[base + i] - mean;
        var_sum += diff * diff;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = var_sum;
    workgroupBarrier();
    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let variance = scratch[0u] / f32(dim);
    let sigma = sqrt(variance + eps);
    let inv_sigma = 1.0 / sigma;

    var sum_dy: f32 = 0.0;
    var sum_dy_xhat: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let gv = grad[base + i];
        let x_hat = (input[base + i] - mean) * inv_sigma;
        sum_dy += gv;
        sum_dy_xhat += gv * x_hat;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum_dy;
    scratch[lid.x + BLOCK_SIZE / 2u] = sum_dy_xhat;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let row_sum_dy = scratch[0u];

    i = lid.x;
    while (i < BLOCK_SIZE / 2u) {
        scratch[i] = scratch[i + BLOCK_SIZE / 2u];
        i += BLOCK_SIZE / 2u;
    }
    workgroupBarrier();

    stride = BLOCK_SIZE / 4u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let row_sum_dy_xhat = scratch[0u];

    let inv_n = 1.0 / f32(dim);

    i = lid.x;
    while (i < dim) {
        let x_hat = (input[base + i] - mean) * inv_sigma;
        let gv = grad[base + i];
        let wv = weight[i];
        dx[base + i] = inv_sigma * (gv - row_sum_dy * inv_n - x_hat * row_sum_dy_xhat * inv_n);
        // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
        loop {
            let prev_bits = atomicLoad(&dw[i]);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + gv * x_hat);
            let res = atomicCompareExchangeWeak(&dw[i], prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
        loop {
            let prev_bits = atomicLoad(&db[i]);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + gv);
            let res = atomicCompareExchangeWeak(&db[i], prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
        i += BLOCK_SIZE;
    }
}
"#;
