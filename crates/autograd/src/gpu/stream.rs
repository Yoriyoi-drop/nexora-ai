use std::time::Duration;
use std::time::Instant;


use super::gpu_tensor::{readback_with_timeout, GpuDtype, GpuTensor};
use super::gpu_types::*;
use super::wgsl::*;
#[cfg(feature = "cuda")]
use super::cuda::CudaTensor;
/// Maximum workgroups per dimension for wgpu/WebGPU (65535).
/// Any dispatch exceeding this must be chunked across multiple dispatches
/// using buffer-slice bindings with per-chunk offsets.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Default workgroup size used by most Nexora WGSL shaders.
const DEFAULT_WORKGROUP_SIZE: u32 = 256;

impl GpuContext {
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
            let buf_size = (query_count as u64 * wgpu::QUERY_SIZE as u64).div_ceil(256) * 256;
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

    pub(crate) fn get_or_create_shader_module(
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

    pub(crate) fn get_or_create_bind_group_layout(
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


    // ── Sampler / utility helpers ──────────────────────────────────────────────

    pub(crate) fn compile_fill_zero(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fill_zero",
            &[storage_binding(0, false)],
            std::borrow::Cow::Borrowed(FILL_ZERO_WGSL),
            "fill_zero_main",
        )
    }

    pub(crate) fn compile_fill_constant(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fill_constant",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(FILL_CONSTANT_WGSL),
            "fill_constant_main",
        )
    }

    pub(crate) fn compile_fill_zero_u32(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "fill_zero_u32",
            &[storage_binding(0, false)],
            std::borrow::Cow::Borrowed(FILL_ZERO_U32_WGSL),
            "fill_zero_u32_main",
        )
    }

    pub(crate) fn compile_moe_scatter_add(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "moe_scatter_add",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
            ],
            std::borrow::Cow::Borrowed(MOE_SCATTER_ADD_WGSL),
            "moe_scatter_add_main",
        )
    }

    pub(crate) fn compile_scale_inplace(&mut self) -> Result<(), GpuError> {
        self.compile_pipeline(
            "scale_inplace",
            &[storage_binding(0, false), uniform_binding(1)],
            std::borrow::Cow::Borrowed(SCALE_INPLACE_WGSL),
            "scale_inplace_main",
        )
    }

    pub(crate) fn compile_gradient_allreduce(&mut self) -> Result<(), GpuError> {
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
    pub(crate) fn dispatch_impl(
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
        let total_groups = numel.div_ceil(wg_size);

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
            let chunk_groups = chunk_elems.div_ceil(wg_size);
            let chunk_bytes = (chunk_elems as u64) * element_size;

            let mut entries = Vec::with_capacity(storage_bindings.len() + uniform_bindings.len());
            for &(binding, buf) in storage_bindings {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: byte_offset,
                        size: wgpu::BufferSize::new(chunk_bytes),
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
        let _slice_numel = (len as usize) * rest;
        let element_size = match tensor.dtype {
            GpuDtype::F32 => 4,
            GpuDtype::F16 | GpuDtype::Bf16 => 2,
            GpuDtype::I8 => 1,
        };
        let _byte_offset = (pos as usize) * dim0 /* actually per-row */;
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
    pub(crate) fn next_query_index(&self) -> (u32, u32) {
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
    pub(crate) fn read_timestamps_from_buffer(
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
            let buf_size = (query_count as u64 * wgpu::QUERY_SIZE as u64).div_ceil(256) * 256;
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
                ),
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
        let wg_m = m.div_ceil(tile_size) as u32;
        let wg_n = n_dim.div_ceil(tile_size) as u32;

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
}


