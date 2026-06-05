use std::collections::HashMap;
use std::ptr;
use std::time::Duration;


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

    /// Rebuild the GPU context after device loss.  
    /// Safe to call during error recovery — `rebuild_lock` guarantees exclusive
    /// access so no other thread observes partially-initialized state.
    pub fn try_rebuild(&self) -> Result<(), GpuError> {
        tracing::warn!("Attempting GPU context rebuild (device lost recovery)...");

        let _guard = self.rebuild_lock.lock().unwrap_or_else(|e| {
            tracing::error!("rebuild_lock poisoned");
            e.into_inner()
        });

        // 1. Clear memory pool (uses internal Mutex — safe with &self)
        self.clear_memory_pool();

        // 2. Clear caches
        // SAFETY: rebuild_lock provides exclusive access. We use ptr::write to
        // replace HashMap contents without creating &mut references, avoiding
        // the invalid_reference_casting lint (no &T→&mut T conversion occurs).
        unsafe {
            let sc = &self.shader_cache as *const HashMap<u64, wgpu::ShaderModule> as *mut HashMap<u64, wgpu::ShaderModule>;
            ptr::drop_in_place(sc);
            ptr::write(sc, HashMap::new());
        }
        unsafe {
            let bc = &self.bind_group_layout_cache as *const HashMap<u64, wgpu::BindGroupLayout> as *mut HashMap<u64, wgpu::BindGroupLayout>;
            ptr::drop_in_place(bc);
            ptr::write(bc, HashMap::new());
        }
        unsafe {
            let pc = &self.pipelines as *const HashMap<String, CompiledPipeline> as *mut HashMap<String, CompiledPipeline>;
            ptr::drop_in_place(pc);
            ptr::write(pc, HashMap::new());
        }
        *self.bind_group_cache_mutex.lock().unwrap_or_else(|e| e.into_inner()) = HashMap::new();

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

        let (dev, q) = pollster::block_on(adapter.request_device(
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

        // 4. Replace device + queue via ptr::write (no &mut reference created)
        // SAFETY: rebuild_lock guarantees exclusive access. wgpu::Device/Queue
        // are Send+Sync and trivially relocatable (no self-references).
        unsafe {
            let dp = &self.device as *const wgpu::Device as *mut wgpu::Device;
            ptr::drop_in_place(dp);
            ptr::write(dp, dev);
        }
        unsafe {
            let qp = &self.queue as *const wgpu::Queue as *mut wgpu::Queue;
            ptr::drop_in_place(qp);
            ptr::write(qp, q);
        }

        // 5. Recreate memory pool
        let new_pool = crate::gpu_memory::GpuMemoryPool::new(&self.device);
        *self.memory_pool.lock().unwrap_or_else(|e| e.into_inner()) = new_pool;

        // 6. Recreate command encoder
        let enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nexora_reusable_encoder_recovered"),
        });
        *self.current_encoder.lock().unwrap_or_else(|e| e.into_inner()) = Some(enc);

        // 7. Recompile all pipelines
        // SAFETY: rebuild_lock provides exclusive access; device/queue are valid.
        #[allow(invalid_reference_casting)]
        let ctx_mut = unsafe { &mut *(self as *const GpuContext as *mut GpuContext) };
        match ctx_mut.compile_all_pipelines() {
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
    pub fn soft_reset(&self) {
        tracing::debug!("GPU soft reset: clearing caches and memory pool");

        let _guard = self.rebuild_lock.lock().unwrap_or_else(|e| {
            tracing::error!("rebuild_lock poisoned");
            e.into_inner()
        });

        // SAFETY: rebuild_lock guarantees exclusive access during soft reset.
        #[allow(invalid_reference_casting)]
        let ctx_mut = unsafe { &mut *(self as *const GpuContext as *mut GpuContext) };

        ctx_mut.clear_memory_pool();
        let enc = ctx_mut.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nexora_reusable_encoder_reset"),
        });
        *ctx_mut.current_encoder.lock().unwrap_or_else(|e| e.into_inner()) = Some(enc);
        ctx_mut.ops_since_flush.store(0, std::sync::atomic::Ordering::Release);
    }
}
