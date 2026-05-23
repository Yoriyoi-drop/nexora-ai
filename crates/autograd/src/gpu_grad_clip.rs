use std::time::Duration;

use crate::gpu::{GpuContext, GpuError, GpuTensor};

/// Hasil gradient clipping GPU-native — tanpa per-tensor CPU readback
pub struct GpuGradClipResult {
    /// Apakah gradient di-clip
    pub was_clipped: bool,
    /// Scale factor yang diaplikasikan (1.0 jika tidak di-clip)
    pub scale_factor: f32,
    /// Global L2 norm (sqrt dari sum of squares)
    pub norm: f32,
}

impl GpuContext {
    /// All-GPU gradient clipping.
    ///
    /// Menggabungkan L2 norm computation + scaling seluruhnya di GPU.
    /// CPU hanya membaca 4 f32 values untuk result struct — bukan N scalars
    /// per gradient tensor seperti implementasi sebelumnya.
    ///
    /// Strategy:
    /// 1. Compute per-tensor L2 norms on GPU (existing, batched)
    /// 2. Sum all norms on GPU via elementwise add
    /// 3. Sqrt on GPU to get global norm
    /// 4. Dispatch gradient_clip shader: compare norm vs max_norm, scale in-place
    /// 5. Read back 4 f32 values [norm, max_norm, scale, clipped]
    ///
    /// Tanpa CPU readback untuk intermediate scalar norms.
    pub fn clip_gradients_gpu(
        &self,
        grad_tensors: &[&GpuTensor],
        max_norm: f32,
    ) -> Result<GpuGradClipResult, GpuError> {
        if grad_tensors.is_empty() || max_norm <= 0.0 {
            return Ok(GpuGradClipResult {
                was_clipped: false,
                scale_factor: 1.0,
                norm: 0.0,
            });
        }

        // 1. Compute per-tensor L2 norm squared on GPU (batched)
        let mut norm_sq_tensors = Vec::with_capacity(grad_tensors.len());
        for g in grad_tensors {
            let norm_sq = self.l2_norm(g)?;
            norm_sq_tensors.push(norm_sq);
        }

        // 2. Sum all norm_sq on GPU → single scalar
        let total_norm_sq = self.reduce_sum_batch(&norm_sq_tensors)?;

        // 3. Sqrt on GPU → global norm (as single-element tensor, kept on GPU)
        let norm_gpu = self.sqrt(&total_norm_sq)?;

        // 4. Dispatch gradient_clip shader for each gradient tensor.
        //    Shader reads norm_sq, computes sqrt, compares with max_norm,
        //    scales gradient in-place, writes [norm, max_norm, scale, clipped].
        //    All dispatches write the same output values — no race concern.
        let pipeline = self
            .pipelines
            .get("gradient_clip")
            .ok_or_else(|| GpuError::Pipeline("gradient_clip not compiled".into()))?;

        let output_buf = self.alloc_buffer(
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let cfg_buf = self.alloc_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let cfg: [u32; 4] = [0, f32::to_bits(max_norm), 0, 0];
        self.queue.write_buffer(&cfg_buf.buffer, 0, bytemuck::cast_slice(&cfg));

        for (i, g) in grad_tensors.iter().enumerate() {
            let numel = g.numel() as u32;
            let bg = self.get_or_create_bind_group_shared(
                &pipeline.bind_group_layout,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: g.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: norm_gpu.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cfg_buf.buffer.as_entire_binding(),
                    },
                ],
                "gradient_clip_bg",
            );
            let wg_count = (numel + 255) / 256;
            self.dispatch(pipeline, &bg, (wg_count, 1, 1));
        }

        // 5. Read back only 16 bytes (4 f32 values) — no per-tensor readback
        self.flush();
        self.sync();

        let staging = self.alloc_buffer(
            16,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        self.with_encoder(|enc| {
            enc.copy_buffer_to_buffer(&output_buf.buffer, 0, &staging.buffer, 0, 16);
        });
        self.flush();

        let slice = staging.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        });
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(|_| GpuError::Timeout("gradient clip readback timed out after 30s".into()))?
            .map_err(|e| GpuError::Device(format!("map_async: {e:?}")))?;

        let mapped = slice.get_mapped_range();
        let data: &[f32] = bytemuck::cast_slice(&*mapped);
        let norm_val = data[0];
        let scale_factor = data[2];
        let clipped = data[3] as u32 != 0;
        staging.buffer.unmap();

        Ok(GpuGradClipResult {
            was_clipped: clipped,
            scale_factor,
            norm: norm_val,
        })
    }

    /// Sum multiple GPU tensors element-wise: result = sum(tensors)
    ///
    /// Semua tensor harus memiliki shape yang sama.
    /// GPU-native — tidak ada CPU readback intermediate.
    pub fn reduce_sum_batch(&self, tensors: &[GpuTensor]) -> Result<GpuTensor, GpuError> {
        if tensors.is_empty() {
            return Err(GpuError::Unsupported("reduce_sum_batch: empty batch".into()));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }
        let mut result = tensors[0].clone();
        for t in &tensors[1..] {
            result = self.add(&result, t)?;
        }
        Ok(result)
    }
}

/// Gradient clipping with CPU readback of N scalars (legacy).
///
/// Optimizations vs naive per-tensor CPU readback:
/// 1. All L2 norm dispatches batched in reusable encoder (single submit)
/// 2. Only N scalars (one per gradient tensor) downloaded to CPU
/// 3. All scaling dispatches batched in reusable encoder (single submit)
///
/// Prefer `clip_gradients_gpu` for new code.
pub fn clip_gradients_batched(
    ctx: &GpuContext,
    grads: &[&GpuTensor],
    max_norm: f32,
) -> Result<(), GpuError> {
    if grads.is_empty() || max_norm <= 0.0 {
        return Ok(());
    }

    // Step 1: Dispatch all L2 norm computations (accumulate in reusable encoder)
    let mut norm_tensors = Vec::with_capacity(grads.len());
    for g in grads {
        let norm_t = ctx.l2_norm(g)?;
        norm_tensors.push(norm_t);
    }

    // Step 2: Flush to ensure all L2 norms are computed on GPU
    ctx.flush();
    ctx.sync();

    // Step 3: Download N scalar norms
    let mut total_sq = 0.0f32;
    for norm_t in &norm_tensors {
        let cpu = norm_t.to_cpu()?;
        let sum_sq = cpu.iter().copied().next().unwrap_or(0.0);
        total_sq += sum_sq;
    }

    // Step 4: Compute scale factor on CPU
    let total_norm = total_sq.sqrt();
    if total_norm <= max_norm || total_norm <= 0.0 {
        return Ok(());
    }
    let scale = max_norm / total_norm;

    // Step 5: Batch-scale all gradients
    for g in grads {
        ctx.scale_inplace(g, scale)?;
    }

    Ok(())
}
