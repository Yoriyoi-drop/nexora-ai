use crate::gpu::{GpuContext, GpuError, GpuTensor};

/// Fused GPU gradient clipping.
///
/// Optimizations vs naive per-tensor CPU readback:
/// 1. All L2 norm dispatches batched in reusable encoder (single submit)
/// 2. Only N scalars (one per gradient tensor) downloaded to CPU
/// 3. All scaling dispatches batched in reusable encoder (single submit)
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
        // l2_norm uses dispatch() which uses with_encoder → accumulates
        let norm_t = ctx.l2_norm(g)?;
        norm_tensors.push(norm_t);
    }

    // Step 2: Flush to ensure all L2 norms are computed on GPU
    ctx.flush();
    ctx.sync();

    // Step 3: Download N scalar norms (negligible — N*4 bytes)
    let mut total_sq = 0.0f32;
    for norm_t in &norm_tensors {
        let cpu = norm_t.to_cpu();
        let sum_sq = cpu.iter().copied().next().unwrap_or(0.0);
        total_sq += sum_sq;
    }

    // Step 4: Compute scale factor on CPU
    let total_norm = total_sq.sqrt();
    if total_norm <= max_norm || total_norm <= 0.0 {
        return Ok(());
    }
    let scale = max_norm / total_norm;

    // Step 5: Batch-scale all gradients (accumulate in reusable encoder)
    for g in grads {
        // scale_inplace uses dispatch() → accumulates
        ctx.scale_inplace(g, scale)?;
    }

    Ok(())
}
