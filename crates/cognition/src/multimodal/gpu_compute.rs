/// GPU-accelerated batch MLP forward: output = gelu(x @ W1 + b1) @ W2 + b2
/// Falls back to None if GPU unavailable
#[cfg(feature = "gpu")]
pub fn try_gpu_mlp_forward(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    batch: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Option<Vec<f32>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let x_t = GpuTensor::from_slice(vec![batch, input_dim], x).ok()?;
    let w1_t = GpuTensor::from_slice(vec![input_dim, hidden_dim], w1).ok()?;
    let b1_t = GpuTensor::from_slice(vec![hidden_dim], b1).ok()?;
    let w2_t = GpuTensor::from_slice(vec![hidden_dim, output_dim], w2).ok()?;
    let b2_t = GpuTensor::from_slice(vec![output_dim], b2).ok()?;

    let h = ctx.matmul(&x_t, &w1_t).ok()?;
    let h = ctx.add(&h, &b1_t).ok()?;
    let h = ctx.gelu(&h).ok()?;
    let o = ctx.matmul(&h, &w2_t).ok()?;
    let o = ctx.add(&o, &b2_t).ok()?;

    let arr = o.to_cpu().ok()?;
    Some(arr.into_iter().collect())
}

/// GPU MLP forward yang return GpuTensor — tanpa readback.
/// Caller bisa chain ke GPU ops berikutnya.
#[cfg(feature = "gpu")]
pub fn try_gpu_mlp_forward_keep_gpu(
    x: &nexora_deeplearning::autograd::gpu::GpuTensor,
    w1: &nexora_deeplearning::autograd::gpu::GpuTensor,
    b1: &nexora_deeplearning::autograd::gpu::GpuTensor,
    w2: &nexora_deeplearning::autograd::gpu::GpuTensor,
    b2: &nexora_deeplearning::autograd::gpu::GpuTensor,
) -> Option<nexora_deeplearning::autograd::gpu::GpuTensor> {
    use nexora_deeplearning::autograd::gpu::{GpuContext};
    let ctx = GpuContext::global().ok()?;

    let h = ctx.matmul(x, w1).ok()?;
    let h = ctx.add(&h, b1).ok()?;
    let h = ctx.gelu(&h).ok()?;
    let o = ctx.matmul(&h, w2).ok()?;
    ctx.add(&o, b2).ok()
}

/// GPU-accelerated batch MLP forward with async readback.
/// Returns `AsyncReadback` — call `.recv()` to get the result.
/// Allows caller to overlap GPU readback with other compute.
#[cfg(feature = "gpu")]
pub fn try_gpu_mlp_forward_async(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    batch: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Option<nexora_deeplearning::autograd::gpu_async::AsyncReadback<Vec<f32>>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let x_t = GpuTensor::from_slice(vec![batch, input_dim], x).ok()?;
    let w1_t = GpuTensor::from_slice(vec![input_dim, hidden_dim], w1).ok()?;
    let b1_t = GpuTensor::from_slice(vec![hidden_dim], b1).ok()?;
    let w2_t = GpuTensor::from_slice(vec![hidden_dim, output_dim], w2).ok()?;
    let b2_t = GpuTensor::from_slice(vec![output_dim], b2).ok()?;

    let h = ctx.matmul(&x_t, &w1_t).ok()?;
    let h = ctx.add(&h, &b1_t).ok()?;
    let h = ctx.gelu(&h).ok()?;
    let o = ctx.matmul(&h, &w2_t).ok()?;
    let o = ctx.add(&o, &b2_t).ok()?;

    o.to_cpu_async(&ctx).ok()
}

/// GPU-accelerated fused attention (FlashAttention-style)
/// q/k/v: [batch, seq, head_dim] flat
#[cfg(feature = "gpu")]
pub fn try_gpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Option<Vec<f32>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let q_t = GpuTensor::from_slice(vec![batch, num_heads, seq, head_dim], q).ok()?;
    let k_t = GpuTensor::from_slice(vec![batch, num_heads, seq, head_dim], k).ok()?;
    let v_t = GpuTensor::from_slice(vec![batch, num_heads, seq, head_dim], v).ok()?;

    let scale = (head_dim as f32).sqrt().recip();
    let o = ctx.fused_attention(&q_t, &k_t, &v_t, scale, causal).ok()?;

    let arr = o.to_cpu().ok()?;
    Some(arr.into_iter().collect())
}

/// GPU attention yang return GpuTensor — tanpa readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_attention_keep_gpu(
    q: &nexora_deeplearning::autograd::gpu::GpuTensor,
    k: &nexora_deeplearning::autograd::gpu::GpuTensor,
    v: &nexora_deeplearning::autograd::gpu::GpuTensor,
    causal: bool,
) -> Option<nexora_deeplearning::autograd::gpu::GpuTensor> {
    use nexora_deeplearning::autograd::gpu::GpuContext;
    let ctx = GpuContext::global().ok()?;
    let head_dim = q.shape().last().copied().unwrap_or(64);
    let scale = (head_dim as f32).sqrt().recip();
    ctx.fused_attention(q, k, v, scale, causal).ok()
}

/// GPU-accelerated fused attention with async readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_attention_async(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Option<nexora_deeplearning::autograd::gpu_async::AsyncReadback<Vec<f32>>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let q_t = GpuTensor::from_slice(vec![batch, num_heads, seq, head_dim], q).ok()?;
    let k_t = GpuTensor::from_slice(vec![batch, num_heads, seq, head_dim], k).ok()?;
    let v_t = GpuTensor::from_slice(vec![batch, num_heads, seq, head_dim], v).ok()?;

    let scale = (head_dim as f32).sqrt().recip();
    let o = ctx.fused_attention(&q_t, &k_t, &v_t, scale, causal).ok()?;

    o.to_cpu_async(&ctx).ok()
}

/// GPU-accelerated batched QKV projection: out = x @ W^T
#[cfg(feature = "gpu")]
pub fn try_gpu_matmul(
    x: &[f32],
    w: &[f32],
    batch: usize,
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<Vec<f32>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let x_t = GpuTensor::from_slice(vec![batch * seq, in_dim], x).ok()?;
    let w_t = GpuTensor::from_slice(vec![in_dim, out_dim], w).ok()?;

    let o = ctx.matmul(&x_t, &w_t).ok()?;
    let arr = o.to_cpu().ok()?;
    Some(arr.into_iter().collect())
}

/// GPU matmul yang return GpuTensor — tanpa readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_matmul_keep_gpu(
    x: &nexora_deeplearning::autograd::gpu::GpuTensor,
    w: &nexora_deeplearning::autograd::gpu::GpuTensor,
) -> Option<nexora_deeplearning::autograd::gpu::GpuTensor> {
    use nexora_deeplearning::autograd::gpu::GpuContext;
    let ctx = GpuContext::global().ok()?;
    ctx.matmul(x, w).ok()
}

/// GPU-accelerated batched matmul with async readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_matmul_async(
    x: &[f32],
    w: &[f32],
    batch: usize,
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<nexora_deeplearning::autograd::gpu_async::AsyncReadback<Vec<f32>>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let x_t = GpuTensor::from_slice(vec![batch * seq, in_dim], x).ok()?;
    let w_t = GpuTensor::from_slice(vec![in_dim, out_dim], w).ok()?;

    let o = ctx.matmul(&x_t, &w_t).ok()?;
    o.to_cpu_async(&ctx).ok()
}

/// GPU-accelerated softmax along last axis
#[cfg(feature = "gpu")]
pub fn try_gpu_softmax(scores: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let s_t = GpuTensor::from_slice(vec![rows, cols], scores).ok()?;
    let o = ctx.softmax(&s_t).ok()?;
    let arr = o.to_cpu().ok()?;
    Some(arr.into_iter().collect())
}

/// GPU-accelerated softmax with async readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_softmax_async(
    scores: &[f32],
    rows: usize,
    cols: usize,
) -> Option<nexora_deeplearning::autograd::gpu_async::AsyncReadback<Vec<f32>>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let s_t = GpuTensor::from_slice(vec![rows, cols], scores).ok()?;
    let o = ctx.softmax(&s_t).ok()?;
    o.to_cpu_async(&ctx).ok()
}

/// GPU-accelerated GELU activation
#[cfg(feature = "gpu")]
pub fn try_gpu_gelu(x: &[f32]) -> Option<Vec<f32>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let n = x.len();
    let x_t = GpuTensor::from_slice(vec![n], x).ok()?;
    let o = ctx.gelu(&x_t).ok()?;
    let arr = o.to_cpu().ok()?;
    Some(arr.into_iter().collect())
}

/// GPU-accelerated GELU with async readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_gelu_async(
    x: &[f32],
) -> Option<nexora_deeplearning::autograd::gpu_async::AsyncReadback<Vec<f32>>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let n = x.len();
    let x_t = GpuTensor::from_slice(vec![n], x).ok()?;
    let o = ctx.gelu(&x_t).ok()?;
    o.to_cpu_async(&ctx).ok()
}

/// GPU-accelerated elementwise add (broadcast)
#[cfg(feature = "gpu")]
pub fn try_gpu_add(a: &[f32], b: &[f32], shape_a: &[usize], shape_b: &[usize]) -> Option<Vec<f32>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let a_t = GpuTensor::from_slice(shape_a.to_vec(), a).ok()?;
    let b_t = GpuTensor::from_slice(shape_b.to_vec(), b).ok()?;
    let o = ctx.add(&a_t, &b_t).ok()?;
    let arr = o.to_cpu().ok()?;
    Some(arr.into_iter().collect())
}

/// GPU-accelerated elementwise add with async readback.
#[cfg(feature = "gpu")]
pub fn try_gpu_add_async(
    a: &[f32],
    b: &[f32],
    shape_a: &[usize],
    shape_b: &[usize],
) -> Option<nexora_deeplearning::autograd::gpu_async::AsyncReadback<Vec<f32>>> {
    use nexora_deeplearning::autograd::gpu::{GpuContext, GpuTensor};
    let ctx = GpuContext::global().ok()?;

    let a_t = GpuTensor::from_slice(shape_a.to_vec(), a).ok()?;
    let b_t = GpuTensor::from_slice(shape_b.to_vec(), b).ok()?;
    let o = ctx.add(&a_t, &b_t).ok()?;
    o.to_cpu_async(&ctx).ok()
}

// --- CPU fallbacks (no GPU needed) ---

#[cfg(not(feature = "gpu"))]
pub fn try_gpu_mlp_forward(
    _x: &[f32], _w1: &[f32], _b1: &[f32], _w2: &[f32], _b2: &[f32],
    _batch: usize, _input_dim: usize, _hidden_dim: usize, _output_dim: usize,
) -> Option<Vec<f32>> { None }

#[cfg(not(feature = "gpu"))]
pub fn try_gpu_attention(
    _q: &[f32], _k: &[f32], _v: &[f32],
    _batch: usize, _seq: usize, _num_heads: usize, _head_dim: usize, _causal: bool,
) -> Option<Vec<f32>> { None }

#[cfg(not(feature = "gpu"))]
pub fn try_gpu_matmul(
    _x: &[f32], _w: &[f32],
    _batch: usize, _seq: usize, _in_dim: usize, _out_dim: usize,
) -> Option<Vec<f32>> { None }

#[cfg(not(feature = "gpu"))]
pub fn try_gpu_softmax(_scores: &[f32], _rows: usize, _cols: usize) -> Option<Vec<f32>> { None }

#[cfg(not(feature = "gpu"))]
pub fn try_gpu_gelu(_x: &[f32]) -> Option<Vec<f32>> { None }

#[cfg(not(feature = "gpu"))]
pub fn try_gpu_add(
    _a: &[f32], _b: &[f32], _shape_a: &[usize], _shape_b: &[usize],
) -> Option<Vec<f32>> { None }
