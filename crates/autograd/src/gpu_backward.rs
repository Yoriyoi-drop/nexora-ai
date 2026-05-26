use ndarray::{ArrayD, IxDyn};

use crate::gpu::{ElemOp, GpuContext, GpuError, GpuTensor, ReduceOp};

// ── Helpers ────────────────────────────────────────────────────────

fn neg(ctx: &GpuContext, t: &GpuTensor) -> Result<GpuTensor, GpuError> {
    ctx.elementwise_unary(t, ElemOp::Neg)
}

/// Reduce all dimensions except the last via matmul with ones.
/// Pure GPU: no CPU readback.
/// Covers the bias-backward pattern where grad has shape [..., D]
/// and target shape is [D].
fn reduce_all_but_last(ctx: &GpuContext, tensor: &GpuTensor) -> Result<GpuTensor, GpuError> {
    let shape = tensor.shape();
    if shape.len() <= 1 {
        return Ok(tensor.clone());
    }
    let last_dim = shape[shape.len() - 1];
    let rest: usize = shape[..shape.len() - 1].iter().product();

    let flat = tensor.reshape(vec![rest, last_dim])?;
    let ones = GpuTensor::from_cpu(&ArrayD::from_elem(IxDyn(&[1, rest]), 1.0f32))
        .map_err(|e| GpuError::Conversion(format!("reduce_all_but_last ones: {e}")))?;
    let result = ctx.matmul(&ones, &flat)?;
    result.reshape(vec![last_dim])
}

/// Reduce a GPU tensor to a target shape (for broadcast backward).
/// Pure GPU for the common cases; CPU readback as fallback.
pub fn reduce_to_shape(
    ctx: &GpuContext,
    tensor: &GpuTensor,
    target_shape: &[usize],
) -> Result<GpuTensor, GpuError> {
    if tensor.shape() == target_shape {
        return Ok(tensor.clone());
    }

    let t_shape = tensor.shape();

    // Target is scalar [1] or []: full sum reduction (GPU-native)
    if target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == 1) {
        return ctx.reduce(tensor, ReduceOp::Sum);
    }

    // Common pattern: bias backward — target is [D], tensor is [..., D]
    if target_shape.len() == 1
        && t_shape.len() > 1
        && target_shape[0] == t_shape[t_shape.len() - 1]
    {
        return reduce_all_but_last(ctx, tensor);
    }

    // CPU fallback for complex broadcast patterns
    let cpu_data = tensor.to_cpu()?;
    let reduced = crate::broadcast::reduce_grad_for_shape(&cpu_data, target_shape);
    GpuTensor::from_cpu(&reduced)
}

// ── MatMul backward ────────────────────────────────────────────────
// da = grad @ b^T,  db = a^T @ grad
pub fn matmul_backward(
    ctx: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
    grad: &GpuTensor,
) -> Result<(GpuTensor, GpuTensor), GpuError> {
    let b_t = ctx.transpose(b)?;
    let da = ctx.matmul(grad, &b_t)?;
    let a_t = ctx.transpose(a)?;
    let db = ctx.matmul(&a_t, grad)?;
    Ok((da, db))
}

// ── Add backward ───────────────────────────────────────────────────
// da = grad, db = grad (gathered to original shape)
pub fn add_backward(
    ctx: &GpuContext,
    a_shape: &[usize],
    b_shape: &[usize],
    grad: &GpuTensor,
) -> Result<(GpuTensor, GpuTensor), GpuError> {
    let da = if a_shape == grad.shape() {
        grad.clone()
    } else {
        reduce_to_shape(ctx, grad, a_shape)?
    };
    let db = if b_shape == grad.shape() {
        grad.clone()
    } else {
        reduce_to_shape(ctx, grad, b_shape)?
    };
    Ok((da, db))
}

// ── Sub backward ───────────────────────────────────────────────────
// da = grad, db = -grad
pub fn sub_backward(
    ctx: &GpuContext,
    a_shape: &[usize],
    b_shape: &[usize],
    grad: &GpuTensor,
) -> Result<(GpuTensor, GpuTensor), GpuError> {
    let (da, db_pos) = add_backward(ctx, a_shape, b_shape, grad)?;
    let db = neg(ctx, &db_pos)?;
    Ok((da, db))
}

// ── Mul backward ───────────────────────────────────────────────────
// da = grad * b,  db = grad * a
pub fn mul_backward(
    ctx: &GpuContext,
    a: &GpuTensor,
    b: &GpuTensor,
    grad: &GpuTensor,
) -> Result<(GpuTensor, GpuTensor), GpuError> {
    let da = ctx.mul(grad, b)?;
    let db = ctx.mul(grad, a)?;
    Ok((da, db))
}

// ── Div backward ───────────────────────────────────────────────────
// da = grad / b,  db = -grad * result / b
pub fn div_backward(
    ctx: &GpuContext,
    b: &GpuTensor,
    result: &GpuTensor,
    grad: &GpuTensor,
) -> Result<(GpuTensor, GpuTensor), GpuError> {
    let da = ctx.div(grad, b)?;
    let neg_grad = neg(ctx, grad)?;
    let db_numer = ctx.mul(&neg_grad, result)?;
    let db = ctx.div(&db_numer, b)?;
    Ok((da, db))
}

// ── Exp backward ───────────────────────────────────────────────────
// da = grad * exp(x) = grad * result
pub fn exp_backward(
    ctx: &GpuContext,
    result: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let da = ctx.mul(grad, result)?;
    Ok(da)
}

// ── Ln backward ────────────────────────────────────────────────────
// da = grad / x
pub fn ln_backward(
    ctx: &GpuContext,
    input: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let da = ctx.div(grad, input)?;
    Ok(da)
}

// ── Powf backward ──────────────────────────────────────────────────
// da = grad * exponent * x^(exponent-1) = grad * exponent * result / x
pub fn powf_backward(
    ctx: &GpuContext,
    input: &GpuTensor,
    result: &GpuTensor,
    exponent: f32,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let x_pow_exp_m1 = ctx.div(result, input)?;
    let exp_t = GpuTensor::from_cpu(&ArrayD::from_elem(IxDyn(&[]), exponent))
        .map_err(|e| GpuError::Conversion(format!("powf_backward exp: {e}")))?;
    let scaled = ctx.mul(grad, &exp_t)?;
    let da = ctx.mul(&scaled, &x_pow_exp_m1)?;
    Ok(da)
}

// ── Sqrt backward ──────────────────────────────────────────────────
// da = grad / (2 * result)
pub fn sqrt_backward(
    ctx: &GpuContext,
    result: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let two = GpuTensor::from_cpu(&ArrayD::from_elem(result.shape(), 2.0f32))
        .map_err(|e| GpuError::Conversion(format!("sqrt_backward two: {e}")))?;
    let denom = ctx.mul(&two, result)?;
    let da = ctx.div(grad, &denom)?;
    Ok(da)
}

// ── Neg backward ───────────────────────────────────────────────────
// da = -grad
pub fn neg_backward(
    ctx: &GpuContext,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    neg(ctx, grad)
}

// ── Relu backward ──────────────────────────────────────────────────
// da = grad * (x > 0)  — pure GPU comparison, no floating-point epsilon
pub fn relu_backward(
    ctx: &GpuContext,
    input: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let mask = ctx.elementwise_unary(input, ElemOp::Step)?;
    let da = ctx.mul(grad, &mask)?;
    Ok(da)
}

// ── Gelu backward ──────────────────────────────────────────────────
// da = grad * gelu'(x) computed entirely with GPU elementwise ops
pub fn gelu_backward(
    ctx: &GpuContext,
    input: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let shape: &[usize] = &input.shape();
    let sqrt_2pi = (2.0 / std::f64::consts::PI) as f32;

    let x_sq = ctx.mul(input, input)?;
    let x_cu = ctx.mul(&x_sq, input)?;

    let p044 = GpuTensor::from_cpu(&ArrayD::from_elem(shape, 0.044715f32))
        .map_err(|e| GpuError::Conversion(format!("gelu_backward const: {e}")))?;
    let x3_scaled = ctx.mul(&x_cu, &p044)?;
    let tanh_inner = ctx.add(input, &x3_scaled)?;

    let sqrt2pi_t = GpuTensor::from_cpu(&ArrayD::from_elem(shape, sqrt_2pi))
        .map_err(|e| GpuError::Conversion(format!("gelu_backward sqrt2pi: {e}")))?;
    let tanh_arg = ctx.mul(&sqrt2pi_t, &tanh_inner)?;

    let t = ctx.elementwise_unary(&tanh_arg, ElemOp::Tanh)
        .map_err(|e| GpuError::Device(format!("gelu_backward tanh: {e}")))?;

    let t_sq = ctx.mul(&t, &t)?;
    let one = GpuTensor::from_cpu(&ArrayD::from_elem(shape, 1.0f32))
        .map_err(|e| GpuError::Conversion(format!("gelu_backward one: {e}")))?;
    let sech2 = ctx.sub(&one, &t_sq)?;

    let one_plus_t = ctx.add(&one, &t)?;
    let half = GpuTensor::from_cpu(&ArrayD::from_elem(shape, 0.5f32))
        .map_err(|e| GpuError::Conversion(format!("gelu_backward half: {e}")))?;
    let term1 = ctx.mul(&half, &one_plus_t)?;

    let p134 = GpuTensor::from_cpu(&ArrayD::from_elem(shape, 0.134145f32))
        .map_err(|e| GpuError::Conversion(format!("gelu_backward p134: {e}")))?;
    let px_sq = ctx.mul(&p134, &x_sq)?;
    let one_px_sq = ctx.add(&one, &px_sq)?;

    let term2a = ctx.mul(&half, input)?;
    let term2b = ctx.mul(&term2a, &sech2)?;
    let term2c = ctx.mul(&term2b, &sqrt2pi_t)?;
    let term2 = ctx.mul(&term2c, &one_px_sq)?;

    let gelu_grad = ctx.add(&term1, &term2)?;
    let da = ctx.mul(grad, &gelu_grad)?;
    Ok(da)
}

// ── Tanh backward ──────────────────────────────────────────────────
// da = grad * (1 - result^2)
pub fn tanh_backward(
    ctx: &GpuContext,
    result: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let r_sq = ctx.mul(result, result)?;
    let one = GpuTensor::from_cpu(&ArrayD::from_elem(result.shape(), 1.0f32))
        .map_err(|e| GpuError::Conversion(format!("tanh_backward one: {e}")))?;
    let one_minus_r2 = ctx.sub(&one, &r_sq)?;
    let da = ctx.mul(grad, &one_minus_r2)?;
    Ok(da)
}

// ── Sigmoid backward ───────────────────────────────────────────────
// da = grad * result * (1 - result)
pub fn sigmoid_backward(
    ctx: &GpuContext,
    result: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let one = GpuTensor::from_cpu(&ArrayD::from_elem(result.shape(), 1.0f32))
        .map_err(|e| GpuError::Conversion(format!("sigmoid_backward one: {e}")))?;
    let one_minus_r = ctx.sub(&one, result)?;
    let r_times_1mr = ctx.mul(result, &one_minus_r)?;
    let da = ctx.mul(grad, &r_times_1mr)?;
    Ok(da)
}

// ── Silu backward ──────────────────────────────────────────────────
// da = grad * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// = grad * sigmoid(x) * (1 + x * sigmoid(x) * (1 - sigmoid(x)) / sigmoid(x))
//
// Using identity: silu(x) = x * sigmoid(x)
// silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
pub fn silu_backward(
    ctx: &GpuContext,
    input: &GpuTensor,
    _result: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let sig = ctx.elementwise_unary(input, ElemOp::Sigmoid)
        .map_err(|e| GpuError::Device(format!("silu_backward sigmoid: {e}")))?;
    let one = GpuTensor::from_cpu(&ArrayD::from_elem(input.shape(), 1.0f32))
        .map_err(|e| GpuError::Conversion(format!("silu_backward one: {e}")))?;
    let one_minus_sig = ctx.sub(&one, &sig)?;
    let x_sig = ctx.mul(input, &sig)?;
    let term2 = ctx.mul(&x_sig, &one_minus_sig)?;
    let local_grad = ctx.add(&sig, &term2)?;
    let da = ctx.mul(grad, &local_grad)?;
    Ok(da)
}

// ── Sum backward ───────────────────────────────────────────────────
// da = grad (broadcast to original shape)
pub fn sum_backward(
    _ctx: &GpuContext,
    orig_shape: &[usize],
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let grad_val = grad.to_cpu_first_element()
        .map_err(|e| GpuError::Device(format!("sum_backward grad readback: {e}")))?;
    let result = GpuTensor::from_cpu(
        &ArrayD::from_elem(IxDyn(orig_shape), grad_val),
    )?;
    Ok(result)
}

// ── Mean backward ──────────────────────────────────────────────────
// da = grad / n (broadcast to original shape)
pub fn mean_backward(
    _ctx: &GpuContext,
    orig_shape: &[usize],
    numel: f32,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let grad_val = grad.to_cpu_first_element()
        .map_err(|e| GpuError::Device(format!("mean_backward grad readback: {e}")))?;
    let fill_val = grad_val / numel;
    let result = GpuTensor::from_cpu(&ArrayD::from_elem(IxDyn(orig_shape), fill_val))?;
    Ok(result)
}

// ── Softmax backward ───────────────────────────────────────────────
// dx = softmax * (grad - sum(softmax * grad, dim=-1, keepdim=True))
// Pure GPU using matmul + elementwise ops
pub fn softmax_backward_last_dim(
    ctx: &GpuContext,
    softmax: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let shape = softmax.shape();
    if shape.len() < 2 {
        let da = ctx.mul(softmax, grad)?;
        return Ok(da);
    }

    let last_dim = shape[shape.len() - 1];
    let rest: usize = shape[..shape.len() - 1].iter().product();

    // Flatten to 2D: [rest, last_dim]
    let s_2d = softmax.reshape(vec![rest, last_dim])?;
    let g_2d = grad.reshape(vec![rest, last_dim])?;

    // sum_sg = (softmax * grad) summed over last dim → [rest, 1]
    let sg = ctx.mul(&s_2d, &g_2d)?;

    let ones_col = GpuTensor::from_cpu(&ArrayD::from_elem(IxDyn(&[last_dim, 1]), 1.0f32))
        .map_err(|e| GpuError::Conversion(format!("softmax_backward ones: {e}")))?;
    let sum_sg = ctx.matmul(&sg, &ones_col)?;

    // dx = s * (g - sum_sg)
    let g_minus_sum = ctx.sub(&g_2d, &sum_sg)?;
    let dx_2d = ctx.mul(&s_2d, &g_minus_sum)?;
    let dx = dx_2d.reshape(shape.to_vec())?;
    Ok(dx)
}

// ── Causal Attention backward ───────────────────────────────────────
// Pure GPU fused kernel. Eliminates 4× to_cpu() from the old CPU fallback.
// Q/K/V/dO/dQ/dK/dV: [batch, heads, seq, dim]
// Returns (dQ, dK, dV) all as GpuTensors.
pub fn attention_backward_gpu(
    ctx: &GpuContext,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    grad: &GpuTensor,
    scale: f32,
) -> Result<(GpuTensor, GpuTensor, GpuTensor), GpuError> {
    ctx.fused_attention_backward(q, k, v, grad, scale, true)
}

// ── Embedding backward ─────────────────────────────────────────────
// Pure GPU scatter-add via atomic CAS f32.
// ids: [N] u32, grad: [N, D] f32, output: d_weight [V, D] f32.
// d_input (grad w.r.t. ids) is always zero (not differentiable).
pub fn embedding_backward_gpu(
    ctx: &GpuContext,
    ids: &GpuTensor,
    grad: &GpuTensor,
    vocab_size: usize,
) -> Result<(GpuTensor, GpuTensor), GpuError> {
    let d_weight = ctx.embedding_backward(ids, grad, vocab_size)?;
    let d_input = GpuTensor::from_cpu(&ndarray::ArrayD::zeros(grad.shape()))
        .map_err(|e| GpuError::Conversion(format!("embedding d_input zeros: {e}")))?;
    Ok((d_input, d_weight))
}

// ── CrossEntropy backward ──────────────────────────────────────────
// Pure GPU: softmax → fused d_logits = grad * (softmax - one_hot(targets)).
// Eliminates 1× to_cpu() for targets and 1× from_cpu() for one-hot upload.
pub fn cross_entropy_backward_gpu(
    ctx: &GpuContext,
    logits: &GpuTensor,
    targets: &GpuTensor,
    grad: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let softmax = ctx.softmax(logits)?;
    ctx.cross_entropy_backward(&softmax, grad, targets)
}

// ── Gradient AllReduce (GPU) ───────────────────────────────────────
// Average gradients from N model replicas.
// Takes a flat buffer of [replica_0_grads | replica_1_grads | ...]
// and writes averaged gradients to `out`.
pub fn gradient_allreduce_gpu(
    ctx: &GpuContext,
    all_grads: &GpuTensor,
    out: &GpuTensor,
    num_replicas: u32,
) -> Result<(), GpuError> {
    ctx.gradient_allreduce(all_grads, out, num_replicas)
}
