//! GPU operations bridge for HLDA-VT
//!
//! Converts between CPU `nexora_atqs::Tensor` and GPU `nexora_autograd::GpuTensor`,
//! runs compute on GPU via `GpuContext`, and reads results back.
//! When GPU is unavailable (`gpu` feature off), all functions return Err.

use crate::types::*;
use nexora_atqs::Tensor;
use std::sync::OnceLock;

// ── Feature-gated GPU types ────────────────────────────────────────

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::{GpuContext, GpuTensor};

/// Cached GPU copy of a weight tensor
pub struct GpuWeight {
    #[cfg(feature = "gpu")]
    tensor: OnceLock<GpuTensor>,
    #[cfg(not(feature = "gpu"))]
    _marker: std::marker::PhantomData<()>,
}

impl GpuWeight {
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "gpu")]
            tensor: OnceLock::new(),
            #[cfg(not(feature = "gpu"))]
            _marker: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn get_or_init(&self, cpu: &Tensor) -> HLDVAResult<&GpuTensor> {
        if let Some(t) = self.tensor.get() {
            return Ok(t);
        }
        let gpu = tensor_to_gpu(cpu)?;
        let _ = self.tensor.set(gpu);
        self.tensor.get().ok_or_else(|| HLDVAError::Device("GpuWeight init race".into()))
    }

    #[cfg(not(feature = "gpu"))]
    pub fn get_or_init(&self, _cpu: &Tensor) -> HLDVAResult<&()> {
        Err(HLDVAError::Device("GPU feature disabled".into()))
    }

    #[cfg(feature = "gpu")]
    pub fn is_cached(&self) -> bool {
        self.tensor.get().is_some()
    }

    #[cfg(not(feature = "gpu"))]
    pub fn is_cached(&self) -> bool {
        false
    }
}

// ── Context management ────────────────────────────────────────────

/// Initialize GPU context
pub fn init_gpu() -> HLDVAResult<()> {
    #[cfg(feature = "gpu")]
    {
        GpuContext::init().map_err(|e| HLDVAError::Device(format!("GPU init: {}", e)))?;
        tracing::info!("GPU context initialized for HLDA-VT");
    }
    #[cfg(not(feature = "gpu"))]
    tracing::warn!("GPU feature disabled, using CPU only");
    Ok(())
}

/// Check if GPU is available at runtime
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GpuContext::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

// ── Tensor conversion (gpu feature only) ───────────────────────────

#[cfg(feature = "gpu")]
fn ctx() -> HLDVAResult<&'static GpuContext> {
    GpuContext::global().map_err(|e| HLDVAError::Device(format!("GPU not available: {}", e)))
}

#[cfg(feature = "gpu")]
pub fn tensor_to_gpu(t: &Tensor) -> HLDVAResult<GpuTensor> {
    GpuTensor::from_slice(t.shape().to_vec(), t.data())
        .map_err(|e| HLDVAError::Tensor(format!("cpu→gpu: {}", e)))
}

#[cfg(feature = "gpu")]
pub fn gpu_to_tensor(g: GpuTensor) -> HLDVAResult<Tensor> {
    let arr = g.to_cpu().map_err(|e| HLDVAError::Tensor(format!("gpu→cpu: {}", e)))?;
    let shape = g.shape();
    let data: Vec<f32> = arr.into_raw_vec();
    Ok(Tensor::new(data, shape))
}

#[cfg(feature = "gpu")]
pub fn gpu_zeros(shape: &[usize]) -> HLDVAResult<GpuTensor> {
    GpuTensor::zeros(shape)
        .map_err(|e| HLDVAError::Tensor(format!("gpu zeros: {}", e)))
}

#[cfg(feature = "gpu")]
pub fn gpu_ones(shape: &[usize]) -> HLDVAResult<GpuTensor> {
    GpuTensor::ones(shape)
        .map_err(|e| HLDVAError::Tensor(format!("gpu ones: {}", e)))
}

// ── GPU ops — always present, return Err when feature off ──────────

fn gpu_err() -> HLDVAResult<Tensor> {
    Err(HLDVAError::Device("GPU feature disabled or GPU unavailable".into()))
}




// Elementwise
#[cfg(feature = "gpu")]
fn gpu_binary_op(a: &Tensor, b: &Tensor, op: fn(&GpuContext, &GpuTensor, &GpuTensor) -> Result<GpuTensor, nexora_autograd::gpu::GpuError>) -> HLDVAResult<Tensor> {
    let c = ctx()?;
    let a_g = tensor_to_gpu(a)?;
    let b_g = tensor_to_gpu(b)?;
    let r = op(c, &a_g, &b_g).map_err(|e| HLDVAError::Tensor(format!("gpu op: {}", e)))?;
    gpu_to_tensor(r)
}

#[cfg(feature = "gpu")]
fn gpu_unary_op(t: &Tensor, op: fn(&GpuContext, &GpuTensor) -> Result<GpuTensor, nexora_autograd::gpu::GpuError>) -> HLDVAResult<Tensor> {
    let c = ctx()?;
    let t_g = tensor_to_gpu(t)?;
    let r = op(c, &t_g).map_err(|e| HLDVAError::Tensor(format!("gpu op: {}", e)))?;
    gpu_to_tensor(r)
}

// ── Public API ─────────────────────────────────────────────────────

pub fn gpu_add(a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_binary_op(a, b, GpuContext::add) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_sub(a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_binary_op(a, b, GpuContext::sub) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_mul(a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_binary_op(a, b, GpuContext::mul) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_div(a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_binary_op(a, b, GpuContext::div) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_scale(t: &Tensor, factor: f32) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let t_g = tensor_to_gpu(t)?;
        let f_g = GpuTensor::from_slice(vec![1], &[factor]).map_err(|e| HLDVAError::Tensor(format!("scale const: {}", e)))?;
        let r = c.mul(&t_g, &f_g).map_err(|e| HLDVAError::Tensor(format!("scale: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_add_scalar(t: &Tensor, scalar: f32) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let t_g = tensor_to_gpu(t)?;
        let s_g = GpuTensor::from_slice(vec![1], &[scalar]).map_err(|e| HLDVAError::Tensor(format!("add_scalar const: {}", e)))?;
        let r = c.add(&t_g, &s_g).map_err(|e| HLDVAError::Tensor(format!("add_scalar: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_gelu(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::gelu) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_relu(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::relu) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_sigmoid(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::sigmoid) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_tanh(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::tanh) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_exp(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::exp) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_sqrt(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::sqrt) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_neg(t: &Tensor) -> HLDVAResult<Tensor> {
    gpu_scale(t, -1.0)
}

pub fn gpu_matmul(a: &Tensor, b: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let a_g = tensor_to_gpu(a)?;
        let b_g = tensor_to_gpu(b)?;
        let r = c.matmul(&a_g, &b_g).map_err(|e| HLDVAError::Tensor(format!("matmul: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_transpose(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let t_g = tensor_to_gpu(t)?;
        let r = c.transpose(&t_g).map_err(|e| HLDVAError::Tensor(format!("transpose: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_softmax(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] { gpu_unary_op(t, GpuContext::softmax) }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let x_g = tensor_to_gpu(x)?;
        let w_g = tensor_to_gpu(weight)?;
        let b_g = tensor_to_gpu(bias)?;
        let r = c.layer_norm(&x_g, &w_g, &b_g, eps).map_err(|e| HLDVAError::Tensor(format!("layer_norm: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let x_g = tensor_to_gpu(x)?;
        let w_g = tensor_to_gpu(weight)?;
        let r = c.rms_norm(&x_g, &w_g, eps).map_err(|e| HLDVAError::Tensor(format!("rms_norm: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_fused_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let q_g = tensor_to_gpu(q)?;
        let k_g = tensor_to_gpu(k)?;
        let v_g = tensor_to_gpu(v)?;
        let r = c.fused_attention(&q_g, &k_g, &v_g, scale, causal).map_err(|e| HLDVAError::Tensor(format!("fused_attention: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_conv2d(input: &Tensor, weight: &Tensor, bias: &Tensor, stride: usize, padding: usize) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        if input_shape.len() < 3 || weight_shape.len() < 4 {
            return Err(HLDVAError::Model("Invalid conv2d input/weight shape".into()));
        }
        let (in_h, in_w, in_c) = (input_shape[0], input_shape[1], input_shape[2]);
        let (out_c, _w_in_c, k_h, k_w) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);
        let out_h = (in_h + 2 * padding).saturating_sub(k_h) / stride + 1;
        let out_w = (in_w + 2 * padding).saturating_sub(k_w) / stride + 1;

        // im2col on CPU
        let mut col = Vec::with_capacity(out_h * out_w * in_c * k_h * k_w);
        let inp = input.data();
        for oy in 0..out_h {
            for ox in 0..out_w {
                for ic in 0..in_c {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            if iy >= padding && iy < in_h + padding && ix >= padding && ix < in_w + padding {
                                let iy_c = iy - padding;
                                let ix_c = ix - padding;
                                let idx = (iy_c * in_w + ix_c) * in_c + ic;
                                col.push(if idx < inp.len() { inp[idx] } else { 0.0 });
                            } else {
                                col.push(0.0);
                            }
                        }
                    }
                }
            }
        }

        let col_t = Tensor::new(col, vec![out_h * out_w, in_c * k_h * k_w]);
        let w_flat_data = weight.data();
        let w_flat = Tensor::new(w_flat_data.to_vec(), vec![out_c, in_c * k_h * k_w]);

        let w_t = gpu_transpose(&w_flat)?;
        let result = gpu_matmul(&col_t, &w_t)?;

        let mut result_data = result.data().to_vec();
        let bias_data = bias.data();
        for i in 0..result_data.len() {
            let c_idx = i % out_c;
            if c_idx < bias_data.len() {
                result_data[i] += bias_data[c_idx];
            }
        }

        Ok(Tensor::new(result_data, vec![out_h, out_w, out_c]))
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_l2_normalize(t: &Tensor) -> HLDVAResult<Tensor> {
    #[cfg(feature = "gpu")] {
        let c = ctx()?;
        let t_g = tensor_to_gpu(t)?;
        let norm = c.l2_norm(&t_g).map_err(|e| HLDVAError::Tensor(format!("l2_norm: {}", e)))?;
        let norm_val = norm.to_cpu_first_element().map_err(|e| HLDVAError::Tensor(format!("l2_norm readback: {}", e)))?;
        if norm_val == 0.0 { return Ok(t.clone()); }
        let inv = GpuTensor::from_slice(vec![1], &[1.0 / norm_val])
            .map_err(|e| HLDVAError::Tensor(format!("l2_norm inv: {}", e)))?;
        let r = c.mul(&t_g, &inv).map_err(|e| HLDVAError::Tensor(format!("l2_norm mul: {}", e)))?;
        gpu_to_tensor(r)
    }
    #[cfg(not(feature = "gpu"))] { gpu_err() }
}

pub fn gpu_cosine_similarity(a: &Tensor, b: &Tensor) -> HLDVAResult<f32> {
    let a_norm = gpu_l2_normalize(a)?;
    let b_norm = gpu_l2_normalize(b)?;
    let dot = gpu_matmul(
        &Tensor::new(a_norm.data().to_vec(), vec![1, a_norm.data().len()]),
        &Tensor::new(b_norm.data().to_vec(), vec![b_norm.data().len(), 1]),
    )?;
    Ok(dot.data()[0])
}

pub fn gpu_sum(t: &Tensor) -> HLDVAResult<f32> {
    #[cfg(feature = "gpu")] {
        let _c = ctx()?;
        let t_g = tensor_to_gpu(t)?;
        t_g.sum_gpu().map_err(|e| HLDVAError::Tensor(format!("sum: {}", e)))
    }
    #[cfg(not(feature = "gpu"))] {
        Err(HLDVAError::Device("GPU feature disabled".into()))
    }
}
