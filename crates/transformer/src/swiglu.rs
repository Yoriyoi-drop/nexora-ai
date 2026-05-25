use ndarray::Array2;
use rand::Rng;
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct SwigluGpuWeights {
    pub w1_t: nexora_autograd::gpu::GpuTensor,
    pub w2_t: nexora_autograd::gpu::GpuTensor,
    pub w3_t: nexora_autograd::gpu::GpuTensor,
}

#[derive(Debug)]
pub struct SwiGLU {
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<SwigluGpuWeights>,
}

#[cfg(not(feature = "gpu"))]
impl Clone for SwiGLU {
    fn clone(&self) -> Self {
        Self {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            w3: self.w3.clone(),
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for SwiGLU {
    fn clone(&self) -> Self {
        Self {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            w3: self.w3.clone(),
            gpu_weights: OnceLock::new(),
        }
    }
}

impl SwiGLU {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (hidden_size as f32).sqrt().recip();

        let w1 = Array2::from_shape_fn((intermediate_size, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let w2 = Array2::from_shape_fn((hidden_size, intermediate_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let w3 = Array2::from_shape_fn((intermediate_size, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });

        Self {
            w1,
            w2,
            w3,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Pure CPU forward path.
        // GPU-resident execution uses `forward_gpu` (no per-layer readback).
        let gate = x.dot(&self.w1.t());
        let hidden = x.dot(&self.w3.t());

        let mut gated = Array2::zeros(gate.dim());

        ndarray::Zip::from(gated.view_mut())
            .and(gate.view())
            .and(hidden.view())
            .for_each(|g, &gv, &hv| {
                let sigmoid = 1.0 / (1.0 + (-gv).exp());
                *g = gv * sigmoid * hv;
            });

        gated.dot(&self.w2.t())
    }
}


#[cfg(feature = "gpu")]
impl SwiGLU {
    /// Pre-upload weights to GPU — call before inference to avoid first-pass latency.
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let mk = |arr: &ndarray::Array2<f32>| -> Result<GpuTensor, nexora_autograd::gpu::GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr.as_slice().ok_or_else(|| {
                nexora_autograd::gpu::GpuError::Unsupported("non-contiguous".into())
            })?.to_vec();
            let cpu_arr = ndarray::ArrayD::from_shape_vec(shape, data)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
            GpuTensor::from_cpu(&cpu_arr)
        };
        let w1 = mk(&self.w1)?;
        let w2 = mk(&self.w2)?;
        let w3 = mk(&self.w3)?;
        self.gpu_weights.set(SwigluGpuWeights {
            w1_t: ctx.transpose(&w1)?,
            w2_t: ctx.transpose(&w2)?,
            w3_t: ctx.transpose(&w3)?,
        }).map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("already set".into()))?;
        Ok(())
    }

    /// GPU forward: SwiGLU FFN using GPU matmul + silu + mul (cached weights).
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        if self.gpu_weights.get().is_none() {
            let mk = |arr: &Array2<f32>| -> Result<GpuTensor, nexora_autograd::gpu::GpuError> {
                let shape = vec![arr.shape()[0], arr.shape()[1]];
                let data = arr.as_slice().ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("non-contiguous weight slice".into())
                })?.to_vec();
                let cpu_arr = ndarray::ArrayD::from_shape_vec(shape, data)
                    .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(format!("shape mismatch for weight: {}", e)))?;
                GpuTensor::from_cpu(&cpu_arr)
            };
            let w1 = mk(&self.w1)?;
            let w2 = mk(&self.w2)?;
            let w3 = mk(&self.w3)?;
            let _ = self.gpu_weights.set(SwigluGpuWeights {
                w1_t: ctx.transpose(&w1)?,
                w2_t: ctx.transpose(&w2)?,
                w3_t: ctx.transpose(&w3)?,
            });
        }
        let cached = self.gpu_weights.get()
            .ok_or_else(|| nexora_autograd::gpu::GpuError::Unsupported("SwiGLU weights not initialized".into()))?;

        let gate = ctx.matmul(x, &cached.w1_t)?;
        let hidden = ctx.matmul(x, &cached.w3_t)?;
        let silu_gate = ctx.silu(&gate)?;
        let gated = ctx.mul(&silu_gate, &hidden)?;
        ctx.matmul(&gated, &cached.w2_t)
    }
}
