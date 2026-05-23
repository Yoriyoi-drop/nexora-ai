use ndarray::Array2;
use rand::Rng;

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::{GpuContext, GpuTensor};

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
    pub(crate) gpu_weights: parking_lot::Mutex<Option<SwigluGpuWeights>>,
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
            gpu_weights: parking_lot::Mutex::new(None),
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
            gpu_weights: parking_lot::Mutex::new(None),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        #[cfg(feature = "gpu")]
        {
            use ndarray::ArrayD;
            if GpuContext::global().is_ok() {
                let x_flat: Vec<f32> = x.iter().copied().collect();
                if let Ok(x_cpu) = ArrayD::from_shape_vec(vec![x.shape()[0], x.shape()[1]], x_flat) {
                    if let Ok(x_gpu) = GpuTensor::from_cpu(&x_cpu) {
                        if let Ok(result) = self.forward_gpu(&x_gpu) {
                            let cpu = result.to_cpu();
                            if let Ok(arr2) = cpu.into_dimensionality::<ndarray::Ix2>() {
                                return arr2.to_owned();
                            }
                        }
                    }
                }
            }
        }

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
        let mut guard = self.gpu_weights.lock();
        if guard.is_some() {
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
        *guard = Some(SwigluGpuWeights {
            w1_t: ctx.transpose(&w1)?,
            w2_t: ctx.transpose(&w2)?,
            w3_t: ctx.transpose(&w3)?,
        });
        Ok(())
    }

    /// GPU forward: SwiGLU FFN using GPU matmul + silu + mul (cached weights).
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        let mut guard = self.gpu_weights.lock();
        let cached = guard.get_or_insert_with(|| {
            let mk = |arr: &Array2<f32>| -> GpuTensor {
                let shape = vec![arr.shape()[0], arr.shape()[1]];
                let data = arr.as_slice().unwrap().to_vec();
                GpuTensor::from_cpu(&ndarray::ArrayD::from_shape_vec(shape, data).unwrap()).unwrap()
            };
            let w1 = mk(&self.w1);
            let w2 = mk(&self.w2);
            let w3 = mk(&self.w3);
            SwigluGpuWeights {
                w1_t: ctx.transpose(&w1).unwrap(),
                w2_t: ctx.transpose(&w2).unwrap(),
                w3_t: ctx.transpose(&w3).unwrap(),
            }
        });

        let gate = ctx.matmul(x, &cached.w1_t)?;
        let hidden = ctx.matmul(x, &cached.w3_t)?;
        let silu_gate = ctx.silu(&gate)?;
        let gated = ctx.mul(&silu_gate, &hidden)?;
        ctx.matmul(&gated, &cached.w2_t)
    }
}
