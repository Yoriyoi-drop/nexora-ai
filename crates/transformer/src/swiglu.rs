use ndarray::Array2;
use rand::Rng;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct SwigluGpuWeights {
    pub w1: nexora_autograd::gpu::GpuTensor,
    pub w2: nexora_autograd::gpu::GpuTensor,
    pub w3: nexora_autograd::gpu::GpuTensor,
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
    pub(crate) gpu_weights: std::sync::Mutex<Option<SwigluGpuWeights>>,
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
            gpu_weights: std::sync::Mutex::new(None),
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
            gpu_weights: std::sync::Mutex::new(None),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let gate = x.dot(&self.w1.t());
        let hidden = x.dot(&self.w3.t());

        let (rows, cols) = gate.dim();
        let mut gated = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let v = gate[[i, j]];
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                gated[[i, j]] = v * sigmoid * hidden[[i, j]];
            }
        }

        gated.dot(&self.w2.t())
    }
}

#[allow(dead_code)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(feature = "gpu")]
impl SwiGLU {
    /// GPU forward: SwiGLU FFN using GPU matmul + silu + mul (cached weights).
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        let mut guard = self.gpu_weights.lock().unwrap();
        let cached = guard.get_or_insert_with(|| {
            let mk = |arr: &Array2<f32>| -> GpuTensor {
                let shape = vec![arr.shape()[0], arr.shape()[1]];
                let data = arr.as_slice().unwrap().to_vec();
                GpuTensor::from_cpu(&ndarray::ArrayD::from_shape_vec(shape, data).unwrap()).unwrap()
            };
            let w1 = mk(&self.w1);
            let w2 = mk(&self.w2);
            let w3 = mk(&self.w3);
            let ctx_ref = GpuContext::global().unwrap();
            SwigluGpuWeights {
                w1_t: ctx_ref.transpose(&w1).unwrap(),
                w2_t: ctx_ref.transpose(&w2).unwrap(),
                w3_t: ctx_ref.transpose(&w3).unwrap(),
                w1, w2, w3,
            }
        });

        let gate = ctx.matmul(x, &cached.w1_t)?;
        let hidden = ctx.matmul(x, &cached.w3_t)?;
        let silu_gate = ctx.silu(&gate)?;
        let gated = ctx.mul(&silu_gate, &hidden)?;
        ctx.matmul(&gated, &cached.w2_t)
    }
}
