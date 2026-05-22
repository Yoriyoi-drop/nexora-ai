use ndarray::{Array1, Array2};

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::{GpuContext, GpuTensor};

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct RmsNormGpuWeights {
    pub weight: GpuTensor,
}

#[derive(Debug)]
pub struct RMSNorm {
    pub weight: Array1<f32>,
    pub eps: f32,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: parking_lot::Mutex<Option<RmsNormGpuWeights>>,
}

#[cfg(not(feature = "gpu"))]
impl Clone for RMSNorm {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            eps: self.eps.clone(),
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for RMSNorm {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            eps: self.eps,
            gpu_weights: parking_lot::Mutex::new(None),
        }
    }
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Array1::from_shape_fn(hidden_size, |_| 1.0);
        Self {
            weight,
            eps,
            #[cfg(feature = "gpu")]
            gpu_weights: parking_lot::Mutex::new(None),
        }
    }

    pub fn from_weights(weight: Array1<f32>, eps: f32) -> Self {
        Self {
            weight,
            eps,
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

        let (batch_size, hidden_size) = x.dim();
        let mut output = Array2::zeros((batch_size, hidden_size));

        for i in 0..batch_size {
            let row = x.row(i);
            let ssq = row.iter().map(|v| v * v).sum::<f32>();
            let rms = (ssq / hidden_size as f32 + self.eps).sqrt();
            for j in 0..hidden_size {
                output[[i, j]] = (row[j] / rms) * self.weight[j];
            }
        }

        output
    }

    pub fn forward_1d(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len();
        let ssq = x.iter().map(|v| v * v).sum::<f32>();
        let rms = (ssq / n as f32 + self.eps).sqrt();
        x.iter()
            .zip(self.weight.iter())
            .map(|(&v, &w)| (v / rms) * w)
            .collect()
    }

    /// Pre-upload weights to GPU — call before inference to avoid first-pass latency.
    #[cfg(feature = "gpu")]
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;
        let _ctx = GpuContext::global()?;
        let mut guard = self.gpu_weights.lock();
        if guard.is_some() {
            return Ok(());
        }
        let weight_shape = vec![self.weight.len()];
        let weight_arr = ndarray::ArrayD::from_shape_vec(weight_shape, self.weight.to_vec())
            .map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("shape error".into()))?;
        *guard = Some(RmsNormGpuWeights {
            weight: nexora_autograd::gpu::GpuTensor::from_cpu(&weight_arr)?,
        });
        Ok(())
    }

    /// GPU forward: runs RMSNorm on GPU via GpuContext::rms_norm.
    /// x: [batch, dim] on GPU, returns normalized output on GPU.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        let mut guard = self.gpu_weights.lock();
        let cached = guard.get_or_insert_with(|| {
            let weight_shape = vec![self.weight.len()];
            let weight_arr = ndarray::ArrayD::from_shape_vec(weight_shape, self.weight.to_vec())
                .unwrap();
            RmsNormGpuWeights {
                weight: GpuTensor::from_cpu(&weight_arr).unwrap(),
            }
        });
        ctx.rms_norm(x, &cached.weight, self.eps)
    }
}
