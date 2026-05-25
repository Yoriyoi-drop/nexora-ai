use ndarray::{Array1, Array2};
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::GpuTensor;

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
    pub(crate) gpu_weights: OnceLock<RmsNormGpuWeights>,
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
            gpu_weights: OnceLock::new(),
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
            gpu_weights: OnceLock::new(),
        }
    }

    pub fn from_weights(weight: Array1<f32>, eps: f32) -> Self {
        Self {
            weight,
            eps,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Pure CPU forward path.
        // GPU-resident execution uses `forward_gpu` (no per-layer readback).
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
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let weight_shape = vec![self.weight.len()];
        let weight_arr = ndarray::ArrayD::from_shape_vec(weight_shape, self.weight.to_vec())
            .map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("shape error".into()))?;
        self.gpu_weights
            .set(RmsNormGpuWeights {
                weight: nexora_autograd::gpu::GpuTensor::from_cpu(&weight_arr)?,
            })
            .map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("already set".into()))?;
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
        if self.gpu_weights.get().is_none() {
            let weight_shape = vec![self.weight.len()];
            let weight_arr = ndarray::ArrayD::from_shape_vec(weight_shape, self.weight.to_vec())
                .map_err(|e| {
                    nexora_autograd::gpu::GpuError::Unsupported(format!(
                        "shape mismatch for RMS norm weight: {}",
                        e
                    ))
                })?;
            let _ = self.gpu_weights.set(RmsNormGpuWeights {
                weight: GpuTensor::from_cpu(&weight_arr)?,
            });
        }
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("RMS norm weights not initialized".into())
        })?;
        ctx.rms_norm(x, &cached.weight, self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rms_norm_forward_shape() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let out = norm.forward(&x);
        assert_eq!(out.dim(), (2, 4));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_forward_unit_weight() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![[3.0, -1.0, 2.0, 0.5]];
        let out = norm.forward(&x);
        let ssq = x.row(0).mapv(|v| v * v).sum();
        let rms = (ssq / 4.0 + 1e-6).sqrt();
        let expected = x.row(0).mapv(|v| v / rms);
        for j in 0..4 {
            assert!((out[[0, j]] - expected[j]).abs() < 1e-5, "mismatch at {j}");
        }
    }

    #[test]
    fn test_rms_norm_forward_1d() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![2.0, 4.0, 6.0, 8.0];
        let out = norm.forward_1d(&x);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_forward_zero_input() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![[0.0, 0.0, 0.0, 0.0]];
        let out = norm.forward(&x);
        for j in 0..4 {
            assert_eq!(
                out[[0, j]],
                0.0,
                "zero input should give zero output at {j}"
            );
        }
    }

    #[test]
    fn test_rms_norm_forward_1d_matches_forward() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![1.5, -2.5, 3.5, -4.5];
        let out_1d = norm.forward_1d(&x);
        let out_2d = norm.forward(&x.view().insert_axis(ndarray::Axis(0)).to_owned());
        for j in 0..4 {
            assert!((out_1d[j] - out_2d[[0, j]]).abs() < 1e-5, "mismatch at {j}");
        }
    }

    #[test]
    fn test_rms_norm_custom_weight() {
        let weight = array![2.0, 0.5, 1.0, 3.0];
        let norm = RMSNorm::from_weights(weight, 1e-6);
        let x = array![[1.0, 1.0, 1.0, 1.0]];
        let out = norm.forward(&x);
        let ssq = 4.0_f32;
        let rms = (ssq / 4.0 + 1e-6).sqrt();
        assert!((out[[0, 0]] - 1.0 * 2.0 / rms).abs() < 1e-5);
        assert!((out[[0, 1]] - 1.0 * 0.5 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_custom_eps() {
        let norm = RMSNorm::from_weights(array![1.0, 1.0, 1.0], 1.0);
        let x = array![[1e-6, 1e-6, 1e-6]];
        let out = norm.forward(&x);
        let rms = (3e-12f32 / 3.0 + 1.0).sqrt();
        let expected = (1e-6 / rms) * 1.0;
        for j in 0..3 {
            assert!((out[[0, j]] - expected).abs() < 1e-12, "mismatch at {j}");
        }
    }
}
