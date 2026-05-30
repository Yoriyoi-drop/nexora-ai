use ndarray::{Array1, Array2};
use std::sync::OnceLock;

use crate::{TransformerError, TransformerResult};

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::GpuTensor;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct RmsNormGpuWeights {
    pub weight: GpuTensor,
}

#[derive(Debug)]
pub struct RMSNorm {
    /// CPU weight — `Some` after load/readback, `None` after GPU upload.
    /// Dropped to free RAM in GPU-only production.
    pub weight: Option<Array1<f32>>,
    pub eps: f32,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<RmsNormGpuWeights>,
}

impl Clone for RMSNorm {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            eps: self.eps,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
        }
    }
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Some(Array1::from_shape_fn(hidden_size, |_| 1.0)),
            eps,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
        }
    }

    /// Create with existing weight array.
    pub fn from_weights(weight: Array1<f32>, eps: f32) -> Self {
        Self {
            weight: Some(weight),
            eps,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
        }
    }

    /// Drop CPU weight after GPU upload to free RAM.
    pub fn drop_cpu_weight(&mut self) {
        self.weight = None;
    }

    /// CPU forward — only works if CPU weight is available.
    /// After GPU upload, call `readback_weight()` first if CPU path needed.
    pub fn forward(&self, x: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        let weight = self.weight.as_ref().ok_or_else(|| {
            TransformerError::Implementation("RMSNorm CPU weight not available — call readback_weight() or use forward_gpu()".into())
        })?;
        let (batch_size, hidden_size) = x.dim();
        let mut output = Array2::zeros((batch_size, hidden_size));

        for i in 0..batch_size {
            let row = x.row(i);
            let ssq = row.iter().map(|v| v * v).sum::<f32>();
            let rms = (ssq / hidden_size as f32 + self.eps).sqrt();
            for j in 0..hidden_size {
                output[[i, j]] = (row[j] / rms) * weight[j];
            }
        }

        Ok(output)
    }

    pub fn forward_1d(&self, x: &Array1<f32>) -> TransformerResult<Array1<f32>> {
        let weight = self.weight.as_ref().ok_or_else(|| {
            TransformerError::Implementation("RMSNorm CPU weight not available".into())
        })?;
        let n = x.len();
        let ssq = x.iter().map(|v| v * v).sum::<f32>();
        let rms = (ssq / n as f32 + self.eps).sqrt();
        Ok(x.iter()
            .zip(weight.iter())
            .map(|(&v, &w)| (v / rms) * w)
            .collect())
    }

    /// Upload weight data directly to GPU — takes `&[f32]` so caller
    /// can provide from safetensors without keeping CPU `Array1`.
    #[cfg(feature = "gpu")]
    pub fn preupload_from_slice(&self, weight_data: &[f32]) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let _ctx = GpuContext::global()?;
        let weight_shape = vec![weight_data.len()];
        let weight_arr = ndarray::ArrayD::from_shape_vec(weight_shape, weight_data.to_vec())
            .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
        self.gpu_weights
            .set(RmsNormGpuWeights {
                weight: nexora_autograd::gpu::GpuTensor::from_cpu(&weight_arr)?,
            })
            .map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("already set".into()))?;
        Ok(())
    }

    /// Pre-upload weights to GPU — reads from CPU weight field.
    /// After this, call `drop_cpu_weight()` to free RAM.
    #[cfg(feature = "gpu")]
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        let weight = self.weight.as_ref().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("RMSNorm CPU weight not available for preupload".into())
        })?;
        if let Some(slice) = weight.as_slice() {
            self.preupload_from_slice(slice)
        } else {
            let v: Vec<_> = weight.iter().copied().collect();
            self.preupload_from_slice(&v)
        }
    }

    /// Readback weight from GPU → populate `weight` field.
    /// Used before checkpoint save or training sync.
    #[cfg(feature = "gpu")]
    pub fn readback_weight(&self) -> Result<Array1<f32>, nexora_autograd::gpu::GpuError> {
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("RMSNorm weights not on GPU".into())
        })?;
        let cpu = cached.weight.to_cpu()?;
        let flat = cpu.as_slice().unwrap_or(&[]);
        Ok(Array1::from_vec(flat.to_vec()))
    }

    /// GPU forward: runs RMSNorm on GPU via GpuContext::rms_norm.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        if self.gpu_weights.get().is_none() {
            if let Some(ref w) = self.weight {
                let weight_shape = vec![w.len()];
                let weight_arr = ndarray::ArrayD::from_shape_vec(weight_shape, w.to_vec())
                    .map_err(|e| {
                        nexora_autograd::gpu::GpuError::Unsupported(format!(
                            "shape mismatch for RMS norm weight: {}",
                            e
                        ))
                    })?;
                let _ = self.gpu_weights.set(RmsNormGpuWeights {
                    weight: GpuTensor::from_cpu(&weight_arr)?,
                });
            } else {
                return Err(nexora_autograd::gpu::GpuError::Unsupported(
                    "RMSNorm weights not initialized on GPU and no CPU weight available".into(),
                ));
            }
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
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dim(), (2, 4));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_forward_unit_weight() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![[3.0, -1.0, 2.0, 0.5]];
        let out = norm.forward(&x).unwrap();
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
        let out = norm.forward_1d(&x).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_forward_zero_input() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = array![[0.0, 0.0, 0.0, 0.0]];
        let out = norm.forward(&x).unwrap();
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
        let out_1d = norm.forward_1d(&x).unwrap();
        let out_2d = norm.forward(&x.view().insert_axis(ndarray::Axis(0)).to_owned()).unwrap();
        for j in 0..4 {
            assert!((out_1d[j] - out_2d[[0, j]]).abs() < 1e-5, "mismatch at {j}");
        }
    }

    #[test]
    fn test_rms_norm_custom_weight() {
        let weight = array![2.0, 0.5, 1.0, 3.0];
        let norm = RMSNorm::from_weights(weight, 1e-6);
        let x = array![[1.0, 1.0, 1.0, 1.0]];
        let out = norm.forward(&x).unwrap();
        let ssq = 4.0_f32;
        let rms = (ssq / 4.0 + 1e-6).sqrt();
        assert!((out[[0, 0]] - 1.0 * 2.0 / rms).abs() < 1e-5);
        assert!((out[[0, 1]] - 1.0 * 0.5 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_custom_eps() {
        let norm = RMSNorm::from_weights(array![1.0, 1.0, 1.0], 1.0);
        let x = array![[1e-6, 1e-6, 1e-6]];
        let out = norm.forward(&x).unwrap();
        let rms = (3e-12f32 / 3.0 + 1.0).sqrt();
        let expected = (1e-6 / rms) * 1.0;
        for j in 0..3 {
            assert!((out[[0, j]] - expected).abs() < 1e-12, "mismatch at {j}");
        }
    }

    #[test]
    fn test_rms_norm_drop_cpu_weight() {
        let mut norm = RMSNorm::new(4, 1e-6);
        assert!(norm.weight.is_some());
        norm.drop_cpu_weight();
        assert!(norm.weight.is_none());
    }
}
