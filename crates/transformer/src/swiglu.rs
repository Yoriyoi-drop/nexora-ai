use ndarray::Array2;
use rand::Rng;
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct SwigluGpuTemps {
    pub w1_t: nexora_autograd::gpu::GpuTensor,
    pub w2_t: nexora_autograd::gpu::GpuTensor,
    pub w3_t: nexora_autograd::gpu::GpuTensor,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct SwigluGpuWeights {
    pub w1_t: nexora_autograd::gpu::GpuTensor,
    pub w2_t: nexora_autograd::gpu::GpuTensor,
    pub w3_t: nexora_autograd::gpu::GpuTensor,
    pub w1_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub w2_f16: Option<nexora_autograd::gpu::GpuTensor>,
    pub w3_f16: Option<nexora_autograd::gpu::GpuTensor>,
}

#[derive(Debug)]
pub struct SwiGLU {
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
    pub w1_f16: Option<Vec<u16>>,
    pub w2_f16: Option<Vec<u16>>,
    pub w3_f16: Option<Vec<u16>>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<SwigluGpuWeights>,
    pub(crate) use_half_precision: bool,
}

#[cfg(not(feature = "gpu"))]
impl Clone for SwiGLU {
    fn clone(&self) -> Self {
        Self {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            w3: self.w3.clone(),
            w1_f16: self.w1_f16.clone(),
            w2_f16: self.w2_f16.clone(),
            w3_f16: self.w3_f16.clone(),
            use_half_precision: self.use_half_precision,
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
            w1_f16: self.w1_f16.clone(),
            w2_f16: self.w2_f16.clone(),
            w3_f16: self.w3_f16.clone(),
            gpu_weights: OnceLock::new(),
            use_half_precision: self.use_half_precision,
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
            w1_f16: None,
            w2_f16: None,
            w3_f16: None,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            use_half_precision: false,
        }
    }

    pub fn pack_f16_weights(&mut self) {
        self.w1_f16 = Some(crate::pack_f32_slice_to_f16(self.w1.as_slice().unwrap()));
        self.w2_f16 = Some(crate::pack_f32_slice_to_f16(self.w2.as_slice().unwrap()));
        self.w3_f16 = Some(crate::pack_f32_slice_to_f16(self.w3.as_slice().unwrap()));
    }

    fn maybe_f16_matmul(&self, x: &Array2<f32>, w: &Array2<f32>, w_f16: &Option<Vec<u16>>) -> Array2<f32> {
        if let Some(f16) = w_f16 {
            let rows = w.shape()[0];
            let cols = w.shape()[1];
            crate::matmul_f16_cpu(x, f16, rows, cols)
        } else {
            x.dot(&w.t())
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Pure CPU forward path.
        // GPU-resident execution uses `forward_gpu` (no per-layer readback).
        let gate = self.maybe_f16_matmul(x, &self.w1, &self.w1_f16);
        let hidden = self.maybe_f16_matmul(x, &self.w3, &self.w3_f16);

        let mut gated = Array2::zeros(gate.dim());

        ndarray::Zip::from(gated.view_mut())
            .and(gate.view())
            .and(hidden.view())
            .for_each(|g, &gv, &hv| {
                let sigmoid = 1.0 / (1.0 + (-gv).exp());
                *g = gv * sigmoid * hv;
            });

        self.maybe_f16_matmul(&gated, &self.w2, &self.w2_f16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_swiglu_forward_shape() {
        let ffn = SwiGLU::new(8, 16);
        let x = array![[1.0; 8], [2.0; 8]];
        let out = ffn.forward(&x);
        assert_eq!(out.dim(), (2, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_forward_batch1() {
        let ffn = SwiGLU::new(4, 6);
        let x = array![[0.5, -0.3, 1.2, -0.7]];
        let out = ffn.forward(&x);
        assert_eq!(out.dim(), (1, 4));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_forward_deterministic() {
        let ffn = SwiGLU::new(4, 8);
        let x = array![[0.1, 0.2, 0.3, 0.4]];
        let out1 = ffn.forward(&x);
        let out2 = ffn.forward(&x);
        for j in 0..4 {
            assert!(
                (out1[[0, j]] - out2[[0, j]]).abs() < 1e-5,
                "mismatch at {j}"
            );
        }
    }

    #[test]
    fn test_swiglu_forward_nonzero_output() {
        let ffn = SwiGLU::new(8, 16);
        let x = array![[1.0; 8]];
        let out = ffn.forward(&x);
        let has_some_signal = out.iter().any(|v| v.abs() > 1e-6);
        assert!(
            has_some_signal,
            "non-zero input to random weights should produce some non-zero output"
        );
    }

    #[test]
    fn test_swiglu_forward_zero_output() {
        let ffn = SwiGLU::new(4, 8);
        let x = array![[0.0; 4]];
        let out = ffn.forward(&x);
        // With zero input, gate=0 so sigmoid(gate)=0.5, silu=0*0.5=0, but hidden=0 also -> 0
        for j in 0..4 {
            assert!(
                out[[0, j]].abs() < 1e-5,
                "zero input should give near-zero output at {j}: {}",
                out[[0, j]]
            );
        }
    }

    #[test]
    fn test_swiglu_silu_property() {
        let ffn = SwiGLU::new(4, 4);
        let x = array![[-10.0, -1.0, 0.0, 10.0]];
        let out = ffn.forward(&x);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_negative_input() {
        let ffn = SwiGLU::new(4, 8);
        let x = array![[-5.0; 4]];
        let out = ffn.forward(&x);
        assert_eq!(out.dim(), (1, 4));
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[cfg(feature = "gpu")]
impl SwiGLU {
    fn ensure_weights_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let ctx = GpuContext::global()?;
        let mk = |arr: &Array2<f32>| -> Result<GpuTensor, nexora_autograd::gpu::GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr
                .as_slice()
                .ok_or_else(|| {
                    nexora_autograd::gpu::GpuError::Unsupported("non-contiguous".into())
                })?
                .to_vec();
            let cpu_arr = ndarray::ArrayD::from_shape_vec(shape, data)
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
            GpuTensor::from_cpu(&cpu_arr)
        };
        let w1 = mk(&self.w1)?;
        let w2 = mk(&self.w2)?;
        let w3 = mk(&self.w3)?;
        let use_f16 = self.use_half_precision;
        let (w1_f16, w2_f16, w3_f16) = if use_f16 {
            (
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w1)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w2)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w3)?)?),
            )
        } else {
            (None, None, None)
        };
        // F16 mode: f32 transposed FFN weights ga dipake —
        // forward_gpu upconvert f16→f32 temp per-call.
        // `zeros(&[1])` DUMMY older (4 bytes per weight, total ~12 bytes),
        // BUKAN 3 matriks FFN ukuran penuh. VRAM hemat: (3 × intermediate × hidden) bytes.
        let (w1_t, w2_t, w3_t) = if use_f16 {
            let d = GpuTensor::zeros(&[1])?;
            (d.clone(), d.clone(), d)
        } else {
            (
                ctx.transpose(&w1)?,
                ctx.transpose(&w2)?,
                ctx.transpose(&w3)?,
            )
        };
        self.gpu_weights
            .set(SwigluGpuWeights {
                w1_t,
                w2_t,
                w3_t,
                w1_f16,
                w2_f16,
                w3_f16,
            })
            .map_err(|_| nexora_autograd::gpu::GpuError::Unsupported("already set".into()))?;
        Ok(())
    }

    /// Pre-upload weights to GPU — call before inference to avoid first-pass latency.
    pub fn preupload_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        self.ensure_weights_gpu()
    }

    fn swiglu_upconvert_f16(
        cached: &SwigluGpuWeights,
        ctx: &nexora_autograd::gpu::GpuContext,
    ) -> Result<SwigluGpuTemps, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuError;
        Ok(SwigluGpuTemps {
            w1_t: ctx.f16_packed_to_f32(cached.w1_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("SwiGLU f16 missing w1".into())
            })?)?,
            w2_t: ctx.f16_packed_to_f32(cached.w2_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("SwiGLU f16 missing w2".into())
            })?)?,
            w3_t: ctx.f16_packed_to_f32(cached.w3_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("SwiGLU f16 missing w3".into())
            })?)?,
        })
    }

    /// GPU forward: SwiGLU FFN using GPU matmul + silu + mul (cached weights).
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        self.ensure_weights_gpu()?;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("SwiGLU weights not initialized".into())
        })?;

        let _f16_temps = if self.use_half_precision {
            Some(Self::swiglu_upconvert_f16(cached, &ctx)?)
        } else {
            None
        };
        let w1_t = _f16_temps.as_ref().map(|t| &t.w1_t).unwrap_or(&cached.w1_t);
        let w2_t = _f16_temps.as_ref().map(|t| &t.w2_t).unwrap_or(&cached.w2_t);
        let w3_t = _f16_temps.as_ref().map(|t| &t.w3_t).unwrap_or(&cached.w3_t);

        let gate = ctx.matmul(x, w1_t)?;
        let hidden = ctx.matmul(x, w3_t)?;
        let silu_gate = ctx.silu(&gate)?;
        let gated = ctx.mul(&silu_gate, &hidden)?;
        ctx.matmul(&gated, w2_t)
    }
}
