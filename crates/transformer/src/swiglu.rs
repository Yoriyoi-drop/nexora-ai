use ndarray::Array2;
use rand::Rng;
use std::sync::OnceLock;

use crate::{TransformerError, TransformerResult};

#[cfg(feature = "gpu")]
use nexora_autograd::gpu::GpuTensor;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct SwigluGpuTemps {
    pub w1_t: nexora_autograd::gpu::GpuTensor,
    pub w2_t: nexora_autograd::gpu::GpuTensor,
    pub w3_t: nexora_autograd::gpu::GpuTensor,
    pub w13_t: nexora_autograd::gpu::GpuTensor,
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
    /// Combined [hidden_size, 2 * ffn_size] weight = concat(w1, w3) column-wise.
    /// Enables single matmul for both gate and hidden projections.
    pub w13_t: nexora_autograd::gpu::GpuTensor,
    pub w13_f16: Option<nexora_autograd::gpu::GpuTensor>,
}

#[derive(Debug)]
pub struct SwiGLU {
    pub w1: Option<Array2<f32>>,
    pub w2: Option<Array2<f32>>,
    pub w3: Option<Array2<f32>>,
    pub w1_f16: Option<Vec<u16>>,
    pub w2_f16: Option<Vec<u16>>,
    pub w3_f16: Option<Vec<u16>>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<SwigluGpuWeights>,
    pub(crate) use_half_precision: bool,
}

impl Clone for SwiGLU {
    fn clone(&self) -> Self {
        tracing::warn!(
            "Cloning SwiGLU — copies 3 weight matrices (~{:.1}M params). Use Arc<SwiGLU> for shared ownership.",
            self.w1.as_ref().map_or(0, |w| w.len()) as f64 / 1_000_000.0 * 3.0
        );
        Self {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            w3: self.w3.clone(),
            w1_f16: self.w1_f16.clone(),
            w2_f16: self.w2_f16.clone(),
            w3_f16: self.w3_f16.clone(),
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            use_half_precision: self.use_half_precision,
        }
    }
}

impl SwiGLU {
    pub fn new(_hidden_size: usize, _intermediate_size: usize) -> Self {
        Self {
            w1: None,
            w2: None,
            w3: None,
            w1_f16: None,
            w2_f16: None,
            w3_f16: None,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            use_half_precision: false,
        }
    }

    /// Initialize CPU weights with random values (for tests / CPU fallback).
    pub fn init_random(&mut self, hidden_size: usize, intermediate_size: usize) {
        let mut rng = rand::thread_rng();
        let scale = (hidden_size as f32).sqrt().recip();
        self.w1 = Some(Array2::from_shape_fn((intermediate_size, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
        self.w2 = Some(Array2::from_shape_fn((hidden_size, intermediate_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
        self.w3 = Some(Array2::from_shape_fn((intermediate_size, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
    }

    pub fn pack_f16_weights(&mut self) {
        if let (Some(w1), Some(w2), Some(w3)) = (&self.w1, &self.w2, &self.w3) {
            let w1_slice = w1.as_slice().unwrap_or_else(|| {
                let v: Vec<f32> = w1.iter().copied().collect();
                // Leak the Vec to get a static &[f32] — safe because pack_f32_slice_to_f16 copies.
                Box::leak(v.into_boxed_slice())
            });
            let w2_slice = w2.as_slice().unwrap_or_else(|| {
                let v: Vec<f32> = w2.iter().copied().collect();
                Box::leak(v.into_boxed_slice())
            });
            let w3_slice = w3.as_slice().unwrap_or_else(|| {
                let v: Vec<f32> = w3.iter().copied().collect();
                Box::leak(v.into_boxed_slice())
            });
            self.w1_f16 = Some(crate::pack_f32_slice_to_f16(w1_slice));
            self.w2_f16 = Some(crate::pack_f32_slice_to_f16(w2_slice));
            self.w3_f16 = Some(crate::pack_f32_slice_to_f16(w3_slice));
        }
    }

    /// Drop CPU weights to free RAM.
    pub fn drop_cpu_weights(&mut self) {
        self.w1 = None;
        self.w2 = None;
        self.w3 = None;
        self.w1_f16 = None;
        self.w2_f16 = None;
        self.w3_f16 = None;
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

    fn get_w1(&self) -> TransformerResult<&Array2<f32>> {
        self.w1.as_ref().ok_or_else(|| {
            TransformerError::Implementation("SwiGLU w1 not available — use readback_weights() or forward_gpu()".into())
        })
    }

    fn get_w2(&self) -> TransformerResult<&Array2<f32>> {
        self.w2.as_ref().ok_or_else(|| {
            TransformerError::Implementation("SwiGLU w2 not available".into())
        })
    }

    fn get_w3(&self) -> TransformerResult<&Array2<f32>> {
        self.w3.as_ref().ok_or_else(|| {
            TransformerError::Implementation("SwiGLU w3 not available".into())
        })
    }

    pub fn forward(&self, x: &Array2<f32>) -> TransformerResult<Array2<f32>> {
        let w1 = self.get_w1()?;
        let w2 = self.get_w2()?;
        let w3 = self.get_w3()?;
        let gate = self.maybe_f16_matmul(x, w1, &self.w1_f16);
        let hidden = self.maybe_f16_matmul(x, w3, &self.w3_f16);

        let mut gated = Array2::zeros(gate.dim());

        ndarray::Zip::from(gated.view_mut())
            .and(gate.view())
            .and(hidden.view())
            .for_each(|g, &gv, &hv| {
                let sigmoid = 1.0 / (1.0 + (-gv).exp());
                *g = gv * sigmoid * hv;
            });

        Ok(self.maybe_f16_matmul(&gated, w2, &self.w2_f16))
    }

    /// Upload weight data directly to GPU from slices.
    #[cfg(feature = "gpu")]
    pub fn preupload_from_slices(
        &self,
        w1_data: &[f32],
        w2_data: &[f32],
        w3_data: &[f32],
        w1_shape: &[usize],
        w2_shape: &[usize],
        w3_shape: &[usize],
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        use ndarray::ArrayD;
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let ctx = GpuContext::global()?;
        let mk = |data: &[f32], shape: &[usize]| -> Result<GpuTensor, GpuError> {
            let arr = ArrayD::from_shape_vec(shape.to_vec(), data.to_vec())
                .map_err(|e| GpuError::Unsupported(e.to_string()))?;
            GpuTensor::from_cpu(&arr)
        };
        let w1 = mk(w1_data, w1_shape)?;
        let w2 = mk(w2_data, w2_shape)?;
        let w3 = mk(w3_data, w3_shape)?;

        // Build combined w13 on CPU
        let w1_shape_2 = [w1_shape[0], w1_shape[1]];
        let w3_shape_2 = [w3_shape[0], w3_shape[1]];
        let w1_cpu = Array2::from_shape_vec(w1_shape_2, w1_data.to_vec())
            .map_err(|e| GpuError::Unsupported(e.to_string()))?;
        let w3_cpu = Array2::from_shape_vec(w3_shape_2, w3_data.to_vec())
            .map_err(|e| GpuError::Unsupported(e.to_string()))?;
        let w13_cpu = crate::swiglu::concat_w1_w3_fused(&w1_cpu, &w3_cpu)
            .map_err(|e| GpuError::Unsupported(e.to_string()))?;
        let w13 = mk(w13_cpu.as_slice().unwrap_or(&[]), &[w13_cpu.nrows(), w13_cpu.ncols()])?;

        let use_f16 = self.use_half_precision;
        let (w1_f16, w2_f16, w3_f16, w13_f16) = if use_f16 {
            (
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w1)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w2)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w3)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w13)?)?),
            )
        } else {
            (None, None, None, None)
        };
        let (w1_t, w2_t, w3_t, w13_t) = if use_f16 {
            let d = GpuTensor::zeros(&[1])?;
            (d.clone(), d.clone(), d.clone(), d)
        } else {
            (
                ctx.transpose(&w1)?,
                ctx.transpose(&w2)?,
                ctx.transpose(&w3)?,
                ctx.transpose(&w13)?,
            )
        };
        self.gpu_weights
            .set(SwigluGpuWeights { w1_t, w2_t, w3_t, w1_f16, w2_f16, w3_f16, w13_t, w13_f16 })
            .map_err(|_| GpuError::Unsupported("already set".into()))
    }

    /// Readback weights from GPU → transpose back to original orientation.
    /// GPU stores w1_t = w1^T for matmul; this returns original w1, w2, w3.
    #[cfg(feature = "gpu")]
    pub fn readback_weights(&self) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuError;
        let cached = self.gpu_weights.get().ok_or_else(|| {
            GpuError::Unsupported("SwiGLU weights not on GPU".into())
        })?;
        let read = |t: &GpuTensor| -> Result<Array2<f32>, nexora_autograd::gpu::GpuError> {
            let cpu = t.to_cpu()?;
            let shape = cpu.shape();
            Array2::from_shape_vec(
                (shape[0], shape[1]),
                cpu.as_slice().unwrap_or(&[]).to_vec(),
            ).map_err(|e| GpuError::Unsupported(e.to_string()))
        };
        let w1_t = read(&cached.w1_t)?;  // [hidden, intermediate]
        let w2_t = read(&cached.w2_t)?;  // [intermediate, hidden]
        let w3_t = read(&cached.w3_t)?;  // [hidden, intermediate]
        // Transpose back to original orientation
        Ok((w1_t.t().to_owned(), w2_t.t().to_owned(), w3_t.t().to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_swiglu_forward_shape() {
        let mut ffn = SwiGLU::new(8, 16);
        ffn.init_random(8, 16);
        let x = array![[1.0; 8], [2.0; 8]];
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.dim(), (2, 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_forward_batch1() {
        let mut ffn = SwiGLU::new(4, 6);
        ffn.init_random(4, 6);
        let x = array![[0.5, -0.3, 1.2, -0.7]];
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.dim(), (1, 4));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_forward_deterministic() {
        let mut ffn = SwiGLU::new(4, 8);
        ffn.init_random(4, 8);
        let x = array![[0.1, 0.2, 0.3, 0.4]];
        let out1 = ffn.forward(&x).unwrap();
        let out2 = ffn.forward(&x).unwrap();
        for j in 0..4 {
            assert!(
                (out1[[0, j]] - out2[[0, j]]).abs() < 1e-5,
                "mismatch at {j}"
            );
        }
    }

    #[test]
    fn test_swiglu_forward_nonzero_output() {
        let mut ffn = SwiGLU::new(8, 16);
        ffn.init_random(8, 16);
        let x = array![[1.0; 8]];
        let out = ffn.forward(&x).unwrap();
        let has_some_signal = out.iter().any(|v| v.abs() > 1e-6);
        assert!(
            has_some_signal,
            "non-zero input to random weights should produce some non-zero output"
        );
    }

    #[test]
    fn test_swiglu_forward_zero_output() {
        let mut ffn = SwiGLU::new(4, 8);
        ffn.init_random(4, 8);
        let x = array![[0.0; 4]];
        let out = ffn.forward(&x).unwrap();
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
        let mut ffn = SwiGLU::new(4, 4);
        ffn.init_random(4, 4);
        let x = array![[-10.0, -1.0, 0.0, 10.0]];
        let out = ffn.forward(&x).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_negative_input() {
        let mut ffn = SwiGLU::new(4, 8);
        ffn.init_random(4, 8);
        let x = array![[-5.0; 4]];
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.dim(), (1, 4));
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

/// Concatenate w1 and w3 along the column dimension (public for loader.rs).
/// Shape: [hidden_size, 2 * ffn_size] = concat([hidden_size, ffn_size], ...)
pub fn concat_w1_w3_fused(w1: &Array2<f32>, w3: &Array2<f32>) -> TransformerResult<Array2<f32>> {
    let (rows, cols1) = w1.dim();
    let (_, cols3) = w3.dim();
    if rows == 0 {
        return Err(TransformerError::Implementation("Empty w1 for concat".into()));
    }
    let mut combined = Array2::zeros((rows, cols1 + cols3));
    combined.slice_mut(ndarray::s![.., ..cols1]).assign(w1);
    combined.slice_mut(ndarray::s![.., cols1..]).assign(w3);
    Ok(combined)
}

#[cfg(feature = "gpu")]
impl SwiGLU {
    fn ensure_weights_gpu(&self) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};
        if self.gpu_weights.get().is_some() {
            return Ok(());
        }
        let ctx = GpuContext::global()?;
        let mk = |arr: &Array2<f32>| -> Result<GpuTensor, GpuError> {
            let shape = vec![arr.shape()[0], arr.shape()[1]];
            let data = arr
                .as_slice()
                .ok_or_else(|| GpuError::Unsupported("non-contiguous".into()))?
                .to_vec();
            let cpu_arr = ndarray::ArrayD::from_shape_vec(shape, data)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?;
            GpuTensor::from_cpu(&cpu_arr)
        };
        let w1 = match self.w1.as_ref() {
            Some(w) => mk(w)?,
            None => return Err(GpuError::Unsupported("SwiGLU w1 not available for GPU upload".into())),
        };
        let w2 = match self.w2.as_ref() {
            Some(w) => mk(w)?,
            None => return Err(GpuError::Unsupported("SwiGLU w2 not available".into())),
        };
        let w3 = match self.w3.as_ref() {
            Some(w) => mk(w)?,
            None => return Err(GpuError::Unsupported("SwiGLU w3 not available".into())),
        };
        let use_f16 = self.use_half_precision;
        // Build combined w13 on CPU before GPU upload
        let w13_cpu = Self::concat_w1_w3(
            self.w1.as_ref().ok_or_else(|| GpuError::Unsupported("SwiGLU w1 not available".into()))?,
            self.w3.as_ref().ok_or_else(|| GpuError::Unsupported("SwiGLU w3 not available".into()))?,
        ).map_err(|e| GpuError::Unsupported(format!("concat_w1_w3: {e}")))?;
        let w13 = mk(&w13_cpu)?;

        let (w1_f16, w2_f16, w3_f16, w13_f16) = if use_f16 {
            (
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w1)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w2)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w3)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&w13)?)?),
            )
        } else {
            (None, None, None, None)
        };
        let (w1_t, w2_t, w3_t, w13_t) = if use_f16 {
            let d = GpuTensor::zeros(&[1])?;
            (d.clone(), d.clone(), d.clone(), d)
        } else {
            (
                ctx.transpose(&w1)?,
                ctx.transpose(&w2)?,
                ctx.transpose(&w3)?,
                ctx.transpose(&w13)?,
            )
        };
        self.gpu_weights
            .set(SwigluGpuWeights { w1_t, w2_t, w3_t, w1_f16, w2_f16, w3_f16, w13_t, w13_f16 })
            .map_err(|_| GpuError::Unsupported("already set".into()))?;
        Ok(())
    }

    /// Pre-upload weights to GPU from CPU Option fields.
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
            w13_t: ctx.f16_packed_to_f32(cached.w13_f16.as_ref().ok_or_else(|| {
                GpuError::Unsupported("SwiGLU f16 missing w13".into())
            })?)?,
        })
    }

    /// Concatenate w1 and w3 along the column (ffn_hidden) dimension.
    /// Delegates to public `concat_w1_w3_fused`.
    fn concat_w1_w3(
        w1: &Array2<f32>,
        w3: &Array2<f32>,
    ) -> TransformerResult<Array2<f32>> {
        concat_w1_w3_fused(w1, w3)
    }

    /// GPU forward: SwiGLU FFN using GPU matmul + silu + mul (cached weights).
    /// Uses 3 well-optimized cuBLAS matmuls (fusion deferred to future work).
    pub fn forward_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::GpuContext;
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
