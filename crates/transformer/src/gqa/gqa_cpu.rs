use std::borrow::Cow;
use std::sync::OnceLock;
use ndarray::{Array1, Array2};
use rand::Rng;
use rayon::prelude::*;
use crate::rope::RoPE;
use super::kv_cache::{KVCacheProvider, KVCacheEntry, PagedCacheReader};
use crate::{TransformerError, TransformerResult};
#[cfg(feature = "gpu")]
use super::gpu::GqaGpuWeights;
#[cfg(feature = "gpu")]
use super::gpu::GpuKVCacheEntry;

#[derive(Debug)]
pub struct GQA {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_groups: usize,
    pub head_dim_rs: f32,
    pub wq: Option<Array2<f32>>,
    pub wk: Option<Array2<f32>>,
    pub wv: Option<Array2<f32>>,
    pub wo: Option<Array2<f32>>,
    pub wq_f16: Option<Vec<u16>>,
    pub wk_f16: Option<Vec<u16>>,
    pub wv_f16: Option<Vec<u16>>,
    pub wo_f16: Option<Vec<u16>>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: OnceLock<GqaGpuWeights>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_scratch: parking_lot::RwLock<Option<GpuKVCacheEntry>>,
    pub(crate) use_half_precision: bool,
}

#[cfg(not(feature = "gpu"))]
impl Clone for GQA {
    fn clone(&self) -> Self {
        tracing::warn!(
            "Cloning GQA — copies all weight matrices (~{:.1}M params). Use Arc<GQA> for shared ownership.",
            (self.num_heads + self.num_kv_heads * 2 + self.num_heads) as f64
                * self.head_dim as f64
                * self.head_dim as f64
                / 1_000_000.0
        );
        Self {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            num_groups: self.num_groups,
            head_dim_rs: self.head_dim_rs,
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            wq_f16: self.wq_f16.clone(),
            wk_f16: self.wk_f16.clone(),
            wv_f16: self.wv_f16.clone(),
            wo_f16: self.wo_f16.clone(),
            use_half_precision: self.use_half_precision,
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for GQA {
    fn clone(&self) -> Self {
        tracing::warn!(
            "Cloning GQA — copies all weight matrices (~{:.1}M params). Use Arc<GQA> for shared ownership.",
            (self.num_heads + self.num_kv_heads * 2 + self.num_heads) as f64
                * self.head_dim as f64
                * self.head_dim as f64
                / 1_000_000.0
        );
        Self {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            num_groups: self.num_groups,
            head_dim_rs: self.head_dim_rs,
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            wq_f16: self.wq_f16.clone(),
            wk_f16: self.wk_f16.clone(),
            wv_f16: self.wv_f16.clone(),
            wo_f16: self.wo_f16.clone(),
            gpu_weights: OnceLock::new(),
            gpu_scratch: parking_lot::RwLock::new(None),
            use_half_precision: self.use_half_precision,
        }
    }
}

impl GQA {
    pub fn new(_hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            num_groups: num_heads / num_kv_heads,
            head_dim_rs: 1.0 / (head_dim as f32).sqrt(),
            wq: None,
            wk: None,
            wv: None,
            wo: None,
            wq_f16: None,
            wk_f16: None,
            wv_f16: None,
            wo_f16: None,
            #[cfg(feature = "gpu")]
            gpu_weights: OnceLock::new(),
            #[cfg(feature = "gpu")]
            gpu_scratch: parking_lot::RwLock::new(None),
            use_half_precision: false,
        }
    }

    pub fn init_random(&mut self, hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) {
        let mut rng = rand::thread_rng();
        let scale = (head_dim as f32).sqrt().recip();
        self.wq = Some(Array2::from_shape_fn((num_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
        self.wk = Some(Array2::from_shape_fn((num_kv_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
        self.wv = Some(Array2::from_shape_fn((num_kv_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
        self.wo = Some(Array2::from_shape_fn((hidden_size, num_heads * head_dim), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        }));
    }

    fn get_wq(&self) -> TransformerResult<&Array2<f32>> {
        self.wq.as_ref().ok_or_else(|| {
            TransformerError::Implementation("GQA wq not available — use readback_weights() or forward_gpu()".into())
        })
    }

    fn get_wk(&self) -> TransformerResult<&Array2<f32>> {
        self.wk.as_ref().ok_or_else(|| {
            TransformerError::Implementation("GQA wk not available".into())
        })
    }

    fn get_wv(&self) -> TransformerResult<&Array2<f32>> {
        self.wv.as_ref().ok_or_else(|| {
            TransformerError::Implementation("GQA wv not available".into())
        })
    }

    fn get_wo(&self) -> TransformerResult<&Array2<f32>> {
        self.wo.as_ref().ok_or_else(|| {
            TransformerError::Implementation("GQA wo not available".into())
        })
    }

    pub fn pack_f16_weights(&mut self) {
        if let (Some(wq), Some(wk), Some(wv), Some(wo)) =
            (&self.wq, &self.wk, &self.wv, &self.wo)
        {
            let wq_contig = wq.iter().copied().collect::<Vec<f32>>();
            let wk_contig = wk.iter().copied().collect::<Vec<f32>>();
            let wv_contig = wv.iter().copied().collect::<Vec<f32>>();
            let wo_contig = wo.iter().copied().collect::<Vec<f32>>();
            self.wq_f16 = Some(crate::pack_f32_slice_to_f16(&wq_contig));
            self.wk_f16 = Some(crate::pack_f32_slice_to_f16(&wk_contig));
            self.wv_f16 = Some(crate::pack_f32_slice_to_f16(&wv_contig));
            self.wo_f16 = Some(crate::pack_f32_slice_to_f16(&wo_contig));
        }
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

    pub fn drop_cpu_weights(&mut self) {
        self.wq = None;
        self.wk = None;
        self.wv = None;
        self.wo = None;
        self.wq_f16 = None;
        self.wk_f16 = None;
        self.wv_f16 = None;
        self.wo_f16 = None;
    }

    #[cfg(feature = "gpu")]
    pub fn preupload_from_slices(
        &self,
        wq_data: &[f32], wk_data: &[f32], wv_data: &[f32], wo_data: &[f32],
        wq_shape: &[usize], wk_shape: &[usize], wv_shape: &[usize], wo_shape: &[usize],
    ) -> Result<(), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};
        let ctx = GpuContext::global()?;
        let mk = |data: &[f32], shape: &[usize]| -> Result<GpuTensor, GpuError> {
            let arr = ndarray::ArrayD::from_shape_vec(shape.to_vec(), data.to_vec())
                .map_err(|e| GpuError::Unsupported(e.to_string()))?;
            GpuTensor::from_cpu(&arr)
        };
        let wq = mk(wq_data, wq_shape)?;
        let wk = mk(wk_data, wk_shape)?;
        let wv = mk(wv_data, wv_shape)?;
        let wo = mk(wo_data, wo_shape)?;
        let use_f16 = self.use_half_precision;
        let (wq_f16, wk_f16, wv_f16, wo_f16) = if use_f16 {
            (
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wq)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wk)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wv)?)?),
                Some(ctx.f32_to_f16_packed(&ctx.transpose(&wo)?)?),
            )
        } else {
            (None, None, None, None)
        };
        // F16 mode: only store f16 packed weights (2× less VRAM). Skip f32 upload.
        // f32→f16 upconversion happens ephemerally per forward pass via gqa_upconvert_f16.
        // F32 mode: store f32 transposed weights as before.
        let (wq_t, wk_t, wv_t, wo_t) = if !use_f16 {
            (
                Some(ctx.transpose(&wq)?),
                Some(ctx.transpose(&wk)?),
                Some(ctx.transpose(&wv)?),
                Some(ctx.transpose(&wo)?),
            )
        } else {
            (None, None, None, None)
        };
        self.gpu_weights
            .set(GqaGpuWeights {
                wq_t, wk_t, wv_t, wo_t,
                wq_f16, wk_f16, wv_f16, wo_f16,
                wq_shape: wq_shape.to_vec(),
                wk_shape: wk_shape.to_vec(),
                wv_shape: wv_shape.to_vec(),
                wo_shape: wo_shape.to_vec(),
                is_f16: use_f16,
            })
            .map_err(|_| GpuError::Unsupported("already set".into()))?;
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn readback_weights(&self) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        let gw = self.gpu_weights.get().ok_or_else(|| {
            nexora_autograd::gpu::GpuError::Unsupported("no gpu weights".into())
        })?;
        let read_f16 = |f16_t: &GpuTensor, orig_shape: &[usize]| -> Result<Array2<f32>, nexora_autograd::gpu::GpuError> {
            let f32_t = ctx.f16_packed_to_f32(f16_t)?;
            let arr_d = f32_t.to_cpu()?;
            let shape = vec![orig_shape[0], orig_shape[1]];
            let arr_dyn = ndarray::ArrayD::from_shape_vec(shape, arr_d.into_raw_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
            arr_dyn.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))
        };
        let read_f32 = |t: &GpuTensor, orig_shape: &[usize]| -> Result<Array2<f32>, nexora_autograd::gpu::GpuError> {
            let t_t = ctx.transpose(t)?;
            let arr_d = t_t.to_cpu()?;
            let shape = vec![orig_shape[0], orig_shape[1]];
            let arr_dyn = ndarray::ArrayD::from_shape_vec(shape, arr_d.into_raw_vec())
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
            arr_dyn.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))
        };
        // Use stored shapes from GqaGpuWeights (not self.wq which may be None after drop_cpu_weights)
        let wq_shape = &gw.wq_shape;
        let wk_shape = &gw.wk_shape;
        let wv_shape = &gw.wv_shape;
        let wo_shape = &gw.wo_shape;
        // Read via f16 path (preferred, 2× less VRAM) or f32 path.
        // When use_half_precision is active, only f16 packed weights exist on GPU.
        let wq = if let Some(ref f16) = gw.wq_f16 {
            read_f16(f16, wq_shape)?
        } else if let Some(ref f32_t) = gw.wq_t {
            read_f32(f32_t, wq_shape)?
        } else {
            return Err(nexora_autograd::gpu::GpuError::Unsupported(
                "GQA weights not available on GPU".into()
            ));
        };
        let wk = if let Some(ref f16) = gw.wk_f16 {
            read_f16(f16, wk_shape)?
        } else if let Some(ref f32_t) = gw.wk_t {
            read_f32(f32_t, wk_shape)?
        } else {
            return Err(nexora_autograd::gpu::GpuError::Unsupported(
                "GQA weights not available on GPU".into()
            ));
        };
        let wv = if let Some(ref f16) = gw.wv_f16 {
            read_f16(f16, wv_shape)?
        } else if let Some(ref f32_t) = gw.wv_t {
            read_f32(f32_t, wv_shape)?
        } else {
            return Err(nexora_autograd::gpu::GpuError::Unsupported(
                "GQA weights not available on GPU".into()
            ));
        };
        let wo = if let Some(ref f16) = gw.wo_f16 {
            read_f16(f16, wo_shape)?
        } else if let Some(ref f32_t) = gw.wo_t {
            read_f32(f32_t, wo_shape)?
        } else {
            return Err(nexora_autograd::gpu::GpuError::Unsupported(
                "GQA weights not available on GPU".into()
            ));
        };
        Ok((wq, wk, wv, wo))
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        cache: Option<&mut Vec<KVCacheEntry>>,
        layer_idx: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> TransformerResult<Array2<f32>> {
        let (batch_size, _) = x.dim();

        let wq = self.get_wq()?;
        let wk = self.get_wk()?;
        let wv = self.get_wv()?;
        let wo = self.get_wo()?;

        let q_proj = self.maybe_f16_matmul(x, wq, &self.wq_f16);
        let k_proj = self.maybe_f16_matmul(x, wk, &self.wk_f16);
        let v_proj = self.maybe_f16_matmul(x, wv, &self.wv_f16);

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_heads, self.head_dim))
            });
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });

        {
            let k_shape = k.dim();
            let k_2d = k
                .into_shape((k_shape.0 * k_shape.1, k_shape.2))
                .unwrap_or_else(|_| {
                    ndarray::Array2::zeros((k_shape.0 * k_shape.1, k_shape.2))
                });
            let half = self.head_dim / 2;
            let cos_1d = ndarray::Array1::from_shape_fn(half, |i| cos[i]);
            let sin_1d = ndarray::Array1::from_shape_fn(half, |i| sin[i]);
            let k_rotated = RoPE::apply(&k_2d, &cos_1d, &sin_1d, self.head_dim);
            k = k_rotated
                .into_shape(k_shape)
                .unwrap_or_else(|_| ndarray::Array3::zeros(k_shape));
        }

        {
            let q_shape = q.dim();
            let q_2d = q
                .into_shape((q_shape.0 * q_shape.1, q_shape.2))
                .unwrap_or_else(|_| {
                    ndarray::Array2::zeros((q_shape.0 * q_shape.1, q_shape.2))
                });
            let half = self.head_dim / 2;
            let cos_1d = ndarray::Array1::from_shape_fn(half, |i| cos[i]);
            let sin_1d = ndarray::Array1::from_shape_fn(half, |i| sin[i]);
            let q_rotated = RoPE::apply(&q_2d, &cos_1d, &sin_1d, self.head_dim);
            q = q_rotated
                .into_shape(q_shape)
                .unwrap_or_else(|_| ndarray::Array3::zeros(q_shape));
        }

        let kv_dim = self.num_kv_heads * self.head_dim;
        let (k_cached, v_cached, total_seq, is_causal): (Cow<[f32]>, Cow<[f32]>, usize, bool) = match cache {
            Some(cache) if layer_idx < cache.len() => {
                let entry = &cache[layer_idx];
                let seq = entry.k.len() / (batch_size * kv_dim);
                (Cow::Borrowed(&entry.k), Cow::Borrowed(&entry.v), seq, false)
            }
            _ => {
                let kf: Vec<f32> = k.iter().copied().collect();
                let vf: Vec<f32> = v.iter().copied().collect();
                // No cache: batch_size = seq_len, all tokens attend in one forward.
                // Apply causal mask so token i only attends to tokens 0..=i.
                (Cow::Owned(kf), Cow::Owned(vf), batch_size, true)
            }
        };

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            let k_start = b * total_seq * kv_dim;
            let k_end = k_start + total_seq * kv_dim;
            let v_start = b * total_seq * kv_dim;
            let v_end = v_start + total_seq * kv_dim;
            if k_end > k_cached.len() || v_end > v_cached.len() {
                continue;
            }
            let Ok(k_2d) = ndarray::ArrayView2::from_shape(
                (total_seq, kv_dim),
                &k_cached[k_start..k_end],
            ) else { continue };
            let Ok(v_2d) = ndarray::ArrayView2::from_shape(
                (total_seq, kv_dim),
                &v_cached[v_start..v_end],
            ) else { continue };

            for kv_h in 0..self.num_kv_heads {
                let kv_off = kv_h * self.head_dim;
                let start_head = kv_h * self.num_groups;
                let end_head = (start_head + self.num_groups).min(self.num_heads);
                let n_heads = end_head - start_head;

                let k_slice = k_2d.slice(ndarray::s![.., kv_off..kv_off + self.head_dim]);
                let v_slice = v_2d.slice(ndarray::s![.., kv_off..kv_off + self.head_dim]);

                let mut q_group = Array2::zeros((n_heads, self.head_dim));
                for (i, h) in (start_head..end_head).enumerate() {
                    let q_row = q.slice(ndarray::s![b, h, ..]);
                    q_group.row_mut(i).assign(&q_row);
                }

                let scores = q_group.dot(&k_slice.t().to_owned());
                let scaled = scores.mapv(|s| s * self.head_dim_rs);

                // Apply causal mask: token b can only attend to positions 0..=b
                let causal_pos = if is_causal { b } else { total_seq - 1 };

                let softmax_out = {
                    let mut out = Array2::zeros((n_heads, total_seq));
                    for r in 0..n_heads {
                        let mut max_v = f32::NEG_INFINITY;
                        for c in 0..=causal_pos {
                            let v = scaled[[r, c]];
                            if v > max_v {
                                max_v = v;
                            }
                        }
                        let mut sum = 0.0;
                        for c in 0..=causal_pos {
                            let v = (scaled[[r, c]] - max_v).exp();
                            out[[r, c]] = v;
                            sum += v;
                        }
                        // Positions beyond causal_pos stay 0 (masked out)
                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for c in 0..=causal_pos {
                                out[[r, c]] *= inv;
                            }
                        }
                    }
                    out
                };

                let attn_out = softmax_out.dot(&v_slice);
                for (i, h) in (start_head..end_head).enumerate() {
                    let out_base = h * self.head_dim;
                    for d in 0..self.head_dim {
                        output[[b, out_base + d]] = attn_out[[i, d]];
                    }
                }
            }
        }

        Ok(self.maybe_f16_matmul(&output, wo, &self.wo_f16))
    }

    pub fn forward_with_kv(
        &self,
        x: &Array2<f32>,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> TransformerResult<Array2<f32>> {
        let (batch_size, _) = x.dim();

        let wq = self.get_wq()?;
        let wk = self.get_wk()?;
        let wv = self.get_wv()?;
        let wo = self.get_wo()?;

        let q_proj = x.dot(&wq.t());
        let k_proj = x.dot(&wk.t());
        let v_proj = x.dot(&wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_heads, self.head_dim))
            });
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        {
            let k_shape = k.dim();
            let k_2d = k
                .into_shape((k_shape.0 * k_shape.1, k_shape.2))
                .unwrap_or_else(|_| {
                    ndarray::Array2::zeros((k_shape.0 * k_shape.1, k_shape.2))
                });
            let half = self.head_dim / 2;
            let cos_1d = ndarray::Array1::from_shape_fn(half, |i| cos[i]);
            let sin_1d = ndarray::Array1::from_shape_fn(half, |i| sin[i]);
            let k_rotated = RoPE::apply(&k_2d, &cos_1d, &sin_1d, self.head_dim);
            k = k_rotated
                .into_shape(k_shape)
                .unwrap_or_else(|_| ndarray::Array3::zeros(k_shape));
        }

        {
            let q_shape = q.dim();
            let q_2d = q
                .into_shape((q_shape.0 * q_shape.1, q_shape.2))
                .unwrap_or_else(|_| {
                    ndarray::Array2::zeros((q_shape.0 * q_shape.1, q_shape.2))
                });
            let half = self.head_dim / 2;
            let cos_1d = ndarray::Array1::from_shape_fn(half, |i| cos[i]);
            let sin_1d = ndarray::Array1::from_shape_fn(half, |i| sin[i]);
            let q_rotated = RoPE::apply(&q_2d, &cos_1d, &sin_1d, self.head_dim);
            q = q_rotated
                .into_shape(q_shape)
                .unwrap_or_else(|_| ndarray::Array3::zeros(q_shape));
        }

        let kv_dim = self.num_kv_heads * self.head_dim;

        if layer_idx < cache.len() {
            let entry = &mut cache[layer_idx];
            entry.k.extend(k.iter().copied());
            entry.v.extend(v.iter().copied());
        } else {
            cache.push(KVCacheEntry {
                k: k.iter().copied().collect(),
                v: v.iter().copied().collect(),
                kv_dim,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                k_compressed: Vec::new(),
                v_compressed: Vec::new(),
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                compressed_seq_len: 0,
            });
        }

        let entry = &mut cache[layer_idx];
        entry.ensure_dequantized();
        let k_slice = &entry.k;
        let v_slice = &entry.v;
        let total_seq = entry.seq_len();

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            let results: Vec<(usize, Vec<f32>)> = (0..self.num_heads)
                .into_par_iter()
                .map(|h| {
                    let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                    let kv_off = kv_h * self.head_dim;

                    let q_slice = q.slice(ndarray::s![b, h, ..]);
                    let mut scores = vec![0.0; total_seq];
                    let mut max_score = f32::NEG_INFINITY;
                    for t in 0..total_seq {
                        let k_idx = t * kv_dim + kv_off;
                        let k_view =
                            ndarray::ArrayView1::from(&k_slice[k_idx..k_idx + self.head_dim]);
                        let score = q_slice.dot(&k_view) * self.head_dim_rs;
                        if score > max_score {
                            max_score = score;
                        }
                        scores[t] = score;
                    }

                    let mut exp_sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }

                    let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                    let mut out_row = vec![0.0; self.head_dim];
                    for d in 0..self.head_dim {
                        let mut weighted = 0.0;
                        for t in 0..total_seq {
                            weighted += scores[t] * inv_exp_sum * v_slice[t * kv_dim + kv_off + d];
                        }
                        out_row[d] = weighted;
                    }
                    (h, out_row)
                })
                .collect();

            for (h, row) in results {
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    output[[b, out_base + d]] = row[d];
                }
            }
        }

        Ok(self.maybe_f16_matmul(&output, wo, &self.wo_f16))
    }

    /// Forward pass using a paged KV cache (reads K/V from blocks per position).
    /// Supports batch_size >= 1 for batched paged attention.
    pub fn forward_with_paged(
        &self,
        x: &Array2<f32>,
        cache: &mut dyn PagedCacheReader,
        seq_id: u64,
        layer_idx: usize,
        token_pos: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> TransformerResult<Array2<f32>> {
        let (batch_size, _) = x.dim();

        let wq = self.get_wq()?;
        let wk = self.get_wk()?;
        let wv = self.get_wv()?;
        let wo = self.get_wo()?;

        let q_proj = self.maybe_f16_matmul(x, wq, &self.wq_f16);
        let k_proj = self.maybe_f16_matmul(x, wk, &self.wk_f16);
        let v_proj = self.maybe_f16_matmul(x, wv, &self.wv_f16);

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_heads, self.head_dim))
            });
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .unwrap_or_else(|_| {
                ndarray::Array3::zeros((batch_size, self.num_kv_heads, self.head_dim))
            });

        {
            let k_shape = k.dim();
            let k_2d = k
                .into_shape((k_shape.0 * k_shape.1, k_shape.2))
                .unwrap_or_else(|_| {
                    ndarray::Array2::zeros((k_shape.0 * k_shape.1, k_shape.2))
                });
            let half = self.head_dim / 2;
            let cos_1d = ndarray::Array1::from_shape_fn(half, |i| cos[i]);
            let sin_1d = ndarray::Array1::from_shape_fn(half, |i| sin[i]);
            let k_rotated = RoPE::apply(&k_2d, &cos_1d, &sin_1d, self.head_dim);
            k = k_rotated
                .into_shape(k_shape)
                .unwrap_or_else(|_| ndarray::Array3::zeros(k_shape));
        }

        {
            let q_shape = q.dim();
            let q_2d = q
                .into_shape((q_shape.0 * q_shape.1, q_shape.2))
                .unwrap_or_else(|_| {
                    ndarray::Array2::zeros((q_shape.0 * q_shape.1, q_shape.2))
                });
            let half = self.head_dim / 2;
            let cos_1d = ndarray::Array1::from_shape_fn(half, |i| cos[i]);
            let sin_1d = ndarray::Array1::from_shape_fn(half, |i| sin[i]);
            let q_rotated = RoPE::apply(&q_2d, &cos_1d, &sin_1d, self.head_dim);
            q = q_rotated
                .into_shape(q_shape)
                .unwrap_or_else(|_| ndarray::Array3::zeros(q_shape));
        }

        // Append this token's K/V to the paged cache
        let k_flat: Vec<f32> = k.iter().copied().collect();
        let v_flat: Vec<f32> = v.iter().copied().collect();
        cache.append(seq_id, layer_idx, token_pos, &k_flat, &v_flat);

        // Batch-read ALL cached positions at once using read_layer_kv()
        // Replaces O(num_tokens) per-token read() calls with O(1) allocation per layer.
        let num_tokens = cache.num_tokens(seq_id).unwrap_or(0);
        if num_tokens == 0 {
            return Ok(Array2::zeros((batch_size, self.num_heads * self.head_dim)));
        }

        let (all_k, all_v) = match cache.read_layer_kv(seq_id, layer_idx) {
            Some(kv) => kv,
            None => return Ok(Array2::zeros((batch_size, self.num_heads * self.head_dim))),
        };

        let kv_dim = self.num_kv_heads * self.head_dim;
        let k_arr = Array2::from_shape_vec((num_tokens, kv_dim), all_k)
            .map_err(|e| TransformerError::Implementation(format!("reshape K: {}", e)))?;
        let v_arr = Array2::from_shape_vec((num_tokens, kv_dim), all_v)
            .map_err(|e| TransformerError::Implementation(format!("reshape V: {}", e)))?;

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                let kv_off = kv_h * self.head_dim;

                let q_slice = q.slice(ndarray::s![b, h, ..]);

                // Batched score: dot product with all cached K positions at once
                let k_slice = k_arr.slice(ndarray::s![.., kv_off..kv_off + self.head_dim]);
                let raw_scores: Vec<f32> = k_slice.dot(&q_slice).iter()
                    .map(|s| *s * self.head_dim_rs)
                    .collect();

                if raw_scores.is_empty() {
                    continue;
                }

                let max_score = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0;
                let mut exp_scores = Vec::with_capacity(num_tokens);
                for &s in &raw_scores {
                    let e = (s - max_score).exp();
                    exp_sum += e;
                    exp_scores.push(e);
                }

                let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };

                // Batched V aggregation: [num_tokens] · [num_tokens, head_dim] → [head_dim]
                let v_slice = v_arr.slice(ndarray::s![.., kv_off..kv_off + self.head_dim]);
                let weights = Array1::from_vec(exp_scores.iter().map(|e| e * inv_exp_sum).collect());
                let out_row_arr = weights.dot(&v_slice);

                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    output[[b, out_base + d]] = out_row_arr[d];
                }
            }
        }

        Ok(self.maybe_f16_matmul(&output, wo, &self.wo_f16))
    }
}
