use std::borrow::Cow;

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use super::rope::RoPE;

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub(crate) struct GqaGpuWeights {
    pub wq_t: nexora_autograd::gpu::GpuTensor,
    pub wk_t: nexora_autograd::gpu::GpuTensor,
    pub wv_t: nexora_autograd::gpu::GpuTensor,
    pub wo_t: nexora_autograd::gpu::GpuTensor,
}

/// Minimal paged cache interface — avoids hard dependency on the inference crate.
/// Inference crate's PagedKVCache implements this trait.
pub trait PagedCacheReader {
    /// Read KV data for a token position. Returns (k_row, v_row).
    fn read(&self, seq_id: u64, layer: usize, token_pos: usize) -> Option<(Vec<f32>, Vec<f32>)>;
    /// Number of tokens stored for a sequence.
    fn num_tokens(&self, seq_id: u64) -> Option<usize>;
    /// Append a token's KV data (single position).
    fn append(&mut self, seq_id: u64, layer: usize, token_pos: usize, k_row: &[f32], v_row: &[f32]);
}

#[derive(Debug)]
pub struct GQA {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_groups: usize,
    pub head_dim_rs: f32,
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    #[cfg(feature = "gpu")]
    pub(crate) gpu_weights: std::sync::Mutex<Option<GqaGpuWeights>>,
}

#[cfg(not(feature = "gpu"))]
impl Clone for GQA {
    fn clone(&self) -> Self {
        Self {
            num_heads: self.num_heads.clone(),
            num_kv_heads: self.num_kv_heads.clone(),
            head_dim: self.head_dim.clone(),
            num_groups: self.num_groups.clone(),
            head_dim_rs: self.head_dim_rs.clone(),
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
        }
    }
}

#[cfg(feature = "gpu")]
impl Clone for GQA {
    fn clone(&self) -> Self {
        Self {
            num_heads: self.num_heads.clone(),
            num_kv_heads: self.num_kv_heads.clone(),
            head_dim: self.head_dim.clone(),
            num_groups: self.num_groups.clone(),
            head_dim_rs: self.head_dim_rs.clone(),
            wq: self.wq.clone(),
            wk: self.wk.clone(),
            wv: self.wv.clone(),
            wo: self.wo.clone(),
            gpu_weights: std::sync::Mutex::new(None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    pub k: Array2<f32>,
    pub v: Array2<f32>,
}

impl GQA {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (head_dim as f32).sqrt().recip();

        let wq = Array2::from_shape_fn((num_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let wk = Array2::from_shape_fn((num_kv_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let wv = Array2::from_shape_fn((num_kv_heads * head_dim, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });
        let wo = Array2::from_shape_fn((hidden_size, num_heads * head_dim), |_| {
            rng.gen::<f32>() * 2.0 * scale - scale
        });

        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            num_groups: num_heads / num_kv_heads,
            head_dim_rs: 1.0 / (head_dim as f32).sqrt(),
            wq,
            wk,
            wv,
            wo,
            #[cfg(feature = "gpu")]
            gpu_weights: std::sync::Mutex::new(None),
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        cache: Option<&mut Vec<KVCacheEntry>>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Array2<f32> {
        let (batch_size, _) = x.dim();

        let q_proj = x.dot(&self.wq.t());
        let k_proj = x.dot(&self.wk.t());
        let v_proj = x.dot(&self.wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .expect("GQA: q shape mismatch");
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .expect("GQA: k shape mismatch");
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .expect("GQA: v shape mismatch");

        for b in 0..batch_size {
            let k_slice_view = k.slice(ndarray::s![b, .., ..]);
            let k_row = k_slice_view.as_slice().unwrap();
            let rotated_k = RoPE::apply_single(k_row, cos, sin, self.head_dim, 0);
            for h in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    k[[b, h, d]] = rotated_k[h * self.head_dim + d];
                }
            }
        }

        for b in 0..batch_size {
            let q_slice_view = q.slice(ndarray::s![b, .., ..]);
            let q_row = q_slice_view.as_slice().unwrap();
            let rotated_q = RoPE::apply_single(q_row, cos, sin, self.head_dim, 0);
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    q[[b, h, d]] = rotated_q[h * self.head_dim + d];
                }
            }
        }

        let (k_cached, v_cached, total_seq): (Cow<Array2<f32>>, Cow<Array2<f32>>, usize) =
            match cache {
                Some(cache) if layer_idx < cache.len() => {
                    let entry = &cache[layer_idx];
                    let seq = entry.k.shape()[0] / batch_size;
                    (Cow::Borrowed(&entry.k), Cow::Borrowed(&entry.v), seq)
                }
                _ => {
                    let kf = k
                        .into_shape((batch_size, self.num_kv_heads * self.head_dim))
                        .expect("GQA: k flat");
                    let vf = v
                        .into_shape((batch_size, self.num_kv_heads * self.head_dim))
                        .expect("GQA: v flat");
                    (Cow::Owned(kf), Cow::Owned(vf), 1)
                }
            };

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                let kv_off = kv_h * self.head_dim;

                let mut scores = Vec::with_capacity(total_seq);
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..total_seq {
                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[b, h, d]] * k_cached[[b * total_seq + t, kv_off + d]];
                    }
                    score *= self.head_dim_rs;
                    if score > max_score {
                        max_score = score;
                    }
                    scores.push(score);
                }

                let mut exp_sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }

                let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    let mut weighted = 0.0;
                    for t in 0..total_seq {
                        weighted +=
                            scores[t] * inv_exp_sum * v_cached[[b * total_seq + t, kv_off + d]];
                    }
                    output[[b, out_base + d]] = weighted;
                }
            }
        }

        output.dot(&self.wo.t())
    }

    pub fn forward_with_kv(
        &self,
        x: &Array2<f32>,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Array2<f32> {
        let (batch_size, _) = x.dim();

        let q_proj = x.dot(&self.wq.t());
        let k_proj = x.dot(&self.wk.t());
        let v_proj = x.dot(&self.wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .expect("GQA forward_with_kv: q shape");
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .expect("GQA forward_with_kv: k shape");
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .expect("GQA forward_with_kv: v shape");

        for b in 0..batch_size {
            let k_slice_view = k.slice(ndarray::s![b, .., ..]);
            let k_row = k_slice_view.as_slice().unwrap();
            let rotated_k = RoPE::apply_single(k_row, cos, sin, self.head_dim, 0);
            for h in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    k[[b, h, d]] = rotated_k[h * self.head_dim + d];
                }
            }
        }

        for b in 0..batch_size {
            let q_slice_view = q.slice(ndarray::s![b, .., ..]);
            let q_row = q_slice_view.as_slice().unwrap();
            let rotated_q = RoPE::apply_single(q_row, cos, sin, self.head_dim, 0);
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    q[[b, h, d]] = rotated_q[h * self.head_dim + d];
                }
            }
        }

        let _seq_len = if layer_idx < cache.len() {
            let entry = &cache[layer_idx];
            entry.k.shape()[0] / batch_size + 1
        } else {
            1
        };

        if layer_idx < cache.len() {
            let entry = &mut cache[layer_idx];
            let k_flat = k
                .into_shape(batch_size * self.num_kv_heads * self.head_dim)
                .expect("GQA: k_flat");
            let v_flat = v
                .into_shape(batch_size * self.num_kv_heads * self.head_dim)
                .expect("GQA: v_flat");
            let new_k = ndarray::concatenate![Axis(0), entry.k.view(), k_flat.insert_axis(Axis(0))];
            let new_v = ndarray::concatenate![Axis(0), entry.v.view(), v_flat.insert_axis(Axis(0))];
            entry.k = new_k;
            entry.v = new_v;
        } else {
            let k_flat = k
                .into_shape(batch_size * self.num_kv_heads * self.head_dim)
                .expect("GQA: k_flat new entry");
            let v_flat = v
                .into_shape(batch_size * self.num_kv_heads * self.head_dim)
                .expect("GQA: v_flat new entry");
            cache.push(KVCacheEntry {
                k: k_flat.insert_axis(Axis(0)).to_owned(),
                v: v_flat.insert_axis(Axis(0)).to_owned(),
            });
        }

        let entry = &cache[layer_idx];
        let k_cached = &entry.k;
        let v_cached = &entry.v;
        let total_seq = k_cached.shape()[0];

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                let kv_off = kv_h * self.head_dim;

                let mut scores = vec![0.0; total_seq];
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..total_seq {
                    let k_row = k_cached.row(t);
                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[b, h, d]] * k_row[kv_off + d];
                    }
                    score *= self.head_dim_rs;
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
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    let mut weighted = 0.0;
                    for t in 0..total_seq {
                        weighted += scores[t] * inv_exp_sum * v_cached[[t, kv_off + d]];
                    }
                    output[[b, out_base + d]] = weighted;
                }
            }
        }

        output.dot(&self.wo.t())
    }

    /// Forward pass using a paged KV cache (reads K/V from blocks per position).
    /// batch_size is always 1 for autoregressive generation.
    pub fn forward_with_paged(
        &self,
        x: &Array2<f32>,
        cache: &mut dyn PagedCacheReader,
        seq_id: u64,
        layer_idx: usize,
        token_pos: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Array2<f32> {
        let (batch_size, _) = x.dim();
        debug_assert_eq!(
            batch_size, 1,
            "forward_with_paged only supports batch_size=1"
        );

        let q_proj = x.dot(&self.wq.t());
        let k_proj = x.dot(&self.wk.t());
        let v_proj = x.dot(&self.wv.t());

        let mut q = q_proj
            .into_shape((batch_size, self.num_heads, self.head_dim))
            .expect("GQA paged: q shape");
        let mut k = k_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .expect("GQA paged: k shape");
        let v = v_proj
            .into_shape((batch_size, self.num_kv_heads, self.head_dim))
            .expect("GQA paged: v shape");

        // RoPE for K
        for b in 0..batch_size {
            let k_slice_view = k.slice(ndarray::s![b, .., ..]);
            let k_row = k_slice_view.as_slice().unwrap();
            let rotated_k = RoPE::apply_single(k_row, cos, sin, self.head_dim, 0);
            for h in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    k[[b, h, d]] = rotated_k[h * self.head_dim + d];
                }
            }
        }

        // RoPE for Q
        for b in 0..batch_size {
            let q_slice_view = q.slice(ndarray::s![b, .., ..]);
            let q_row = q_slice_view.as_slice().unwrap();
            let rotated_q = RoPE::apply_single(q_row, cos, sin, self.head_dim, 0);
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    q[[b, h, d]] = rotated_q[h * self.head_dim + d];
                }
            }
        }

        // Append this token's K/V to the paged cache
        let k_flat: Vec<f32> = k
            .view()
            .into_shape((batch_size * self.num_kv_heads * self.head_dim,))
            .expect("GQA paged: k flat")
            .iter()
            .copied()
            .collect();
        let v_flat: Vec<f32> = v
            .view()
            .into_shape((batch_size * self.num_kv_heads * self.head_dim,))
            .expect("GQA paged: v flat")
            .iter()
            .copied()
            .collect();
        cache.append(seq_id, layer_idx, token_pos, &k_flat, &v_flat);

        // Read all cached positions for attention computation
        let num_tokens = cache.num_tokens(seq_id).unwrap_or(0);
        if num_tokens == 0 {
            return Array2::zeros((batch_size, self.num_heads * self.head_dim));
        }

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);
                let kv_off = kv_h * self.head_dim;

                let mut max_score = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(num_tokens);
                for t in 0..num_tokens {
                    let (k_row, _) = match cache.read(seq_id, layer_idx, t) {
                        Some(r) => r,
                        None => continue,
                    };
                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[b, h, d]] * k_row[kv_off + d];
                    }
                    score *= self.head_dim_rs;
                    if score > max_score {
                        max_score = score;
                    }
                    scores.push(score);
                }

                if scores.is_empty() {
                    continue;
                }

                let mut exp_sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }

                let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
                let out_base = h * self.head_dim;
                for d in 0..self.head_dim {
                    let mut weighted = 0.0;
                    for (t, &s) in scores.iter().enumerate() {
                        let (_, v_row) = match cache.read(seq_id, layer_idx, t) {
                            Some(r) => r,
                            None => continue,
                        };
                        weighted += s * inv_exp_sum * v_row[kv_off + d];
                    }
                    output[[b, out_base + d]] = weighted;
                }
            }
        }

        output.dot(&self.wo.t())
    }

    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &self,
        x_gpu: &nexora_autograd::gpu::GpuTensor,
        cache: &mut Vec<KVCacheEntry>,
        layer_idx: usize,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuError, GpuTensor};
        use ndarray::ArrayD;

        let ctx = GpuContext::global()?;
        let batch_size = 1;

        // 1. Lazy-init cached GPU weights (pre-transposed for matmul)
        let mut guard = self.gpu_weights.lock().unwrap();
        if guard.is_none() {
            let mk = |arr: &Array2<f32>| -> Result<GpuTensor, GpuError> {
                let shape = vec![arr.shape()[0], arr.shape()[1]];
                let data = arr.as_slice().ok_or_else(|| {
                    GpuError::Unsupported("non-contiguous weight".into())
                })?.to_vec();
                Ok(GpuTensor::from_cpu(
                    &ArrayD::from_shape_vec(shape, data).map_err(|e| GpuError::Unsupported(e.to_string()))?
                )?)
            };
            let ctx_ref = GpuContext::global()?;
            let wq = mk(&self.wq)?;
            let wk = mk(&self.wk)?;
            let wv = mk(&self.wv)?;
            let wo = mk(&self.wo)?;
            *guard = Some(GqaGpuWeights {
                wq_t: ctx_ref.transpose(&wq)?,
                wk_t: ctx_ref.transpose(&wk)?,
                wv_t: ctx_ref.transpose(&wv)?,
                wo_t: ctx_ref.transpose(&wo)?,
            });
        }
        let cached = guard.as_ref().unwrap();

        // 2. QKV projection on GPU: matmul(x, W^T) with cached pre-transposed weights
        let q_proj = ctx.matmul(x_gpu, &cached.wq_t)?;
        let k_proj = ctx.matmul(x_gpu, &cached.wk_t)?;
        let v_proj = ctx.matmul(x_gpu, &cached.wv_t)?;

        // 3. Upload cos/sin for RoPE
        let half = cos.len();
        let cos_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, half], cos.to_vec()).map_err(|e| GpuError::Unsupported(e.to_string()))?
        )?;
        let sin_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, half], sin.to_vec()).map_err(|e| GpuError::Unsupported(e.to_string()))?
        )?;

        // Apply RoPE on GPU for Q and K
        let q_rotated = ctx.rotary_embedding(&q_proj, &cos_gpu, &sin_gpu, self.head_dim as u32)?;
        let k_rotated = ctx.rotary_embedding(&k_proj, &cos_gpu, &sin_gpu, self.head_dim as u32)?;

        // 4. Reshape Q to [1, num_heads, 1, head_dim] for fused_attention
        let q_4d = q_rotated.reshape(vec![batch_size, self.num_heads, 1, self.head_dim])?;

        // 5. Download K, V (1 token each) to CPU, append to cache
        let k_cpu_arr = k_rotated.to_cpu();
        let k_flat: Vec<f32> = k_cpu_arr.iter().copied().collect();
        let k_cpu = Array2::from_shape_vec(
            (batch_size, self.num_kv_heads * self.head_dim),
            k_flat,
        ).map_err(|e| GpuError::Unsupported(e.to_string()))?;
        let v_cpu_arr = v_proj.to_cpu();
        let v_flat: Vec<f32> = v_cpu_arr.iter().copied().collect();
        let v_cpu = Array2::from_shape_vec(
            (batch_size, self.num_kv_heads * self.head_dim),
            v_flat,
        ).map_err(|e| GpuError::Unsupported(e.to_string()))?;

        if layer_idx < cache.len() {
            let entry = &mut cache[layer_idx];
            let new_k = ndarray::concatenate![ndarray::Axis(0), entry.k.view(), k_cpu.view()];
            let new_v = ndarray::concatenate![ndarray::Axis(0), entry.v.view(), v_cpu.view()];
            entry.k = new_k;
            entry.v = new_v;
        } else {
            cache.push(KVCacheEntry {
                k: k_cpu.to_owned(),
                v: v_cpu.to_owned(),
            });
        }

        // 6. Upload full KV cache to GPU, repeat heads for GQA if needed
        let entry = &cache[layer_idx];
        let total_seq = entry.k.shape()[0];

        // Build K/V with head repeating for GQA:
        // If num_heads != num_kv_heads, repeat K/V heads so fused_attention
        // sees [B, num_heads, S, D] for both Q and K/V.
        let eff_heads = if self.num_heads > self.num_kv_heads {
            self.num_heads
        } else {
            self.num_kv_heads
        };

        let k_slice = entry.k.as_slice().unwrap();
        let v_slice = entry.v.as_slice().unwrap();
        let mut k_4d_data = vec![0.0; total_seq * eff_heads * self.head_dim];
        let mut v_4d_data = vec![0.0; total_seq * eff_heads * self.head_dim];

        let groups = self.num_heads / self.num_kv_heads.max(1);
        for s in 0..total_seq {
            for h in 0..eff_heads {
                let kv_h = if self.num_heads > self.num_kv_heads {
                    (h / groups).min(self.num_kv_heads - 1)
                } else {
                    h
                };
                let cache_off = s * (self.num_kv_heads * self.head_dim) + kv_h * self.head_dim;
                let target_off = h * total_seq * self.head_dim + s * self.head_dim;
                for d in 0..self.head_dim {
                    k_4d_data[target_off + d] = k_slice[cache_off + d];
                    v_4d_data[target_off + d] = v_slice[cache_off + d];
                }
            }
        }

        let kv_shape = vec![1, eff_heads, total_seq, self.head_dim];
        let k_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(kv_shape.clone(), k_4d_data)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?
        )?;
        let v_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(kv_shape, v_4d_data)
                .map_err(|e| GpuError::Unsupported(e.to_string()))?
        )?;

        // 7. Fused attention on GPU: O(n^2) on GPU instead of CPU
        let attn_output_4d = ctx.fused_attention(&q_4d, &k_gpu, &v_gpu, self.head_dim_rs, false)?;

        // 8. Reshape attention output: [1, num_heads, 1, head_dim] -> [1, num_heads * head_dim]
        let attn_flat = attn_output_4d.reshape(vec![batch_size, self.num_heads * self.head_dim])?;

        // 9. Output projection on GPU: matmul(attn_flat, wo^T) with cached weight
        ctx.matmul(&attn_flat, &cached.wo_t)
    }
}
