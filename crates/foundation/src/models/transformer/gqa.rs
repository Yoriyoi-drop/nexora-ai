use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use super::rope::RoPE;

#[derive(Debug, Clone)]
pub struct GQA {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_groups: usize,
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
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
            wq,
            wk,
            wv,
            wo,
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

        let q = q_proj.into_shape((batch_size, self.num_heads, self.head_dim)).unwrap();
        let mut k = k_proj.into_shape((batch_size, self.num_kv_heads, self.head_dim)).unwrap();
        let mut v = v_proj.into_shape((batch_size, self.num_kv_heads, self.head_dim)).unwrap();

        for b in 0..batch_size {
            let k_row = k.slice(ndarray::s![b, .., ..]).to_owned().into_shape(self.num_kv_heads * self.head_dim).unwrap();
            let rotated_k = RoPE::apply_single(&k_row, cos, sin, self.head_dim, 0);
            let rotated_k = rotated_k.into_shape((self.num_kv_heads, self.head_dim)).unwrap();
            for h in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    k[[b, h, d]] = rotated_k[[h, d]];
                }
            }
        }

        let (k_cached, v_cached) = if let Some(cache) = cache {
            let entry = if layer_idx < cache.len() {
                &cache[layer_idx]
            } else {
                return x.to_owned();
            };
            (&entry.k, &entry.v)
        } else {
            return x.to_owned();
        };

        let total_seq = 1;

        let mut output = Array2::zeros((batch_size, self.num_heads * self.head_dim));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let kv_h = (h / self.num_groups).min(self.num_kv_heads - 1);

                let mut scores = Vec::with_capacity(total_seq);
                let mut max_score = f32::NEG_INFINITY;

                for t in 0..total_seq {
                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[b, h, d]] * k_cached[[b * total_seq + t, kv_h * self.head_dim + d]];
                    }
                    score /= (self.head_dim as f32).sqrt();
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

                for d in 0..self.head_dim {
                    let mut weighted = 0.0;
                    for t in 0..total_seq {
                        let attn = scores[t] / exp_sum;
                        weighted += attn * v_cached[[b * total_seq + t, kv_h * self.head_dim + d]];
                    }
                    output[[b, h * self.head_dim + d]] = weighted;
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

        let q = q_proj.into_shape((batch_size, self.num_heads, self.head_dim)).unwrap();
        let mut k = k_proj.into_shape((batch_size, self.num_kv_heads, self.head_dim)).unwrap();
        let mut v = v_proj.into_shape((batch_size, self.num_kv_heads, self.head_dim)).unwrap();

        for b in 0..batch_size {
            let k_row = k.slice(ndarray::s![b, .., ..]).to_owned()
                .into_shape(self.num_kv_heads * self.head_dim).unwrap();
            let rotated_k = RoPE::apply_single(&k_row, cos, sin, self.head_dim, 0);
            let rotated_k = rotated_k.into_shape((self.num_kv_heads, self.head_dim)).unwrap();
            for h in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    k[[b, h, d]] = rotated_k[[h, d]];
                }
            }
        }

        for b in 0..batch_size {
            let q_row = q.slice(ndarray::s![b, .., ..]).to_owned()
                .into_shape(self.num_heads * self.head_dim).unwrap();
            let rotated_q = RoPE::apply_single(&q_row, cos, sin, self.head_dim, 0);
            let rotated_q = rotated_q.into_shape((self.num_heads, self.head_dim)).unwrap();
        }

        let seq_len = if layer_idx < cache.len() {
            let entry = &cache[layer_idx];
            entry.k.shape()[0] / batch_size + 1
        } else {
            1
        };

        if layer_idx < cache.len() {
            let entry = &mut cache[layer_idx];
            let k_flat = k.into_shape(batch_size * self.num_kv_heads * self.head_dim).unwrap();
            let v_flat = v.into_shape(batch_size * self.num_kv_heads * self.head_dim).unwrap();
            let new_k = ndarray::concatenate![Axis(0), entry.k.view(), k_flat.insert_axis(Axis(0))];
            let new_v = ndarray::concatenate![Axis(0), entry.v.view(), v_flat.insert_axis(Axis(0))];
            entry.k = new_k;
            entry.v = new_v;
        } else {
            let k_flat = k.into_shape(batch_size * self.num_kv_heads * self.head_dim).unwrap();
            let v_flat = v.into_shape(batch_size * self.num_kv_heads * self.head_dim).unwrap();
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

                let mut scores = Vec::with_capacity(total_seq);
                let mut max_score = f32::NEG_INFINITY;

                for t in 0..total_seq {
                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[b, h, d]] * k_cached[[t, kv_h * self.head_dim + d]];
                    }
                    score /= (self.head_dim as f32).sqrt();
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

                for d in 0..self.head_dim {
                    let mut weighted = 0.0;
                    for t in 0..total_seq {
                        let attn = scores[t] / exp_sum;
                        weighted += attn * v_cached[[t, kv_h * self.head_dim + d]];
                    }
                    output[[b, h * self.head_dim + d]] = weighted;
                }
            }
        }

        output.dot(&self.wo.t())
    }
}
