use crate::multimodal::error::Result;
use crate::multimodal::types::*;
use ndarray::{Array1, Array2, ArrayD};
use rand::Rng;

struct FramePatchMLP {
    pub(crate) w1: Array2<f32>,
    pub(crate) b1: Array1<f32>,
    pub(crate) w2: Array2<f32>,
    pub(crate) b2: Array1<f32>,
}

impl FramePatchMLP {
    fn new(patch_pixels: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / patch_pixels as f32).sqrt();
        let scale2 = (2.0 / hidden_dim as f32).sqrt();
        Self {
            w1: Array2::from_shape_fn((patch_pixels, hidden_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale1 - scale1
            }),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::from_shape_fn((hidden_dim, output_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale2 - scale2
            }),
            b2: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let hidden = x.dot(&self.w1) + &self.b1;
        let gelu = hidden.mapv(|v| {
            v * 0.5 * (1.0 + (v * 0.7978845608 * (1.0 + 0.044715 * v * v)).tanh())
        });
        gelu.dot(&self.w2) + &self.b2
    }

    #[cfg(feature = "gpu")]
    fn forward_batched_gpu(&self, inputs: &[Array1<f32>]) -> Option<Vec<Array1<f32>>> {
        use crate::multimodal::gpu_compute;
        let batch = inputs.len();
        if batch == 0 { return None; }
        let input_dim = inputs[0].len();
        let hidden_dim = self.w1.shape()[1];
        let output_dim = self.w2.shape()[1];
        let flat_input: Vec<f32> = inputs.iter().flat_map(|i| i.iter().copied()).collect();
        let w1_slice: Vec<f32> = self.w1.iter().copied().collect();
        let b1_slice: Vec<f32> = self.b1.iter().copied().collect();
        let w2_slice: Vec<f32> = self.w2.iter().copied().collect();
        let b2_slice: Vec<f32> = self.b2.iter().copied().collect();
        let result = gpu_compute::try_gpu_mlp_forward(
            &flat_input, &w1_slice, &b1_slice, &w2_slice, &b2_slice,
            batch, input_dim, hidden_dim, output_dim,
        )?;
        Some(result.chunks(output_dim).map(|c| Array1::from_vec(c.to_vec())).collect())
    }
}

struct TemporalAttention {
    num_heads: usize,
    pub(crate) wq: Array2<f32>,
    pub(crate) wk: Array2<f32>,
    pub(crate) wv: Array2<f32>,
    pub(crate) wo: Array2<f32>,
}

impl TemporalAttention {
    fn new(embed_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let _head_dim = embed_dim / num_heads;
        let scale = (2.0 / embed_dim as f32).sqrt();
        Self {
            num_heads,
            wq: Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            }),
            wk: Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            }),
            wv: Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            }),
            wo: Array2::from_shape_fn((embed_dim, embed_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            }),
        }
    }

    fn forward(&self, x: &[f32], seq_len: usize, embed_dim: usize) -> Vec<f32> {
        // Try GPU-accelerated path first
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_gpu(x, seq_len, embed_dim) {
            return result;
        }

        // CPU fallback
        let head_dim = embed_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * embed_dim];

        let q = matmul_flat(x, &self.wq, seq_len, embed_dim, embed_dim);
        let k = matmul_flat(x, &self.wk, seq_len, embed_dim, embed_dim);
        let v = matmul_flat(x, &self.wv, seq_len, embed_dim, embed_dim);

        for h in 0..self.num_heads {
            let hd = h * head_dim;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[i * embed_dim + hd + d] * k[j * embed_dim + hd + d];
                    }
                    score *= scale;
                    let exp_score = score.exp();
                    let mut sum_exp = 0.0f32;
                    for kk in 0..seq_len {
                        let mut s = 0.0f32;
                        for d in 0..head_dim {
                            s += q[i * embed_dim + hd + d] * k[kk * embed_dim + hd + d];
                        }
                        sum_exp += (s * scale).exp();
                    }
                    let attn = if sum_exp > 0.0 { exp_score / sum_exp } else { 0.0 };
                    for d in 0..head_dim {
                        output[i * embed_dim + hd + d] += attn * v[j * embed_dim + hd + d];
                    }
                }
            }
        }

        matmul_flat(&output, &self.wo, seq_len, embed_dim, embed_dim)
    }

    #[cfg(feature = "gpu")]
    fn forward_gpu(&self, x: &[f32], seq_len: usize, embed_dim: usize) -> Option<Vec<f32>> {
        use crate::multimodal::gpu_compute;
        let nh = self.num_heads;
        let dh = embed_dim / nh;
        let wq_slice: Vec<f32> = self.wq.iter().copied().collect();
        let wk_slice: Vec<f32> = self.wk.iter().copied().collect();
        let wv_slice: Vec<f32> = self.wv.iter().copied().collect();
        let wo_slice: Vec<f32> = self.wo.iter().copied().collect();

        // QKV projections via GPU matmul — submit all 3 asynchronously
        let q_async = gpu_compute::try_gpu_matmul_async(x, &wq_slice, 1, seq_len, embed_dim, embed_dim)?;
        let k_async = gpu_compute::try_gpu_matmul_async(x, &wk_slice, 1, seq_len, embed_dim, embed_dim)?;
        let v_async = gpu_compute::try_gpu_matmul_async(x, &wv_slice, 1, seq_len, embed_dim, embed_dim)?;
        let q = q_async.recv().ok()?;
        let k = k_async.recv().ok()?;
        let v = v_async.recv().ok()?;

        // Reshape to [1, nh, S, dh]
        let mut q_4d = vec![0.0f32; nh * seq_len * dh];
        let mut k_4d = vec![0.0f32; nh * seq_len * dh];
        let mut v_4d = vec![0.0f32; nh * seq_len * dh];
        for h in 0..nh {
            for i in 0..seq_len {
                for d in 0..dh {
                    let src = i * embed_dim + h * dh + d;
                    let dst = h * seq_len * dh + i * dh + d;
                    q_4d[dst] = q[src];
                    k_4d[dst] = k[src];
                    v_4d[dst] = v[src];
                }
            }
        }

        // Fused attention
        let attended = gpu_compute::try_gpu_attention(&q_4d, &k_4d, &v_4d, 1, seq_len, nh, dh, false)?;

        // attended: [1, nh, S, dh] -> concat heads: [S, d]
        let mut concat = vec![0.0f32; seq_len * embed_dim];
        for h in 0..nh {
            for i in 0..seq_len {
                for d in 0..dh {
                    let src = h * seq_len * dh + i * dh + d;
                    let dst = i * embed_dim + h * dh + d;
                    concat[dst] = attended[src];
                }
            }
        }

        // Output projection via GPU
        gpu_compute::try_gpu_matmul(&concat, &wo_slice, 1, seq_len, embed_dim, embed_dim)
    }
}

fn matmul_flat(a: &[f32], w: &Array2<f32>, m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    let gpu_result = {
        let w_slice: Vec<f32> = w.iter().copied().collect();
        crate::multimodal::gpu_compute::try_gpu_matmul(a, &w_slice, 1, m, k, n)
    };
    #[cfg(feature = "gpu")]
    if let Some(result) = gpu_result {
        return result;
    }

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * w[[kk, j]];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

pub struct VideoEncoder {
    config: crate::multimodal::config::VideoEncoderConfig,
    model_loaded: bool,
    _temporal_dim: usize,
    patch_projection: FramePatchMLP,
    temporal_attention: TemporalAttention,
}

impl VideoEncoder {
    pub fn new(config: crate::multimodal::config::VideoEncoderConfig) -> Result<Self> {
        let patch_size = 16;
        let channels = 3;
        let patch_pixels = patch_size * patch_size * channels;
        let hidden_dim = config.output_dim.max(64);
        let output_dim = config.output_dim;
        let num_frames = config.num_frames;
        let num_heads = config.output_dim.max(4) / 4;
        let num_heads = num_heads.max(2).next_power_of_two().min(output_dim);

        Ok(Self {
            _temporal_dim: num_frames,
            model_loaded: false,
            config,
            patch_projection: FramePatchMLP::new(patch_pixels, hidden_dim, output_dim),
            temporal_attention: TemporalAttention::new(output_dim, num_heads),
        })
    }

    pub fn load_model(&mut self) -> Result<()> {
        self.model_loaded = true;
        Ok(())
    }

    pub fn encode(&mut self, input: &VideoInput) -> Result<ArrayD<f32>> {
        if !self.model_loaded {
            self.load_model()?;
        }

        let sampled_frames = self.sample_frames(&input.frames)?;

        let mut frame_features = Vec::new();
        for frame in &sampled_frames {
            let frame_feature = self.encode_frame(frame)?;
            frame_features.push(frame_feature);
        }

        let temporal_features = self.apply_temporal_modeling(&frame_features)?;
        Ok(temporal_features)
    }

    fn sample_frames(&self, frames: &[ImageInput]) -> Result<Vec<ImageInput>> {
        if frames.len() <= self.config.num_frames {
            return Ok(frames.to_vec());
        }
        let mut sampled = Vec::new();
        let step = frames.len() / self.config.num_frames;
        for i in 0..self.config.num_frames {
            let frame_idx = i * step;
            if frame_idx < frames.len() {
                sampled.push(frames[frame_idx].clone());
            }
        }
        Ok(sampled)
    }

    pub fn encode_frame(&self, frame: &ImageInput) -> Result<ArrayD<f32>> {
        let patch_size = 16;
        let cols = (frame.width / patch_size).max(1);
        let rows = (frame.height / patch_size).max(1);
        let num_patches = cols * rows;
        let embed_dim = self.config.output_dim;
        let channels = 3;

        let raw: Vec<f32> = frame.data.iter().map(|&b| b as f32 / 255.0).collect();
        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];
        let pixels: Vec<f32> = raw
            .chunks(frame.data.len() / (frame.width * frame.height).max(1) * channels)
            .flat_map(|pixel| {
                pixel.iter().enumerate().map(|(c, &v)| {
                    if c < 3 { (v - mean[c]) / std[c] } else { v }
                })
            })
            .collect();

        let channels_in_data = if frame.data.len() >= frame.width * frame.height * 3 {
            3
        } else {
            1
        };

        // Collect all patch vectors
        let mut patch_vecs: Vec<Array1<f32>> = Vec::with_capacity(num_patches);
        for p in 0..num_patches {
            let patch_col = p % cols;
            let patch_row = p / cols;
            let mut patch_vec = Vec::with_capacity(patch_size * patch_size * 3);

            for py in 0..patch_size.min(frame.height - patch_row * patch_size) {
                for px in 0..patch_size.min(frame.width - patch_col * patch_size) {
                    let idx = ((patch_row * patch_size + py) * frame.width
                        + (patch_col * patch_size + px))
                        * channels_in_data;
                    if channels_in_data >= 3 && idx + 2 < pixels.len() {
                        patch_vec.push(pixels[idx]);
                        patch_vec.push(pixels[idx + 1]);
                        patch_vec.push(pixels[idx + 2]);
                    } else if idx < pixels.len() {
                        let v = pixels[idx];
                        patch_vec.push(v);
                        patch_vec.push(v);
                        patch_vec.push(v);
                    } else {
                        patch_vec.push(0.0);
                        patch_vec.push(0.0);
                        patch_vec.push(0.0);
                    }
                }
            }

            while patch_vec.len() < patch_size * patch_size * 3 {
                patch_vec.push(0.0);
            }
            patch_vecs.push(Array1::from_vec(patch_vec));
        }

        // Try GPU batched MLP first, fall back to per-patch CPU
        let mut features = vec![0.0f32; num_patches * embed_dim];
        #[cfg(feature = "gpu")]
        if let Some(gpu_results) = self.patch_projection.forward_batched_gpu(&patch_vecs) {
            for (p, projected) in gpu_results.iter().enumerate() {
                for d in 0..embed_dim {
                    features[p * embed_dim + d] = projected[d];
                }
            }
        } else {
            for (p, patch_arr) in patch_vecs.iter().enumerate() {
                let projected = self.patch_projection.forward(patch_arr);
                for d in 0..embed_dim {
                    features[p * embed_dim + d] = projected[d];
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        for (p, patch_arr) in patch_vecs.iter().enumerate() {
            let projected = self.patch_projection.forward(patch_arr);
            for d in 0..embed_dim {
                features[p * embed_dim + d] = projected[d];
            }
        }

        let shape = vec![num_patches, embed_dim];
        Ok(ArrayD::from_shape_vec(shape, features)?)
    }

    fn apply_temporal_modeling(&self, frame_features: &[ArrayD<f32>]) -> Result<ArrayD<f32>> {
        if frame_features.is_empty() {
            return Err(crate::multimodal::error::CaffeineError::encoder(
                "No frame features to process",
            ));
        }

        let num_frames = frame_features.len();
        let first_shape = frame_features[0].shape();
        let seq_len = first_shape[0];
        let embed_dim = first_shape[1];

        let mut temporal_data = vec![0.0f32; num_frames * seq_len * embed_dim];

        for (t, frame_feature) in frame_features.iter().enumerate() {
            for i in 0..seq_len {
                for d in 0..embed_dim {
                    if let Some(&val) = frame_feature.get([i, d]) {
                        let pos_enc_sin = (t as f32 / 10000.0_f32.powf(d as f32 / embed_dim as f32)).sin();
                        let pos_enc_cos = (t as f32 / 10000.0_f32.powf(d as f32 / embed_dim as f32)).cos();
                        let temporal_encoding = if d % 2 == 0 { pos_enc_sin } else { pos_enc_cos };
                        let output_idx = t * seq_len * embed_dim + i * embed_dim + d;
                        temporal_data[output_idx] = val + temporal_encoding;
                    }
                }
            }
        }

        let attended = self.temporal_attention.forward(&temporal_data, num_frames * seq_len, embed_dim);

        let shape = vec![1, num_frames * seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(shape, attended)?)
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    pub fn config(&self) -> &crate::multimodal::config::VideoEncoderConfig {
        &self.config
    }

    /// Collect all trainable weights for checkpoint
    pub(crate) fn collect_weights(&self) -> Vec<(String, ndarray::ArrayD<f32>)> {
        let mut weights = Vec::new();
        weights.push(("video_encoder.patch_proj.w1".to_string(), self.patch_projection.w1.clone().into_dyn()));
        weights.push(("video_encoder.patch_proj.b1".to_string(), self.patch_projection.b1.clone().into_dyn()));
        weights.push(("video_encoder.patch_proj.w2".to_string(), self.patch_projection.w2.clone().into_dyn()));
        weights.push(("video_encoder.patch_proj.b2".to_string(), self.patch_projection.b2.clone().into_dyn()));
        weights.push(("video_encoder.temporal_attn.wq".to_string(), self.temporal_attention.wq.clone().into_dyn()));
        weights.push(("video_encoder.temporal_attn.wk".to_string(), self.temporal_attention.wk.clone().into_dyn()));
        weights.push(("video_encoder.temporal_attn.wv".to_string(), self.temporal_attention.wv.clone().into_dyn()));
        weights.push(("video_encoder.temporal_attn.wo".to_string(), self.temporal_attention.wo.clone().into_dyn()));
        weights
    }
}
