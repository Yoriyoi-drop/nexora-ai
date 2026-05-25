use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct RoPE {
    pub inv_freq: Array1<f32>,
    pub max_seq_len: usize,
    pub head_dim: usize,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let inv_freq: Array1<f32> =
            Array1::from_shape_fn(half, |i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32));
        Self {
            inv_freq,
            max_seq_len,
            head_dim,
        }
    }

    pub fn precompute_freqs_cis(&self) -> (Array2<f32>, Array2<f32>) {
        let half = self.head_dim / 2;
        let mut cos = Array2::zeros((self.max_seq_len, half));
        let mut sin = Array2::zeros((self.max_seq_len, half));
        for pos in 0..self.max_seq_len {
            for i in 0..half {
                let freq = self.inv_freq[i];
                cos[[pos, i]] = (pos as f32 * freq).cos();
                sin[[pos, i]] = (pos as f32 * freq).sin();
            }
        }
        (cos, sin)
    }

    pub fn apply(
        x: &Array2<f32>,
        cos: &Array1<f32>,
        sin: &Array1<f32>,
        head_dim: usize,
    ) -> Array2<f32> {
        // Pure CPU forward path.
        // GPU-resident execution uses `apply_gpu` (no per-layer readback).
        let (batch_size, dim) = x.dim();
        let half = head_dim / 2;
        let mut output = x.clone();

        for b in 0..batch_size {
            for i in 0..half.min(dim / 2) {
                let idx1 = i;
                let idx2 = i + half;
                if idx2 >= dim {
                    break;
                }
                let c = cos.get(i).copied().unwrap_or(1.0);
                let s = sin.get(i).copied().unwrap_or(0.0);
                let v1 = x[[b, idx1]];
                let v2 = x[[b, idx2]];
                output[[b, idx1]] = v1 * c - v2 * s;
                output[[b, idx2]] = v1 * s + v2 * c;
            }
        }

        output
    }

    pub fn apply_single(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        head_dim: usize,
        pos: usize,
    ) -> Array1<f32> {
        let dim = x.len();
        let half = head_dim / 2;
        let mut output = ndarray::Array::from(x.to_vec());

        for i in 0..half.min(dim / 2) {
            let idx1 = i;
            let idx2 = i + half;
            if idx2 >= dim {
                break;
            }
            let c = cos.get(pos * half + i).copied().unwrap_or(1.0);
            let s = sin.get(pos * half + i).copied().unwrap_or(0.0);
            let v1 = x[idx1];
            let v2 = x[idx2];
            output[idx1] = v1 * c - v2 * s;
            output[idx2] = v1 * s + v2 * c;
        }

        output
    }

    /// Apply RoPE on GPU: uses rotary_embedding WGSL kernel.
    /// x: [total_rows, head_dim] on GPU, cos/sin: [half] on CPU for current position.
    /// Returns rotated tensor on GPU.
    #[cfg(feature = "gpu")]
    pub fn apply_gpu(
        &self,
        x: &nexora_autograd::gpu::GpuTensor,
        cos_cpu: &Array1<f32>,
        sin_cpu: &Array1<f32>,
    ) -> Result<nexora_autograd::gpu::GpuTensor, nexora_autograd::gpu::GpuError> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()?;
        let half_vec = vec![cos_cpu.len()];
        let cos_arr = ndarray::ArrayD::from_shape_vec(half_vec.clone(), cos_cpu.to_vec())
            .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
        let cos_gpu = GpuTensor::from_cpu(&cos_arr)?;
        let sin_arr = ndarray::ArrayD::from_shape_vec(half_vec, sin_cpu.to_vec())
            .map_err(|e| nexora_autograd::gpu::GpuError::Unsupported(e.to_string()))?;
        let sin_gpu = GpuTensor::from_cpu(&sin_arr)?;
        ctx.rotary_embedding(x, &cos_gpu, &sin_gpu, self.head_dim as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rope_new() {
        let rope = RoPE::new(64, 128, 10000.0);
        assert_eq!(rope.head_dim, 64);
        assert_eq!(rope.max_seq_len, 128);
        assert_eq!(rope.inv_freq.len(), 32);
        assert!(rope.inv_freq[0] > rope.inv_freq[31]);
    }

    #[test]
    fn test_precompute_freqs_cis_shape() {
        let rope = RoPE::new(64, 128, 10000.0);
        let (cos, sin) = rope.precompute_freqs_cis();
        assert_eq!(cos.dim(), (128, 32));
        assert_eq!(sin.dim(), (128, 32));
    }

    #[test]
    fn test_precompute_freqs_cis_values() {
        let rope = RoPE::new(4, 1, 10000.0);
        let (cos, sin) = rope.precompute_freqs_cis();
        assert!((cos[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((sin[[0, 0]] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_preserves_magnitude() {
        let half = 4;
        let head_dim = half * 2;
        let rope = RoPE::new(head_dim, 1, 10000.0);
        let (all_cos, all_sin) = rope.precompute_freqs_cis();
        let cos_1d = all_cos.row(0).to_owned();
        let sin_1d = all_sin.row(0).to_owned();
        let x = array![[0.5, -0.2, 0.3, 0.8, -0.1, 0.6, -0.7, 0.4]];
        let out = RoPE::apply(&x, &cos_1d, &sin_1d, head_dim);
        let in_norm = x.iter().map(|v| v * v).sum::<f32>();
        let out_norm = out.iter().map(|v| v * v).sum::<f32>();
        assert!(
            (in_norm - out_norm).abs() < 1e-5,
            "RoPE should preserve magnitude"
        );
    }

    #[test]
    fn test_apply_single_shape() {
        let rope = RoPE::new(8, 10, 10000.0);
        let (cos, sin) = rope.precompute_freqs_cis();
        let cos_flat: Vec<f32> = cos.iter().copied().collect();
        let sin_flat: Vec<f32> = sin.iter().copied().collect();
        let x = vec![1.0; 8];
        let out = RoPE::apply_single(&x, &cos_flat, &sin_flat, 8, 3);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_apply_single_matches_apply_batch1() {
        let half = 4;
        let head_dim = half * 2;
        let rope = RoPE::new(head_dim, 5, 10000.0);
        let (cos, sin) = rope.precompute_freqs_cis();
        let cos_flat: Vec<f32> = cos.iter().copied().collect();
        let sin_flat: Vec<f32> = sin.iter().copied().collect();
        let x_vec = vec![0.3, -0.5, 0.7, -0.9, 1.1, -1.3, 1.5, -1.7];
        let pos = 3;
        let single = RoPE::apply_single(&x_vec, &cos_flat, &sin_flat, head_dim, pos);
        let cos_row = cos.row(pos).to_owned();
        let sin_row = sin.row(pos).to_owned();
        let x_arr: [f32; 8] = x_vec.clone().try_into().unwrap();
        let x_2d = ndarray::arr2(&[x_arr]);
        let batch = RoPE::apply(&x_2d, &cos_row, &sin_row, head_dim);
        for j in 0..head_dim {
            assert!((single[j] - batch[[0, j]]).abs() < 1e-5, "mismatch at {j}");
        }
    }

    #[test]
    fn test_apply_zero_input() {
        let rope = RoPE::new(8, 1, 10000.0);
        let (cos, sin) = rope.precompute_freqs_cis();
        let cos_1d = cos.row(0).to_owned();
        let sin_1d = sin.row(0).to_owned();
        let x = array![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let out = RoPE::apply(&x, &cos_1d, &sin_1d, 8);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_apply_single_bounds() {
        let rope = RoPE::new(4, 1, 10000.0);
        let (cos, sin) = rope.precompute_freqs_cis();
        let cos_flat: Vec<f32> = cos.iter().copied().collect();
        let sin_flat: Vec<f32> = sin.iter().copied().collect();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = RoPE::apply_single(&x, &cos_flat, &sin_flat, 4, 0);
        assert_eq!(out.len(), 4);
    }
}
