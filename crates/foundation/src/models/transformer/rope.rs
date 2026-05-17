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
        let inv_freq: Array1<f32> = Array1::from_shape_fn(half, |i| {
            1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)
        });
        Self { inv_freq, max_seq_len, head_dim }
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
        cos: &Array1<f32>,
        sin: &Array1<f32>,
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
}
