use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Array1<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = Array1::from_shape_fn(hidden_size, |_| 1.0);
        Self { weight, eps }
    }

    pub fn from_weights(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
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
        x.iter().zip(self.weight.iter()).map(|(&v, &w)| (v / rms) * w).collect()
    }
}
