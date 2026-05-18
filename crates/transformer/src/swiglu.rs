use ndarray::Array2;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SwiGLU {
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
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

        Self { w1, w2, w3 }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let gate = x.dot(&self.w1.t());
        let hidden = x.dot(&self.w3.t());

        let (rows, cols) = gate.dim();
        let mut gated = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let v = gate[[i, j]];
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                gated[[i, j]] = v * sigmoid * hidden[[i, j]];
            }
        }

        gated.dot(&self.w2.t())
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
