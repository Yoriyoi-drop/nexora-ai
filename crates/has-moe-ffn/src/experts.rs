//! Expert modules for HAS-MoE-FFN

use serde::{Deserialize, Serialize};

/// GELU activation function (standalone, for sharing across modules)
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let x_cubed = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
    x * 0.5 * (1.0 + tanh_arg.tanh())
}

/// Expert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_dropout: bool,
    pub dropout_rate: f32,
}

/// Individual expert in the MoE system
pub struct Expert {
    config: ExpertConfig,
    // Feed-forward network weights
    fc1_weights: Vec<Vec<f32>>,
    fc1_bias: Vec<f32>,
    fc2_weights: Vec<Vec<f32>>,
    fc2_bias: Vec<f32>,
}

impl Expert {
    /// Create new expert
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        use_dropout: bool,
        dropout_rate: f32,
    ) -> Self {
        let config = ExpertConfig {
            hidden_size,
            intermediate_size,
            use_dropout,
            dropout_rate,
        };

        // Initialize feed-forward network weights
        let fc1_weights = Self::init_weights(hidden_size, intermediate_size);
        let fc1_bias = vec![0.0; intermediate_size];
        let fc2_weights = Self::init_weights(intermediate_size, hidden_size);
        let fc2_bias = vec![0.0; hidden_size];

        Self {
            config,
            fc1_weights,
            fc1_bias,
            fc2_weights,
            fc2_bias,
        }
    }

    /// Initialize weights with Xavier/Glorot initialization
    fn init_weights(input_size: usize, output_size: usize) -> Vec<Vec<f32>> {
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let row: Vec<f32> = (0..input_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();
            weights.push(row);
        }
        weights
    }

    /// Forward pass through expert
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(result) = self.forward_gpu(input) {
            return result;
        }

        // First linear layer + activation
        let hidden = self.fc1_forward(input);
        let activated = self.apply_gelu(&hidden);

        // Apply dropout if enabled
        let dropped = if self.config.use_dropout {
            self.apply_dropout(&activated)
        } else {
            activated
        };

        // Second linear layer
        let output = self.fc2_forward(&dropped);

        output
    }

    /// GPU-accelerated forward pass
    #[cfg(feature = "gpu")]
    fn forward_gpu(&self, input: &[f32]) -> Option<Vec<f32>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        use ndarray::ArrayD;

        use tracing::warn;
        let ctx = match GpuContext::global() {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!("GPU forward failed (GPU unavailable): {e}");
                return None;
            }
        };
        let dim = input.len();

        // Upload input as [1, hidden_size]
        let input_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, dim], input.to_vec()).ok()?
        ).ok()?;

        // fc1 weights: [intermediate_size, hidden_size] → transpose → [hidden_size, intermediate_size]
        let fc1_w: Vec<f32> = self.fc1_weights.iter().flat_map(|r| r.iter()).copied().collect();
        let w1_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![self.fc1_weights.len(), dim], fc1_w).ok()?
        ).ok()?;
        let w1t_gpu = ctx.transpose(&w1_gpu).ok()?;
        let b1_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, self.fc1_bias.len()], self.fc1_bias.clone()).ok()?
        ).ok()?;

        let hidden_gpu = ctx.matmul(&input_gpu, &w1t_gpu).ok()?;
        let hidden_gpu = ctx.add(&hidden_gpu, &b1_gpu).ok()?;

        // GELU on CPU (GELU GPU kernel not assumed available)
        let hidden_cpu = hidden_gpu.to_cpu();
        let hidden_slice: Vec<f32> = hidden_cpu.iter().copied().collect();
        let activated: Vec<f32> = hidden_slice.iter().map(|&x| gelu(x)).collect();

        // Dropout (CPU, same as original)
        let dropped = if self.config.use_dropout {
            self.apply_dropout(&activated)
        } else {
            activated
        };

        // fc2 weights: [hidden_size, intermediate_size] → transpose → [intermediate_size, hidden_size]
        let fc2_w: Vec<f32> = self.fc2_weights.iter().flat_map(|r| r.iter()).copied().collect();
        let w2_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![self.fc2_weights.len(), dropped.len()], fc2_w).ok()?
        ).ok()?;
        let w2t_gpu = ctx.transpose(&w2_gpu).ok()?;
        let b2_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, self.fc2_bias.len()], self.fc2_bias.clone()).ok()?
        ).ok()?;
        let dropped_gpu = GpuTensor::from_cpu(
            &ArrayD::from_shape_vec(vec![1, dropped.len()], dropped).ok()?
        ).ok()?;

        let out_gpu = ctx.matmul(&dropped_gpu, &w2t_gpu).ok()?;
        let out_gpu = ctx.add(&out_gpu, &b2_gpu).ok()?;

        let out_cpu = out_gpu.to_cpu();
        Some(out_cpu.iter().copied().collect())
    }

    /// First linear layer forward pass
    fn fc1_forward(&self, input: &[f32]) -> Vec<f32> {
        self.fc1_weights
            .iter()
            .zip(self.fc1_bias.iter())
            .map(|(weights, bias)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                sum
            })
            .collect()
    }

    /// Second linear layer forward pass
    fn fc2_forward(&self, input: &[f32]) -> Vec<f32> {
        self.fc2_weights
            .iter()
            .zip(self.fc2_bias.iter())
            .map(|(weights, bias)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                sum
            })
            .collect()
    }

    /// GELU activation function
    fn apply_gelu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| gelu(x)).collect()
    }

    /// Apply dropout during training
    fn apply_dropout(&self, input: &[f32]) -> Vec<f32> {
        let rate = self.config.dropout_rate;
        let scale = 1.0 / (1.0 - rate);
        input
            .iter()
            .map(|&x| {
                if rand::random::<f32>() < rate {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect()
    }

    /// Get expert configuration
    pub fn config(&self) -> &ExpertConfig {
        &self.config
    }

    /// Get expert size information
    pub fn size_info(&self) -> (usize, usize) {
        (self.config.hidden_size, self.config.intermediate_size)
    }
}
