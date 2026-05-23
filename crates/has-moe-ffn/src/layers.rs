//! Layer implementations for HAS-MoE-FFN

use serde::{Deserialize, Serialize};

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub use_bias: bool,
    pub activation: String,
}

/// Base layer trait
pub trait Layer {
    fn forward(&self, input: &[f32]) -> Vec<f32>;
    fn config(&self) -> &LayerConfig;
}

/// Dense layer implementation
pub struct DenseLayer {
    config: LayerConfig,
    weights: Vec<Vec<f32>>,
    bias: Option<Vec<f32>>,
}

impl DenseLayer {
    /// Create new dense layer
    pub fn new(config: LayerConfig) -> Self {
        let weights = Self::init_weights(config.input_size, config.output_size);
        let bias = if config.use_bias {
            Some(Self::init_bias(config.output_size))
        } else {
            None
        };

        Self {
            config,
            weights,
            bias,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Perform matrix multiplication: output = input * weights^T + bias
        let mut output = Vec::with_capacity(self.config.output_size);

        for i in 0..self.config.output_size {
            let mut sum = 0.0;

            // Matrix multiplication
            for j in 0..self.config.input_size {
                sum += input[j] * self.weights[i][j];
            }

            // Add bias if present
            if let Some(ref bias) = self.bias {
                sum += bias[i];
            }

            // Apply activation function
            let activated = self.apply_activation(sum);
            output.push(activated);
        }

        output
    }

    fn config(&self) -> &LayerConfig {
        &self.config
    }
}

impl DenseLayer {
    /// Apply activation function
    fn apply_activation(&self, x: f32) -> f32 {
        match self.config.activation.as_str() {
            "relu" => x.max(0.0),
            "gelu" => crate::experts::gelu(x),
            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "tanh" => x.tanh(),
            "swish" => x * (1.0 / (1.0 + (-x).exp())),
            _ => x, // No activation
        }
    }

    /// Initialize weights with Xavier/Glorot initialization
    fn init_weights(input_size: usize, output_size: usize) -> Vec<Vec<f32>> {
        let scale = (6.0 / (input_size + output_size) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let row: Vec<f32> = (0..input_size)
                .map(|j| {
                    // Uniform initialization in [-scale, scale]
                    let normalized = (i as f32 + j as f32) / (input_size * output_size) as f32;
                    (normalized - 0.5) * 2.0 * scale
                })
                .collect();
            weights.push(row);
        }

        weights
    }

    /// Initialize bias with zeros
    fn init_bias(size: usize) -> Vec<f32> {
        vec![0.0; size]
    }

    /// Get layer parameters count
    pub fn param_count(&self) -> usize {
        let weight_params = self.config.input_size * self.config.output_size;
        let bias_params = if self.config.use_bias {
            self.config.output_size
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Get layer shape information
    pub fn shape(&self) -> (usize, usize) {
        (self.config.input_size, self.config.output_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_layer(activation: &str) -> DenseLayer {
        DenseLayer::new(LayerConfig {
            input_size: 4,
            output_size: 3,
            use_bias: true,
            activation: activation.to_string(),
        })
    }

    #[test]
    fn test_dense_forward_output_dim() {
        let layer = dense_layer("relu");
        let input = vec![1.0, 0.5, 0.0, -0.5];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_dense_forward_relu_clamps_negative() {
        let mut layer = DenseLayer::new(LayerConfig {
            input_size: 1,
            output_size: 1,
            use_bias: false,
            activation: "relu".into(),
        });
        layer.weights = vec![vec![-1.0]];
        let out = layer.forward(&[5.0]);
        assert!(out[0] <= 0.0);
    }

    #[test]
    fn test_dense_forward_without_bias() {
        let mut layer = DenseLayer::new(LayerConfig {
            input_size: 2,
            output_size: 2,
            use_bias: false,
            activation: "none".into(),
        });
        layer.weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let output = layer.forward(&[3.0, 5.0]);
        assert!((output[0] - 3.0).abs() < 1e-5);
        assert!((output[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations_relu() {
        let layer = dense_layer("relu");
        let out = layer.apply_activation(-5.0);
        assert_eq!(out, 0.0);
        assert_eq!(layer.apply_activation(3.0), 3.0);
    }

    #[test]
    fn test_activations_gelu() {
        let layer = dense_layer("gelu");
        let out = layer.apply_activation(0.0);
        assert!((out - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations_sigmoid() {
        let layer = dense_layer("sigmoid");
        let out = layer.apply_activation(0.0);
        assert!((out - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_activations_tanh() {
        let layer = dense_layer("tanh");
        let out = layer.apply_activation(0.0);
        assert!((out - 0.0).abs() < 1e-5);
        assert!((layer.apply_activation(100.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations_swish() {
        let layer = dense_layer("swish");
        let out = layer.apply_activation(0.0);
        assert!((out - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations_none() {
        let layer = dense_layer("unknown");
        assert_eq!(layer.apply_activation(42.0), 42.0);
    }

    #[test]
    fn test_config() {
        let layer = dense_layer("relu");
        let cfg = layer.config();
        assert_eq!(cfg.input_size, 4);
        assert_eq!(cfg.output_size, 3);
    }

    #[test]
    fn test_param_count_with_bias() {
        let layer = dense_layer("relu");
        assert_eq!(layer.param_count(), 4 * 3 + 3);
    }

    #[test]
    fn test_param_count_without_bias() {
        let layer = DenseLayer::new(LayerConfig {
            input_size: 4,
            output_size: 3,
            use_bias: false,
            activation: "relu".into(),
        });
        assert_eq!(layer.param_count(), 12);
    }

    #[test]
    fn test_shape() {
        let layer = dense_layer("relu");
        assert_eq!(layer.shape(), (4, 3));
    }
}
