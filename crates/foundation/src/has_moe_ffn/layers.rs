//! Layer implementations for HAS-MoE-FFN

use serde::{Serialize, Deserialize};

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
        
        Self { config, weights, bias }
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
            "gelu" => crate::has_moe_ffn::experts::gelu(x),
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
        let bias_params = if self.config.use_bias { self.config.output_size } else { 0 };
        weight_params + bias_params
    }
    
    /// Get layer shape information
    pub fn shape(&self) -> (usize, usize) {
        (self.config.input_size, self.config.output_size)
    }
}
