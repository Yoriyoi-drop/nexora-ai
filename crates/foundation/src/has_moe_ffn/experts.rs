//! Expert modules for HAS-MoE-FFN

use serde::{Serialize, Deserialize};

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
    pub fn new(hidden_size: usize, intermediate_size: usize, use_dropout: bool, dropout_rate: f32) -> Self {
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
    
    /// First linear layer forward pass
    fn fc1_forward(&self, input: &[f32]) -> Vec<f32> {
        self.fc1_weights.iter().zip(self.fc1_bias.iter()).map(|(weights, bias)| {
            let mut sum = *bias;
            for (w, x) in weights.iter().zip(input.iter()) {
                sum += w * x;
            }
            sum
        }).collect()
    }
    
    /// Second linear layer forward pass
    fn fc2_forward(&self, input: &[f32]) -> Vec<f32> {
        self.fc2_weights.iter().zip(self.fc2_bias.iter()).map(|(weights, bias)| {
            let mut sum = *bias;
            for (w, x) in weights.iter().zip(input.iter()) {
                sum += w * x;
            }
            sum
        }).collect()
    }
    
    /// GELU activation function
    fn apply_gelu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| gelu(x)).collect()
    }
    
    /// Apply dropout during training
    fn apply_dropout(&self, input: &[f32]) -> Vec<f32> {
        let rate = self.config.dropout_rate;
        let scale = 1.0 / (1.0 - rate);
        input.iter().map(|&x| {
            if rand::random::<f32>() < rate {
                0.0
            } else {
                x * scale
            }
        }).collect()
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
