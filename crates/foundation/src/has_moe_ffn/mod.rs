/// Heterogeneous Attention-based Sparse MoE Feed-Forward Network (HAS-MoE-FFN)
/// 
/// Advanced sparse mixture of experts implementation for transformer models

pub mod layers;
pub mod experts;
pub mod routing;
pub mod attention;
pub mod config;
pub mod types;
pub mod error;

// Re-export main components
pub use layers::*;
pub use experts::*;
pub use routing::*;
pub use attention::*;
pub use config::*;
pub use types::*;
pub use error::*;

use serde::{Serialize, Deserialize};

/// Main HAS-MoE-FFN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasMoeFFNConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub use_dropout: bool,
    pub dropout_rate: f32,
}

impl Default for HasMoeFFNConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            hidden_size: 768,
            intermediate_size: 3072,
            use_dropout: true,
            dropout_rate: 0.1,
        }
    }
}

/// Main HAS-MoE-FFN implementation
pub struct HasMoeFFN {
    config: HasMoeFFNConfig,
    experts: Vec<crate::has_moe_ffn::experts::Expert>,
    router: crate::has_moe_ffn::routing::Router,
}

impl HasMoeFFN {
    /// Create new HAS-MoE-FFN
    pub fn new(config: HasMoeFFNConfig) -> Self {
        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            experts.push(crate::has_moe_ffn::experts::Expert::new(
                config.hidden_size,
                config.intermediate_size,
                config.use_dropout,
                config.dropout_rate,
            ));
        }
        
        let router = crate::has_moe_ffn::routing::Router::new(
            config.hidden_size,
            config.num_experts,
            config.top_k,
        );
        
        Self {
            config,
            experts,
            router,
        }
    }
    
    /// Forward pass through HAS-MoE-FFN
    pub fn forward(&self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        let (batch_size, hidden_size) = input.dim();
        
        // Route to experts
        let routing_weights = self.router.forward(input);
        
        // Initialize output tensor
        let mut output = ndarray::Array2::zeros((batch_size, hidden_size));
        
        for i in 0..batch_size {
            let row_view = input.row(i);
            let token_slice = row_view.as_slice().unwrap_or(&[]);
            let mut token_output = vec![0.0; hidden_size];
            
            let top_experts = self.get_top_experts(&routing_weights, i);
            
            for &expert_idx in &top_experts {
                let routing_weight = routing_weights[[i, expert_idx]];
                
                let expert_output = self.experts[expert_idx].forward(token_slice);
                
                // Add weighted contribution to output
                for (j, &val) in expert_output.iter().enumerate() {
                    token_output[j] += val * routing_weight;
                }
            }
            
            // Store token output
            for (j, &val) in token_output.iter().enumerate() {
                output[[i, j]] = val;
            }
        }
        
        output
    }
    
    /// Get top-k experts for a specific token
    fn get_top_experts(&self, routing_weights: &ndarray::Array2<f32>, token_idx: usize) -> Vec<usize> {
        let mut expert_scores: Vec<(usize, f32)> = (0..self.config.num_experts)
            .map(|j| (j, routing_weights[[token_idx, j]]))
            .collect();
        
        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        expert_scores
            .iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect()
    }
    
    /// Get configuration
    pub fn config(&self) -> &HasMoeFFNConfig {
        &self.config
    }
}

impl Default for HasMoeFFN {
    fn default() -> Self {
        Self::new(HasMoeFFNConfig::default())
    }
}
