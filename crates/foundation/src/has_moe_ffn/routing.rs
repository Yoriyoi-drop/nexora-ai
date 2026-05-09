//! Routing mechanism for HAS-MoE-FFN

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub load_balance_score: f32,
    pub expert_utilization: Vec<f32>,
    pub total_routes: usize,
}

/// Router for expert selection
pub struct Router {
    config: RouterConfig,
    routing_stats: HashMap<usize, usize>,
}

impl Router {
    /// Create new router
    pub fn new(hidden_size: usize, num_experts: usize, top_k: usize) -> Self {
        let config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
        };
        
        Self { 
            config,
            routing_stats: HashMap::new(),
        }
    }
    
    /// Compute gating weight for a specific expert
    fn compute_gating_weight(&self, input: &ndarray::Array1<f32>, expert_idx: usize) -> f32 {
        // Simple gating: use expert index as a bias and compute dot product
        let expert_bias = expert_idx as f32 * 0.1;
        let dot_product: f32 = input.iter().enumerate()
            .map(|(i, &x)| x * ((i + expert_idx) as f32 * 0.01).cos())
            .sum();
        dot_product + expert_bias
    }
    
    /// Apply softmax to a row of gating weights
    fn apply_softmax_row(&self, gating_weights: &mut ndarray::Array2<f32>, row_idx: usize) {
        let mut row_vals: Vec<f32> = (0..self.config.num_experts)
            .map(|j| gating_weights[[row_idx, j]])
            .collect();
        
        let softmax_vals = self.softmax(&row_vals);
        
        for (j, &val) in softmax_vals.iter().enumerate() {
            gating_weights[[row_idx, j]] = val;
        }
    }
    
    /// Softmax function
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp and sum
        let exp_vals: Vec<f32> = input.iter()
            .map(|x| (x - max_val).exp())
            .collect();
        let sum: f32 = exp_vals.iter().sum();
        
        // Normalize
        if sum > 0.0 {
            exp_vals.iter().map(|x| x / sum).collect()
        } else {
            vec![1.0 / input.len() as f32; input.len()]
        }
    }
    
    /// Forward pass through router
    pub fn forward(&self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        let (batch_size, hidden_size) = input.dim();
        let mut gating_weights = ndarray::Array2::zeros((batch_size, self.config.num_experts));
        
        // Compute gating weights for each token
        for i in 0..batch_size {
            for j in 0..self.config.num_experts {
                // Simple gating: compute similarity between token and expert
                let token_vec = input.slice(ndarray::s![i, ..]).to_owned();
                let expert_weight = self.compute_gating_weight(&token_vec, j);
                gating_weights[[i, j]] = expert_weight;
            }
            
            // Apply softmax to get probability distribution
            self.apply_softmax_row(&mut gating_weights, i);
        }
        
        gating_weights
    }
    
    /// Route single input
    pub fn route_single(&self, input: &ndarray::Array1<f32>) -> Result<Vec<usize>, String> {
        // Compute gating weights for this input
        let mut gating_weights = Vec::with_capacity(self.config.num_experts);
        for j in 0..self.config.num_experts {
            let weight = self.compute_gating_weight(input, j);
            gating_weights.push(weight);
        }
        
        // Apply softmax
        let softmax_weights = self.softmax(&gating_weights);
        
        // Get top-k experts
        let mut expert_scores: Vec<(usize, f32)> = softmax_weights
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_experts: Vec<usize> = expert_scores
            .iter()
            .take(self.config.top_k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect();
        
        Ok(top_experts)
    }
    
    /// Route batch of inputs
    pub fn route(&mut self, input: &ndarray::Array2<f32>) -> Result<Vec<Vec<usize>>, String> {
        let (batch_size, _) = input.dim();
        let mut all_routes = Vec::new();
        
        for i in 0..batch_size {
            let row = input.slice(ndarray::s![i, ..]).to_owned();
            let route = self.route_single(&row)?;
            
            // Update routing stats
            for &expert_id in &route {
                *self.routing_stats.entry(expert_id).or_insert(0) += 1;
            }
            
            all_routes.push(route);
        }
        
        Ok(all_routes)
    }
    
    /// Get routing statistics
    pub fn get_routing_stats(&self) -> RoutingStats {
        let total_routes: usize = self.routing_stats.values().sum();
        let mut expert_utilization = vec![0.0; self.config.num_experts];
        
        for (expert_id, count) in &self.routing_stats {
            if *expert_id < self.config.num_experts {
                expert_utilization[*expert_id] = *count as f32 / total_routes.max(1) as f32;
            }
        }
        
        // Calculate load balance score (1.0 = perfectly balanced, 0.0 = completely imbalanced)
        let avg_utilization = expert_utilization.iter().sum::<f32>() / self.config.num_experts as f32;
        let variance = expert_utilization.iter()
            .map(|u| (u - avg_utilization).powi(2))
            .sum::<f32>() / self.config.num_experts as f32;
        let load_balance_score = 1.0 / (1.0 + variance);
        
        RoutingStats {
            load_balance_score,
            expert_utilization,
            total_routes,
        }
    }
}
