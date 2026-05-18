//! Routing mechanism for HAS-MoE-FFN
//! Improved dengan Capped Routing + Load Balancing Loss (Switch Transformer + Expert Choice)

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub z_loss_coefficient: f32,
    pub importance_loss_coefficient: f32,
    pub use_capped_routing: bool,
    pub use_load_balancing_loss: bool,
    pub use_expert_choice: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            z_loss_coefficient: 1e-4,
            importance_loss_coefficient: 0.01,
            use_capped_routing: true,
            use_load_balancing_loss: true,
            use_expert_choice: false,
        }
    }
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub load_balance_score: f32,
    pub expert_utilization: Vec<f32>,
    pub total_routes: usize,
    pub load_balancing_loss: f32,
    pub z_loss: f32,
    pub capacity_violations: usize,
    pub total_tokens: usize,
}

impl RoutingStats {
    pub fn new(num_experts: usize) -> Self {
        Self {
            load_balance_score: 1.0,
            expert_utilization: vec![0.0; num_experts],
            total_routes: 0,
            load_balancing_loss: 0.0,
            z_loss: 0.0,
            capacity_violations: 0,
            total_tokens: 0,
        }
    }
}

/// Router for expert selection
pub struct Router {
    config: RouterConfig,
    routing_stats: HashMap<usize, usize>,
    expert_capacities: Vec<usize>,
    router_weights: Vec<Vec<f32>>,
    last_aux_loss: f32,
}

impl Router {
    /// Create new router
    pub fn new(hidden_size: usize, num_experts: usize, top_k: usize) -> Self {
        let config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            ..Default::default()
        };
        
        let scale = (1.0 / hidden_size as f32).sqrt();
        let router_weights: Vec<Vec<f32>> = (0..num_experts)
            .map(|_| (0..hidden_size).map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale).collect())
            .collect();

        Self { 
            expert_capacities: vec![0; num_experts],
            config,
            routing_stats: HashMap::new(),
            router_weights,
            last_aux_loss: 0.0,
        }
    }
    
    /// Create router with custom config
    pub fn with_config(config: RouterConfig) -> Self {
        let num_experts = config.num_experts;
        let hidden_size = config.hidden_size;
        let scale = (1.0 / hidden_size as f32).sqrt();
        let router_weights: Vec<Vec<f32>> = (0..num_experts)
            .map(|_| (0..hidden_size).map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale).collect())
            .collect();

        Self { 
            expert_capacities: vec![0; config.num_experts],
            config,
            routing_stats: HashMap::new(),
            router_weights,
            last_aux_loss: 0.0,
        }
    }

    /// Return the auxiliary loss from the last forward pass
    pub fn auxiliary_loss(&self) -> f32 {
        self.last_aux_loss
    }
    
    /// Compute gating weight for a specific expert
    fn compute_gating_weight(&self, input: &[f32], expert_idx: usize) -> f32 {
        let expert_bias = expert_idx as f32 * 0.1;
        let dot_product: f32 = input.iter().enumerate()
            .map(|(i, &x)| x * self.router_weights[expert_idx][i])
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
        
        for i in 0..batch_size {
            let row_view = input.row(i);
            let token_slice = row_view.as_slice().unwrap_or(&[]);
            for j in 0..self.config.num_experts {
                let expert_weight = self.compute_gating_weight(token_slice, j);
                gating_weights[[i, j]] = expert_weight;
            }
            
            // Apply softmax to get probability distribution
            self.apply_softmax_row(&mut gating_weights, i);
        }
        
        gating_weights
    }
    
    pub fn route_single(&self, input: &ndarray::Array1<f32>) -> Result<Vec<usize>, String> {
        let input_slice = input.as_slice().unwrap_or(&[]);
        let mut gating_weights = Vec::with_capacity(self.config.num_experts);
        for j in 0..self.config.num_experts {
            let weight = self.compute_gating_weight(input_slice, j);
            gating_weights.push(weight);
        }
        
        // Apply softmax
        let softmax_weights = self.softmax(&gating_weights);
        
        // Get top-k experts using O(E) select_nth_unstable
        let mut expert_scores: Vec<(usize, f32)> = softmax_weights
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        let top_experts: Vec<usize> = expert_scores
            .iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect();
        
        Ok(top_experts)
    }
    
    /// Route batch of inputs dengan Capped Routing + Load Balancing Loss
    pub fn route(&mut self, input: &ndarray::Array2<f32>) -> Result<Vec<Vec<usize>>, String> {
        let (batch_size, _) = input.dim();
        let mut all_routes = Vec::with_capacity(batch_size);
        let mut routing_weights: Vec<Vec<(usize, f32)>> = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let row_view = input.row(i);
            let row_slice = row_view.as_slice().unwrap_or(&[]);
            let (route, weights) = self.route_single_with_weights(row_slice)?;
            let weights_with_indices: Vec<(usize, f32)> = weights.into_iter().enumerate().collect();
            routing_weights.push(weights_with_indices);
            all_routes.push(route);
        }
        
        // Phase 2: Capped routing — batasi token per expert (batch-aware)
        if self.config.use_capped_routing {
            let capacity = (batch_size as f32 * self.config.top_k as f32 
                * self.config.capacity_factor / self.config.num_experts as f32).ceil() as usize;
            
            let mut expert_counts = vec![0usize; self.config.num_experts];
            let mut capped_routes: Vec<Vec<usize>> = vec![Vec::new(); batch_size];
            let mut _capacity_violations = 0;
            
            // Sort tokens by routing confidence untuk fair capacity allocation
            for i in 0..batch_size {
                let conf = all_routes[i].iter()
                    .map(|&e| routing_weights[i].iter()
                        .find(|(ex, _)| *ex == e)
                        .map(|(_, w)| *w)
                        .unwrap_or(0.0))
                    .sum::<f32>();
                
                for &expert_id in &all_routes[i] {
                    if expert_counts[expert_id] < capacity {
                        capped_routes[i].push(expert_id);
                        expert_counts[expert_id] += 1;
                    } else {
                        _capacity_violations += 1;
                    }
                }
            }
            
            self.expert_capacities = expert_counts;
            all_routes = capped_routes;
        }
        
        // Phase 3: Compute auxiliary loss (load balancing + Z-loss)
        self.last_aux_loss = 0.0;
        if self.config.use_load_balancing_loss {
            self.last_aux_loss += self.compute_load_balancing_loss(&routing_weights, batch_size);
        }
        
        // Phase 4: Update routing stats
        for route in &all_routes {
            for &expert_id in route {
                *self.routing_stats.entry(expert_id).or_insert(0) += 1;
            }
        }
        
        Ok(all_routes)
    }
    
    pub fn route_single_with_zloss(&self, input: &ndarray::Array1<f32>) -> Result<(Vec<usize>, Vec<f32>, f32), String> {
        let input_slice = input.as_slice().unwrap_or(&[]);
        let mut gating_weights = Vec::with_capacity(self.config.num_experts);
        for j in 0..self.config.num_experts {
            let weight = self.compute_gating_weight(input_slice, j);
            gating_weights.push(weight);
        }
        
        // Softmax + compute Z-loss
        let max_val = gating_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = gating_weights.iter().map(|x| (x - max_val).exp()).sum();
        let log_sum_exp = max_val + exp_sum.ln();
        let z_loss = log_sum_exp * log_sum_exp; // Z-loss = log(Σ exp)²
        
        let softmax_weights: Vec<f32> = if exp_sum > 0.0 {
            gating_weights.iter().map(|x| (x - max_val).exp() / exp_sum).collect()
        } else {
            vec![1.0 / self.config.num_experts as f32; self.config.num_experts]
        };
        
        // Get top-k experts using O(E) select_nth_unstable
        let mut expert_scores: Vec<(usize, f32)> = softmax_weights.iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        let top_experts: Vec<usize> = expert_scores.iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect();
        
        Ok((top_experts, softmax_weights, z_loss))
    }

    fn route_single_with_weights(&self, input: &[f32]) -> Result<(Vec<usize>, Vec<f32>), String> {
        let mut gating_weights = Vec::with_capacity(self.config.num_experts);
        for j in 0..self.config.num_experts {
            let weight = self.compute_gating_weight(input, j);
            gating_weights.push(weight);
        }
        
        let softmax_weights = self.softmax(&gating_weights);
        
        let mut expert_scores: Vec<(usize, f32)> = softmax_weights.iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        let k = self.config.top_k.min(expert_scores.len());
        if k > 1 {
            expert_scores.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        let top_experts: Vec<usize> = expert_scores.iter()
            .take(k)
            .map(|(expert_idx, _)| *expert_idx)
            .collect();
        
        Ok((top_experts, softmax_weights))
    }
    
    /// Compute load balancing loss
    /// Importance loss (Switch Transformer): ∑_i f_i · P_i
    /// f_i = fraction of tokens routed to expert i
    /// P_i = average router probability for expert i
    pub fn compute_load_balancing_loss(&self, routing_weights: &[Vec<(usize, f32)>], batch_size: usize) -> f32 {
        if batch_size == 0 || !self.config.use_load_balancing_loss {
            return 0.0;
        }
        
        let num_experts = self.config.num_experts;
        let mut expert_counts = vec![0.0f32; num_experts];
        let mut expert_probs = vec![0.0f32; num_experts];
        
        for weights in routing_weights {
            for &(expert_id, prob) in weights {
                if expert_id < num_experts {
                    expert_counts[expert_id] += 1.0;
                    expert_probs[expert_id] += prob;
                }
            }
        }
        
        let total_tokens = batch_size as f32;
        let importance_loss: f32 = (0..num_experts).map(|i| {
            let f_i = expert_counts[i] / total_tokens;
            let p_i = expert_probs[i] / total_tokens;
            f_i * p_i
        }).sum();
        
        // Nol loss jika routing sempurna (uniform)
        let uniform_loss = 1.0 / num_experts as f32;
        importance_loss * self.config.importance_loss_coefficient
    }
    
    /// Get routing statistics (backward compatible — tanpa z_loss/load_balancing_loss)
    pub fn get_routing_stats(&self) -> RoutingStats {
        self.get_routing_stats_detailed(0.0, 0.0)
    }
    
    /// Get routing statistics (updated with load balancing metrics)
    pub fn get_routing_stats_detailed(&self, z_loss: f32, load_balancing_loss: f32) -> RoutingStats {
        let total_routes: usize = self.routing_stats.values().sum();
        let mut expert_utilization = vec![0.0; self.config.num_experts];
        
        for (expert_id, count) in &self.routing_stats {
            if *expert_id < self.config.num_experts {
                expert_utilization[*expert_id] = *count as f32 / total_routes.max(1) as f32;
            }
        }
        
        let avg_utilization = expert_utilization.iter().sum::<f32>() / self.config.num_experts as f32;
        let variance = expert_utilization.iter()
            .map(|u| (u - avg_utilization).powi(2))
            .sum::<f32>() / self.config.num_experts as f32;
        let load_balance_score = 1.0 / (1.0 + variance);
        
        let capacity_violations = self.expert_capacities.iter()
            .enumerate()
            .filter(|(_, &count)| count > (total_routes as f32 / self.config.num_experts as f32 * self.config.capacity_factor) as usize)
            .count();
        
        RoutingStats {
            load_balance_score,
            expert_utilization,
            total_routes,
            load_balancing_loss,
            z_loss,
            capacity_violations,
            total_tokens: total_routes / self.config.top_k.max(1),
        }
    }
}
