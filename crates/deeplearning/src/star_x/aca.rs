//! Adaptive Compute Allocation (ACA)
//!
//! Dynamic compute routing untuk efisiensi:
//! - Tidak semua token membutuhkan compute penuh
//! - Routing berdasarkan complexity
//! - Compute fokus ke token penting
//! - Scaling yang lebih efisien

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::AdaptiveCompute;
use ndarray::{ArrayD, Array1, Array2};
use std::collections::HashMap;
use rand;

/// Compute level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeLevel {
    Micro,      // Hanya TGH
    Meso,       // TGH + SCA
    Full,       // Full STAR-X
}

impl From<ComputeLevel> for usize {
    fn from(level: ComputeLevel) -> usize {
        match level {
            ComputeLevel::Micro => 0,
            ComputeLevel::Meso => 1,
            ComputeLevel::Full => 2,
        }
    }
}

impl ComputeLevel {
    pub fn as_usize(&self) -> usize {
        (*self).into()
    }
    
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => ComputeLevel::Micro,
            1 => ComputeLevel::Meso,
            _ => ComputeLevel::Full,
        }
    }
}

/// Adaptive Compute Allocation implementation
#[derive(Debug, Clone)]
pub struct AdaptiveComputeAllocation {
    // Compute routing network
    routing_weights: Array2<f32>,
    routing_bias: Array1<f32>,
    
    // Complexity estimation parameters
    complexity_weights: Array2<f32>,
    complexity_bias: Array1<f32>,
    
    // Configuration
    input_size: usize,
    hidden_size: usize,
    compute_thresholds: Vec<f32>,
    
    // Compute statistics
    level_usage: HashMap<ComputeLevel, usize>,
    total_computations: usize,
    compute_efficiency: f32,
    
    // Adaptive parameters
    adaptation_rate: f32,
    target_efficiency: f32,
    
    // Routing strategy
    routing_strategy: RoutingStrategy,
}

/// Routing strategy enumeration
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    Threshold,      // Static threshold-based
    Adaptive,       // Adaptive threshold
    Complexity,     // Complexity-based
    Importance,     // Importance-based
    Hybrid,         // Hybrid approach
}

impl AdaptiveComputeAllocation {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        compute_thresholds: Vec<f32>,
    ) -> DLResult<Self> {
        if compute_thresholds.len() != 3 {
            return Err(DeepLearningError::Configuration {
                reason: "Must provide exactly 3 compute thresholds".to_string(),
            });
        }
        
        // Initialize routing network
        let routing_weights = Self::xavier_init(input_size + hidden_size, 3); // 3 compute levels
        let routing_bias = Array1::zeros(3);
        
        // Initialize complexity network
        let complexity_weights = Self::xavier_init(input_size, 1);
        let complexity_bias = Array1::zeros(1);
        
        let mut level_usage = HashMap::new();
        level_usage.insert(ComputeLevel::Micro, 0);
        level_usage.insert(ComputeLevel::Meso, 0);
        level_usage.insert(ComputeLevel::Full, 0);
        
        Ok(Self {
            routing_weights,
            routing_bias,
            complexity_weights,
            complexity_bias,
            input_size,
            hidden_size,
            compute_thresholds,
            level_usage,
            total_computations: 0,
            compute_efficiency: 0.0,
            adaptation_rate: 0.01,
            target_efficiency: 0.7,
            routing_strategy: RoutingStrategy::Adaptive,
        })
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            (rand::random::<f32>() * 2.0 - 1.0) * limit
        })
    }
    
    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Apply sigmoid ke array
    fn sigmoid_array(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        for val in arr.iter_mut() {
            *val = self.sigmoid(*val);
        }
        arr
    }
    
    /// Concatenate input dan hidden state
    fn concatenate(&self, input: &ArrayD<f32>, hidden_state: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().expect("tensor should be contiguous");
        let hidden_flat = hidden_state.as_slice().expect("tensor should be contiguous");
        
        let mut concatenated = Vec::with_capacity(input_flat.len() + hidden_flat.len());
        concatenated.extend_from_slice(input_flat);
        concatenated.extend_from_slice(hidden_flat);
        
        Ok(Array1::from_vec(concatenated).into_dyn())
    }
    
    /// Matrix multiplication
    fn matmul(&self, weights: &Array2<f32>, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().expect("tensor should be contiguous");
        if input_flat.len() != weights.shape()[0] {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![weights.shape()[0]],
                actual: vec![input_flat.len()],
            });
        }
        
        let mut output = vec![0.0; weights.shape()[1]];
        for (i, row) in weights.outer_iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                output[j] += input_flat[i] * weight;
            }
        }
        
        Ok(Array1::from_vec(output).into_dyn())
    }
    
    /// Add bias
    fn add_bias(&self, output: &mut ArrayD<f32>, bias: &Array1<f32>) {
        let output_flat = output.as_slice_mut().expect("tensor should be contiguous");
        for (i, &b) in bias.iter().enumerate().take(output_flat.len()) {
            output_flat[i] += b;
        }
    }
    
    /// Compute input complexity
    pub fn compute_complexity(&self, input: &ArrayD<f32>) -> DLResult<f32> {
        // Linear projection
        let complexity_linear = self.matmul(&self.complexity_weights, input)?;
        let mut complexity_output = complexity_linear;
        self.add_bias(&mut complexity_output, &self.complexity_bias);
        
        // Apply sigmoid
        let complexity_prob = self.sigmoid_array(complexity_output);
        let complexity_flat = complexity_prob.as_slice().expect("tensor should be contiguous");
        
        Ok(complexity_flat[0])
    }
    
    /// Compute input importance
    pub fn compute_importance(&self, input: &ArrayD<f32>, hidden_state: &ArrayD<f32>) -> DLResult<f32> {
        let concatenated = self.concatenate(input, hidden_state)?;
        
        // Compute routing scores
        let routing_linear = self.matmul(&self.routing_weights, &concatenated)?;
        let mut routing_output = routing_linear;
        self.add_bias(&mut routing_output, &self.routing_bias);
        
        // Apply softmax
        let routing_probs = self.softmax_array(&routing_output)?;
        let routing_flat = routing_probs.as_slice().expect("tensor should be contiguous");
        
        // Importance = weighted sum of compute levels
        let importance = routing_flat[0] * 0.0 + routing_flat[1] * 0.5 + routing_flat[2] * 1.0;
        Ok(importance)
    }
    
    /// Softmax untuk routing probabilities
    fn softmax_array(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().expect("tensor should be contiguous");
        
        // Find max for numerical stability
        let max_val = input_flat.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp and sum
        let exp_vals: Vec<f32> = input_flat.iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        
        // Normalize
        let softmax_vals: Vec<f32> = exp_vals.iter()
            .map(|&e| e / sum_exp)
            .collect();
        
        Ok(Array1::from_vec(softmax_vals).into_dyn())
    }
    
    /// Update compute statistics
    fn update_statistics(&mut self, level: ComputeLevel) {
        *self.level_usage.entry(level).or_insert(0) += 1;
        self.total_computations += 1;
        
        // Update efficiency
        let micro_usage = *self.level_usage.get(&ComputeLevel::Micro).unwrap_or(&0) as f32;
        let meso_usage = *self.level_usage.get(&ComputeLevel::Meso).unwrap_or(&0) as f32;
        let full_usage = *self.level_usage.get(&ComputeLevel::Full).unwrap_or(&0) as f32;
        
        // Efficiency = weighted average (lower compute levels = higher efficiency)
        self.compute_efficiency = (micro_usage * 1.0 + meso_usage * 0.7 + full_usage * 0.3) / 
                                self.total_computations as f32;
    }
    
    /// Adaptive threshold adjustment
    fn adapt_thresholds(&mut self) {
        let efficiency_diff = self.target_efficiency - self.compute_efficiency;
        
        if efficiency_diff > 0.1 {
            // Efficiency too low, raise thresholds (use more compute)
            for threshold in &mut self.compute_thresholds {
                *threshold *= 0.95;
            }
        } else if efficiency_diff < -0.1 {
            // Efficiency too high, lower thresholds (use less compute)
            for threshold in &mut self.compute_thresholds {
                *threshold *= 1.05;
            }
        }
        
        // Clamp thresholds
        for threshold in &mut self.compute_thresholds {
            *threshold = threshold.clamp(0.1, 0.9);
        }
    }
    
    /// Get compute cost for level
    pub fn get_compute_cost(&self, level: ComputeLevel) -> f32 {
        match level {
            ComputeLevel::Micro => 0.3,   // 30% of full compute
            ComputeLevel::Meso => 0.6,    // 60% of full compute
            ComputeLevel::Full => 1.0,    // 100% of full compute
        }
    }
    
    /// Set routing strategy
    pub fn set_routing_strategy(&mut self, strategy: RoutingStrategy) {
        self.routing_strategy = strategy;
    }
    
    /// Get compute statistics
    pub fn get_compute_stats(&self) -> (f32, f32, f32, HashMap<ComputeLevel, usize>) {
        let utilization = self.total_computations as f32 / 1000.0; // Normalized
        let cost = self.compute_efficiency; // Inverse of efficiency
        
        (self.compute_efficiency, utilization, cost, self.level_usage.clone())
    }
}

impl AdaptiveCompute for AdaptiveComputeAllocation {
    fn determine_compute_level(&self, 
        input: &ArrayD<f32>,
        hidden_state: &ArrayD<f32>
    ) -> DLResult<usize> {
        
        let level = match self.routing_strategy {
            RoutingStrategy::Threshold => {
                self.threshold_routing(input)?
            },
            RoutingStrategy::Adaptive => {
                self.adaptive_routing(input, hidden_state)?
            },
            RoutingStrategy::Complexity => {
                self.complexity_routing(input)?
            },
            RoutingStrategy::Importance => {
                self.importance_routing(input, hidden_state)?
            },
            RoutingStrategy::Hybrid => {
                self.hybrid_routing(input, hidden_state)?
            },
        };
        
        Ok(level.as_usize())
    }
    
    fn get_compute_stats(&self) -> (f32, f32, f32) {
        (self.compute_efficiency, self.get_compute_cost(ComputeLevel::Full), 1.0 - self.compute_efficiency)
    }
    
    fn set_compute_thresholds(&mut self, thresholds: Vec<f32>) -> DLResult<()> {
        if thresholds.len() != 3 {
            return Err(DeepLearningError::Configuration {
                reason: "Must provide exactly 3 compute thresholds".to_string(),
            });
        }
        
        self.compute_thresholds = thresholds;
        Ok(())
    }
}

/// Routing strategy implementations
impl AdaptiveComputeAllocation {
    /// Threshold-based routing
    fn threshold_routing(&self, input: &ArrayD<f32>) -> DLResult<ComputeLevel> {
        let complexity = self.compute_complexity(input)?;
        
        if complexity < self.compute_thresholds[0] {
            Ok(ComputeLevel::Micro)
        } else if complexity < self.compute_thresholds[1] {
            Ok(ComputeLevel::Meso)
        } else {
            Ok(ComputeLevel::Full)
        }
    }
    
    /// Adaptive routing dengan feedback
    fn adaptive_routing(&self, input: &ArrayD<f32>, hidden_state: &ArrayD<f32>) -> DLResult<ComputeLevel> {
        let importance = self.compute_importance(input, hidden_state)?;
        
        // Adjust thresholds based on current efficiency
        let adjusted_thresholds = self.compute_thresholds.iter()
            .map(|&t| t * (2.0 - self.compute_efficiency)) // Adjust based on efficiency
            .collect::<Vec<_>>();
        
        if importance < adjusted_thresholds[0] {
            Ok(ComputeLevel::Micro)
        } else if importance < adjusted_thresholds[1] {
            Ok(ComputeLevel::Meso)
        } else {
            Ok(ComputeLevel::Full)
        }
    }
    
    /// Complexity-based routing
    fn complexity_routing(&self, input: &ArrayD<f32>) -> DLResult<ComputeLevel> {
        let complexity = self.compute_complexity(input)?;
        
        // Use non-linear thresholds for complexity
        if complexity < 0.3 {
            Ok(ComputeLevel::Micro)
        } else if complexity < 0.7 {
            Ok(ComputeLevel::Meso)
        } else {
            Ok(ComputeLevel::Full)
        }
    }
    
    /// Importance-based routing
    fn importance_routing(&self, input: &ArrayD<f32>, hidden_state: &ArrayD<f32>) -> DLResult<ComputeLevel> {
        let importance = self.compute_importance(input, hidden_state)?;
        
        // Direct importance mapping
        if importance < 0.33 {
            Ok(ComputeLevel::Micro)
        } else if importance < 0.67 {
            Ok(ComputeLevel::Meso)
        } else {
            Ok(ComputeLevel::Full)
        }
    }
    
    /// Hybrid routing combining multiple strategies
    fn hybrid_routing(&self, input: &ArrayD<f32>, hidden_state: &ArrayD<f32>) -> DLResult<ComputeLevel> {
        let complexity = self.compute_complexity(input)?;
        let importance = self.compute_importance(input, hidden_state)?;
        
        // Weighted combination
        let combined_score = 0.6 * importance + 0.4 * complexity;
        
        if combined_score < 0.35 {
            Ok(ComputeLevel::Micro)
        } else if combined_score < 0.65 {
            Ok(ComputeLevel::Meso)
        } else {
            Ok(ComputeLevel::Full)
        }
    }
}

/// Advanced compute allocation features
impl AdaptiveComputeAllocation {
    /// Batch compute level determination
    pub fn batch_determine_levels(&self, 
        inputs: &[ArrayD<f32>],
        hidden_states: &[ArrayD<f32>]
    ) -> DLResult<Vec<ComputeLevel>> {
        
        if inputs.len() != hidden_states.len() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![inputs.len()],
                actual: vec![hidden_states.len()],
            });
        }
        
        let mut levels = Vec::with_capacity(inputs.len());
        
        for (input, hidden) in inputs.iter().zip(hidden_states.iter()) {
            let level_idx = self.determine_compute_level(input, hidden)?;
            let level = ComputeLevel::from_usize(level_idx);
            levels.push(level);
        }
        
        Ok(levels)
    }
    
    /// Compute allocation with memory constraints
    pub fn memory_constrained_allocation(&self,
        inputs: &[ArrayD<f32>],
        hidden_states: &[ArrayD<f32>],
        memory_budget_mb: usize
    ) -> DLResult<Vec<ComputeLevel>> {
        
        let base_levels = self.batch_determine_levels(inputs, hidden_states)?;
        
        // Estimate memory usage per level
        let memory_per_micro = 10; // MB
        let memory_per_meso = 25;   // MB
        let memory_per_full = 50;   // MB
        
        let mut total_memory = 0;
        let mut adjusted_levels = Vec::with_capacity(base_levels.len());
        
        for (_i, &level) in base_levels.iter().enumerate() {
            let memory_cost = match level {
                ComputeLevel::Micro => memory_per_micro,
                ComputeLevel::Meso => memory_per_meso,
                ComputeLevel::Full => memory_per_full,
            };
            
            if total_memory + memory_cost <= memory_budget_mb {
                total_memory += memory_cost;
                adjusted_levels.push(level);
            } else {
                // Downgrade to lower compute level
                let downgraded = match level {
                    ComputeLevel::Full => ComputeLevel::Meso,
                    ComputeLevel::Meso => ComputeLevel::Micro,
                    ComputeLevel::Micro => ComputeLevel::Micro,
                };
                adjusted_levels.push(downgraded);
            }
        }
        
        Ok(adjusted_levels)
    }
    
    /// Update routing dengan reinforcement learning
    pub fn update_routing_with_feedback(&mut self, 
        actual_performance: f32,
        expected_performance: f32
    ) -> DLResult<()> {
        
        let performance_error = expected_performance - actual_performance;
        
        // Update routing weights based on performance feedback
        let adjustment = self.adaptation_rate * performance_error;
        
        // Simple adjustment - in practice would use gradient-based update
        for threshold in &mut self.compute_thresholds {
            *threshold += adjustment;
            *threshold = threshold.clamp(0.1, 0.9);
        }
        
        // Adapt thresholds based on efficiency
        if self.total_computations % 100 == 0 {
            self.adapt_thresholds();
        }
        
        Ok(())
    }
    
    /// Get allocation recommendations
    pub fn get_allocation_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::with_capacity(3);
        
        if self.compute_efficiency < 0.5 {
            recommendations.push("Consider lowering compute thresholds to improve efficiency".to_string());
        }
        
        if self.compute_efficiency > 0.9 {
            recommendations.push("Consider raising compute thresholds to utilize more capacity".to_string());
        }
        
        let micro_ratio = *self.level_usage.get(&ComputeLevel::Micro).unwrap_or(&0) as f32 / 
                         self.total_computations as f32;
        
        if micro_ratio > 0.8 {
            recommendations.push("Most inputs use micro compute - consider increasing model capacity".to_string());
        }
        
        let full_ratio = *self.level_usage.get(&ComputeLevel::Full).unwrap_or(&0) as f32 / 
                        self.total_computations as f32;
        
        if full_ratio > 0.5 {
            recommendations.push("High full compute usage - consider model optimization".to_string());
        }
        
        recommendations
    }
    
    /// Allocate compute level based on input and relevance
    pub fn allocate_compute(&mut self, 
        _input: &ArrayD<f32>, 
        relevance: &ArrayD<f32>, 
        _input_size: usize, 
        _hidden_size: usize
    ) -> ComputeLevel {
        // Simple heuristic based on relevance magnitude
        let relevance_sum = relevance.iter().sum::<f32>();
        let relevance_avg = relevance_sum / relevance.len() as f32;
        
        let level = if relevance_avg > 0.8 {
            ComputeLevel::Full
        } else if relevance_avg > 0.5 {
            ComputeLevel::Meso
        } else {
            ComputeLevel::Micro
        };
        
        // Update statistics
        self.total_computations += 1;
        *self.level_usage.entry(level).or_insert(0) += 1;
        
        level
    }
}
