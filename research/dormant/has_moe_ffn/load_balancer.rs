//! Load Balancer for Optimal Computation Distribution

use crate::has_moe_ffn::{
    error::{HasMoeFfnError, Result},
    types::*,
};
use ndarray::ArrayD;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Load Balancer for distributing computation across experts
pub struct LoadBalancer {
    config: LoadBalancerConfig,
    expert_states: Vec<ExpertState>,
    strategy: Box<dyn LoadBalancingStrategyTrait>,
    metrics_collector: MetricsCollector,
    rebalance_timer: Instant,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(config: LoadBalancerConfig) -> Result<Self> {
        let strategy: Box<dyn LoadBalancingStrategyTrait> = match config.strategy {
            LoadBalancingStrategy::RoundRobin => Box::new(RoundRobinStrategy::new()),
            LoadBalancingStrategy::LeastLoaded => Box::new(LeastLoadedStrategy::new()),
            LoadBalancingStrategy::WeightedRoundRobin => Box::new(WeightedRoundRobinStrategy::new()),
            LoadBalancingStrategy::Adaptive => Box::new(AdaptiveStrategy::new()),
        };
        
        let expert_states = (0..8) // Default number of experts
            .map(|i| ExpertState::new(i))
            .collect();
        
        let metrics_collector = MetricsCollector::new();
        
        Ok(Self {
            config,
            expert_states,
            strategy,
            metrics_collector,
            rebalance_timer: Instant::now(),
        })
    }
    
    /// Balance routing decisions across experts
    pub fn balance(
        &mut self,
        routing_decisions: Vec<RoutingDecision>,
        experts: &[crate::has_moe_ffn::experts::StructuredSwiGLUExpert],
    ) -> Result<Vec<(usize, ArrayD<f32>)>> {
        // Update expert states based on current load
        self.update_expert_states(experts)?;
        
        // Apply load balancing strategy
        let balanced_decisions = self.strategy.apply(
            routing_decisions,
            &self.expert_states,
            &self.config,
        )?;
        
        // Create balanced expert assignments
        let mut assignments = Vec::new();
        
        for decision in balanced_decisions {
            // Check if expert is available
            if self.is_expert_available(decision.expert_id) {
                // Create input tensor for this expert
                let expert_input = self.create_expert_input(&decision)?;
                assignments.push((decision.expert_id, expert_input));
                
                // Update expert state
                self.record_expert_assignment(decision.expert_id);
            } else {
                // Find alternative expert
                if let Some(alternative_id) = self.find_alternative_expert(decision.expert_id) {
                    let expert_input = self.create_expert_input(&decision)?;
                    assignments.push((alternative_id, expert_input));
                    self.record_expert_assignment(alternative_id);
                } else {
                    return Err(HasMoeFfnError::all_experts_busy());
                }
            }
        }
        
        // Check if rebalancing is needed
        if self.should_rebalance() {
            self.rebalance()?;
        }
        
        Ok(assignments)
    }
    
    /// Update expert states
    fn update_expert_states(
        &mut self,
        experts: &[crate::has_moe_ffn::experts::StructuredSwiGLUExpert],
    ) -> Result<()> {
        // Collect performance metrics first
        let mut performance_metrics = Vec::new();
        for expert in experts {
            let avg_time = expert.performance_metrics()
                .get("avg_forward_time_ms")
                .copied()
                .unwrap_or(0.0);
            performance_metrics.push(avg_time);
        }
        
        // Update states
        for i in 0..self.expert_states.len() {
            if i < experts.len() {
                let state = &mut self.expert_states[i];
                
                // Update performance metrics from expert
                state.average_response_time = performance_metrics[i];
                
                // Update queue length based on recent assignments
                state.queue_length = self.metrics_collector.get_queue_length(i);
                
                // Update success rate
                state.success_rate = self.metrics_collector.get_success_rate(i);
                
                // Calculate current load separately to avoid borrow issues
                let queue_factor = state.queue_length as f32 / self.config.max_queue_length as f32;
                let response_factor = state.average_response_time / 1000.0;
                let success_factor = 1.0 - state.success_rate;
                state.current_load = (queue_factor * 0.4 + response_factor * 0.4 + success_factor * 0.2).min(1.0);
            }
        }
        
        Ok(())
    }
    
    /// Calculate expert load
    fn calculate_expert_load(&self, state: &ExpertState) -> f32 {
        let queue_factor = state.queue_length as f32 / self.config.max_queue_length as f32;
        let response_factor = state.average_response_time / 1000.0; // Normalize to seconds
        let success_factor = 1.0 - state.success_rate;
        
        // Weighted combination of factors
        (queue_factor * 0.4 + response_factor * 0.4 + success_factor * 0.2).min(1.0)
    }
    
    /// Check if expert is available
    fn is_expert_available(&self, expert_id: usize) -> bool {
        if expert_id >= self.expert_states.len() {
            return false;
        }
        
        let state = &self.expert_states[expert_id];
        state.queue_length < self.config.max_queue_length &&
        state.current_load < 0.9 && // 90% load threshold
        state.is_healthy
    }
    
    /// Find alternative expert
    fn find_alternative_expert(&self, preferred_id: usize) -> Option<usize> {
        // Find expert with lowest load
        let mut best_candidate = None;
        let mut lowest_load = f32::INFINITY;
        
        for (i, state) in self.expert_states.iter().enumerate() {
            if i != preferred_id && self.is_expert_available(i) {
                if state.current_load < lowest_load {
                    lowest_load = state.current_load;
                    best_candidate = Some(i);
                }
            }
        }
        
        best_candidate
    }
    
    /// Create expert input tensor
    fn create_expert_input(&self, decision: &RoutingDecision) -> Result<ArrayD<f32>> {
        // This is a simplified implementation
        // In practice, this would extract the relevant tokens from the input
        let input_dim = 4096; // Default dimension
        let mut input_data = vec![0.0f32; input_dim];
        
        // Fill with some pattern based on the decision
        for (i, &token_id) in decision.input_tokens.iter().enumerate() {
            if i < input_dim {
                input_data[i] = token_id as f32 * decision.confidence;
            }
        }
        
        Ok(ndarray::ArrayD::from_shape_vec(vec![input_dim], input_data)
            .map_err(|_| HasMoeFfnError::tensor("Failed to create input tensor"))?)
    }
    
    /// Record expert assignment
    fn record_expert_assignment(&mut self, expert_id: usize) {
        if expert_id < self.expert_states.len() {
            let state = &mut self.expert_states[expert_id];
            state.queue_length = state.queue_length.saturating_add(1);
            state.last_assignment = Instant::now();
            
            // Record in metrics collector
            self.metrics_collector.record_assignment(expert_id);
        }
    }
    
    /// Check if rebalancing is needed
    fn should_rebalance(&self) -> bool {
        self.rebalance_timer.elapsed() >= Duration::from_millis(self.config.rebalance_interval_ms)
    }
    
    /// Rebalance expert loads
    fn rebalance(&mut self) -> Result<()> {
        // Calculate load imbalance
        let loads: Vec<f32> = self.expert_states.iter()
            .map(|state| state.current_load)
            .collect();
        
        if loads.is_empty() {
            return Ok(());
        }
        
        let mean_load = loads.iter().sum::<f32>() / loads.len() as f32;
        let variance = loads.iter()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f32>() / loads.len() as f32;
        
        // If variance is high, trigger rebalancing
        if variance > 0.1 {
            self.perform_rebalancing()?;
        }
        
        self.rebalance_timer = Instant::now();
        Ok(())
    }
    
    /// Perform actual rebalancing
    fn perform_rebalancing(&mut self) -> Result<()> {
        // Sort experts by load
        let mut expert_loads: Vec<(usize, f32)> = self.expert_states.iter()
            .enumerate()
            .map(|(i, state)| (i, state.current_load))
            .collect();
        
        expert_loads.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Redistribute load from most loaded to least loaded
        let (most_loaded_id, most_loaded) = expert_loads.last().unwrap();
        let (least_loaded_id, _) = expert_loads.first().unwrap();
        
        if *most_loaded > 0.8 && *least_loaded_id != *most_loaded_id {
            // Mark for potential redistribution
            self.expert_states[*most_loaded_id].needs_redistribution = true;
            self.expert_states[*least_loaded_id].can_accept_more = true;
        }
        
        Ok(())
    }
    
    /// Get load balancing statistics
    pub fn get_load_stats(&self) -> LoadBalancingStats {
        let total_load: f32 = self.expert_states.iter()
            .map(|state| state.current_load)
            .sum();
        
        let average_load = total_load / self.expert_states.len() as f32;
        
        let max_load = self.expert_states.iter()
            .map(|state| state.current_load)
            .fold(0.0f32, f32::max);
        
        let min_load = self.expert_states.iter()
            .map(|state| state.current_load)
            .fold(f32::INFINITY, f32::min);
        
        let utilization_variance = self.expert_states.iter()
            .map(|state| (state.current_load - average_load).powi(2))
            .sum::<f32>() / self.expert_states.len() as f32;
        
        LoadBalancingStats {
            average_load,
            max_load,
            min_load,
            utilization_variance,
            total_experts: self.expert_states.len(),
            healthy_experts: self.expert_states.iter().filter(|s| s.is_healthy).count(),
        }
    }
}

/// Expert state for load tracking
#[derive(Debug, Clone)]
pub struct ExpertState {
    pub expert_id: usize,
    pub current_load: f32,
    pub average_response_time: f32,
    pub queue_length: usize,
    pub success_rate: f32,
    pub is_healthy: bool,
    pub last_assignment: Instant,
    pub needs_redistribution: bool,
    pub can_accept_more: bool,
}

impl ExpertState {
    pub fn new(expert_id: usize) -> Self {
        Self {
            expert_id,
            current_load: 0.0,
            average_response_time: 0.0,
            queue_length: 0,
            success_rate: 1.0,
            is_healthy: true,
            last_assignment: Instant::now(),
            needs_redistribution: false,
            can_accept_more: true,
        }
    }
}

/// Load balancing strategy trait
pub trait LoadBalancingStrategyTrait {
    fn apply(
        &mut self,
        routing_decisions: Vec<RoutingDecision>,
        expert_states: &[ExpertState],
        config: &LoadBalancerConfig,
    ) -> Result<Vec<RoutingDecision>>;
}

/// Round-robin load balancing strategy
pub struct RoundRobinStrategy {
    current_expert: usize,
}

impl RoundRobinStrategy {
    pub fn new() -> Self {
        Self { current_expert: 0 }
    }
}

impl LoadBalancingStrategyTrait for RoundRobinStrategy {
    fn apply(
        &mut self,
        mut routing_decisions: Vec<RoutingDecision>,
        expert_states: &[ExpertState],
        _config: &LoadBalancerConfig,
    ) -> Result<Vec<RoutingDecision>> {
        for decision in &mut routing_decisions {
            // Find next available expert
            let mut attempts = 0;
            while attempts < expert_states.len() {
                if expert_states[self.current_expert].is_healthy &&
                   expert_states[self.current_expert].queue_length < 100 {
                    decision.expert_id = self.current_expert;
                    break;
                }
                
                self.current_expert = (self.current_expert + 1) % expert_states.len();
                attempts += 1;
            }
            
            self.current_expert = (self.current_expert + 1) % expert_states.len();
        }
        
        Ok(routing_decisions)
    }
}

/// Least-loaded load balancing strategy
pub struct LeastLoadedStrategy;

impl LeastLoadedStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl LoadBalancingStrategyTrait for LeastLoadedStrategy {
    fn apply(
        &mut self,
        mut routing_decisions: Vec<RoutingDecision>,
        expert_states: &[ExpertState],
        _config: &LoadBalancerConfig,
    ) -> Result<Vec<RoutingDecision>> {
        for decision in &mut routing_decisions {
            // Find expert with minimum load
            let mut best_expert = 0;
            let mut min_load = f32::INFINITY;
            
            for (i, state) in expert_states.iter().enumerate() {
                if state.is_healthy && state.current_load < min_load {
                    min_load = state.current_load;
                    best_expert = i;
                }
            }
            
            decision.expert_id = best_expert;
        }
        
        Ok(routing_decisions)
    }
}

/// Weighted round-robin load balancing strategy
pub struct WeightedRoundRobinStrategy {
    current_weights: Vec<f32>,
}

impl WeightedRoundRobinStrategy {
    pub fn new() -> Self {
        Self {
            current_weights: Vec::new(),
        }
    }
}

impl LoadBalancingStrategyTrait for WeightedRoundRobinStrategy {
    fn apply(
        &mut self,
        mut routing_decisions: Vec<RoutingDecision>,
        expert_states: &[ExpertState],
        _config: &LoadBalancerConfig,
    ) -> Result<Vec<RoutingDecision>> {
        // Initialize weights if needed
        if self.current_weights.len() != expert_states.len() {
            self.current_weights = vec![1.0; expert_states.len()];
        }
        
        for decision in &mut routing_decisions {
            // Select expert based on weights
            let total_weight: f32 = self.current_weights.iter().sum();
            let mut random_value = fastrand::f32() * total_weight;
            
            for (i, &weight) in self.current_weights.iter().enumerate() {
                if random_value <= weight && expert_states[i].is_healthy {
                    decision.expert_id = i;
                    self.current_weights[i] *= 0.9; // Decrease weight
                    break;
                }
                random_value -= weight;
            }
            
            // Gradually restore weights
            for weight in &mut self.current_weights {
                *weight = (*weight * 1.1).min(2.0);
            }
        }
        
        Ok(routing_decisions)
    }
}

/// Adaptive load balancing strategy
pub struct AdaptiveStrategy {
    performance_history: HashMap<usize, VecDeque<f32>>,
}

impl AdaptiveStrategy {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
        }
    }
}

impl LoadBalancingStrategyTrait for AdaptiveStrategy {
    fn apply(
        &mut self,
        mut routing_decisions: Vec<RoutingDecision>,
        expert_states: &[ExpertState],
        config: &LoadBalancerConfig,
    ) -> Result<Vec<RoutingDecision>> {
        // Update performance history
        for state in expert_states {
            let history = self.performance_history.entry(state.expert_id)
                .or_insert_with(|| VecDeque::with_capacity(100));
            
            history.push_back(state.success_rate);
            if history.len() > 100 {
                history.pop_front();
            }
        }
        
        for decision in &mut routing_decisions {
            let mut best_expert = 0;
            let mut best_score = f32::NEG_INFINITY;
            
            for (i, state) in expert_states.iter().enumerate() {
                if !state.is_healthy {
                    continue;
                }
                
                // Calculate adaptive score
                let load_score = 1.0 - state.current_load;
                let performance_score = self.get_average_performance(i);
                let queue_score = 1.0 - (state.queue_length as f32 / config.max_queue_length as f32);
                
                let combined_score = load_score * 0.4 + performance_score * 0.4 + queue_score * 0.2;
                
                if combined_score > best_score {
                    best_score = combined_score;
                    best_expert = i;
                }
            }
            
            decision.expert_id = best_expert;
        }
        
        Ok(routing_decisions)
    }
}

impl AdaptiveStrategy {
    fn get_average_performance(&self, expert_id: usize) -> f32 {
        if let Some(history) = self.performance_history.get(&expert_id) {
            if history.is_empty() {
                return 0.5;
            }
            history.iter().sum::<f32>() / history.len() as f32
        } else {
            0.5
        }
    }
}

/// Metrics collector for load balancing
pub struct MetricsCollector {
    assignment_history: HashMap<usize, VecDeque<Instant>>,
    success_history: HashMap<usize, VecDeque<bool>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            assignment_history: HashMap::new(),
            success_history: HashMap::new(),
        }
    }
    
    pub fn record_assignment(&mut self, expert_id: usize) {
        let history = self.assignment_history.entry(expert_id)
            .or_insert_with(|| VecDeque::with_capacity(1000));
        
        history.push_back(Instant::now());
        
        // Keep only recent assignments
        while history.len() > 1000 {
            history.pop_front();
        }
    }
    
    pub fn record_success(&mut self, expert_id: usize, success: bool) {
        let history = self.success_history.entry(expert_id)
            .or_insert_with(|| VecDeque::with_capacity(1000));
        
        history.push_back(success);
        
        // Keep only recent successes
        while history.len() > 1000 {
            history.pop_front();
        }
    }
    
    pub fn get_queue_length(&self, expert_id: usize) -> usize {
        self.assignment_history.get(&expert_id)
            .map(|history| history.len())
            .unwrap_or(0)
    }
    
    pub fn get_success_rate(&self, expert_id: usize) -> f32 {
        if let Some(history) = self.success_history.get(&expert_id) {
            if history.is_empty() {
                return 1.0;
            }
            history.iter().map(|&success| if success { 1.0 } else { 0.0 }).sum::<f32>() / history.len() as f32
        } else {
            1.0
        }
    }
}

/// Load balancing statistics
#[derive(Debug, Clone)]
pub struct LoadBalancingStats {
    pub average_load: f32,
    pub max_load: f32,
    pub min_load: f32,
    pub utilization_variance: f32,
    pub total_experts: usize,
    pub healthy_experts: usize,
}
