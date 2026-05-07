//! Expert Router with Context-Aware Selection

use crate::has_moe_ffn::{
    error::{HasMoeFfnError, Result},
    types::*,
};
use ndarray::{ArrayD, Array1, Array2};
use std::collections::HashMap;

/// Expert Router for context-aware expert selection
pub struct ExpertRouter {
    config: RouterConfig,
    gate_weights: Array2<f32>,
    context_analyzer: ContextAnalyzer,
    routing_history: Vec<RoutingDecision>,
    load_tracker: LoadTracker,
}

impl ExpertRouter {
    /// Create new expert router
    pub fn new(config: RouterConfig) -> Result<Self> {
        let gate_weights = Array2::zeros((config.num_experts, config.num_experts));
        let context_analyzer = ContextAnalyzer::new(config.use_context_analysis)?;
        let load_tracker = LoadTracker::new(config.num_experts);
        
        Ok(Self {
            config,
            gate_weights,
            context_analyzer,
            routing_history: Vec::new(),
            load_tracker,
        })
    }
    
    /// Route input to selected experts
    pub fn route(&mut self, input: &ArrayD<f32>) -> Result<Vec<RoutingDecision>> {
        // Step 1: Analyze context
        let context_analysis = self.context_analyzer.analyze(input)?;
        
        // Step 2: Compute expert scores
        let expert_scores = self.compute_expert_scores(input, &context_analysis)?;
        
        // Step 3: Apply load balancing
        let balanced_scores = self.apply_load_balancing(&expert_scores)?;
        
        // Step 4: Select top-k experts
        let selected_experts = self.select_top_k_experts(&balanced_scores)?;
        
        // Step 5: Create routing decisions
        let routing_decisions = self.create_routing_decisions(
            selected_experts,
            &context_analysis,
            input,
        )?;
        
        // Step 6: Update routing history
        self.update_routing_history(&routing_decisions);
        
        Ok(routing_decisions)
    }
    
    /// Compute expert scores based on input and context
    fn compute_expert_scores(
        &self,
        input: &ArrayD<f32>,
        context: &ContextAnalysis,
    ) -> Result<Array1<f32>> {
        let mut scores = Array1::zeros(self.config.num_experts);
        
        for i in 0..self.config.num_experts {
            // Base score from gate weights
            let base_score = self.compute_gate_score(input, i)?;
            
            // Context-aware score
            let context_score = self.compute_context_score(context, i)?;
            
            // Load penalty
            let load_penalty = self.load_tracker.get_load_penalty(i);
            
            // Combine scores
            let combined_score = base_score * context_score * (1.0 - load_penalty);
            scores[i] = combined_score;
        }
        
        // Apply temperature and softmax
        self.apply_temperature_softmax(scores)
    }
    
    /// Compute gate score for specific expert
    fn compute_gate_score(&self, input: &ArrayD<f32>, expert_id: usize) -> Result<f32> {
        // Simple dot product for demonstration
        // In practice, this would be a learned neural network
        let input_flat = input.as_slice().ok_or_else(|| {
            HasMoeFfnError::tensor("Input cannot be flattened")
        })?;
        
        if input_flat.is_empty() {
            return Ok(0.0);
        }
        
        let input_sum: f32 = input_flat.iter().sum();
        let normalized_input = input_sum / input_flat.len() as f32;
        
        // Use expert_id as seed for pseudo-random but consistent scoring
        let expert_factor = (expert_id as f32 + 1.0) / (self.config.num_experts as f32);
        
        Ok(normalized_input * expert_factor)
    }
    
    /// Compute context-aware score
    fn compute_context_score(&self, context: &ContextAnalysis, expert_id: usize) -> Result<f32> {
        // Map content type to expert specialization
        let specialization_score = match context.content_type {
            ExpertSpecialization::Reasoning => if expert_id % 4 == 0 { 1.2 } else { 0.8 },
            ExpertSpecialization::Coding => if expert_id % 4 == 1 { 1.3 } else { 0.7 },
            ExpertSpecialization::Mathematics => if expert_id % 4 == 2 { 1.4 } else { 0.6 },
            ExpertSpecialization::Language => if expert_id % 4 == 3 { 1.2 } else { 0.8 },
            _ => 1.0,
        };
        
        // Adjust by complexity score
        let complexity_factor = 1.0 + context.complexity_score * 0.2;
        
        Ok(specialization_score * complexity_factor)
    }
    
    /// Apply temperature and softmax to scores
    fn apply_temperature_softmax(&self, mut scores: Array1<f32>) -> Result<Array1<f32>> {
        // Apply temperature
        scores.mapv_inplace(|x| x / self.config.temperature);
        
        // Numerical stability: subtract max
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        scores.mapv_inplace(|x| x - max_score);
        
        // Exponentiate and normalize
        scores.mapv_inplace(|x| x.exp());
        let sum: f32 = scores.sum();
        
        if sum == 0.0 {
            return Err(HasMoeFfnError::router("Score sum is zero"));
        }
        
        scores.mapv_inplace(|x| x / sum);
        Ok(scores)
    }
    
    /// Apply load balancing to scores
    fn apply_load_balancing(&self, scores: &Array1<f32>) -> Result<Array1<f32>> {
        let mut balanced_scores = scores.clone();
        
        for (i, &score) in scores.iter().enumerate() {
            let load_factor = self.load_tracker.get_load_factor(i);
            let penalty = self.config.load_balance_factor * load_factor;
            balanced_scores[i] = score * (1.0 - penalty);
        }
        
        // Re-normalize
        let sum: f32 = balanced_scores.sum();
        if sum > 0.0 {
            balanced_scores.mapv_inplace(|x| x / sum);
        }
        
        Ok(balanced_scores)
    }
    
    /// Select top-k experts
    fn select_top_k_experts(&self, scores: &Array1<f32>) -> Result<Vec<(usize, f32)>> {
        let mut expert_scores: Vec<(usize, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        // Sort by score (descending)
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        let top_k = self.config.top_k.min(expert_scores.len());
        Ok(expert_scores.into_iter().take(top_k).collect())
    }
    
    /// Create routing decisions
    fn create_routing_decisions(
        &self,
        selected_experts: Vec<(usize, f32)>,
        context: &ContextAnalysis,
        input: &ArrayD<f32>,
    ) -> Result<Vec<RoutingDecision>> {
        let mut decisions = Vec::new();
        
        for (expert_id, confidence) in selected_experts {
            let decision = RoutingDecision {
                expert_id,
                confidence,
                specialization: context.content_type.clone(),
                input_tokens: self.extract_input_tokens(input)?,
            };
            decisions.push(decision);
        }
        
        Ok(decisions)
    }
    
    /// Extract input tokens (simplified)
    fn extract_input_tokens(&self, input: &ArrayD<f32>) -> Result<Vec<usize>> {
        // This is a simplified token extraction
        // In practice, this would work with actual token embeddings
        let input_flat = input.as_slice().ok_or_else(|| {
            HasMoeFfnError::tensor("Input cannot be flattened")
        })?;
        
        let mut tokens = Vec::new();
        for (i, &value) in input_flat.iter().enumerate() {
            if value.abs() > 0.1 { // Threshold for "active" tokens
                tokens.push(i);
            }
        }
        
        Ok(tokens)
    }
    
    /// Update routing history
    fn update_routing_history(&mut self, decisions: &[RoutingDecision]) {
        self.routing_history.extend(decisions.iter().cloned());
        
        // Keep only recent history (last 1000 decisions)
        if self.routing_history.len() > 1000 {
            let remove_count = self.routing_history.len() - 1000;
            self.routing_history.drain(0..remove_count);
        }
        
        // Update load tracker
        for decision in decisions {
            self.load_tracker.record_usage(decision.expert_id);
        }
    }
    
    /// Route input and return single best routing decision for SACA integration
    pub fn route_single(&mut self, input: &ArrayD<f32>) -> Result<RoutingDecision> {
        let decisions = self.route(input)?;
        
        // Return the decision with highest confidence
        let best_decision = decisions.into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or_else(|| HasMoeFfnError::router("No routing decisions available"))?;
        
        Ok(best_decision)
    }
    
    /// Get routing statistics
    pub fn get_routing_stats(&self) -> RoutingStats {
        let mut expert_usage = HashMap::new();
        
        for decision in &self.routing_history {
            *expert_usage.entry(decision.expert_id).or_insert(0) += 1;
        }
        
        let total_decisions = self.routing_history.len();
        let expert_utilization = expert_usage
            .iter()
            .map(|(&id, &count)| (id, count as f32 / total_decisions as f32))
            .collect();
        
        RoutingStats {
            total_routings: total_decisions,
            expert_utilization,
            average_confidence: self.calculate_average_confidence(),
            load_balance_score: self.calculate_load_balance_score(),
        }
    }
    
    fn calculate_average_confidence(&self) -> f32 {
        if self.routing_history.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = self.routing_history.iter().map(|d| d.confidence).sum();
        sum / self.routing_history.len() as f32
    }
    
    fn calculate_load_balance_score(&self) -> f32 {
        let loads: Vec<f32> = (0..self.config.num_experts)
            .map(|i| self.load_tracker.get_load_factor(i))
            .collect();
        
        if loads.is_empty() {
            return 0.0;
        }
        
        let mean = loads.iter().sum::<f32>() / loads.len() as f32;
        let variance = loads.iter().map(|&load| (load - mean).powi(2)).sum::<f32>() / loads.len() as f32;
        
        // Lower variance = better load balancing
        (1.0 - variance).max(0.0)
    }
}

/// Context Analyzer for understanding input content
pub struct ContextAnalyzer {
    enabled: bool,
}

impl ContextAnalyzer {
    pub fn new(enabled: bool) -> Result<Self> {
        Ok(Self { enabled })
    }
    
    pub fn analyze(&self, input: &ArrayD<f32>) -> Result<ContextAnalysis> {
        if !self.enabled {
            return Ok(ContextAnalysis {
                content_type: ExpertSpecialization::General,
                complexity_score: 0.5,
                required_experts: vec![0],
                confidence_scores: vec![1.0],
            });
        }
        
        // Simplified context analysis
        let input_flat = input.as_slice().ok_or_else(|| {
            HasMoeFfnError::tensor("Input cannot be flattened")
        })?;
        
        let complexity_score = self.calculate_complexity(input_flat);
        let content_type = self.detect_content_type(input_flat);
        
        Ok(ContextAnalysis {
            content_type,
            complexity_score,
            required_experts: vec![0, 1], // Default to first two experts
            confidence_scores: vec![0.8, 0.6],
        })
    }
    
    fn calculate_complexity(&self, input: &[f32]) -> f32 {
        if input.is_empty() {
            return 0.0;
        }
        
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        
        // Normalize complexity score to [0, 1]
        (variance.sqrt() / (1.0 + variance.sqrt())).min(1.0)
    }
    
    fn detect_content_type(&self, input: &[f32]) -> ExpertSpecialization {
        if input.is_empty() {
            return ExpertSpecialization::General;
        }
        
        // Simplified content type detection based on input patterns
        let pattern_score = self.analyze_patterns(input);
        
        match pattern_score {
            score if score > 0.8 => ExpertSpecialization::Mathematics,
            score if score > 0.6 => ExpertSpecialization::Reasoning,
            score if score > 0.4 => ExpertSpecialization::Coding,
            score if score > 0.2 => ExpertSpecialization::Language,
            _ => ExpertSpecialization::General,
        }
    }
    
    fn analyze_patterns(&self, input: &[f32]) -> f32 {
        // Simplified pattern analysis
        let positive_ratio = input.iter().filter(|&&x| x > 0.0).count() as f32 / input.len() as f32;
        let magnitude_avg = input.iter().map(|&x| x.abs()).sum::<f32>() / input.len() as f32;
        
        (positive_ratio + magnitude_avg) / 2.0
    }
}

/// Load Tracker for monitoring expert usage
pub struct LoadTracker {
    expert_loads: Vec<f32>,
    usage_history: Vec<Vec<usize>>,
}

impl LoadTracker {
    pub fn new(num_experts: usize) -> Self {
        Self {
            expert_loads: vec![0.0; num_experts],
            usage_history: Vec::new(),
        }
    }
    
    pub fn record_usage(&mut self, expert_id: usize) {
        if expert_id < self.expert_loads.len() {
            self.expert_loads[expert_id] += 1.0;
        }
    }
    
    pub fn get_load_factor(&self, expert_id: usize) -> f32 {
        if expert_id >= self.expert_loads.len() {
            return 1.0;
        }
        
        let total_load: f32 = self.expert_loads.iter().sum();
        if total_load == 0.0 {
            return 0.0;
        }
        
        self.expert_loads[expert_id] / total_load
    }
    
    pub fn get_load_penalty(&self, expert_id: usize) -> f32 {
        let load_factor = self.get_load_factor(expert_id);
        // Exponential penalty for high load
        (load_factor * 2.0).min(1.0)
    }
    
    /// Get routing statistics
    pub fn get_routing_stats(&self) -> RoutingStats {
        use std::collections::HashMap;
        
        let mut expert_utilization = HashMap::new();
        for (i, &load) in self.expert_loads.iter().enumerate() {
            expert_utilization.insert(i, load);
        }
        
        RoutingStats {
            total_routings: 0,
            expert_utilization,
            average_confidence: 0.0,
            load_balance_score: 1.0,
        }
    }
    
    /// Get efficiency score
    pub fn get_efficiency_score(&self) -> f32 {
        // Simplified efficiency calculation
        let stats = self.get_routing_stats();
        stats.load_balance_score
    }
}

/// Routing statistics
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub total_routings: usize,
    pub expert_utilization: HashMap<usize, f32>,
    pub average_confidence: f32,
    pub load_balance_score: f32,
}
