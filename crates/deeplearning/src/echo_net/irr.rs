//! Iterative Resonance Reasoner (IRR)
//!
//! Block 6 dari ECHO-Net Ω
//!
//! Upgrade terbesar dengan reasoning multi-step internal:
//! 
//! Initial query:
//! q_0 = W_q * h_t
//!
//! Loop reasoning:
//! R_n = Re(q_n * H)
//! q_{n+1} = f(q_n, R_n)
//!
//! dengan recurrent refinement:
//! q_{n+1} = q_n + α * tanh(W_r * R_n)
//!
//! Artinya:
//! - Model berpikir iteratif
//! - Query berkembang
//! - Reasoning bisa mendalam
//! - Mirip internal deliberation

use crate::{DLResult, DeepLearningError};
use crate::autograd::Tensor;
use crate::echo_net::utils::ResonanceCalculator;
use ndarray::{ArrayD, Array2, Array1};

/// Reasoning step result
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub query: ArrayD<f32>,
    pub resonance: ArrayD<f32>,
    pub confidence: f32,
    pub novelty: f32,
    pub coherence: f32,
}

impl ReasoningStep {
    pub fn new(step: usize, query: ArrayD<f32>, resonance: ArrayD<f32>) -> Self {
        Self {
            step_number: step,
            query,
            resonance,
            confidence: 0.0,
            novelty: 0.0,
            coherence: 0.0,
        }
    }
}

/// Iterative Resonance Reasoner implementation
#[derive(Debug, Clone)]
pub struct IterativeResonanceReasoner {
    // Reasoning parameters
    reasoning_steps: usize,
    reasoning_alpha: f32,
    convergence_threshold: f32,
    max_iterations: usize,
    
    // Query transformation weights
    query_weights: Array2<f32>,
    refinement_weights: Array2<f32>,
    output_weights: Array2<f32>,
    
    // Resonance computation
    resonance_calculator: ResonanceCalculator,
    
    // Reasoning history
    reasoning_history: Vec<ReasoningStep>,
    query_evolution: Vec<ArrayD<f32>>,
    
    // Convergence tracking
    convergence_scores: Vec<f32>,
    novelty_scores: Vec<f32>,
    coherence_scores: Vec<f32>,
    
    // Adaptive reasoning
    adaptive_steps: bool,
    early_termination: bool,
    step_importance_weights: Array1<f32>,
    
    // Memory integration
    memory_integration_weight: f32,
    attention_weights: Array1<f32>,
    
    // Reasoning state
    current_step: usize,
    is_converged: bool,
    final_reasoning: Option<ArrayD<f32>>,
    
    // Performance metrics
    average_confidence: f32,
    reasoning_depth: f32,
    query_stability: f32,
}

impl IterativeResonanceReasoner {
    /// Create new Iterative Resonance Reasoner
    pub fn new(
        input_dim: usize,
        reasoning_steps: usize,
        reasoning_alpha: f32,
        convergence_threshold: f32,
    ) -> DLResult<Self> {
        // Initialize weights
        let query_weights = Self::xavier_init(input_dim, input_dim);
        let refinement_weights = Self::xavier_init(input_dim, input_dim);
        let output_weights = Self::xavier_init(input_dim, input_dim);
        
        // Initialize step importance weights
        let step_importance_weights = Array1::from_vec(vec![0.5, 0.3, 0.15, 0.05]); // Decreasing importance
        
        Ok(Self {
            reasoning_steps,
            reasoning_alpha,
            convergence_threshold,
            max_iterations: reasoning_steps * 2,
            query_weights,
            refinement_weights,
            output_weights,
            resonance_calculator: ResonanceCalculator,
            reasoning_history: Vec::new(),
            query_evolution: Vec::new(),
            convergence_scores: Vec::new(),
            novelty_scores: Vec::new(),
            coherence_scores: Vec::new(),
            adaptive_steps: true,
            early_termination: true,
            step_importance_weights,
            memory_integration_weight: 0.3,
            attention_weights: Array1::ones(reasoning_steps),
            current_step: 0,
            is_converged: false,
            final_reasoning: None,
            average_confidence: 0.0,
            reasoning_depth: 0.0,
            query_stability: 0.0,
        })
    }
    
    /// Forward pass - perform iterative resonance reasoning
    pub fn forward(&mut self, input: &ArrayD<f32>, holographic_memory: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        // Reset reasoning state
        self.reset_reasoning_state()?;
        
        // Initialize query
        let mut current_query = self.initialize_query(input)?;
        
        // Perform iterative reasoning
        for step in 0..self.max_iterations {
            self.current_step = step;
            
            // Compute resonance with holographic memory
            let resonance = self.compute_resonance(&current_query, holographic_memory)?;
            
            // Create reasoning step
            let mut reasoning_step = ReasoningStep::new(step, current_query.clone(), resonance.clone());
            
            // Calculate step metrics
            reasoning_step.confidence = self.calculate_confidence(&resonance)?;
            reasoning_step.novelty = self.calculate_novelty(&current_query, &resonance)?;
            reasoning_step.coherence = self.calculate_coherence(&current_query, &resonance)?;
            
            // Store reasoning step
            self.reasoning_history.push(reasoning_step.clone());
            self.query_evolution.push(current_query.clone());
            
            // Update metrics
            self.convergence_scores.push(reasoning_step.confidence);
            self.novelty_scores.push(reasoning_step.novelty);
            self.coherence_scores.push(reasoning_step.coherence);
            
            // Check for convergence
            if self.check_convergence() {
                self.is_converged = true;
                break;
            }
            
            // Refine query for next step
            current_query = self.refine_query(&current_query, &resonance)?;
            
            // Check for early termination
            if self.early_termination && step >= self.reasoning_steps {
                break;
            }
        }
        
        // Generate final reasoning output
        let final_output = self.generate_final_output()?;
        self.final_reasoning = Some(final_output.clone());
        
        // Update performance metrics
        self.update_performance_metrics()?;
        
        Ok(final_output)
    }
    
    /// Initialize query from input
    fn initialize_query(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_view: Array2<f32> = input.view().into_dimensionality().expect("input is 2D").to_owned();
        let query = input_view.dot(&self.query_weights.t());
        
        // Apply non-linearity
        let activated_query = query.mapv(|x| x.tanh());
        
        Ok(activated_query.into_dimensionality().expect("query is 2D"))
    }
    
    /// Compute resonance between query and holographic memory
    fn compute_resonance(&self, query: &ArrayD<f32>, holographic_memory: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        // Reshape for resonance calculation
        let query_1d = self.to_1d_array(query)?;
        let memory_1d = self.to_1d_array(holographic_memory)?;
        
        // Compute resonance coefficient
        let resonance_coeff = ResonanceCalculator::resonance_coefficient(&query_1d, &memory_1d);
        
        // Apply resonance operation
        let mut resonance = holographic_memory.clone();
        resonance.mapv_inplace(|x| x * resonance_coeff);
        
        // Add attention-weighted contribution
        let attention_weighted = self.apply_attention(query, holographic_memory)?;
        
        // Combine resonance with attention
        let combined_resonance = resonance.mapv(|r| r * self.memory_integration_weight) + 
                               attention_weighted.mapv(|a| a * (1.0 - self.memory_integration_weight));
        
        Ok(combined_resonance)
    }
    
    /// Apply attention to query-memory interaction
    fn apply_attention(&self, query: &ArrayD<f32>, memory: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let query_1d = self.to_1d_array(query)?;
        let memory_1d = self.to_1d_array(memory)?;
        
        // Compute attention weights
        let chunk_size = query_1d.len().min(memory_1d.len());
        let mut attention_scores = Vec::with_capacity(chunk_size);
        
        for i in 0..chunk_size {
            let query_chunk = Array1::from(vec![query_1d[i]]);
            let memory_chunk = Array1::from(vec![memory_1d[i]]);
            
            let attention_score = self.cosine_similarity(&query_chunk, &memory_chunk);
            attention_scores.push(attention_score);
        }
        
        // Normalize attention scores
        let total_score: f32 = attention_scores.iter().sum();
        if total_score > 0.0 {
            for score in &mut attention_scores {
                *score /= total_score;
            }
        }
        
        // Apply attention to memory
        let mut attended_memory = memory_1d.clone();
        for (i, &attention_score) in attention_scores.iter().enumerate() {
            if i < attended_memory.len() {
                attended_memory[i] *= attention_score;
            }
        }
        
        // Convert back to original shape
        self.to_original_shape(&attended_memory, memory.shape())
    }
    
    /// Refine query for next reasoning step
    fn refine_query(&self, current_query: &ArrayD<f32>, resonance: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        // Apply refinement transformation
        let resonance_view: Array2<f32> = resonance.view().into_dimensionality().expect("resonance is 2D").to_owned();
        let refined = resonance_view.dot(&self.refinement_weights.t());
        
        // Apply tanh activation
        let activated_refined = refined.mapv(|x| x.tanh());
        
        // Combine with current query
        let mut new_query = current_query.clone();
        for (i, &refined_val) in activated_refined.iter().enumerate() {
            if i < new_query.len() {
                new_query[i] = new_query[i] + self.reasoning_alpha * refined_val;
            }
        }
        
        Ok(new_query)
    }
    
    /// Calculate confidence score
    fn calculate_confidence(&self, resonance: &ArrayD<f32>) -> DLResult<f32> {
        // Confidence based on resonance strength and stability
        let resonance_energy: f32 = resonance.iter().map(|&x| x * x).sum();
        let resonance_magnitude = resonance_energy.sqrt();
        
        // Normalize by size
        let normalized_magnitude = resonance_magnitude / resonance.len() as f32;
        
        // Apply sigmoid for confidence
        Ok(1.0 / (1.0 + (-normalized_magnitude * 10.0).exp()))
    }
    
    /// Calculate novelty score
    fn calculate_novelty(&self, query: &ArrayD<f32>, resonance: &ArrayD<f32>) -> DLResult<f32> {
        // Novelty based on difference between query and resonance
        let similarity = self.cosine_similarity(&self.to_1d_array(query)?, &self.to_1d_array(resonance)?);
        
        // Novelty = 1 - similarity
        Ok(1.0 - similarity)
    }
    
    /// Calculate coherence score
    fn calculate_coherence(&self, query: &ArrayD<f32>, resonance: &ArrayD<f32>) -> DLResult<f32> {
        // Coherence based on phase alignment and energy consistency
        let query_energy: f32 = query.iter().map(|&x| x * x).sum();
        let resonance_energy: f32 = resonance.iter().map(|&x| x * x).sum();
        
        if query_energy == 0.0 || resonance_energy == 0.0 {
            return Ok(0.0);
        }
        
        let energy_ratio = (query_energy / resonance_energy).min(resonance_energy / query_energy);
        Ok(energy_ratio.sqrt())
    }
    
    /// Check for convergence
    fn check_convergence(&self) -> bool {
        if self.convergence_scores.len() < 3 {
            return false;
        }
        
        let recent_scores: Vec<f32> = self.convergence_scores.iter()
            .rev()
            .take(3)
            .cloned()
            .collect();
        
        // Check if scores are stable
        let score_variance = self.calculate_variance(&recent_scores);
        score_variance < self.convergence_threshold
    }
    
    /// Calculate variance of scores
    fn calculate_variance(&self, scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        
        let mean: f32 = scores.iter().sum();
        let mean = mean / scores.len() as f32;
        
        let variance: f32 = scores.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        variance / scores.len() as f32
    }
    
    /// Generate final reasoning output
    fn generate_final_output(&self) -> DLResult<ArrayD<f32>> {
        if self.reasoning_history.is_empty() {
            return Err(DeepLearningError::Configuration {
                reason: "No reasoning steps performed".to_string(),
            });
        }
        
        // Weighted combination of reasoning steps
        let mut final_output = ArrayD::zeros(self.reasoning_history[0].query.shape().to_vec());
        
        for (step_idx, reasoning_step) in self.reasoning_history.iter().enumerate() {
            let weight = if step_idx < self.step_importance_weights.len() {
                self.step_importance_weights[step_idx]
            } else {
                0.01 // Very small weight for additional steps
            };
            
            // Weighted sum of query and resonance
            for (i, &query_val) in reasoning_step.query.iter().enumerate() {
                if i < final_output.len() {
                    final_output[i] += weight * query_val;
                }
            }
            
            for (i, &resonance_val) in reasoning_step.resonance.iter().enumerate() {
                if i < final_output.len() {
                    final_output[i] += weight * resonance_val;
                }
            }
        }
        
        // Apply final transformation
        let output_view: Array2<f32> = final_output.view().into_dimensionality().expect("output is 2D").to_owned();
        let transformed = output_view.dot(&self.output_weights.t());
        let activated = transformed.mapv(|x| x.tanh());
        
        Ok(activated.into_dimensionality().expect("activated is 2D"))
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self) -> DLResult<()> {
        if !self.convergence_scores.is_empty() {
            self.average_confidence = self.convergence_scores.iter().sum::<f32>() / self.convergence_scores.len() as f32;
        }
        
        self.reasoning_depth = self.reasoning_history.len() as f32;
        
        // Calculate query stability
        if self.query_evolution.len() >= 2 {
            let mut stabilities = Vec::with_capacity(self.query_evolution.len().saturating_sub(1));
            
            for i in 1..self.query_evolution.len() {
                let similarity = self.cosine_similarity(
                    &self.to_1d_array(&self.query_evolution[i-1])?,
                    &self.to_1d_array(&self.query_evolution[i])?
                );
                stabilities.push(similarity);
            }
            
            self.query_stability = stabilities.iter().sum::<f32>() / stabilities.len() as f32;
        }
        
        Ok(())
    }
    
    /// Utility: Convert to 1D array
    fn to_1d_array(&self, tensor: &ArrayD<f32>) -> DLResult<Array1<f32>> {
        let vec: Vec<f32> = tensor.iter().cloned().collect();
        Ok(Array1::from(vec))
    }
    
    /// Utility: Convert back to original shape
    fn to_original_shape(&self, array_1d: &Array1<f32>, original_shape: &[usize]) -> DLResult<ArrayD<f32>> {
        let total_size = original_shape.iter().product();
        let vec: Vec<f32> = array_1d.iter().take(total_size).cloned().collect();
        Ok(ArrayD::from_shape_vec(original_shape.to_vec(), vec).expect("data length matches shape"))
    }
    
    /// Utility: Cosine similarity
    fn cosine_similarity(&self, arr1: &Array1<f32>, arr2: &Array1<f32>) -> f32 {
        let dot_product: f32 = arr1.iter().zip(arr2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = arr1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = arr2.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let scale = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            rand::random::<f32>() * 2.0 * scale - scale
        })
    }
    
    /// Reset reasoning state
    fn reset_reasoning_state(&mut self) -> DLResult<()> {
        self.reasoning_history.clear();
        self.query_evolution.clear();
        self.convergence_scores.clear();
        self.novelty_scores.clear();
        self.coherence_scores.clear();
        self.current_step = 0;
        self.is_converged = false;
        self.final_reasoning = None;
        
        Ok(())
    }
    
    /// Get reasoning history
    pub fn get_query_weights(&self) -> Tensor {
        let data = ArrayD::from_shape_vec(
            vec![self.query_weights.shape()[0], self.query_weights.shape()[1]],
            self.query_weights.iter().copied().collect(),
        ).expect("data length matches shape");
        Tensor::new(data)
    }
    pub fn get_refinement_weights(&self) -> Tensor {
        let data = ArrayD::from_shape_vec(
            vec![self.refinement_weights.shape()[0], self.refinement_weights.shape()[1]],
            self.refinement_weights.iter().copied().collect(),
        ).expect("data length matches shape");
        Tensor::new(data)
    }
    pub fn get_output_weights(&self) -> Tensor {
        let data = ArrayD::from_shape_vec(
            vec![self.output_weights.shape()[0], self.output_weights.shape()[1]],
            self.output_weights.iter().copied().collect(),
        ).expect("data length matches shape");
        Tensor::new(data)
    }
    pub fn set_query_weights(&mut self, t: &Tensor) {
        let d = t.data();
        self.query_weights = d.clone().into_shape(self.query_weights.dim()).unwrap_or(self.query_weights.clone());
    }
    pub fn set_refinement_weights(&mut self, t: &Tensor) {
        let d = t.data();
        self.refinement_weights = d.clone().into_shape(self.refinement_weights.dim()).unwrap_or(self.refinement_weights.clone());
    }
    pub fn set_output_weights(&mut self, t: &Tensor) {
        let d = t.data();
        self.output_weights = d.clone().into_shape(self.output_weights.dim()).unwrap_or(self.output_weights.clone());
    }
    pub fn get_reasoning_history(&self) -> &[ReasoningStep] {
        &self.reasoning_history
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> ReasoningMetrics {
        ReasoningMetrics {
            reasoning_steps: self.reasoning_history.len(),
            is_converged: self.is_converged,
            average_confidence: self.average_confidence,
            reasoning_depth: self.reasoning_depth,
            query_stability: self.query_stability,
            final_confidence: self.convergence_scores.last().copied().unwrap_or(0.0),
        }
    }
    
    /// Get final reasoning result
    pub fn get_final_reasoning(&self) -> Option<&ArrayD<f32>> {
        self.final_reasoning.as_ref()
    }
    
    /// Set reasoning parameters
    pub fn set_reasoning_alpha(&mut self, alpha: f32) {
        self.reasoning_alpha = alpha;
    }
    
    /// Set convergence threshold
    pub fn set_convergence_threshold(&mut self, threshold: f32) {
        self.convergence_threshold = threshold;
    }
    
    /// Enable/disable adaptive steps
    pub fn set_adaptive_steps(&mut self, adaptive: bool) {
        self.adaptive_steps = adaptive;
    }
    
    /// Enable/disable early termination
    pub fn set_early_termination(&mut self, early_termination: bool) {
        self.early_termination = early_termination;
    }
}

/// Reasoning performance metrics
#[derive(Debug, Clone)]
pub struct ReasoningMetrics {
    pub reasoning_steps: usize,
    pub is_converged: bool,
    pub average_confidence: f32,
    pub reasoning_depth: f32,
    pub query_stability: f32,
    pub final_confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    
    #[test]
    fn test_irr_creation() {
        let irr = IterativeResonanceReasoner::new(512, 3, 0.1, 0.01).unwrap();
        assert_eq!(irr.reasoning_steps, 3);
        assert_eq!(irr.reasoning_alpha, 0.1);
        assert_eq!(irr.convergence_threshold, 0.01);
    }
    
    #[test]
    fn test_reasoning_step_creation() {
        let query = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let resonance = ArrayD::from_shape_vec(vec![3], vec![0.5, 1.0, 1.5]).unwrap();
        let step = ReasoningStep::new(0, query.clone(), resonance.clone());
        
        assert_eq!(step.step_number, 0);
        assert_eq!(step.query, query);
        assert_eq!(step.resonance, resonance);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let irr = IterativeResonanceReasoner::new(3, 3, 0.1, 0.01).unwrap();
        
        let vec1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let vec2 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let vec3 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        
        let sim1 = irr.cosine_similarity(&vec1, &vec2);
        let sim2 = irr.cosine_similarity(&vec1, &vec3);
        
        assert!((sim1 - 1.0).abs() < 1e-6);
        assert!((sim2 - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_variance_calculation() {
        let irr = IterativeResonanceReasoner::new(3, 3, 0.1, 0.01).unwrap();
        
        let scores1 = vec![1.0, 1.0, 1.0]; // No variance
        let scores2 = vec![0.0, 1.0, 2.0]; // Some variance
        
        let var1 = irr.calculate_variance(&scores1);
        let var2 = irr.calculate_variance(&scores2);
        
        assert!((var1 - 0.0).abs() < 1e-6);
        assert!(var2 > var1);
    }
}
