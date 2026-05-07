//! Aggregation Layer for Expert Output Fusion

use crate::has_moe_ffn::{
    error::{HasMoeFfnError, Result},
    types::*,
};
use ndarray::{ArrayD, Array2, Array1, ArrayView};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand;
use std::collections::HashMap;

/// Aggregation Layer for fusing expert outputs
pub struct AggregationLayer {
    config: AggregationConfig,
    method: Box<dyn AggregationStrategy>,
    attention_weights: Option<Array2<f32>>,
    gating_network: Option<GatingNetwork>,
    learned_weights: Option<Array1<f32>>,
    normalization_layer: Option<NormalizationLayer>,
}

impl AggregationLayer {
    /// Create new aggregation layer
    pub fn new(config: AggregationConfig) -> Result<Self> {
        let method: Box<dyn AggregationStrategy> = match config.method {
            AggregationMethod::WeightedSum => Box::new(WeightedSumMethod::new()),
            AggregationMethod::Attention => Box::new(AttentionMethod::new()),
            AggregationMethod::Gating => Box::new(GatingMethod::new()),
            AggregationMethod::LearnedMixing => Box::new(LearnedMixingMethod::new()),
        };
        
        let mut aggregation = Self {
            config: config.clone(),
            method,
            attention_weights: None,
            gating_network: None,
            learned_weights: None,
            normalization_layer: None,
        };
        
        // Initialize components based on configuration
        aggregation.initialize_components()?;
        
        Ok(aggregation)
    }
    
    /// Initialize aggregation components
    fn initialize_components(&mut self) -> Result<()> {
        match self.config.method {
            AggregationMethod::Attention => {
                self.attention_weights = Some(Array2::zeros((4096, 8))); // model_dim x num_experts
            }
            AggregationMethod::Gating => {
                self.gating_network = Some(GatingNetwork::new(4096, 8)?);
            }
            AggregationMethod::LearnedMixing => {
                self.learned_weights = Some(Array1::from_shape_fn(8, 
                    |_| fastrand::f32() * 0.2 - 0.1));
            }
            _ => {}
        }
        
        if self.config.normalization {
            self.normalization_layer = Some(NormalizationLayer::new(4096)?);
        }
        
        Ok(())
    }
    
    /// Aggregate expert outputs
    pub fn aggregate(
        &mut self,
        expert_outputs: Vec<ExpertOutput>,
        original_input: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        if expert_outputs.is_empty() {
            return Err(HasMoeFfnError::aggregation("No expert outputs to aggregate"));
        }
        
        // Validate all outputs have same dimensions
        self.validate_output_dimensions(&expert_outputs)?;
        
        // Apply aggregation method
        let aggregated = self.method.aggregate(
            expert_outputs,
            original_input,
            &self.config,
            &mut self.attention_weights,
            &mut self.gating_network,
            &mut self.learned_weights,
        )?;
        
        // Apply normalization if configured
        let final_output = if let Some(normalization) = &self.normalization_layer {
            normalization.normalize(&aggregated)?
        } else {
            aggregated
        };
        
        Ok(final_output)
    }
    
    /// Validate output dimensions
    fn validate_output_dimensions(&self, expert_outputs: &[ExpertOutput]) -> Result<()> {
        if expert_outputs.is_empty() {
            return Ok(());
        }
        
        let first_shape = expert_outputs[0].output.shape();
        
        for output in expert_outputs.iter().skip(1) {
            if output.output.shape() != first_shape {
                return Err(HasMoeFfnError::aggregation(
                    "Expert outputs have inconsistent dimensions"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get aggregation statistics
    pub fn get_aggregation_stats(&self) -> AggregationStats {
        AggregationStats {
            method: format!("{:?}", self.config.method),
            has_attention_weights: self.attention_weights.is_some(),
            has_gating_network: self.gating_network.is_some(),
            has_learned_weights: self.learned_weights.is_some(),
            has_normalization: self.normalization_layer.is_some(),
        }
    }
}

/// Aggregation method trait
pub trait AggregationStrategy {
    fn aggregate(
        &mut self,
        expert_outputs: Vec<ExpertOutput>,
        original_input: &ArrayD<f32>,
        config: &AggregationConfig,
        attention_weights: &mut Option<Array2<f32>>,
        gating_network: &mut Option<GatingNetwork>,
        learned_weights: &mut Option<Array1<f32>>,
    ) -> Result<ArrayD<f32>>;
}

/// Weighted sum aggregation method
pub struct WeightedSumMethod;

impl WeightedSumMethod {
    pub fn new() -> Self {
        Self
    }
}

impl AggregationStrategy for WeightedSumMethod {
    fn aggregate(
        &mut self,
        expert_outputs: Vec<ExpertOutput>,
        _original_input: &ArrayD<f32>,
        _config: &AggregationConfig,
        _attention_weights: &mut Option<Array2<f32>>,
        _gating_network: &mut Option<GatingNetwork>,
        _learned_weights: &mut Option<Array1<f32>>,
    ) -> Result<ArrayD<f32>> {
        if expert_outputs.is_empty() {
            return Err(HasMoeFfnError::aggregation("No expert outputs for weighted sum"));
        }
        
        let output_dim = expert_outputs[0].output.shape()[0];
        let mut weighted_sum = Array1::zeros(output_dim);
        let mut total_weight = 0.0;
        
        for output in expert_outputs {
            let weight = output.confidence;
            let output_view = output.output.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| HasMoeFfnError::tensor("Cannot convert output to 1D"))?;
            
            for (i, &val) in output_view.iter().enumerate() {
                weighted_sum[i] += val * weight;
            }
            total_weight += weight;
        }
        
        // Normalize by total weight
        if total_weight > 0.0 {
            weighted_sum.mapv_inplace(|x| x / total_weight);
        }
        
        Ok(weighted_sum.into_dimensionality::<ndarray::Ix1>().map_err(|_| {
            HasMoeFfnError::tensor("Cannot convert weighted sum to dynamic")
        })?.into_dyn())
    }
}

/// Attention-based aggregation method
pub struct AttentionMethod {
    query_projection: Option<Array2<f32>>,
    key_projection: Option<Array2<f32>>,
    value_projection: Option<Array2<f32>>,
}

impl AttentionMethod {
    pub fn new() -> Self {
        Self {
            query_projection: None,
            key_projection: None,
            value_projection: None,
        }
    }
}

impl AggregationStrategy for AttentionMethod {
    fn aggregate(
        &mut self,
        expert_outputs: Vec<ExpertOutput>,
        original_input: &ArrayD<f32>,
        _config: &AggregationConfig,
        attention_weights: &mut Option<Array2<f32>>,
        _gating_network: &mut Option<GatingNetwork>,
        _learned_weights: &mut Option<Array1<f32>>,
    ) -> Result<ArrayD<f32>> {
        if expert_outputs.is_empty() {
            return Err(HasMoeFfnError::aggregation("No expert outputs for attention aggregation"));
        }
        
        let output_dim = expert_outputs[0].output.shape()[0];
        let num_experts = expert_outputs.len();
        
        // Initialize attention weights if needed
        if attention_weights.is_none() {
            *attention_weights = Some(Array2::from_shape_fn((output_dim, num_experts),
                |_| fastrand::f32() * 0.2 - 0.1));
        }
        
        let attention_weights = attention_weights.as_mut().unwrap();
        
        // Compute attention scores
        let mut attention_scores = Vec::new();
        for (i, output) in expert_outputs.iter().enumerate() {
            // Simple attention score based on confidence and computation cost
            let score = output.confidence / (1.0 + output.computation_cost);
            attention_scores.push(score);
        }
        
        // Normalize attention scores
        let total_score: f32 = attention_scores.iter().sum();
        if total_score > 0.0 {
            for score in &mut attention_scores {
                *score /= total_score;
            }
        }
        
        // Apply attention weights to outputs
        let mut attended_output = Array1::zeros(output_dim);
        
        for (i, output) in expert_outputs.iter().enumerate() {
            let output_view = output.output.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| HasMoeFfnError::tensor("Cannot convert output to 1D"))?;
            
            for (j, &val) in output_view.iter().enumerate() {
                attended_output[j] += val * attention_scores[i];
            }
        }
        
        Ok(attended_output.into_dimensionality::<ndarray::Ix1>().map_err(|_| {
            HasMoeFfnError::tensor("Cannot convert attended output to dynamic")
        })?.into_dyn())
    }
}

/// Gating-based aggregation method
pub struct GatingMethod;

impl GatingMethod {
    pub fn new() -> Self {
        Self
    }
}

impl AggregationStrategy for GatingMethod {
    fn aggregate(
        &mut self,
        expert_outputs: Vec<ExpertOutput>,
        original_input: &ArrayD<f32>,
        _config: &AggregationConfig,
        _attention_weights: &mut Option<Array2<f32>>,
        gating_network: &mut Option<GatingNetwork>,
        _learned_weights: &mut Option<Array1<f32>>,
    ) -> Result<ArrayD<f32>> {
        if expert_outputs.is_empty() {
            return Err(HasMoeFfnError::aggregation("No expert outputs for gating aggregation"));
        }
        
        let output_dim = expert_outputs[0].output.shape()[0];
        let num_experts = expert_outputs.len();
        
        // Initialize gating network if needed
        if gating_network.is_none() {
            *gating_network = Some(GatingNetwork::new(output_dim, num_experts)?);
        }
        
        let gating_network = gating_network.as_mut().unwrap();
        
        // Compute gating weights based on input
        let gating_weights = gating_network.compute_gates(original_input)?;
        
        // Apply gating weights to expert outputs
        let mut gated_output = Array1::zeros(output_dim);
        
        for (i, output) in expert_outputs.iter().enumerate() {
            let gate_weight = if i < gating_weights.len() {
                gating_weights[i]
            } else {
                1.0 / num_experts as f32
            };
            
            let output_view = output.output.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| HasMoeFfnError::tensor("Cannot convert output to 1D"))?;
            
            for (j, &val) in output_view.iter().enumerate() {
                gated_output[j] += val * gate_weight;
            }
        }
        
        Ok(gated_output.into_dimensionality::<ndarray::Ix1>().map_err(|_| {
            HasMoeFfnError::tensor("Cannot convert gated output to dynamic")
        })?.into_dyn())
    }
}

/// Learned mixing aggregation method
pub struct LearnedMixingMethod;

impl LearnedMixingMethod {
    pub fn new() -> Self {
        Self
    }
}

impl AggregationStrategy for LearnedMixingMethod {
    fn aggregate(
        &mut self,
        expert_outputs: Vec<ExpertOutput>,
        _original_input: &ArrayD<f32>,
        _config: &AggregationConfig,
        _attention_weights: &mut Option<Array2<f32>>,
        _gating_network: &mut Option<GatingNetwork>,
        learned_weights: &mut Option<Array1<f32>>,
    ) -> Result<ArrayD<f32>> {
        if expert_outputs.is_empty() {
            return Err(HasMoeFfnError::aggregation("No expert outputs for learned mixing"));
        }
        
        let output_dim = expert_outputs[0].output.shape()[0];
        let num_experts = expert_outputs.len();
        
        // Initialize learned weights if needed
        if learned_weights.is_none() {
            *learned_weights = Some(Array1::from_shape_fn(8, 
                |_| fastrand::f32() * 0.2 - 0.1));
        }
        
        let learned_weights = learned_weights.as_mut().unwrap();
        
        // Apply softmax to learned weights
        let mut softmax_weights = learned_weights.clone();
        let max_weight = softmax_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        softmax_weights.mapv_inplace(|x| (x - max_weight).exp());
        let sum_weights: f32 = softmax_weights.iter().sum();
        if sum_weights > 0.0 {
            softmax_weights.mapv_inplace(|x| x / sum_weights);
        }
        
        // Apply learned weights to expert outputs
        let mut mixed_output = Array1::zeros(output_dim);
        
        for (i, output) in expert_outputs.iter().enumerate() {
            let weight = if i < softmax_weights.len() {
                softmax_weights[i]
            } else {
                1.0 / num_experts as f32
            };
            
            let output_view = output.output.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| HasMoeFfnError::tensor("Cannot convert output to 1D"))?;
            
            for (j, &val) in output_view.iter().enumerate() {
                mixed_output[j] += val * weight;
            }
        }
        
        Ok(mixed_output.into_dimensionality::<ndarray::Ix1>().map_err(|_| {
            HasMoeFfnError::tensor("Cannot convert mixed output to dynamic")
        })?.into_dyn())
    }
}

/// Gating network for dynamic expert selection
pub struct GatingNetwork {
    input_weights: Array2<f32>,
    output_weights: Array2<f32>,
    hidden_dim: usize,
    num_experts: usize,
}

impl GatingNetwork {
    pub fn new(input_dim: usize, num_experts: usize) -> Result<Self> {
        let hidden_dim = input_dim / 2; // Hidden layer size
        
        let input_weights = Array2::from_shape_fn((input_dim, hidden_dim),
            |_| fastrand::f32() * 0.2 - 0.1);
        let output_weights = Array2::from_shape_fn((hidden_dim, num_experts),
            |_| fastrand::f32() * 0.2 - 0.1);
        
        Ok(Self {
            input_weights,
            output_weights,
            hidden_dim,
            num_experts,
        })
    }
    
    pub fn compute_gates(&mut self, input: &ArrayD<f32>) -> Result<Vec<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| HasMoeFfnError::tensor("Cannot convert input to 1D"))?;
        
        // Forward pass through gating network
        let hidden = self.input_weights.t().dot(&input_view);
        let hidden_activated = hidden.mapv(|x| x.max(0.0)); // ReLU activation
        let gates = self.output_weights.t().dot(&hidden_activated);
        
        // Apply softmax
        let max_gate = gates.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let softmax_gates: Vec<f32> = gates.iter()
            .map(|&x| ((x - max_gate).exp()))
            .collect();
        
        let sum_gates: f32 = softmax_gates.iter().sum();
        if sum_gates > 0.0 {
            Ok(softmax_gates.iter().map(|&x| x / sum_gates).collect())
        } else {
            Ok(vec![1.0 / self.num_experts as f32; self.num_experts])
        }
    }
}

/// Normalization layer
pub struct NormalizationLayer {
    normalized_shape: usize,
    epsilon: f32,
}

impl NormalizationLayer {
    pub fn new(normalized_shape: usize) -> Result<Self> {
        Ok(Self {
            normalized_shape,
            epsilon: 1e-6,
        })
    }
    
    pub fn normalize(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| HasMoeFfnError::tensor("Cannot convert input to 1D"))?;
        
        let mean = input_view.mean().unwrap_or(0.0);
        let variance = input_view.var(0.0);
        let std_dev = (variance + self.epsilon).sqrt();
        
        let normalized: Array1<f32> = input_view.mapv(|x| (x - mean) / std_dev);
        
        Ok(normalized.into_dimensionality::<ndarray::Ix1>().map_err(|_| {
            HasMoeFfnError::tensor("Cannot convert normalized output to dynamic")
        })?.into_dyn())
    }
}

/// Aggregation statistics
#[derive(Debug, Clone)]
pub struct AggregationStats {
    pub method: String,
    pub has_attention_weights: bool,
    pub has_gating_network: bool,
    pub has_learned_weights: bool,
    pub has_normalization: bool,
}
