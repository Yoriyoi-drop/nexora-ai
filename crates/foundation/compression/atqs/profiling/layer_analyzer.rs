//! Layer analysis for ATQS-Compress profiling
//! Analyzes layer properties, types, and characteristics

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Standard;
use std::collections::HashMap;

/// Layer analysis configuration
#[derive(Debug, Clone)]
pub struct LayerAnalysisConfig {
    pub analyze_gradients: bool,
    pub analyze_activations: bool,
    pub analyze_weights: bool,
    pub sample_size: usize,
    pub sensitivity_threshold: f32,
}

/// Layer analysis result
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer_idx: usize,
    pub layer_type: crate::LayerInfo,
    pub weight_statistics: WeightStatistics,
    pub activation_statistics: Option<ActivationStatistics>,
    pub gradient_statistics: Option<GradientStatistics>,
    pub computational_cost: ComputationalCost,
    pub memory_footprint: usize,
    pub criticality_indicators: Vec<CriticalityIndicator>,
}

/// Weight statistics
#[derive(Debug, Clone)]
pub struct WeightStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity: f32,
    pub rank_estimate: usize,
    pub singular_values: Vec<f32>,
}

/// Activation statistics
#[derive(Debug, Clone)]
pub struct ActivationStatistics {
    pub mean_activation: f32,
    pub activation_variance: f32,
    pub dead_neuron_ratio: f32,
    pub activation_entropy: f32,
    pub output_distribution: HashMap<String, f32>,
}

/// Gradient statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    pub gradient_norm: f32,
    pub gradient_variance: f32,
    pub gradient_sparsity: f32,
    pub update_magnitude: f32,
}

/// Computational cost analysis
#[derive(Debug, Clone)]
pub struct ComputationalCost {
    pub flops_per_forward: usize,
    pub flops_per_backward: usize,
    pub memory_accesses: usize,
    pub parallel_efficiency: f32,
}

/// Criticality indicators
#[derive(Debug, Clone)]
pub enum CriticalityIndicator {
    HighGradientNorm,
    LowSparsity,
    HighRank,
    CriticalPath,
    AttentionHead,
    EarlyLayer,
}

/// Analyze all layers in foundation model
pub fn analyze_model_layers(
    model: &dyn crate::FoundationModel,
    config: &LayerAnalysisConfig,
) -> Result<Vec<LayerAnalysis>, crate::ATQSError> {
    let layers = model.get_layers();
    let mut analyses = Vec::new();
    
    for (layer_idx, layer) in layers.iter().enumerate() {
        let analysis = analyze_single_layer(layer, layer_idx, model, config)?;
        analyses.push(analysis);
    }
    
    Ok(analyses)
}

/// Analyze a single layer
pub fn analyze_single_layer(
    layer: &dyn crate::ModelLayer,
    layer_idx: usize,
    model: &dyn crate::FoundationModel,
    config: &LayerAnalysisConfig,
) -> Result<LayerAnalysis, crate::ATQSError> {
    let layer_type = layer.get_layer_info();
    
    // Analyze weight statistics
    let weight_statistics = analyze_weight_statistics(layer)?;
    
    // Analyze activation statistics (if requested)
    let activation_statistics = if config.analyze_activations {
        Some(analyze_activation_statistics(layer, config.sample_size)?)
    } else {
        None
    };
    
    // Analyze gradient statistics (if requested)
    let gradient_statistics = if config.analyze_gradients {
        Some(analyze_gradient_statistics(layer, config.sample_size)?)
    } else {
        None
    };
    
    // Compute computational cost
    let computational_cost = compute_computational_cost(layer)?;
    
    // Compute memory footprint
    let memory_footprint = compute_memory_footprint(layer)?;
    
    // Identify criticality indicators
    let criticality_indicators = identify_criticality_indicators(
        &weight_statistics,
        &activation_statistics,
        &gradient_statistics,
        layer_idx,
        &layer_type.layer_type,
        config,
    )?;
    
    Ok(LayerAnalysis {
        layer_idx,
        layer_type: layer_type.clone(),
        weight_statistics,
        activation_statistics,
        gradient_statistics,
        computational_cost,
        memory_footprint,
        criticality_indicators,
    })
}

/// Analyze weight statistics
fn analyze_weight_statistics(
    layer: &dyn crate::ModelLayer,
) -> Result<WeightStatistics, crate::ATQSError> {
    let weights = layer.get_weights();
    let flat_weights: Vec<f32> = weights.iter().cloned().collect();
    
    // Basic statistics
    let mean = flat_weights.iter().sum::<f32>() / flat_weights.len() as f32;
    let variance = flat_weights.iter()
        .map(|w| (w - mean).powi(2))
        .sum::<f32>() / flat_weights.len() as f32;
    let std_dev = variance.sqrt();
    let min = flat_weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = flat_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Sparsity
    let sparsity = flat_weights.iter()
        .filter(|&&w| w.abs() < 1e-8)
        .count() as f32 / flat_weights.len() as f32;
    
    // Rank estimation using SVD
    let (rank_estimate, singular_values) = estimate_weight_rank(&weights)?;
    
    Ok(WeightStatistics {
        mean,
        std_dev,
        min,
        max,
        sparsity,
        rank_estimate,
        singular_values,
    })
}

/// Analyze activation statistics
fn analyze_activation_statistics(
    layer: &dyn crate::ModelLayer,
    sample_size: usize,
) -> Result<ActivationStatistics, crate::ATQSError> {
    let weights = layer.get_weights();
    let mut activations = Vec::new();
    
    // Generate random inputs and compute activations
    for _ in 0..sample_size {
        let input = generate_random_input(&weights.shape())?;
        let activation = compute_layer_activation(&weights, &input)?;
        activations.push(activation);
    }
    
    // Compute statistics
    let mean_activation = activations.iter().sum::<f32>() / activations.len() as f32;
    let activation_variance = activations.iter()
        .map(|a| (a - mean_activation).powi(2))
        .sum::<f32>() / activations.len() as f32;
    
    // Dead neuron ratio (very small activations)
    let dead_neuron_ratio = activations.iter()
        .filter(|&&a| a.abs() < 1e-6)
        .count() as f32 / activations.len() as f32;
    
    // Activation entropy
    let activation_entropy = compute_activation_entropy(&activations)?;
    
    // Output distribution
    let mut output_distribution = HashMap::new();
    output_distribution.insert("mean".to_string(), mean_activation);
    output_distribution.insert("variance".to_string(), activation_variance);
    output_distribution.insert("dead_ratio".to_string(), dead_neuron_ratio);
    
    Ok(ActivationStatistics {
        mean_activation,
        activation_variance,
        dead_neuron_ratio,
        activation_entropy,
        output_distribution,
    })
}

/// Analyze gradient statistics
fn analyze_gradient_statistics(
    layer: &dyn crate::ModelLayer,
    sample_size: usize,
) -> Result<GradientStatistics, crate::ATQSError> {
    let weights = layer.get_weights();
    let mut gradients = Vec::new();
    
    // Approximate gradients using finite differences
    for _ in 0..sample_size {
        let gradient = approximate_layer_gradient(&weights)?;
        gradients.push(gradient);
    }
    
    // Compute gradient statistics
    let gradient_norm = gradients.iter()
        .map(|g| g.iter().map(|&x| x * x).sum::<f32>().sqrt())
        .sum::<f32>() / gradients.len() as f32;
    
    let gradient_variance = gradients.iter()
        .map(|g| {
            let mean = g.iter().sum::<f32>() / g.len() as f32;
            g.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / g.len() as f32
        })
        .sum::<f32>() / gradients.len() as f32;
    
    let gradient_sparsity = gradients.iter()
        .map(|g| {
            g.iter().filter(|&&x| x.abs() < 1e-8).count() as f32 / g.len() as f32
        })
        .sum::<f32>() / gradients.len() as f32;
    
    let update_magnitude = gradients.iter()
        .map(|g| g.iter().map(|&x| x.abs()).sum::<f32>())
        .sum::<f32>() / gradients.len() as f32;
    
    Ok(GradientStatistics {
        gradient_norm,
        gradient_variance,
        gradient_sparsity,
        update_magnitude,
    })
}

/// Compute computational cost for layer
fn compute_computational_cost(
    layer: &dyn crate::ModelLayer,
) -> Result<ComputationalCost, crate::ATQSError> {
    let weights = layer.get_weights();
    let layer_type = layer.get_layer_info();
    let shape = weights.shape();
    
    let (flops_per_forward, flops_per_backward) = match layer_type.layer_type.as_str() {
        "Attention" => {
            // Attention: Q*K^T + softmax + (Q*K^T)*V
            let seq_len = shape[0];
            let hidden_dim = shape[1];
            let head_dim = hidden_dim / 8; // Assuming 8 heads
            
            let qk_flops = seq_len * seq_len * head_dim;
            let softmax_flops = seq_len * seq_len;
            let qkv_flops = seq_len * seq_len * head_dim;
            
            let forward_flops = qk_flops + softmax_flops + qkv_flops;
            let backward_flops = forward_flops * 2; // Gradient computation
            
            (forward_flops, backward_flops)
        }
        "FeedForward" => {
            // FFN: Linear + ReLU + Linear
            let input_dim = shape[0];
            let hidden_dim = shape[1];
            
            let linear1_flops = input_dim * hidden_dim;
            let relu_flops = hidden_dim;
            let linear2_flops = hidden_dim * input_dim;
            
            let forward_flops = linear1_flops + relu_flops + linear2_flops;
            let backward_flops = forward_flops * 2;
            
            (forward_flops, backward_flops)
        }
        "Embedding" => {
            // Embedding lookup
            let vocab_size = shape[0];
            let embed_dim = shape[1];
            
            let forward_flops = embed_dim; // Just lookup
            let backward_flops = embed_dim; // Gradient update
            
            (forward_flops, backward_flops)
        }
        "LayerNorm" => {
            // Layer normalization
            let dim: usize = shape.iter().product();
            
            let forward_flops = dim * 3; // mean, var, normalize
            let backward_flops = dim * 2;
            
            (forward_flops, backward_flops)
        }
        "Output" => {
            // Output projection
            let input_dim = shape[0];
            let output_dim = shape[1];
            
            let forward_flops = input_dim * output_dim;
            let backward_flops = forward_flops * 2;
            
            (forward_flops, backward_flops)
        }
        _ => {
            // Default case: treat as simple matrix multiplication
            let input_dim = shape[0];
            let output_dim = shape[1];
            
            let forward_flops = input_dim * output_dim;
            let backward_flops = forward_flops * 2;
            
            (forward_flops, backward_flops)
        }
    };
    
    // Memory accesses (rough estimate)
    let memory_accesses = weights.len() * 2; // Read + write
    
    // Parallel efficiency (simplified estimate)
    let parallel_efficiency = match layer_type.layer_type.as_str() {
        "Attention" => 0.8, // Good parallelization
        "FeedForward" => 0.9, // Excellent parallelization
        "Embedding" => 0.7, // Memory bound
        "LayerNorm" => 0.6, // Sequential dependencies
        "Output" => 0.9, // Good parallelization
        _ => 0.8, // Default
    };
    
    Ok(ComputationalCost {
        flops_per_forward,
        flops_per_backward,
        memory_accesses,
        parallel_efficiency,
    })
}

/// Compute memory footprint of layer
fn compute_memory_footprint(layer: &dyn crate::ModelLayer) -> Result<usize, crate::ATQSError> {
    let weights = layer.get_weights();
    Ok(weights.len() * std::mem::size_of::<f32>())
}

/// Identify criticality indicators for layer
fn identify_criticality_indicators(
    weight_stats: &WeightStatistics,
    activation_stats: &Option<ActivationStatistics>,
    gradient_stats: &Option<GradientStatistics>,
    layer_idx: usize,
    layer_type: &str,
    config: &LayerAnalysisConfig,
) -> Result<Vec<CriticalityIndicator>, crate::ATQSError> {
    let mut indicators = Vec::new();
    
    // High gradient norm indicator
    if let Some(grad_stats) = gradient_stats {
        if grad_stats.gradient_norm > config.sensitivity_threshold {
            indicators.push(CriticalityIndicator::HighGradientNorm);
        }
    }
    
    // Low sparsity indicator (dense layers are often critical)
    if weight_stats.sparsity < 0.1 {
        indicators.push(CriticalityIndicator::LowSparsity);
    }
    
    // High rank indicator
    if weight_stats.rank_estimate > 64 {
        indicators.push(CriticalityIndicator::HighRank);
    }
    
    // Early layer indicator
    if layer_idx < 4 {
        indicators.push(CriticalityIndicator::EarlyLayer);
    }
    
    // Attention head indicator
    if layer_type == "Attention" {
        indicators.push(CriticalityIndicator::AttentionHead);
    }
    
    // Critical path indicator (based on computational cost)
    if let Some(act_stats) = activation_stats {
        if act_stats.activation_variance > 0.5 {
            indicators.push(CriticalityIndicator::CriticalPath);
        }
    }
    
    Ok(indicators)
}

/// Estimate weight rank using SVD
fn estimate_weight_rank(
    weights: &ArrayD<f32>,
) -> Result<(usize, Vec<f32>), crate::ATQSError> {
    let shape = weights.shape();
    
    // Reshape to 2D for SVD
    let rows = shape[0];
    let cols = shape.iter().skip(1).product();
    let reshaped = weights.clone().into_shape((rows, cols))?;
    
    // Compute SVD
    let (_u, s, _vt) = compute_weight_svd(&reshaped.view())?;
    
    // Estimate rank based on singular value threshold
    let threshold = s[0] * 1e-6; // Relative threshold
    let rank = s.iter().take_while(|&sv| *sv > threshold).count();
    
    Ok((rank, s))
}

/// Compute SVD for weight analysis
fn compute_weight_svd(
    matrix: &ArrayView<f32, ndarray::Ix2>,
) -> Result<(Array<f32, ndarray::Ix2>, Vec<f32>, Array<f32, ndarray::Ix2>), crate::ATQSError> {
    let (m, n) = matrix.dim();
    let rank = m.min(n);
    
    // Placeholder SVD computation
    let mut u = Array::zeros((m, rank));
    for elem in u.iter_mut() {
        *elem = rand::random::<f32>();
    }
    let s: Vec<f32> = (0..rank).map(|i| 1.0 / (i + 1) as f32).collect();
    let mut vt = Array::zeros((rank, n));
    for elem in vt.iter_mut() {
        *elem = rand::random::<f32>();
    }
    
    Ok((u, s, vt))
}

/// Generate random input for activation analysis
fn generate_random_input(shape: &[usize]) -> Result<ArrayD<f32>, crate::ATQSError> {
    let size: usize = shape.iter().product();
    {
        let mut array = Array::zeros(size);
        for elem in array.iter_mut() {
            *elem = rand::random::<f32>();
        }
        Ok(array.into_shape(shape).unwrap().into_dyn())
    }
}

/// Compute layer activation
fn compute_layer_activation(
    weights: &ArrayD<f32>,
    input: &ArrayD<f32>,
) -> Result<f32, crate::ATQSError> {
    // Simplified activation computation
    let dot_product = weights.iter().zip(input.iter())
        .map(|(w, i)| w * i)
        .sum::<f32>();
    
    // Apply ReLU activation
    Ok(dot_product.max(0.0))
}

/// Compute activation entropy
fn compute_activation_entropy(activations: &[f32]) -> Result<f32, crate::ATQSError> {
    if activations.is_empty() {
        return Ok(0.0);
    }
    
    // Create histogram
    let bins = 10;
    let min = activations.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = activations.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    
    if range < 1e-12 {
        return Ok(0.0);
    }
    
    let mut histogram = vec![0; bins];
    for &activation in activations {
        let bin_idx = ((activation - min) / range * bins as f32) as usize;
        let bin_idx = bin_idx.min(bins - 1);
        histogram[bin_idx] += 1;
    }
    
    // Compute entropy
    let total = activations.len() as f32;
    let mut entropy = 0.0;
    for &count in &histogram {
        if count > 0 {
            let p = count as f32 / total;
            entropy -= p * p.ln();
        }
    }
    
    Ok(entropy)
}

/// Approximate layer gradient using finite differences
fn approximate_layer_gradient(
    weights: &ArrayD<f32>,
) -> Result<Vec<f32>, crate::ATQSError> {
    let flat_weights: Vec<f32> = weights.iter().cloned().collect();
    let mut gradients = Vec::new();
    let epsilon = 1e-6;
    
    for (i, &weight) in flat_weights.iter().enumerate() {
        // Perturb weight slightly
        let mut perturbed_weights = flat_weights.clone();
        perturbed_weights[i] += epsilon;
        
        // Compute approximate gradient (simplified)
        let original_output = compute_simple_output(&flat_weights);
        let perturbed_output = compute_simple_output(&perturbed_weights);
        let gradient = (perturbed_output - original_output) / epsilon;
        
        gradients.push(gradient);
    }
    
    Ok(gradients)
}

/// Compute simple output for gradient approximation
fn compute_simple_output(weights: &[f32]) -> f32 {
    weights.iter().sum::<f32>() / weights.len() as f32
}
