//! Attention-aware analysis for ATQS-Compress
//! Implements attention-guided tensor decomposition and refinement

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use std::collections::HashMap;

/// Attention patterns and metrics
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub attention_weights: ArrayD<f32>,
    pub head_importance: Vec<f32>,
    pub layer_importance: f32,
    pub entropy: f32,
    pub sparsity: f32,
}

/// Attention-aware decomposition configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub attention_threshold: f32,
    pub head_selection_ratio: f32,
    pub layer_weight_factor: f32,
    pub entropy_weight: f32,
    pub sparsity_weight: f32,
}

/// Global attention map across layers
#[derive(Debug, Clone)]
pub struct GlobalAttentionMap {
    pub layer_patterns: Vec<AttentionPattern>,
    pub cross_layer_correlations: HashMap<(usize, usize), f32>,
    pub critical_paths: Vec<Vec<usize>>,
    pub redundancy_scores: Vec<f32>,
}

/// Analyze attention patterns in transformer model
pub fn analyze_attention_patterns(
    attention_weights: &[ArrayD<f32>],
    config: &AttentionConfig,
) -> Result<GlobalAttentionMap, crate::ATQSError> {
    let mut layer_patterns = Vec::new();
    let mut cross_layer_correlations = HashMap::new();
    
    // Analyze each layer's attention patterns
    for (layer_idx, weights) in attention_weights.iter().enumerate() {
        let pattern = analyze_layer_attention(weights, config)?;
        layer_patterns.push(pattern);
    }
    
    // Compute cross-layer correlations
    for i in 0..layer_patterns.len() {
        for j in i+1..layer_patterns.len() {
            let correlation = compute_attention_correlation(
                &layer_patterns[i], 
                &layer_patterns[j]
            )?;
            cross_layer_correlations.insert((i, j), correlation);
        }
    }
    
    // Identify critical attention paths
    let critical_paths = identify_critical_paths(&layer_patterns, config)?;
    
    // Compute redundancy scores
    let redundancy_scores = compute_layer_redundancy(&layer_patterns)?;
    
    Ok(GlobalAttentionMap {
        layer_patterns,
        cross_layer_correlations,
        critical_paths,
        redundancy_scores,
    })
}

/// Guide tensor decomposition using attention information
pub fn guide_tensor_decomposition(
    tensor: &ArrayD<f32>,
    attention_pattern: &AttentionPattern,
    base_rank: usize,
) -> Result<Vec<usize>, crate::ATQSError> {
    let shape = tensor.shape();
    let mut adaptive_ranks = Vec::new();
    
    for (dim_idx, &dim_size) in shape.iter().enumerate() {
        // Adjust rank based on attention importance
        let attention_factor = get_attention_importance_for_dim(
            attention_pattern, 
            dim_idx
        )?;
        
        let entropy_factor = attention_pattern.entropy * attention_pattern.entropy;
        let sparsity_factor = 1.0 - attention_pattern.sparsity;
        
        // Compute adaptive rank
        let adaptive_rank = ((base_rank as f32) * 
            attention_factor * 
            (1.0 + entropy_factor) * 
            (1.0 + sparsity_factor)
        ) as usize;
        
        // Ensure rank is within valid bounds
        let final_rank = adaptive_rank.min(dim_size).max(1);
        adaptive_ranks.push(final_rank);
    }
    
    Ok(adaptive_ranks)
}

/// Perform attention-aware joint optimization across layers
pub fn optimize_joint_attention(
    tensors: &[ArrayD<f32>],
    attention_map: &GlobalAttentionMap,
    config: &AttentionConfig,
) -> Result<Vec<ArrayD<f32>>, crate::ATQSError> {
    let mut optimized_tensors = Vec::new();
    
    // Group layers by critical paths
    let mut path_groups = HashMap::new();
    for (path_idx, path) in attention_map.critical_paths.iter().enumerate() {
        for &layer_idx in path {
            path_groups.insert(layer_idx, path_idx);
        }
    }
    
    // Optimize each path jointly
    for path in &attention_map.critical_paths {
        let path_tensors: Vec<&ArrayD<f32>> = path.iter()
            .map(|&idx| &tensors[idx])
            .collect();
        
        let path_patterns: Vec<&AttentionPattern> = path.iter()
            .map(|&idx| &attention_map.layer_patterns[idx])
            .collect();
        
        let optimized_path = optimize_attention_path(
            &path_tensors, 
            &path_patterns, 
            config
        )?;
        
        optimized_tensors.extend(optimized_path);
    }
    
    // Handle non-critical layers individually
    for (idx, tensor) in tensors.iter().enumerate() {
        if !path_groups.contains_key(&idx) {
            let pattern = &attention_map.layer_patterns[idx];
            let optimized = optimize_single_layer_attention(tensor, pattern, config)?;
            optimized_tensors.push(optimized);
        }
    }
    
    Ok(optimized_tensors)
}

/// Analyze attention pattern for a single layer
fn analyze_layer_attention(
    weights: &ArrayD<f32>,
    config: &AttentionConfig,
) -> Result<AttentionPattern, crate::ATQSError> {
    let shape = weights.shape();
    
    // Compute attention entropy
    let entropy = compute_attention_entropy(weights)?;
    
    // Compute attention sparsity
    let sparsity = compute_attention_sparsity(weights, config.attention_threshold)?;
    
    // Compute head importance (assuming multi-head attention)
    let num_heads = shape.get(0).unwrap_or(&1);
    let mut head_importance = Vec::new();
    
    for head_idx in 0..*num_heads {
        let head_weights = if shape.len() > 2 {
            weights.slice_axis(ndarray::Axis(0), ndarray::Slice::from(head_idx..=head_idx))
                .to_owned()
                .into_dyn()
        } else {
            weights.clone()
        };
        
        let importance = compute_head_importance(&head_weights)?;
        head_importance.push(importance);
    }
    
    // Compute overall layer importance
    let layer_importance = head_importance.iter().sum::<f32>() / head_importance.len() as f32;
    
    Ok(AttentionPattern {
        attention_weights: weights.clone(),
        head_importance,
        layer_importance,
        entropy,
        sparsity,
    })
}

/// Compute attention entropy
fn compute_attention_entropy(weights: &ArrayD<f32>) -> Result<f32, crate::ATQSError> {
    // Flatten weights and normalize to probability distribution
    let flat = weights.iter().cloned().collect::<Vec<f32>>();
    let total: f32 = flat.iter().sum();
    
    if total <= 0.0 {
        return Ok(0.0);
    }
    
    let mut entropy = 0.0;
    for &weight in &flat {
        if weight > 0.0 {
            let p = weight / total;
            entropy -= p * p.ln();
        }
    }
    
    Ok(entropy)
}

/// Compute attention sparsity
fn compute_attention_sparsity(
    weights: &ArrayD<f32>, 
    threshold: f32
) -> Result<f32, crate::ATQSError> {
    let total_elements = weights.len();
    let sparse_elements = weights.iter()
        .filter(|&&w| w < threshold)
        .count();
    
    Ok(sparse_elements as f32 / total_elements as f32)
}

/// Compute head importance score
fn compute_head_importance(weights: &ArrayD<f32>) -> Result<f32, crate::ATQSError> {
    // Importance based on attention concentration and magnitude
    let entropy = compute_attention_entropy(weights)?;
    let magnitude = weights.iter().map(|&w| w.abs()).sum::<f32>();
    
    // Lower entropy + higher magnitude = more important
    let importance = magnitude * (1.0 - entropy / (weights.len() as f32).ln());
    Ok(importance)
}

/// Compute correlation between two attention patterns
fn compute_attention_correlation(
    pattern1: &AttentionPattern,
    pattern2: &AttentionPattern,
) -> Result<f32, crate::ATQSError> {
    // Flatten attention weights
    let flat1: Vec<f32> = pattern1.attention_weights.iter().cloned().collect();
    let flat2: Vec<f32> = pattern2.attention_weights.iter().cloned().collect();
    
    if flat1.len() != flat2.len() {
        return Ok(0.0); // Different dimensions, no correlation
    }
    
    // Compute Pearson correlation
    let mean1 = flat1.iter().sum::<f32>() / flat1.len() as f32;
    let mean2 = flat2.iter().sum::<f32>() / flat2.len() as f32;
    
    let mut numerator = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;
    
    for (&w1, &w2) in flat1.iter().zip(flat2.iter()) {
        let diff1 = w1 - mean1;
        let diff2 = w2 - mean2;
        numerator += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }
    
    if var1 == 0.0 || var2 == 0.0 {
        return Ok(0.0);
    }
    
    Ok(numerator / (var1 * var2).sqrt())
}

/// Identify critical attention paths through the network
fn identify_critical_paths(
    patterns: &[AttentionPattern],
    config: &AttentionConfig,
) -> Result<Vec<Vec<usize>>, crate::ATQSError> {
    let mut paths = Vec::new();
    let mut visited = vec![false; patterns.len()];
    
    // Sort layers by importance
    let mut indexed_patterns: Vec<(usize, &AttentionPattern)> = patterns
        .iter()
        .enumerate()
        .collect();
    
    indexed_patterns.sort_by(|a, b| {
        b.1.layer_importance.partial_cmp(&a.1.layer_importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Select top layers based on importance threshold
    let num_selected = ((patterns.len() as f32) * config.head_selection_ratio) as usize;
    
    for &(idx, pattern) in indexed_patterns.iter().take(num_selected) {
        if !visited[idx] && pattern.layer_importance > config.attention_threshold {
            let path = trace_attention_path(idx, patterns, &mut visited, config)?;
            paths.push(path);
        }
    }
    
    Ok(paths)
}

/// Trace attention path starting from a given layer
fn trace_attention_path(
    start_idx: usize,
    patterns: &[AttentionPattern],
    visited: &mut [bool],
    config: &AttentionConfig,
) -> Result<Vec<usize>, crate::ATQSError> {
    let mut path = vec![start_idx];
    visited[start_idx] = true;
    
    let mut current_idx = start_idx;
    
    // Trace forward through connected layers
    while current_idx + 1 < patterns.len() {
        let next_idx = current_idx + 1;
        
        if !visited[next_idx] {
            let correlation = compute_attention_correlation(
                &patterns[current_idx],
                &patterns[next_idx]
            )?;
            
            if correlation > config.attention_threshold {
                path.push(next_idx);
                visited[next_idx] = true;
                current_idx = next_idx;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    
    Ok(path)
}

/// Compute redundancy scores for layers
fn compute_layer_redundancy(patterns: &[AttentionPattern]) -> Result<Vec<f32>, crate::ATQSError> {
    let mut redundancy_scores = Vec::new();
    
    for (idx, pattern) in patterns.iter().enumerate() {
        let mut redundancy = 0.0;
        
        // Compare with other layers
        for (other_idx, other_pattern) in patterns.iter().enumerate() {
            if idx != other_idx {
                let correlation = compute_attention_correlation(pattern, other_pattern)?;
                redundancy += correlation;
            }
        }
        
        // Normalize by number of comparisons
        redundancy /= (patterns.len() - 1) as f32;
        redundancy_scores.push(redundancy);
    }
    
    Ok(redundancy_scores)
}

/// Get attention importance for specific tensor dimension
fn get_attention_importance_for_dim(
    pattern: &AttentionPattern,
    dim_idx: usize,
) -> Result<f32, crate::ATQSError> {
    // Map tensor dimensions to attention importance
    // This is a simplified mapping
    let importance = match dim_idx {
        0 => pattern.layer_importance, // Usually sequence dimension
        1 => pattern.layer_importance * 0.8, // Usually feature dimension
        2 => pattern.layer_importance * 0.6, // Usually head dimension
        _ => pattern.layer_importance * 0.5,
    };
    
    Ok(importance.min(1.0).max(0.1))
}

/// Optimize attention path jointly
fn optimize_attention_path(
    path_tensors: &[&ArrayD<f32>],
    path_patterns: &[&AttentionPattern],
    config: &AttentionConfig,
) -> Result<Vec<ArrayD<f32>>, crate::ATQSError> {
    let mut optimized = Vec::new();
    
    // Compute global attention statistics for the path
    let global_entropy = path_patterns.iter()
        .map(|p| p.entropy)
        .sum::<f32>() / path_patterns.len() as f32;
    
    let global_sparsity = path_patterns.iter()
        .map(|p| p.sparsity)
        .sum::<f32>() / path_patterns.len() as f32;
    
    // Apply joint optimization to each tensor in path
    for (tensor, pattern) in path_tensors.iter().zip(path_patterns.iter()) {
        let optimized_tensor = apply_joint_attention_optimization(
            tensor,
            pattern,
            global_entropy,
            global_sparsity,
            config,
        )?;
        optimized.push(optimized_tensor);
    }
    
    Ok(optimized)
}

/// Optimize single layer with attention guidance
fn optimize_single_layer_attention(
    tensor: &ArrayD<f32>,
    pattern: &AttentionPattern,
    config: &AttentionConfig,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    apply_joint_attention_optimization(
        tensor,
        pattern,
        pattern.entropy,
        pattern.sparsity,
        config,
    )
}

/// Apply joint attention optimization to a tensor
fn apply_joint_attention_optimization(
    tensor: &ArrayD<f32>,
    pattern: &AttentionPattern,
    global_entropy: f32,
    global_sparsity: f32,
    config: &AttentionConfig,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let shape = tensor.shape();
    let mut result = tensor.clone();
    
    // Apply attention-weighted smoothing
    let attention_weight = pattern.layer_importance * config.layer_weight_factor;
    let entropy_factor = 1.0 + (pattern.entropy - global_entropy) * config.entropy_weight;
    let sparsity_factor = 1.0 + (global_sparsity - pattern.sparsity) * config.sparsity_weight;
    
    // Apply scaling based on attention factors
    let scale_factor = attention_weight * entropy_factor * sparsity_factor;
    
    result.mapv_inplace(|x| x * scale_factor);
    
    Ok(result)
}
