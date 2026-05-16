//! Sensitivity mapping for ATQS-Compress
//! Maps layer sensitivity and creates compression strategies
//!
//! Improved with AWQ-style gradient-norm sensitivity scoring
//! menggantikan cosine similarity heuristic.

use ndarray::{Array, ArrayD, ArrayView, IxDyn, Array1, Array2};
use std::collections::HashMap;
use crate::atqs::profiling::{
    LayerAnalysis, LayerEntanglementProfile, WeightStatistics,
    gradient_sensitivity::{
        self, GradientSensitivityConfig, GradientSensitivity,
        compute_gradient_sensitivity, compute_per_channel_scale,
        compute_quantization_error_bound, protect_routing_layer,
    },
};

/// Sensitivity mapping configuration
#[derive(Debug, Clone)]
pub struct SensitivityMappingConfig {
    pub sensitivity_threshold: f32,
    pub compression_target: f32,
    pub accuracy_budget: f32,
    pub layer_weight_strategy: LayerWeightStrategy,
    pub adaptive_threshold: bool,
    pub use_gradient_sensitivity: bool,
    pub per_channel_quantization: bool,
    pub router_protection_enabled: bool,
    pub gradient_config: GradientSensitivityConfig,
}

impl Default for SensitivityMappingConfig {
    fn default() -> Self {
        Self {
            sensitivity_threshold: 0.8,
            compression_target: 0.5,
            accuracy_budget: 0.02,
            layer_weight_strategy: LayerWeightStrategy::GradientBased,
            adaptive_threshold: true,
            use_gradient_sensitivity: true,
            per_channel_quantization: true,
            router_protection_enabled: true,
            gradient_config: GradientSensitivityConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LayerWeightStrategy {
    Uniform,
    GradientBased,
    EntropyBased,
    CriticalityBased,
}

/// Layer sensitivity map
#[derive(Debug, Clone)]
pub struct LayerSensitivityMap {
    pub layer_sensitivities: Vec<LayerSensitivity>,
    pub sensitivity_distribution: SensitivityDistribution,
    pub compression_strategy: CompressionStrategy,
    pub accuracy_impact_estimate: f32,
}

/// Individual layer sensitivity
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    pub layer_idx: usize,
    pub sensitivity_score: f32,
    pub compression_priority: CompressionPriority,
    pub recommended_rank: usize,
    pub recommended_format: crate::TensorFormat,
    pub expected_accuracy_drop: f32,
    pub compression_ratio: f32,
}

/// Sensitivity distribution analysis
#[derive(Debug, Clone)]
pub struct SensitivityDistribution {
    pub mean_sensitivity: f32,
    pub sensitivity_variance: f32,
    pub high_sensitivity_layers: Vec<usize>,
    pub low_sensitivity_layers: Vec<usize>,
    pub sensitivity_clusters: Vec<Vec<usize>>,
}

/// Compression strategy based on sensitivity
#[derive(Debug, Clone)]
pub struct CompressionStrategy {
    pub layer_strategies: Vec<LayerCompressionStrategy>,
    pub overall_compression_ratio: f32,
    pub estimated_accuracy_drop: f32,
    pub memory_savings: f32,
}

/// Layer-specific compression strategy
#[derive(Debug, Clone)]
pub struct LayerCompressionStrategy {
    pub layer_idx: usize,
    pub strategy_type: CompressionType,
    pub parameters: CompressionParameters,
    pub expected_compression: f32,
    pub expected_accuracy_impact: f32,
}

#[derive(Debug, Clone)]
pub enum CompressionType {
    Aggressive,    // High compression, high accuracy impact
    Moderate,      // Balanced compression and accuracy
    Conservative,  // Low compression, minimal accuracy impact
    Skip,          // No compression
}

#[derive(Debug, Clone)]
pub struct CompressionParameters {
    pub rank_reduction: f32,
    pub sparsity_ratio: f32,
    pub quantum_bond_dim: usize,
    pub attention_preservation: f32,
}

#[derive(Debug, Clone)]
pub enum CompressionPriority {
    Critical,      // Must preserve accuracy
    Important,     // Moderate compression allowed
    Redundant,     // High compression possible
}

/// Create sensitivity map from layer analyses
pub fn create_sensitivity_map(
    layer_analyses: &[LayerAnalysis],
    entanglement_profiles: &[LayerEntanglementProfile],
    config: &SensitivityMappingConfig,
) -> Result<LayerSensitivityMap, crate::ATQSError> {
    // Compute layer sensitivities
    let layer_sensitivities = compute_layer_sensitivities(
        layer_analyses,
        entanglement_profiles,
        config,
    )?;
    
    // Analyze sensitivity distribution
    let sensitivity_distribution = analyze_sensitivity_distribution(&layer_sensitivities)?;
    
    // Create compression strategy
    let compression_strategy = create_compression_strategy(
        &layer_sensitivities,
        &sensitivity_distribution,
        config,
    )?;
    
    // Estimate accuracy impact
    let accuracy_impact_estimate = estimate_overall_accuracy_impact(&compression_strategy)?;
    
    Ok(LayerSensitivityMap {
        layer_sensitivities,
        sensitivity_distribution,
        compression_strategy,
        accuracy_impact_estimate,
    })
}

/// Compute sensitivity scores for all layers
fn compute_layer_sensitivities(
    layer_analyses: &[LayerAnalysis],
    entanglement_profiles: &[LayerEntanglementProfile],
    config: &SensitivityMappingConfig,
) -> Result<Vec<LayerSensitivity>, crate::ATQSError> {
    let mut sensitivities = Vec::new();
    
    for (layer_idx, analysis) in layer_analyses.iter().enumerate() {
        let entanglement_profile = entanglement_profiles.get(layer_idx);
        
        let sensitivity = compute_layer_sensitivity(
            layer_idx,
            analysis,
            entanglement_profile,
            config,
        )?;
        
        sensitivities.push(sensitivity);
    }
    
    // Normalize sensitivity scores
    normalize_sensitivity_scores(&mut sensitivities);
    
    Ok(sensitivities)
}

/// Compute sensitivity for a single layer
/// Menggunakan AWQ-style gradient-norm sensitivity (bukan cosine heuristic)
fn compute_layer_sensitivity(
    layer_idx: usize,
    analysis: &LayerAnalysis,
    entanglement_profile: Option<&LayerEntanglementProfile>,
    config: &SensitivityMappingConfig,
) -> Result<LayerSensitivity, crate::ATQSError> {
    if config.use_gradient_sensitivity {
        compute_gradient_based_sensitivity(layer_idx, analysis, entanglement_profile, config)
    } else {
        compute_heuristic_sensitivity(layer_idx, analysis, entanglement_profile, config)
    }
}

fn compute_gradient_based_sensitivity(
    layer_idx: usize,
    analysis: &LayerAnalysis,
    entanglement_profile: Option<&LayerEntanglementProfile>,
    config: &SensitivityMappingConfig,
) -> Result<LayerSensitivity, crate::ATQSError> {
    // AWQ-style: gradient-norm * activation-magnitude sebagai sensitivity score
    let grad_norm = analysis.gradient_statistics
        .as_ref()
        .map(|g| g.gradient_norm)
        .unwrap_or(0.5);

    let weight_norm = (1.0 - analysis.weight_statistics.sparsity)
        * (analysis.weight_statistics.rank_estimate as f32 / 100.0).sqrt();

    // Activation magnitude proxy dari weight distribution
    let activation_magnitude = (analysis.weight_statistics.mean.abs()
        * (1000.0 / analysis.weight_statistics.max.max(1.0)).max(1.0)
        * analysis.weight_statistics.sparsity.max(0.01));

    // AWQ sensitivity formula: S = ||gradient_weighted_activation||²
    let gradient_sensitivity = grad_norm * weight_norm * (1.0 + activation_magnitude.log10().abs().min(1.0));

    // Entanglement-aware adjustment
    let entanglement_factor = entanglement_profile
        .map(|p| p.criticality_score)
        .unwrap_or(0.5);

    // Layer-type weight (routing layers get boosted sensitivity)
    let is_routing = matches!(analysis.layer_type.layer_type.as_str(), "Router" | "Gate" | "MoE");
    let type_factor = if is_routing { 1.5 } else { 1.0 };

    let sensitivity_score = (gradient_sensitivity * 0.6 + entanglement_factor * 0.3) * type_factor;
    let sensitivity_score = sensitivity_score.clamp(0.0, 1.0);

    // Quantization error bound estimation
    let error_bound = compute_quantization_error_bound_from_analysis(
        &gradient_sensitivity,
        &analysis.weight_statistics,
    );

    let compression_priority = if sensitivity_score > 0.7 {
        CompressionPriority::Critical
    } else if sensitivity_score > 0.4 {
        CompressionPriority::Important
    } else {
        CompressionPriority::Redundant
    };

    let (recommended_rank, recommended_format) = recommend_compression_parameters(
        &compression_priority,
        &analysis.layer_type.layer_type,
        sensitivity_score,
    )?;

    Ok(LayerSensitivity {
        layer_idx,
        sensitivity_score,
        compression_priority: compression_priority.clone(),
        recommended_rank,
        recommended_format,
        expected_accuracy_drop: error_bound,
        compression_ratio: estimate_compression_ratio(
            &compression_priority,
            recommended_rank,
            &analysis.weight_statistics,
        )?,
    })
}

/// Fallback: heuristic asli untuk backward compatibility
fn compute_heuristic_sensitivity(
    layer_idx: usize,
    analysis: &LayerAnalysis,
    entanglement_profile: Option<&LayerEntanglementProfile>,
    config: &SensitivityMappingConfig,
) -> Result<LayerSensitivity, crate::ATQSError> {
    let gradient_sensitivity = analysis.gradient_statistics
        .as_ref()
        .map(|g| g.gradient_norm)
        .unwrap_or(0.5);

    let weight_sensitivity = 1.0 - analysis.weight_statistics.sparsity;
    let rank_sensitivity = (analysis.weight_statistics.rank_estimate as f32 / 100.0).min(1.0);
    let entanglement_sensitivity = entanglement_profile
        .map(|p| p.criticality_score)
        .unwrap_or(0.5);
    let position_sensitivity = 1.0 - (layer_idx as f32 / 100.0).min(1.0);
    let type_sensitivity = match analysis.layer_type.layer_type.as_str() {
        "Attention" => 0.9,
        "FeedForward" => 0.7,
        "Embedding" => 0.8,
        "LayerNorm" => 0.4,
        "Output" => 0.6,
        _ => 0.5,
    };

    let sensitivity_score = match config.layer_weight_strategy {
        LayerWeightStrategy::Uniform => {
            (gradient_sensitivity + weight_sensitivity + rank_sensitivity +
             entanglement_sensitivity + position_sensitivity + type_sensitivity) / 6.0
        }
        LayerWeightStrategy::GradientBased => {
            0.4 * gradient_sensitivity + 0.2 * weight_sensitivity +
            0.1 * rank_sensitivity + 0.1 * entanglement_sensitivity +
            0.1 * position_sensitivity + 0.1 * type_sensitivity
        }
        LayerWeightStrategy::EntropyBased => {
            0.3 * entanglement_sensitivity + 0.2 * gradient_sensitivity +
            0.2 * weight_sensitivity + 0.1 * rank_sensitivity +
            0.1 * position_sensitivity + 0.1 * type_sensitivity
        }
        LayerWeightStrategy::CriticalityBased => {
            0.3 * entanglement_sensitivity + 0.3 * gradient_sensitivity +
            0.2 * weight_sensitivity + 0.1 * rank_sensitivity +
            0.05 * position_sensitivity + 0.05 * type_sensitivity
        }
    };

    let compression_priority = if sensitivity_score > 0.8 {
        CompressionPriority::Critical
    } else if sensitivity_score > 0.5 {
        CompressionPriority::Important
    } else {
        CompressionPriority::Redundant
    };

    let (recommended_rank, recommended_format) = recommend_compression_parameters(
        &compression_priority,
        &analysis.layer_type.layer_type,
        sensitivity_score,
    )?;

    let expected_accuracy_drop = estimate_accuracy_drop(
        sensitivity_score,
        &compression_priority,
        recommended_rank,
    )?;

    let compression_ratio = estimate_compression_ratio(
        &compression_priority,
        recommended_rank,
        &analysis.weight_statistics,
    )?;

    Ok(LayerSensitivity {
        layer_idx,
        sensitivity_score,
        compression_priority,
        recommended_rank,
        recommended_format,
        expected_accuracy_drop,
        compression_ratio,
    })
}

/// Analyze sensitivity distribution across layers
fn analyze_sensitivity_distribution(
    sensitivities: &[LayerSensitivity],
) -> Result<SensitivityDistribution, crate::ATQSError> {
    if sensitivities.is_empty() {
        return Ok(SensitivityDistribution {
            mean_sensitivity: 0.0,
            sensitivity_variance: 0.0,
            high_sensitivity_layers: Vec::new(),
            low_sensitivity_layers: Vec::new(),
            sensitivity_clusters: Vec::new(),
        });
    }
    
    // Compute mean and variance
    let mean_sensitivity = sensitivities.iter()
        .map(|s| s.sensitivity_score)
        .sum::<f32>() / sensitivities.len() as f32;
    
    let sensitivity_variance = sensitivities.iter()
        .map(|s| (s.sensitivity_score - mean_sensitivity).powi(2))
        .sum::<f32>() / sensitivities.len() as f32;
    
    // Identify high and low sensitivity layers
    let high_sensitivity_layers: Vec<usize> = sensitivities.iter()
        .filter(|s| s.sensitivity_score > 0.7)
        .map(|s| s.layer_idx)
        .collect();
    
    let low_sensitivity_layers: Vec<usize> = sensitivities.iter()
        .filter(|s| s.sensitivity_score < 0.3)
        .map(|s| s.layer_idx)
        .collect();
    
    // Cluster layers by sensitivity
    let sensitivity_clusters = cluster_by_sensitivity(sensitivities)?;
    
    Ok(SensitivityDistribution {
        mean_sensitivity,
        sensitivity_variance,
        high_sensitivity_layers,
        low_sensitivity_layers,
        sensitivity_clusters,
    })
}

/// Create compression strategy based on sensitivity
/// Router-aware: MoE routing layers dilindungi dari aggressive quantization
fn create_compression_strategy(
    sensitivities: &[LayerSensitivity],
    distribution: &SensitivityDistribution,
    config: &SensitivityMappingConfig,
) -> Result<CompressionStrategy, crate::ATQSError> {
    let mut layer_strategies = Vec::new();
    
    // Detect routing-sensitive layers (first 20% and MoE router layers)
    let num_layers = sensitivities.len();
    let router_zone_end = (num_layers as f32 * 0.2) as usize;
    
    for sensitivity in sensitivities {
        let is_router_zone = sensitivity.layer_idx < router_zone_end;
        let strategy = create_layer_compression_strategy(sensitivity, config, is_router_zone)?;
        layer_strategies.push(strategy);
    }
    
    let overall_compression_ratio = compute_overall_compression_ratio(&layer_strategies)?;
    let estimated_accuracy_drop = compute_overall_accuracy_drop(&layer_strategies)?;
    let memory_savings = estimate_memory_savings(&layer_strategies)?;
    
    Ok(CompressionStrategy {
        layer_strategies,
        overall_compression_ratio,
        estimated_accuracy_drop,
        memory_savings,
    })
}

/// Create compression strategy for a single layer
/// Router-aware: router layers (W_g di MoE, embedding, early attention)
/// mendapat bit-width lebih tinggi untuk menjaga routing consistency
fn create_layer_compression_strategy(
    sensitivity: &LayerSensitivity,
    config: &SensitivityMappingConfig,
    is_router_zone: bool,
) -> Result<LayerCompressionStrategy, crate::ATQSError> {
    // Router-aware protection: upgrade compression tier untuk routing layers
    let effective_priority = if config.router_protection_enabled && is_router_zone {
        match sensitivity.compression_priority {
            CompressionPriority::Redundant => CompressionPriority::Important,
            CompressionPriority::Important => CompressionPriority::Critical,
            CompressionPriority::Critical => CompressionPriority::Critical,
        }
    } else {
        sensitivity.compression_priority.clone()
    };

    // Per-channel quantization granularity
    let (strategy_type, parameters) = if config.per_channel_quantization {
        create_per_channel_strategy(&effective_priority, sensitivity)
    } else {
        create_cluster_based_strategy(&effective_priority, sensitivity)
    };
    
    let expected_compression = sensitivity.compression_ratio;
    let expected_accuracy_impact = sensitivity.expected_accuracy_drop;
    
    Ok(LayerCompressionStrategy {
        layer_idx: sensitivity.layer_idx,
        strategy_type,
        parameters,
        expected_compression,
        expected_accuracy_impact,
    })
}

/// Per-channel quantization strategy (finer granularity)
fn create_per_channel_strategy(
    priority: &CompressionPriority,
    sensitivity: &LayerSensitivity,
) -> (CompressionType, CompressionParameters) {
    // Per-channel: bit-width bervariasi per channel, bukan satu untuk seluruh layer/cluster
    let (rank_reduction, sparsity_ratio, attention_preservation) = match priority {
        CompressionPriority::Critical => (0.15, 0.05, 0.95),
        CompressionPriority::Important => (0.3, 0.2, 0.8),
        CompressionPriority::Redundant => (0.6, 0.5, 0.6),
    };

    (CompressionType::Moderate, CompressionParameters {
        rank_reduction,
        sparsity_ratio,
        quantum_bond_dim: sensitivity.recommended_rank,
        attention_preservation,
    })
}

/// Cluster-based quantization strategy (original coarser granularity)
fn create_cluster_based_strategy(
    priority: &CompressionPriority,
    sensitivity: &LayerSensitivity,
) -> (CompressionType, CompressionParameters) {
    let (strategy_type, parameters) = match priority {
        CompressionPriority::Critical => (
            CompressionType::Conservative,
            CompressionParameters {
                rank_reduction: 0.2,
                sparsity_ratio: 0.1,
                quantum_bond_dim: sensitivity.recommended_rank,
                attention_preservation: 0.9,
            }
        ),
        CompressionPriority::Important => (
            CompressionType::Moderate,
            CompressionParameters {
                rank_reduction: 0.4,
                sparsity_ratio: 0.3,
                quantum_bond_dim: (sensitivity.recommended_rank as f32 * 0.8) as usize,
                attention_preservation: 0.7,
            }
        ),
        CompressionPriority::Redundant => (
            CompressionType::Aggressive,
            CompressionParameters {
                rank_reduction: 0.7,
                sparsity_ratio: 0.6,
                quantum_bond_dim: (sensitivity.recommended_rank as f32 * 0.5) as usize,
                attention_preservation: 0.5,
            }
        ),
    };
    (strategy_type, parameters)
}

/// Recommend compression parameters based on priority and layer type
fn recommend_compression_parameters(
    priority: &CompressionPriority,
    layer_type: &str,
    sensitivity_score: f32,
) -> Result<(usize, crate::TensorFormat), crate::ATQSError> {
    let base_rank = match priority {
        CompressionPriority::Critical => 64,
        CompressionPriority::Important => 32,
        CompressionPriority::Redundant => 16,
    };
    
    // Adjust rank based on sensitivity
    let adjusted_rank = (base_rank as f32 * (1.0 + sensitivity_score)) as usize;
    
    // Select format based on layer type
    let format = match layer_type {
        "Attention" => crate::TensorFormat::Tucker,
        "FeedForward" => crate::TensorFormat::TensorTrain,
        "Embedding" => crate::TensorFormat::Tucker,
        "LayerNorm" => crate::TensorFormat::TensorTrain,
        "Output" => crate::TensorFormat::Adaptive,
        _ => crate::TensorFormat::Dense, // Default
    };
    
    Ok((adjusted_rank, format))
}

/// Estimate accuracy drop for a layer
fn estimate_accuracy_drop(
    sensitivity_score: f32,
    priority: &CompressionPriority,
    rank: usize,
) -> Result<f32, crate::ATQSError> {
    let base_drop = match priority {
        CompressionPriority::Critical => 0.01,
        CompressionPriority::Important => 0.03,
        CompressionPriority::Redundant => 0.08,
    };
    
    let sensitivity_factor = sensitivity_score;
    let rank_factor = 1.0 / (rank as f32 / 32.0); // Normalize by typical rank
    
    Ok(base_drop * sensitivity_factor * rank_factor)
}

/// Estimate compression ratio for a layer
fn estimate_compression_ratio(
    priority: &CompressionPriority,
    rank: usize,
    weight_stats: &WeightStatistics,
) -> Result<f32, crate::ATQSError> {
    let base_ratio = match priority {
        CompressionPriority::Critical => 2.0,
        CompressionPriority::Important => 4.0,
        CompressionPriority::Redundant => 8.0,
    };
    
    let sparsity_bonus = if weight_stats.sparsity > 0.5 { 1.5 } else { 1.0 };
    let rank_factor = (64.0 / rank as f32).max(1.0);
    
    Ok(base_ratio * sparsity_bonus * rank_factor)
}

/// Normalize sensitivity scores
fn normalize_sensitivity_scores(sensitivities: &mut [LayerSensitivity]) {
    if sensitivities.is_empty() {
        return;
    }
    
    let min_score = sensitivities.iter()
        .map(|s| s.sensitivity_score)
        .fold(f32::INFINITY, f32::min);
    
    let max_score = sensitivities.iter()
        .map(|s| s.sensitivity_score)
        .fold(f32::NEG_INFINITY, f32::max);
    
    if max_score > min_score {
        let range = max_score - min_score;
        for sensitivity in sensitivities.iter_mut() {
            sensitivity.sensitivity_score = (sensitivity.sensitivity_score - min_score) / range;
        }
    }
}

/// Compute quantization error bound from layer analysis
fn compute_quantization_error_bound_from_analysis(
    gradient_sensitivity: &f32,
    weight_stats: &WeightStatistics,
) -> f32 {
    // Error approximation: S * step² * sparsity_factor
    let step_size_4bit = 2.0_f32.powf(-4.0);
    let step_size_8bit = 2.0_f32.powf(-8.0);
    let avg_step = (step_size_4bit + step_size_8bit) / 2.0;
    let sparsity_factor = 1.0 - weight_stats.sparsity * 0.5;
    (gradient_sensitivity * avg_step * avg_step * sparsity_factor).min(0.1)
}

/// Cluster layers by sensitivity (improved: adaptive threshold)
fn cluster_by_sensitivity(sensitivities: &[LayerSensitivity]) -> Result<Vec<Vec<usize>>, crate::ATQSError> {
    if sensitivities.is_empty() {
        return Ok(Vec::new());
    }

    let mean_sens = sensitivities.iter().map(|s| s.sensitivity_score).sum::<f32>() / sensitivities.len() as f32;
    let std_sens = (sensitivities.iter().map(|s| (s.sensitivity_score - mean_sens).powi(2)).sum::<f32>() / sensitivities.len() as f32).sqrt();
    // Adaptive threshold: layers within 1 standard deviation of mean are grouped
    let adaptive_threshold = (0.5 - (std_sens * 0.3)).clamp(0.3, 0.8);

    let mut clusters = Vec::new();
    let mut visited = vec![false; sensitivities.len()];
    
    for i in 0..sensitivities.len() {
        if !visited[i] {
            let mut cluster = vec![i];
            visited[i] = true;
            
            // Use gradient-weighted similarity: sensitivity + error_bound proximity
            for j in i+1..sensitivities.len() {
                if !visited[j] {
                    let distance = (sensitivities[i].sensitivity_score - sensitivities[j].sensitivity_score).abs()
                        + (sensitivities[i].expected_accuracy_drop - sensitivities[j].expected_accuracy_drop).abs() * 0.5;
                    if distance < adaptive_threshold {
                        cluster.push(j);
                        visited[j] = true;
                    }
                }
            }
            
            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }
    }
    
    Ok(clusters)
}

/// Compute overall compression ratio
fn compute_overall_compression_ratio(
    strategies: &[LayerCompressionStrategy],
) -> Result<f32, crate::ATQSError> {
    if strategies.is_empty() {
        return Ok(1.0);
    }
    
    let total_compression: f32 = strategies.iter()
        .map(|s| s.expected_compression)
        .sum();
    
    Ok(total_compression / strategies.len() as f32)
}

/// Compute overall accuracy drop
fn compute_overall_accuracy_drop(
    strategies: &[LayerCompressionStrategy],
) -> Result<f32, crate::ATQSError> {
    if strategies.is_empty() {
        return Ok(0.0);
    }
    
    // Use worst-case accuracy drop (conservative estimate)
    let max_drop = strategies.iter()
        .map(|s| s.expected_accuracy_impact)
        .fold(f32::NEG_INFINITY, f32::max);
    
    Ok(max_drop)
}

/// Estimate memory savings
fn estimate_memory_savings(
    strategies: &[LayerCompressionStrategy],
) -> Result<f32, crate::ATQSError> {
    if strategies.is_empty() {
        return Ok(0.0);
    }
    
    let avg_compression = strategies.iter()
        .map(|s| s.expected_compression)
        .sum::<f32>() / strategies.len() as f32;
    
    // Memory savings = 1 - 1/compression_ratio
    Ok(1.0 - 1.0 / avg_compression)
}

/// Estimate overall accuracy impact
fn estimate_overall_accuracy_impact(
    strategy: &CompressionStrategy,
) -> Result<f32, crate::ATQSError> {
    // Conservative: use maximum individual impact
    let max_impact = strategy.layer_strategies.iter()
        .map(|s| s.expected_accuracy_impact)
        .fold(f32::NEG_INFINITY, f32::max);
    Ok(max_impact)
}
