use ndarray::{Array1, Array2, ArrayD, s};
use std::collections::HashMap;

/// AWQ-style gradient-norm sensitivity estimator
/// Mengganti cosine similarity heuristic dengan mathematical sensitivity score
#[derive(Debug, Clone)]
pub struct GradientSensitivityConfig {
    pub calibration_samples: usize,
    pub hessian_trace_rank: usize,
    pub use_activation_scaling: bool,
    pub router_protection_bits: usize,
    pub channel_group_size: usize,
}

impl Default for GradientSensitivityConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 128,
            hessian_trace_rank: 32,
            use_activation_scaling: true,
            router_protection_bits: 8,
            channel_group_size: 128,
        }
    }
}

/// Per-channel sensitivity analysis
#[derive(Debug, Clone)]
pub struct ChannelSensitivity {
    pub channel_idx: usize,
    pub activation_magnitude: f32,
    pub weight_norm: f32,
    pub sensitivity_score: f32,
    pub scale_factor: f32,
    pub recommended_bits: usize,
}

/// Per-layer gradient sensitivity
#[derive(Debug, Clone)]
pub struct GradientSensitivity {
    pub layer_idx: usize,
    pub layer_name: String,
    pub is_routing_layer: bool,
    pub overall_sensitivity: f32,
    pub channel_sensitivities: Vec<ChannelSensitivity>,
    pub hessian_trace_approx: f32,
    pub recommended_bit_width: usize,
}

/// Compute gradient-norm sensitivity (AWQ-style)
/// Formula: S(l) = ||gradient_weighted_activation||² ≈ ||activation_magnitude * weight||²
pub fn compute_gradient_sensitivity(
    weights: &Array2<f32>,
    calibration_activations: &[Array1<f32>],
    config: &GradientSensitivityConfig,
) -> GradientSensitivity {
    let (output_channels, _input_dim) = weights.dim();
    let mut channel_sensitivities = Vec::with_capacity(output_channels);

    // Compute per-channel activation magnitude (AWQ key insight)
    let activation_magnitudes = if calibration_activations.is_empty() {
        vec![1.0; output_channels]
    } else {
        compute_channel_activation_magnitudes(calibration_activations, output_channels)
    };

    // Per-channel weight norms
    let channel_weight_norms: Vec<f32> = (0..output_channels)
        .map(|c| weights.row(c).mapv(|x| x * x).sum().sqrt())
        .collect();

    // Compute sensitivity and scale per channel
    let max_activation = activation_magnitudes
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-8);

    for c in 0..output_channels {
        // AWQ sensitivity: activation_magnitude * weight_norm
        let sensitivity = activation_magnitudes[c] * channel_weight_norms[c];
        let sensitivity_score = (sensitivity / (1e-8 + max_activation)).min(1.0);

        // AWQ scaling: s = max(|x|)^α / mean(|x|) for channel
        let scale_factor = if config.use_activation_scaling {
            compute_awq_scale_factor(&activation_magnitudes, c)
        } else {
            1.0
        };

        let recommended_bits = if sensitivity_score > 0.7 {
            8
        } else if sensitivity_score > 0.4 {
            6
        } else {
            4
        };

        channel_sensitivities.push(ChannelSensitivity {
            channel_idx: c,
            activation_magnitude: activation_magnitudes[c],
            weight_norm: channel_weight_norms[c],
            sensitivity_score,
            scale_factor,
            recommended_bits,
        });
    }

    let overall_sensitivity = channel_sensitivities
        .iter()
        .map(|c| c.sensitivity_score)
        .sum::<f32>()
        / output_channels.max(1) as f32;

    let avg_bits = channel_sensitivities
        .iter()
        .map(|c| c.recommended_bits as f32)
        .sum::<f32>()
        / output_channels.max(1) as f32;

    GradientSensitivity {
        layer_idx: 0,
        layer_name: String::new(),
        is_routing_layer: false,
        overall_sensitivity,
        channel_sensitivities,
        hessian_trace_approx: 0.0,
        recommended_bit_width: avg_bits.round() as usize,
    }
}

/// Compute Hessian trace approximation (GPTQ-style)
/// H = diag(X·Xᵀ) where X is activation matrix
pub fn compute_hessian_trace_approximation(
    weights: &Array2<f32>,
    calibration_activations: &[Array1<f32>],
    config: &GradientSensitivityConfig,
) -> Array1<f32> {
    let (_output_channels, input_dim) = weights.dim();

    if calibration_activations.is_empty() {
        return Array1::ones(input_dim);
    }

    let n = calibration_activations.len().min(config.calibration_samples);
    let mut hessian_diag: Array1<f32> = Array1::zeros(input_dim);

    for act in calibration_activations.iter().take(n) {
        let act_clipped = if act.len() > input_dim {
            act.slice(s![..input_dim]).to_owned()
        } else {
            act.clone()
        };
        // H_ii = Σ x_i² for each input dimension
        hessian_diag = hessian_diag + &act_clipped.mapv(|x| x * x);
    }

    let result: Array1<f32> = hessian_diag.mapv(|x: f32| x / n.max(1) as f32);
    result
}

/// Apply AWQ-style per-channel scaling before quantization
/// s_c = max(|X_c|)^α / mean(|X_c|) ; scaled_W = W_c / s_c ; quantize(scaled_W)
pub fn compute_per_channel_scale(
    weights: &Array2<f32>,
    activations: &[Array1<f32>],
    alpha: f32,
) -> Vec<f32> {
    let (output_channels, _input_dim) = weights.dim();

    if activations.is_empty() {
        return vec![1.0; output_channels];
    }

    let mut scales = Vec::with_capacity(output_channels);

    for c in 0..output_channels {
        let mut channel_acts: Vec<f32> = activations
            .iter()
            .filter_map(|a| a.get(c).copied())
            .collect();

        if channel_acts.is_empty() {
            scales.push(1.0);
            continue;
        }

        let max_abs = channel_acts
            .iter()
            .map(|x| x.abs())
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1e-8);
        let mean_abs = channel_acts.iter().map(|x| x.abs()).sum::<f32>()
            / channel_acts.len() as f32;

        // s = max(|x|)^α / mean(|x|)
        let scale = max_abs.powf(alpha) / mean_abs.max(1e-8);
        scales.push(scale.max(0.1).min(10.0));
    }

    scales
}

/// Apply per-channel quantization with scaling
pub fn apply_per_channel_quantization(
    weights: &Array2<f32>,
    per_channel_scales: &[f32],
    bits: usize,
) -> Array2<f32> {
    let (output_channels, input_dim) = weights.dim();
    let mut quantized = weights.clone();

    let max_level = (1 << (bits - 1)) as f32;

    for c in 0..output_channels.min(per_channel_scales.len()) {
        let scale = per_channel_scales[c];
        if scale.abs() < 1e-8 {
            continue;
        }

        for j in 0..input_dim {
            // Scale weight down, quantize, scale back up
            let scaled = weights[(c, j)] / scale;
            let clamped = scaled.clamp(-max_level, max_level);
            let rounded = clamped.round();
            quantized[(c, j)] = rounded * scale;
        }
    }

    quantized
}

/// Protect routing layers (W_g in MoE) from aggressive quantization
pub fn protect_routing_layer(
    weights: &Array2<f32>,
    min_bits: usize,
) -> Vec<f32> {
    let (output_channels, _input_dim) = weights.dim();

    // Routing layers get uniform high-precision treatment
    let mut scales = Vec::with_capacity(output_channels);

    for c in 0..output_channels {
        let row = weights.row(c);
        let max_abs = row.iter().map(|x| x.abs()).fold(f32::NEG_INFINITY, f32::max).max(1e-8);
        let scale = max_abs / ((1 << (min_bits - 1)) as f32);
        scales.push(scale);
    }

    scales
}

/// Compute quantization error bound for a layer
pub fn compute_quantization_error_bound(
    sensitivity: &GradientSensitivity,
    bits: usize,
) -> f32 {
    let avg_sensitivity = sensitivity.overall_sensitivity;
    let step_size = 2.0_f32.powf(-(bits as f32));
    // Error ∝ sensitivity * step_size² (second-order)
    avg_sensitivity * step_size * step_size
}

fn compute_channel_activation_magnitudes(
    activations: &[Array1<f32>],
    num_channels: usize,
) -> Vec<f32> {
    let mut magnitudes = vec![0.0f32; num_channels];
    let mut counts = vec![0usize; num_channels];

    for act in activations {
        let len = act.len().min(num_channels);
        for i in 0..len {
            magnitudes[i] += act[i].abs();
            counts[i] += 1;
        }
    }

    for (mag, &cnt) in magnitudes.iter_mut().zip(counts.iter()) {
        if cnt > 0 {
            *mag /= cnt as f32;
        }
    }

    magnitudes
}

fn compute_awq_scale_factor(activation_magnitudes: &[f32], channel: usize) -> f32 {
    let max_val = activation_magnitudes
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-8);
    let mean_val = activation_magnitudes.iter().sum::<f32>()
        / activation_magnitudes.len().max(1) as f32;

    if mean_val < 1e-8 {
        return 1.0;
    }

    // s = max(|x|)^α / mean(|x|) with α=0.5 (AWQ default)
    let alpha = 0.5;
    let s = max_val.powf(alpha) / mean_val;
    s.clamp(0.1, 10.0)
}
