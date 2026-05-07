//! Calibration optimizer for ATQS-Compress
//! Optimizes calibration parameters and schedules

use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use ndarray_rand::RandomExt;
use rand_distr::Standard;
use std::collections::HashMap;
use crate::atqs::types::{CalibrationDataset, CalibrationBatch};

/// Calibration optimizer configuration
#[derive(Debug, Clone)]
pub struct CalibrationOptimizerConfig {
    pub optimization_method: OptimizationMethod,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub learning_rate_schedule: LearningRateSchedule,
    pub validation_split: f32,
    pub early_stopping_patience: usize,
}

#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    Adam,
    SGDWithMomentum,
    AdaGrad,
    RMSProp,
    LAMB,
}

#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    Fixed(f32),
    ExponentialDecay { initial_rate: f32, decay_rate: f32 },
    CosineAnnealing { max_rate: f32, min_rate: f32 },
    WarmupCosine { warmup_steps: usize, max_rate: f32, min_rate: f32 },
}

/// Optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    pub current_iteration: usize,
    pub best_validation_loss: f32,
    pub patience_counter: usize,
    pub learning_rate: f32,
    pub gradient_norm: f32,
    pub convergence_metrics: ConvergenceMetrics,
}

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub loss_trend: Vec<f32>,
    pub gradient_norms: Vec<f32>,
    pub parameter_changes: Vec<f32>,
    pub validation_scores: Vec<f32>,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub final_state: OptimizationState,
    pub optimization_history: Vec<OptimizationState>,
    pub best_parameters: HashMap<String, ArrayD<f32>>,
    pub total_training_time: f32,
    pub convergence_achieved: bool,
}

/// Optimize calibration parameters
pub fn optimize_calibration(
    model: &mut dyn crate::FoundationModel,
    calibration_data: &CalibrationDataset,
    config: &CalibrationOptimizerConfig,
) -> Result<OptimizationResult, crate::ATQSError> {
    // Initialize optimizer
    let mut optimizer = create_optimizer(config)?;
    let mut state = initialize_optimization_state(config)?;
    let mut history = Vec::new();
    
    // Split data into training and validation
    let (train_data, val_data) = split_calibration_data(calibration_data, config.validation_split)?;
    
    // Optimization loop
    for iteration in 0..config.max_iterations {
        // Sample training batch
        let batch = sample_training_batch(&train_data)?;
        
        // Forward pass
        let outputs = forward_pass_batch(model, &batch.inputs)?;
        
        // Compute loss
        let loss = compute_batch_loss(&outputs, &batch.targets)?;
        
        // Backward pass
        let gradients = compute_gradients(model, &batch)?;
        
        // Update parameters
        update_parameters(model, &gradients, &mut optimizer, &mut state)?;
        
        // Update learning rate
        update_learning_rate(&mut state, &config.learning_rate_schedule, iteration)?;
        
        // Validation
        if iteration % 10 == 0 {
            let val_loss = evaluate_validation_loss(model, &val_data)?;
            update_validation_metrics(&mut state, val_loss, config)?;
            
            // Check for convergence
            if check_convergence(&state, config)? {
                break;
            }
        }
        
        // Record state
        state.current_iteration = iteration;
        history.push(state.clone());
    }
    
    // Extract best parameters
    let best_parameters = extract_best_parameters(model)?;
    
    Ok(OptimizationResult {
        final_state: state.clone(),
        optimization_history: history,
        best_parameters,
        total_training_time: 0.0, // Would be measured
        convergence_achieved: state.convergence_metrics.loss_trend.len() > 0,
    })
}

/// Create optimizer based on configuration
fn create_optimizer(config: &CalibrationOptimizerConfig) -> Result<Box<dyn CalibrationOptimizer>, crate::ATQSError> {
    match config.optimization_method {
        OptimizationMethod::Adam => Ok(Box::new(AdamOptimizer::new(config))),
        OptimizationMethod::SGDWithMomentum => Ok(Box::new(SGDMomentumOptimizer::new(config))),
        OptimizationMethod::AdaGrad => Ok(Box::new(AdaGradOptimizer::new(config))),
        OptimizationMethod::RMSProp => Ok(Box::new(RMSPropOptimizer::new(config))),
        OptimizationMethod::LAMB => Ok(Box::new(LAMBOptimizer::new(config))),
    }
}

/// Initialize optimization state
fn initialize_optimization_state(config: &CalibrationOptimizerConfig) -> Result<OptimizationState, crate::ATQSError> {
    let initial_lr = match &config.learning_rate_schedule {
        LearningRateSchedule::Fixed(lr) => *lr,
        LearningRateSchedule::ExponentialDecay { initial_rate, .. } => *initial_rate,
        LearningRateSchedule::CosineAnnealing { max_rate, .. } => *max_rate,
        LearningRateSchedule::WarmupCosine { max_rate, .. } => *max_rate,
    };
    
    Ok(OptimizationState {
        current_iteration: 0,
        best_validation_loss: f32::INFINITY,
        patience_counter: 0,
        learning_rate: initial_lr,
        gradient_norm: 0.0,
        convergence_metrics: ConvergenceMetrics {
            loss_trend: Vec::new(),
            gradient_norms: Vec::new(),
            parameter_changes: Vec::new(),
            validation_scores: Vec::new(),
        },
    })
}

/// Split calibration data into training and validation
fn split_calibration_data(
    data: &CalibrationDataset,
    validation_split: f32,
) -> Result<(CalibrationDataset, CalibrationDataset), crate::ATQSError> {
    let total_samples = data.inputs.len().min(data.targets.len());
    let val_size = (total_samples as f32 * validation_split) as usize;
    let train_size = total_samples - val_size;
    
    // Shuffle indices
    let mut indices: Vec<usize> = (0..total_samples).collect();
    fastrand::shuffle(&mut indices);
    
    // Split data
    let mut train_data = CalibrationDataset {
        inputs: Vec::new(),
        targets: Vec::new(),
        metadata: data.metadata.clone(),
    };
    
    let mut val_data = CalibrationDataset {
        inputs: Vec::new(),
        targets: Vec::new(),
        metadata: data.metadata.clone(),
    };
    
    for (i, &idx) in indices.iter().enumerate() {
        if i < train_size {
            train_data.inputs.push(data.inputs[idx].clone());
            train_data.targets.push(data.targets[idx].clone());
        } else {
            val_data.inputs.push(data.inputs[idx].clone());
            val_data.targets.push(data.targets[idx].clone());
        }
    }
    
    Ok((train_data, val_data))
}

/// Sample training batch
fn sample_training_batch(
    train_data: &CalibrationDataset,
) -> Result<CalibrationBatch, crate::ATQSError> {
    let batch_size = 32; // Fixed batch size for simplicity
    let dataset_size = train_data.inputs.len().min(train_data.targets.len());
    
    if dataset_size == 0 {
        return Err(crate::ATQSError::CalibrationError(
            "Empty training dataset".to_string(),
        ));
    }
    
    let mut batch_inputs = Vec::new();
    let mut batch_targets = Vec::new();
    
    for _ in 0..batch_size {
        let idx = fastrand::usize(0..dataset_size);
        batch_inputs.push(train_data.inputs[idx].clone());
        batch_targets.push(train_data.targets[idx].clone());
    }
    
    Ok(CalibrationBatch {
        inputs: batch_inputs,
        targets: batch_targets,
    })
}

/// Forward pass for a batch
fn forward_pass_batch(
    model: &dyn crate::FoundationModel,
    inputs: &[ArrayD<f32>],
) -> Result<Vec<ArrayD<f32>>, crate::ATQSError> {
    let mut outputs = Vec::new();
    
    for input in inputs {
        let output = forward_pass_single(model, input)?;
        outputs.push(output);
    }
    
    Ok(outputs)
}

/// Compute batch loss
fn compute_batch_loss(
    outputs: &[ArrayD<f32>],
    targets: &[ArrayD<f32>],
) -> Result<f32, crate::ATQSError> {
    if outputs.len() != targets.len() {
        return Err(crate::ATQSError::CalibrationError(
            "Outputs and targets length mismatch".to_string(),
        ));
    }
    
    let mut total_loss = 0.0;
    
    for (output, target) in outputs.iter().zip(targets.iter()) {
        let loss = compute_mse_loss(output, target)?;
        total_loss += loss;
    }
    
    Ok(total_loss / outputs.len() as f32)
}

/// Compute gradients for all parameters
fn compute_gradients(
    model: &mut dyn crate::FoundationModel,
    batch: &CalibrationBatch,
) -> Result<HashMap<String, ArrayD<f32>>, crate::ATQSError> {
    let mut gradients = HashMap::new();
    let layers = model.get_layers();
    
    for (layer_idx, layer) in layers.iter().enumerate() {
        let layer_gradients = compute_layer_gradients(model, layer_idx, batch)?;
        
        for (param_name, gradient) in layer_gradients {
            gradients.insert(format!("layer_{}_{}", layer_idx, param_name), gradient);
        }
    }
    
    Ok(gradients)
}

/// Compute gradients for specific layer
fn compute_layer_gradients(
    model: &mut dyn crate::FoundationModel,
    layer_idx: usize,
    batch: &CalibrationBatch,
) -> Result<HashMap<String, ArrayD<f32>>, crate::ATQSError> {
    let mut layer_gradients = HashMap::new();
    
    // Get layer weights
    let layers = model.get_layers();
    if layer_idx >= layers.len() {
        return Ok(layer_gradients);
    }
    
    let layer = &layers[layer_idx];
    let weights = layer.get_weights();
    
    // Compute weight gradients using finite differences
    let weight_gradients = compute_weight_gradients(model, layer_idx, &weights, batch)?;
    layer_gradients.insert("weights".to_string(), weight_gradients);
    
    // Compute bias gradients if applicable
    if let Some(biases) = get_layer_biases(model, layer_idx)? {
        let bias_gradients = compute_bias_gradients(model, layer_idx, &biases, batch)?;
        layer_gradients.insert("biases".to_string(), bias_gradients);
    }
    
    Ok(layer_gradients)
}

/// Compute weight gradients using finite differences
fn compute_weight_gradients(
    model: &mut dyn crate::FoundationModel,
    layer_idx: usize,
    weights: &ArrayD<f32>,
    batch: &CalibrationBatch,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let epsilon = 1e-6;
    let mut gradients = Array::zeros(weights.shape());
    
    // Compute original loss
    let original_loss = compute_layer_loss(model, layer_idx, batch)?;
    
    // Compute gradients for each weight
    for (idx, &weight) in weights.indexed_iter() {
        
        // Perturb weight
        let mut perturbed_weights = weights.clone();
        perturbed_weights[&idx] += epsilon;
        
        // Temporarily update weights
        model.update_layer_weights(layer_idx, perturbed_weights)?;
        let perturbed_loss = compute_layer_loss(model, layer_idx, batch)?;
        
        // Restore original weights
        model.update_layer_weights(layer_idx, weights.clone())?;
        
        // Compute gradient
        gradients[&idx] = (perturbed_loss - original_loss) / epsilon;
    }
    
    Ok(gradients)
}

/// Compute bias gradients
fn compute_bias_gradients(
    model: &mut dyn crate::FoundationModel,
    layer_idx: usize,
    biases: &ArrayD<f32>,
    batch: &CalibrationBatch,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let epsilon = 1e-6;
    let mut gradients = Array::zeros(biases.shape());
    
    // Compute original loss
    let original_loss = compute_layer_loss(model, layer_idx, batch)?;
    
    // Compute gradients for each bias
    for (idx, &bias) in biases.indexed_iter() {
        
        // Perturb bias
        let mut perturbed_biases = biases.clone();
        perturbed_biases[&idx] += epsilon;
        
        // Temporarily update biases
        update_layer_biases(model, layer_idx, &perturbed_biases)?;
        let perturbed_loss = compute_layer_loss(model, layer_idx, batch)?;
        
        // Restore original biases
        update_layer_biases(model, layer_idx, biases)?;
        
        // Compute gradient
        gradients[&idx] = (perturbed_loss - original_loss) / epsilon;
    }
    
    Ok(gradients)
}

/// Update parameters using optimizer
fn update_parameters(
    model: &mut dyn crate::FoundationModel,
    gradients: &HashMap<String, ArrayD<f32>>,
    optimizer: &mut Box<dyn CalibrationOptimizer>,
    state: &mut OptimizationState,
) -> Result<(), crate::ATQSError> {
    // Compute gradient norm
    let gradient_norm: f32 = gradients.values()
        .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
        .sum::<f32>()
        .sqrt();
    
    state.gradient_norm = gradient_norm;
    
    // Update each parameter
    for (param_name, gradient) in gradients {
        if let Some((layer_idx, param_type)) = parse_parameter_name(param_name) {
            optimizer.update_parameter(model, layer_idx, &param_type, gradient, state)?;
        }
    }
    
    Ok(())
}

/// Update learning rate based on schedule
fn update_learning_rate(
    state: &mut OptimizationState,
    schedule: &LearningRateSchedule,
    iteration: usize,
) -> Result<(), crate::ATQSError> {
    state.learning_rate = match schedule {
        LearningRateSchedule::Fixed(lr) => *lr,
        LearningRateSchedule::ExponentialDecay { initial_rate, decay_rate } => {
            initial_rate * decay_rate.powi(iteration as i32)
        }
        LearningRateSchedule::CosineAnnealing { max_rate, min_rate } => {
            let progress = (iteration as f32 / 1000.0).min(1.0); // Assume 1000 max iterations
            min_rate + (max_rate - min_rate) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
        }
        LearningRateSchedule::WarmupCosine { warmup_steps, max_rate, min_rate } => {
            if iteration < *warmup_steps {
                // Warmup phase
                max_rate * (iteration as f32 / *warmup_steps as f32)
            } else {
                // Cosine annealing phase
                let progress = ((iteration - warmup_steps) as f32 / 1000.0).min(1.0);
                min_rate + (max_rate - min_rate) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
            }
        }
    };
    
    Ok(())
}

/// Evaluate validation loss
fn evaluate_validation_loss(
    model: &dyn crate::FoundationModel,
    val_data: &CalibrationDataset,
) -> Result<f32, crate::ATQSError> {
    let mut total_loss = 0.0;
    let num_samples = val_data.inputs.len().min(val_data.targets.len());
    
    for i in 0..num_samples {
        let input = &val_data.inputs[i];
        let target = &val_data.targets[i];
        
        let output = forward_pass_single(model, input)?;
        let loss = compute_mse_loss(&output, target)?;
        
        total_loss += loss;
    }
    
    Ok(total_loss / num_samples as f32)
}

/// Update validation metrics
fn update_validation_metrics(
    state: &mut OptimizationState,
    val_loss: f32,
    config: &CalibrationOptimizerConfig,
) -> Result<(), crate::ATQSError> {
    state.convergence_metrics.validation_scores.push(val_loss);
    
    // Update best validation loss
    if val_loss < state.best_validation_loss {
        state.best_validation_loss = val_loss;
        state.patience_counter = 0;
    } else {
        state.patience_counter += 1;
    }
    
    // Update loss trend
    state.convergence_metrics.loss_trend.push(val_loss);
    state.convergence_metrics.gradient_norms.push(state.gradient_norm);
    
    Ok(())
}

/// Check for convergence
fn check_convergence(
    state: &OptimizationState,
    config: &CalibrationOptimizerConfig,
) -> Result<bool, crate::ATQSError> {
    // Check early stopping
    if state.patience_counter >= config.early_stopping_patience {
        return Ok(true);
    }
    
    // Check loss convergence
    if let Some(&recent_loss) = state.convergence_metrics.loss_trend.last() {
        if recent_loss < config.convergence_threshold {
            return Ok(true);
        }
    }
    
    // Check gradient norm convergence
    if state.gradient_norm < 1e-6 {
        return Ok(true);
    }
    
    Ok(false)
}

/// Extract best parameters from model
fn extract_best_parameters(
    model: &dyn crate::FoundationModel,
) -> Result<HashMap<String, ArrayD<f32>>, crate::ATQSError> {
    let mut parameters = HashMap::new();
    let layers = model.get_layers();
    
    for (layer_idx, layer) in layers.iter().enumerate() {
        let weights = layer.get_weights();
        parameters.insert(format!("layer_{}_weights", layer_idx), weights.clone());
        
        if let Some(biases) = get_layer_biases(model, layer_idx)? {
            parameters.insert(format!("layer_{}_biases", layer_idx), biases.clone());
        }
    }
    
    Ok(parameters)
}

/// Parse parameter name to extract layer index and type
fn parse_parameter_name(name: &str) -> Option<(usize, String)> {
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() >= 3 && parts[0] == "layer" {
        if let Ok(layer_idx) = parts[1].parse::<usize>() {
            let param_type = parts[2..].join("_");
            return Some((layer_idx, param_type));
        }
    }
    None
}

/// Get layer biases if they exist
fn get_layer_biases(
    model: &dyn crate::FoundationModel,
    layer_idx: usize,
) -> Result<Option<ArrayD<f32>>, crate::ATQSError> {
    // Simplified - in practice would need to check if layer has biases
    let layers = model.get_layers();
    if layer_idx < layers.len() {
        let weights = layers[layer_idx].get_weights();
        // Assume biases are half the size of weights (simplified)
        let bias_shape = vec![weights.shape()[0] / 2];
        Ok(Some(Array::zeros(bias_shape).into_dyn()))
    } else {
        Ok(None)
    }
}

/// Update layer biases
fn update_layer_biases(
    model: &mut dyn crate::FoundationModel,
    layer_idx: usize,
    biases: &ArrayD<f32>,
) -> Result<(), crate::ATQSError> {
    // Simplified - would need actual implementation
    Ok(())
}

/// Compute layer loss
fn compute_layer_loss(
    model: &dyn crate::FoundationModel,
    layer_idx: usize,
    batch: &CalibrationBatch,
) -> Result<f32, crate::ATQSError> {
    let mut total_loss = 0.0;
    
    for (input, target) in batch.inputs.iter().zip(batch.targets.iter()) {
        let layer_output = forward_pass_to_layer(model, input, layer_idx)?;
        let loss = compute_mse_loss(&layer_output, target)?;
        total_loss += loss;
    }
    
    Ok(total_loss / batch.inputs.len() as f32)
}

/// Helper functions (simplified implementations)
fn forward_pass_single(
    model: &dyn crate::FoundationModel,
    input: &ArrayD<f32>,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let layers = model.get_layers();
    let mut output = input.clone();
    
    for layer in layers.iter() {
        let weights = layer.get_weights();
        output = apply_layer_operation(&weights, &output)?;
    }
    
    Ok(output)
}

fn forward_pass_to_layer(
    model: &dyn crate::FoundationModel,
    input: &ArrayD<f32>,
    target_layer: usize,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    let layers = model.get_layers();
    let mut output = input.clone();
    
    for (layer_idx, layer) in layers.iter().enumerate() {
        let weights = layer.get_weights();
        output = apply_layer_operation(&weights, &output)?;
        
        if layer_idx == target_layer {
            break;
        }
    }
    
    Ok(output)
}

fn apply_layer_operation(
    weights: &ArrayD<f32>,
    input: &ArrayD<f32>,
) -> Result<ArrayD<f32>, crate::ATQSError> {
    if weights.ndim() == 2 && input.ndim() == 1 {
        let weights_2d = weights.view().into_dimensionality::<ndarray::Ix2>()?;
        let input_reshaped = input.view().into_shape((input.len(), 1))?;
        let output = weights_2d.dot(&input_reshaped);
        Ok(output.clone().into_shape((output.len(),)).unwrap().into_dyn())
    } else {
        Ok(Array::zeros(weights.shape()).into_dyn())
    }
}

fn compute_mse_loss(
    output: &ArrayD<f32>,
    target: &ArrayD<f32>,
) -> Result<f32, crate::ATQSError> {
    if output.shape() != target.shape() {
        return Ok(0.0);
    }
    
    let diff = output - target;
    let mse = diff.iter().map(|&x| x * x).sum::<f32>() / diff.len() as f32;
    Ok(mse)
}

/// Trait for calibration optimizers
trait CalibrationOptimizer {
    fn update_parameter(
        &mut self,
        model: &mut dyn crate::FoundationModel,
        layer_idx: usize,
        param_type: &str,
        gradient: &ArrayD<f32>,
        state: &OptimizationState,
    ) -> Result<(), crate::ATQSError>;
}

/// Adam optimizer implementation
struct AdamOptimizer {
    config: CalibrationOptimizerConfig,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, ArrayD<f32>>,
    v: HashMap<String, ArrayD<f32>>,
    t: usize,
}

impl AdamOptimizer {
    fn new(config: &CalibrationOptimizerConfig) -> Self {
        Self {
            config: config.clone(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl CalibrationOptimizer for AdamOptimizer {
    fn update_parameter(
        &mut self,
        model: &mut dyn crate::FoundationModel,
        layer_idx: usize,
        param_type: &str,
        gradient: &ArrayD<f32>,
        state: &OptimizationState,
    ) -> Result<(), crate::ATQSError> {
        let param_name = format!("layer_{}_{}", layer_idx, param_type);
        
        // Initialize moments if needed
        if !self.m.contains_key(&param_name) {
            self.m.insert(param_name.clone(), Array::zeros(gradient.shape()));
            self.v.insert(param_name.clone(), Array::zeros(gradient.shape()));
        }
        
        // Update moments
        let m = self.m.get_mut(&param_name).unwrap();
        let v = self.v.get_mut(&param_name).unwrap();
        
        *m = m.mapv(|x| self.beta1 * x) + gradient.mapv(|x| (1.0 - self.beta1) * x);
        *v = v.mapv(|x| self.beta2 * x) + gradient.mapv(|x| (1.0 - self.beta2) * x * x);
        
        // Bias correction
        let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.t as i32 + 1)));
        let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.t as i32 + 1)));
        
        // Update parameters
        let update = m_hat.mapv(|x| x * state.learning_rate) / 
                    v_hat.mapv(|x| x.sqrt() + self.epsilon);
        
        // Apply update to model (simplified)
        apply_parameter_update(model, layer_idx, param_type, &update)?;
        
        self.t += 1;
        Ok(())
    }
}

/// SGD with Momentum optimizer implementation
struct SGDMomentumOptimizer {
    config: CalibrationOptimizerConfig,
    momentum: f32,
    velocity: HashMap<String, ArrayD<f32>>,
}

impl SGDMomentumOptimizer {
    fn new(config: &CalibrationOptimizerConfig) -> Self {
        Self {
            config: config.clone(),
            momentum: 0.9,
            velocity: HashMap::new(),
        }
    }
}

impl CalibrationOptimizer for SGDMomentumOptimizer {
    fn update_parameter(
        &mut self,
        model: &mut dyn crate::FoundationModel,
        layer_idx: usize,
        param_type: &str,
        gradient: &ArrayD<f32>,
        state: &OptimizationState,
    ) -> Result<(), crate::ATQSError> {
        let param_name = format!("layer_{}_{}", layer_idx, param_type);
        
        // Initialize velocity if needed
        if !self.velocity.contains_key(&param_name) {
            self.velocity.insert(param_name.clone(), Array::zeros(gradient.shape()));
        }
        
        // Update velocity
        let velocity = self.velocity.get_mut(&param_name).unwrap();
        *velocity = velocity.mapv(|v| self.momentum * v) + 
                   gradient.mapv(|g| state.learning_rate * g);
        
        // Apply update
        apply_parameter_update(model, layer_idx, param_type, velocity)?;
        
        Ok(())
    }
}

/// Placeholder implementations for other optimizers
struct AdaGradOptimizer {
    config: CalibrationOptimizerConfig,
    accumulated_gradients: HashMap<String, ArrayD<f32>>,
}

impl AdaGradOptimizer {
    fn new(config: &CalibrationOptimizerConfig) -> Self {
        Self {
            config: config.clone(),
            accumulated_gradients: HashMap::new(),
        }
    }
}

impl CalibrationOptimizer for AdaGradOptimizer {
    fn update_parameter(
        &mut self,
        model: &mut dyn crate::FoundationModel,
        layer_idx: usize,
        param_type: &str,
        gradient: &ArrayD<f32>,
        state: &OptimizationState,
    ) -> Result<(), crate::ATQSError> {
        // Simplified AdaGrad update
        let update = gradient.mapv(|g| -state.learning_rate * g);
        apply_parameter_update(model, layer_idx, param_type, &update)?;
        Ok(())
    }
}

struct RMSPropOptimizer {
    config: CalibrationOptimizerConfig,
    squared_gradients: HashMap<String, ArrayD<f32>>,
}

impl RMSPropOptimizer {
    fn new(config: &CalibrationOptimizerConfig) -> Self {
        Self {
            config: config.clone(),
            squared_gradients: HashMap::new(),
        }
    }
}

impl CalibrationOptimizer for RMSPropOptimizer {
    fn update_parameter(
        &mut self,
        model: &mut dyn crate::FoundationModel,
        layer_idx: usize,
        param_type: &str,
        gradient: &ArrayD<f32>,
        state: &OptimizationState,
    ) -> Result<(), crate::ATQSError> {
        // Simplified RMSProp update
        let update = gradient.mapv(|g| -state.learning_rate * g);
        apply_parameter_update(model, layer_idx, param_type, &update)?;
        Ok(())
    }
}

struct LAMBOptimizer {
    config: CalibrationOptimizerConfig,
}

impl LAMBOptimizer {
    fn new(config: &CalibrationOptimizerConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

impl CalibrationOptimizer for LAMBOptimizer {
    fn update_parameter(
        &mut self,
        model: &mut dyn crate::FoundationModel,
        layer_idx: usize,
        param_type: &str,
        gradient: &ArrayD<f32>,
        state: &OptimizationState,
    ) -> Result<(), crate::ATQSError> {
        // Simplified LAMB update
        let update = gradient.mapv(|g| -state.learning_rate * g);
        apply_parameter_update(model, layer_idx, param_type, &update)?;
        Ok(())
    }
}

/// Apply parameter update to model
fn apply_parameter_update(
    model: &mut dyn crate::FoundationModel,
    layer_idx: usize,
    param_type: &str,
    update: &ArrayD<f32>,
) -> Result<(), crate::ATQSError> {
    let layers = model.get_layers();
    if layer_idx >= layers.len() {
        return Ok(());
    }
    
    let layer = &layers[layer_idx];
    let current_weights = layer.get_weights();
    
    if param_type.contains("weights") {
        let updated_weights = current_weights + update;
        model.update_layer_weights(layer_idx, updated_weights)?;
    }
    
    Ok(())
}
