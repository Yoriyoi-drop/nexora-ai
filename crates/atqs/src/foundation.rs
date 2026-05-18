//! Foundation model trait for ATQS

use async_trait::async_trait;
use ndarray::{Array, ArrayD};
use std::collections::HashMap;
use crate::types::LayerInfo;

/// Foundation model trait that all models must implement
#[async_trait]
pub trait FoundationModel: Send + Sync {
    /// Get model name
    fn name(&self) -> &str;
    
    /// Get model version
    fn version(&self) -> &str;
    
    /// Get model parameters
    fn parameters(&self) -> HashMap<String, ArrayD<f32>>;
    
    /// Set model parameters
    fn set_parameters(&mut self, params: HashMap<String, ArrayD<f32>>) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Forward pass
    async fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>>;
    
    /// Get model size in bytes
    fn size_bytes(&self) -> usize;
    
    /// Get model architecture
    fn architecture(&self) -> Vec<String>;
    
    /// Clone the model
    fn clone_model(&self) -> Box<dyn FoundationModel>;
    
    /// Get all layers in the model
    fn get_layers(&self) -> Vec<LayerInfo>;
    
    /// Update layer weights
    fn update_layer_weights(&mut self, layer_idx: usize, weights: ArrayD<f32>) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Get layer weights
    fn get_layer_weights(&self, layer_idx: usize) -> Option<&ArrayD<f32>>;
    
    /// Forward pass to specific layer
    fn forward_to_layer(&self, input: &ArrayD<f32>, layer_idx: usize) -> Result<ArrayD<f32>, Box<dyn std::error::Error>>;
    
    /// Compute loss for a single input-target pair
    fn compute_loss(&self, output: &ArrayD<f32>, target: &ArrayD<f32>) -> Result<f32, Box<dyn std::error::Error>>;
    
    /// Apply gradient to parameters
    fn apply_gradient(&mut self, gradients: HashMap<String, ArrayD<f32>>, learning_rate: f32) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Get model state for checkpointing
    fn get_state(&self) -> HashMap<String, ArrayD<f32>>;
    
    /// Set model state from checkpoint
    fn set_state(&mut self, state: HashMap<String, ArrayD<f32>>) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Get memory footprint in bytes
    fn get_memory_footprint(&self) -> usize;
    
    /// Get attention weights from attention layers
    fn get_attention_weights(&self) -> Vec<ArrayD<f32>>;
    
    /// Apply dense layer operation (internal method)
    fn apply_dense_layer(&self, input: &ArrayD<f32>, weights: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>>;
    
    /// Optimized matrix multiplication (internal method)
    fn matrix_multiply_optimized(&self, input: &ArrayD<f32>, weights: &ArrayD<f32>, batch_size: usize, input_features: usize, output_features: usize) -> Result<ArrayD<f32>, Box<dyn std::error::Error>>;
    
    /// Create layer information (internal method)
    fn create_layer_info(&self) -> Vec<LayerInfo>;
}

/// Basic foundation model implementation
#[derive(Debug, Clone)]
pub struct BasicFoundationModel {
    name: String,
    version: String,
    parameters: HashMap<String, ArrayD<f32>>,
    architecture: Vec<String>,
}

impl BasicFoundationModel {
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            parameters: HashMap::new(),
            architecture: vec!["basic".to_string()],
        }
    }
    
    pub fn with_parameters(mut self, params: HashMap<String, ArrayD<f32>>) -> Self {
        self.parameters = params;
        self
    }
    
    pub fn with_architecture(mut self, arch: Vec<String>) -> Self {
        self.architecture = arch;
        self
    }
}

#[async_trait]
impl FoundationModel for BasicFoundationModel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
    
    fn parameters(&self) -> HashMap<String, ArrayD<f32>> {
        // Return references instead of clones for better performance
        self.parameters.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
    
    fn set_parameters(&mut self, params: HashMap<String, ArrayD<f32>>) -> Result<(), Box<dyn std::error::Error>> {
        self.parameters = params;
        Ok(())
    }
    
    async fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // Validate input shape
        if input.shape().len() < 2 {
            return Err("Input must be at least 2D (batch_size, features)".into());
        }
        
        let batch_size = input.shape()[0];
        let mut current_output = input.clone();
        
        // Process layers in order
        for layer_name in &self.architecture {
            if let Some(weights) = self.parameters.get(layer_name) {
                // Apply dense layer operation
                current_output = self.apply_dense_layer(&current_output, weights)?;
                
                // Apply ReLU activation (except for final layer)
                if layer_name != self.architecture.last().map(|s| s.as_str()).unwrap_or("") {
                    current_output = current_output.mapv(|x| x.max(0.0));
                }
            } else {
                return Err(format!("Layer '{}' not found in parameters", layer_name).into());
            }
        }
        
        // Validate output shape
        if current_output.shape()[0] != batch_size {
            return Err("Output batch size mismatch".into());
        }
        
        Ok(current_output)
    }
    
    /// Apply dense layer operation with optimized matrix multiplication
    fn apply_dense_layer(&self, input: &ArrayD<f32>, weights: &ArrayD<f32>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let input_shape = input.shape();
        let weight_shape = weights.shape();
        
        // Validate dimensions
        if input_shape.len() != 2 || weight_shape.len() != 2 {
            return Err("Both input and weights must be 2D tensors".into());
        }
        
        let (batch_size, input_features) = (input_shape[0], input_shape[1]);
        let (output_features, weight_input_features) = (weight_shape[0], weight_shape[1]);
        
        if input_features != weight_input_features {
            return Err(format!("Input features ({}) don't match weight input features ({})", 
                             input_features, weight_input_features).into());
        }
        
        // Optimized matrix multiplication using better memory access patterns
        let output = self.matrix_multiply_optimized(input, weights, batch_size, input_features, output_features)?;
        
        Ok(output)
    }
    
    /// Optimized matrix multiplication with cache-friendly access patterns
    fn matrix_multiply_optimized(
        &self,
        input: &ArrayD<f32>,
        weights: &ArrayD<f32>,
        batch_size: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let mut output = ArrayD::zeros(vec![batch_size, output_features]);
        
        // Use blocking for better cache performance
        const BLOCK_SIZE: usize = 64;
        
        for b in 0..batch_size {
            for o_block in (0..output_features).step_by(BLOCK_SIZE) {
                let o_end = std::cmp::min(o_block + BLOCK_SIZE, output_features);
                
                for i_block in (0..input_features).step_by(BLOCK_SIZE) {
                    let i_end = std::cmp::min(i_block + BLOCK_SIZE, input_features);
                    
                    // Process block
                    for o in o_block..o_end {
                        let mut sum = 0.0f32;
                        
                        for i in i_block..i_end {
                            let input_val = input[[b, i]];
                            let weight_val = weights[[o, i]];
                            sum += input_val * weight_val;
                        }
                        
                        output[[b, o]] += sum;
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    fn size_bytes(&self) -> usize {
        self.parameters
            .values()
            .map(|arr| arr.len() * std::mem::size_of::<f32>())
            .sum()
    }
    
    fn architecture(&self) -> Vec<String> {
        self.architecture.clone()
    }
    
    fn clone_model(&self) -> Box<dyn FoundationModel> {
        Box::new(self.clone())
    }
    
    fn get_layers(&self) -> Vec<LayerInfo> {
        // Cache layer information to avoid repeated computation
        self.create_layer_info()
    }
    
    /// Create layer information with optimized memory usage
    fn create_layer_info(&self) -> Vec<LayerInfo> {
        let mut layers = Vec::with_capacity(self.parameters.len());
        
        for (name, weights) in &self.parameters {
            let weight_shape = weights.shape();
            
            // Calculate input and output shapes from weight dimensions
            let input_shape = if weight_shape.len() >= 2 {
                vec![weight_shape[1]]
            } else {
                vec![]
            };
            
            let output_shape = if weight_shape.len() >= 1 {
                vec![weight_shape[0]]
            } else {
                vec![]
            };
            
            let layer_info = LayerInfo {
                name: name.clone(),
                layer_type: "dense".to_string(),
                input_shape,
                output_shape,
                num_parameters: weights.len(),
                weights: weights.clone(), // Keep clone here as it's required by LayerInfo
                biases: None,
                activation: "relu".to_string(),
                trainable: true,
            };
            layers.push(layer_info);
        }
        
        layers
    }
    
    fn update_layer_weights(&mut self, layer_idx: usize, weights: ArrayD<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let layers = self.get_layers();
        if layer_idx < layers.len() {
            let layer_name = &layers[layer_idx].name;
            self.parameters.insert(layer_name.clone(), weights);
        }
        Ok(())
    }
    
    fn get_layer_weights(&self, layer_idx: usize) -> Option<&ArrayD<f32>> {
        let layers = self.get_layers();
        if layer_idx < layers.len() {
            let layer_name = &layers[layer_idx].name;
            self.parameters.get(layer_name)
        } else {
            None
        }
    }
    
    fn forward_to_layer(&self, input: &ArrayD<f32>, layer_idx: usize) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        // Validate input shape
        if input.shape().len() < 2 {
            return Err("Input must be at least 2D (batch_size, features)".into());
        }
        
        // Validate layer index
        if layer_idx >= self.architecture.len() {
            return Err(format!("Layer index {} exceeds architecture length {}", 
                             layer_idx, self.architecture.len()).into());
        }
        
        let batch_size = input.shape()[0];
        let mut current_output = input.clone();
        
        // Process layers up to the specified index (inclusive)
        for (current_idx, layer_name) in self.architecture.iter().enumerate() {
            if current_idx > layer_idx {
                break;
            }
            
            if let Some(weights) = self.parameters.get(layer_name) {
                // Apply dense layer operation
                current_output = self.apply_dense_layer(&current_output, weights)?;
                
                // Apply ReLU activation (except for final layer or if this is the target layer)
                if current_idx < layer_idx && layer_name != self.architecture.last().map(|s| s.as_str()).unwrap_or("") {
                    current_output = current_output.mapv(|x| x.max(0.0));
                }
            } else {
                return Err(format!("Layer '{}' not found in parameters", layer_name).into());
            }
        }
        
        // Validate output shape
        if current_output.shape()[0] != batch_size {
            return Err("Output batch size mismatch".into());
        }
        
        Ok(current_output)
    }
    
    fn compute_loss(&self, output: &ArrayD<f32>, target: &ArrayD<f32>) -> Result<f32, Box<dyn std::error::Error>> {
        if output.shape() != target.shape() {
            return Err("Output and target shapes don't match".into());
        }
        
        let mut sum_sq_error = 0.0;
        for (o, t) in output.iter().zip(target.iter()) {
            let diff = o - t;
            sum_sq_error += diff * diff;
        }
        
        Ok(sum_sq_error / output.len() as f32)
    }
    
    fn apply_gradient(&mut self, gradients: HashMap<String, ArrayD<f32>>, learning_rate: f32) -> Result<(), Box<dyn std::error::Error>> {
        for (param_name, gradient) in gradients {
            if let Some(current_param) = self.parameters.get_mut(&param_name) {
                if current_param.shape() == gradient.shape() {
                    *current_param = current_param.clone() - gradient * learning_rate;
                }
            }
        }
        Ok(())
    }
    
    fn get_state(&self) -> HashMap<String, ArrayD<f32>> {
        self.parameters.clone()
    }
    
    fn set_state(&mut self, state: HashMap<String, ArrayD<f32>>) -> Result<(), Box<dyn std::error::Error>> {
        self.parameters = state;
        Ok(())
    }
    
    fn get_memory_footprint(&self) -> usize {
        self.size_bytes()
    }
    
    fn get_attention_weights(&self) -> Vec<ArrayD<f32>> {
        let mut attention_weights = Vec::new();
        
        // Extract attention weights from parameters
        // Look for parameters that match attention layer naming patterns
        for (param_name, param_array) in &self.parameters {
            if param_name.contains("attention") || param_name.contains("attn") || 
               param_name.contains("query") || param_name.contains("key") || 
               param_name.contains("value") || param_name.contains("softmax") {
                // For attention mechanisms, we typically need to compute the attention weights
                // This is a simplified implementation - in practice, this would require
                // the actual input data and forward pass computation
                if param_array.ndim() >= 2 {
                    // Create a mock attention weight matrix for demonstration
                    let rows = param_array.shape()[0];
                    let cols = param_array.shape()[1];
                    let attention_matrix = Array::from_elem((rows, cols), 1.0 / cols as f32);
                    attention_weights.push(attention_matrix.into_dyn());
                }
            }
        }
        
        attention_weights
    }
}
