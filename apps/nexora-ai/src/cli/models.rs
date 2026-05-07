//! Model definitions for CLI operations

use serde::{Deserialize, Serialize};

/// Training sample for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub input: String,
    pub target: String,
}

/// Trait for trainable models
#[async_trait::async_trait]
pub trait TrainingModel: Send + Sync {
    /// Train a single step and return loss
    async fn train_step(&mut self, input: &str, target: &str, learning_rate: f32) -> anyhow::Result<f32>;
    
    /// Serialize model for saving
    fn serialize(&self) -> anyhow::Result<Vec<u8>>;
    
    /// Get model name
    fn model_name(&self) -> &str;
}

/// Simple neural network model for training
#[derive(Debug)]
pub struct SimpleNeuralModel {
    pub training_steps: u64,
    pub total_loss: f32,
    pub learning_rate: f32,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    // Simple weight matrices (flattened for simplicity)
    pub weights_input_hidden: Vec<f32>,
    pub weights_hidden_output: Vec<f32>,
    pub bias_hidden: Vec<f32>,
    pub bias_output: Vec<f32>,
}

impl SimpleNeuralModel {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let weights_input_hidden = Self::initialize_weights(input_size * hidden_size);
        let weights_hidden_output = Self::initialize_weights(hidden_size * output_size);
        let bias_hidden = vec![0.0; hidden_size];
        let bias_output = vec![0.0; output_size];
        
        Self {
            training_steps: 0,
            total_loss: 0.0,
            learning_rate: 0.01,
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
    }
    
    fn initialize_weights(size: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut weights = Vec::with_capacity(size);
        let mut hasher = DefaultHasher::new();
        
        for i in 0..size {
            i.hash(&mut hasher);
            let seed = hasher.finish();
            // Simple pseudo-random initialization using hash
            let weight = ((seed as f32) / (u64::MAX as f32)) * 2.0 - 1.0; // [-1, 1]
            weights.push(weight * 0.1); // Scale down for stability
        }
        
        weights
    }
    
    fn text_to_features(&self, text: &str) -> Vec<f32> {
        // Simple text encoding: character frequency + length
        let mut features = vec![0.0; self.input_size];
        
        if text.is_empty() {
            return features;
        }
        
        // Basic character frequency encoding
        for (_i, ch) in text.chars().enumerate() {
            let char_code = ch as usize;
            let feature_idx = char_code % self.input_size;
            if feature_idx < self.input_size {
                features[feature_idx] += 1.0;
            }
        }
        
        // Normalize
        let sum: f32 = features.iter().sum();
        if sum > 0.0 {
            for feature in &mut features {
                *feature /= sum;
            }
        }
        
        features
    }
    
    fn forward(&self, input: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        // Input to hidden layer
        let mut hidden = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                hidden[i] += input[j] * self.weights_input_hidden[i * self.input_size + j];
            }
            hidden[i] += self.bias_hidden[i];
            hidden[i] = hidden[i].max(0.0); // ReLU activation
        }
        
        // Hidden to output layer
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                output[i] += hidden[j] * self.weights_hidden_output[i * self.hidden_size + j];
            }
            output[i] += self.bias_output[i];
        }
        
        (hidden, output)
    }
    
    fn calculate_loss(&self, output: &Vec<f32>, target: &str) -> f32 {
        // Simple loss: mean squared error with target encoding
        let target_features = self.text_to_features(target);
        let mut loss = 0.0;
        
        for i in 0..self.output_size.min(target_features.len()) {
            let diff = output[i] - target_features[i];
            loss += diff * diff;
        }
        
        loss / self.output_size as f32
    }
    
    fn backward(&mut self, input: &Vec<f32>, hidden: &Vec<f32>, output: &Vec<f32>, target: &str) {
        let target_features = self.text_to_features(target);
        
        // Output layer gradients
        let mut output_grad = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            if i < target_features.len() {
                output_grad[i] = 2.0 * (output[i] - target_features[i]) / self.output_size as f32;
            }
        }
        
        // Hidden layer gradients
        let mut hidden_grad = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                hidden_grad[i] += output_grad[j] * self.weights_hidden_output[j * self.hidden_size + i];
            }
            // ReLU derivative
            hidden_grad[i] = if hidden[i] > 0.0 { hidden_grad[i] } else { 0.0 };
        }
        
        // Update weights and biases (simple gradient descent)
        // Update hidden to output weights
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                let grad = output_grad[i] * hidden[j];
                self.weights_hidden_output[i * self.hidden_size + j] -= self.learning_rate * grad;
            }
            self.bias_output[i] -= self.learning_rate * output_grad[i];
        }
        
        // Update input to hidden weights
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                let grad = hidden_grad[i] * input[j];
                self.weights_input_hidden[i * self.input_size + j] -= self.learning_rate * grad;
            }
            self.bias_hidden[i] -= self.learning_rate * hidden_grad[i];
        }
    }
}

#[async_trait::async_trait]
impl TrainingModel for SimpleNeuralModel {
    async fn train_step(&mut self, input: &str, target: &str, learning_rate: f32) -> anyhow::Result<f32> {
        self.learning_rate = learning_rate;
        
        // Convert text to features
        let input_features = self.text_to_features(input);
        
        // Forward pass
        let (hidden, output) = self.forward(&input_features);
        
        // Calculate loss
        let loss = self.calculate_loss(&output, target);
        
        // Backward pass
        self.backward(&input_features, &hidden, &output, target);
        
        // Update training stats
        self.training_steps += 1;
        self.total_loss += loss;
        
        Ok(loss)
    }
    
    fn serialize(&self) -> anyhow::Result<Vec<u8>> {
        let data = serde_json::json!({
            "model_type": "simple_neural",
            "training_steps": self.training_steps,
            "total_loss": self.total_loss,
            "learning_rate": self.learning_rate,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_hidden": self.bias_hidden,
            "bias_output": self.bias_output,
        });
        Ok(serde_json::to_vec(&data)?)
    }
    
    fn model_name(&self) -> &str {
        "simple_neural_model"
    }
}

/// Result of a single evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub input: String,
    pub target: String,
    pub predicted: String,
    pub loss: f32,
    pub correct: bool,
}

/// Complete evaluation report
#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub model_path: String,
    pub test_data_path: String,
    pub total_samples: usize,
    pub average_loss: f32,
    pub accuracy: f32,
    pub correct_predictions: usize,
    pub timestamp: String,
    pub detailed_results: Vec<EvaluationResult>,
}

/// Memory entry structure - re-export from nexora_memory
pub use nexora_memory::MemoryEntry;

/// Memory export data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryExportData {
    pub export_timestamp: String,
    pub short_term: Vec<MemoryEntry>,
    pub session: Vec<MemoryEntry>,
    pub long_term: Vec<MemoryEntry>,
    pub knowledge: Vec<MemoryEntry>,
    pub metadata: MemoryMetadata,
}

/// Memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub version: String,
    pub total_entries: usize,
    pub export_format: String,
}
