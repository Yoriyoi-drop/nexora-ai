// Model Traits for Foundation Components
//
// Traits specific to AI model operations and management

use nexora_atqs::Tensor;
use std::fmt;

/// Trait for model inference operations
/// 
/// This trait defines the interface for running inference with AI models.
/// Implementations should handle the forward pass through the model and
/// return predictions or outputs for given inputs.
/// 
/// # Type Parameters
/// - `Input`: The type of input data the model expects
/// - `Output`: The type of output/predictions the model produces
/// - `Error`: Error type for inference failures
/// 
/// # Examples
/// ```rust
/// struct MyModel {
///     // model fields
/// }
/// 
/// impl ModelInference for MyModel {
///     type Input = Vec<f32>;
///     type Output = Vec<f32>;
///     type Error = MyError;
///     
///     fn infer(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
///         // Run forward pass
///         Ok(self.forward_pass(&input))
///     }
///     
///     fn input_shape(&self) -> &[usize] {
///         &[784] // MNIST flattened image
///     }
///     
///     fn output_shape(&self) -> &[usize] {
///         &[10] // 10 classes
///     }
/// }
/// ```
pub trait ModelInference {
    /// The input type for the model
    type Input;
    
    /// The output type produced by the model
    type Output;
    
    /// Error type for inference operations
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Run inference with the given input
    /// 
    /// # Arguments
    /// * `input` - The input data to process
    /// 
    /// # Returns
    /// * `Ok(output)` - The model's prediction/output
    /// * `Err(error)` - Inference failure
    /// 
    /// # Examples
    /// ```rust
    /// let output = model.infer(input_data)?;
    /// ```
    fn infer(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Batch inference with multiple inputs
    /// 
    /// Default implementation processes inputs sequentially.
    /// Override for optimized batch processing.
    /// 
    /// # Arguments
    /// * `inputs` - Vector of input data to process
    /// 
    /// # Returns
    /// * Vector of outputs corresponding to each input
    /// 
    /// # Performance Note
    /// Consider overriding this method for GPU acceleration or
    /// other batch optimizations.
    fn batch_infer(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>, Self::Error> {
        inputs.into_iter().map(|input| self.infer(input)).collect()
    }
    
    /// Get model input shape requirements
    /// 
    /// # Returns
    /// Slice describing the expected input dimensions
    /// 
    /// # Examples
    /// ```rust
    /// let shape = model.input_shape(); // [224, 224, 3] for RGB image
    /// ```
    fn input_shape(&self) -> &[usize];
    
    /// Get model output shape
    /// 
    /// # Returns
    /// Slice describing the output dimensions
    /// 
    /// # Examples
    /// ```rust
    /// let shape = model.output_shape(); // [1000] for ImageNet classes
    /// ```
    fn output_shape(&self) -> &[usize];
}

/// Trait for model training operations
/// 
/// This trait defines the interface for training AI models. Implementations
/// should handle parameter updates, loss computation, and validation.
/// 
/// # Type Parameters
/// - `TrainingData`: The type of training data the model expects
/// - `Error`: Error type for training failures
/// 
/// # Examples
/// ```rust
/// struct MyTrainer {
///     model: MyModel,
///     optimizer: MyOptimizer,
/// }
/// 
/// impl ModelTraining for MyTrainer {
///     type TrainingData = (Vec<Vec<f32>>, Vec<usize>);
///     type Error = MyError;
///     
///     fn train(&mut self, data: Self::TrainingData) -> Result<(), Self::Error> {
///         for epoch in 0..100 {
///             let loss = self.train_epoch(&data)?;
///             println!("Epoch {} loss: {}", epoch, loss);
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait ModelTraining {
    /// The training data type expected by the model
    type TrainingData;
    
    /// Error type for training operations
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Train the model with the given data
    /// 
    /// This method should perform complete training on the provided dataset.
    /// Default implementation may call train_epoch repeatedly.
    /// 
    /// # Arguments
    /// * `data` - Training dataset
    /// 
    /// # Returns
    /// * `Ok(())` - Training completed successfully
    /// * `Err(error)` - Training failed
    /// 
    /// # Examples
    /// ```rust
    /// trainer.train((inputs, targets))?;
    /// ```
    fn train(&mut self, data: Self::TrainingData) -> Result<(), Self::Error>;
    
    /// Train for one epoch
    /// 
    /// Process the entire training dataset once and return the average loss.
    /// 
    /// # Arguments
    /// * `data` - Training dataset for this epoch
    /// 
    /// # Returns
    /// * `Ok(loss)` - Average loss for this epoch
    /// * `Err(error)` - Training step failed
    /// 
    /// # Examples
    /// ```rust
    /// let epoch_loss = trainer.train_epoch(&batch_data)?;
    /// ```
    fn train_epoch(&mut self, data: Self::TrainingData) -> Result<f32, Self::Error>;
    
    /// Validate the model
    /// 
    /// Evaluate model performance on validation data without updating parameters.
    /// 
    /// # Arguments
    /// * `data` - Validation dataset
    /// 
    /// # Returns
    /// * `Ok(loss)` - Validation loss
    /// * `Err(error)` - Validation failed
    /// 
    /// # Examples
    /// ```rust
    /// let val_loss = trainer.validate(&val_data)?;
    /// ```
    fn validate(&self, data: Self::TrainingData) -> Result<f32, Self::Error>;
    
    /// Get current training loss
    /// 
    /// # Returns
    /// * `Some(loss)` - Current training loss if available
    /// * `None` - Loss not available (e.g., before training starts)
    /// 
    /// # Examples
    /// ```rust
    /// if let Some(loss) = trainer.training_loss() {
    ///     println!("Current loss: {}", loss);
    /// }
    /// ```
    fn training_loss(&self) -> Option<f32>;
    
    /// Get current validation loss
    /// 
    /// # Returns
    /// * `Some(loss)` - Current validation loss if available
    /// * `None` - Validation loss not available
    /// 
    /// # Examples
    /// ```rust
    /// if let Some(val_loss) = trainer.validation_loss() {
    ///     println!("Validation loss: {}", val_loss);
    /// }
    /// ```
    fn validation_loss(&self) -> Option<f32>;
}

/// Trait for model optimization
pub trait ModelOptimization {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Optimize the model (e.g., quantization, pruning)
    fn optimize(&mut self) -> Result<(), Self::Error>;
    
    /// Apply quantization
    fn quantize(&mut self, bits: u8) -> Result<(), Self::Error>;
    
    /// Apply pruning
    fn prune(&mut self, ratio: f32) -> Result<(), Self::Error>;
    
    /// Get model size in bytes
    fn model_size(&self) -> usize;
    
    /// Get number of parameters
    fn parameter_count(&self) -> usize;
}

/// Trait for model serialization
pub trait ModelSerializable {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Save model to file
    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Self::Error>;
    
    /// Load model from file
    fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Self::Error>;
    
    /// Export model to bytes
    fn export(&self) -> Result<Vec<u8>, Self::Error>;
    
    /// Import model from bytes
    fn import(&mut self, data: &[u8]) -> Result<(), Self::Error>;
}

/// Trait for model configuration
pub trait ModelConfigurable<C> {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Set model configuration
    fn set_config(&mut self, config: C) -> Result<(), Self::Error>;
    
    /// Get current configuration
    fn get_config(&self) -> &C;
    
    /// Validate configuration
    fn validate_config(&self, config: &C) -> Result<(), Self::Error>;
}

/// Trait for model metrics and monitoring
pub trait ModelMetrics {
    type Metrics: fmt::Debug + Clone;
    
    /// Get current metrics
    fn get_metrics(&self) -> Self::Metrics;
    
    /// Reset metrics
    fn reset_metrics(&mut self);
    
    /// Log metrics
    fn log_metrics(&self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Trait for model layers
pub trait ModelLayer {
    type Input;
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Forward pass through the layer
    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Get layer parameters
    fn parameters(&self) -> Vec<Tensor>;
    
    /// Get layer gradients
    fn gradients(&self) -> Vec<Tensor>;
    
    /// Update layer parameters
    fn update_parameters(&mut self, learning_rate: f32) -> Result<(), Self::Error>;
    
    /// Get layer name
    fn name(&self) -> &str;
    
    /// Get layer input shape
    fn input_shape(&self) -> &[usize];
    
    /// Get layer output shape
    fn output_shape(&self) -> &[usize];
}

/// Trait for model activation functions
pub trait ActivationFunction {
    /// Apply activation function
    fn activate(&self, x: f32) -> f32;
    
    /// Apply activation function derivative
    fn derivative(&self, x: f32) -> f32;
    
    /// Apply activation to tensor
    fn activate_tensor(&self, tensor: &Tensor) -> Tensor;
    
    /// Apply activation derivative to tensor
    fn derivative_tensor(&self, tensor: &Tensor) -> Tensor;
    
    /// Get activation function name
    fn name(&self) -> &str;
}

/// Trait for model loss functions
pub trait LossFunction {
    type Target;
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Compute loss
    fn compute(&self, prediction: &Tensor, target: Self::Target) -> Result<f32, Self::Error>;
    
    /// Compute loss gradient
    fn gradient(&self, prediction: &Tensor, target: Self::Target) -> Result<Tensor, Self::Error>;
    
    /// Get loss function name
    fn name(&self) -> &str;
}

/// Trait for model optimizers
pub trait Optimizer {
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Update parameters using gradients
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<(), Self::Error>;
    
    /// Reset optimizer state
    fn reset(&mut self);
    
    /// Get learning rate
    fn learning_rate(&self) -> f32;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
    
    /// Get optimizer name
    fn name(&self) -> &str;
}

/// Trait for model evaluation
pub trait ModelEvaluation {
    type TestData;
    type Error: std::error::Error + Send + Sync + 'static;
    
    /// Evaluate model on test data
    fn evaluate(&self, test_data: Self::TestData) -> Result<EvaluationMetrics, Self::Error>;
    
    /// Compute accuracy
    fn accuracy(&self, test_data: Self::TestData) -> Result<f32, Self::Error>;
    
    /// Compute precision
    fn precision(&self, test_data: Self::TestData) -> Result<f32, Self::Error>;
    
    /// Compute recall
    fn recall(&self, test_data: Self::TestData) -> Result<f32, Self::Error>;
    
    /// Compute F1 score
    fn f1_score(&self, test_data: Self::TestData) -> Result<f32, Self::Error>;
}

/// Evaluation metrics for model performance
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub loss: f32,
    pub inference_time_ms: f64,
}

impl Default for EvaluationMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            loss: f32::INFINITY,
            inference_time_ms: 0.0,
        }
    }
}
