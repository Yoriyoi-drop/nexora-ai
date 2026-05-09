//! NXR Base Model Trait
//! 
//! Core trait that all NXR models must implement

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    model_identity::{ModelMeta, NxrModelId},
    capability_spec::CapabilityVector,
};

/// Base trait for all NXR models
#[async_trait]
pub trait NxrModel: Send + Sync {
    /// Model configuration type
    type Config: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;
    
    /// Model metrics type
    type Metrics: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;
    
    /// Model state type
    type State: Clone + Send + Sync;

    /// Get model identity
    fn identity(&self) -> &ModelMeta;

    /// Get model capabilities
    fn capabilities(&self) -> &CapabilityVector;

    /// Get model configuration
    fn config(&self) -> &Self::Config;

    /// Get current model state
    async fn state(&self) -> Result<Self::State, NxrModelError>;

    /// Initialize the model
    async fn initialize(&mut self, config: Self::Config) -> Result<(), NxrModelError>;

    /// Reset the model to initial state
    async fn reset(&self) -> Result<(), NxrModelError>;

    /// Get model metrics
    async fn metrics(&self) -> Result<Self::Metrics, NxrModelError>;

    /// Perform inference
    async fn infer(&self, input: &NxrInput) -> Result<NxrOutput, NxrModelError>;

    /// Stream inference
    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> Result<(), NxrModelError>;

    /// Update model configuration
    async fn update_config(&mut self, config: Self::Config) -> Result<(), NxrModelError>;

    /// Validate model state
    async fn validate(&self) -> Result<ValidationResult, NxrModelError>;

    /// Get model statistics
    async fn statistics(&self) -> Result<ModelStatistics, NxrModelError>;

    /// Check if model is ready for inference
    async fn is_ready(&self) -> bool;

    /// Get resource usage
    async fn resource_usage(&self) -> Result<ResourceUsage, NxrModelError>;
}

/// Base model implementation with shared functionality
pub struct BaseNxrModel<C, M, S> {
    /// Model metadata
    meta: ModelMeta,
    /// Model capabilities
    capabilities: CapabilityVector,
    /// Model configuration
    config: Arc<RwLock<C>>,
    /// Current state
    state: Arc<RwLock<S>>,
    /// Model metrics
    metrics: Arc<RwLock<M>>,
    /// Initialization status
    initialized: Arc<RwLock<bool>>,
    /// Model statistics
    statistics: Arc<RwLock<ModelStatistics>>,
}

impl<C, M, S> BaseNxrModel<C, M, S>
where
    C: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    M: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    S: Clone + Send + Sync,
{
    /// Create new base model
    pub fn new(
        meta: ModelMeta,
        capabilities: CapabilityVector,
        config: C,
        initial_state: S,
        initial_metrics: M,
    ) -> Self {
        Self {
            meta,
            capabilities,
            config: Arc::new(RwLock::new(config)),
            state: Arc::new(RwLock::new(initial_state)),
            metrics: Arc::new(RwLock::new(initial_metrics)),
            initialized: Arc::new(RwLock::new(false)),
            statistics: Arc::new(RwLock::new(ModelStatistics::default())),
        }
    }

    /// Get model identity
    pub fn identity(&self) -> &ModelMeta {
        &self.meta
    }

    /// Get model capabilities
    pub fn capabilities(&self) -> &CapabilityVector {
        &self.capabilities
    }

    /// Get model configuration
    pub async fn config(&self) -> C {
        self.config.read().await.clone()
    }

    /// Update model configuration
    pub async fn update_config(&self, new_config: C) -> Result<(), NxrModelError> {
        let mut config = self.config.write().await;
        *config = new_config;
        Ok(())
    }

    /// Get current state
    pub async fn state(&self) -> Result<S, NxrModelError> {
        let state = self.state.read().await;
        Ok(state.clone())
    }

    /// Update state
    pub async fn update_state(&self, new_state: S) -> Result<(), NxrModelError> {
        let mut state = self.state.write().await;
        *state = new_state;
        Ok(())
    }

    /// Get metrics
    pub async fn metrics(&self) -> Result<M, NxrModelError> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }

    /// Update metrics
    pub async fn update_metrics(&self, new_metrics: M) -> Result<(), NxrModelError> {
        let mut metrics = self.metrics.write().await;
        *metrics = new_metrics;
        Ok(())
    }

    /// Mark as initialized
    pub async fn mark_initialized(&self) {
        let mut initialized = self.initialized.write().await;
        *initialized = true;
    }

    /// Check if initialized
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Get statistics
    pub async fn statistics(&self) -> Result<ModelStatistics, NxrModelError> {
        let stats = self.statistics.read().await;
        Ok(stats.clone())
    }

    /// Update statistics
    pub async fn update_statistics(&self, updater: impl FnOnce(&mut ModelStatistics)) -> Result<(), NxrModelError> {
        let mut stats = self.statistics.write().await;
        updater(&mut stats);
        Ok(())
    }
}

/// NXR Model Error
#[derive(Debug, thiserror::Error)]
pub enum NxrModelError {
    #[error("Model not initialized: {0}")]
    NotInitialized(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
    
    #[error("Resource error: {0}")]
    Resource(String),
    
    #[error("State error: {0}")]
    State(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Capability not supported: {0}")]
    UnsupportedCapability(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
}

impl From<&str> for NxrModelError {
    fn from(s: &str) -> Self {
        NxrModelError::Internal(s.to_string())
    }
}

impl From<String> for NxrModelError {
    fn from(s: String) -> Self {
        NxrModelError::Internal(s)
    }
}

pub type NxrModelResult<T> = Result<T, NxrModelError>;

/// NXR Input for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NxrInput {
    /// Input ID
    pub id: uuid::Uuid,
    /// Input timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Input data
    pub data: InputData,
    /// Input parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Request metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Input data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputData {
    /// Text input
    Text(String),
    /// Token input
    Tokens(Vec<u32>),
    /// Image input (base64)
    Image(String),
    /// Audio input (base64)
    Audio(String),
    /// Multimodal input
    Multimodal {
        text: Option<String>,
        images: Vec<String>,
        audio: Option<String>,
    },
    /// Structured input
    Structured(serde_json::Value),
}

/// NXR Output from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NxrOutput {
    /// Output ID
    pub id: uuid::Uuid,
    /// Input ID this output responds to
    pub input_id: uuid::Uuid,
    /// Output timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Output data
    pub data: OutputData,
    /// Generation metadata
    pub metadata: GenerationMetadata,
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// Output data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputData {
    /// Text output
    Text(String),
    /// Token output with probabilities
    Tokens(Vec<TokenOutput>),
    /// Image output (base64)
    Image(String),
    /// Audio output (base64)
    Audio(String),
    /// Multimodal output
    Multimodal {
        text: Option<String>,
        images: Vec<String>,
        audio: Option<String>,
    },
    /// Structured output
    Structured(serde_json::Value),
}

/// Token with probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOutput {
    /// Token ID
    pub token_id: u32,
    /// Token text
    pub text: String,
    /// Log probability
    pub log_prob: f32,
    /// Token position
    pub position: usize,
}

/// Generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Model version used
    pub model_version: String,
    /// Seed used (if deterministic)
    pub seed: Option<u64>,
}

/// Finish reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    /// Maximum tokens reached
    MaxTokens,
    /// Stop sequence encountered
    StopSequence,
    /// End of sequence token
    EndOfSequence,
    /// Timeout
    Timeout,
    /// Cancelled
    Cancelled,
    /// Error
    Error(String),
    /// Unknown reason
    Unknown,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Memory usage in GB
    pub memory_usage_gb: f32,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f32>,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Network usage in MB/s
    pub network_usage_mbps: Option<f32>,
}

/// Stream chunk for streaming inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NxrStreamChunk {
    /// Chunk ID
    pub id: uuid::Uuid,
    /// Input ID
    pub input_id: uuid::Uuid,
    /// Chunk timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Chunk data
    pub data: StreamChunkData,
    /// Is this the final chunk?
    pub is_final: bool,
}

/// Stream chunk data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamChunkData {
    /// Text delta
    TextDelta(String),
    /// Token delta
    TokenDelta(TokenOutput),
    /// Progress update (0.0 - 1.0)
    Progress(f32),
    /// Status update
    Status(String),
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation score (0.0 - 1.0)
    pub score: f32,
}

/// Model statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStatistics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Model uptime in seconds
    pub uptime_seconds: u64,
    /// Last activity timestamp
    pub last_activity: Option<chrono::DateTime<chrono::Utc>>,
}

/// Resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory usage in GB
    pub memory_gb: f32,
    /// CPU usage percentage (0.0 - 100.0)
    pub cpu_percent: f32,
    /// GPU usage percentage (if applicable)
    pub gpu_percent: Option<f32>,
    /// GPU memory usage in GB (if applicable)
    pub gpu_memory_gb: Option<f32>,
    /// Disk usage in GB
    pub disk_gb: f32,
    /// Network usage in MB/s
    pub network_mbps: f32,
    /// Active connections
    pub active_connections: u32,
    /// Queue size
    pub queue_size: u32,
}

impl Default for NxrInput {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            data: InputData::Text(String::new()),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for NxrOutput {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            input_id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            data: OutputData::Text(String::new()),
            metadata: GenerationMetadata {
                finish_reason: FinishReason::Unknown,
                total_tokens: 0,
                generation_time_ms: 0,
                model_version: "1.0.0".to_string(),
                seed: None,
            },
            performance: PerformanceMetrics {
                tokens_per_second: 0.0,
                memory_usage_gb: 0.0,
                gpu_utilization: None,
                cpu_utilization: 0.0,
                network_usage_mbps: None,
            },
        }
    }
}
