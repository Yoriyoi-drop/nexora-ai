//! NXR Model Configuration
//! 
//! Universal configuration schema for all NXR models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Universal NXR Model Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NxrModelConfig {
    /// Model identifier
    pub model_id: crate::model_identity::NxrModelId,
    /// Model version
    pub version: String,
    /// Model architecture configuration
    pub architecture: ArchitectureConfig,
    /// Inference configuration
    pub inference: InferenceConfig,
    /// Resource configuration
    pub resources: ResourceConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Feature flags
    pub features: FeatureFlags,
    /// Sandbox configuration for high-risk models
    pub sandbox: SandboxConfig,
    /// Custom configuration parameters
    pub custom: HashMap<String, serde_json::Value>,
}

/// Architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Model type
    pub model_type: ModelType,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<usize>,
    /// Intermediate size
    pub intermediate_size: Option<usize>,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// MoE configuration (if applicable)
    pub moe_config: Option<MoeConfig>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Normalization type
    pub normalization: NormalizationType,
    /// Position embedding type
    pub position_embedding: PositionEmbeddingType,
    /// Rope theta (for RoPE)
    pub rope_theta: Option<f32>,
    /// Use bias
    pub use_bias: bool,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Attention dropout rate
    pub attention_dropout_rate: f32,
}

/// Model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Standard transformer
    Transformer,
    /// Mixture of Experts
    MixtureOfExperts,
    /// Diffusion model
    Diffusion,
    /// Convolutional neural network
    Convolutional,
    /// Recurrent neural network
    Recurrent,
    /// Hybrid architecture
    Hybrid { components: Vec<ModelType> },
    /// Custom architecture
    Custom { name: String },
}

/// MoE (Mixture of Experts) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to activate
    pub num_experts_per_token: usize,
    /// Expert capacity factor
    pub capacity_factor: f32,
    /// Load balancing loss coefficient
    pub load_balancing_loss_coef: f32,
    /// Expert routing method
    pub routing_method: RoutingMethod,
}

/// Expert routing method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingMethod {
    /// Top-k routing
    TopK,
    /// Noisy top-k routing
    NoisyTopK,
    /// Learned routing
    Learned,
    /// Random routing
    Random,
}

/// Activation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// ReLU
    ReLU,
    /// GELU
    GELU,
    /// SwiGLU
    SwiGLU,
    /// SiLU
    SiLU,
    /// Tanh
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Custom activation
    Custom { name: String },
}

/// Normalization type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Layer normalization
    LayerNorm,
    /// RMS normalization
    RMSNorm,
    /// Batch normalization
    BatchNorm,
    /// Group normalization
    GroupNorm { num_groups: usize },
}

/// Position embedding type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionEmbeddingType {
    /// Absolute position embeddings
    Absolute,
    /// Relative position embeddings
    Relative,
    /// Rotary position embeddings (RoPE)
    Rotary,
    /// ALiBi position embeddings
    Alibi,
    /// No position embeddings
    None,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Sampling method
    pub sampling_method: SamplingMethod,
    /// Use KV cache
    pub use_kv_cache: bool,
    /// Beam search configuration
    pub beam_search_config: Option<BeamSearchConfig>,
}

/// Sampling method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Greedy sampling
    Greedy,
    /// Temperature sampling
    Temperature,
    /// Top-p sampling
    TopP,
    /// Top-k sampling
    TopK,
    /// Combined top-p and top-k
    TopPK,
    /// Beam search
    BeamSearch,
    /// Custom sampling
    Custom { name: String },
}

/// Beam search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamSearchConfig {
    /// Number of beams
    pub num_beams: usize,
    /// Length penalty
    pub length_penalty: f32,
    /// Early stopping
    pub early_stopping: bool,
    /// Divergence penalty
    pub divergence_penalty: f32,
}

/// Resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Memory configuration
    pub memory: MemoryConfig,
    /// Compute configuration
    pub compute: ComputeConfig,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Network configuration
    pub network: NetworkConfig,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Minimum memory in GB
    pub min_memory_gb: f32,
    /// Maximum memory in GB
    pub max_memory_gb: Option<f32>,
    /// Memory optimization level
    pub optimization_level: MemoryOptimizationLevel,
    /// Use memory mapping
    pub use_mmap: bool,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Memory optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// KV cache size in GB
    pub kv_cache_gb: Option<f32>,
    /// Use gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Use offloading
    pub offloading: bool,
    /// Offload directory
    pub offload_dir: Option<String>,
}

/// Compute configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    /// Number of CPU cores
    pub num_cpu_cores: Option<usize>,
    /// GPU configuration
    pub gpu_config: Option<GpuConfig>,
    /// Batch size
    pub batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Use parallel processing
    pub use_parallel: bool,
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// GPU device indices
    pub device_ids: Vec<u32>,
    /// GPU memory in GB
    pub memory_gb: f32,
    /// Use mixed precision
    pub mixed_precision: bool,
    /// Tensor parallel size
    pub tensor_parallel_size: Option<usize>,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Model directory
    pub model_dir: String,
    /// Cache directory
    pub cache_dir: String,
    /// Use compression
    pub use_compression: bool,
    /// Compression level
    pub compression_level: Option<u8>,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Enable network access
    pub enable_network: bool,
    /// API endpoints
    pub api_endpoints: Vec<String>,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Retry configuration
    pub retry_config: Option<RetryConfig>,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retries
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Exponential backoff
    pub exponential_backoff: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Profiling configuration
    pub profiling: ProfilingConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// Optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Performance optimization
    Performance,
    /// Maximum optimization
    Maximum,
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable profiling
    pub enabled: bool,
    /// Profile memory
    pub profile_memory: bool,
    /// Profile compute
    pub profile_compute: bool,
    /// Profile output directory
    pub output_dir: Option<String>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,
    /// Enable health checks
    pub health_checks: bool,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
}

/// Sandbox configuration for high-risk models (Cipher, Genesis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enable sandbox isolation
    pub enabled: bool,
    /// Network access (default: blocked for high-risk models)
    pub network_access: bool,
    /// Maximum memory in MB
    pub max_memory_mb: u64,
    /// Maximum CPU cores
    pub max_cpu_cores: u32,
    /// Maximum execution time in seconds
    pub max_execution_seconds: u64,
    /// Allow filesystem write
    pub allow_filesystem_write: bool,
    /// Monitor all operations
    pub enable_monitoring: bool,
    /// Allow external API calls
    pub allow_external_apis: bool,
    /// Consent token required for sensitive operations
    pub consent_required: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            network_access: false,
            max_memory_mb: 4096,
            max_cpu_cores: 2,
            max_execution_seconds: 300,
            allow_filesystem_write: false,
            enable_monitoring: true,
            allow_external_apis: false,
            consent_required: true,
        }
    }
}

/// Feature flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable streaming inference
    pub enable_streaming: bool,
    /// Enable batch inference
    pub enable_batch: bool,
    /// Enable async processing
    pub enable_async: bool,
    /// Enable caching
    pub enable_cache: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// Enable encryption
    pub enable_encryption: bool,
    /// Enable debug mode
    pub debug_mode: bool,
    /// Enable experimental features
    pub experimental_features: bool,
}

impl Default for NxrModelConfig {
    fn default() -> Self {
        Self {
            model_id: crate::model_identity::NxrModelId::Omnis,
            version: "1.0.0".to_string(),
            architecture: ArchitectureConfig::default(),
            inference: InferenceConfig::default(),
            resources: ResourceConfig::default(),
            performance: PerformanceConfig::default(),
            features: FeatureFlags::default(),
            sandbox: SandboxConfig::default(),
            custom: HashMap::new(),
        }
    }
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Transformer,
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            num_kv_heads: None,
            intermediate_size: Some(3072),
            max_sequence_length: 2048,
            vocab_size: 50257,
            moe_config: None,
            activation: ActivationFunction::GELU,
            normalization: NormalizationType::RMSNorm,
            position_embedding: PositionEmbeddingType::Rotary,
            rope_theta: Some(10000.0),
            use_bias: true,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.1,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: Vec::new(),
            sampling_method: SamplingMethod::Temperature,
            use_kv_cache: true,
            beam_search_config: None,
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            memory: MemoryConfig::default(),
            compute: ComputeConfig::default(),
            storage: StorageConfig::default(),
            network: NetworkConfig::default(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            min_memory_gb: 4.0,
            max_memory_gb: None,
            optimization_level: MemoryOptimizationLevel::Basic,
            use_mmap: false,
            cache_config: CacheConfig::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            kv_cache_gb: None,
            gradient_checkpointing: false,
            offloading: false,
            offload_dir: None,
        }
    }
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            num_cpu_cores: None,
            gpu_config: None,
            batch_size: 1,
            max_batch_size: 8,
            use_parallel: true,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            model_dir: "./models".to_string(),
            cache_dir: "./cache".to_string(),
            use_compression: false,
            compression_level: None,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enable_network: false,
            api_endpoints: Vec::new(),
            timeout_seconds: 30,
            retry_config: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            profiling: ProfilingConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            profile_memory: false,
            profile_compute: false,
            output_dir: None,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_interval_seconds: 60,
            health_checks: true,
            health_check_interval_seconds: 30,
        }
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            enable_batch: true,
            enable_async: true,
            enable_cache: true,
            enable_compression: false,
            enable_encryption: false,
            debug_mode: false,
            experimental_features: false,
        }
    }
}

impl NxrModelConfig {
    /// Create configuration for specific model
    pub fn for_model(model_id: crate::model_identity::NxrModelId) -> Self {
        let mut config = Self::default();
        config.model_id = model_id;
        
        // Apply model-specific defaults
        match model_id {
            crate::model_identity::NxrModelId::Omnis => {
                config.architecture.hidden_size = 8192;
                config.architecture.num_layers = 64;
                config.architecture.num_attention_heads = 64;
                config.architecture.max_sequence_length = 10_000_000;
                config.architecture.model_type = ModelType::MixtureOfExperts;
                config.architecture.moe_config = Some(MoeConfig {
                    num_experts: 8,
                    num_experts_per_token: 2,
                    capacity_factor: 1.25,
                    load_balancing_loss_coef: 0.01,
                    routing_method: RoutingMethod::TopK,
                });
                config.resources.memory.min_memory_gb = 64.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0, 1, 2, 3],
                    memory_gb: 48.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(4),
                });
            }
            crate::model_identity::NxrModelId::Swift => {
                config.architecture.hidden_size = 512;
                config.architecture.num_layers = 8;
                config.architecture.num_attention_heads = 8;
                config.architecture.max_sequence_length = 65536;
                config.resources.memory.min_memory_gb = 1.0;
                config.resources.compute.gpu_config = None; // CPU only for edge
                config.features.enable_streaming = true;
                config.features.enable_compression = true;
            }
            crate::model_identity::NxrModelId::Vortex => {
                config.architecture.hidden_size = 4096;
                config.architecture.num_layers = 32;
                config.architecture.num_attention_heads = 32;
                config.architecture.max_sequence_length = 1_000_000;
                config.resources.memory.min_memory_gb = 32.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0, 1],
                    memory_gb: 32.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(2),
                });
                config.features.enable_streaming = true;
            }
            crate::model_identity::NxrModelId::Nexum => {
                config.architecture.hidden_size = 2048;
                config.architecture.num_layers = 16;
                config.architecture.num_attention_heads = 16;
                config.architecture.max_sequence_length = 512_000;
                config.resources.memory.min_memory_gb = 16.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0],
                    memory_gb: 24.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(1),
                });
            }
            crate::model_identity::NxrModelId::Spectra => {
                config.architecture.hidden_size = 1024;
                config.architecture.num_layers = 12;
                config.architecture.num_attention_heads = 12;
                config.architecture.max_sequence_length = 256_000;
                config.resources.memory.min_memory_gb = 8.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0],
                    memory_gb: 16.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(1),
                });
            }
            crate::model_identity::NxrModelId::Genesis => {
                config.architecture.hidden_size = 6144;
                config.architecture.num_layers = 48;
                config.architecture.num_attention_heads = 48;
                config.architecture.max_sequence_length = 2_000_000;
                config.architecture.model_type = ModelType::MixtureOfExperts;
                config.architecture.moe_config = Some(MoeConfig {
                    num_experts: 6,
                    num_experts_per_token: 2,
                    capacity_factor: 1.2,
                    load_balancing_loss_coef: 0.01,
                    routing_method: RoutingMethod::TopK,
                });
                config.resources.memory.min_memory_gb = 48.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0, 1, 2],
                    memory_gb: 40.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(3),
                });
                config.sandbox = SandboxConfig {
                    enabled: true,
                    network_access: false,
                    max_memory_mb: 8192,
                    max_cpu_cores: 4,
                    max_execution_seconds: 600,
                    allow_filesystem_write: false,
                    enable_monitoring: true,
                    allow_external_apis: false,
                    consent_required: false,
                };
            }
            crate::model_identity::NxrModelId::Kronos => {
                config.architecture.hidden_size = 3072;
                config.architecture.num_layers = 24;
                config.architecture.num_attention_heads = 24;
                config.architecture.max_sequence_length = 128_000;
                config.resources.memory.min_memory_gb = 24.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0],
                    memory_gb: 32.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(1),
                });
            }
            crate::model_identity::NxrModelId::Cipher => {
                config.architecture.hidden_size = 768;
                config.architecture.num_layers = 6;
                config.architecture.num_attention_heads = 8;
                config.architecture.max_sequence_length = 64_000;
                config.features.enable_encryption = true;
                config.resources.memory.min_memory_gb = 4.0;
                config.resources.compute.gpu_config = None;
                config.sandbox = SandboxConfig {
                    enabled: true,
                    network_access: false,
                    max_memory_mb: 4096,
                    max_cpu_cores: 2,
                    max_execution_seconds: 300,
                    allow_filesystem_write: false,
                    enable_monitoring: true,
                    allow_external_apis: false,
                    consent_required: true,
                };
            }
            crate::model_identity::NxrModelId::Axiom => {
                config.architecture.hidden_size = 2048;
                config.architecture.num_layers = 16;
                config.architecture.num_attention_heads = 16;
                config.architecture.max_sequence_length = 256_000;
                config.architecture.model_type = ModelType::MixtureOfExperts;
                config.architecture.moe_config = Some(MoeConfig {
                    num_experts: 4,
                    num_experts_per_token: 2,
                    capacity_factor: 1.1,
                    load_balancing_loss_coef: 0.005,
                    routing_method: RoutingMethod::TopK,
                });
                config.resources.memory.min_memory_gb = 16.0;
                config.resources.compute.gpu_config = Some(GpuConfig {
                    device_ids: vec![0],
                    memory_gb: 24.0,
                    mixed_precision: true,
                    tensor_parallel_size: Some(1),
                });
            }
            _ => {}
        }
        
        config
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate architecture
        if self.architecture.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }
        
        if self.architecture.num_layers == 0 {
            return Err("num_layers must be > 0".to_string());
        }
        
        if self.architecture.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }
        
        if self.architecture.max_sequence_length == 0 {
            return Err("max_sequence_length must be > 0".to_string());
        }
        
        // Validate MoE configuration
        if let Some(moe) = &self.architecture.moe_config {
            if moe.num_experts == 0 {
                return Err("num_experts must be > 0".to_string());
            }
            
            if moe.num_experts_per_token == 0 || moe.num_experts_per_token > moe.num_experts {
                return Err("num_experts_per_token must be > 0 and <= num_experts".to_string());
            }
        }
        
        // Validate inference parameters
        if self.inference.max_new_tokens == 0 {
            return Err("max_new_tokens must be > 0".to_string());
        }
        
        if !(0.0..=2.0).contains(&self.inference.temperature) {
            return Err("temperature must be between 0.0 and 2.0".to_string());
        }
        
        if !(0.0..=1.0).contains(&self.inference.top_p) {
            return Err("top_p must be between 0.0 and 1.0".to_string());
        }
        
        Ok(())
    }

    /// Get model-specific configuration override
    pub fn get_custom<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.custom.get(key).and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Set custom configuration
    pub fn set_custom<T>(&mut self, key: String, value: T) -> Result<(), serde_json::Error>
    where
        T: Serialize,
    {
        let json_value = serde_json::to_value(value)?;
        self.custom.insert(key, json_value);
        Ok(())
    }
}
