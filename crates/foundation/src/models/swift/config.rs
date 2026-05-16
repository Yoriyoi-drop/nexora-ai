//! NXR-SWIFT Configuration
//! 
//! Model-specific configuration for NXR-SWIFT

use serde::{Deserialize, Serialize};
use crate::shared::{model_config::NxrModelConfig, deeplearning_integration::DeepLearningConfig, gnac_integration::GnacIntegrationConfig};

/// NXR-SWIFT Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Stream processing configuration
    pub stream_processing: StreamProcessingConfig,
    /// Workflow integration configuration
    pub workflow_integration: WorkflowIntegrationConfig,
    /// Agent configuration
    pub agents: AgentConfig,
    /// Deep learning configuration
    pub deep_learning: DeepLearningConfig,
    /// GNAC integration configuration
    pub gnac: GnacIntegrationConfig,
}

/// Performance Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Latency optimization
    pub latency_optimization: LatencyOptimization,
    /// Throughput optimization
    pub throughput_optimization: ThroughputOptimization,
    /// Resource optimization
    pub resource_optimization: ResourceOptimization,
}

/// Latency Optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimization {
    /// Target latency
    pub target_latency_ms: u32,
    /// Optimization strategy
    pub optimization_strategy: LatencyStrategy,
    /// Caching enabled
    pub caching_enabled: bool,
    /// Cache size
    pub cache_size_mb: u32,
}

/// Latency Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyStrategy {
    /// Aggressive optimization
    Aggressive,
    /// Balanced optimization
    Balanced,
    /// Conservative optimization
    Conservative,
}

/// Throughput Optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputOptimization {
    /// Target throughput
    pub target_throughput_rps: u32,
    /// Batch size
    pub batch_size: u32,
    /// Parallelization enabled
    pub parallelization_enabled: bool,
}

/// Resource Optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptimization {
    /// CPU optimization
    pub cpu_optimization: CpuOptimization,
    /// Memory optimization
    pub memory_optimization: MemoryOptimization,
    /// GPU optimization
    pub gpu_optimization: GpuOptimization,
}

/// CpuOptimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuOptimization {
    /// Single core
    SingleCore,
    /// Multi-core
    MultiCore { cores: u32 },
    /// Dynamic scaling
    DynamicScaling,
}

/// MemoryOptimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Memory limit
    pub memory_limit_mb: u32,
    /// Garbage collection
    pub garbage_collection: GarbageCollection,
}

/// GarbageCollection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GarbageCollection {
    /// Manual GC
    Manual,
    /// Automatic GC
    Automatic,
    /// Aggressive GC
    Aggressive,
}

/// GpuOptimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuOptimization {
    /// No GPU
    None,
    /// Single GPU
    Single,
    /// Multi GPU
    Multi { gpus: u32 },
}

/// Stream Processing Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessingConfig {
    /// Buffer size
    pub buffer_size: u32,
    /// Processing mode
    pub processing_mode: ProcessingMode,
    /// Backpressure handling
    pub backpressure_handling: BackpressureHandling,
}

/// ProcessingMode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingMode {
    /// Sequential processing
    Sequential,
    /// Parallel processing
    Parallel,
    /// Batch processing
    Batch,
}

/// BackpressureHandling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackpressureHandling {
    /// Drop oldest
    DropOldest,
    /// Drop newest
    DropNewest,
    /// Block
    Block,
    /// Buffer
    Buffer,
}

/// Workflow Integration Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowIntegrationConfig {
    /// Integration APIs
    pub integration_apis: Vec<IntegrationApi>,
    /// Workflow templates
    pub workflow_templates: Vec<WorkflowTemplate>,
    /// Event handling
    pub event_handling: EventHandling,
}

/// IntegrationApi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationApi {
    /// API name
    pub name: String,
    /// API type
    pub api_type: ApiType,
    /// Authentication
    pub authentication: Authentication,
    /// Rate limiting
    pub rate_limiting: RateLimiting,
}

/// ApiType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiType {
    /// REST API
    REST,
    /// GraphQL API
    GraphQL,
    /// WebSocket API
    WebSocket,
    /// gRPC API
    gRPC,
}

/// Authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Authentication {
    /// No authentication
    None,
    /// API key
    ApiKey { key: String },
    /// OAuth
    OAuth { token: String },
    /// JWT
    JWT { token: String },
}

/// RateLimiting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiting {
    /// Requests per second
    pub requests_per_second: u32,
    /// Burst size
    pub burst_size: u32,
}

/// WorkflowTemplate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    /// Template ID
    pub id: uuid::Uuid,
    /// Template name
    pub name: String,
    /// Template steps
    pub steps: Vec<WorkflowStep>,
}

/// WorkflowStep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Step ID
    pub id: uuid::Uuid,
    /// Step name
    pub name: String,
    /// Step type
    pub step_type: StepType,
    /// Step parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// StepType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    /// Processing step
    Processing,
    /// Transformation step
    Transformation,
    /// Validation step
    Validation,
    /// Output step
    Output,
}

/// EventHandling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventHandling {
    /// Event sources
    pub event_sources: Vec<EventSource>,
    /// Event processing
    pub event_processing: EventProcessing,
    /// Batch size for batch processing
    pub batch_size: u32,
}

/// EventSource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSource {
    /// HTTP webhook
    HttpWebhook { url: String },
    /// Message queue
    MessageQueue { queue: String },
    /// Database trigger
    DatabaseTrigger { table: String },
    /// Timer
    Timer { interval_seconds: u64 },
}

/// EventProcessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventProcessing {
    /// Synchronous processing
    Synchronous,
    /// Asynchronous processing
    Asynchronous,
    /// Batch processing
    Batch { batch_size: u32 },
}

/// Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// SPEED-BOOST configuration
    pub speed_boost: SpeedBoostConfig,
    /// FLOW-MANAGER configuration
    pub flow_manager: FlowManagerConfig,
    /// STREAM-PROCESSOR configuration
    pub stream_processor: StreamProcessorConfig,
    /// EDGE-ADAPTER configuration
    pub edge_adapter: EdgeAdapterConfig,
}

/// SpeedBoostConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedBoostConfig {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Cache strategy
    pub cache_strategy: CacheStrategy,
    /// Prefetch enabled
    pub prefetch_enabled: bool,
}

/// OptimizationLevel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Maximum optimization
    Maximum,
}

/// CacheStrategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// No caching
    None,
    /// LRU cache
    LRU,
    /// LFU cache
    LFU,
    /// Custom cache
    Custom { strategy: String },
}

/// FlowManagerConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowManagerConfig {
    /// Workflow orchestration
    pub workflow_orchestration: bool,
    /// Dependency management
    pub dependency_management: bool,
    /// Error handling
    pub error_handling: ErrorHandling,
}

/// ErrorHandling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Fail fast
    FailFast,
    /// Retry
    Retry { max_attempts: u32 },
    /// Circuit breaker
    CircuitBreaker { threshold: u32 },
}

/// StreamProcessorConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessorConfig {
    /// Stream type
    pub stream_type: StreamType,
    /// Window size
    pub window_size: u32,
    /// Sliding window
    pub sliding_window: bool,
}

/// StreamType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamType {
    /// Real-time stream
    RealTime,
    /// Batch stream
    Batch,
    /// Hybrid stream
    Hybrid,
}

/// EdgeAdapterConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeAdapterConfig {
    /// Deployment mode
    pub deployment_mode: DeploymentMode,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Offline capability
    pub offline_capability: bool,
}

/// DeploymentMode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentMode {
    /// Cloud deployment
    Cloud,
    /// Edge deployment
    Edge,
    /// Hybrid deployment
    Hybrid,
}

/// ResourceConstraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// CPU limit
    pub cpu_limit: f32,
    /// Memory limit
    pub memory_limit_mb: u32,
    /// Storage limit
    pub storage_limit_mb: u32,
}

impl Default for SwiftConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Swift),
            performance: PerformanceConfig::default(),
            stream_processing: StreamProcessingConfig::default(),
            workflow_integration: WorkflowIntegrationConfig::default(),
            agents: AgentConfig::default(),
            deep_learning: DeepLearningConfig::star_x(),
            gnac: GnacIntegrationConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            latency_optimization: LatencyOptimization::default(),
            throughput_optimization: ThroughputOptimization::default(),
            resource_optimization: ResourceOptimization::default(),
        }
    }
}

impl Default for LatencyOptimization {
    fn default() -> Self {
        Self {
            target_latency_ms: 50,
            optimization_strategy: LatencyStrategy::Balanced,
            caching_enabled: true,
            cache_size_mb: 1024,
        }
    }
}

impl Default for ThroughputOptimization {
    fn default() -> Self {
        Self {
            target_throughput_rps: 1000,
            batch_size: 32,
            parallelization_enabled: true,
        }
    }
}

impl Default for ResourceOptimization {
    fn default() -> Self {
        Self {
            cpu_optimization: CpuOptimization::DynamicScaling,
            memory_optimization: MemoryOptimization {
                memory_limit_mb: 8192,
                garbage_collection: GarbageCollection::Automatic,
            },
            gpu_optimization: GpuOptimization::Single,
        }
    }
}

impl Default for StreamProcessingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            processing_mode: ProcessingMode::Parallel,
            backpressure_handling: BackpressureHandling::Buffer,
        }
    }
}

impl Default for WorkflowIntegrationConfig {
    fn default() -> Self {
        Self {
            integration_apis: vec![],
            workflow_templates: vec![],
            event_handling: EventHandling {
                event_sources: vec![],
                event_processing: EventProcessing::Asynchronous,
                batch_size: 32,
            },
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            speed_boost: SpeedBoostConfig::default(),
            flow_manager: FlowManagerConfig::default(),
            stream_processor: StreamProcessorConfig::default(),
            edge_adapter: EdgeAdapterConfig::default(),
        }
    }
}

impl Default for SpeedBoostConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Aggressive,
            cache_strategy: CacheStrategy::LRU,
            prefetch_enabled: true,
        }
    }
}

impl Default for FlowManagerConfig {
    fn default() -> Self {
        Self {
            workflow_orchestration: true,
            dependency_management: true,
            error_handling: ErrorHandling::Retry { max_attempts: 3 },
        }
    }
}

impl Default for StreamProcessorConfig {
    fn default() -> Self {
        Self {
            stream_type: StreamType::RealTime,
            window_size: 1000,
            sliding_window: true,
        }
    }
}

impl Default for EdgeAdapterConfig {
    fn default() -> Self {
        Self {
            deployment_mode: DeploymentMode::Hybrid,
            resource_constraints: ResourceConstraints {
                cpu_limit: 2.0,
                memory_limit_mb: 4096,
                storage_limit_mb: 10240,
            },
            offline_capability: true,
        }
    }
}

impl SwiftConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate performance configuration
        if self.performance.latency_optimization.target_latency_ms < 10 {
            return Err("Target latency too low (minimum 10ms)".to_string());
        }

        if self.performance.throughput_optimization.target_throughput_rps == 0 {
            return Err("Target throughput must be > 0".to_string());
        }

        // Validate stream processing configuration
        if self.stream_processing.buffer_size == 0 {
            return Err("Buffer size must be > 0".to_string());
        }

        // Validate workflow integration
        if self.workflow_integration.event_handling.batch_size == 0 {
            return Err("Batch size must be > 0".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific agent
    pub fn get_agent_config(&self, agent_name: &str) -> Option<serde_json::Value> {
        match agent_name {
            "speed_boost" => Some(serde_json::to_value(&self.agents.speed_boost).unwrap_or_default()),
            "flow_manager" => Some(serde_json::to_value(&self.agents.flow_manager).unwrap_or_default()),
            "stream_processor" => Some(serde_json::to_value(&self.agents.stream_processor).unwrap_or_default()),
            "edge_adapter" => Some(serde_json::to_value(&self.agents.edge_adapter).unwrap_or_default()),
            _ => None,
        }
    }
}
