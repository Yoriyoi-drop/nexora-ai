//! Nexora-AI Main Library - Pure Rust Entry Point
//!
//! Main entry point for Nexora AI system - delegates to specialized crates

use crate::error::{NexoraError, NexoraResult};
use chrono::Utc;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

pub mod api;
pub mod auth;
pub mod cli;
pub mod config;
pub mod core;
pub mod error;
pub mod metrics;
pub mod security;
pub mod server;

pub use api::{ApiConfig, ApiResponse, NexoraApi};
pub use cli::Cli;
pub use nexora_api;
pub use config::NexoraConfig;
pub use core::*;
pub use server::{NexoraServer, ServerConfig};

// --- Foundation model integration ---
use nexora_foundation::shared::{model_identity::NxrModelId, model_registry::global_registry};
use nexora_tokenizer::{BpeConfig, BpeTokenizer};
/// Create an InferenceEngine sharing the model from the registry via with_model().
/// This avoids InferenceEngine::new() which creates a separate uninitialized CausalLM.
pub async fn create_inference_engine(
    registry: &nexora_foundation::shared::model_registry::NxrModelRegistry,
    model_id: &NxrModelId,
    config: nexora_inference::InferenceConfig,
) -> Result<nexora_inference::InferenceEngineStruct, NexoraError> {
    create_inference_engine_inner(
        registry,
        model_id,
        config,
        false,
        "127.0.0.1:8080",
        &[],
        1000,
    )
    .await
}

/// Create an InferenceEngine with optional distributed cluster support.
pub async fn create_inference_engine_inner(
    registry: &nexora_foundation::shared::model_registry::NxrModelRegistry,
    model_id: &NxrModelId,
    mut config: nexora_inference::InferenceConfig,
    enable_distributed: bool,
    listen_address: &str,
    seed_nodes: &[String],
    gossip_interval_ms: u64,
) -> Result<nexora_inference::InferenceEngineStruct, NexoraError> {
    let model_raw = registry
        .get_model_raw(model_id)
        .await
        .map_err(|e| NexoraError::model(format!("Model {} not found: {}", model_id, e)))?;

    let causal_lm_model = model_raw
        .downcast_ref::<nexora_foundation::causal_lm_model::CausalLmModel>()
        .ok_or_else(|| NexoraError::model(format!("Model {} is not a CausalLmModel", model_id)))?;

    let model_arc = causal_lm_model
        .get_model_arc()
        .await
        .ok_or_else(|| NexoraError::model(format!("Model {} not loaded", model_id)))?;

    let tokenizer = Arc::new(Mutex::new(BpeTokenizer::new(BpeConfig {
        vocab_size: model_arc.config.vocab_size,
        ..Default::default()
    })));

    if enable_distributed {
        config.enable_distributed = true;
        config.distributed_listen_address = listen_address.to_string();
        let cluster_cfg = nexora_runtime::cluster::ClusterConfig {
            node_id: uuid::Uuid::new_v4(),
            listen_address: listen_address.to_string(),
            gossip_interval_ms,
            seed_nodes: seed_nodes.to_vec(),
            ..Default::default()
        };
        let node_registry =
            Arc::new(nexora_runtime::cluster::NodeRegistry::new(cluster_cfg));
        let node_id = node_registry.local_info().await.node_id;
        let mut engine = nexora_inference::InferenceEngineStruct::with_model(
            model_arc,
            Some(tokenizer),
            config,
        )
        .with_distributed(node_registry, node_id);
        engine.initialize().await.map_err(|e| {
            NexoraError::system(format!("Failed to initialize inference engine: {}", e))
        })?;
        Ok(engine)
    } else {
        let mut engine =
            nexora_inference::InferenceEngineStruct::with_model(model_arc, Some(tokenizer), config);
        engine.initialize().await.map_err(|e| {
            NexoraError::system(format!("Failed to initialize inference engine: {}", e))
        })?;
        Ok(engine)
    }
}

/// Adapter that bridges the inference engine to the agent system's InferenceEngine trait
struct NexoraInferenceEngine {
    engine: Arc<nexora_inference::InferenceEngineStruct>,
    model_id: String,
}

impl NexoraInferenceEngine {
    fn new(engine: Arc<nexora_inference::InferenceEngineStruct>, model_id: String) -> Self {
        Self { engine, model_id }
    }
}

#[async_trait::async_trait]
impl nexora_agent::inference_agent::InferenceEngine for NexoraInferenceEngine {
    async fn start_session(&self, _session_id: Uuid, _model_id: &str, _config: &Value) -> nexora_agent::Result<()> {
        Ok(())
    }

    async fn generate_tokens(
        &self,
        _session_id: Uuid,
        prompt: &str,
        max_tokens: u32,
    ) -> nexora_agent::Result<String> {
        let request = nexora_inference::InferenceRequest {
            request_id: Uuid::new_v4(),
            model_id: self.model_id.clone(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.7,
            top_k: 50,
            top_p: 1.0,
            streaming: false,
            ..Default::default()
        };
        let response = self
            .engine
            .generate_internal(request)
            .await
            .map_err(|e| nexora_agent::AgentError::ProcessingError {
                operation: "generate_tokens".to_string(),
                reason: e.to_string(),
            })?;
        Ok(response.text)
    }

    async fn stream_tokens(
        &self,
        _session_id: Uuid,
        prompt: &str,
        max_tokens: u32,
    ) -> nexora_agent::Result<Box<dyn futures::Stream<Item = nexora_agent::Result<String>> + Send>> {
        let request = nexora_inference::InferenceRequest {
            request_id: Uuid::new_v4(),
            model_id: self.model_id.clone(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 0.7,
            top_k: 50,
            top_p: 1.0,
            streaming: false,
            ..Default::default()
        };
        let response = self
            .engine
            .generate_internal(request)
            .await
            .map_err(|e| nexora_agent::AgentError::ProcessingError {
                operation: "stream_tokens".to_string(),
                reason: e.to_string(),
            })?;
        let stream = futures::stream::once(async move { Ok(response.text) });
        Ok(Box::new(stream))
    }

    async fn stop_session(&self, _session_id: Uuid) -> nexora_agent::Result<()> {
        Ok(())
    }

    async fn get_session_status(&self, _session_id: Uuid) -> nexora_agent::Result<nexora_agent::inference_agent::InferenceSessionStatus> {
        Ok(nexora_agent::inference_agent::InferenceSessionStatus::Ready)
    }

    async fn health_check(&self) -> nexora_agent::Result<bool> {
        Ok(true)
    }
}

/// Nexora AI System - Main orchestrator for all AI components
///
/// This is the central coordinator that manages all Nexora AI components including:
/// - Model management and loading
/// - Request processing and routing
/// - System monitoring and health checks
/// - Text generation and chat functionality
/// - Code analysis and generation
///
/// The system uses the NXR foundation model series for all AI inference.
#[derive(Clone)]
pub struct NexoraAI {
    /// Foundation model registry (all registered NXR models)
    registry: Arc<nexora_foundation::shared::model_registry::NxrModelRegistry>,

    /// Active model ID currently in use
    active_model_id: NxrModelId,

    /// Inference engine (primary path — wraps CausalLM + BpeTokenizer + scheduler)
    inference_engine: Arc<nexora_inference::InferenceEngineStruct>,

    /// System configuration
    config: NexoraConfig,

    /// System startup timestamp for uptime tracking
    start_time: chrono::DateTime<Utc>,

    /// Cached system information to avoid frequent system calls
    system_info_cache: Arc<RwLock<Option<SystemInfo>>>,

    /// Total request counter for metrics
    request_count: Arc<AtomicU64>,

    /// System monitoring and health check component
    system_monitor: SystemMonitor,

    /// Request processor for handling incoming requests
    request_processor: RequestProcessor,

    /// Agent manager for multi-agent orchestration
    agent_manager: Arc<nexora_agent::AgentManager>,

    /// Shared training metrics store (written by CLI, served to dashboard)
    pub train_metrics: Arc<RwLock<Value>>,
}

impl NexoraAI {
    /// Create new Nexora AI instance with foundation models
    pub async fn new(config: NexoraConfig) -> NexoraResult<Self> {
        info!("Initializing Nexora AI system with NXR foundation models...");

        config
            .validate()
            .map_err(|e| NexoraError::config(format!("Configuration validation failed: {}", e)))?;

        // Step 0: Initialize GPU context — if unavailable, all GPU paths are disabled
        #[cfg(feature = "gpu")]
        let gpu_ok = {
            match nexora_deeplearning::gpu::GpuContext::init() {
                Ok(ctx) => {
                    info!("GPU initialized: {}", ctx.adapter_info().name);
                    true
                }
                Err(e) => {
                    warn!("GPU initialization failed ({}), disabling GPU inference paths", e);
                    false
                }
            }
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_ok = false;

        // Step 1: Initialize foundation models — all 10 NXR models get real CausalLM instances
        nexora_foundation::init::initialize_foundation_models()
            .await
            .map_err(|e| {
                NexoraError::system(format!("Failed to initialize foundation models: {}", e))
            })?;

        // Step 1b: Initialize model-internal agents — all 10 NXR model agent systems go live
        nexora_foundation::model_agent_manager::init_model_agents().await;

        let registry = global_registry();

        let active_model_id = config.models.active_model.unwrap_or(NxrModelId::Omnis);
        info!(
            "Active model: {} ({})",
            active_model_id,
            active_model_id.fullname()
        );

        // Step 2: Initialize system monitoring and shared state
        let mut system = sysinfo::System::new_all();
        system.refresh_all();

        debug!(
            "System initialized with {} cores, {}MB total memory",
            system.cpus().len(),
            system.total_memory() / (1024 * 1024)
        );

        let system_info_cache = Arc::new(RwLock::new(None));
        let request_count = Arc::new(AtomicU64::new(0));

        let system_monitor = SystemMonitor::new(
            registry.clone(),
            config.clone(),
            Utc::now(),
            system_info_cache.clone(),
            request_count.clone(),
        );

        let train_metrics = Arc::new(RwLock::new(serde_json::json!({
            "status": "idle",
            "history": { "steps": [], "losses": [], "lrs": [], "val_losses": [] }
        })));

        // Step 3: Initialize inference engine (primary generation path)
        let engine_config = nexora_inference::InferenceConfig {
            default_model_id: active_model_id.to_string(),
            max_concurrent_requests: 4,
            enable_streaming: true,
            use_gpu: gpu_ok,
            #[cfg(feature = "gpu")]
            use_gpu_resident: gpu_ok,
            enable_distributed: config.core.enable_distributed,
            distributed_listen_address: config.core.distributed_listen_address.clone(),
            distributed_seed_nodes: config.core.distributed_seed_nodes.clone(),
            ..Default::default()
        };

        let inference_engine = Arc::new(
            create_inference_engine_inner(
                &registry,
                &active_model_id,
                engine_config,
                config.core.enable_distributed,
                &config.core.distributed_listen_address,
                &config.core.distributed_seed_nodes,
                config.core.distributed_gossip_interval_ms,
            )
            .await
            .map_err(|e| {
                NexoraError::system(format!("Failed to initialize inference engine: {}", e))
            })?,
        );
        info!("Inference engine ready (model: {})", active_model_id);

        // Step 4: Initialize multi-agent system — wire engine, start, spawn 7 agents
        let agent_config = nexora_agent::agent_manager::AgentManagerConfig::default();
        let agent_manager = Arc::new(nexora_agent::AgentManager::new(agent_config));

        // Wire inference engine into agent system
        let nexus_engine: Arc<dyn nexora_agent::inference_agent::InferenceEngine> =
            Arc::new(NexoraInferenceEngine::new(
                Arc::clone(&inference_engine),
                active_model_id.to_string(),
            ));
        agent_manager.set_inference_engine(nexus_engine).await;

        agent_manager
            .start()
            .await
            .map_err(|e| NexoraError::system(format!("Failed to start agent manager: {}", e)))?;

        // Spawn all 7 agents: context, routing, inference, memory, planner, response, validation
        let agent_types = vec![
            "context",
            "routing",
            "inference",
            "memory",
            "planner",
            "response",
            "validation",
        ];
        let mut spawned = Vec::new();
        for agent_type in agent_types {
            let cfg = nexora_agent::AgentConfig::default();
            let (tx, rx) = tokio::sync::oneshot::channel();
            agent_manager
                .command_sender()
                .send(nexora_agent::agent_manager::ManagerCommand::SpawnAgent {
                    agent_type: agent_type.to_string(),
                    config: cfg,
                    response_tx: tx,
                })
                .await
                .map_err(|e| {
                    NexoraError::system(format!("Failed to spawn {} agent: {}", agent_type, e))
                })?;
            match rx.await {
                Ok(Ok(agent_id)) => {
                    spawned.push((agent_type, agent_id));
                }
                Ok(Err(e)) => {
                    info!("Agent {} could not spawn: {} (non-fatal)", agent_type, e);
                }
                Err(_) => {
                    info!(
                        "Agent {} spawn response channel closed (non-fatal)",
                        agent_type
                    );
                }
            }
        }
        info!("Agent system active: {}/7 agents spawned", spawned.len());

        Ok(Self {
            registry: registry.clone(),
            active_model_id,
            inference_engine,
            config,
            start_time: Utc::now(),
            system_info_cache,
            request_count: request_count.clone(),
            system_monitor,
            request_processor: RequestProcessor::new(
                request_count.clone(),
                registry.clone(),
                active_model_id,
            ),
            agent_manager,
            train_metrics,
        })
    }

    fn model_id(&self) -> NxrModelId {
        self.active_model_id
    }

    /// Get system information
    pub async fn get_system_info(&self) -> NexoraResult<SystemInfo> {
        self.system_monitor
            .get_system_info()
            .await
            .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))
    }

    /// Health check
    pub async fn health_check(&self) -> NexoraResult<HealthStatus> {
        self.system_monitor
            .health_check()
            .await
            .map_err(|e| NexoraError::system(format!("Health check failed: {}", e)))
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> NexoraResult<serde_json::Value> {
        let system_info = self.get_system_info().await?;
        Ok(serde_json::json!({
            "cpu_usage": system_info.cpu_usage,
            "memory_usage": system_info.memory_usage,
            "uptime": system_info.uptime,
            "active_models": system_info.active_models.len(),
            "timestamp": system_info.last_updated,
            "request_count": self.request_count.load(Ordering::Relaxed),
            "uptime_seconds": (Utc::now() - self.start_time).num_seconds(),
            "active_model": format!("{}", self.active_model_id),
        }))
    }

    /// Process a request
    pub async fn process_request(&self, input: &str) -> NexoraResult<String> {
        if input.trim().is_empty() {
            return Err(NexoraError::validation("input", "Input cannot be empty"));
        }

        info!("Processing request: {} chars", input.len());
        self.request_processor
            .process_request(input)
            .await
            .map_err(|e| NexoraError::processing(format!("Request processing failed: {}", e)))
    }

    /// Generate text via streaming inference engine, returns a Receiver of tokens
    pub async fn generate_text_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> NexoraResult<tokio::sync::mpsc::Receiver<Arc<nexora_inference::GeneratedToken>>> {
        if prompt.trim().is_empty() {
            return Err(NexoraError::validation("prompt", "Prompt cannot be empty"));
        }
        if max_tokens == 0 {
            return Err(NexoraError::validation(
                "max_tokens",
                "Max tokens must be greater than 0",
            ));
        }
        if !(0.0..=2.0).contains(&temperature) {
            return Err(NexoraError::validation(
                "temperature",
                "Temperature must be between 0.0 and 2.0",
            ));
        }

        info!(
            "Streaming text via inference engine ({} model): prompt={}, max_tokens={}, temperature={}",
            self.active_model_id,
            prompt.len(),
            max_tokens,
            temperature
        );

        let request = nexora_inference::InferenceRequest {
            request_id: uuid::Uuid::new_v4(),
            model_id: self.active_model_id.to_string(),
            prompt: prompt.to_string(),
            max_tokens: max_tokens as u32,
            temperature,
            top_k: 50,
            top_p: 1.0,
            streaming: true,
            ..Default::default()
        };

        self.inference_engine
            .submit_streaming_request(request)
            .await
            .map_err(|e| NexoraError::model(format!("Streaming inference failed: {}", e)))
    }

    /// Generate text using foundation model inference
    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> NexoraResult<String> {
        if prompt.trim().is_empty() {
            return Err(NexoraError::validation("prompt", "Prompt cannot be empty"));
        }
        if max_tokens == 0 {
            return Err(NexoraError::validation(
                "max_tokens",
                "Max tokens must be greater than 0",
            ));
        }
        if !(0.0..=2.0).contains(&temperature) {
            return Err(NexoraError::validation(
                "temperature",
                "Temperature must be between 0.0 and 2.0",
            ));
        }

        info!(
            "Generating text via inference engine ({} model): prompt={}, max_tokens={}, temperature={}",
            self.active_model_id,
            prompt.len(),
            max_tokens,
            temperature
        );

        let request = nexora_inference::InferenceRequest {
            request_id: uuid::Uuid::new_v4(),
            model_id: self.active_model_id.to_string(),
            prompt: prompt.to_string(),
            max_tokens: max_tokens as u32,
            temperature,
            top_k: 50,
            top_p: 1.0,
            streaming: false,
            ..Default::default()
        };

        let response = self
            .inference_engine
            .generate_internal(request)
            .await
            .map_err(|e| NexoraError::model(format!("Inference failed: {}", e)))?;

        Ok(response.text)
    }

    /// Chat conversation using inference engine
    pub async fn chat(
        &self,
        message: &str,
        conversation_id: Option<String>,
    ) -> NexoraResult<String> {
        if message.trim().is_empty() {
            return Err(NexoraError::validation(
                "message",
                "Message cannot be empty",
            ));
        }

        info!(
            "Chat via inference engine ({} model): {} chars, conversation_id: {:?}",
            self.active_model_id,
            message.len(),
            conversation_id
        );

        let request = nexora_inference::InferenceRequest {
            request_id: uuid::Uuid::new_v4(),
            model_id: self.active_model_id.to_string(),
            prompt: message.to_string(),
            max_tokens: 1024,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            streaming: false,
            ..Default::default()
        };

        let response = self
            .inference_engine
            .generate_internal(request)
            .await
            .map_err(|e| NexoraError::model(format!("Chat inference failed: {}", e)))?;

        Ok(response.text)
    }

    /// Analyze code using inference engine
    pub async fn analyze_code(&self, code: &str, language: &str) -> NexoraResult<String> {
        if code.trim().is_empty() {
            return Err(NexoraError::validation("code", "Code cannot be empty"));
        }
        if language.trim().is_empty() {
            return Err(NexoraError::validation(
                "language",
                "Language cannot be empty",
            ));
        }

        let prompt = format!(
            "Analyze this {} code and return its structure:\n```{}\n{}\n```",
            language, language, code
        );

        info!(
            "Analyzing {} code ({} chars, prompt={}) via inference engine",
            language,
            code.len(),
            prompt.len()
        );

        self.generate_text(&prompt, 2048, 0.3).await
    }

    /// Generate code using inference engine
    pub async fn generate_code(&self, description: &str, language: &str) -> NexoraResult<String> {
        if description.trim().is_empty() {
            return Err(NexoraError::validation(
                "description",
                "Description cannot be empty",
            ));
        }
        if language.trim().is_empty() {
            return Err(NexoraError::validation(
                "language",
                "Language cannot be empty",
            ));
        }

        let prompt = format!(
            "Generate {} code. Task: {}\nReturn ONLY the code with no explanation.",
            language, description
        );

        info!(
            "Generating {} code via inference engine ({} model): prompt={}",
            language,
            self.active_model_id,
            prompt.len()
        );

        self.generate_text(&prompt, 2048, 0.2).await
    }

    /// Get training model information from registry
    pub async fn get_training_model(&self) -> NexoraResult<String> {
        info!("Retrieving training model information from registry...");

        let models = self.registry.list_models().await;
        if models.is_empty() {
            return Err(NexoraError::not_found("No training models configured"));
        }

        let model_name = format!("{} ({})", models[0], models[0].fullname());
        info!("Found training model: {}", model_name);
        Ok(model_name)
    }

    /// Get agent manager for agent orchestration
    pub fn agent_manager(&self) -> &Arc<nexora_agent::AgentManager> {
        &self.agent_manager
    }
}
