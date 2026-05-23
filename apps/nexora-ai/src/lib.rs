//! Nexora-AI Main Library - Pure Rust Entry Point
//!
//! Main entry point for Nexora AI system - delegates to specialized crates

use crate::error::{NexoraError, NexoraResult};
use chrono::Utc;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

pub mod api;
pub mod cli;
pub mod config;
pub mod core;
pub mod error;
pub mod security;
pub mod server;

pub use api::{ApiConfig, ApiResponse, NexoraApi};
pub use cli::Cli;
pub use config::NexoraConfig;
pub use core::*;
pub use server::{NexoraServer, ServerConfig};

// --- Foundation model integration ---
use nexora_foundation::shared::{
    base_model::{InputData, NxrInput},
    model_identity::NxrModelId,
    model_registry::global_registry,
};
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

        // Step 1: Initialize foundation models — all 10 NXR models get real CausalLM instances
        nexora_foundation::init::initialize_foundation_models()
            .await
            .map_err(|e| {
                NexoraError::system(format!("Failed to initialize foundation models: {}", e))
            })?;

        // Step 1b: Initialize model-internal agents — all 10 NXR model agent systems go live
        nexora_foundation::model_agent_manager::init_model_agents().await;

        let registry = global_registry();

        let active_model_id = NxrModelId::Omnis;
        info!(
            "Active model: {} ({})",
            active_model_id,
            active_model_id.fullname()
        );

        // Step 2: Initialize multi-agent system — spawn all 7 runtime agents
        let agent_config = nexora_agent::agent_manager::AgentManagerConfig::default();
        let agent_manager = Arc::new(nexora_agent::AgentManager::new(agent_config));
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

        // Initialize system monitoring
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

        Ok(Self {
            registry,
            active_model_id,
            config,
            start_time: Utc::now(),
            system_info_cache,
            request_count: request_count.clone(),
            system_monitor,
            request_processor: RequestProcessor::new(request_count.clone()),
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
            "Generating text via {} model: prompt={}, max_tokens={}, temperature={}",
            self.active_model_id,
            prompt.len(),
            max_tokens,
            temperature
        );

        let model_id = self.model_id();
        let model =
            self.registry.get_model(&model_id).await.map_err(|e| {
                NexoraError::model(format!("Model {} not available: {}", model_id, e))
            })?;

        let input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(prompt.to_string()),
            parameters: [
                ("max_tokens".to_string(), serde_json::json!(max_tokens)),
                ("temperature".to_string(), serde_json::json!(temperature)),
            ]
            .into(),
            metadata: std::collections::HashMap::new(),
        };

        let output = model
            .infer(&input)
            .await
            .map_err(|e| NexoraError::model(format!("Inference failed: {}", e)))?;

        let result = match output.data {
            nexora_foundation::shared::base_model::OutputData::Text(text) => text,
            _ => format!("{:?}", output.data),
        };

        Ok(result)
    }

    /// Chat conversation using foundation model inference
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
            "Chat via {} model: {} chars, conversation_id: {:?}",
            self.active_model_id,
            message.len(),
            conversation_id
        );

        let model_id = self.model_id();
        let model =
            self.registry.get_model(&model_id).await.map_err(|e| {
                NexoraError::model(format!("Model {} not available: {}", model_id, e))
            })?;

        let mut metadata = std::collections::HashMap::new();
        if let Some(conv_id) = &conversation_id {
            metadata.insert("conversation_id".to_string(), serde_json::json!(conv_id));
        }

        let input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(message.to_string()),
            parameters: [("mode".to_string(), serde_json::json!("chat"))].into(),
            metadata,
        };

        let output = model
            .infer(&input)
            .await
            .map_err(|e| NexoraError::model(format!("Chat inference failed: {}", e)))?;

        let result = match output.data {
            nexora_foundation::shared::base_model::OutputData::Text(text) => text,
            _ => format!("{:?}", output.data),
        };

        Ok(result)
    }

    /// Analyze code using foundation model + ORACLE
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

        info!(
            "Analyzing {} code ({} chars) via foundation model",
            language,
            code.len()
        );

        let model_id = self.model_id();
        let model =
            self.registry.get_model(&model_id).await.map_err(|e| {
                NexoraError::model(format!("Model {} not available: {}", model_id, e))
            })?;

        let prompt = format!(
            "Analyze this {} code and return its structure:\n```{}\n{}\n```",
            language, language, code
        );

        let input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(prompt),
            parameters: [
                ("mode".to_string(), serde_json::json!("code_analysis")),
                ("language".to_string(), serde_json::json!(language)),
            ]
            .into(),
            metadata: std::collections::HashMap::new(),
        };

        let output = model
            .infer(&input)
            .await
            .map_err(|e| NexoraError::model(format!("Code analysis failed: {}", e)))?;

        let result = match output.data {
            nexora_foundation::shared::base_model::OutputData::Text(text) => text,
            _ => "Analysis complete".to_string(),
        };

        Ok(result)
    }

    /// Generate code using foundation model inference
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

        info!(
            "Generating {} code via {} model: {}",
            language, self.active_model_id, description
        );

        let model_id = self.model_id();
        let model =
            self.registry.get_model(&model_id).await.map_err(|e| {
                NexoraError::model(format!("Model {} not available: {}", model_id, e))
            })?;

        let prompt = format!(
            "Generate {} code. Task: {}\nReturn ONLY the code with no explanation.",
            language, description
        );

        let input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(prompt),
            parameters: [
                ("mode".to_string(), serde_json::json!("code_generation")),
                ("language".to_string(), serde_json::json!(language)),
            ]
            .into(),
            metadata: std::collections::HashMap::new(),
        };

        let output = model
            .infer(&input)
            .await
            .map_err(|e| NexoraError::model(format!("Code generation failed: {}", e)))?;

        let result = match output.data {
            nexora_foundation::shared::base_model::OutputData::Text(text) => text,
            _ => "// Code generation complete".to_string(),
        };

        Ok(result)
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
}
