//! Agent Manager
//!
//! Supervisor untuk semua agent dalam sistem Nexora.
//! Bertanggung jawab untuk spawn, stop, dan monitoring agent.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc as StdArc;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::communication::MessageBus;
use crate::inference_agent::InferenceEngine;
use crate::lifecycle::LifecycleManager;
use crate::registry::AgentRegistry;
use crate::state::AgentState;
use crate::{
    Agent, AgentConfig, AgentError, AgentMessage, AgentResponse, AgentStats, AgentStatus, Result,
};

/// Konfigurasi untuk AgentManager
#[derive(Debug, Clone)]
pub struct AgentManagerConfig {
    /// Maximum concurrent agents
    pub max_concurrent_agents: usize,
    /// Default timeout untuk agent operations (dalam seconds)
    pub default_timeout_seconds: u64,
    /// Health check interval (dalam seconds)
    pub health_check_interval_seconds: u64,
    /// Enable auto-restart untuk failed agents
    pub auto_restart_failed_agents: bool,
    /// Maximum restart attempts
    pub max_restart_attempts: u32,
}

impl Default for AgentManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_agents: 100,
            default_timeout_seconds: 30,
            health_check_interval_seconds: 3600,
            auto_restart_failed_agents: true,
            max_restart_attempts: 3,
        }
    }
}

/// Manager untuk semua agent
pub struct AgentManager {
    /// Registry untuk tracking agent
    registry: StdArc<AgentRegistry>,
    /// Lifecycle manager
    lifecycle: StdArc<LifecycleManager>,
    /// Message bus untuk komunikasi
    message_bus: StdArc<MessageBus>,
    /// Shared state
    state: StdArc<AgentState>,
    /// Konfigurasi
    config: AgentManagerConfig,
    /// Channel untuk menerima command (bounded, buffer=256)
    command_rx: StdArc<RwLock<Option<mpsc::Receiver<ManagerCommand>>>>,
    /// Channel untuk mengirim command
    command_tx: StdArc<mpsc::Sender<ManagerCommand>>,
    /// Shared memory store singleton (not created per-call)
    memory_store: StdArc<std::sync::Mutex<nexora_memory::MemoryLayers>>,
    /// Health check loop cancellation flag
    is_running: StdArc<AtomicBool>,
    /// Tracked background task handles for cleanup
    background_handles: StdArc<std::sync::Mutex<Vec<JoinHandle<()>>>>,
    /// Global inference engine for agent inference
    inference_engine: StdArc<tokio::sync::RwLock<Option<StdArc<dyn InferenceEngine>>>>,
}

/// Command yang bisa dikirim ke AgentManager
#[derive(Debug)]
pub enum ManagerCommand {
    /// Spawn new agent
    SpawnAgent {
        agent_type: String,
        config: AgentConfig,
        response_tx: oneshot::Sender<Result<Uuid>>,
    },
    /// Stop agent
    StopAgent {
        agent_id: Uuid,
        response_tx: oneshot::Sender<Result<()>>,
    },
    /// Restart agent
    RestartAgent {
        agent_id: Uuid,
        response_tx: oneshot::Sender<Result<()>>,
    },
    /// Send message to agent
    SendMessage {
        agent_id: Uuid,
        message: AgentMessage,
        response_tx: oneshot::Sender<Result<AgentResponse>>,
    },
    /// Get agent status
    GetStatus {
        agent_id: Uuid,
        response_tx: oneshot::Sender<Result<AgentStatus>>,
    },
    /// Get agent stats
    GetStats {
        agent_id: Uuid,
        response_tx: oneshot::Sender<Result<AgentStats>>,
    },
    /// List all agents
    ListAgents {
        response_tx: oneshot::Sender<Result<Vec<(Uuid, String, AgentStatus)>>>,
    },
    /// Health check all agents
    HealthCheck {
        response_tx: oneshot::Sender<Result<HashMap<Uuid, bool>>>,
    },
    /// Dispatch plan steps to workers
    DispatchPlan {
        plan_id: Uuid,
        response_tx: oneshot::Sender<Result<()>>,
    },
    /// Get plan status from planner
    PlanStatus {
        plan_id: Uuid,
        response_tx: oneshot::Sender<Result<Value>>,
    },
    /// Get agent IDs grouped by type
    ListAgentIds {
        response_tx: oneshot::Sender<HashMap<String, Vec<Uuid>>>,
    },
    /// Shutdown manager
    Shutdown {
        response_tx: oneshot::Sender<Result<()>>,
    },
}

impl AgentManager {
    /// Create new agent manager
    pub fn new(config: AgentManagerConfig) -> Self {
        let (command_tx, command_rx) = mpsc::channel(256);
        let memory_store = StdArc::new(std::sync::Mutex::new(nexora_memory::MemoryLayers::new()));

        Self {
            registry: StdArc::new(AgentRegistry::new()),
            lifecycle: StdArc::new(LifecycleManager::new(config.clone())),
            message_bus: StdArc::new(MessageBus::new()),
            state: StdArc::new(AgentState::new().with_memory_store(
                StdArc::new(tokio::sync::Mutex::new(nexora_memory::MemoryLayers::new())),
            )),
            config,
            background_handles: StdArc::new(std::sync::Mutex::new(Vec::new())),
            command_rx: StdArc::new(RwLock::new(Some(command_rx))),
            command_tx: StdArc::new(command_tx),
            memory_store,
            is_running: StdArc::new(AtomicBool::new(true)),
            inference_engine: StdArc::new(tokio::sync::RwLock::new(None)),
        }
    }

    /// Get command sender untuk external communication
    pub fn command_sender(&self) -> mpsc::Sender<ManagerCommand> {
        (*self.command_tx).clone()
    }

    /// Set inference engine for inference agent
    /// Must be called before spawning inference agents
    pub async fn set_inference_engine(&self, engine: StdArc<dyn InferenceEngine>) {
        let mut guard = self.inference_engine.write().await;
        *guard = Some(engine);
    }

    /// Start agent manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting AgentManager with config: {:?}", self.config);

        // Restore state from memory
        if let Ok((sessions, agents)) = self.state.restore_from_memory().await {
            if sessions > 0 || agents > 0 {
                info!("Restored {} sessions and {} agents from memory", sessions, agents);
            }
        }

        // Start background tasks
        let manager = self.clone();
        let handle1 = tokio::spawn(async move {
            let fut = std::panic::AssertUnwindSafe(manager.run_command_loop());
            if let Err(e) = futures::future::FutureExt::catch_unwind(fut).await {
                error!("AgentManager command loop panicked: {:?}", e);
            }
        });
        if let Ok(mut handles) = self.background_handles.lock() {
            handles.push(handle1);
        }

        // Start health check loop
        if self.config.health_check_interval_seconds > 0 {
            let manager = self.clone();
            let handle2 = tokio::spawn(async move {
                let fut = std::panic::AssertUnwindSafe(manager.run_health_check_loop());
                if let Err(e) = futures::future::FutureExt::catch_unwind(fut).await {
                    error!("AgentManager health check loop panicked: {:?}", e);
                }
            });
            if let Ok(mut handles) = self.background_handles.lock() {
                handles.push(handle2);
            }
        }

        info!("AgentManager started successfully");
        Ok(())
    }

    /// Main command processing loop
    async fn run_command_loop(&self) {
        info!("Starting command loop");

        let mut rx_guard = self.command_rx.write().await;
        if let Some(mut rx) = rx_guard.take() {
            while let Some(command) = rx.recv().await {
                debug!("Received command: {:?}", std::mem::discriminant(&command));

                match command {
                    ManagerCommand::SpawnAgent {
                        agent_type,
                        config,
                        response_tx,
                    } => {
                        let result = self.spawn_agent_internal(agent_type, config).await;
                        if response_tx.send(result).is_err() {
                            warn!("SpawnAgent response channel closed");
                        }
                    }
                    ManagerCommand::StopAgent {
                        agent_id,
                        response_tx,
                    } => {
                        let result = self.stop_agent_internal(agent_id).await;
                        if response_tx.send(result).is_err() {
                            warn!("StopAgent response channel closed");
                        }
                    }
                    ManagerCommand::RestartAgent {
                        agent_id,
                        response_tx,
                    } => {
                        let result = self.restart_agent_internal(agent_id).await;
                        if response_tx.send(result).is_err() {
                            warn!("RestartAgent response channel closed");
                        }
                    }
                    ManagerCommand::SendMessage {
                        agent_id,
                        message,
                        response_tx,
                    } => {
                        let result = self.send_message_internal(agent_id, message).await;
                        if response_tx.send(result).is_err() {
                            warn!("SendMessage response channel closed");
                        }
                    }
                    ManagerCommand::GetStatus {
                        agent_id,
                        response_tx,
                    } => {
                        let result = self.get_status_internal(agent_id).await;
                        if response_tx.send(result).is_err() {
                            warn!("GetStatus response channel closed");
                        }
                    }
                    ManagerCommand::GetStats {
                        agent_id,
                        response_tx,
                    } => {
                        let result = self.get_stats_internal(agent_id).await;
                        if response_tx.send(result).is_err() {
                            warn!("GetStats response channel closed");
                        }
                    }
                    ManagerCommand::ListAgents { response_tx } => {
                        let result = self.list_agents_internal().await;
                        if response_tx.send(result).is_err() {
                            warn!("ListAgents response channel closed");
                        }
                    }
                    ManagerCommand::HealthCheck { response_tx } => {
                        let result = self.health_check_all_internal().await;
                        if response_tx.send(result).is_err() {
                            warn!("HealthCheck response channel closed");
                        }
                    }
                    ManagerCommand::DispatchPlan { plan_id, response_tx } => {
                        let result = self.dispatch_plan_internal(plan_id).await;
                        if response_tx.send(result).is_err() {
                            warn!("DispatchPlan response channel closed");
                        }
                    }
                    ManagerCommand::PlanStatus { plan_id, response_tx } => {
                        let result = self.plan_status_internal(plan_id).await;
                        if response_tx.send(result).is_err() {
                            warn!("PlanStatus response channel closed");
                        }
                    }
                    ManagerCommand::ListAgentIds { response_tx } => {
                        let result = self.list_agent_ids_internal().await;
                        if response_tx.send(result).is_err() {
                            warn!("ListAgentIds response channel closed");
                        }
                    }
                    ManagerCommand::Shutdown { response_tx } => {
                        let result = self.shutdown_internal().await;
                        if response_tx.send(result).is_err() {
                            warn!("Shutdown response channel closed");
                        }
                        break;
                    }
                }
            }
        }

        info!("Command loop ended");
    }

    /// Health check loop (cancellable via is_running flag)
    async fn run_health_check_loop(&self) {
        info!("Starting health check loop");

        let interval = tokio::time::Duration::from_secs(self.config.health_check_interval_seconds);
        let mut interval_timer = tokio::time::interval(interval);

        while self.is_running.load(Ordering::Relaxed) {
            interval_timer.tick().await;

            if !self.is_running.load(Ordering::Relaxed) {
                break;
            }

            if let Err(e) = self.perform_health_check().await {
                error!("Health check failed: {}", e);
            }
        }

        info!("Health check loop ended");
    }

    /// Internal spawn agent implementation
    async fn spawn_agent_internal(&self, agent_type: String, config: AgentConfig) -> Result<Uuid> {
        info!("Spawning agent of type: {}", agent_type);

        // Check concurrent agent limit
        if self.registry.agent_count().await >= self.config.max_concurrent_agents {
            return Err(AgentError::ProcessingError {
                operation: "spawn_agent".to_string(),
                reason: format!(
                    "Maximum concurrent agents ({}) reached",
                    self.config.max_concurrent_agents
                ),
            });
        }

        // Create agent instance based on type
        let mut agent = self.create_agent_instance(&agent_type).await?;
        let agent_id = agent.id();

        // Initialize agent before wrapping
        agent.initialize(config.clone()).await?;

        // Register agent (wraps in Arc<Mutex> internally)
        self.registry
            .register_agent(agent_id, agent_type.clone(), agent)
            .await?;

        // Start agent lifecycle
        self.lifecycle.start_agent(agent_id).await?;

        info!(
            "Agent {} (type: {}) spawned successfully",
            agent_id, agent_type
        );
        Ok(agent_id)
    }

    /// Internal stop agent implementation
    async fn stop_agent_internal(&self, agent_id: Uuid) -> Result<()> {
        info!("Stopping agent: {}", agent_id);

        // Stop lifecycle
        self.lifecycle.stop_agent(agent_id).await?;

        // Unregister agent
        self.registry.unregister_agent(agent_id).await?;

        info!("Agent {} stopped successfully", agent_id);
        Ok(())
    }

    /// Internal restart agent implementation
    async fn restart_agent_internal(&self, agent_id: Uuid) -> Result<()> {
        info!("Restarting agent: {}", agent_id);

        // Get agent info before stopping
        let agent_info = self
            .registry
            .get_agent_info(agent_id)
            .await?
            .ok_or_else(|| AgentError::AgentNotFound {
                agent_id: agent_id.to_string(),
            })?;

        // Stop agent
        self.stop_agent_internal(agent_id).await?;

        // Spawn new agent with same config
        self.spawn_agent_internal(agent_info.agent_type, agent_info.config)
            .await?;

        info!("Agent {} restarted successfully", agent_id);
        Ok(())
    }

    /// Internal send message implementation
    async fn send_message_internal(
        &self,
        agent_id: Uuid,
        message: AgentMessage,
    ) -> Result<AgentResponse> {
        debug!("Sending message to agent: {}", agent_id);

        // Step 1: Receive message
        {
            let mut agent = self.registry.get_agent(agent_id).await?;
            agent.receive(message.clone()).await?;
        }

        // Create context and bridge message payload into parameters
        let mut context = crate::AgentContext::new(Uuid::new_v4());
        context.parameters.insert(
            "action".to_string(),
            serde_json::json!(message.message_type),
        );
        if let serde_json::Value::Object(map) = &message.payload {
            for (k, v) in map {
                context.parameters.insert(k.clone(), v.clone());
            }
        } else {
            context.parameters.insert("payload".to_string(), message.payload.clone());
        }

        // Step 2: Process message
        let response = {
            let mut agent = self.registry.get_agent(agent_id).await?;
            agent.process(context).await?
        };

        // Step 3: Send response
        {
            let mut agent = self.registry.get_agent(agent_id).await?;
            agent.respond(response.clone()).await?;
        }

        debug!("Message processed successfully for agent: {}", agent_id);
        Ok(response)
    }

    /// Internal get status implementation
    async fn get_status_internal(&self, agent_id: Uuid) -> Result<AgentStatus> {
        let agent = self.registry.get_agent(agent_id).await?;
        Ok(agent.status())
    }

    /// Internal get stats implementation
    async fn get_stats_internal(&self, agent_id: Uuid) -> Result<AgentStats> {
        let agent = self.registry.get_agent(agent_id).await?;
        Ok(agent.get_stats())
    }

    /// Internal list agents implementation
    async fn list_agents_internal(&self) -> Result<Vec<(Uuid, String, AgentStatus)>> {
        self.registry.list_agents().await
    }

    /// Internal health check all implementation
    async fn health_check_all_internal(&self) -> Result<HashMap<Uuid, bool>> {
        let agents = self.registry.list_agents().await?;
        let mut results = HashMap::new();

        for (agent_id, _, _) in agents {
            let agent = self.registry.get_agent(agent_id).await?;
            let healthy = agent.health_check().await.unwrap_or(false);
            results.insert(agent_id, healthy);
        }

        Ok(results)
    }

    /// Dispatch plan steps from planner to workers
    async fn dispatch_plan_internal(&self, plan_id: Uuid) -> Result<()> {
        info!("Dispatching plan {} to workers", plan_id);

        let planner_ids = self.registry.get_agents_by_type("planner").await?;
        let planner_id = planner_ids.first().ok_or_else(|| AgentError::ProcessingError {
            operation: "dispatch_plan".to_string(),
            reason: "No planner agent available".to_string(),
        })?;

        let worker_ids = self.registry.get_agents_by_type("worker").await?;
        if worker_ids.is_empty() {
            return Err(AgentError::ProcessingError {
                operation: "dispatch_plan".to_string(),
                reason: "No worker agents available".to_string(),
            });
        }

        let plan_id_str = plan_id.to_string();
        let mut dispatched = 0usize;
        let mut has_failure = false;

        loop {
            // Get plan from planner (refreshed each iteration)
            let get_msg = crate::AgentMessage::new(
                "get_plan",
                serde_json::json!({"plan_id": plan_id_str.clone()}),
            );
            let plan_resp = self.send_message_internal(*planner_id, get_msg).await?;

            let plan_status = plan_resp.payload.get("plan")
                .and_then(|p| p.get("status"))
                .and_then(|s| s.as_str())
                .unwrap_or("");
            if plan_status == "Completed" || plan_status == "Failed" {
                info!("Plan {} is already {} ({} steps dispatched)", plan_id, plan_status, dispatched);
                return Ok(());
            }

            let plan_steps = plan_resp.payload.get("plan")
                .and_then(|p| p.get("steps"))
                .and_then(|s| s.as_array())
                .ok_or_else(|| AgentError::ProcessingError {
                    operation: "dispatch_plan".to_string(),
                    reason: "Invalid plan response from planner".to_string(),
                })?;

            let pending_steps: Vec<(usize, &Value)> = plan_steps.iter()
                .enumerate()
                .filter(|(_, s)| {
                    s.get("status").and_then(|v| v.as_str()) == Some("Pending")
                })
                .collect();

            if pending_steps.is_empty() {
                // All done — check completion via execute_next_step
                let exec_msg = crate::AgentMessage::new(
                    "execute_step",
                    serde_json::json!({"plan_id": plan_id_str.clone()}),
                );
                let _ = self.send_message_internal(*planner_id, exec_msg).await;
                break;
            }

            for (i, step) in &pending_steps {
                let step_id = step.get("step_id")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| AgentError::ProcessingError {
                        operation: "dispatch_plan".to_string(),
                        reason: "Step missing step_id".to_string(),
                    })?;

                let description = step.get("description")
                    .and_then(|s| s.as_str())
                    .unwrap_or("Execute step");
                let step_type = step.get("step_type")
                    .and_then(|s| s.as_str())
                    .unwrap_or("Processing");

                // Round-robin worker selection
                let worker_id = worker_ids[i % worker_ids.len()];

                let exec_msg = crate::AgentMessage::new(
                    "execute_step",
                    serde_json::json!({
                        "plan_id": plan_id_str.clone(),
                        "step_id": step_id,
                        "description": description,
                        "step_type": step_type,
                    }),
                );

                let exec_resp = self.send_message_internal(worker_id, exec_msg).await;
                match exec_resp {
                    Ok(resp) => {
                        let success = resp.payload.get("success").and_then(|v| v.as_bool()).unwrap_or(false);

                        // Notify planner about step completion
                        let complete_msg = crate::AgentMessage::new(
                            "complete_step",
                            serde_json::json!({
                                "plan_id": plan_id_str.clone(),
                                "step_id": step_id,
                                "result": resp.payload.get("output").cloned().unwrap_or(serde_json::json!(null)),
                            }),
                        );
                        let _ = self.send_message_internal(*planner_id, complete_msg).await;

                        dispatched += 1;
                        info!("Step {} done via worker {} (success={})", step_id, worker_id, success);

                        if !success {
                            has_failure = true;
                            warn!("Step {} failed, attempting replan", step_id);
                            // Replan: mark step as failed and let loop re-fetch
                            let fail_msg = crate::AgentMessage::new(
                                "fail_step",
                                serde_json::json!({
                                    "plan_id": plan_id_str.clone(),
                                    "step_id": step_id,
                                    "error": resp.payload.get("error").cloned().unwrap_or(json!("Unknown error")),
                                }),
                            );
                            let _ = self.send_message_internal(*planner_id, fail_msg).await;
                        }
                    }
                    Err(e) => {
                        has_failure = true;
                        warn!("Worker dispatch failed for step {}: {}", step_id, e);
                        let fail_msg = crate::AgentMessage::new(
                            "fail_step",
                            serde_json::json!({
                                "plan_id": plan_id_str.clone(),
                                "step_id": step_id,
                                "error": json!(e.to_string()),
                            }),
                        );
                        let _ = self.send_message_internal(*planner_id, fail_msg).await;
                    }
                }
            }
        }

        if has_failure {
            warn!("Plan {} completed with some step failures", plan_id);
        } else {
            info!("Plan {} completed successfully ({} steps)", plan_id, dispatched);
        }
        Ok(())
    }

    /// Get plan status from planner
    async fn plan_status_internal(&self, plan_id: Uuid) -> Result<Value> {
        let planner_ids = self.registry.get_agents_by_type("planner").await?;
        let planner_id = planner_ids.first().ok_or_else(|| AgentError::ProcessingError {
            operation: "plan_status".to_string(),
            reason: "No planner agent available".to_string(),
        })?;

        let msg = crate::AgentMessage::new(
            "get_plan",
            serde_json::json!({"plan_id": plan_id.to_string()}),
        );
        let resp = self.send_message_internal(*planner_id, msg).await?;
        Ok(resp.payload)
    }

    /// List agent IDs grouped by type
    async fn list_agent_ids_internal(&self) -> HashMap<String, Vec<Uuid>> {
        let mut grouped = HashMap::<String, Vec<Uuid>>::new();
        if let Ok(agents) = self.registry.list_agents().await {
            for (id, agent_type, _) in agents {
                grouped.entry(agent_type).or_default().push(id);
            }
        }
        grouped
    }

    /// Internal shutdown implementation
    async fn shutdown_internal(&self) -> Result<()> {
        info!("Shutting down AgentManager");

        // Signal health check loop to stop
        self.is_running.store(false, Ordering::Relaxed);

        // Save state to memory before shutdown
        if let Err(e) = self.state.snapshot_to_memory().await {
            warn!("Failed to snapshot state: {}", e);
        }

        // Stop all agents
        let agents = self.registry.list_agents().await?;
        for (agent_id, _, _) in agents {
            if let Err(e) = self.stop_agent_internal(agent_id).await {
                warn!("Failed to stop agent {}: {}", agent_id, e);
            }
        }

        info!("AgentManager shutdown complete");
        Ok(())
    }

    /// Perform health check
    async fn perform_health_check(&self) -> Result<()> {
        debug!("Performing health check");

        let health_results = self.health_check_all_internal().await?;

        for (agent_id, healthy) in health_results {
            if !healthy {
                warn!("Agent {} failed health check", agent_id);

                if self.config.auto_restart_failed_agents {
                    info!("Attempting to restart failed agent: {}", agent_id);
                    if let Err(e) = self.restart_agent_internal(agent_id).await {
                        error!("Failed to restart agent {}: {}", agent_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Create agent instance based on type
    async fn create_agent_instance(&self, agent_type: &str) -> Result<Box<dyn Agent>> {
        match agent_type {
            "context" => Ok(Box::new(crate::context_agent::ContextAgent::new(
                StdArc::new(nexora_memory::MemoryLayers::new()),
                crate::context_agent::ContextAgentConfig::default(),
            ))),
            "routing" => Ok(Box::new(crate::routing_agent::RoutingAgent::new(
                StdArc::new(HashMap::new()),
                crate::routing_agent::RoutingAgentConfig::default(),
            ))),
            "inference" => {
                let mut agent = crate::inference_agent::InferenceAgent::new(
                    crate::inference_agent::InferenceAgentConfig::default(),
                );
                if let Ok(guard) = self.inference_engine.try_read() {
                    if let Some(engine) = &*guard {
                        agent.set_inference_engine(engine.clone());
                    }
                }
                Ok(Box::new(agent))
            }
            "memory" => Ok(Box::new(crate::memory_agent::MemoryAgent::new(
                StdArc::new(tokio::sync::RwLock::new(nexora_memory::MemoryLayers::new())),
                crate::memory_agent::MemoryAgentConfig::default(),
            ))),
            "planner" => {
                let store = StdArc::new(tokio::sync::Mutex::new(nexora_memory::MemoryLayers::new()));
                Ok(Box::new(
                    crate::planner_agent::PlannerAgent::new(
                        crate::planner_agent::PlannerAgentConfig::default(),
                    )
                    .with_memory_store(store),
                ))
            }
            "response" => Ok(Box::new(crate::response_agent::ResponseAgent::new(
                crate::response_agent::ResponseAgentConfig::default(),
            ))),
            "validation" => Ok(Box::new(crate::validation_agent::ValidationAgent::new(
                crate::validation_agent::ValidationAgentConfig::default(),
            ))),
            "worker" => {
                let store = StdArc::new(tokio::sync::Mutex::new(
                    nexora_memory::MemoryLayers::new(),
                ));
                let mut agent = crate::worker_agent::WorkerAgent::new(
                    crate::worker_agent::WorkerAgentConfig::default(),
                )
                .with_memory_store(store);
                if let Ok(guard) = self.inference_engine.try_read() {
                    if let Some(engine) = &*guard {
                        agent = agent.with_inference_engine(engine.clone());
                    }
                }
                Ok(Box::new(agent))
            }
            _ => Err(AgentError::ProcessingError {
                operation: "create_agent".to_string(),
                reason: format!("Unknown agent type: {}", agent_type),
            }),
        }
    }

    /// Get memory store singleton
    fn get_memory_store(&self) -> StdArc<std::sync::Mutex<nexora_memory::MemoryLayers>> {
        self.memory_store.clone()
    }
}

impl Clone for AgentManager {
    fn clone(&self) -> Self {
        Self {
            registry: StdArc::clone(&self.registry),
            lifecycle: StdArc::clone(&self.lifecycle),
            message_bus: StdArc::clone(&self.message_bus),
            state: StdArc::clone(&self.state),
            config: self.config.clone(),
            command_rx: StdArc::clone(&self.command_rx),
            command_tx: StdArc::clone(&self.command_tx),
            memory_store: StdArc::clone(&self.memory_store),
            is_running: StdArc::clone(&self.is_running),
            inference_engine: StdArc::clone(&self.inference_engine),
            background_handles: StdArc::clone(&self.background_handles),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_manager_config_default() {
        let config = AgentManagerConfig::default();
        assert_eq!(config.max_concurrent_agents, 100);
        assert_eq!(config.default_timeout_seconds, 30);
        assert_eq!(config.health_check_interval_seconds, 3600);
        assert!(config.auto_restart_failed_agents);
        assert_eq!(config.max_restart_attempts, 3);
    }

    #[test]
    fn test_agent_manager_config_clone_debug() {
        let config = AgentManagerConfig::default();
        let cloned = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", cloned));
    }

    #[test]
    fn test_manager_command_spawn_agent() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::SpawnAgent {
            agent_type: "test".into(),
            config: AgentConfig::default(),
            response_tx: tx,
        };
        assert!(matches!(cmd, ManagerCommand::SpawnAgent { .. }));
    }

    #[test]
    fn test_manager_command_stop_agent() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::StopAgent {
            agent_id: Uuid::new_v4(),
            response_tx: tx,
        };
        assert!(matches!(cmd, ManagerCommand::StopAgent { .. }));
    }

    #[test]
    fn test_manager_command_send_message() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::SendMessage {
            agent_id: Uuid::new_v4(),
            message: AgentMessage::new("test", serde_json::json!({})),
            response_tx: tx,
        };
        assert!(matches!(cmd, ManagerCommand::SendMessage { .. }));
    }

    #[test]
    fn test_manager_command_get_status() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::GetStatus {
            agent_id: Uuid::new_v4(),
            response_tx: tx,
        };
        assert!(matches!(cmd, ManagerCommand::GetStatus { .. }));
    }

    #[test]
    fn test_manager_command_list_agents() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::ListAgents { response_tx: tx };
        assert!(matches!(cmd, ManagerCommand::ListAgents { .. }));
    }

    #[test]
    fn test_manager_command_health_check() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::HealthCheck { response_tx: tx };
        assert!(matches!(cmd, ManagerCommand::HealthCheck { .. }));
    }

    #[test]
    fn test_manager_command_shutdown() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let cmd = ManagerCommand::Shutdown { response_tx: tx };
        assert!(matches!(cmd, ManagerCommand::Shutdown { .. }));
    }

    #[tokio::test]
    async fn test_agent_manager_new() {
        let config = AgentManagerConfig::default();
        let manager = AgentManager::new(config.clone());
        assert_eq!(
            manager.config.max_concurrent_agents,
            config.max_concurrent_agents
        );
        let _sender = manager.command_sender();
    }

    #[tokio::test]
    async fn test_agent_manager_get_memory_store() {
        let manager = AgentManager::new(AgentManagerConfig::default());
        let store = manager.get_memory_store();
        // Memory store should be valid Arc<MemoryLayers>
        assert!(std::sync::Arc::strong_count(&store) >= 1);
    }

    #[tokio::test]
    async fn test_dispatch_plan_full_flow() {
        let manager = AgentManager::new(AgentManagerConfig {
            health_check_interval_seconds: 0,
            ..Default::default()
        });
        let cmd_tx = manager.command_sender();
        manager
            .start()
            .await
            .expect("AgentManager should start");

        // Spawn planner + 2 worker agents
        let agent_types = vec!["planner", "worker", "worker"];
        for agent_type in agent_types {
            let (tx, rx) = tokio::sync::oneshot::channel();
            cmd_tx
                .send(ManagerCommand::SpawnAgent {
                    agent_type: agent_type.to_string(),
                    config: AgentConfig::default(),
                    response_tx: tx,
                })
                .await
                .expect("SpawnAgent should send");
            let result = rx.await.expect("Spawn response should arrive");
            assert!(result.is_ok(), "Agent {} should spawn: {:?}", agent_type, result.err());
        }

        // Create plan via planner
        let (list_tx, list_rx) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ManagerCommand::ListAgentIds {
                response_tx: list_tx,
            })
            .await
            .expect("ListAgentIds should send");
        let grouped = list_rx.await.unwrap_or_default();
        let planner_id = *grouped
            .get("planner")
            .and_then(|v| v.first())
            .expect("Planner agent should exist");

        let create_msg = AgentMessage::new(
            "create_plan",
            serde_json::json!({"task": "Write a short hello world program in Rust and test it"}),
        );
        let (create_tx, create_rx) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ManagerCommand::SendMessage {
                agent_id: planner_id,
                message: create_msg,
                response_tx: create_tx,
            })
            .await
            .expect("create_plan should send");
        let create_resp = create_rx
            .await
            .expect("create_plan response should arrive")
            .expect("create_plan should succeed");
        let plan_id_str = create_resp
            .payload
            .get("plan_id")
            .and_then(|v| v.as_str())
            .expect("plan_id should be in response");
        let plan_id = Uuid::parse_str(plan_id_str).expect("plan_id should be valid UUID");
        let (dispatch_tx, dispatch_rx) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ManagerCommand::DispatchPlan {
                plan_id,
                response_tx: dispatch_tx,
            })
            .await
            .expect("DispatchPlan should send");
        let dispatch_result = dispatch_rx
            .await
            .expect("Dispatch response should arrive");
        assert!(dispatch_result.is_ok(), "Dispatch should succeed: {:?}", dispatch_result.err());

        // Check plan completed
        let (status_tx, status_rx) = tokio::sync::oneshot::channel();
        cmd_tx
            .send(ManagerCommand::PlanStatus {
                plan_id,
                response_tx: status_tx,
            })
            .await
            .expect("PlanStatus should send");
        let status_payload = status_rx
            .await
            .expect("PlanStatus response should arrive")
            .expect("PlanStatus should succeed");

        let plan_status = status_payload
            .get("plan")
            .and_then(|p| p.get("status"))
            .and_then(|v| v.as_str())
            .expect("plan status should be present");
        assert_eq!(
            plan_status, "Completed",
            "Plan should complete after dispatch, got: {}",
            plan_status
        );
    }
}
