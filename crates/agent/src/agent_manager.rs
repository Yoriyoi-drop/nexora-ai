//! Agent Manager
//! 
//! Supervisor untuk semua agent dalam sistem Nexora.
//! Bertanggung jawab untuk spawn, stop, dan monitoring agent.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, oneshot};
use uuid::Uuid;
use tracing::{info, warn, error, debug};

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentStats, AgentConfig
};
use crate::registry::AgentRegistry;
use crate::lifecycle::LifecycleManager;
use crate::communication::MessageBus;
use crate::state::AgentState;

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
            health_check_interval_seconds: 60,
            auto_restart_failed_agents: true,
            max_restart_attempts: 3,
        }
    }
}

/// Manager untuk semua agent
pub struct AgentManager {
    /// Registry untuk tracking agent
    registry: Arc<AgentRegistry>,
    /// Lifecycle manager
    lifecycle: Arc<LifecycleManager>,
    /// Message bus untuk komunikasi
    message_bus: Arc<MessageBus>,
    /// Shared state
    state: Arc<AgentState>,
    /// Konfigurasi
    config: AgentManagerConfig,
    /// Channel untuk menerima command
    command_rx: RwLock<Option<mpsc::UnboundedReceiver<ManagerCommand>>>,
    /// Channel untuk mengirim command
    command_tx: mpsc::UnboundedSender<ManagerCommand>,
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
    /// Shutdown manager
    Shutdown {
        response_tx: oneshot::Sender<Result<()>>,
    },
}

impl AgentManager {
    /// Create new agent manager
    pub fn new(config: AgentManagerConfig) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        
        Self {
            registry: Arc::new(AgentRegistry::new()),
            lifecycle: Arc::new(LifecycleManager::new(config.clone())),
            message_bus: Arc::new(MessageBus::new()),
            state: Arc::new(AgentState::new()),
            config,
            command_rx: RwLock::new(Some(command_rx)),
            command_tx,
        }
    }
    
    /// Get command sender untuk external communication
    pub fn command_sender(&self) -> mpsc::UnboundedSender<ManagerCommand> {
        self.command_tx.clone()
    }
    
    /// Start agent manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting AgentManager with config: {:?}", self.config);
        
        // Start background tasks
        let manager = self.clone();
        tokio::spawn(async move {
            manager.run_command_loop().await;
        });
        
        // Start health check loop
        if self.config.health_check_interval_seconds > 0 {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.run_health_check_loop().await;
            });
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
                ManagerCommand::SpawnAgent { agent_type, config, response_tx } => {
                    let result = self.spawn_agent_internal(agent_type, config).await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::StopAgent { agent_id, response_tx } => {
                    let result = self.stop_agent_internal(agent_id).await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::RestartAgent { agent_id, response_tx } => {
                    let result = self.restart_agent_internal(agent_id).await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::SendMessage { agent_id, message, response_tx } => {
                    let result = self.send_message_internal(agent_id, message).await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::GetStatus { agent_id, response_tx } => {
                    let result = self.get_status_internal(agent_id).await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::GetStats { agent_id, response_tx } => {
                    let result = self.get_stats_internal(agent_id).await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::ListAgents { response_tx } => {
                    let result = self.list_agents_internal().await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::HealthCheck { response_tx } => {
                    let result = self.health_check_all_internal().await;
                    let _ = response_tx.send(result);
                }
                ManagerCommand::Shutdown { response_tx } => {
                    let result = self.shutdown_internal().await;
                    let _ = response_tx.send(result);
                    break;
                }
            }
        }
        }
        
        info!("Command loop ended");
    }
    
    /// Health check loop
    async fn run_health_check_loop(&self) {
        info!("Starting health check loop");
        
        let interval = tokio::time::Duration::from_secs(self.config.health_check_interval_seconds);
        let mut interval_timer = tokio::time::interval(interval);
        
        loop {
            interval_timer.tick().await;
            
            if let Err(e) = self.perform_health_check().await {
                error!("Health check failed: {}", e);
            }
        }
    }
    
    /// Internal spawn agent implementation
    async fn spawn_agent_internal(&self, agent_type: String, config: AgentConfig) -> Result<Uuid> {
        info!("Spawning agent of type: {}", agent_type);
        
        // Check concurrent agent limit
        if self.registry.agent_count().await >= self.config.max_concurrent_agents {
            return Err(AgentError::ProcessingError(
                format!("Maximum concurrent agents ({}) reached", self.config.max_concurrent_agents)
            ));
        }
        
        // Create agent instance based on type
        let agent = self.create_agent_instance(&agent_type).await?;
        let agent_id = agent.id();
        
        // Initialize agent
        let mut agent_clone = agent;
        agent_clone.initialize(config.clone()).await?;
        
        // Register agent
        self.registry.register_agent(agent_id, agent_type.clone(), agent_clone).await?;
        
        // Start agent lifecycle
        self.lifecycle.start_agent(agent_id).await?;
        
        info!("Agent {} (type: {}) spawned successfully", agent_id, agent_type);
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
        let agent_info = self.registry.get_agent_info(agent_id).await?
            .ok_or_else(|| AgentError::AgentNotFound(agent_id.to_string()))?;
        
        // Stop agent
        self.stop_agent_internal(agent_id).await?;
        
        // Spawn new agent with same config
        self.spawn_agent_internal(agent_info.agent_type, agent_info.config).await?;
        
        info!("Agent {} restarted successfully", agent_id);
        Ok(())
    }
    
    /// Internal send message implementation
    async fn send_message_internal(&self, agent_id: Uuid, message: AgentMessage) -> Result<AgentResponse> {
        debug!("Sending message to agent: {}", agent_id);
        
        let agent = self.registry.get_agent(agent_id).await?
            .ok_or_else(|| AgentError::AgentNotFound(agent_id.to_string()))?;
        
        // Send message to agent
        let mut agent_clone = agent;
        agent_clone.receive(message).await?;
        
        // Create context
        let context = crate::AgentContext::new(Uuid::new_v4());
        
        // Process message
        let response = agent_clone.process(context).await?;
        
        // Send response
        agent_clone.respond(response.clone()).await?;
        
        debug!("Message processed successfully for agent: {}", agent_id);
        Ok(response)
    }
    
    /// Internal get status implementation
    async fn get_status_internal(&self, agent_id: Uuid) -> Result<AgentStatus> {
        let agent = self.registry.get_agent(agent_id).await?
            .ok_or_else(|| AgentError::AgentNotFound(agent_id.to_string()))?;
        
        Ok(agent.status())
    }
    
    /// Internal get stats implementation
    async fn get_stats_internal(&self, agent_id: Uuid) -> Result<AgentStats> {
        let agent = self.registry.get_agent(agent_id).await?
            .ok_or_else(|| AgentError::AgentNotFound(agent_id.to_string()))?;
        
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
            let agent = self.registry.get_agent(agent_id).await?
                .ok_or_else(|| AgentError::AgentNotFound(agent_id.to_string()))?;
            
            let healthy = agent.health_check().await.unwrap_or(false);
            results.insert(agent_id, healthy);
        }
        
        Ok(results)
    }
    
    /// Internal shutdown implementation
    async fn shutdown_internal(&self) -> Result<()> {
        info!("Shutting down AgentManager");
        
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
        // Implement agent factory based on type
        match agent_type {
            "context" => {
                // Create context agent
                let memory_store = self.get_memory_store().await?;
                let config = crate::context_agent::ContextAgentConfig::default();
                let agent = crate::context_agent::ContextAgent::new(memory_store, config);
                Ok(Box::new(agent))
            }
            "routing" => {
                // Create routing agent
                let specialist_models = Arc::new(HashMap::new());
                let config = crate::routing_agent::RoutingAgentConfig::default();
                let agent = crate::routing_agent::RoutingAgent::new(specialist_models, config);
                Ok(Box::new(agent))
            }
            "inference" => {
                // Create inference agent
                let config = crate::inference_agent::InferenceAgentConfig::default();
                let agent = crate::inference_agent::InferenceAgent::new(config);
                Ok(Box::new(agent))
            }
            "memory" => {
                // Create memory agent
                let memory_store = Arc::new(tokio::sync::RwLock::new(nexora_memory::MemoryLayers::new()));
                let config = crate::memory_agent::MemoryAgentConfig::default();
                let agent = crate::memory_agent::MemoryAgent::new(memory_store, config);
                Ok(Box::new(agent))
            }
            "planner" => {
                // Create planner agent
                let config = crate::planner_agent::PlannerAgentConfig::default();
                let agent = crate::planner_agent::PlannerAgent::new(config);
                Ok(Box::new(agent))
            }
            "response" => {
                // Create response agent
                let config = crate::response_agent::ResponseAgentConfig::default();
                let agent = crate::response_agent::ResponseAgent::new(config);
                Ok(Box::new(agent))
            }
            "validation" => {
                // Create validation agent
                let config = crate::validation_agent::ValidationAgentConfig::default();
                let agent = crate::validation_agent::ValidationAgent::new(config);
                Ok(Box::new(agent))
            }
            _ => Err(AgentError::ProcessingError(
                format!("Unknown agent type: {}", agent_type)
            ))
        }
    }
    
    /// Get memory store instance
    async fn get_memory_store(&self) -> Result<Arc<nexora_memory::MemoryLayers>> {
        // Create or get memory store instance
        // In a real implementation, this might use dependency injection
        let _memory_config = nexora_memory::lru_memory::MemoryConfig::default();
        let memory_store = Arc::new(nexora_memory::MemoryLayers::new());
        Ok(memory_store)
    }
}

// Implement Clone untuk AgentManager
impl Clone for AgentManager {
    fn clone(&self) -> Self {
        // Create new command channel for the cloned manager
        let (new_tx, new_rx) = tokio::sync::mpsc::unbounded_channel();
        
        Self {
            registry: Arc::clone(&self.registry),
            lifecycle: Arc::clone(&self.lifecycle),
            message_bus: Arc::clone(&self.message_bus),
            state: Arc::clone(&self.state),
            config: self.config.clone(),
            command_rx: RwLock::new(Some(new_rx)),
            command_tx: new_tx,
        }
    }
}
