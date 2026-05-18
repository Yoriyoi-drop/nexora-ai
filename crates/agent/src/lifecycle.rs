//! Lifecycle Manager
//! 
//! Mengelola startup, shutdown, dan restart agent.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use tracing::{info, debug};
use chrono::{DateTime, Utc};

use crate::{AgentError, Result, AgentStatus};
use crate::agent_manager::AgentManagerConfig;

/// Event lifecycle agent
#[derive(Debug, Clone)]
pub enum AgentLifecycleEvent {
    /// Agent mulai diinisialisasi
    Initializing { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent siap
    Ready { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent mulai memproses
    Processing { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent selesai memproses
    ProcessingComplete { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent di-pause
    Paused { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent resume
    Resumed { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent error
    Error { agent_id: Uuid, error: String, timestamp: DateTime<Utc> },
    /// Agent shutdown
    Shutdown { agent_id: Uuid, timestamp: DateTime<Utc> },
    /// Agent restart
    Restarted { agent_id: Uuid, timestamp: DateTime<Utc> },
}

/// Status detail untuk lifecycle tracking
#[derive(Debug, Clone)]
pub struct AgentLifecycleStatus {
    /// Agent ID
    pub agent_id: Uuid,
    /// Status saat ini
    pub status: AgentStatus,
    /// Waktu mulai
    pub started_at: DateTime<Utc>,
    /// Waktu last update
    pub last_updated: DateTime<Utc>,
    /// Jumlah restart
    pub restart_count: u32,
    /// Total processing time (milliseconds)
    pub total_processing_time_ms: u64,
    /// Error terakhir (jika ada)
    pub last_error: Option<String>,
}

/// Lifecycle manager untuk semua agent
pub struct LifecycleManager {
    /// Tracking status per agent
    agent_status: Arc<RwLock<HashMap<Uuid, AgentLifecycleStatus>>>,
    /// Event channel untuk lifecycle events
    event_tx: mpsc::UnboundedSender<AgentLifecycleEvent>,
    /// Event receiver
    _event_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<AgentLifecycleEvent>>>>,
    /// Event subscribers (bounded, buffer=32 per subscriber)
    event_subscribers: Arc<tokio::sync::Mutex<Vec<mpsc::UnboundedSender<AgentLifecycleEvent>>>>,
    /// Konfigurasi
    config: AgentManagerConfig,
}

impl LifecycleManager {
    /// Create new lifecycle manager
    pub fn new(config: AgentManagerConfig) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        
        Self {
            agent_status: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            _event_rx: Arc::new(RwLock::new(Some(event_rx))),
            event_subscribers: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            config,
        }
    }
    
    /// Emit lifecycle event
    async fn emit_event(&self, event: AgentLifecycleEvent) -> Result<()> {
        let _ = self.event_tx.send(event);
        Ok(())
    }
    
    /// Start agent lifecycle
    pub async fn start_agent(&self, agent_id: Uuid) -> Result<()> {
        info!("Starting lifecycle for agent: {}", agent_id);
        
        let now = Utc::now();
        let status = AgentLifecycleStatus {
            agent_id,
            status: AgentStatus::Initializing,
            started_at: now,
            last_updated: now,
            restart_count: 0,
            total_processing_time_ms: 0,
            last_error: None,
        };
        
        // Update status
        {
            let mut agent_status = self.agent_status.write().await;
            agent_status.insert(agent_id, status);
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Initializing { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        // Transition to ready
        self.transition_to_ready(agent_id).await?;
        
        Ok(())
    }
    
    /// Stop agent lifecycle
    pub async fn stop_agent(&self, agent_id: Uuid) -> Result<()> {
        info!("Stopping lifecycle for agent: {}", agent_id);
        
        let now = Utc::now();
        
        // Update status to shutting down
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::ShuttingDown;
                status.last_updated = now;
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Shutdown { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        // Final status update
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Shutdown;
                status.last_updated = now;
            }
        }
        
        Ok(())
    }
    
    /// Restart agent
    pub async fn restart_agent(&self, agent_id: Uuid) -> Result<()> {
        info!("Restarting agent: {}", agent_id);
        
        let now = Utc::now();
        
        // Check restart limit
        {
            let agent_status = self.agent_status.read().await;
            if let Some(status) = agent_status.get(&agent_id) {
                if status.restart_count >= self.config.max_restart_attempts {
                    return Err(AgentError::LifecycleError(
                        format!("Agent {} exceeded maximum restart attempts ({})", 
                                agent_id, self.config.max_restart_attempts)
                    ));
                }
            }
        }
        
        // Update restart count
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.restart_count += 1;
                status.last_error = None; // Clear error on restart
            }
        }
        
        // Emit restart event
        let event = AgentLifecycleEvent::Restarted { agent_id, timestamp: now };
        self.emit_event(event).await?;
        
        // Restart lifecycle
        self.start_agent(agent_id).await?;
        
        Ok(())
    }
    
    /// Pause agent
    pub async fn pause_agent(&self, agent_id: Uuid) -> Result<()> {
        info!("Pausing agent: {}", agent_id);
        
        let now = Utc::now();
        
        // Update status
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Paused;
                status.last_updated = now;
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Paused { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        Ok(())
    }
    
    /// Resume agent
    pub async fn resume_agent(&self, agent_id: Uuid) -> Result<()> {
        info!("Resuming agent: {}", agent_id);
        
        let now = Utc::now();
        
        // Update status to ready
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Ready;
                status.last_updated = now;
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Resumed { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        Ok(())
    }
    
    /// Mark agent as processing
    pub async fn mark_processing(&self, agent_id: Uuid) -> Result<()> {
        let now = Utc::now();
        
        // Update status
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Processing;
                status.last_updated = now;
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Processing { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        Ok(())
    }
    
    /// Mark agent processing complete
    pub async fn mark_processing_complete(&self, agent_id: Uuid, processing_time_ms: u64) -> Result<()> {
        let now = Utc::now();
        
        // Update status
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Ready;
                status.last_updated = now;
                status.total_processing_time_ms += processing_time_ms;
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::ProcessingComplete { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        Ok(())
    }
    
    /// Mark agent as error
    pub async fn mark_error(&self, agent_id: Uuid, error: String) -> Result<()> {
        let now = Utc::now();
        
        // Update status
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Error(error.clone());
                status.last_updated = now;
                status.last_error = Some(error.clone());
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Error { agent_id, error: error.clone(), timestamp: now };
        let _ = self.event_tx.send(event);
        
        Ok(())
    }
    
    /// Get agent lifecycle status
    pub async fn get_agent_status(&self, agent_id: Uuid) -> Result<Option<AgentLifecycleStatus>> {
        let agent_status = self.agent_status.read().await;
        Ok(agent_status.get(&agent_id).cloned())
    }
    
    /// Get all agent statuses
    pub async fn get_all_agent_statuses(&self) -> HashMap<Uuid, AgentLifecycleStatus> {
        let agent_status = self.agent_status.read().await;
        agent_status.clone()
    }
    
    /// Get agents by status
    pub async fn get_agents_by_status(&self, target_status: AgentStatus) -> Vec<Uuid> {
        let agent_status = self.agent_status.read().await;
        agent_status.iter()
            .filter(|(_, status)| status.status == target_status)
            .map(|(agent_id, _)| *agent_id)
            .collect()
    }
    
    /// Get event subscriber
    pub async fn get_event_subscriber(&self) -> Option<mpsc::UnboundedReceiver<AgentLifecycleEvent>> {
        // Implement proper subscription mechanism
        let (tx, rx) = mpsc::unbounded_channel();
        
        // Add subscriber to the list
        let mut subscribers = self.event_subscribers.lock().await;
        subscribers.push(tx);
        
        Some(rx)
    }
    
    /// Cleanup old agent statuses
    pub async fn cleanup_old_statuses(&self, max_age_hours: u64) -> Result<usize> {
        let now = Utc::now();
        let mut removed_count = 0;
        
        {
            let mut agent_status = self.agent_status.write().await;
            agent_status.retain(|agent_id, status| {
                let age_hours = (now - status.last_updated).num_hours() as u64;
                let should_keep = age_hours <= max_age_hours || 
                                matches!(status.status, AgentStatus::Ready | AgentStatus::Processing);
                
                if !should_keep {
                    debug!("Cleaning up old status for agent: {}", agent_id);
                    removed_count += 1;
                }
                
                should_keep
            });
        }
        
        Ok(removed_count)
    }
    
    /// Get lifecycle statistics
    pub async fn get_lifecycle_stats(&self) -> LifecycleStats {
        let agent_status = self.agent_status.read().await;
        let mut stats = LifecycleStats::default();
        
        for status in agent_status.values() {
            stats.total_agents += 1;
            
            match status.status {
                AgentStatus::Ready => stats.ready_agents += 1,
                AgentStatus::Processing => stats.processing_agents += 1,
                AgentStatus::Paused => stats.paused_agents += 1,
                AgentStatus::Error(_) => stats.error_agents += 1,
                AgentStatus::Shutdown => stats.shutdown_agents += 1,
                AgentStatus::ShuttingDown => stats.shutting_down_agents += 1,
                AgentStatus::Initializing => stats.initializing_agents += 1,
            }
            
            stats.total_processing_time_ms += status.total_processing_time_ms;
            stats.total_restarts += status.restart_count;
        }
        
        stats
    }
    
    /// Transition agent to ready state
    async fn transition_to_ready(&self, agent_id: Uuid) -> Result<()> {
        let now = Utc::now();
        
        // Update status
        {
            let mut agent_status = self.agent_status.write().await;
            if let Some(status) = agent_status.get_mut(&agent_id) {
                status.status = AgentStatus::Ready;
                status.last_updated = now;
            }
        }
        
        // Emit event
        let event = AgentLifecycleEvent::Ready { agent_id, timestamp: now };
        let _ = self.event_tx.send(event);
        
        Ok(())
    }
}

/// Lifecycle statistics
#[derive(Debug, Clone, Default)]
pub struct LifecycleStats {
    /// Total agents
    pub total_agents: usize,
    /// Agents yang ready
    pub ready_agents: usize,
    /// Agents yang sedang processing
    pub processing_agents: usize,
    /// Agents yang di-pause
    pub paused_agents: usize,
    /// Agents yang error
    pub error_agents: usize,
    /// Agents yang shutdown
    pub shutdown_agents: usize,
    /// Agents yang sedang shutdown
    pub shutting_down_agents: usize,
    /// Agents yang initializing
    pub initializing_agents: usize,
    /// Total processing time
    pub total_processing_time_ms: u64,
    /// Total restarts
    pub total_restarts: u32,
}
