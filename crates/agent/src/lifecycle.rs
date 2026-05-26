//! Lifecycle Manager
//!
//! Mengelola startup, shutdown, dan restart agent.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::agent_manager::AgentManagerConfig;
use crate::{AgentError, AgentStatus, Result};

/// Event lifecycle agent
#[derive(Debug, Clone)]
pub enum AgentLifecycleEvent {
    /// Agent mulai diinisialisasi
    Initializing {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent siap
    Ready {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent mulai memproses
    Processing {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent selesai memproses
    ProcessingComplete {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent di-pause
    Paused {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent resume
    Resumed {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent error
    Error {
        agent_id: Uuid,
        error: String,
        timestamp: DateTime<Utc>,
    },
    /// Agent shutdown
    Shutdown {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Agent restart
    Restarted {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
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
    /// Event subscribers (buffer=1024 per subscriber)
    event_subscribers: Arc<tokio::sync::Mutex<Vec<mpsc::Sender<AgentLifecycleEvent>>>>,
    /// Konfigurasi
    config: AgentManagerConfig,
}

impl LifecycleManager {
    /// Create new lifecycle manager
    pub fn new(config: AgentManagerConfig) -> Self {
        Self {
            agent_status: Arc::new(RwLock::new(HashMap::new())),
            event_subscribers: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            config,
        }
    }

    /// Emit lifecycle event to all subscribers
    async fn emit_event(&self, event: AgentLifecycleEvent) {
        let mut subscribers = self.event_subscribers.lock().await;
        let mut dead = Vec::new();
        for (i, subscriber) in subscribers.iter().enumerate() {
            match subscriber.try_send(event.clone()) {
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!("Lifecycle subscriber channel full — dropping event");
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    dead.push(i);
                }
                Ok(_) => {}
            }
        }
        for &i in dead.iter().rev() {
            subscribers.remove(i);
        }
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
        let event = AgentLifecycleEvent::Initializing {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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
        let event = AgentLifecycleEvent::Shutdown {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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
                    return Err(AgentError::LifecycleError { reason: format!(
                        "Agent {} exceeded maximum restart attempts ({})",
                        agent_id, self.config.max_restart_attempts
                    ) });
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
        let event = AgentLifecycleEvent::Restarted {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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
        let event = AgentLifecycleEvent::Paused {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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
        let event = AgentLifecycleEvent::Resumed {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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
        let event = AgentLifecycleEvent::Processing {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

        Ok(())
    }

    /// Mark agent processing complete
    pub async fn mark_processing_complete(
        &self,
        agent_id: Uuid,
        processing_time_ms: u64,
    ) -> Result<()> {
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
        let event = AgentLifecycleEvent::ProcessingComplete {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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
        let event = AgentLifecycleEvent::Error {
            agent_id,
            error: error.clone(),
            timestamp: now,
        };
        self.emit_event(event).await;

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
        agent_status
            .iter()
            .filter(|(_, status)| status.status == target_status)
            .map(|(agent_id, _)| *agent_id)
            .collect()
    }

    /// Get event subscriber
    pub async fn get_event_subscriber(
        &self,
    ) -> Option<mpsc::Receiver<AgentLifecycleEvent>> {
        // Implement proper subscription mechanism
        let (tx, rx) = mpsc::channel(1024);

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
                let age_hours = (now - status.last_updated).num_hours().max(0) as u64;
                let should_keep = age_hours <= max_age_hours
                    || matches!(status.status, AgentStatus::Ready | AgentStatus::Processing);

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
        let event = AgentLifecycleEvent::Ready {
            agent_id,
            timestamp: now,
        };
        self.emit_event(event).await;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_lifecycle_event_variants() {
        let agent_id = Uuid::new_v4();
        let now = Utc::now();
        let init = AgentLifecycleEvent::Initializing { agent_id, timestamp: now };
        let ready = AgentLifecycleEvent::Ready { agent_id, timestamp: now };
        let shutdown = AgentLifecycleEvent::Shutdown { agent_id, timestamp: now };
        assert!(matches!(init, AgentLifecycleEvent::Initializing { .. }));
        assert!(matches!(ready, AgentLifecycleEvent::Ready { .. }));
        assert!(matches!(shutdown, AgentLifecycleEvent::Shutdown { .. }));
    }

    #[test]
    fn test_agent_lifecycle_event_error() {
        let agent_id = Uuid::new_v4();
        let event = AgentLifecycleEvent::Error {
            agent_id,
            error: "crash".into(),
            timestamp: Utc::now(),
        };
        if let AgentLifecycleEvent::Error { error, .. } = &event {
            assert_eq!(error, "crash");
        } else {
            panic!("Expected Error variant");
        }
    }

    #[test]
    fn test_agent_lifecycle_status_creation() {
        let agent_id = Uuid::new_v4();
        let now = Utc::now();
        let status = AgentLifecycleStatus {
            agent_id,
            status: AgentStatus::Ready,
            started_at: now,
            last_updated: now,
            restart_count: 0,
            total_processing_time_ms: 0,
            last_error: None,
        };
        assert_eq!(status.agent_id, agent_id);
        assert_eq!(status.status, AgentStatus::Ready);
        assert_eq!(status.restart_count, 0);
    }

    #[test]
    fn test_agent_lifecycle_status_with_error() {
        let status = AgentLifecycleStatus {
            agent_id: Uuid::new_v4(),
            status: AgentStatus::Error("oops".into()),
            started_at: Utc::now(),
            last_updated: Utc::now(),
            restart_count: 2,
            total_processing_time_ms: 500,
            last_error: Some("oops".into()),
        };
        assert_eq!(status.restart_count, 2);
        assert_eq!(status.last_error, Some("oops".into()));
    }

    #[test]
    fn test_lifecycle_stats_default() {
        let stats = LifecycleStats::default();
        assert_eq!(stats.total_agents, 0);
        assert_eq!(stats.ready_agents, 0);
        assert_eq!(stats.processing_agents, 0);
        assert_eq!(stats.paused_agents, 0);
        assert_eq!(stats.error_agents, 0);
        assert_eq!(stats.total_restarts, 0);
    }

    #[test]
    fn test_lifecycle_stats_with_values() {
        let stats = LifecycleStats {
            total_agents: 5,
            ready_agents: 3,
            processing_agents: 1,
            paused_agents: 0,
            error_agents: 1,
            shutdown_agents: 0,
            shutting_down_agents: 0,
            initializing_agents: 0,
            total_processing_time_ms: 1000,
            total_restarts: 2,
        };
        assert_eq!(stats.total_agents, 5);
        assert_eq!(stats.total_restarts, 2);
    }

    #[test]
    fn test_lifecycle_manager_new() {
        let config = AgentManagerConfig::default();
        let manager = LifecycleManager::new(config);
        let statuses = futures::executor::block_on(manager.get_all_agent_statuses());
        assert!(statuses.is_empty());
    }

    #[test]
    fn test_lifecycle_manager_get_nonexistent_status() {
        let config = AgentManagerConfig::default();
        let manager = LifecycleManager::new(config);
        let status = futures::executor::block_on(manager.get_agent_status(Uuid::new_v4())).unwrap();
        assert!(status.is_none());
    }

    #[test]
    fn test_lifecycle_manager_cleanup_empty() {
        let config = AgentManagerConfig::default();
        let manager = LifecycleManager::new(config);
        let count = futures::executor::block_on(manager.cleanup_old_statuses(1)).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_lifecycle_stats_debug_clone() {
        let stats = LifecycleStats::default();
        let cloned = stats.clone();
        assert_eq!(format!("{:?}", stats), format!("{:?}", cloned));
    }
}
