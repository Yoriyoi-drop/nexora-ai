//! Agent State
//! 
//! Shared runtime state untuk semua agent.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::{AgentError, Result};

/// Global state yang bisa diakses semua agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalState {
    /// System-wide configuration
    pub system_config: HashMap<String, serde_json::Value>,
    /// Global counters
    pub counters: HashMap<String, u64>,
    /// Global flags
    pub flags: HashMap<String, bool>,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Session-specific state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Session ID
    pub session_id: Uuid,
    /// User ID (jika ada)
    pub user_id: Option<Uuid>,
    /// Session data
    pub data: HashMap<String, serde_json::Value>,
    /// Session metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Session status
    pub status: SessionStatus,
}

/// Session status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Idle,
    Suspended,
    Closed,
}

/// Agent-specific state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpecificState {
    /// Agent ID
    pub agent_id: Uuid,
    /// Agent type
    pub agent_type: String,
    /// Private data
    pub private_data: HashMap<String, serde_json::Value>,
    /// Shared data (bisa diakses agent lain)
    pub shared_data: HashMap<String, serde_json::Value>,
    /// Performance metrics
    pub metrics: AgentMetrics,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Performance metrics untuk agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Total errors
    pub total_errors: u64,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Last request timestamp
    pub last_request_at: Option<DateTime<Utc>>,
}

/// Shared runtime state manager
pub struct AgentState {
    /// Global state
    global_state: Arc<RwLock<GlobalState>>,
    /// Session states
    session_states: Arc<RwLock<HashMap<Uuid, SessionState>>>,
    /// Agent-specific states
    agent_states: Arc<RwLock<HashMap<Uuid, AgentSpecificState>>>,
    /// State change listeners
    state_listeners: Arc<RwLock<Vec<StateChangeListener>>>,
}

/// Listener untuk state changes
pub type StateChangeListener = Box<dyn Fn(StateChangeEvent) + Send + Sync>;

/// State change event
#[derive(Debug, Clone)]
pub enum StateChangeEvent {
    /// Global state changed
    GlobalStateChanged { key: String, old_value: Option<serde_json::Value>, new_value: serde_json::Value },
    /// Session state changed
    SessionStateChanged { session_id: Uuid, key: String, old_value: Option<serde_json::Value>, new_value: serde_json::Value },
    /// Agent state changed
    AgentStateChanged { agent_id: Uuid, key: String, old_value: Option<serde_json::Value>, new_value: serde_json::Value },
    /// Session created
    SessionCreated { session_id: Uuid },
    /// Session closed
    SessionClosed { session_id: Uuid },
    /// Agent registered
    AgentRegistered { agent_id: Uuid },
    /// Agent unregistered
    AgentUnregistered { agent_id: Uuid },
}

impl AgentState {
    /// Create new agent state manager
    pub fn new() -> Self {
        Self {
            global_state: Arc::new(RwLock::new(GlobalState {
                system_config: HashMap::new(),
                counters: HashMap::new(),
                flags: HashMap::new(),
                last_updated: Utc::now(),
            })),
            session_states: Arc::new(RwLock::new(HashMap::new())),
            agent_states: Arc::new(RwLock::new(HashMap::new())),
            state_listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Get global state
    pub async fn get_global_state(&self) -> GlobalState {
        self.global_state.read().await.clone()
    }
    
    /// Update global state
    pub async fn update_global_state<F>(&self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut GlobalState),
    {
        let mut global_state = self.global_state.write().await;
        let old_state = global_state.clone();
        
        updater(&mut global_state);
        global_state.last_updated = Utc::now();
        
        // Emit changes
        self.emit_global_state_changes(&old_state, &global_state).await;
        
        Ok(())
    }
    
    /// Get global config value
    pub async fn get_global_config(&self, key: &str) -> Option<serde_json::Value> {
        let global_state = self.global_state.read().await;
        global_state.system_config.get(key).cloned()
    }
    
    /// Set global config value
    pub async fn set_global_config(&self, key: String, value: serde_json::Value) -> Result<()> {
        self.update_global_state(|state| {
            state.system_config.insert(key, value);
        }).await
    }
    
    /// Get global counter
    pub async fn get_global_counter(&self, key: &str) -> Option<u64> {
        let global_state = self.global_state.read().await;
        global_state.counters.get(key).cloned()
    }
    
    /// Increment global counter
    pub async fn increment_global_counter(&self, key: String, amount: u64) -> Result<u64> {
        let mut new_value = 0u64;
        self.update_global_state(|state| {
            let counter = state.counters.entry(key.clone()).or_insert(0);
            *counter += amount;
            new_value = *counter;
        }).await?;
        
        Ok(new_value)
    }
    
    /// Get global flag
    pub async fn get_global_flag(&self, key: &str) -> Option<bool> {
        let global_state = self.global_state.read().await;
        global_state.flags.get(key).cloned()
    }
    
    /// Set global flag
    pub async fn set_global_flag(&self, key: String, value: bool) -> Result<()> {
        self.update_global_state(|state| {
            state.flags.insert(key, value);
        }).await
    }
    
    /// Create new session
    pub async fn create_session(&self, session_id: Uuid, user_id: Option<Uuid>) -> Result<()> {
        let session_state = SessionState {
            session_id,
            user_id,
            data: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            status: SessionStatus::Active,
        };
        
        let mut session_states = self.session_states.write().await;
        session_states.insert(session_id, session_state);
        
        // Emit event
        self.emit_state_change(StateChangeEvent::SessionCreated { session_id }).await;
        
        Ok(())
    }
    
    /// Get session state
    pub async fn get_session_state(&self, session_id: Uuid) -> Option<SessionState> {
        let session_states = self.session_states.read().await;
        session_states.get(&session_id).cloned()
    }
    
    /// Update session state
    pub async fn update_session_state<F>(&self, session_id: Uuid, updater: F) -> Result<()>
    where
        F: FnOnce(&mut SessionState),
    {
        let mut session_states = self.session_states.write().await;
        
        if let Some(session_state) = session_states.get_mut(&session_id) {
            let old_data = session_state.data.clone();
            
            updater(session_state);
            session_state.last_activity = Utc::now();
            
            // Emit changes
            self.emit_session_state_changes(session_id, &old_data, &session_state.data).await;
            
            Ok(())
        } else {
            Err(AgentError::StateError(format!("Session {} not found", session_id)))
        }
    }
    
    /// Set session data
    pub async fn set_session_data(&self, session_id: Uuid, key: String, value: serde_json::Value) -> Result<()> {
        self.update_session_state(session_id, |state| {
            state.data.insert(key, value);
        }).await
    }
    
    /// Get session data
    pub async fn get_session_data(&self, session_id: Uuid, key: &str) -> Option<serde_json::Value> {
        let session_states = self.session_states.read().await;
        session_states.get(&session_id)
            .and_then(|state| state.data.get(key).cloned())
    }
    
    /// Close session
    pub async fn close_session(&self, session_id: Uuid) -> Result<()> {
        self.update_session_state(session_id, |state| {
            state.status = SessionStatus::Closed;
        }).await?;
        
        // Emit event
        self.emit_state_change(StateChangeEvent::SessionClosed { session_id }).await;
        
        Ok(())
    }
    
    /// Register agent state
    pub async fn register_agent(&self, agent_id: Uuid, agent_type: String) -> Result<()> {
        let agent_state = AgentSpecificState {
            agent_id,
            agent_type,
            private_data: HashMap::new(),
            shared_data: HashMap::new(),
            metrics: AgentMetrics::default(),
            last_updated: Utc::now(),
        };
        
        let mut agent_states = self.agent_states.write().await;
        agent_states.insert(agent_id, agent_state);
        
        // Emit event
        self.emit_state_change(StateChangeEvent::AgentRegistered { agent_id }).await;
        
        Ok(())
    }
    
    /// Unregister agent state
    pub async fn unregister_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut agent_states = self.agent_states.write().await;
        agent_states.remove(&agent_id);
        
        // Emit event
        self.emit_state_change(StateChangeEvent::AgentUnregistered { agent_id }).await;
        
        Ok(())
    }
    
    /// Get agent state
    pub async fn get_agent_state(&self, agent_id: Uuid) -> Option<AgentSpecificState> {
        let agent_states = self.agent_states.read().await;
        agent_states.get(&agent_id).cloned()
    }
    
    /// Update agent state
    pub async fn update_agent_state<F>(&self, agent_id: Uuid, updater: F) -> Result<()>
    where
        F: FnOnce(&mut AgentSpecificState),
    {
        let mut agent_states = self.agent_states.write().await;
        
        if let Some(agent_state) = agent_states.get_mut(&agent_id) {
            let old_shared_data = agent_state.shared_data.clone();
            
            updater(agent_state);
            agent_state.last_updated = Utc::now();
            
            // Emit changes
            self.emit_agent_state_changes(agent_id, &old_shared_data, &agent_state.shared_data).await;
            
            Ok(())
        } else {
            Err(AgentError::StateError(format!("Agent {} not found", agent_id)))
        }
    }
    
    /// Set agent private data
    pub async fn set_agent_private_data(&self, agent_id: Uuid, key: String, value: serde_json::Value) -> Result<()> {
        self.update_agent_state(agent_id, |state| {
            state.private_data.insert(key, value);
        }).await
    }
    
    /// Get agent private data
    pub async fn get_agent_private_data(&self, agent_id: Uuid, key: &str) -> Option<serde_json::Value> {
        let agent_states = self.agent_states.read().await;
        agent_states.get(&agent_id)
            .and_then(|state| state.private_data.get(key).cloned())
    }
    
    /// Set agent shared data
    pub async fn set_agent_shared_data(&self, agent_id: Uuid, key: String, value: serde_json::Value) -> Result<()> {
        self.update_agent_state(agent_id, |state| {
            state.shared_data.insert(key, value);
        }).await
    }
    
    /// Get agent shared data
    pub async fn get_agent_shared_data(&self, agent_id: Uuid, key: &str) -> Option<serde_json::Value> {
        let agent_states = self.agent_states.read().await;
        agent_states.get(&agent_id)
            .and_then(|state| state.shared_data.get(key).cloned())
    }
    
    /// Update agent metrics
    pub async fn update_agent_metrics<F>(&self, agent_id: Uuid, updater: F) -> Result<()>
    where
        F: FnOnce(&mut AgentMetrics),
    {
        self.update_agent_state(agent_id, |state| {
            updater(&mut state.metrics);
        }).await
    }
    
    /// Add state change listener
    pub async fn add_state_listener(&self, listener: StateChangeListener) {
        let mut listeners = self.state_listeners.write().await;
        listeners.push(listener);
    }
    
    /// Get all active sessions
    pub async fn get_active_sessions(&self) -> Vec<SessionState> {
        let session_states = self.session_states.read().await;
        session_states.values()
            .filter(|state| state.status == SessionStatus::Active)
            .cloned()
            .collect()
    }
    
    /// Cleanup old sessions
    pub async fn cleanup_old_sessions(&self, max_idle_hours: u64) -> Result<usize> {
        let now = Utc::now();
        let mut removed_count = 0;
        
        {
            let mut session_states = self.session_states.write().await;
            session_states.retain(|_session_id, state| {
                let idle_hours = (now - state.last_activity).num_hours() as u64;
                let should_keep = idle_hours <= max_idle_hours || 
                                state.status == SessionStatus::Active;
                
                if !should_keep {
                    removed_count += 1;
                }
                
                should_keep
            });
        }
        
        Ok(removed_count)
    }
    
    /// Emit state change event
    async fn emit_state_change(&self, event: StateChangeEvent) {
        let listeners = self.state_listeners.read().await;
        for listener in listeners.iter() {
            listener(event.clone());
        }
    }
    
    /// Emit global state changes
    async fn emit_global_state_changes(&self, old_state: &GlobalState, new_state: &GlobalState) {
        // Check for config changes
        for (key, new_value) in &new_state.system_config {
            let old_value = old_state.system_config.get(key);
            if old_value != Some(new_value) {
                let event = StateChangeEvent::GlobalStateChanged {
                    key: key.clone(),
                    old_value: old_value.cloned(),
                    new_value: new_value.clone(),
                };
                self.emit_state_change(event).await;
            }
        }
        
        // Check for counter changes
        for (key, &new_value) in &new_state.counters {
            let old_value = old_state.counters.get(key);
            if old_value != Some(&new_value) {
                let event = StateChangeEvent::GlobalStateChanged {
                    key: key.clone(),
                    old_value: old_value.map(|v| serde_json::Value::Number(serde_json::Number::from(*v))),
                    new_value: serde_json::Value::Number(serde_json::Number::from(new_value)),
                };
                self.emit_state_change(event).await;
            }
        }
        
        // Check for flag changes
        for (key, &new_value) in &new_state.flags {
            let old_value = old_state.flags.get(key);
            if old_value != Some(&new_value) {
                let event = StateChangeEvent::GlobalStateChanged {
                    key: key.clone(),
                    old_value: old_value.map(|v| serde_json::Value::Bool(*v)),
                    new_value: serde_json::Value::Bool(new_value),
                };
                self.emit_state_change(event).await;
            }
        }
    }
    
    /// Emit session state changes
    async fn emit_session_state_changes(&self, session_id: Uuid, old_data: &HashMap<String, serde_json::Value>, new_data: &HashMap<String, serde_json::Value>) {
        for (key, new_value) in new_data {
            let old_value = old_data.get(key);
            if old_value != Some(new_value) {
                let event = StateChangeEvent::SessionStateChanged {
                    session_id,
                    key: key.clone(),
                    old_value: old_value.cloned(),
                    new_value: new_value.clone(),
                };
                self.emit_state_change(event).await;
            }
        }
    }
    
    /// Emit agent state changes
    async fn emit_agent_state_changes(&self, agent_id: Uuid, old_data: &HashMap<String, serde_json::Value>, new_data: &HashMap<String, serde_json::Value>) {
        for (key, new_value) in new_data {
            let old_value = old_data.get(key);
            if old_value != Some(new_value) {
                let event = StateChangeEvent::AgentStateChanged {
                    agent_id,
                    key: key.clone(),
                    old_value: old_value.cloned(),
                    new_value: new_value.clone(),
                };
                self.emit_state_change(event).await;
            }
        }
    }
}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            system_config: HashMap::new(),
            counters: HashMap::new(),
            flags: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_errors: 0,
            avg_response_time_ms: 0.0,
            memory_usage_bytes: 0,
            cpu_usage_percent: 0.0,
            last_request_at: None,
        }
    }
}
