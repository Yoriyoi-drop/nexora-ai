//! Agent Registry
//! 
//! Registry untuk tracking semua agent aktif dan mapping intent ke agent.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, warn};

use crate::{Agent, AgentError, Result, AgentStatus, AgentConfig};

/// Information tentang registered agent
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Unique ID agent
    pub agent_id: Uuid,
    /// Tipe agent
    pub agent_type: String,
    /// Konfigurasi agent
    pub config: AgentConfig,
    /// Status saat ini
    pub status: AgentStatus,
    /// Waktu registrasi
    pub registered_at: DateTime<Utc>,
    /// Waktu last update
    pub last_updated: DateTime<Utc>,
    /// Jumlah restart attempts
    pub restart_attempts: u32,
}

/// Mapping dari intent ke agent type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentMapping {
    /// Intent pattern (regex atau string)
    pub intent_pattern: String,
    /// Agent type yang handle intent ini
    pub agent_type: String,
    /// Priority untuk mapping ini
    pub priority: u8,
    /// Apakah mapping ini aktif
    pub active: bool,
}

/// Registry untuk semua agent
pub struct AgentRegistry {
    /// Map dari agent ID ke Agent instance
    agents: Arc<RwLock<HashMap<Uuid, Box<dyn Agent>>>>,
    /// Map dari agent ID ke AgentInfo
    agent_info: Arc<RwLock<HashMap<Uuid, AgentInfo>>>,
    /// Intent mapping
    intent_mappings: Arc<RwLock<Vec<IntentMapping>>>,
    /// Map dari agent type ke list of agent IDs
    type_index: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
}

impl AgentRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            agent_info: Arc::new(RwLock::new(HashMap::new())),
            intent_mappings: Arc::new(RwLock::new(Vec::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register new agent
    pub async fn register_agent(
        &self,
        agent_id: Uuid,
        agent_type: String,
        agent: Box<dyn Agent>,
    ) -> Result<()> {
        let now = Utc::now();
        
        // Check jika agent sudah terdaftar
        {
            let agents = self.agents.read().await;
            if agents.contains_key(&agent_id) {
                return Err(AgentError::AgentAlreadyExists(agent_id.to_string()));
            }
        }
        
        // Get config from agent
        let agent_config = agent.get_config();
        
        // Create agent info
        let info = AgentInfo {
            agent_id,
            agent_type: agent_type.clone(),
            config: agent_config,
            status: AgentStatus::Initializing,
            registered_at: now,
            last_updated: now,
            restart_attempts: 0,
        };
        
        // Register agent
        {
            let mut agents = self.agents.write().await;
            agents.insert(agent_id, agent);
        }
        
        // Register info
        {
            let mut agent_info = self.agent_info.write().await;
            agent_info.insert(agent_id, info);
        }
        
        // Update type index
        {
            let mut type_index = self.type_index.write().await;
            type_index.entry(agent_type).or_insert_with(Vec::new).push(agent_id);
        }
        
        Ok(())
    }
    
    /// Unregister agent
    pub async fn unregister_agent(&self, agent_id: Uuid) -> Result<()> {
        // Get agent type before removing
        let agent_type = {
            let agent_info = self.agent_info.read().await;
            agent_info.get(&agent_id)
                .map(|info| info.agent_type.clone())
        };
        
        if let Some(agent_type) = agent_type {
            // Remove from agents
            {
                let mut agents = self.agents.write().await;
                agents.remove(&agent_id);
            }
            
            // Remove from info
            {
                let mut agent_info = self.agent_info.write().await;
                agent_info.remove(&agent_id);
            }
            
            // Update type index
            {
                let mut type_index = self.type_index.write().await;
                if let Some(agents) = type_index.get_mut(&agent_type) {
                    agents.retain(|&id| id != agent_id);
                    if agents.is_empty() {
                        type_index.remove(&agent_type);
                    }
                }
            }
            
            Ok(())
        } else {
            Err(AgentError::AgentNotFound(agent_id.to_string()))
        }
    }
    
    /// Get agent instance
    pub async fn get_agent(&self, _agent_id: Uuid) -> Result<Option<Box<dyn Agent>>> {
        // Note: We can't return a reference to Box<dyn Agent> from async function
        // This needs to be redesigned - perhaps store Arc<dyn Agent> instead
        Err(AgentError::ProcessingError("get_agent needs redesign for async".to_string()))
    }
    
    /// Get agent info
    pub async fn get_agent_info(&self, agent_id: Uuid) -> Result<Option<AgentInfo>> {
        let agent_info = self.agent_info.read().await;
        Ok(agent_info.get(&agent_id).cloned())
    }
    
    /// Update agent status
    pub async fn update_agent_status(&self, agent_id: Uuid, status: AgentStatus) -> Result<()> {
        let mut agent_info = self.agent_info.write().await;
        if let Some(info) = agent_info.get_mut(&agent_id) {
            info.status = status.clone();
            info.last_updated = Utc::now();
            Ok(())
        } else {
            Err(AgentError::AgentNotFound(agent_id.to_string()))
        }
    }
    
    /// List all agents
    pub async fn list_agents(&self) -> Result<Vec<(Uuid, String, AgentStatus)>> {
        let agent_info = self.agent_info.read().await;
        let mut result = Vec::with_capacity(agent_info.len());
        
        for (agent_id, info) in agent_info.iter() {
            result.push((*agent_id, info.agent_type.clone(), info.status.clone()));
        }
        
        Ok(result)
    }
    
    /// Get agents by type
    pub async fn get_agents_by_type(&self, agent_type: &str) -> Result<Vec<Uuid>> {
        let type_index = self.type_index.read().await;
        Ok(type_index.get(agent_type).cloned().unwrap_or_default())
    }
    
    /// Get agent count
    pub async fn agent_count(&self) -> usize {
        let agents = self.agents.read().await;
        agents.len()
    }
    
    /// Add intent mapping
    pub async fn add_intent_mapping(&self, mapping: IntentMapping) -> Result<()> {
        let mut mappings = self.intent_mappings.write().await;
        mappings.push(mapping);
        Ok(())
    }
    
    /// Remove intent mapping
    pub async fn remove_intent_mapping(&self, index: usize) -> Result<()> {
        let mut mappings = self.intent_mappings.write().await;
        if index < mappings.len() {
            mappings.remove(index);
            Ok(())
        } else {
            Err(AgentError::ProcessingError("Invalid mapping index".to_string()))
        }
    }
    
    /// Get agent for intent
    pub async fn get_agent_for_intent(&self, intent: &str) -> Result<Option<Uuid>> {
        let mappings = self.intent_mappings.read().await;
        
        // Find matching mapping (sorted by priority)
        let mut matching_mappings: Vec<_> = mappings.iter()
            .filter(|m| m.active && self.intent_matches(intent, &m.intent_pattern))
            .collect();
        
        matching_mappings.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        if let Some(mapping) = matching_mappings.first() {
            // Get available agent of this type
            let agents = self.get_agents_by_type(&mapping.agent_type).await?;
            
            // Return first available agent
            for agent_id in agents {
                if let Ok(Some(info)) = self.get_agent_info(agent_id).await {
                    if matches!(info.status, AgentStatus::Ready) {
                        return Ok(Some(agent_id));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Check if intent matches pattern
    fn intent_matches(&self, intent: &str, pattern: &str) -> bool {
        // Implement regex matching
        match regex::Regex::new(pattern) {
            Ok(regex) => {
                let matches = regex.is_match(intent);
                debug!("Intent '{}' matches pattern '{}': {}", intent, pattern, matches);
                matches
            }
            Err(e) => {
                warn!("Invalid regex pattern '{}': {}, falling back to simple matching", pattern, e);
                // Fallback to simple string matching
                intent.to_lowercase().contains(&pattern.to_lowercase()) ||
                pattern.to_lowercase().contains(&intent.to_lowercase())
            }
        }
    }
    
    /// Get all intent mappings
    pub async fn get_intent_mappings(&self) -> Vec<IntentMapping> {
        let mappings = self.intent_mappings.read().await;
        mappings.clone()
    }
    
    /// Get agents by status
    pub async fn get_agents_by_status(&self, status: AgentStatus) -> Result<Vec<Uuid>> {
        let agent_info = self.agent_info.read().await;
        let mut result = Vec::with_capacity(agent_info.len());
        
        for (agent_id, info) in agent_info.iter() {
            if info.status == status {
                result.push(*agent_id);
            }
        }
        
        Ok(result)
    }
    
    /// Increment restart attempts
    pub async fn increment_restart_attempts(&self, agent_id: Uuid) -> Result<u32> {
        let mut agent_info = self.agent_info.write().await;
        if let Some(info) = agent_info.get_mut(&agent_id) {
            info.restart_attempts += 1;
            info.last_updated = Utc::now();
            Ok(info.restart_attempts)
        } else {
            Err(AgentError::AgentNotFound(agent_id.to_string()))
        }
    }
    
    /// Reset restart attempts
    pub async fn reset_restart_attempts(&self, agent_id: Uuid) -> Result<()> {
        let mut agent_info = self.agent_info.write().await;
        if let Some(info) = agent_info.get_mut(&agent_id) {
            info.restart_attempts = 0;
            info.last_updated = Utc::now();
            Ok(())
        } else {
            Err(AgentError::AgentNotFound(agent_id.to_string()))
        }
    }
    
    /// Cleanup old/dead agents
    pub async fn cleanup_dead_agents(&self, max_idle_seconds: u64) -> Result<usize> {
        let now = Utc::now();
        let mut dead_agents = Vec::new();
        
        {
            let agent_info = self.agent_info.read().await;
            for (agent_id, info) in agent_info.iter() {
                let idle_duration = (now - info.last_updated).num_seconds();
                
                if idle_duration > max_idle_seconds as i64 &&
                   matches!(info.status, AgentStatus::Error(_) | AgentStatus::Shutdown) {
                    dead_agents.push(*agent_id);
                }
            }
        }
        
        // Unregister dead agents
        for agent_id in dead_agents.iter() {
            if let Err(e) = self.unregister_agent(*agent_id).await {
                eprintln!("Failed to unregister dead agent {}: {}", agent_id, e);
            }
        }
        
        Ok(dead_agents.len())
    }
}

impl Default for IntentMapping {
    fn default() -> Self {
        Self {
            intent_pattern: String::new(),
            agent_type: String::new(),
            priority: 50,
            active: true,
        }
    }
}

impl IntentMapping {
    /// Create new intent mapping
    pub fn new(intent_pattern: String, agent_type: String) -> Self {
        Self {
            intent_pattern,
            agent_type,
            priority: 50,
            active: true,
        }
    }
    
    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set active status
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}
