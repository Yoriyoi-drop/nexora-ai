//! NXR-ÆTHER Agents System
//! 
//! Refactored implementation using individual agent modules

use std::collections::HashMap;
use crate::shared::base_model::NxrModelResult;
use super::config::AetherConfig;
use super::coordinator::AetherCoordinator;
use super::agents::{
    EmpathyPrimeAgent, PsycheAnalyzerAgent, EmotionWeaverAgent, CultureAdapterAgent,
    EmpathyPrimeConfig, PsycheAnalyzerConfig, EmotionWeaverConfig, CultureAdapterConfig,
};

/// NXR-ÆTHER Agents System
#[derive(Debug, Clone)]
pub struct AetherAgents {
    /// Empathy Prime Agent - Core empathy synthesis
    pub empathy_prime: EmpathyPrimeAgent,
    /// Psyche Analyzer Agent - Psychological analysis
    pub psyche_analyzer: PsycheAnalyzerAgent,
    /// Emotion Weaver Agent - Emotional processing
    pub emotion_weaver: EmotionWeaverAgent,
    /// Culture Adapter Agent - Cultural adaptation
    pub culture_adapter: CultureAdapterAgent,
    /// Agent coordination system
    pub coordinator: AetherCoordinator,
}

impl AetherAgents {
    /// Create a new Aether Agents system
    pub fn new(config: AetherConfig) -> NxrModelResult<Self> {
        // Create individual agents with their configurations
        let empathy_prime = EmpathyPrimeAgent::new(config.empathy_prime_config);
        let psyche_analyzer = PsycheAnalyzerAgent::new(config.psyche_analyzer_config);
        let emotion_weaver = EmotionWeaverAgent::new(config.emotion_weaver_config);
        let culture_adapter = CultureAdapterAgent::new(config.culture_adapter_config);
        
        // Create coordinator
        let coordinator = AetherCoordinator::new_aether();
        
        Ok(Self {
            empathy_prime,
            psyche_analyzer,
            emotion_weaver,
            culture_adapter,
            coordinator,
        })
    }
    
    /// Initialize all agents
    pub async fn initialize(&mut self) -> NxrModelResult<()> {
        // Initialize individual agents would happen here
        // For now, we'll assume they're already initialized
        Ok(())
    }
    
    /// Get agent by name
    pub fn get_agent(&self, agent_name: &str) -> Option<&dyn std::any::Any> {
        match agent_name {
            "empathy_prime" => Some(&self.empathy_prime),
            "psyche_analyzer" => Some(&self.psyche_analyzer),
            "emotion_weaver" => Some(&self.emotion_weaver),
            "culture_adapter" => Some(&self.culture_adapter),
            _ => None,
        }
    }
    
    /// Get all agent names
    pub fn get_agent_names(&self) -> Vec<String> {
        vec![
            "empathy_prime".to_string(),
            "psyche_analyzer".to_string(),
            "emotion_weaver".to_string(),
            "culture_adapter".to_string(),
        ]
    }
    
    /// Get system status
    pub fn get_system_status(&self) -> SystemStatus {
        SystemStatus {
            total_agents: 4,
            active_agents: 4, // Simplified - would check actual status
            coordinator_status: "active".to_string(),
            last_activity: chrono::Utc::now(),
        }
    }
}

/// System Status
#[derive(Debug, Clone)]
pub struct SystemStatus {
    /// Total number of agents
    pub total_agents: u32,
    /// Number of active agents
    pub active_agents: u32,
    /// Coordinator status
    pub coordinator_status: String,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

impl Default for AetherAgents {
    fn default() -> Self {
        let config = AetherConfig::default();
        Self::new(config).expect("AetherAgents::new with default config should succeed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aether_agents_creation() {
        let agents = AetherAgents::default();
        assert_eq!(agents.get_agent_names().len(), 4);
        assert!(agents.get_agent_names().contains(&"empathy_prime".to_string()));
        assert!(agents.get_agent_names().contains(&"psyche_analyzer".to_string()));
        assert!(agents.get_agent_names().contains(&"emotion_weaver".to_string()));
        assert!(agents.get_agent_names().contains(&"culture_adapter".to_string()));
    }

    #[test]
    fn test_system_status() {
        let agents = AetherAgents::default();
        let status = agents.get_system_status();
        assert_eq!(status.total_agents, 4);
        assert_eq!(status.active_agents, 4);
        assert_eq!(status.coordinator_status, "active");
    }
}
