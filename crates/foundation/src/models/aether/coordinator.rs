//! NXR-ÆTHER Agent Coordinator
//! 
//! Coordination system for emotional intelligence and psychological analysis agents

use std::collections::HashMap;
use crate::shared::{
    agent_coordinator::{AgentCoordinator, CoordinationStrategy},
    agent_types::{TaskRoutingRule, CommunicationChannel, AgentResult},
};
// Agents are looked up by string name, not imported types


/// Aether-specific Agent Coordinator
pub struct AetherCoordinator {
    inner: AgentCoordinator<AetherCoordinatorData>,
}

/// Data for Aether coordinator
#[derive(Clone)]
pub struct AetherCoordinatorData {
    // Add any aether-specific data here
}

impl AetherCoordinator {
    /// Create a new Aether coordinator with empathy-driven strategy
    pub fn new_aether() -> Self {
        let inner_coordinator = AgentCoordinator::new(CoordinationStrategy::EmpathyDriven { 
            empathy_weight: 0.9 
        });
        
        let mut coordinator = Self {
            inner: inner_coordinator,
        };
        
        // Set up default routing rules for emotional tasks
        coordinator.setup_default_routing_rules();
        
        // Set up agent weights based on emotional capabilities
        coordinator.setup_agent_weights();
        
        // Set up communication channels
        coordinator.setup_communication_channels();
        
        coordinator
    }
    
    /// Setup default routing rules for emotional tasks
    fn setup_default_routing_rules(&mut self) {
        // Empathy tasks
        let empathy_rule = TaskRoutingRule {
            pattern: "empathy|understand|feel|compassion".to_string(),
            target_agent: "empathy_prime".to_string(),
            priority: 4,
            weight: Some(0.35),
        };
        
        // Psychological analysis tasks
        let psyche_rule = TaskRoutingRule {
            pattern: "analyze|psychology|mental|behavior".to_string(),
            target_agent: "psyche_analyzer".to_string(),
            priority: 3,
            weight: Some(0.3),
        };
        
        // Emotional processing tasks
        let emotion_rule = TaskRoutingRule {
            pattern: "emotion|feeling|mood|affect".to_string(),
            target_agent: "emotion_weaver".to_string(),
            priority: 2,
            weight: Some(0.2),
        };
        
        // Cultural adaptation tasks
        let culture_rule = TaskRoutingRule {
            pattern: "culture|cultural|adapt|context".to_string(),
            target_agent: "culture_adapter".to_string(),
            priority: 1,
            weight: Some(0.15),
        };
        
        self.inner.add_routing_rule(empathy_rule);
        self.inner.add_routing_rule(psyche_rule);
        self.inner.add_routing_rule(emotion_rule);
        self.inner.add_routing_rule(culture_rule);
    }
    
    /// Setup agent weights based on emotional capabilities
    fn setup_agent_weights(&mut self) {
        self.inner.add_agent_weight("empathy_prime".to_string(), 0.35);
        self.inner.add_agent_weight("psyche_analyzer".to_string(), 0.3);
        self.inner.add_agent_weight("emotion_weaver".to_string(), 0.2);
        self.inner.add_agent_weight("culture_adapter".to_string(), 0.15);
    }
    
    /// Setup communication channels between agents
    fn setup_communication_channels(&mut self) {
        // Empathy channel
        let empathy_channel = CommunicationChannel {
            id: "empathy_network".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::Topic("empathy".to_string()),
            connected_agents: vec![
                "empathy_prime".to_string(),
                "emotion_weaver".to_string(),
            ],
            config: HashMap::new(),
        };
        
        // Psychological analysis channel
        let psyche_channel = CommunicationChannel {
            id: "psyche_analysis".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::Broadcast,
            connected_agents: vec![
                "psyche_analyzer".to_string(),
                "empathy_prime".to_string(),
            ],
            config: HashMap::new(),
        };
        
        // Cultural context channel
        let culture_channel = CommunicationChannel {
            id: "cultural_context".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::RequestResponse,
            connected_agents: vec![
                "culture_adapter".to_string(),
                "empathy_prime".to_string(),
                "emotion_weaver".to_string(),
            ],
            config: HashMap::new(),
        };
        
        // Emotional synthesis channel
        let synthesis_channel = CommunicationChannel {
            id: "emotional_synthesis".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::Broadcast,
            connected_agents: vec![
                "empathy_prime".to_string(),
                "psyche_analyzer".to_string(),
                "emotion_weaver".to_string(),
                "culture_adapter".to_string(),
            ],
            config: HashMap::new(),
        };
        
        self.inner.add_communication_channel(empathy_channel);
        self.inner.add_communication_channel(psyche_channel);
        self.inner.add_communication_channel(culture_channel);
        self.inner.add_communication_channel(synthesis_channel);
    }
    
    /// Route emotional task to appropriate agent
    pub fn route_emotional_task(&self, task_description: &str, emotional_context: &str) -> AgentResult<String> {
        let enhanced_description = format!("{} context:{}", task_description, emotional_context);
        self.inner.route_task(&enhanced_description)
    }
    
    /// Get agents by emotional capability
    pub fn get_agents_by_capability(&self, capability: &str) -> Vec<String> {
        match capability {
            "empathy" => vec!["empathy_prime".to_string()],
            "psychology" => vec!["psyche_analyzer".to_string()],
            "emotion" => vec!["emotion_weaver".to_string()],
            "culture" => vec!["culture_adapter".to_string()],
            "emotional_intelligence" => vec![
                "empathy_prime".to_string(),
                "psyche_analyzer".to_string(),
                "emotion_weaver".to_string(),
            ],
            "comprehensive" => vec![
                "empathy_prime".to_string(),
                "psyche_analyzer".to_string(),
                "emotion_weaver".to_string(),
                "culture_adapter".to_string(),
            ],
            _ => vec![],
        }
    }
    
    /// Update coordination strategy based on emotional context
    pub fn update_emotional_strategy(&mut self, empathy_level: f32) {
        let new_strategy = if empathy_level > 0.8 {
            CoordinationStrategy::EmpathyDriven { 
                empathy_weight: empathy_level 
            }
        } else if empathy_level > 0.5 {
            CoordinationStrategy::Parallel
        } else {
            CoordinationStrategy::Sequential
        };
        
        self.inner.update_strategy(new_strategy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aether_coordinator_creation() {
        let coordinator = AetherCoordinator::new_aether();
        assert!(coordinator.inner.routing_rules.len() > 0);
        assert!(coordinator.inner.agent_weights.len() == 4);
        assert!(coordinator.inner.communication_channels.len() == 4);
    }

    #[test]
    fn test_emotional_task_routing() {
        let coordinator = AetherCoordinator::new_aether();
        
        let result = coordinator.route_emotional_task("help me understand their feelings", "empathy");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "empathy_prime");
    }

    #[test]
    fn test_agents_by_capability() {
        let coordinator = AetherCoordinator::new_aether();
        
        let empathy_agents = coordinator.get_agents_by_capability("empathy");
        assert_eq!(empathy_agents.len(), 1);
        assert_eq!(empathy_agents[0], "empathy_prime");
        
        let comprehensive_agents = coordinator.get_agents_by_capability("comprehensive");
        assert_eq!(comprehensive_agents.len(), 4);
    }
}
