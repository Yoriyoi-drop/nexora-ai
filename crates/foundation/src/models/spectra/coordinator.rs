//! NXR-SPECTRA Agent Coordinator
//! 
//! Coordination system for creative multimodal synthesis agents

use std::collections::HashMap;
use crate::shared::{
    agent_coordinator::{AgentCoordinator, CoordinationStrategy},
    agent_types::{TaskRoutingRule, CommunicationChannel, AgentResult},
};
use super::agents::{
    CreativeMuseAgent, ArtisticWeaverAgent, StyleAdapterAgent, InnovationEngineAgent,
};

/// Spectra-specific Agent Coordinator
pub struct SpectraCoordinator {
    inner: AgentCoordinator<SpectraCoordinatorData>,
}

/// Data for Spectra coordinator
#[derive(Clone)]
pub struct SpectraCoordinatorData {
    // Add any spectra-specific data here
}

impl SpectraCoordinator {
    /// Create a new Spectra coordinator with creative-driven strategy
    pub fn new_spectra() -> Self {
        let inner_coordinator = AgentCoordinator::new(CoordinationStrategy::CreativeDriven { 
            creativity_weight: 0.8 
        });
        
        let mut coordinator = Self {
            inner: inner_coordinator,
        };
        
        // Set up default routing rules for creative tasks
        coordinator.setup_default_routing_rules();
        
        // Set up agent weights based on creative capabilities
        coordinator.setup_agent_weights();
        
        // Set up communication channels
        coordinator.setup_communication_channels();
        
        coordinator
    }
    
    /// Setup default routing rules for creative tasks
    fn setup_default_routing_rules(&mut self) {
        // Visual creative tasks
        let visual_rule = TaskRoutingRule {
            pattern: "visual|image|graphic|design".to_string(),
            target_agent: "creative_muse".to_string(),
            priority: 3,
            weight: Some(0.3),
        };
        
        // Style adaptation tasks
        let style_rule = TaskRoutingRule {
            pattern: "style|adapt|transform|convert".to_string(),
            target_agent: "style_adapter".to_string(),
            priority: 2,
            weight: Some(0.25),
        };
        
        // Innovation tasks
        let innovation_rule = TaskRoutingRule {
            pattern: "innovate|novel|new|original".to_string(),
            target_agent: "innovation_engine".to_string(),
            priority: 4,
            weight: Some(0.35),
        };
        
        // Artistic tasks
        let artistic_rule = TaskRoutingRule {
            pattern: "artistic|art|creative|compose".to_string(),
            target_agent: "artistic_weaver".to_string(),
            priority: 2,
            weight: Some(0.1),
        };
        
        self.inner.add_routing_rule(visual_rule);
        self.inner.add_routing_rule(style_rule);
        self.inner.add_routing_rule(innovation_rule);
        self.inner.add_routing_rule(artistic_rule);
    }
    
    /// Setup agent weights based on creative capabilities
    fn setup_agent_weights(&mut self) {
        self.inner.add_agent_weight("creative_muse".to_string(), 0.3);
        self.inner.add_agent_weight("artistic_weaver".to_string(), 0.2);
        self.inner.add_agent_weight("style_adapter".to_string(), 0.25);
        self.inner.add_agent_weight("innovation_engine".to_string(), 0.25);
    }
    
    /// Setup communication channels between agents
    fn setup_communication_channels(&mut self) {
        // Creative inspiration channel
        let inspiration_channel = CommunicationChannel {
            id: "creative_inspiration".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::Topic("inspiration".to_string()),
            connected_agents: vec![
                "creative_muse".to_string(),
                "innovation_engine".to_string(),
            ],
            config: HashMap::new(),
        };
        
        // Style coordination channel
        let style_channel = CommunicationChannel {
            id: "style_coordination".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::Broadcast,
            connected_agents: vec![
                "artistic_weaver".to_string(),
                "style_adapter".to_string(),
            ],
            config: HashMap::new(),
        };
        
        // Multimodal synthesis channel
        let multimodal_channel = CommunicationChannel {
            id: "multimodal_synthesis".to_string(),
            channel_type: crate::shared::agent_types::ChannelType::RequestResponse,
            connected_agents: vec![
                "creative_muse".to_string(),
                "artistic_weaver".to_string(),
                "style_adapter".to_string(),
                "innovation_engine".to_string(),
            ],
            config: HashMap::new(),
        };
        
        self.inner.add_communication_channel(inspiration_channel);
        self.inner.add_communication_channel(style_channel);
        self.inner.add_communication_channel(multimodal_channel);
    }
    
    /// Route creative task to appropriate agent
    pub fn route_creative_task(&self, task_description: &str, creative_domain: &str) -> AgentResult<String> {
        let enhanced_description = format!("{} domain:{}", task_description, creative_domain);
        self.inner.route_task(&enhanced_description)
    }
    
    /// Get agents by creative capability
    pub fn get_agents_by_capability(&self, capability: &str) -> Vec<String> {
        match capability {
            "visual" => vec!["creative_muse".to_string()],
            "style" => vec!["artistic_weaver".to_string(), "style_adapter".to_string()],
            "innovation" => vec!["innovation_engine".to_string()],
            "multimodal" => vec![
                "creative_muse".to_string(),
                "artistic_weaver".to_string(),
                "style_adapter".to_string(),
                "innovation_engine".to_string(),
            ],
            _ => vec![],
        }
    }
    
    /// Update coordination strategy based on creative context
    pub fn update_creative_strategy(&mut self, creativity_level: f32) {
        let new_strategy = if creativity_level > 0.8 {
            CoordinationStrategy::CreativeDriven { 
                creativity_weight: creativity_level 
            }
        } else if creativity_level > 0.5 {
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
    fn test_spectra_coordinator_creation() {
        let coordinator = SpectraCoordinator::new_spectra();
        assert!(coordinator.inner.routing_rules.len() > 0);
        assert!(coordinator.inner.agent_weights.len() == 4);
        assert!(coordinator.inner.communication_channels.len() == 3);
    }

    #[test]
    fn test_creative_task_routing() {
        let coordinator = SpectraCoordinator::new_spectra();
        
        let result = coordinator.route_creative_task("create a beautiful image", "visual");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "creative_muse");
    }

    #[test]
    fn test_agents_by_capability() {
        let coordinator = SpectraCoordinator::new_spectra();
        
        let visual_agents = coordinator.get_agents_by_capability("visual");
        assert_eq!(visual_agents.len(), 1);
        assert_eq!(visual_agents[0], "creative_muse");
        
        let multimodal_agents = coordinator.get_agents_by_capability("multimodal");
        assert_eq!(multimodal_agents.len(), 4);
    }
}
