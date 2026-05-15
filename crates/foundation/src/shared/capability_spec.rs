//! NXR Model Capability Specification
//! 
//! Defines capabilities, domains, and performance specs for NXR models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Capability Domain
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityDomain {
    /// Text generation and understanding
    Text,
    /// Code generation and analysis
    Code,
    /// Mathematical reasoning
    Mathematics,
    /// Logical reasoning
    Logic,
    /// Visual processing
    Vision,
    /// Audio processing
    Audio,
    /// Multimodal synthesis
    Multimodal,
    /// Emotional intelligence
    Emotional,
    /// Strategic planning
    Strategy,
    /// Knowledge retrieval
    Knowledge,
    /// Security analysis
    Security,
    /// Creative generation
    Creative,
    /// Agent orchestration
    Orchestration,
    /// Self-improvement
    SelfImprovement,
    /// Edge computing
    Edge,
    /// Consensus building
    Consensus,
    /// Coordination
    Coordination,
    /// Alignment
    Alignment,
    /// Resource management
    ResourceManagement,
    /// Communication
    Communication,
    /// Support
    Support,
    /// Simulation
    Simulation,
    /// Decision making
    Decision,
    /// Monitoring
    Monitoring,
}

/// Capability Level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CapabilityLevel {
    /// No capability
    None = 0,
    /// Basic capability
    Basic = 1,
    /// Intermediate capability
    Intermediate = 2,
    /// Advanced capability
    Advanced = 3,
    /// Expert level capability
    Expert = 4,
    /// Master level capability
    Master = 5,
    /// Transcendent capability
    Transcendent = 6,
}

/// Capability Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySpec {
    /// Domain
    pub domain: CapabilityDomain,
    /// Level in this domain
    pub level: CapabilityLevel,
    /// Specific sub-capabilities
    pub sub_capabilities: Vec<String>,
    /// Performance metrics (if applicable)
    pub metrics: HashMap<String, f32>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Resource Requirements for capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum memory in GB
    pub min_memory_gb: f32,
    /// Minimum compute units
    pub min_compute_units: u32,
    /// GPU requirement
    pub requires_gpu: bool,
    /// Minimum GPU memory in GB
    pub min_gpu_memory_gb: Option<f32>,
    /// Network requirement
    pub requires_network: bool,
    /// Latency requirement in ms
    pub max_latency_ms: Option<u32>,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            min_memory_gb: 1.0,
            min_compute_units: 1,
            requires_gpu: false,
            min_gpu_memory_gb: None,
            requires_network: false,
            max_latency_ms: None,
        }
    }
}

impl CapabilitySpec {
    /// Create new capability specification
    pub fn new(domain: CapabilityDomain, level: CapabilityLevel) -> Self {
        Self {
            domain,
            level,
            sub_capabilities: Vec::new(),
            metrics: HashMap::new(),
            resource_requirements: ResourceRequirements::default(),
        }
    }

    /// Add sub-capability
    pub fn with_sub_capability(mut self, capability: String) -> Self {
        self.sub_capabilities.push(capability);
        self
    }

    /// Add multiple sub-capabilities
    pub fn with_sub_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.sub_capabilities.extend(capabilities);
        self
    }

    /// Add performance metric
    pub fn with_metric(mut self, key: String, value: f32) -> Self {
        self.metrics.insert(key, value);
        self
    }

    /// Set resource requirements
    pub fn with_resources(mut self, resources: ResourceRequirements) -> Self {
        self.resource_requirements = resources;
        self
    }

    /// Check if capability meets minimum level
    pub fn meets_minimum(&self, min_level: CapabilityLevel) -> bool {
        self.level >= min_level
    }

    /// Get capability score (0.0 - 1.0)
    pub fn score(&self) -> f32 {
        self.level as u8 as f32 / 6.0
    }
}

/// Model Capability Vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityVector {
    /// Model identifier
    pub model_id: crate::shared::model_identity::NxrModelId,
    /// Capability specifications
    pub capabilities: HashMap<CapabilityDomain, CapabilitySpec>,
    /// Overall capability score
    pub overall_score: f32,
    /// Specialization domains (top 3)
    pub specializations: Vec<CapabilityDomain>,
}

impl CapabilityVector {
    /// Create new capability vector
    pub fn new(model_id: crate::shared::model_identity::NxrModelId) -> Self {
        Self {
            model_id,
            capabilities: HashMap::new(),
            overall_score: 0.0,
            specializations: Vec::new(),
        }
    }

    /// Add capability
    pub fn with_capability(mut self, capability: CapabilitySpec) -> Self {
        let domain = capability.domain.clone();
        self.capabilities.insert(domain, capability);
        self
    }

    /// Calculate overall score
    pub fn calculate_score(mut self) -> Self {
        let total_score: f32 = self.capabilities
            .values()
            .map(|cap| cap.score())
            .sum();
        
        self.overall_score = if self.capabilities.is_empty() {
            0.0
        } else {
            total_score / self.capabilities.len() as f32
        };

        // Calculate specializations (top 3 domains by level)
        let mut domains: Vec<_> = self.capabilities
            .iter()
            .map(|(domain, cap)| (domain.clone(), cap.level))
            .collect();
        
        domains.sort_by(|a, b| b.1.cmp(&a.1));
        self.specializations = domains.into_iter().take(3).map(|(d, _)| d).collect();

        self
    }

    /// Get capability by domain
    pub fn get_capability(&self, domain: &CapabilityDomain) -> Option<&CapabilitySpec> {
        self.capabilities.get(domain)
    }

    /// Check if model has capability in domain
    pub fn has_capability(&self, domain: &CapabilityDomain, min_level: CapabilityLevel) -> bool {
        self.capabilities
            .get(domain)
            .map(|cap| cap.meets_minimum(min_level))
            .unwrap_or(false)
    }

    /// Get domains where model excels (Expert+)
    pub fn expert_domains(&self) -> Vec<&CapabilityDomain> {
        self.capabilities
            .iter()
            .filter(|(_, cap)| cap.level >= CapabilityLevel::Expert)
            .map(|(domain, _)| domain)
            .collect()
    }
}

/// Predefined capability specifications for each model
pub mod predefined {
    use super::*;

    /// NXR-OMNIS capabilities
    pub fn omnis_capabilities() -> CapabilityVector {
        CapabilityVector::new(crate::shared::model_identity::NxrModelId::Omnis)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "long-chain reasoning".to_string(),
                    "world modeling".to_string(),
                    "context synthesis".to_string(),
                ])
                .with_metric("accuracy".to_string(), 0.997)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 64.0,
                    min_compute_units: 128,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(48.0),
                    requires_network: true,
                    max_latency_ms: Some(1000),
                }))
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "formal reasoning".to_string(),
                    "proof generation".to_string(),
                    "paradox resolution".to_string(),
                ]))
            .with_capability(CapabilitySpec::new(CapabilityDomain::Mathematics, CapabilityLevel::Master)
                .with_sub_capabilities(vec![
                    "advanced calculus".to_string(),
                    "abstract algebra".to_string(),
                    "statistical analysis".to_string(),
                ]))
            .with_capability(CapabilitySpec::new(CapabilityDomain::Multimodal, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "text-vision synthesis".to_string(),
                    "cross-modal reasoning".to_string(),
                ]))
            .calculate_score()
    }

    /// NXR-VORTEX capabilities
    pub fn vortex_capabilities() -> CapabilityVector {
        CapabilityVector::new(crate::shared::model_identity::NxrModelId::Vortex)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Code, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "system architecture".to_string(),
                    "debugging".to_string(),
                    "optimization".to_string(),
                    "security review".to_string(),
                ])
                .with_metric("human_eval".to_string(), 0.972)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: false,
                    max_latency_ms: Some(500),
                }))
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "program logic".to_string(),
                    "algorithm design".to_string(),
                ]))
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "technical documentation".to_string(),
                    "code explanation".to_string(),
                ]))
            .calculate_score()
    }

    /// NXR-ÆTHER capabilities
    pub fn aether_capabilities() -> CapabilityVector {
        CapabilityVector::new(crate::shared::model_identity::NxrModelId::Aether)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Emotional, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "empathy synthesis".to_string(),
                    "psychological analysis".to_string(),
                    "emotional support".to_string(),
                    "tone adaptation".to_string(),
                ])
                .with_metric("eq_score".to_string(), 0.965)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(12.0),
                    requires_network: false,
                    max_latency_ms: Some(800),
                }))
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "nuanced conversation".to_string(),
                    "contextual understanding".to_string(),
                ]))
            .calculate_score()
    }

    /// Get capabilities for any model
    pub fn get_capabilities(model_id: crate::shared::model_identity::NxrModelId) -> CapabilityVector {
        match model_id {
            crate::shared::model_identity::NxrModelId::Omnis => omnis_capabilities(),
            crate::shared::model_identity::NxrModelId::Vortex => vortex_capabilities(),
            crate::shared::model_identity::NxrModelId::Aether => aether_capabilities(),
            crate::shared::model_identity::NxrModelId::Nexum |
            crate::shared::model_identity::NxrModelId::Spectra |
            crate::shared::model_identity::NxrModelId::Swift |
            crate::shared::model_identity::NxrModelId::Genesis |
            crate::shared::model_identity::NxrModelId::Kronos |
            crate::shared::model_identity::NxrModelId::Cipher |
            crate::shared::model_identity::NxrModelId::Axiom => {
                let mut vector = CapabilityVector::new(model_id);
                vector.reasoning_score = 0.6;
                vector.general_knowledge_score = 0.7;
                vector.code_generation_score = 0.5;
                vector.multimodal_score = 0.4;
                vector.memory_capacity = 0.6;
                vector.context_window = 4096;
                vector
            }
            _ => CapabilityVector::new(model_id).calculate_score(),
        }
    }
}
