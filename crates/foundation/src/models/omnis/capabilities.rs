//! NXR-OMNIS Capabilities
//! 
//! Capability vector and specifications for NXR-OMNIS

use crate::shared::{
    capability_spec::{CapabilityVector, CapabilitySpec, CapabilityDomain, CapabilityLevel, ResourceRequirements},
    model_identity::NxrModelId,
};

/// NXR-OMNIS Capabilities Manager
pub struct OmnisCapabilities {
    /// Capability vector
    vector: CapabilityVector,
}

impl OmnisCapabilities {
    /// Create new capabilities for NXR-OMNIS
    pub fn new() -> Self {
        let vector = Self::create_capability_vector();
        Self { vector }
    }

    /// Get capability vector
    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }

    /// Create capability vector for NXR-OMNIS
    fn create_capability_vector() -> CapabilityVector {
        CapabilityVector::new(NxrModelId::Omnis)
            // Text capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "long-chain reasoning".to_string(),
                    "world modeling".to_string(),
                    "context synthesis".to_string(),
                    "meta-reasoning".to_string(),
                    "truth arbitration".to_string(),
                    "chain execution".to_string(),
                    "cross-modal synthesis".to_string(),
                    "unlimited context understanding".to_string(),
                ])
                .with_metric("accuracy".to_string(), 0.997)
                .with_metric("reasoning_depth".to_string(), 10.0)
                .with_metric("context_window".to_string(), 10_000_000.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 64.0,
                    min_compute_units: 128,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(48.0),
                    requires_network: true,
                    max_latency_ms: Some(1000),
                }))
            
            // Logic capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "formal reasoning".to_string(),
                    "proof generation".to_string(),
                    "paradox resolution".to_string(),
                    "logical decomposition".to_string(),
                    "inference optimization".to_string(),
                    "contradiction detection".to_string(),
                ])
                .with_metric("proof_accuracy".to_string(), 0.995)
                .with_metric("inference_speed".to_string(), 850.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: false,
                    max_latency_ms: Some(500),
                }))
            
            // Mathematics capabilities - Master level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Mathematics, CapabilityLevel::Master)
                .with_sub_capabilities(vec![
                    "advanced calculus".to_string(),
                    "abstract algebra".to_string(),
                    "statistical analysis".to_string(),
                    "numerical methods".to_string(),
                    "optimization theory".to_string(),
                    "probability theory".to_string(),
                    "discrete mathematics".to_string(),
                ])
                .with_metric("math_accuracy".to_string(), 0.98)
                .with_metric("problem_complexity".to_string(), 9.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(16.0),
                    requires_network: false,
                    max_latency_ms: Some(300),
                }))
            
            // Multimodal capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Multimodal, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "text-vision synthesis".to_string(),
                    "cross-modal reasoning".to_string(),
                    "multimodal understanding".to_string(),
                    "modality translation".to_string(),
                    "sensory integration".to_string(),
                ])
                .with_metric("modality_accuracy".to_string(), 0.92)
                .with_metric("integration_score".to_string(), 0.88)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 48.0,
                    min_compute_units: 96,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(32.0),
                    requires_network: true,
                    max_latency_ms: Some(800),
                }))
            
            // Emotional capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Emotional, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "empathy synthesis".to_string(),
                    "emotional context analysis".to_string(),
                    "tone adaptation".to_string(),
                    "psychological modeling".to_string(),
                    "emotional intelligence".to_string(),
                ])
                .with_metric("emotional_accuracy".to_string(), 0.91)
                .with_metric("empathy_score".to_string(), 0.89)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: false,
                    max_latency_ms: Some(600),
                }))
            
            // Strategy capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Strategy, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "strategic planning".to_string(),
                    "game theory application".to_string(),
                    "risk assessment".to_string(),
                    "decision optimization".to_string(),
                    "long-term planning".to_string(),
                    "competitive analysis".to_string(),
                ])
                .with_metric("strategy_accuracy".to_string(), 0.96)
                .with_metric("planning_horizon".to_string(), 365.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(1200),
                }))
            
            // Knowledge capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Knowledge, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "knowledge synthesis".to_string(),
                    "fact verification".to_string(),
                    "source evaluation".to_string(),
                    "knowledge integration".to_string(),
                    "semantic understanding".to_string(),
                    "concept mapping".to_string(),
                ])
                .with_metric("knowledge_accuracy".to_string(), 0.998)
                .with_metric("fact_coverage".to_string(), 0.95)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 128.0,
                    min_compute_units: 256,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(64.0),
                    requires_network: true,
                    max_latency_ms: Some(2000),
                }))
            
            // Security capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Security, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "threat analysis".to_string(),
                    "vulnerability assessment".to_string(),
                    "security reasoning".to_string(),
                    "risk evaluation".to_string(),
                    "security protocol analysis".to_string(),
                ])
                .with_metric("security_accuracy".to_string(), 0.89)
                .with_metric("threat_detection".to_string(), 0.87)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(400),
                }))
            
            // Creative capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Creative, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "creative synthesis".to_string(),
                    "novel concept generation".to_string(),
                    "artistic understanding".to_string(),
                    "creative problem solving".to_string(),
                    "innovation facilitation".to_string(),
                ])
                .with_metric("creativity_score".to_string(), 0.93)
                .with_metric("novelty_score".to_string(), 0.91)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: false,
                    max_latency_ms: Some(1500),
                }))
            
            // Orchestration capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Orchestration, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "multi-agent coordination".to_string(),
                    "task decomposition".to_string(),
                    "resource optimization".to_string(),
                    "workflow orchestration".to_string(),
                    "agent communication".to_string(),
                    "consensus building".to_string(),
                ])
                .with_metric("coordination_accuracy".to_string(), 0.98)
                .with_metric("agent_efficiency".to_string(), 0.96)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 48.0,
                    min_compute_units: 96,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(32.0),
                    requires_network: true,
                    max_latency_ms: Some(800),
                }))
            
            // Self-improvement capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::SelfImprovement, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "self-reflection".to_string(),
                    "performance monitoring".to_string(),
                    "strategy adaptation".to_string(),
                    "learning optimization".to_string(),
                    "meta-learning".to_string(),
                ])
                .with_metric("improvement_rate".to_string(), 0.85)
                .with_metric("adaptation_speed".to_string(), 0.82)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: false,
                    max_latency_ms: Some(1000),
                }))
            
            // Edge computing capabilities - Basic level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Edge, CapabilityLevel::Basic)
                .with_sub_capabilities(vec![
                    "lightweight inference".to_string(),
                    "resource optimization".to_string(),
                    "edge deployment".to_string(),
                ])
                .with_metric("edge_efficiency".to_string(), 0.75)
                .with_metric("resource_usage".to_string(), 0.8)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 4.0,
                    min_compute_units: 8,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(200),
                }))
            .calculate_score()
    }

    /// Check if model supports specific capability
    pub fn supports_capability(&self, domain: &CapabilityDomain, min_level: CapabilityLevel) -> bool {
        self.vector.has_capability(domain, min_level)
    }

    /// Get capability score for domain
    pub fn get_capability_score(&self, domain: &CapabilityDomain) -> f32 {
        self.vector
            .get_capability(domain)
            .map(|cap| cap.score())
            .unwrap_or(0.0)
    }

    /// Get expert domains
    pub fn expert_domains(&self) -> Vec<&CapabilityDomain> {
        self.vector.expert_domains()
    }

    /// Get specializations
    pub fn specializations(&self) -> &Vec<CapabilityDomain> {
        &self.vector.specializations
    }

    /// Get overall capability score
    pub fn overall_score(&self) -> f32 {
        self.vector.overall_score
    }

    /// Get resource requirements for domain
    pub fn get_resource_requirements(&self, domain: &CapabilityDomain) -> Option<&ResourceRequirements> {
        self.vector.get_capability(domain).map(|cap| &cap.resource_requirements)
    }

    /// Validate capabilities
    pub fn validate(&self) -> Result<(), String> {
        // Check that core capabilities are at transcendent level
        let core_domains = vec![
            CapabilityDomain::Text,
            CapabilityDomain::Logic,
            CapabilityDomain::Strategy,
            CapabilityDomain::Knowledge,
            CapabilityDomain::Orchestration,
        ];

        for domain in core_domains {
            if !self.supports_capability(&domain, CapabilityLevel::Transcendent) {
                return Err(format!("Core capability {:?} not at transcendent level", domain));
            }
        }

        // Check that overall score is high enough
        if self.overall_score() < 0.9 {
            return Err("Overall capability score too low".to_string());
        }

        Ok(())
    }

    /// Update capability based on performance
    pub fn update_capability(&mut self, domain: CapabilityDomain, performance_score: f32) {
        // This would update the capability based on actual performance
        // For now, just log the update
        println!("Updating capability {:?} with score: {}", domain, performance_score);
    }

    /// Get capability summary
    pub fn capability_summary(&self) -> CapabilitySummary {
        let mut expert_domains = Vec::new();
        let mut advanced_domains = Vec::new();
        let mut intermediate_domains = Vec::new();
        let mut basic_domains = Vec::new();

        for (domain, capability) in &self.vector.capabilities {
            match capability.level {
                CapabilityLevel::Transcendent | CapabilityLevel::Master => {
                    expert_domains.push(domain.clone());
                }
                CapabilityLevel::Expert | CapabilityLevel::Advanced => {
                    advanced_domains.push(domain.clone());
                }
                CapabilityLevel::Intermediate => {
                    intermediate_domains.push(domain.clone());
                }
                CapabilityLevel::Basic => {
                    basic_domains.push(domain.clone());
                }
                CapabilityLevel::None => {}
            }
        }

        CapabilitySummary {
            expert_domains,
            advanced_domains,
            intermediate_domains,
            basic_domains,
            overall_score: self.overall_score(),
            specializations: self.specializations().clone(),
        }
    }
}

/// Capability summary
#[derive(Debug, Clone)]
pub struct CapabilitySummary {
    /// Expert-level domains
    pub expert_domains: Vec<CapabilityDomain>,
    /// Advanced-level domains
    pub advanced_domains: Vec<CapabilityDomain>,
    /// Intermediate-level domains
    pub intermediate_domains: Vec<CapabilityDomain>,
    /// Basic-level domains
    pub basic_domains: Vec<CapabilityDomain>,
    /// Overall score
    pub overall_score: f32,
    /// Specializations
    pub specializations: Vec<CapabilityDomain>,
}

impl CapabilitySummary {
    /// Get performance tier
    pub fn performance_tier(&self) -> PerformanceTier {
        if self.overall_score >= 0.95 {
            PerformanceTier::Transcendent
        } else if self.overall_score >= 0.9 {
            PerformanceTier::Master
        } else if self.overall_score >= 0.8 {
            PerformanceTier::Expert
        } else if self.overall_score >= 0.7 {
            PerformanceTier::Advanced
        } else if self.overall_score >= 0.6 {
            PerformanceTier::Intermediate
        } else {
            PerformanceTier::Basic
        }
    }

    /// Get capability count by level
    pub fn capability_counts(&self) -> CapabilityCounts {
        CapabilityCounts {
            expert: self.expert_domains.len(),
            advanced: self.advanced_domains.len(),
            intermediate: self.intermediate_domains.len(),
            basic: self.basic_domains.len(),
            total: self.expert_domains.len() + self.advanced_domains.len() + 
                   self.intermediate_domains.len() + self.basic_domains.len(),
        }
    }
}

/// Performance tier
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceTier {
    /// Transcendent performance
    Transcendent,
    /// Master performance
    Master,
    /// Expert performance
    Expert,
    /// Advanced performance
    Advanced,
    /// Intermediate performance
    Intermediate,
    /// Basic performance
    Basic,
}

/// Capability counts
#[derive(Debug, Clone)]
pub struct CapabilityCounts {
    /// Expert-level count
    pub expert: usize,
    /// Advanced-level count
    pub advanced: usize,
    /// Intermediate-level count
    pub intermediate: usize,
    /// Basic-level count
    pub basic: usize,
    /// Total count
    pub total: usize,
}

impl Default for OmnisCapabilities {
    fn default() -> Self {
        Self::new()
    }
}
