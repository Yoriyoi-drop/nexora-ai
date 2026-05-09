//! NXR-KRONOS Capabilities
//! 
//! Capability vector and specifications for NXR-KRONOS

use crate::shared::{
    capability_spec::{CapabilityVector, CapabilitySpec, CapabilityDomain, CapabilityLevel, ResourceRequirements},
    model_identity::NxrModelId,
};

/// NXR-KRONOS Capabilities Manager
pub struct KronosCapabilities {
    /// Capability vector
    vector: CapabilityVector,
}

impl KronosCapabilities {
    /// Create new capabilities for NXR-KRONOS
    pub fn new() -> Self {
        let vector = Self::create_capability_vector();
        Self { vector }
    }

    /// Get capability vector
    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }

    /// Create capability vector for NXR-KRONOS
    fn create_capability_vector() -> CapabilityVector {
        CapabilityVector::new(NxrModelId::Kronos)
            // Knowledge capabilities - Expert level (primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Knowledge, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "knowledge_retrieval".to_string(),
                    "semantic_search".to_string(),
                    "knowledge_graph_management".to_string(),
                    "information_extraction".to_string(),
                    "knowledge_synthesis".to_string(),
                    "document_indexing".to_string(),
                ])
                .with_metric("knowledge_accuracy".to_string(), 0.978)
                .with_metric("retrieval_precision".to_string(), 0.94)
                .with_metric("indexing_speed".to_string(), 0.89)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(200),
                }))
            
            // Text capabilities - Expert level (primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "document_processing".to_string(),
                    "text_extraction".to_string(),
                    "semantic_understanding".to_string(),
                    "summarization".to_string(),
                    "text_generation".to_string(),
                ])
                .with_metric("text_accuracy".to_string(), 0.965)
                .with_metric("extraction_accuracy".to_string(), 0.93)
                .with_metric("summarization_quality".to_string(), 0.90)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: false,
                    max_latency_ms: Some(300),
                }))
            
            // Orchestration capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Orchestration, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "indexing_orchestration".to_string(),
                    "search_orchestration".to_string(),
                    "graph_orchestration".to_string(),
                    "pipeline_orchestration".to_string(),
                ])
                .with_metric("orchestration_accuracy".to_string(), 0.88)
                .with_metric("pipeline_efficiency".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(16.0),
                    requires_network: true,
                    max_latency_ms: Some(400),
                }))
            
            // Simulation capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Simulation, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "index_simulation".to_string(),
                    "search_simulation".to_string(),
                    "workflow_simulation".to_string(),
                ])
                .with_metric("simulation_accuracy".to_string(), 0.82)
                .with_metric("simulation_speed".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 12.0,
                    min_compute_units: 24,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(500),
                }))
            
            // Decision capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Decision, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "retrieval_decisions".to_string(),
                    "indexing_decisions".to_string(),
                    "synthesis_decisions".to_string(),
                ])
                .with_metric("decision_accuracy".to_string(), 0.83)
                .with_metric("decision_speed".to_string(), 0.86)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 12.0,
                    min_compute_units: 24,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(200),
                }))
            
            // Logic capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "graph_logic".to_string(),
                    "relation_logic".to_string(),
                    "extraction_logic".to_string(),
                ])
                .with_metric("logic_accuracy".to_string(), 0.81)
                .with_metric("reasoning_depth".to_string(), 5.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 8.0,
                    min_compute_units: 16,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(300),
                }))
            
            // Mathematics capabilities - Basic level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Mathematics, CapabilityLevel::Basic)
                .with_sub_capabilities(vec![
                    "vector_operations".to_string(),
                    "similarity_calculations".to_string(),
                ])
                .with_metric("math_accuracy".to_string(), 0.76)
                .with_metric("calculation_speed".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 4.0,
                    min_compute_units: 8,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(100),
                }))
            
            // Strategy capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Strategy, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "indexing_strategy".to_string(),
                    "search_strategy".to_string(),
                    "storage_strategy".to_string(),
                ])
                .with_metric("strategy_accuracy".to_string(), 0.82)
                .with_metric("optimization_efficiency".to_string(), 0.84)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 12.0,
                    min_compute_units: 24,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(300),
                }))
            
            // Security capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Security, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "access_control".to_string(),
                    "data_encryption".to_string(),
                    "privacy_preservation".to_string(),
                ])
                .with_metric("security_accuracy".to_string(), 0.83)
                .with_metric("protection_level".to_string(), 0.80)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 8.0,
                    min_compute_units: 16,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(200),
                }))
            
            // Monitoring capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Monitoring, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "index_monitoring".to_string(),
                    "search_monitoring".to_string(),
                    "graph_monitoring".to_string(),
                    "performance_monitoring".to_string(),
                ])
                .with_metric("monitoring_accuracy".to_string(), 0.90)
                .with_metric("alert_precision".to_string(), 0.87)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 12.0,
                    min_compute_units: 24,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(150),
                }))
            
            // Multimodal capabilities - Basic level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Multimodal, CapabilityLevel::Basic)
                .with_sub_capabilities(vec![
                    "document_visualization".to_string(),
                    "graph_visualization".to_string(),
                ])
                .with_metric("modality_accuracy".to_string(), 0.71)
                .with_metric("visualization_quality".to_string(), 0.68)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 8.0,
                    min_compute_units: 16,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(8.0),
                    requires_network: false,
                    max_latency_ms: Some(400),
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
        // Check that core capabilities are at expert level or higher
        let core_domains = vec![
            CapabilityDomain::Knowledge,
            CapabilityDomain::Text,
        ];

        for domain in core_domains {
            if !self.supports_capability(&domain, CapabilityLevel::Expert) {
                return Err(format!("Core capability {:?} not at expert level", domain));
            }
        }

        // Check that overall score is high enough
        if self.overall_score() < 0.85 {
            return Err("Overall capability score too low".to_string());
        }

        Ok(())
    }

    /// Update capability based on performance
    pub fn update_capability(&mut self, domain: CapabilityDomain, performance_score: f32) {
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

impl Default for KronosCapabilities {
    fn default() -> Self {
        Self::new()
    }
}
