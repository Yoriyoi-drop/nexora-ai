//! NXR-AXIOM Capabilities
//! 
//! Capability vector and specifications for NXR-AXIOM

use crate::shared::{
    capability_spec::{CapabilityVector, CapabilitySpec, CapabilityDomain, CapabilityLevel, ResourceRequirements},
    model_identity::NxrModelId,
};

/// NXR-AXIOM Capabilities Manager
pub struct AxiomCapabilities {
    /// Capability vector
    vector: CapabilityVector,
}

impl AxiomCapabilities {
    /// Create new capabilities for NXR-AXIOM
    pub fn new() -> Self {
        let vector = Self::create_capability_vector();
        Self { vector }
    }

    /// Get capability vector
    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }

    /// Create capability vector for NXR-AXIOM
    fn create_capability_vector() -> CapabilityVector {
        CapabilityVector::new(NxrModelId::Axiom)
            // Logic capabilities - Master level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Master)
                .with_sub_capabilities(vec![
                    "formal logic".to_string(),
                    "propositional logic".to_string(),
                    "first-order logic".to_string(),
                    "higher-order logic".to_string(),
                    "modal logic".to_string(),
                    "temporal logic".to_string(),
                    "proof generation".to_string(),
                    "theorem proving".to_string(),
                ])
                .with_metric("logical_accuracy".to_string(), 0.94)
                .with_metric("reasoning_depth".to_string(), 8.0)
                .with_metric("inference_speed".to_string(), 750.0)
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
                    "arithmetic operations".to_string(),
                    "algebraic manipulation".to_string(),
                    "geometric reasoning".to_string(),
                    "calculus operations".to_string(),
                    "statistical analysis".to_string(),
                    "number theory".to_string(),
                    "combinatorics".to_string(),
                    "graph theory".to_string(),
                ])
                .with_metric("math_accuracy".to_string(), 0.92)
                .with_metric("problem_complexity".to_string(), 9.0)
                .with_metric("symbolic_computation".to_string(), 0.88)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: false,
                    max_latency_ms: Some(600),
                }))
            
            // Strategy capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Strategy, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "strategic planning".to_string(),
                    "risk assessment".to_string(),
                    "decision optimization".to_string(),
                    "game theory application".to_string(),
                    "long-term planning".to_string(),
                    "scenario analysis".to_string(),
                ])
                .with_metric("strategy_accuracy".to_string(), 0.89)
                .with_metric("planning_horizon".to_string(), 365.0)
                .with_metric("risk_detection".to_string(), 0.87)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(1200),
                }))
            
            // Knowledge capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Knowledge, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "theorem knowledge".to_string(),
                    "axiom management".to_string(),
                    "knowledge synthesis".to_string(),
                    "formal system mastery".to_string(),
                    "proof verification".to_string(),
                ])
                .with_metric("knowledge_accuracy".to_string(), 0.91)
                .with_metric("theorem_coverage".to_string(), 0.85)
                .with_metric("verification_accuracy".to_string(), 0.96)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 64.0,
                    min_compute_units: 128,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(32.0),
                    requires_network: true,
                    max_latency_ms: Some(1500),
                }))
            
            // Security capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Security, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "risk analysis".to_string(),
                    "vulnerability assessment".to_string(),
                    "audit compliance".to_string(),
                    "threat detection".to_string(),
                ])
                .with_metric("security_accuracy".to_string(), 0.87)
                .with_metric("risk_detection".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(400),
                }))
            
            // Text capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "formal proof writing".to_string(),
                    "mathematical notation".to_string(),
                    "logical expression".to_string(),
                    "technical documentation".to_string(),
                ])
                .with_metric("text_accuracy".to_string(), 0.88)
                .with_metric("formal_correctness".to_string(), 0.92)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(300),
                }))
            
            // Simulation capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Simulation, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "world simulation".to_string(),
                    "scenario generation".to_string(),
                    "future prediction".to_string(),
                    "trend analysis".to_string(),
                    "monte_carlo_simulation".to_string(),
                ])
                .with_metric("simulation_accuracy".to_string(), 0.86)
                .with_metric("prediction_horizon".to_string(), 100.0)
                .with_metric("scenario_diversity".to_string(), 0.88)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 48.0,
                    min_compute_units: 96,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(32.0),
                    requires_network: true,
                    max_latency_ms: Some(2000),
                }))
            
            // Decision capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Decision, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "decision optimization".to_string(),
                    "multi_criteria_analysis".to_string(),
                    "cost_benefit_analysis".to_string(),
                    "sensitivity_analysis".to_string(),
                    "decision explanation".to_string(),
                ])
                .with_metric("decision_accuracy".to_string(), 0.89)
                .with_metric("explanation_quality".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: false,
                    max_latency_ms: Some(800),
                }))
            
            // Orchestration capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Orchestration, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "agent coordination".to_string(),
                    "task decomposition".to_string(),
                    "workflow orchestration".to_string(),
                    "resource optimization".to_string(),
                ])
                .with_metric("coordination_accuracy".to_string(), 0.87)
                .with_metric("agent_efficiency".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(1000),
                }))
            
            // Monitoring capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Monitoring, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "audit compliance".to_string(),
                    "performance monitoring".to_string(),
                    "alert generation".to_string(),
                    "violation detection".to_string(),
                    "report generation".to_string(),
                ])
                .with_metric("monitoring_accuracy".to_string(), 0.91)
                .with_metric("alert_precision".to_string(), 0.88)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(300),
                }))
            
            // Multimodal capabilities - Basic level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Multimodal, CapabilityLevel::Basic)
                .with_sub_capabilities(vec![
                    "text_diagram_generation".to_string(),
                    "mathematical_visualization".to_string(),
                ])
                .with_metric("modality_accuracy".to_string(), 0.75)
                .with_metric("visualization_quality".to_string(), 0.72)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 8.0,
                    min_compute_units: 16,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(8.0),
                    requires_network: false,
                    max_latency_ms: Some(500),
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
            CapabilityDomain::Logic,
            CapabilityDomain::Mathematics,
            CapabilityDomain::Strategy,
            CapabilityDomain::Knowledge,
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

impl Default for AxiomCapabilities {
    fn default() -> Self {
        Self::new()
    }
}
