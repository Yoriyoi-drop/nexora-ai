//! NXR-CIPHER Capabilities
//! 
//! Capability vector and specifications for NXR-CIPHER

use crate::shared::{
    capability_spec::{CapabilityVector, CapabilitySpec, CapabilityDomain, CapabilityLevel, ResourceRequirements},
    model_identity::NxrModelId,
};

/// NXR-CIPHER Capabilities Manager
pub struct CipherCapabilities {
    /// Capability vector
    vector: CapabilityVector,
}

impl CipherCapabilities {
    /// Create new capabilities for NXR-CIPHER
    pub fn new() -> Self {
        let vector = Self::create_capability_vector();
        Self { vector }
    }

    /// Get capability vector
    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }

    /// Create capability vector for NXR-CIPHER
    fn create_capability_vector() -> CapabilityVector {
        CapabilityVector::new(NxrModelId::Cipher)
            // Security capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Security, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "vulnerability_assessment".to_string(),
                    "penetration_testing".to_string(),
                    "threat_hunting".to_string(),
                    "incident_response".to_string(),
                    "security_audit".to_string(),
                    "malware_analysis".to_string(),
                    "forensics".to_string(),
                ])
                .with_metric("security_accuracy".to_string(), 0.986)
                .with_metric("vulnerability_detection".to_string(), 0.97)
                .with_metric("threat_identification".to_string(), 0.94)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(800),
                }))
            
            // Strategy capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Strategy, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "security_strategy".to_string(),
                    "risk_management".to_string(),
                    "incident_response_planning".to_string(),
                    "threat_modeling".to_string(),
                ])
                .with_metric("strategy_accuracy".to_string(), 0.89)
                .with_metric("risk_assessment".to_string(), 0.92)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: true,
                    max_latency_ms: Some(1000),
                }))
            
            // Knowledge capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Knowledge, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "cve_database".to_string(),
                    "threat_intelligence".to_string(),
                    "security_best_practices".to_string(),
                    "compliance_standards".to_string(),
                    "attack_patterns".to_string(),
                ])
                .with_metric("knowledge_coverage".to_string(), 0.95)
                .with_metric("intelligence_accuracy".to_string(), 0.91)
                .with_metric("compliance_coverage".to_string(), 0.88)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 64.0,
                    min_compute_units: 128,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(32.0),
                    requires_network: true,
                    max_latency_ms: Some(1500),
                }))
            
            // Text capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Text, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "security_report_generation".to_string(),
                    "vulnerability_description".to_string(),
                    "threat_explanation".to_string(),
                    "technical_documentation".to_string(),
                ])
                .with_metric("text_accuracy".to_string(), 0.91)
                .with_metric("report_quality".to_string(), 0.88)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(400),
                }))
            
            // Logic capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "attack_logic_analysis".to_string(),
                    "vulnerability_reasoning".to_string(),
                    "security_logic".to_string(),
                ])
                .with_metric("logic_accuracy".to_string(), 0.85)
                .with_metric("reasoning_depth".to_string(), 5.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(500),
                }))
            
            // Mathematics capabilities - Basic level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Mathematics, CapabilityLevel::Basic)
                .with_sub_capabilities(vec![
                    "risk_calculation".to_string(),
                    "statistical_analysis".to_string(),
                ])
                .with_metric("math_accuracy".to_string(), 0.78)
                .with_metric("calculation_speed".to_string(), 0.82)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 8.0,
                    min_compute_units: 16,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(300),
                }))
            
            // Monitoring capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Monitoring, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "real_time_monitoring".to_string(),
                    "anomaly_detection".to_string(),
                    "alert_generation".to_string(),
                    "log_analysis".to_string(),
                    "traffic_analysis".to_string(),
                ])
                .with_metric("monitoring_accuracy".to_string(), 0.94)
                .with_metric("detection_precision".to_string(), 0.91)
                .with_metric("alert_response_time".to_string(), 0.89)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: true,
                    max_latency_ms: Some(200),
                }))
            
            // Simulation capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Simulation, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "attack_simulation".to_string(),
                    "vulnerability_simulation".to_string(),
                    "threat_scenario_simulation".to_string(),
                    "zero_day_simulation".to_string(),
                ])
                .with_metric("simulation_accuracy".to_string(), 0.87)
                .with_metric("scenario_diversity".to_string(), 0.85)
                .with_metric("prediction_confidence".to_string(), 0.82)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(1500),
                }))
            
            // Decision capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Decision, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "security_decision_support".to_string(),
                    "incident_response_decisions".to_string(),
                    "risk_mitigation_decisions".to_string(),
                    "compliance_decisions".to_string(),
                ])
                .with_metric("decision_accuracy".to_string(), 0.88)
                .with_metric("response_effectiveness".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: true,
                    max_latency_ms: Some(600),
                }))
            
            // Orchestration capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Orchestration, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "agent_coordination".to_string(),
                    "scan_orchestration".to_string(),
                    "response_orchestration".to_string(),
                ])
                .with_metric("coordination_accuracy".to_string(), 0.84)
                .with_metric("orchestration_efficiency".to_string(), 0.82)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: true,
                    max_latency_ms: Some(800),
                }))
            
            // Multimodal capabilities - Basic level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Multimodal, CapabilityLevel::Basic)
                .with_sub_capabilities(vec![
                    "network_diagram_generation".to_string(),
                    "threat_visualization".to_string(),
                ])
                .with_metric("modality_accuracy".to_string(), 0.72)
                .with_metric("visualization_quality".to_string(), 0.70)
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
            CapabilityDomain::Security,
            CapabilityDomain::Knowledge,
            CapabilityDomain::Monitoring,
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

impl Default for CipherCapabilities {
    fn default() -> Self {
        Self::new()
    }
}
