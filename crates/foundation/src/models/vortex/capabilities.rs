//! NXR-VORTEX Capabilities
//! 
//! Capability vector and specifications for NXR-VORTEX

use std::collections::HashMap;

use crate::shared::{
    capability_spec::{CapabilityVector, CapabilitySpec, CapabilityDomain, CapabilityLevel, ResourceRequirements},
    model_identity::NxrModelId,
};

/// NXR-VORTEX Capabilities Manager
#[derive(Clone)]
pub struct _VortexCapabilities {
    /// Capability vector
    vector: CapabilityVector,
    /// Performance metrics
    performance_metrics: CapabilityPerformanceMetrics,
    /// Resource requirements
    resource_requirements: ResourceRequirements,
}

/// Capability Performance Metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapabilityPerformanceMetrics {
    /// Code generation accuracy
    pub code_generation_accuracy: f32,
    /// Debug success rate
    pub debug_success_rate: f32,
    /// Architecture analysis accuracy
    pub arch_analysis_accuracy: f32,
    /// Test generation quality
    pub test_generation_quality: f32,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Resource utilization
    pub resource_utilization: f32,
}

impl _VortexCapabilities {
    /// Create new capabilities for NXR-VORTEX
    pub fn new() -> Self {
        let vector = Self::create_capability_vector();
        let performance_metrics = CapabilityPerformanceMetrics {
            code_generation_accuracy: 0.972,
            debug_success_rate: 0.95,
            arch_analysis_accuracy: 0.94,
            test_generation_quality: 0.91,
            avg_response_time_ms: 450.0,
            resource_utilization: 0.75,
        };
        
        let resource_requirements = ResourceRequirements {
            min_memory_gb: 32.0,
            min_compute_units: 64,
            requires_gpu: true,
            min_gpu_memory_gb: Some(24.0),
            requires_network: true,
            max_latency_ms: Some(1000),
        };

        Self {
            vector,
            performance_metrics,
            resource_requirements,
        }
    }

    /// Get capability vector
    pub fn vector(&self) -> &CapabilityVector {
        &self.vector
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &CapabilityPerformanceMetrics {
        &self.performance_metrics
    }

    /// Get resource requirements
    pub fn resource_requirements(&self) -> &ResourceRequirements {
        &self.resource_requirements
    }

    /// Create capability vector for NXR-VORTEX
    fn create_capability_vector() -> CapabilityVector {
        CapabilityVector::new(NxrModelId::Vortex)
            // Code capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Code, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "code generation".to_string(),
                    "code analysis".to_string(),
                    "code optimization".to_string(),
                    "code refactoring".to_string(),
                    "debugging".to_string(),
                    "architecture analysis".to_string(),
                    "test generation".to_string(),
                    "documentation generation".to_string(),
                    "multi-language support".to_string(),
                    "security analysis".to_string(),
                    "performance optimization".to_string(),
                ])
                .with_metric("accuracy".to_string(), 0.972)
                .with_metric("reasoning_depth".to_string(), 9.0)
                .with_metric("context_window".to_string(), 2_000_000.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
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
                    "algorithmic reasoning".to_string(),
                    "computational complexity analysis".to_string(),
                ])
                .with_metric("proof_accuracy".to_string(), 0.995)
                .with_metric("inference_speed".to_string(), 950.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: false,
                    max_latency_ms: Some(800),
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
                    "algorithm analysis".to_string(),
                    "computational mathematics".to_string(),
                ])
                .with_metric("math_accuracy".to_string(), 0.98)
                .with_metric("problem_complexity".to_string(), 9.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(16.0),
                    requires_network: false,
                    max_latency_ms: Some(600),
                }))
            
            // Security capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Security, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "vulnerability analysis".to_string(),
                    "threat assessment".to_string(),
                    "security reasoning".to_string(),
                    "risk evaluation".to_string(),
                    "security protocol analysis".to_string(),
                    "penetration testing".to_string(),
                    "security audit".to_string(),
                    "compliance checking".to_string(),
                ])
                .with_metric("security_accuracy".to_string(), 0.97)
                .with_metric("threat_detection".to_string(), 0.94)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 20.0,
                    min_compute_units: 40,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(18.0),
                    requires_network: true,
                    max_latency_ms: Some(700),
                }))
            
            // Strategy capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Strategy, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "strategic planning".to_string(),
                    "game theory application".to_string(),
                    "risk assessment".to_string(),
                    "decision optimization".to_string(),
                    "long-term planning".to_string(),
                    "competitive analysis".to_string(),
                    "resource optimization".to_string(),
                ])
                .with_metric("strategy_accuracy".to_string(), 0.89)
                .with_metric("planning_horizon".to_string(), 180.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: true,
                    max_latency_ms: Some(900),
                }))
            
            // Knowledge capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Knowledge, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "knowledge synthesis".to_string(),
                    "fact verification".to_string(),
                    "source evaluation".to_string(),
                    "knowledge integration".to_string(),
                    "semantic understanding".to_string(),
                    "concept mapping".to_string(),
                    "domain expertise".to_string(),
                ])
                .with_metric("knowledge_accuracy".to_string(), 0.91)
                .with_metric("fact_coverage".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 28.0,
                    min_compute_units: 56,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(22.0),
                    requires_network: true,
                    max_latency_ms: Some(1200),
                }))
            
            // Creative capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Creative, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "creative synthesis".to_string(),
                    "novel concept generation".to_string(),
                    "artistic understanding".to_string(),
                    "creative problem solving".to_string(),
                    "innovation facilitation".to_string(),
                    "design thinking".to_string(),
                ])
                .with_metric("creativity_score".to_string(), 0.88)
                .with_metric("novelty_score".to_string(), 0.86)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 20.0,
                    min_compute_units: 40,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(18.0),
                    requires_network: false,
                    max_latency_ms: Some(800),
                }))
            
            // Orchestration capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Orchestration, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "multi-agent coordination".to_string(),
                    "task decomposition".to_string(),
                    "resource optimization".to_string(),
                    "workflow orchestration".to_string(),
                    "agent communication".to_string(),
                    "consensus building".to_string(),
                ])
                .with_metric("coordination_accuracy".to_string(), 0.92)
                .with_metric("agent_efficiency".to_string(), 0.89)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: true,
                    max_latency_ms: Some(800),
                }))
            
            // Self-improvement capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::SelfImprovement, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "self-reflection".to_string(),
                    "performance monitoring".to_string(),
                    "strategy adaptation".to_string(),
                    "learning optimization".to_string(),
                    "meta-learning".to_string(),
                ])
                .with_metric("improvement_rate".to_string(), 0.75)
                .with_metric("adaptation_speed".to_string(), 0.72)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(16.0),
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
                .with_metric("edge_efficiency".to_string(), 0.65)
                .with_metric("resource_usage".to_string(), 0.7)
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
            CapabilityDomain::Code,
            CapabilityDomain::Logic,
        ];

        for domain in core_domains {
            if !self.supports_capability(&domain, CapabilityLevel::Transcendent) {
                return Err(format!("Core capability {:?} not at transcendent level", domain));
            }
        }

        // Check that overall score is high enough
        if self.overall_score() < 0.85 {
            return Err("Overall capability score too low".to_string());
        }

        // Validate performance metrics
        if self.performance_metrics.code_generation_accuracy < 0.9 {
            return Err("Code generation accuracy too low".to_string());
        }

        if self.performance_metrics.debug_success_rate < 0.8 {
            return Err("Debug success rate too low".to_string());
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
            performance_metrics: self.performance_metrics.clone(),
            resource_requirements: self.resource_requirements.clone(),
        }
    }

    /// Get detailed capability breakdown
    pub fn detailed_breakdown(&self) -> DetailedCapabilityBreakdown {
        let mut breakdown = DetailedCapabilityBreakdown::new();

        for (domain, capability) in &self.vector.capabilities {
            let capability_detail = CapabilityDetail {
                domain: domain.clone(),
                level: capability.level.clone(),
                score: capability.score(),
                sub_capabilities: capability.sub_capabilities.clone(),
                metrics: capability.metrics.clone(),
                resource_requirements: capability.resource_requirements.clone(),
            };
            
            breakdown.capabilities.insert(domain.clone(), capability_detail);
        }

        breakdown
    }

    /// Compare with another capability set
    pub fn compare_with(&self, other: &_VortexCapabilities) -> CapabilityComparison {
        let mut comparison = CapabilityComparison::new();

        for domain in &self.vector.specializations {
            let self_score = self.get_capability_score(domain);
            let other_score = other.get_capability_score(domain);
            
            comparison.domain_comparisons.insert(
                domain.clone(),
                DomainComparison {
                    domain: domain.clone(),
                    self_score,
                    other_score,
                    difference: self_score - other_score,
                    self_level: self.vector.get_capability(domain).map(|c| c.level.clone()),
                    other_level: other.vector.get_capability(domain).map(|c| c.level.clone()),
                }
            );
        }

        comparison.overall_difference = self.overall_score() - other.overall_score();
        comparison

    }

    /// Update performance metrics
    pub fn update_performance_metrics(&mut self, new_metrics: CapabilityPerformanceMetrics) {
        self.performance_metrics = new_metrics;
    }

    /// Get capability optimization suggestions
    pub fn get_optimization_suggestions(&self) -> Vec<_OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Check for underperforming capabilities
        for (domain, capability) in &self.vector.capabilities {
            if capability.score() < 0.8 {
                suggestions.push(_OptimizationSuggestion {
                    domain: domain.clone(),
                    suggestion_type: SuggestionType::ImproveCapability,
                    description: format!("Capability {:?} is underperforming with score {:.2}", domain, capability.score()),
                    priority: if capability.score() < 0.6 { SuggestionPriority::High } else { SuggestionPriority::Medium },
                    estimated_improvement: 0.15,
                    resource_cost: ResourceCost::Medium,
                });
            }
        }

        // Check resource usage
        if self.performance_metrics.resource_utilization > 0.9 {
            suggestions.push(_OptimizationSuggestion {
                domain: CapabilityDomain::Edge,
                suggestion_type: SuggestionType::OptimizeResources,
                description: "Resource utilization is too high, consider optimization".to_string(),
                priority: SuggestionPriority::High,
                estimated_improvement: 0.2,
                resource_cost: ResourceCost::Low,
            });
        }

        suggestions
    }

    /// Apply capability improvement
    pub fn apply_improvement(&self, improvements: &HashMap<CapabilityDomain, f32>) -> _VortexCapabilities {
        let mut new_capabilities = self.clone();
        
        for (domain, improvement) in improvements {
            if let Some(capability) = new_capabilities.vector.capabilities.get_mut(domain) {
                // Simulate improvement by increasing score
                let current_score = capability.score();
                let new_score = (current_score + improvement).min(1.0);
                
                // Update capability metrics
                capability.metrics.insert("improved_score".to_string(), new_score);
            }
        }

        new_capabilities
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
    /// Performance metrics
    pub performance_metrics: CapabilityPerformanceMetrics,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
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

    /// Get efficiency rating
    pub fn efficiency_rating(&self) -> f32 {
        let capability_score = self.overall_score;
        let resource_efficiency = 1.0 - (self.performance_metrics.resource_utilization - 0.5).abs();
        
        (capability_score + resource_efficiency) / 2.0
    }

    /// Get specialization score
    pub fn specialization_score(&self) -> f32 {
        if self.specializations.is_empty() {
            0.0
        } else {
            let total_score: f32 = self.specializations
                .iter()
                .map(|domain| self.get_domain_score(domain))
                .sum();
            
            total_score / self.specializations.len() as f32
        }
    }

    /// Get domain score
    fn get_domain_score(&self, domain: &CapabilityDomain) -> f32 {
        match domain {
            CapabilityDomain::Code => 0.92,
            CapabilityDomain::Logic => 0.88,
            CapabilityDomain::Mathematics => 0.82,
            CapabilityDomain::Strategy => 0.85,
            CapabilityDomain::Knowledge => 0.80,
            CapabilityDomain::Text => 0.78,
            CapabilityDomain::Creative => 0.75,
            CapabilityDomain::Security => 0.73,
            CapabilityDomain::Decision => 0.85,
            _ => 0.75,
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

/// Detailed capability breakdown
#[derive(Debug, Clone)]
pub struct DetailedCapabilityBreakdown {
    pub capabilities: HashMap<CapabilityDomain, CapabilityDetail>,
}

impl DetailedCapabilityBreakdown {
    pub fn new() -> Self {
        Self {
            capabilities: HashMap::new(),
        }
    }

    /// Get capability by domain
    pub fn get_capability(&self, domain: &CapabilityDomain) -> Option<&CapabilityDetail> {
        self.capabilities.get(domain)
    }

    /// Get top capabilities
    pub fn get_top_capabilities(&self, limit: usize) -> Vec<(&CapabilityDomain, &CapabilityDetail)> {
        let mut capabilities: Vec<_> = self.capabilities.iter().collect();
        capabilities.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));
        capabilities.into_iter().take(limit).collect()
    }

    /// Get capability statistics
    pub fn get_statistics(&self) -> CapabilityStatistics {
        let mut stats = CapabilityStatistics::new();
        
        for capability in self.capabilities.values() {
            match capability.level {
                CapabilityLevel::Transcendent => stats.transcendent_count += 1,
                CapabilityLevel::Master => stats.master_count += 1,
                CapabilityLevel::Expert => stats.expert_count += 1,
                CapabilityLevel::Advanced => stats.advanced_count += 1,
                CapabilityLevel::Intermediate => stats.intermediate_count += 1,
                CapabilityLevel::Basic => stats.basic_count += 1,
                CapabilityLevel::None => {}
            }
        }

        stats.total_capabilities = self.capabilities.len();
        stats.average_score = self.capabilities.values().map(|c| c.score).sum::<f32>() / self.capabilities.len() as f32;

        stats
    }
}

/// Capability detail
#[derive(Debug, Clone)]
pub struct CapabilityDetail {
    pub domain: CapabilityDomain,
    pub level: CapabilityLevel,
    pub score: f32,
    pub sub_capabilities: Vec<String>,
    pub metrics: HashMap<String, f32>,
    pub resource_requirements: ResourceRequirements,
}

/// Capability statistics
#[derive(Debug, Clone)]
pub struct CapabilityStatistics {
    pub total_capabilities: usize,
    pub transcendent_count: usize,
    pub master_count: usize,
    pub expert_count: usize,
    pub advanced_count: usize,
    pub intermediate_count: usize,
    pub basic_count: usize,
    pub average_score: f32,
}

impl CapabilityStatistics {
    pub fn new() -> Self {
        Self {
            total_capabilities: 0,
            transcendent_count: 0,
            master_count: 0,
            expert_count: 0,
            advanced_count: 0,
            intermediate_count: 0,
            basic_count: 0,
            average_score: 0.0,
        }
    }
}

/// Capability comparison
#[derive(Debug, Clone)]
pub struct CapabilityComparison {
    pub domain_comparisons: HashMap<CapabilityDomain, DomainComparison>,
    pub overall_difference: f32,
}

impl CapabilityComparison {
    pub fn new() -> Self {
        Self {
            domain_comparisons: HashMap::new(),
            overall_difference: 0.0,
        }
    }

    /// Get improvement opportunities
    pub fn get_improvement_opportunities(&self) -> Vec<ImprovementOpportunity> {
        let mut opportunities = Vec::new();

        for (domain, comparison) in &self.domain_comparisons {
            if comparison.difference < -0.1 {
                opportunities.push(ImprovementOpportunity {
                    domain: domain.clone(),
                    gap: -comparison.difference,
                    priority: if comparison.difference < -0.3 { OpportunityPriority::High } else { OpportunityPriority::Medium },
                    potential_improvement: 0.2,
                });
            }
        }

        opportunities.sort_by(|a, b| b.gap.partial_cmp(&a.gap).unwrap_or(std::cmp::Ordering::Equal));
        opportunities
    }
}

/// Domain comparison
#[derive(Debug, Clone)]
pub struct DomainComparison {
    pub domain: CapabilityDomain,
    pub self_score: f32,
    pub other_score: f32,
    pub difference: f32,
    pub self_level: Option<CapabilityLevel>,
    pub other_level: Option<CapabilityLevel>,
}

/// Improvement opportunity
#[derive(Debug, Clone)]
pub struct ImprovementOpportunity {
    pub domain: CapabilityDomain,
    pub gap: f32,
    pub priority: OpportunityPriority,
    pub potential_improvement: f32,
}

/// Opportunity priority
#[derive(Debug, Clone, PartialEq)]
pub enum OpportunityPriority {
    High,
    Medium,
    Low,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct _OptimizationSuggestion {
    pub domain: CapabilityDomain,
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub priority: SuggestionPriority,
    pub estimated_improvement: f32,
    pub resource_cost: ResourceCost,
}

/// Suggestion type
#[derive(Debug, Clone)]
pub enum SuggestionType {
    ImproveCapability,
    OptimizeResources,
    UpgradeInfrastructure,
    AddTraining,
    RefineModel,
}

/// Suggestion priority
#[derive(Debug, Clone, PartialEq)]
pub enum SuggestionPriority {
    High,
    Medium,
    Low,
}

/// Resource cost
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceCost {
    Low,
    Medium,
    High,
}

impl Default for _VortexCapabilities {
    fn default() -> Self {
        Self::new()
    }
}
