//! NXR-SPECTRA Capabilities
//! 
//! Capability vector and specifications for NXR-SPECTRA

use std::collections::HashMap;

use crate::shared::{
    capability_spec::{CapabilityVector, CapabilitySpec, CapabilityDomain, CapabilityLevel, ResourceRequirements},
    model_identity::NxrModelId,
};

/// NXR-SPECTRA Capabilities Manager
#[derive(Clone)]
pub struct SpectraCapabilities {
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
    /// Creative generation accuracy
    pub creative_generation_accuracy: f32,
    /// Style adaptation accuracy
    pub style_adaptation_accuracy: f32,
    /// Multimodal synthesis quality
    pub multimodal_synthesis_quality: f32,
    /// Innovation generation quality
    pub innovation_generation_quality: f32,
    /// Artistic quality score
    pub artistic_quality_score: f32,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Resource utilization
    pub resource_utilization: f32,
}

impl SpectraCapabilities {
    /// Create new capabilities for NXR-SPECTRA
    pub fn new() -> Self {
        let vector = Self::create_capability_vector();
        let performance_metrics = CapabilityPerformanceMetrics {
            creative_generation_accuracy: 0.948,
            style_adaptation_accuracy: 0.91,
            multimodal_synthesis_quality: 0.89,
            innovation_generation_quality: 0.87,
            artistic_quality_score: 0.93,
            avg_response_time_ms: 420.0,
            resource_utilization: 0.78,
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

    /// Create capability vector for NXR-SPECTRA
    fn create_capability_vector() -> CapabilityVector {
        CapabilityVector::new(NxrModelId::Spectra)
            // Creative capabilities - Transcendent level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Creative, CapabilityLevel::Transcendent)
                .with_sub_capabilities(vec![
                    "creative synthesis".to_string(),
                    "artistic generation".to_string(),
                    "style adaptation".to_string(),
                    "concept generation".to_string(),
                    "multimodal creativity".to_string(),
                    "cross-modal synthesis".to_string(),
                    "innovative thinking".to_string(),
                    "aesthetic assessment".to_string(),
                    "creative collaboration".to_string(),
                    "artistic expression".to_string(),
                ])
                .with_metric("creativity_score".to_string(), 0.948)
                .with_metric("originality_score".to_string(), 0.91)
                .with_metric("context_window".to_string(), 1_000_000.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 32.0,
                    min_compute_units: 64,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(24.0),
                    requires_network: true,
                    max_latency_ms: Some(1000),
                }))
            
            // Multimedia capabilities - Master level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Multimedia, CapabilityLevel::Master)
                .with_sub_capabilities(vec![
                    "visual generation".to_string(),
                    "audio generation".to_string(),
                    "video synthesis".to_string(),
                    "3D content creation".to_string(),
                    "interactive media".to_string(),
                    "multimodal fusion".to_string(),
                    "cross-modal attention".to_string(),
                    "media adaptation".to_string(),
                ])
                .with_metric("multimedia_quality".to_string(), 0.94)
                .with_metric("synthesis_accuracy".to_string(), 0.89)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 28.0,
                    min_compute_units: 56,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(22.0),
                    requires_network: true,
                    max_latency_ms: Some(1200),
                }))
            
            // Style capabilities - Expert level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Style, CapabilityLevel::Expert)
                .with_sub_capabilities(vec![
                    "style recognition".to_string(),
                    "style adaptation".to_string(),
                    "style synthesis".to_string(),
                    "style learning".to_string(),
                    "cross-style creativity".to_string(),
                    "historical style awareness".to_string(),
                    "cultural style understanding".to_string(),
                    "style evolution".to_string(),
                ])
                .with_metric("style_accuracy".to_string(), 0.91)
                .with_metric("adaptation_speed".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 24.0,
                    min_compute_units: 48,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(20.0),
                    requires_network: true,
                    max_latency_ms: Some(800),
                }))
            
            // Innovation capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Innovation, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "concept generation".to_string(),
                    "novelty evaluation".to_string(),
                    "innovation synthesis".to_string(),
                    "cross-domain innovation".to_string(),
                    "disruptive thinking".to_string(),
                    "creative problem solving".to_string(),
                    "idea generation".to_string(),
                ])
                .with_metric("innovation_score".to_string(), 0.87)
                .with_metric("novelty_score".to_string(), 0.83)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 20.0,
                    min_compute_units: 40,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(18.0),
                    requires_network: false,
                    max_latency_ms: Some(600),
                }))
            
            // Communication capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Communication, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "artistic communication".to_string(),
                    "creative expression".to_string(),
                    "visual storytelling".to_string(),
                    "multimedia communication".to_string(),
                    "aesthetic communication".to_string(),
                    "creative collaboration".to_string(),
                ])
                .with_metric("communication_quality".to_string(), 0.89)
                .with_metric("expression_clarity".to_string(), 0.86)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 16.0,
                    min_compute_units: 32,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(16.0),
                    requires_network: false,
                    max_latency_ms: Some(500),
                }))
            
            // Logic capabilities - Intermediate level (not primary focus)
            .with_capability(CapabilitySpec::new(CapabilityDomain::Logic, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "creative reasoning".to_string(),
                    "artistic logic".to_string(),
                    "aesthetic reasoning".to_string(),
                ])
                .with_metric("logic_accuracy".to_string(), 0.78)
                .with_metric("reasoning_depth".to_string(), 5.0)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 12.0,
                    min_compute_units: 24,
                    requires_gpu: false,
                    min_gpu_memory_gb: None,
                    requires_network: false,
                    max_latency_ms: Some(400),
                }))
            
            // Knowledge capabilities - Advanced level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Knowledge, CapabilityLevel::Advanced)
                .with_sub_capabilities(vec![
                    "artistic knowledge".to_string(),
                    "style knowledge".to_string(),
                    "cultural knowledge".to_string(),
                    "creative techniques".to_string(),
                    "art history".to_string(),
                    "design principles".to_string(),
                ])
                .with_metric("knowledge_accuracy".to_string(), 0.88)
                .with_metric("knowledge_coverage".to_string(), 0.85)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 18.0,
                    min_compute_units: 36,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(17.0),
                    requires_network: true,
                    max_latency_ms: Some(900),
                }))
            
            // Support capabilities - Intermediate level
            .with_capability(CapabilitySpec::new(CapabilityDomain::Support, CapabilityLevel::Intermediate)
                .with_sub_capabilities(vec![
                    "creative guidance".to_string(),
                    "artistic advice".to_string(),
                    "style recommendations".to_string(),
                    "creative inspiration".to_string(),
                ])
                .with_metric("support_quality".to_string(), 0.82)
                .with_metric("guidance_effectiveness".to_string(), 0.79)
                .with_resources(ResourceRequirements {
                    min_memory_gb: 14.0,
                    min_compute_units: 28,
                    requires_gpu: true,
                    min_gpu_memory_gb: Some(14.0),
                    requires_network: false,
                    max_latency_ms: Some(600),
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
            CapabilityDomain::Creative,
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
        if self.performance_metrics.creative_generation_accuracy < 0.9 {
            return Err("Creative generation accuracy too low".to_string());
        }

        if self.performance_metrics.style_adaptation_accuracy < 0.8 {
            return Err("Style adaptation accuracy too low".to_string());
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
    pub fn compare_with(&self, other: &SpectraCapabilities) -> CapabilityComparison {
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
    pub fn get_optimization_suggestions(&self) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Check for underperforming capabilities
        for (domain, capability) in &self.vector.capabilities {
            if capability.score() < 0.8 {
                suggestions.push(OptimizationSuggestion {
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
        if self.performance_metrics.resource_utilization > 0.85 {
            suggestions.push(OptimizationSuggestion {
                domain: CapabilityDomain::Creative,
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
    pub fn apply_improvement(&self, improvements: &HashMap<CapabilityDomain, f32>) -> SpectraCapabilities {
        let mut new_capabilities = self.clone();
        
        for (domain, improvement) in improvements {
            if let Some(capability) = new_capabilities.vector.capabilities.get_mut(domain) {
                // Simulate improvement by increasing score
                let current_score = capability.score();
                let new_score = (current_score + improvement).min(1.0_f32);
                
                // Update capability metrics
                capability.metrics.insert("improved_score".to_string(), new_score);
            }
        }

        new_capabilities
    }

    /// Get creative intelligence metrics
    pub fn get_creative_intelligence_metrics(&self) -> CreativeIntelligenceMetrics {
        CreativeIntelligenceMetrics {
            creativity_score: self.performance_metrics.creative_generation_accuracy,
            style_adaptation: self.performance_metrics.style_adaptation_accuracy,
            multimodal_synthesis: self.performance_metrics.multimodal_synthesis_quality,
            innovation_generation: self.performance_metrics.innovation_generation_quality,
            artistic_quality: self.performance_metrics.artistic_quality_score,
            response_efficiency: ((1000.0 / self.performance_metrics.avg_response_time_ms).min(1.0)) as f32,
        }
    }

    /// Get multimedia capabilities breakdown
    pub fn get_multimedia_capabilities(&self) -> MultimediaCapabilitiesBreakdown {
        let mut breakdown = MultimediaCapabilitiesBreakdown::new();

        if let Some(multimedia_capability) = self.vector.get_capability(&CapabilityDomain::Multimedia) {
            breakdown.supported_modalities = multimedia_capability.sub_capabilities.clone();
            breakdown.synthesis_quality = self.performance_metrics.multimodal_synthesis_quality;
            breakdown.fusion_accuracy = multimedia_capability.metrics.get("synthesis_accuracy").copied().unwrap_or(0.0);
        }

        breakdown
    }

    /// Get style adaptation capabilities
    pub fn get_style_adaptation_capabilities(&self) -> StyleAdaptationCapabilities {
        let mut capabilities = StyleAdaptationCapabilities::new();

        if let Some(style_capability) = self.vector.get_capability(&CapabilityDomain::Style) {
            capabilities.adaptation_accuracy = self.performance_metrics.style_adaptation_accuracy;
            capabilities.adaptation_methods = style_capability.sub_capabilities.clone();
            capabilities.adaptation_speed = style_capability.metrics.get("adaptation_speed").copied().unwrap_or(0.0);
        }

        capabilities
    }

    /// Get innovation capabilities breakdown
    pub fn get_innovation_capabilities(&self) -> InnovationCapabilitiesBreakdown {
        let mut breakdown = InnovationCapabilitiesBreakdown::new();

        if let Some(innovation_capability) = self.vector.get_capability(&CapabilityDomain::Innovation) {
            breakdown.innovation_quality = self.performance_metrics.innovation_generation_quality;
            breakdown.innovation_methods = innovation_capability.sub_capabilities.clone();
            breakdown.novelty_score = innovation_capability.metrics.get("novelty_score").copied().unwrap_or(0.0);
        }

        breakdown
    }

    /// Get creative generation capabilities
    pub fn get_creative_generation_capabilities(&self) -> CreativeGenerationCapabilities {
        let mut capabilities = CreativeGenerationCapabilities::new();

        if let Some(creative_capability) = self.vector.get_capability(&CapabilityDomain::Creative) {
            capabilities.generation_accuracy = self.performance_metrics.creative_generation_accuracy;
            capabilities.generation_methods = creative_capability.sub_capabilities.clone();
            capabilities.creativity_score = creative_capability.metrics.get("creativity_score").copied().unwrap_or(0.0);
        }

        capabilities
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
            CapabilityDomain::Creative => 0.92,
            CapabilityDomain::Vision => 0.88,
            CapabilityDomain::Multimodal => 0.90,
            CapabilityDomain::Text => 0.85,
            CapabilityDomain::Knowledge => 0.87,
            CapabilityDomain::Simulation => 0.82,
            CapabilityDomain::Decision => 0.80,
            CapabilityDomain::Strategy => 0.78,
            CapabilityDomain::Emotional => 0.76,
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
pub struct OptimizationSuggestion {
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

/// Creative intelligence metrics
#[derive(Debug, Clone)]
pub struct CreativeIntelligenceMetrics {
    pub creativity_score: f32,
    pub style_adaptation: f32,
    pub multimodal_synthesis: f32,
    pub innovation_generation: f32,
    pub artistic_quality: f32,
    pub response_efficiency: f32,
}

/// Multimedia capabilities breakdown
#[derive(Debug, Clone)]
pub struct MultimediaCapabilitiesBreakdown {
    pub supported_modalities: Vec<String>,
    pub synthesis_quality: f32,
    pub fusion_accuracy: f32,
}

impl MultimediaCapabilitiesBreakdown {
    pub fn new() -> Self {
        Self {
            supported_modalities: Vec::new(),
            synthesis_quality: 0.0,
            fusion_accuracy: 0.0,
        }
    }
}

/// Style adaptation capabilities
#[derive(Debug, Clone)]
pub struct StyleAdaptationCapabilities {
    pub adaptation_accuracy: f32,
    pub adaptation_methods: Vec<String>,
    pub adaptation_speed: f32,
}

impl StyleAdaptationCapabilities {
    pub fn new() -> Self {
        Self {
            adaptation_accuracy: 0.0,
            adaptation_methods: Vec::new(),
            adaptation_speed: 0.0,
        }
    }
}

/// Innovation capabilities breakdown
#[derive(Debug, Clone)]
pub struct InnovationCapabilitiesBreakdown {
    pub innovation_quality: f32,
    pub innovation_methods: Vec<String>,
    pub novelty_score: f32,
}

impl InnovationCapabilitiesBreakdown {
    pub fn new() -> Self {
        Self {
            innovation_quality: 0.0,
            innovation_methods: Vec::new(),
            novelty_score: 0.0,
        }
    }
}

/// Creative generation capabilities
#[derive(Debug, Clone)]
pub struct CreativeGenerationCapabilities {
    pub generation_accuracy: f32,
    pub generation_methods: Vec<String>,
    pub creativity_score: f32,
}

impl CreativeGenerationCapabilities {
    pub fn new() -> Self {
        Self {
            generation_accuracy: 0.0,
            generation_methods: Vec::new(),
            creativity_score: 0.0,
        }
    }
}

impl Default for SpectraCapabilities {
    fn default() -> Self {
        Self::new()
    }
}
