//! Expert Domains — shared + tier-specific expert pools for NXR MoE.
//!
//! 324 experts divided into:
//! - 128 **Shared** (math, language, logic, science, code, factual, reasoning, general)
//! - 64 **Ultra** (adv. reasoning, multimodal, metalearning, deep_code, scientific_discovery)
//! - 48 **Apex** (code_review, emotional, task_planning, system_design)
//! - 32 **Pro** (creative, security, data_analysis)
//! - 28 **Core** (temporal, retrieval, summarization)
//! - 24 **Edge** (fast_path, classification, extraction, translation)
//!
//! Router selects top-32 from available experts, biased toward domain-relevant ones.

use serde::{Deserialize, Serialize};

/// Expert specialization domain — what each expert is good at.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertDomain {
    // ── Shared (128 experts, all tiers) ─────────────────────────────────
    /// Mathematical reasoning and computation
    Math,
    /// Natural language understanding and generation
    Language,
    /// Formal logic and structured reasoning
    Logic,
    /// Scientific domain knowledge (physics, chemistry, biology)
    Science,
    /// General-purpose programming and software engineering
    Code,
    /// Factual knowledge and world knowledge
    Factual,
    /// Multi-step reasoning chains
    Reasoning,
    /// General fallback — catches everything else
    General,
    // ── Ultra (64 experts, Ultra tier only) ────────────────────────────
    /// Advanced mathematical and scientific reasoning
    AdvReasoning,
    /// Multi-modal perception and fusion (text, image, audio)
    MultiModal,
    /// Meta-learning and few-shot adaptation
    MetaLearning,
    /// Deep systems programming (kernel, compiler, embedded)
    DeepCode,
    /// Scientific discovery (simulation, analysis, hypothesis)
    ScientificDiscovery,
    // ── Apex (48 experts, Apex tier only) ──────────────────────────────
    /// Code review and static analysis
    CodeReview,
    /// Emotional intelligence and affective computing
    Emotional,
    /// Hierarchical task decomposition and planning
    TaskPlanning,
    /// System architecture and design patterns
    SystemDesign,
    // ── Pro (32 experts, Pro tier only) ────────────────────────────────
    /// Creative writing, art, and divergent thinking
    Creative,
    /// Security analysis, threat modeling, and vulnerability research
    Security,
    /// Data analysis, statistics, and visualization
    DataAnalysis,
    // ── Core (28 experts, Core tier only) ──────────────────────────────
    /// Temporal reasoning with time-dependent context
    Temporal,
    /// Knowledge retrieval and information extraction
    Retrieval,
    /// Abstractive and extractive summarization
    Summarization,
    // ── Edge (24 experts, Edge tier only) ──────────────────────────────
    /// Latency-optimized fast path for simple queries
    FastPath,
    /// Text classification and categorization
    Classification,
    /// Information extraction (NER, relation extraction)
    Extraction,
    /// Machine translation
    Translation,
}

impl ExpertDomain {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ExpertDomain::Math => "math",
            ExpertDomain::Language => "language",
            ExpertDomain::Logic => "logic",
            ExpertDomain::Science => "science",
            ExpertDomain::Code => "code",
            ExpertDomain::Factual => "factual",
            ExpertDomain::Reasoning => "reasoning",
            ExpertDomain::General => "general",
            ExpertDomain::AdvReasoning => "adv_reasoning",
            ExpertDomain::MultiModal => "multimodal",
            ExpertDomain::MetaLearning => "metalearning",
            ExpertDomain::DeepCode => "deep_code",
            ExpertDomain::ScientificDiscovery => "sci_discovery",
            ExpertDomain::CodeReview => "code_review",
            ExpertDomain::Emotional => "emotional",
            ExpertDomain::TaskPlanning => "task_planning",
            ExpertDomain::SystemDesign => "system_design",
            ExpertDomain::Creative => "creative",
            ExpertDomain::Security => "security",
            ExpertDomain::DataAnalysis => "data_analysis",
            ExpertDomain::Temporal => "temporal",
            ExpertDomain::Retrieval => "retrieval",
            ExpertDomain::Summarization => "summarization",
            ExpertDomain::FastPath => "fast_path",
            ExpertDomain::Classification => "classification",
            ExpertDomain::Extraction => "extraction",
            ExpertDomain::Translation => "translation",
        }
    }

    /// Which tier does this domain belong to?
    pub fn tier(&self) -> &'static str {
        match self {
            ExpertDomain::Math | ExpertDomain::Language | ExpertDomain::Logic
            | ExpertDomain::Science | ExpertDomain::Code | ExpertDomain::Factual
            | ExpertDomain::Reasoning | ExpertDomain::General => "shared",
            ExpertDomain::AdvReasoning | ExpertDomain::MultiModal
            | ExpertDomain::MetaLearning | ExpertDomain::DeepCode
            | ExpertDomain::ScientificDiscovery => "ultra",
            ExpertDomain::CodeReview | ExpertDomain::Emotional
            | ExpertDomain::TaskPlanning | ExpertDomain::SystemDesign => "apex",
            ExpertDomain::Creative | ExpertDomain::Security
            | ExpertDomain::DataAnalysis => "pro",
            ExpertDomain::Temporal | ExpertDomain::Retrieval
            | ExpertDomain::Summarization => "core",
            ExpertDomain::FastPath | ExpertDomain::Classification
            | ExpertDomain::Extraction | ExpertDomain::Translation => "edge",
        }
    }
}

/// Assignment of a domain to a specific expert index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertAssignment {
    pub expert_id: usize,
    pub domain: ExpertDomain,
    pub tier: String,
}

/// Per-domain expert counts for a pool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCount {
    pub domain: ExpertDomain,
    pub count: usize,
}

/// Expert pool configuration: partitions 324 experts by domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertPoolConfig {
    /// List of (domain, count) pairs. Sum must equal total experts.
    pub domain_counts: Vec<DomainCount>,
    /// Total experts across all domains.
    pub total_experts: usize,
    /// Bias toward shared experts during routing (0.0–1.0).
    pub shared_bias: f64,
    /// How many of top-32 should come from shared pool (approximate).
    pub shared_slots: usize,
}

impl Default for ExpertPoolConfig {
    fn default() -> Self {
        Self::shared_ultra_apex_pro_core_edge()
    }
}

impl ExpertPoolConfig {
    /// Full 324-expert pool: shared + all tiers.
    pub fn shared_ultra_apex_pro_core_edge() -> Self {
        Self {
            domain_counts: vec![
                // Shared (128)
                DomainCount { domain: ExpertDomain::Math, count: 20 },
                DomainCount { domain: ExpertDomain::Language, count: 24 },
                DomainCount { domain: ExpertDomain::Logic, count: 12 },
                DomainCount { domain: ExpertDomain::Science, count: 16 },
                DomainCount { domain: ExpertDomain::Code, count: 20 },
                DomainCount { domain: ExpertDomain::Factual, count: 16 },
                DomainCount { domain: ExpertDomain::Reasoning, count: 12 },
                DomainCount { domain: ExpertDomain::General, count: 8 },
                // Ultra (64)
                DomainCount { domain: ExpertDomain::AdvReasoning, count: 16 },
                DomainCount { domain: ExpertDomain::MultiModal, count: 16 },
                DomainCount { domain: ExpertDomain::MetaLearning, count: 8 },
                DomainCount { domain: ExpertDomain::DeepCode, count: 12 },
                DomainCount { domain: ExpertDomain::ScientificDiscovery, count: 12 },
                // Apex (48)
                DomainCount { domain: ExpertDomain::CodeReview, count: 16 },
                DomainCount { domain: ExpertDomain::Emotional, count: 12 },
                DomainCount { domain: ExpertDomain::TaskPlanning, count: 12 },
                DomainCount { domain: ExpertDomain::SystemDesign, count: 8 },
                // Pro (32)
                DomainCount { domain: ExpertDomain::Creative, count: 12 },
                DomainCount { domain: ExpertDomain::Security, count: 10 },
                DomainCount { domain: ExpertDomain::DataAnalysis, count: 10 },
                // Core (28)
                DomainCount { domain: ExpertDomain::Temporal, count: 10 },
                DomainCount { domain: ExpertDomain::Retrieval, count: 10 },
                DomainCount { domain: ExpertDomain::Summarization, count: 8 },
                // Edge (24)
                DomainCount { domain: ExpertDomain::FastPath, count: 8 },
                DomainCount { domain: ExpertDomain::Classification, count: 6 },
                DomainCount { domain: ExpertDomain::Extraction, count: 6 },
                DomainCount { domain: ExpertDomain::Translation, count: 4 },
            ],
            total_experts: 324,
            shared_bias: 0.7,
            shared_slots: 22,
        }
    }

    /// Validate that domain counts sum to total_experts.
    pub fn validate(&self) -> Result<(), String> {
        let sum: usize = self.domain_counts.iter().map(|d| d.count).sum();
        if sum != self.total_experts {
            return Err(format!(
                "Expert pool counts sum to {} but expected {}", sum, self.total_experts
            ));
        }
        Ok(())
    }

    /// Build expert assignments — linear mapping of domain → expert index range.
    pub fn build_assignments(&self) -> Vec<ExpertAssignment> {
        let mut assignments = Vec::with_capacity(self.total_experts);
        let mut expert_id = 0;
        for dc in &self.domain_counts {
            for _ in 0..dc.count {
                assignments.push(ExpertAssignment {
                    expert_id,
                    domain: dc.domain,
                    tier: dc.domain.tier().to_string(),
                });
                expert_id += 1;
            }
        }
        assignments
    }

    /// Get domain for a specific expert index.
    pub fn domain_for_expert(&self, expert_id: usize) -> Option<ExpertDomain> {
        let mut cursor = 0;
        for dc in &self.domain_counts {
            if expert_id < cursor + dc.count {
                return Some(dc.domain);
            }
            cursor += dc.count;
        }
        None
    }

    /// Expert index range for a given domain.
    pub fn expert_range(&self, domain: ExpertDomain) -> Option<(usize, usize)> {
        let mut start = 0;
        for dc in &self.domain_counts {
            let end = start + dc.count;
            if dc.domain == domain {
                return Some((start, end));
            }
            start = end;
        }
        None
    }

    /// All expert indices in a given tier.
    pub fn experts_in_tier(&self, tier: &str) -> Vec<usize> {
        let mut result = Vec::new();
        let mut idx = 0;
        for dc in &self.domain_counts {
            if dc.domain.tier() == tier {
                for offset in 0..dc.count {
                    result.push(idx + offset);
                }
            }
            idx += dc.count;
        }
        result
    }

    /// All shared expert indices.
    pub fn shared_experts(&self) -> Vec<usize> {
        self.experts_in_tier("shared")
    }

    /// All tier-specific expert indices.
    pub fn tier_experts(&self, tier: &str) -> Vec<usize> {
        self.experts_in_tier(tier)
    }

    /// Domain bias weights for routing (higher = more likely to be selected).
    /// Used by domain-aware router to prefer relevant experts.
    pub fn domain_bias(&self, domain: ExpertDomain) -> f64 {
        match domain {
            // Shared always get high bias
            ExpertDomain::Math => 1.0,
            ExpertDomain::Language => 1.0,
            ExpertDomain::Logic => 0.9,
            ExpertDomain::Science => 0.9,
            ExpertDomain::Code => 1.0,
            ExpertDomain::Factual => 0.8,
            ExpertDomain::Reasoning => 1.0,
            ExpertDomain::General => 0.7,
            // Tier-specific get moderate bias
            ExpertDomain::AdvReasoning => 1.2,
            ExpertDomain::MultiModal => 0.8,
            ExpertDomain::MetaLearning => 0.7,
            ExpertDomain::DeepCode => 1.1,
            ExpertDomain::ScientificDiscovery => 0.9,
            ExpertDomain::CodeReview => 1.1,
            ExpertDomain::Emotional => 0.7,
            ExpertDomain::TaskPlanning => 1.0,
            ExpertDomain::SystemDesign => 0.9,
            ExpertDomain::Creative => 0.8,
            ExpertDomain::Security => 0.9,
            ExpertDomain::DataAnalysis => 0.8,
            ExpertDomain::Temporal => 0.9,
            ExpertDomain::Retrieval => 0.8,
            ExpertDomain::Summarization => 0.7,
            ExpertDomain::FastPath => 0.6,
            ExpertDomain::Classification => 0.6,
            ExpertDomain::Extraction => 0.6,
            ExpertDomain::Translation => 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_total_324() {
        let pool = ExpertPoolConfig::default();
        assert!(pool.validate().is_ok());
    }

    #[test]
    fn test_assignments_count() {
        let pool = ExpertPoolConfig::default();
        let assignments = pool.build_assignments();
        assert_eq!(assignments.len(), 324);
    }

    #[test]
    fn test_domain_for_expert() {
        let pool = ExpertPoolConfig::default();
        // First 20 experts should be Math
        assert_eq!(pool.domain_for_expert(0), Some(ExpertDomain::Math));
        assert_eq!(pool.domain_for_expert(19), Some(ExpertDomain::Math));
        // Next 24 should be Language
        assert_eq!(pool.domain_for_expert(20), Some(ExpertDomain::Language));
        assert_eq!(pool.domain_for_expert(43), Some(ExpertDomain::Language));
    }

    #[test]
    fn test_shared_expert_count() {
        let pool = ExpertPoolConfig::default();
        let shared = pool.shared_experts();
        assert_eq!(shared.len(), 128);
    }

    #[test]
    fn test_tier_experts() {
        let pool = ExpertPoolConfig::default();
        let ultra = pool.tier_experts("ultra");
        assert_eq!(ultra.len(), 64);
        let apex = pool.tier_experts("apex");
        assert_eq!(apex.len(), 48);
        let pro = pool.tier_experts("pro");
        assert_eq!(pro.len(), 32);
        let core = pool.tier_experts("core");
        assert_eq!(core.len(), 28);
        let edge = pool.tier_experts("edge");
        assert_eq!(edge.len(), 24);
    }

    #[test]
    fn test_expert_range() {
        let pool = ExpertPoolConfig::default();
        let (start, end) = pool.expert_range(ExpertDomain::Math).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 20);
    }

    #[test]
    fn test_domain_bias_non_zero() {
        let pool = ExpertPoolConfig::default();
        for dc in &pool.domain_counts {
            let bias = pool.domain_bias(dc.domain);
            assert!(bias > 0.0, "domain {:?} has zero bias", dc.domain);
        }
    }
}
