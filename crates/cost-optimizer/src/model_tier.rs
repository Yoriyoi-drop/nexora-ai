use serde::{Deserialize, Serialize};

/// Model tier — semakin besar semakin mahal tapi capable
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    Regex,
    RuleEngine,
    Small,
    Medium,
    Large,
}

impl ModelTier {
    pub fn name(&self) -> &'static str {
        match self {
            ModelTier::Regex => "regex",
            ModelTier::RuleEngine => "rule-engine",
            ModelTier::Small => "small",
            ModelTier::Medium => "medium",
            ModelTier::Large => "large",
        }
    }

    /// Estimated cost per 1K tokens (USD)
    pub fn cost_per_1k_tokens(&self) -> f64 {
        match self {
            ModelTier::Regex => 0.0,
            ModelTier::RuleEngine => 0.0,
            ModelTier::Small => 0.0005,
            ModelTier::Medium => 0.002,
            ModelTier::Large => 0.01,
        }
    }

    /// Estimated latency (ms)
    pub fn latency_ms(&self) -> u64 {
        match self {
            ModelTier::Regex => 1,
            ModelTier::RuleEngine => 5,
            ModelTier::Small => 100,
            ModelTier::Medium => 500,
            ModelTier::Large => 2000,
        }
    }

    /// Berapa banyak request di tier ini yang bisa di-handle sebelum fallthrough
    pub fn capacity(&self) -> f64 {
        match self {
            ModelTier::Regex => 0.20,
            ModelTier::RuleEngine => 0.25,
            ModelTier::Small => 0.20,
            ModelTier::Medium => 0.20,
            ModelTier::Large => 0.15,
        }
    }
}
