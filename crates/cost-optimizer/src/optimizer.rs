use std::sync::Arc;

use super::cascade::CostCascade;
use super::cost_tracker::CostTracker;

/// AI Cost Optimizer — entry point untuk cost optimization
pub struct CostOptimizer {
    cascade: Arc<CostCascade>,
}

impl CostOptimizer {
    pub fn new() -> Self {
        Self {
            cascade: Arc::new(CostCascade::new()),
        }
    }

    /// Route request ke tier paling cost-effective
    pub fn optimize(&self, input: &str) -> OptimizerResult {
        let (tier, cost) = self.cascade.route(input);
        let is_large = matches!(tier, super::model_tier::ModelTier::Large);

        OptimizerResult {
            recommended_tier: tier.name().to_string(),
            estimated_cost: cost,
            savings_vs_large: if is_large {
                0.0
            } else {
                super::model_tier::ModelTier::Large.cost_per_1k_tokens() - cost
            },
            use_large_model: is_large,
        }
    }

    pub fn cascade(&self) -> &Arc<CostCascade> {
        &self.cascade
    }

    pub fn cost_tracker(&self) -> &Arc<CostTracker> {
        &self.cascade.cost_tracker()
    }

    pub fn savings_rate(&self) -> f64 {
        self.cascade.savings_rate()
    }
}

impl Default for CostOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OptimizerResult {
    pub recommended_tier: String,
    pub estimated_cost: f64,
    pub savings_vs_large: f64,
    pub use_large_model: bool,
}
