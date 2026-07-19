use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::debug;

use super::model_tier::ModelTier;
use super::rule_engine::RuleEngine;
use crate::cost_tracker::CostTracker;

/// Cascading model router — regex → rule → small → medium → large
///
/// Setiap request masuk, coba handle di tier paling murah dulu.
/// Fallthrough ke tier berikutnya jika tier saat ini tidak bisa handle.
pub struct CostCascade {
    rule_engine: Arc<RuleEngine>,
    cost_tracker: Arc<CostTracker>,
    // Counter per tier
    regex_count: AtomicU64,
    rule_count: AtomicU64,
    small_count: AtomicU64,
    medium_count: AtomicU64,
    large_count: AtomicU64,
}

impl CostCascade {
    pub fn new() -> Self {
        Self {
            rule_engine: Arc::new(RuleEngine::new()),
            cost_tracker: Arc::new(CostTracker::new()),
            regex_count: AtomicU64::new(0),
            rule_count: AtomicU64::new(0),
            small_count: AtomicU64::new(0),
            medium_count: AtomicU64::new(0),
            large_count: AtomicU64::new(0),
        }
    }

    /// Route request ke model tier yang tepat
    /// Return: (tier_yang_digunakan, estimated_cost)
    pub fn route(&self, input: &str) -> (ModelTier, f64) {
        // Tier 1: Regex patterns (gratis)
        if let Some(result) = self.rule_engine.evaluate(input) {
            self.rule_count.fetch_add(1, Ordering::Relaxed);
            let cost = result.tier.cost_per_1k_tokens();
            self.cost_tracker
                .record_route(result.tier, 0.0);
            debug!(
                "cost cascade: {} → {} (rule: {})",
                input.chars().take(50).collect::<String>(),
                result.tier.name(),
                result.matched_rule
            );
            return (result.tier, cost);
        }

        // Tier 2: Heuristic routing berdasarkan panjang dan kompleksitas
        let input_lower = input.to_lowercase();
        let word_count = input.split_whitespace().count();
        let has_code = input_lower.contains("fn ") || input_lower.contains("def ") || input_lower.contains("```");
        let is_technical = has_code || input_lower.contains("explain") || input_lower.contains("how does");
        let is_short = word_count < 10;
        let is_medium = word_count < 50;

        let tier = if is_short && !is_technical {
            self.small_count.fetch_add(1, Ordering::Relaxed);
            ModelTier::Small
        } else if is_medium && !is_technical {
            self.medium_count.fetch_add(1, Ordering::Relaxed);
            ModelTier::Medium
        } else {
            self.large_count.fetch_add(1, Ordering::Relaxed);
            ModelTier::Large
        };

        let cost = tier.cost_per_1k_tokens();
        self.cost_tracker.record_route(tier, cost);
        debug!("cost cascade: {} → {}", input.chars().take(50).collect::<String>(), tier.name());
        (tier, cost)
    }

    /// Persentase request yang tidak perlu large model
    pub fn savings_rate(&self) -> f64 {
        let total = self.total_requests() as f64;
        if total == 0.0 {
            return 0.0;
        }
        let non_large = total - self.large_count.load(Ordering::Relaxed) as f64;
        non_large / total
    }

    pub fn total_requests(&self) -> u64 {
        self.regex_count.load(Ordering::Relaxed)
            + self.rule_count.load(Ordering::Relaxed)
            + self.small_count.load(Ordering::Relaxed)
            + self.medium_count.load(Ordering::Relaxed)
            + self.large_count.load(Ordering::Relaxed)
    }

    pub fn tier_counts(&self) -> Vec<(ModelTier, u64)> {
        vec![
            (ModelTier::Regex, self.regex_count.load(Ordering::Relaxed)),
            (ModelTier::RuleEngine, self.rule_count.load(Ordering::Relaxed)),
            (ModelTier::Small, self.small_count.load(Ordering::Relaxed)),
            (ModelTier::Medium, self.medium_count.load(Ordering::Relaxed)),
            (ModelTier::Large, self.large_count.load(Ordering::Relaxed)),
        ]
    }

    pub fn cost_tracker(&self) -> &Arc<CostTracker> {
        &self.cost_tracker
    }
}

impl Default for CostCascade {
    fn default() -> Self {
        Self::new()
    }
}
