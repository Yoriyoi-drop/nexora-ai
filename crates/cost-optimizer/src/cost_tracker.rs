use std::sync::atomic::{AtomicU64, Ordering};

use super::model_tier::ModelTier;

/// Track biaya LLM — total cost, saved cost, per-tier breakdown
pub struct CostTracker {
    total_cost_micro: AtomicU64,
    total_saved_micro: AtomicU64,
    total_tokens: AtomicU64,

    // Per-tier
    small_cost: AtomicU64,
    medium_cost: AtomicU64,
    large_cost: AtomicU64,

    small_tokens: AtomicU64,
    medium_tokens: AtomicU64,
    large_tokens: AtomicU64,
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            total_cost_micro: AtomicU64::new(0),
            total_saved_micro: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            small_cost: AtomicU64::new(0),
            medium_cost: AtomicU64::new(0),
            large_cost: AtomicU64::new(0),
            small_tokens: AtomicU64::new(0),
            medium_tokens: AtomicU64::new(0),
            large_tokens: AtomicU64::new(0),
        }
    }

    pub fn record_route(&self, tier: ModelTier, cost: f64) {
        // Cost in micro-dollars
        let cost_micro = (cost * 1_000_000.0) as u64;

        self.total_cost_micro
            .fetch_add(cost_micro, Ordering::Relaxed);

        // Jika pakai small/medium, catat savings vs large
        let large_cost = ModelTier::Large.cost_per_1k_tokens() * 1_000_000.0;
        match tier {
            ModelTier::Small => {
                let saved = (large_cost - cost * 1_000_000.0) as u64;
                self.total_saved_micro.fetch_add(saved, Ordering::Relaxed);
                self.small_cost
                    .fetch_add(cost_micro, Ordering::Relaxed);
                self.small_tokens.fetch_add(100, Ordering::Relaxed);
            }
            ModelTier::Medium => {
                let saved = (large_cost - cost * 1_000_000.0) as u64;
                self.total_saved_micro.fetch_add(saved, Ordering::Relaxed);
                self.medium_cost
                    .fetch_add(cost_micro, Ordering::Relaxed);
                self.medium_tokens.fetch_add(200, Ordering::Relaxed);
            }
            ModelTier::Large => {
                self.large_cost
                    .fetch_add(cost_micro, Ordering::Relaxed);
                self.large_tokens.fetch_add(500, Ordering::Relaxed);
            }
            _ => {}
        }

        // Approx tokens: small=100, medium=200, large=500 per request
        let approx_tokens = match tier {
            ModelTier::Regex | ModelTier::RuleEngine => 0,
            ModelTier::Small => 100,
            ModelTier::Medium => 200,
            ModelTier::Large => 500,
        };
        self.total_tokens
            .fetch_add(approx_tokens, Ordering::Relaxed);
    }

    pub fn total_cost(&self) -> f64 {
        self.total_cost_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn total_saved(&self) -> f64 {
        self.total_saved_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> CostStats {
        CostStats {
            total_cost: self.total_cost(),
            total_saved: self.total_saved(),
            total_tokens: self.total_tokens(),
            small_cost: self.small_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            medium_cost: self.medium_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            large_cost: self.large_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        }
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CostStats {
    pub total_cost: f64,
    pub total_saved: f64,
    pub total_tokens: u64,
    pub small_cost: f64,
    pub medium_cost: f64,
    pub large_cost: f64,
}
