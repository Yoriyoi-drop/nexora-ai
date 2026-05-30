pub mod enforcer;
pub mod plan;
pub mod usage;

pub use enforcer::{QuotaDecision, QuotaEnforcer};
pub use plan::{BillingPlan, TierName};
pub use usage::{UsageRecord, UsageSummary, UsageTracker};

use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct BillingSystem {
    pub enforcer: Arc<QuotaEnforcer>,
    pub config: Arc<RwLock<crate::config::billing::BillingConfig>>,
}

impl BillingSystem {
    pub fn new(config: crate::config::billing::BillingConfig) -> Self {
        let usage_tracker = UsageTracker::new();
        let enforcer = QuotaEnforcer::new(usage_tracker);
        Self {
            enforcer: Arc::new(enforcer),
            config: Arc::new(RwLock::new(config)),
        }
    }

    pub fn disabled() -> Self {
        Self::new(crate::config::billing::BillingConfig {
            enable_billing: false,
            ..Default::default()
        })
    }

    pub async fn resolve_tier(&self, api_key: &str) -> String {
        let config = self.config.read().await;
        config.default_tier.clone()
    }

    pub async fn is_enabled(&self) -> bool {
        self.config.read().await.enable_billing
    }
}
