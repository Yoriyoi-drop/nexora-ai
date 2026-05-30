use super::plan::{BillingPlan, TierName};
use super::usage::UsageTracker;

pub enum QuotaDecision {
    Allowed,
    DeniedDaily { limit: u32, used: u32 },
    DeniedMonthly { quota: u64, used: u64 },
    DeniedNoPlan,
}

impl QuotaDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, QuotaDecision::Allowed)
    }
}

pub struct QuotaEnforcer {
    usage_tracker: UsageTracker,
}

impl QuotaEnforcer {
    pub fn new(usage_tracker: UsageTracker) -> Self {
        Self { usage_tracker }
    }

    pub fn usage_tracker(&self) -> &UsageTracker {
        &self.usage_tracker
    }

    pub async fn check_quota(&self, api_key: &str, tier_name: &str) -> QuotaDecision {
        let plan = match BillingPlan::find(tier_name) {
            Some(p) => p,
            None => return QuotaDecision::DeniedNoPlan,
        };

        let daily_used = self.usage_tracker.get_daily_count(api_key).await;
        if daily_used >= plan.daily_request_limit {
            return QuotaDecision::DeniedDaily {
                limit: plan.daily_request_limit,
                used: daily_used,
            };
        }

        let monthly_used = self.usage_tracker.get_monthly_tokens(api_key).await;
        if monthly_used >= plan.monthly_token_quota {
            return QuotaDecision::DeniedMonthly {
                quota: plan.monthly_token_quota,
                used: monthly_used,
            };
        }

        QuotaDecision::Allowed
    }

    pub async fn get_remaining(&self, api_key: &str, tier_name: &str) -> (u32, u64) {
        let plan = BillingPlan::find(tier_name);
        let daily_limit = plan.as_ref().map(|p| p.daily_request_limit).unwrap_or(0);
        let monthly_quota = plan.as_ref().map(|p| p.monthly_token_quota).unwrap_or(0);

        let daily_remaining = self.usage_tracker.get_daily_remaining(api_key, daily_limit).await;
        let monthly_remaining = self.usage_tracker.get_monthly_remaining(api_key, monthly_quota).await;

        (daily_remaining, monthly_remaining)
    }
}
