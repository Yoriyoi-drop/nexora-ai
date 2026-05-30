use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    pub enable_billing: bool,
    pub default_tier: String,
    pub stripe_secret_key: Option<String>,
    pub stripe_webhook_secret: Option<String>,
    pub stripe_price_lookup: bool,
    pub usage_sync_interval_secs: u64,
}

impl Default for BillingConfig {
    fn default() -> Self {
        Self {
            enable_billing: false,
            default_tier: "free".to_string(),
            stripe_secret_key: None,
            stripe_webhook_secret: None,
            stripe_price_lookup: false,
            usage_sync_interval_secs: 300,
        }
    }
}
