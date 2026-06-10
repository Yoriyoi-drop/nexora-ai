use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    pub enable_billing: bool,
    pub default_tier: String,
    pub stripe_secret_key: Option<String>,
    pub stripe_webhook_secret: Option<String>,
    pub stripe_price_lookup: bool,
    pub stripe_pro_price_id: Option<String>,
    pub stripe_enterprise_price_id: Option<String>,
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
            stripe_pro_price_id: std::env::var("STRIPE_PRO_PRICE_ID").ok(),
            stripe_enterprise_price_id: std::env::var("STRIPE_ENTERPRISE_PRICE_ID").ok(),
            usage_sync_interval_secs: 300,
        }
    }
}
