use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TierName {
    Free,
    Pro,
    Enterprise,
}

impl TierName {
    pub fn as_str(&self) -> &'static str {
        match self {
            TierName::Free => "free",
            TierName::Pro => "pro",
            TierName::Enterprise => "enterprise",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "free" => Some(TierName::Free),
            "pro" => Some(TierName::Pro),
            "enterprise" => Some(TierName::Enterprise),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingPlan {
    pub id: i32,
    pub name: TierName,
    pub display_name: String,
    pub monthly_token_quota: u64,
    pub daily_request_limit: u32,
    pub max_concurrent_requests: u32,
    pub price_monthly: f64,
    pub features: HashMap<String, serde_json::Value>,
}

impl BillingPlan {
    pub fn builtin() -> Vec<BillingPlan> {
        vec![
            BillingPlan {
                id: 1,
                name: TierName::Free,
                display_name: "Free".to_string(),
                monthly_token_quota: 100_000,
                daily_request_limit: 100,
                max_concurrent_requests: 5,
                price_monthly: 0.0,
                features: {
                    let mut m = HashMap::new();
                    m.insert("models".into(), serde_json::json!(["nexora-1b"]));
                    m.insert("context_length".into(), serde_json::json!(4096));
                    m.insert("priority".into(), serde_json::json!(false));
                    m.insert("dedicated".into(), serde_json::json!(false));
                    m
                },
            },
            BillingPlan {
                id: 2,
                name: TierName::Pro,
                display_name: "Pro".to_string(),
                monthly_token_quota: 10_000_000,
                daily_request_limit: 1000,
                max_concurrent_requests: 20,
                price_monthly: 29.99,
                features: {
                    let mut m = HashMap::new();
                    m.insert("models".into(), serde_json::json!(["nexora-1b", "nexora-7b"]));
                    m.insert("context_length".into(), serde_json::json!(8192));
                    m.insert("priority".into(), serde_json::json!(true));
                    m.insert("dedicated".into(), serde_json::json!(false));
                    m
                },
            },
            BillingPlan {
                id: 3,
                name: TierName::Enterprise,
                display_name: "Enterprise".to_string(),
                monthly_token_quota: 1_000_000_000,
                daily_request_limit: 10_000,
                max_concurrent_requests: 100,
                price_monthly: 299.99,
                features: {
                    let mut m = HashMap::new();
                    m.insert("models".into(), serde_json::json!(["nexora-1b", "nexora-7b", "nexora-13b"]));
                    m.insert("context_length".into(), serde_json::json!(32768));
                    m.insert("priority".into(), serde_json::json!(true));
                    m.insert("dedicated".into(), serde_json::json!(true));
                    m
                },
            },
        ]
    }

    pub fn find(name: &str) -> Option<BillingPlan> {
        BillingPlan::builtin().into_iter().find(|p| p.name.as_str() == name)
    }
}
