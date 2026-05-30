use axum::{Extension, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

use crate::billing::{BillingPlan, TierName};
use crate::NexoraAI;

#[derive(Debug, Serialize)]
pub struct PlanResponse {
    pub plans: Vec<PlanInfo>,
}

#[derive(Debug, Serialize)]
pub struct PlanInfo {
    pub id: i32,
    pub name: String,
    pub display_name: String,
    pub monthly_token_quota: u64,
    pub daily_request_limit: u32,
    pub max_concurrent_requests: u32,
    pub price_monthly: f64,
    pub features: Value,
}

impl From<BillingPlan> for PlanInfo {
    fn from(p: BillingPlan) -> Self {
        PlanInfo {
            id: p.id,
            name: p.name.as_str().to_string(),
            display_name: p.display_name,
            monthly_token_quota: p.monthly_token_quota,
            daily_request_limit: p.daily_request_limit,
            max_concurrent_requests: p.max_concurrent_requests,
            price_monthly: p.price_monthly,
            features: serde_json::to_value(p.features).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SubscribeRequest {
    pub plan: String,
}

#[derive(Debug, Serialize)]
pub struct UsageResponse {
    pub api_key: String,
    pub tier: String,
    pub monthly_tokens_used: u64,
    pub monthly_token_quota: u64,
    pub monthly_remaining: u64,
    pub daily_requests_used: u32,
    pub daily_request_limit: u32,
    pub daily_remaining: u32,
    pub timestamp: u64,
}

#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WebhookPayload {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Option<Value>,
}

pub async fn list_billing_plans() -> Json<Value> {
    let plans: Vec<PlanInfo> = BillingPlan::builtin()
        .into_iter()
        .map(|p| PlanInfo::from(p))
        .collect();
    Json(json!({ "success": true, "plans": plans }))
}

pub async fn get_usage(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(query): Json<UsageQuery>,
) -> Json<Value> {
    let billing = &nexora.billing;
    let api_key = query.api_key.unwrap_or_else(|| "default".to_string());
    let tier = billing.resolve_tier(&api_key).await;

    let enforcer = &billing.enforcer;
    let usage = enforcer.usage_tracker().usage_summary(&api_key).await;
    let (daily_remaining, monthly_remaining) = enforcer.get_remaining(&api_key, &tier).await;

    let plan = BillingPlan::find(&tier);

    Json(json!({
        "success": true,
        "usage": {
            "api_key": api_key,
            "tier": tier,
            "monthly_tokens_used": usage.monthly_tokens_used,
            "monthly_token_quota": plan.as_ref().map(|p| p.monthly_token_quota).unwrap_or(0),
            "monthly_remaining": monthly_remaining,
            "daily_requests_used": usage.daily_requests,
            "daily_request_limit": plan.as_ref().map(|p| p.daily_request_limit).unwrap_or(0),
            "daily_remaining": daily_remaining,
            "timestamp": usage.timestamp,
        }
    }))
}

pub async fn get_subscription(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(query): Json<UsageQuery>,
) -> Json<Value> {
    let api_key = query.api_key.unwrap_or_else(|| "default".to_string());
    let tier_name = nexora.billing.resolve_tier(&api_key).await;
    let plan = BillingPlan::find(&tier_name);

    match plan {
        Some(p) => Json(json!({
            "success": true,
            "subscription": {
                "plan": PlanInfo::from(p),
                "status": "active",
            }
        })),
        None => Json(json!({
            "success": false,
            "error": format!("Unknown tier: {}", tier_name)
        })),
    }
}

pub async fn subscribe(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<SubscribeRequest>,
) -> Json<Value> {
    if !nexora.billing.is_enabled().await {
        return Json(json!({
            "success": false,
            "error": "Billing system is disabled. Set enable_billing = true in config."
        }));
    }

    let plan_name = payload.plan.to_lowercase();
    if TierName::from_str(&plan_name).is_none() {
        return Json(json!({
            "success": false,
            "error": format!("Unknown plan: {}. Available: free, pro, enterprise", plan_name)
        }));
    }

    let mut meta = HashMap::new();
    meta.insert("plan".to_string(), plan_name.clone());
    nexora.telemetry.recorder.record_event("billing.subscribe", None, meta).await;

    info!("Subscription request for plan: {}", plan_name);
    Json(json!({
        "success": true,
        "message": format!("Subscribed to {} plan. Stripe checkout URL generation not yet implemented.", plan_name),
        "plan": plan_name,
        "note": "In production, this returns a Stripe Checkout Session URL."
    }))
}

pub async fn stripe_webhook(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let event_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
    let mut meta = HashMap::new();
    meta.insert("event_type".to_string(), event_type.to_string());
    nexora.telemetry.recorder.record_event("billing.webhook", None, meta).await;
    warn!("Stripe webhook received (stub): event_type={}", event_type);
    Json(json!({
        "success": true,
        "message": "Webhook received (stub)"
    }))
}
