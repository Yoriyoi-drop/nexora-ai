use axum::{
    body::Bytes,
    http::HeaderMap,
    Extension, Json,
};
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

use crate::billing::{BillingPlan, TierName};
use crate::NexoraAI;

type HmacSha256 = Hmac<Sha256>;

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
    pub success_url: Option<String>,
    pub cancel_url: Option<String>,
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

/// Map plan name to Stripe Price ID from config.
/// Set STRIPE_PRO_PRICE_ID / STRIPE_ENTERPRISE_PRICE_ID env vars or config values.
fn stripe_price_id(plan_name: &str, config: &super::config::billing::BillingConfig) -> Option<String> {
    match plan_name {
        "pro" => config.stripe_pro_price_id.clone().or_else(|| {
            tracing::warn!("STRIPE_PRO_PRICE_ID not configured; set env var or add to config");
            None
        }),
        "enterprise" => config.stripe_enterprise_price_id.clone().or_else(|| {
            tracing::warn!("STRIPE_ENTERPRISE_PRICE_ID not configured; set env var or add to config");
            None
        }),
        _ => None,
    }
}

/// Create a Stripe Checkout Session via the Stripe API.
/// Returns the checkout URL or an error message.
async fn create_stripe_checkout(
    secret_key: &str,
    plan_name: &str,
    plan: &BillingPlan,
    success_url: &str,
    cancel_url: &str,
    config: &super::config::billing::BillingConfig,
) -> Result<String, String> {
    let price = stripe_price_id(plan_name, config).ok_or_else(|| {
        format!("Plan '{}' does not have a Stripe price configured. Set STRIPE_PRO_PRICE_ID or STRIPE_ENTERPRISE_PRICE_ID", plan_name)
    })?;

    let client = reqwest::Client::new();
    let params = serde_json::json!({
        "mode": "subscription",
        "line_items": [{
            "price": price,
            "quantity": 1
        }],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "metadata": {
            "plan": plan_name,
            "price_monthly": plan.price_monthly
        }
    });

    let resp = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .header("Authorization", format!("Bearer {}", secret_key))
        .header("Content-Type", "application/json")
        .json(&params)
        .send()
        .await
        .map_err(|e| format!("Stripe API request failed: {}", e))?;

    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse Stripe response: {}", e))?;

    body.get("url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            let error_msg = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            format!("Stripe checkout creation failed: {}", error_msg)
        })
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
    let Some(tier) = TierName::from_str(&plan_name) else {
        return Json(json!({
            "success": false,
            "error": format!("Unknown plan: {}. Available: free, pro, enterprise", plan_name)
        }));
    };

    let plan = BillingPlan::find(&plan_name);
    let plan = match plan {
        Some(p) => p,
        None => {
            return Json(json!({
                "success": false,
                "error": format!("Plan configuration not found for: {}", plan_name)
            }));
        }
    };

    // If free plan, activate immediately without Stripe
    if plan_name == "free" {
        return Json(json!({
            "success": true,
            "plan": PlanInfo::from(plan),
            "status": "active",
            "checkout_url": null,
            "note": "Free plan activated immediately."
        }));
    }

    // Try to create Stripe Checkout Session if secret key is configured
    let config = nexora.billing.config.read().await;
    let checkout_url = if let Some(ref secret_key) = config.stripe_secret_key {
        let success_url = payload
            .success_url
            .unwrap_or_else(|| "https://nexora.ai/billing/success".to_string());
        let cancel_url = payload
            .cancel_url
            .unwrap_or_else(|| "https://nexora.ai/billing/cancel".to_string());

        match create_stripe_checkout(secret_key, &plan_name, &plan, &success_url, &cancel_url, &config).await
        {
            Ok(url) => Some(url),
            Err(e) => {
                warn!("Stripe checkout failed: {}", e);
                None
            }
        }
    } else {
        warn!("No stripe_secret_key configured; cannot create checkout");
        None
    };
    drop(config);

    info!("Subscription checkout created for plan: {}", plan_name);

    Json(json!({
        "success": true,
        "plan": PlanInfo::from(plan),
        "status": "pending",
        "checkout_url": checkout_url,
    }))
}

pub async fn stripe_webhook(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    body: Bytes,
) -> Json<Value> {
    let config = nexora.billing.config.read().await;
    let webhook_secret = config.stripe_webhook_secret.clone();
    drop(config);

    if let Some(ref secret) = webhook_secret {
        let sig_header = headers
            .get("stripe-signature")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        match sig_header {
            Some(sig) => {
                let mut expected = match HmacSha256::new_from_slice(secret.as_bytes()) {
                    Ok(mac) => mac,
                    Err(e) => {
                        tracing::error!("HMAC key invalid: {e}. Skipping webhook signature verification.");
                        return Json(json!({"success": false, "error": "Invalid webhook configuration"}));
                    }
                };
                expected.update(&body);
                let expected_sig = hex::encode(expected.finalize().into_bytes());

                let parts: Vec<&str> = sig.split(',').collect();
                let sig_valid = parts.iter().any(|p| {
                    if let Some(v1) = p.strip_prefix("v1=") {
                        expected_sig == v1
                    } else {
                        false
                    }
                });

                if !sig_valid {
                    warn!("Stripe webhook signature verification failed");
                    return Json(json!({
                        "success": false,
                        "error": "Invalid webhook signature"
                    }));
                }
            }
            None => {
                warn!("Stripe webhook missing Stripe-Signature header");
                return Json(json!({
                    "success": false,
                    "error": "Missing Stripe-Signature header"
                }));
            }
        }
    } else {
        info!("No stripe_webhook_secret configured; skipping signature verification");
    }

    let payload: Value = serde_json::from_slice(&body).unwrap_or_default();
    let event_type = payload
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let event_id = payload
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("no-id");

    info!("Stripe webhook received: type={}, id={}", event_type, event_id);

    match event_type {
        "checkout.session.completed" => {
            let session = payload.get("data").and_then(|d| d.get("object"));
            if let Some(session) = session {
                let customer_email = session
                    .get("customer_email")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let plan_name = session
                    .get("metadata")
                    .and_then(|m| m.get("plan"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                info!(
                    "Checkout completed: email={}, plan={}",
                    customer_email, plan_name
                );
                let customer_email = session
                    .get("customer_email")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let plan_name = session
                    .get("metadata")
                    .and_then(|m| m.get("plan"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("free");
                info!(
                    "Provisioning subscription for {} on plan {}",
                    customer_email, plan_name
                );
                match nexora.billing.enforcer.provision_subscription(customer_email, plan_name).await {
                    Ok(_) => info!("Subscription provisioned successfully for {}", customer_email),
                    Err(e) => warn!("Failed to provision subscription: {}", e),
                }
            }
        }
        "customer.subscription.updated" | "customer.subscription.deleted" => {
            let subscription = payload.get("data").and_then(|d| d.get("object"));
            if let Some(sub) = subscription {
                let status = sub
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let sub_id = sub
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let customer_email = sub
                    .get("customer_email")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                info!(
                    "Subscription {} status updated to: {} for {}",
                    sub_id, status, customer_email
                );
                match nexora.billing.enforcer.update_subscription_tier(customer_email, status).await {
                    Ok(_) => info!("Tier updated for {}", customer_email),
                    Err(e) => warn!("Failed to update tier: {}", e),
                }
            }
        }
        "invoice.payment_succeeded" => {
            info!("Payment succeeded for event {}", event_id);
        }
        "invoice.payment_failed" => {
            warn!("Payment failed for event {}", event_id);
        }
        _ => {
            info!("Unhandled webhook event type: {}", event_type);
        }
    }

    Json(json!({
        "success": true,
        "message": "Webhook received and processed",
        "event_type": event_type,
        "event_id": event_id,
    }))
}
