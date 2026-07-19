use axum::{Extension, Json};
use serde_json::Value;
use std::sync::Arc;

use crate::NexoraAI;

/// GET /api/system/status — comprehensive system dashboard
pub async fn system_dashboard(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    let json_str = nexora.system.dashboard_json();
    let data: Value = serde_json::from_str(&json_str).unwrap_or_default();
    Json(data)
}

/// GET /api/system/metrics — prometheus-format metrics
pub async fn system_metrics_prometheus(Extension(nexora): Extension<Arc<NexoraAI>>) -> String {
    nexora.system.prometheus_metrics()
}

/// GET /api/system/optimizer — cost optimizer stats
pub async fn system_cost_stats(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    let tracker = nexora.system.cost_optimizer.cost_tracker();
    let stats = tracker.stats();
    Json(serde_json::json!({
        "total_cost_usd": stats.total_cost,
        "total_saved_usd": stats.total_saved,
        "total_tokens": stats.total_tokens,
        "savings_rate": format!("{:.1}%", nexora.system.cost_optimizer.savings_rate() * 100.0),
        "tier_counts": nexora.system.cost_optimizer.cascade().tier_counts()
            .iter().map(|(tier, count)| serde_json::json!({
                "tier": tier.name(),
                "requests": count
            })).collect::<Vec<_>>(),
    }))
}
