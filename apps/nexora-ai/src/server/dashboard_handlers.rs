use axum::{Extension, Json};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::NexoraAI;

pub async fn dashboard_stats(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let stats = nexora.telemetry.store.get_dashboard_stats().await;
    Json(json!(stats))
}

pub async fn dashboard_usage(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let agg = nexora.telemetry.store.get_aggregate().await;
    Json(json!(agg))
}

pub async fn dashboard_events(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let events = nexora.telemetry.store.get_events().await;
    Json(json!(events))
}

pub async fn dashboard_requests(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let requests = nexora.telemetry.store.get_requests().await;
    Json(json!(requests))
}

pub async fn dashboard_metrics(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let snapshots = nexora.telemetry.store.get_metric_snapshots().await;
    Json(json!(snapshots))
}
