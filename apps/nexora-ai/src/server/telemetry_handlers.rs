use axum::{Extension, Json};
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::oneshot;

use nexora_agent::agent_manager::ManagerCommand;
use crate::NexoraAI;

pub async fn telemetry_inference(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let snapshot = nexora_inference::inference_trait::observability_snapshot();
    let uptime_secs = (chrono::Utc::now() - nexora.start_time).num_seconds() as f64;

    Json(json!({
        "tokens_per_second": snapshot.tokens_per_sec,
        "batch_efficiency_pct": snapshot.batching_efficiency_pct,
        "cache_hit_rate": snapshot.prefix_cache_hit_ratio,
        "cache_pressure": snapshot.kv_cache_pressure,
        "active_requests": snapshot.scheduler_queue_depth.max(0) as u32,
        "total_requests": nexora.request_count.load(Ordering::Relaxed) as u64,
        "gpu_tokens": snapshot.gpu_tokens_generated,
        "cpu_tokens": snapshot.cpu_tokens_generated,
        "gpu_fallbacks": snapshot.gpu_cpu_fallbacks,
        "gpu_forward_errors": snapshot.gpu_forward_errors,
        "gpu_alive": snapshot.gpu_alive,
        "prefix_cache_hits": snapshot.prefix_cache_hits,
        "prefix_cache_misses": snapshot.prefix_cache_misses,
        "uptime_secs": uptime_secs,
    }))
}

pub async fn telemetry_agents(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let cmd_tx = nexora.agent_manager().command_sender();
    let (tx, rx) = oneshot::channel();
    let _ = cmd_tx.send(ManagerCommand::ListAgentIds { response_tx: tx }).await;
    let grouped: Vec<Value> = rx.await.unwrap_or_default().into_iter().map(|(agent_type, ids)| {
        json!({
            "agent_type": agent_type,
            "count": ids.len(),
            "ids": ids,
        })
    }).collect();
    Json(json!(grouped))
}

pub async fn telemetry_memory(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let snapshot = nexora_inference::inference_trait::observability_snapshot();
    let total_nodes = if nexora.distributed().is_some() { 2 } else { 1 };
    Json(json!({
        "nodes": [],
        "summary": {
            "total_nodes": total_nodes,
            "active_nodes": total_nodes,
            "memory_type": "distributed",
            "kv_cache_pressure": snapshot.kv_cache_pressure,
            "gpu_alive": snapshot.gpu_alive,
        },
    }))
}

pub async fn telemetry_pipeline(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let snapshot = nexora_inference::inference_trait::observability_snapshot();
    Json(json!([{
        "stage": "inference",
        "active_requests": snapshot.scheduler_queue_depth.max(0) as u32,
        "gpu_vs_cpu_ratio": if snapshot.cpu_tokens_generated.max(1) > 0 {
            snapshot.gpu_tokens_generated as f64 / snapshot.cpu_tokens_generated.max(1) as f64
        } else { 0.0 },
        "cache_hit_rate": snapshot.prefix_cache_hit_ratio,
        "batch_efficiency_pct": snapshot.batching_efficiency_pct,
    }]))
}

pub async fn telemetry_hallucination() -> Json<Value> {
    let guard = &*nexora_hallucination::monitoring::GLOBAL_HALLUCINATION_MONITOR;
    Json(json!({
        "total_checked": guard.total_checked(),
        "total_blocked": guard.total_blocked(),
        "total_flagged": guard.total_flagged(),
        "hallucination_rate": guard.hallucination_rate(),
    }))
}

pub async fn telemetry_training(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let train = nexora.train_metrics.read().await;
    Json(json!({
        "is_training": train.get("is_training").and_then(|v| v.as_bool()).unwrap_or(false),
        "current_epoch": train.get("epoch").and_then(|v| v.as_u64()).unwrap_or(0),
        "total_epochs": train.get("total_epochs").and_then(|v| v.as_u64()).unwrap_or(0),
        "current_step": train.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
        "total_steps": train.get("total_steps").and_then(|v| v.as_u64()).unwrap_or(0),
        "loss": train.get("loss").and_then(|v| v.as_f64()),
        "learning_rate": train.get("learning_rate").and_then(|v| v.as_f64()),
        "grad_norm": train.get("grad_norm").and_then(|v| v.as_f64()),
        "tokens_per_second": train.get("tokens_per_second").and_then(|v| v.as_f64()),
        "samples_processed": train.get("samples_processed").and_then(|v| v.as_u64()).unwrap_or(0),
        "time_elapsed_secs": train.get("time_elapsed_secs").and_then(|v| v.as_f64()),
        "estimated_time_remaining_secs": train.get("estimated_time_remaining_secs").and_then(|v| v.as_f64()),
    }))
}

pub async fn telemetry_models(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let model_id = nexora.model_id();
    Json(json!([{
        "id": model_id.to_string(),
        "name": model_id.to_string(),
        "status": "loaded",
    }]))
}

pub async fn telemetry_token_flows() -> Json<Value> {
    let snapshot = nexora_inference::inference_trait::observability_snapshot();
    Json(json!([{
        "source": "inference_engine",
        "gpu_tokens": snapshot.gpu_tokens_generated,
        "cpu_tokens": snapshot.cpu_tokens_generated,
        "gpu_fallbacks": snapshot.gpu_cpu_fallbacks,
        "prefix_cache_hits": snapshot.prefix_cache_hits,
        "prefix_cache_misses": snapshot.prefix_cache_misses,
    }]))
}
