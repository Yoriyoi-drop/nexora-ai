use axum::{Extension, Json};
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::NexoraAI;

pub async fn telemetry_inference(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let snapshot = nexora_inference::inference_trait::observability_snapshot();
    let perf = nexora.get_performance_metrics().await.unwrap_or_default();
    let uptime_secs = (chrono::Utc::now() - nexora.start_time).num_seconds() as f64;

    Json(json!({
        "model_name": format!("{}", nexora.model_id()),
        "tokens_per_second": snapshot.tokens_per_sec,
        "latency_ms": perf.get("uptime_seconds").and_then(|v| v.as_f64()),
        "latency_p50_ms": null,
        "latency_p99_ms": null,
        "context_length": null,
        "batch_size": snapshot.batching_efficiency,
        "cache_hit_rate": snapshot.prefix_cache_hit_ratio,
        "cache_total_entries": snapshot.kv_cache_used,
        "cache_memory_bytes": snapshot.kv_cache_pressure as u64,
        "active_requests": snapshot.scheduler_depth.max(0) as u32,
        "total_requests": nexora.request_count.load(Ordering::Relaxed) as u64,
        "error_rate": snapshot.gpu_forward_errors as f64 / (snapshot.gpu_tokens + 1).max(1) as f64,
        "speculative_acceptance_rate": null,
        "kv_cache_usage_pct": if snapshot.kv_cache_total > 0 { (snapshot.kv_cache_used as f64 / snapshot.kv_cache_total as f64) * 100.0 } else { 0.0 },
    }))
}

pub async fn telemetry_agents(
    nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let agent_mgr = nexora.agent_manager();
    let agents = agent_mgr.list_agents().await.unwrap_or_default();
    let agent_list: Vec<Value> = agents.iter().map(|a| json!({
        "id": a.id(),
        "name": a.name(),
        "agent_type": format!("{:?}", a.agent_type()),
        "status": "active",
        "config": {},
    })).collect();
    Json(json!(agent_list))
}

pub async fn telemetry_memory(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    // TODO: implement — wire real MemoryStore and return persistent memory graph
    Json(json!({
        "nodes": [],
        "summary": {
            "total_nodes": 0,
            "active_nodes": 0,
            "memory_type": "distributed",
        },
    }))
}

pub async fn telemetry_pipeline(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    // TODO: implement — return active DataStream pipeline status and throughput
    let pipelines: Vec<Value> = vec![];
    Json(json!(pipelines))
}

pub async fn telemetry_hallucination(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    // TODO: implement — wire nexora-hallucination detector and return real stats
    Json(json!({
        "total_checked": 0,
        "total_blocked": 0,
        "total_flagged": 0,
        "hallucination_rate": 0.0,
        "risk_score_avg": 0.0,
        "pre_gen_blocked": 0,
        "in_gen_corrected": 0,
        "post_gen_flagged": 0,
        "top_risk_sources": [],
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
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    // TODO: implement — read loaded models from model registry instead of hardcoded list
    let models: Vec<Value> = vec![
        json!({"id": "omnis", "name": "Omnis", "status": "loaded", "parameters": "7B"}),
        json!({"id": "swift", "name": "Swift", "status": "loaded", "parameters": "1B"}),
        json!({"id": "vortex", "name": "Vortex", "status": "loaded", "parameters": "7B"}),
    ];
    Json(json!(models))
}

pub async fn telemetry_token_flows(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let snapshot = nexora_inference::inference_trait::observability_snapshot();
    Json(json!([{
        "source": "inference_engine",
        "gpu_tokens": snapshot.gpu_tokens,
        "cpu_tokens": snapshot.cpu_tokens,
        "gpu_fallbacks": snapshot.gpu_fallbacks,
        "prefix_cache_hits": snapshot.prefix_cache_hits,
        "prefix_cache_misses": snapshot.prefix_cache_misses,
    }]))
}
