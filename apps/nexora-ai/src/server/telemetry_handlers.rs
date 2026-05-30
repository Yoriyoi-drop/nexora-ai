use axum::{Extension, Json};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::NexoraAI;

fn null_if<T: Serialize>(v: Option<T>) -> serde_json::Value {
    match v {
        Some(v) => serde_json::to_value(v).unwrap_or(Value::Null),
        None => Value::Null,
    }
}

fn empty_vec<T>() -> Vec<T> {
    vec![]
}

pub async fn telemetry_inference(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    Json(json!({
        "model_name": null_if(None::<String>),
        "tokens_per_second": null_if(None::<f64>),
        "latency_ms": null_if(None::<f64>),
        "latency_p50_ms": null_if(None::<f64>),
        "latency_p99_ms": null_if(None::<f64>),
        "context_length": null_if(None::<u32>),
        "batch_size": null_if(None::<u32>),
        "cache_hit_rate": null_if(None::<f64>),
        "cache_total_entries": null_if(None::<u64>),
        "cache_memory_bytes": null_if(None::<u64>),
        "active_requests": null_if(None::<u32>),
        "total_requests": null_if(None::<u64>),
        "error_rate": null_if(None::<f64>),
        "speculative_acceptance_rate": null_if(None::<f64>),
        "kv_cache_usage_pct": null_if(None::<f64>),
    }))
}

pub async fn telemetry_agents(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let agents: Vec<Value> = vec![];
    Json(json!(agents))
}

pub async fn telemetry_memory(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    Json(json!({
        "nodes": [],
        "summary": null_if(None::<Value>),
    }))
}

pub async fn telemetry_pipeline(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let pipelines: Vec<Value> = vec![];
    Json(json!(pipelines))
}

pub async fn telemetry_hallucination(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    Json(json!({
        "total_checked": null_if(None::<u64>),
        "total_blocked": null_if(None::<u64>),
        "total_flagged": null_if(None::<u64>),
        "hallucination_rate": null_if(None::<f64>),
        "risk_score_avg": null_if(None::<f64>),
        "pre_gen_blocked": null_if(None::<u64>),
        "in_gen_corrected": null_if(None::<u64>),
        "post_gen_flagged": null_if(None::<u64>),
        "top_risk_sources": null_if(None::<Vec<(String, f64)>>),
    }))
}

pub async fn telemetry_training(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    Json(json!({
        "is_training": null_if(None::<bool>),
        "current_epoch": null_if(None::<u32>),
        "total_epochs": null_if(None::<u32>),
        "current_step": null_if(None::<u64>),
        "total_steps": null_if(None::<u64>),
        "loss": null_if(None::<f64>),
        "learning_rate": null_if(None::<f64>),
        "grad_norm": null_if(None::<f64>),
        "tokens_per_second": null_if(None::<f64>),
        "samples_processed": null_if(None::<u64>),
        "time_elapsed_secs": null_if(None::<f64>),
        "estimated_time_remaining_secs": null_if(None::<f64>),
    }))
}

pub async fn telemetry_models(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let models: Vec<Value> = vec![];
    Json(json!(models))
}

pub async fn telemetry_token_flows(
    _nexora: Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let flows: Vec<Value> = vec![];
    Json(json!(flows))
}
