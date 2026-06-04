/// HTTP-based all-reduce for sharded tensor parallelism.
///
/// Architecture:
///   Shard 0 runs the reduce coordinator. All shards POST their partial
///   attention/FFN outputs to shard 0. Shard 0 accumulates and, when all
///   partials are received, returns the sum.
///
///   Each layer × phase (attn/ffn) gets a unique phase ID for ordering.
///
/// Flow (on each shard):
///   1. Compute local partial output
///   2. POST to coordinator: `{ "phase_id": "L3_attn", "shard_rank": 1, "data": [...] }`
///   3. Coordinator accumulates, responds with `{ "full_data": [...] }` when all received
///   4. Each shard continues with the full (summed) output

use std::collections::HashMap;
use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use ndarray::Array2;

use nexora_transformer::TransformerResult;

// ---------------------------------------------------------------------------
// Payload types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceRequest {
    /// Opaque phase identifier, e.g. "0_attn", "3_ffn"
    pub phase_id: String,
    /// Which shard is sending this partial
    pub shard_rank: usize,
    /// Total number of shards (for validation)
    pub num_shards: usize,
    /// Flat f32 data of the partial output
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceResponse {
    /// The fully reduced (summed) data, present once all shards have reported
    pub full_data: Option<Vec<f32>>,
    /// true if we're still waiting for more shards
    pub waiting: bool,
}

// ---------------------------------------------------------------------------
// Global coordinator (singleton per process)
// ---------------------------------------------------------------------------

struct PhaseAccumulator {
    partials: Vec<Option<Vec<f32>>>,
    num_shards: usize,
    waiters: Vec<tokio::sync::oneshot::Sender<Vec<f32>>>,
}

pub struct ReduceCoordinator {
    phases: Mutex<HashMap<String, PhaseAccumulator>>,
}

impl ReduceCoordinator {
    pub fn new() -> Self {
        Self {
            phases: Mutex::new(HashMap::new()),
        }
    }
}

static GLOBAL_COORDINATOR: std::sync::OnceLock<ReduceCoordinator> = std::sync::OnceLock::new();

fn global_coordinator() -> &'static ReduceCoordinator {
    GLOBAL_COORDINATOR.get_or_init(|| ReduceCoordinator::new())
}

// ---------------------------------------------------------------------------
// HTTP handler (axum)
// ---------------------------------------------------------------------------

async fn handle_reduce(
    Json(req): Json<ReduceRequest>,
) -> Json<ReduceResponse> {
    let coord = global_coordinator();
    let mut phases = coord.phases.lock().await;
    let entry = phases.entry(req.phase_id.clone()).or_insert_with(|| {
        let mut partials = Vec::with_capacity(req.num_shards);
        partials.resize_with(req.num_shards, || None);
        PhaseAccumulator {
            partials,
            num_shards: req.num_shards,
            waiters: Vec::new(),
        }
    });

    entry.partials[req.shard_rank] = Some(req.data);

    if entry.partials.iter().all(|p| p.is_some()) {
        let mut sum: Option<Vec<f32>> = None;
        for p in entry.partials.iter() {
            if let Some(data) = p {
                match &mut sum {
                    None => sum = Some(data.clone()),
                    Some(ref mut acc) => {
                        for (a, b) in acc.iter_mut().zip(data.iter()) {
                            *a += b;
                        }
                    }
                }
            }
        }
        let full = sum.unwrap_or_default();
        for waker in entry.waiters.drain(..) {
            let _ = waker.send(full.clone());
        }
        phases.remove(&req.phase_id);
        Json(ReduceResponse {
            full_data: Some(full),
            waiting: false,
        })
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        entry.waiters.push(tx);
        drop(phases);

        match rx.await {
            Ok(full) => Json(ReduceResponse {
                full_data: Some(full),
                waiting: false,
            }),
            Err(_) => Json(ReduceResponse {
                full_data: None,
                waiting: false,
            }),
        }
    }
}

/// Register the reduce endpoint on an axum router.
pub fn add_reduce_route(router: Router) -> Router {
    router.route("/shard/reduce", post(handle_reduce))
}

// ---------------------------------------------------------------------------
// Client — runs on every shard
// ---------------------------------------------------------------------------

/// Synchronous HTTP all-reduce client.
/// Sends the local partial to the coordinator, waits for the full sum.
pub fn all_reduce_http(
    coordinator_url: &str,
    phase_id: &str,
    shard_rank: usize,
    num_shards: usize,
    data: &[f32],
) -> TransformerResult<Vec<f32>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| {
            nexora_transformer::TransformerError::Implementation(format!(
                "HTTP client build: {e}"
            ))
        })?;

    let payload = ReduceRequest {
        phase_id: phase_id.to_string(),
        shard_rank,
        num_shards,
        data: data.to_vec(),
    };

    let resp = client
        .post(format!("{}/shard/reduce", coordinator_url))
        .json(&payload)
        .send()
        .map_err(|e| {
            nexora_transformer::TransformerError::Implementation(format!(
                "HTTP reduce POST: {e}"
            ))
        })?;

    let result: ReduceResponse = resp.json().map_err(|e| {
        nexora_transformer::TransformerError::Implementation(format!(
            "HTTP reduce decode: {e}"
        ))
    })?;

    result.full_data.ok_or_else(|| {
        nexora_transformer::TransformerError::Implementation(
            "HTTP reduce got empty response".into(),
        )
    })
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a `CallbackCollective` that uses HTTP all-reduce against the given
/// coordinator URL.
pub fn create_http_collective(
    coordinator_url: String,
    shard_rank: usize,
    num_shards: usize,
    hidden_size: usize,
) -> nexora_transformer::sharded::CallbackCollective {
    let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let reduce_attn = {
        let url = coordinator_url.clone();
        let c = counter.clone();
        move |local: &Array2<f32>| {
            let phase_id = c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let data = local.as_slice().unwrap_or(&[]);
            let full = all_reduce_http(
                &url,
                &format!("{phase_id}_attn"),
                shard_rank,
                num_shards,
                data,
            )?;
            let rows = local.shape()[0];
            let cols = local.shape()[1];
            if rows * cols == full.len() {
                Ok(Array2::from_shape_vec((rows, cols), full).map_err(|e| {
                    nexora_transformer::TransformerError::Implementation(format!(
                        "HTTP reduce shape: {e}"
                    ))
                })?)
            } else {
                // Fallback: treat as flat (should not happen)
                Ok(Array2::from_shape_vec((full.len() / hidden_size, hidden_size), full).map_err(|e| {
                    nexora_transformer::TransformerError::Implementation(format!(
                        "HTTP reduce shape2: {e}"
                    ))
                })?)
            }
        }
    };

    let reduce_ffn = {
        let url = coordinator_url;
        let c = counter;
        move |local: &Array2<f32>| {
            let phase_id = c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let data = local.as_slice().unwrap_or(&[]);
            let full = all_reduce_http(
                &url,
                &format!("{phase_id}_ffn"),
                shard_rank,
                num_shards,
                data,
            )?;
            let rows = local.shape()[0];
            let cols = local.shape()[1];
            if rows * cols == full.len() {
                Ok(Array2::from_shape_vec((rows, cols), full).map_err(|e| {
                    nexora_transformer::TransformerError::Implementation(format!(
                        "HTTP reduce shape: {e}"
                    ))
                })?)
            } else {
                Ok(Array2::from_shape_vec((full.len() / hidden_size, hidden_size), full).map_err(|e| {
                    nexora_transformer::TransformerError::Implementation(format!(
                        "HTTP reduce shape2: {e}"
                    ))
                })?)
            }
        }
    };

    nexora_transformer::sharded::CallbackCollective::new(
        num_shards,
        shard_rank,
        reduce_attn,
        reduce_ffn,
    )
}
