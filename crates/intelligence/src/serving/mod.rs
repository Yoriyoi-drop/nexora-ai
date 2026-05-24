//! Model Serving
//!
//! Real serving layer: OpenAI-compatible endpoint wired to an InferenceBackend trait.
//! AppState holds an Arc<dyn InferenceBackend> that is REQUIRED at startup.
//! The server will not start without a registered backend.

pub mod unified_api;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

// ─── InferenceBackend trait ───────────────────────────────────────────────────

/// Single interface required by the serving layer.
/// Implement this on any struct that wraps a real model (nexora-intelligence,
/// or a remote endpoint).
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Generate text for a prompt.
    /// Returns `(generated_text, tokens_generated)`.
    async fn generate(
        &self,
        model_id: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<(String, usize)>;

    /// Backend identifier for logging.
    fn backend_name(&self) -> &str;
}

// ─── Serving config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub request_timeout_ms: u64,
    pub enable_load_balancing: bool,
    pub health_check_interval_ms: u64,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8081,
            max_concurrent_requests: 100,
            request_timeout_ms: 30000,
            enable_load_balancing: true,
            health_check_interval_ms: 5000,
        }
    }
}

// ─── HTTP types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct HealthResponse {
    status: String,
    version: &'static str,
    uptime_seconds: u64,
    active_requests: usize,
    backend: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InferenceRequest {
    model_id: String,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct InferenceResponse {
    generated_text: String,
    tokens_generated: usize,
    inference_time_ms: u64,
}

// ─── AppState ─────────────────────────────────────────────────────────────────

struct ServerState {
    started_at: tokio::time::Instant,
    request_count: std::sync::atomic::AtomicUsize,
}

impl ServerState {
    fn new() -> Self {
        Self {
            started_at: tokio::time::Instant::now(),
            request_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

struct AppState {
    config: ServingConfig,
    server_state: ServerState,
    semaphore: Semaphore,
    /// The real inference backend (always present - required at startup).
    inference: Arc<dyn InferenceBackend>,
}

// ─── ModelServer ──────────────────────────────────────────────────────────────

pub struct ModelServer {
    config: ServingConfig,
    inference: Arc<dyn InferenceBackend>,
}

impl ModelServer {
    pub fn new(config: ServingConfig, backend: Arc<dyn InferenceBackend>) -> Self {
        Self { config, inference: backend }
    }

    pub async fn start(&self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        info!(
            "Starting model server on {} (backend: {})",
            addr,
            self.inference.backend_name()
        );

        let app_state = Arc::new(AppState {
            config: self.config.clone(),
            server_state: ServerState::new(),
            semaphore: Semaphore::new(self.config.max_concurrent_requests),
            inference: self.inference.clone(),
        });

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/infer", post(infer_handler))
            .route("/v1/chat/completions", post(openai_chat_handler))
            .with_state(app_state);

        let listener = TcpListener::bind(&addr).await?;
        info!("Model server listening on {}", addr);
        axum::serve(listener, app).await?;
        Ok(())
    }
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let uptime = state.server_state.started_at.elapsed().as_secs();
    let active = state
        .server_state
        .request_count
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: uptime,
        active_requests: active,
        backend: Some(state.inference.backend_name().to_string()),
    })
}

async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (StatusCode::SERVICE_UNAVAILABLE, "Too many concurrent requests".to_string())
    })?;
    state.server_state.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let backend = state.inference.clone();
    let timeout_ms = state.config.request_timeout_ms;

    if req.prompt.is_empty() {
        state.server_state.request_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        return Err((StatusCode::BAD_REQUEST, "Empty prompt".to_string()));
    }

    let task = async move {
        let start = std::time::Instant::now();
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(1.0);
        backend
            .generate(&req.model_id, &req.prompt, max_tokens, temperature)
            .await
            .map(|(text, tokens)| {
                let ms = start.elapsed().as_millis() as u64;
                info!("Inference for '{}' done in {}ms ({} tokens)", req.model_id, ms, tokens);
                Json(InferenceResponse {
                    generated_text: text,
                    tokens_generated: tokens,
                    inference_time_ms: ms,
                })
            })
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
    };

    let result =
        tokio::time::timeout(Duration::from_millis(timeout_ms), task).await;

    state.server_state.request_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok(r) => r,
        Err(_) => Err((StatusCode::REQUEST_TIMEOUT, "Request timed out".to_string())),
    }
}

/// OpenAI-compatible `/v1/chat/completions` endpoint.
/// Extracts messages, concatenates them into a prompt, and routes through the
/// registered InferenceBackend. Returns an OpenAI-schema response.
async fn openai_chat_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        (StatusCode::SERVICE_UNAVAILABLE, "Too many concurrent requests".to_string())
    })?;
    state.server_state.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let backend = state.inference.clone();
    let timeout_ms = state.config.request_timeout_ms;

    let model_id = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("nexora-default")
        .to_string();

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;

    let temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    // Flatten messages into a single prompt string (role: content format)
    let prompt = body
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|messages| {
            messages
                .iter()
                .filter_map(|m| {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if content.is_empty() {
                        None
                    } else {
                        Some(format!("{}: {}", role, content))
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    if prompt.is_empty() {
        state.server_state.request_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        return Err((StatusCode::BAD_REQUEST, "No message content provided.".to_string()));
    }

    info!(
        "OpenAI /v1/chat/completions: model='{}', prompt_len={}, max_tokens={}, temperature={}",
        model_id,
        prompt.len(),
        max_tokens,
        temperature
    );

    let model_id_clone = model_id.clone();
    let task = async move {
        let start = std::time::Instant::now();
        backend
            .generate(&model_id_clone, &prompt, max_tokens, temperature)
            .await
            .map(|(text, tokens_generated)| {
                let elapsed = start.elapsed().as_millis() as u64;
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                // Approximate prompt token count (4 chars/token heuristic)
                let prompt_tokens = prompt.len() / 4;

                Json(serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id_clone,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": tokens_generated,
                        "total_tokens": prompt_tokens + tokens_generated
                    },
                    "inference_time_ms": elapsed
                }))
            })
            .map_err(|e| {
                error!("Inference backend error: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            })
    };

    let result = tokio::time::timeout(Duration::from_millis(timeout_ms), task).await;
    state.server_state.request_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

    match result {
        Ok(r) => r,
        Err(_) => {
            warn!("OpenAI endpoint timed out for model '{}'", model_id);
            Err((StatusCode::REQUEST_TIMEOUT, "Inference timed out".to_string()))
        }
    }
}
