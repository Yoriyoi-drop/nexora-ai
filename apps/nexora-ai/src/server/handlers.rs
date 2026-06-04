use axum::{
    extract::Path,
    http::StatusCode,
    response::sse::{Event, Sse},
    response::Html,
    response::IntoResponse,
    Extension, Json,
};
use axum::http::HeaderMap;
use futures::stream::{self, Stream};
use nexora_monitoring::MetricsCollector;
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::{error, info, warn};

use crate::security::{SecurityConfig, SecurityValidator};
use crate::NexoraAI;

static METRICS: OnceLock<Arc<MetricsCollector>> = OnceLock::new();
static SECURITY: OnceLock<SecurityValidator> = OnceLock::new();

fn init_security_validator() -> &'static SecurityValidator {
    SECURITY.get_or_init(|| {
        SecurityValidator::new(SecurityConfig::default())
    })
}

fn validate_prompt(prompt: &str) -> Result<(), (StatusCode, Json<Value>)> {
    let validator = init_security_validator();
    validator.validate_input(prompt).map_err(|e| {
        warn!("Security validation rejected input: {}", e);
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "Input rejected by security validator",
                "detail": e.to_string()
            })),
        )
    })
}

pub fn init_metrics() -> anyhow::Result<Arc<MetricsCollector>> {
    let collector = Arc::new(
        MetricsCollector::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize metrics collector: {}", e))?,
    );
    METRICS.set(collector.clone()).ok();
    Ok(collector)
}

pub fn metrics_collector() -> Option<&'static Arc<MetricsCollector>> {
    METRICS.get()
}

pub async fn health_check(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    let start = std::time::Instant::now();
    match nexora.health_check().await {
        Ok(health) => {
            if let Some(m) = metrics_collector() {
                m.record_request(true, start.elapsed().as_secs_f64());
            }
            Json(json!({
                "healthy": health.healthy,
                "timestamp": health.last_check,
                "version": env!("CARGO_PKG_VERSION")
            }))
        }
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Health check failed: {}", e);
            Json(json!({
                "healthy": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn detailed_health_check(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    match nexora.health_check().await {
        Ok(health) => Json(json!(health)),
        Err(e) => {
            error!("Detailed health check failed: {}", e);
            Json(json!({
                "healthy": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn metrics_handler() -> axum::response::Response {
    let body = match metrics_collector() {
        Some(m) => m.gather_prometheus(),
        None => String::new(),
    };
    axum::http::Response::builder()
        .header("Content-Type", "text/plain; charset=utf-8")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|e| {
            axum::http::Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("metrics error: {}", e)))
                .unwrap_or_else(|_| {
                    axum::http::Response::new(axum::body::Body::from("metrics error"))
                })
        })
}

pub async fn gpu_health_handler() -> Json<serde_json::Value> {
    let obs = nexora_inference::inference_trait::observability_snapshot();
    Json(serde_json::json!({
        "gpu_alive": obs.gpu_alive,
        "gpu_forward_errors": obs.gpu_forward_errors,
        "gpu_cpu_fallbacks": obs.gpu_cpu_fallbacks,
        "gpu_resident_successes": obs.gpu_resident_successes,
        "gpu_resident_fallbacks": obs.gpu_resident_fallbacks,
        "gpu_tokens_generated": obs.gpu_tokens_generated,
        "cpu_tokens_generated": obs.cpu_tokens_generated,
        "gpu_resident_ratio": obs.gpu_resident_ratio,
    }))
}

pub async fn system_info(Extension(nexora): Extension<Arc<NexoraAI>>) -> impl IntoResponse {
    match nexora.get_system_info().await {
        Ok(info) => Json(json!(info)),
        Err(e) => {
            error!("Failed to get system info: {}", e);
            Json(json!({
                "healthy": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn performance_metrics(Extension(nexora): Extension<Arc<NexoraAI>>) -> impl IntoResponse {
    match nexora.get_performance_metrics().await {
        Ok(metrics) => Json(metrics),
        Err(e) => {
            error!("Failed to get performance metrics: {}", e);
            Json(json!({
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn memory_stats(Extension(nexora): Extension<Arc<NexoraAI>>) -> impl IntoResponse {
    match nexora.get_system_info().await {
        Ok(info) => Json(json!({
            "total_memory": info.memory_stats.total_memory,
            "used_memory": info.memory_stats.used_memory,
            "available_memory": info.memory_stats.available_memory,
            "cache_size": info.memory_stats.cache_size,
            "usage_percent": info.memory_usage,
            "timestamp": info.last_updated
        })),
        Err(e) => {
            error!("Failed to get memory stats: {}", e);
            Json(json!({
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn process_request(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let input = payload.get("input").and_then(|v| v.as_str()).unwrap_or("");

    if let Err((_status, err_json)) = validate_prompt(input) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let truncated: String = input.chars().take(100).collect();
    info!(
        "Processing request: {} [truncated {} chars]",
        truncated,
        input.len().min(100)
    );

    match nexora.process_request(input).await {
        Ok(response) => Json(json!({
            "success": true,
            "response": response,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Request processing failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn generate_text(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    if let Err((_status, err_json)) = validate_prompt(prompt) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let max_tokens = payload
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as usize;

    let temperature = payload
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let truncated: String = prompt.chars().take(100).collect();
    info!(
        "Generating text: prompt='{}' [truncated {} chars], max_tokens={}, temperature={}",
        truncated,
        prompt.len().min(100),
        max_tokens,
        temperature
    );

    match nexora.generate_text(prompt, max_tokens, temperature).await {
        Ok(generated) => Json(json!({
            "success": true,
            "generated_text": generated,
            "parameters": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Text generation failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn generate_text_stream(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Result<Sse<impl Stream<Item = Result<Event, anyhow::Error>>>, (StatusCode, Json<Value>)> {
    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    if prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Prompt cannot be empty" })),
        ));
    }

    validate_prompt(prompt)?;

    let max_tokens = payload
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;

    let temperature = payload
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    info!(
        "Streaming text generation: prompt={}, max_tokens={}, temperature={}",
        prompt.len(),
        max_tokens,
        temperature
    );

    let rx = nexora
        .generate_text_stream(prompt, max_tokens, temperature)
        .await
        .map_err(|e| {
            error!("Streaming inference init failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
        })?;

    let stream = stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Some(token) => {
                let data = serde_json::json!({
                    "token": token.token_text.to_string(),
                    "position": token.position,
                });
                let event = Event::default().data(data.to_string());
                Some((Ok::<_, anyhow::Error>(event), rx))
            }
            None => {
                let event = Event::default().data("[DONE]");
                Some((Ok(event), rx))
            }
        }
    });

    Ok(Sse::new(stream))
}

pub async fn chat(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let message = payload
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if let Err((_status, err_json)) = validate_prompt(message) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let conversation_id = payload
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let truncated = &message[..message.len().min(100)];
    info!(
        "Chat message: {} [truncated {} chars] (conversation_id: {:?})",
        truncated,
        message.len().min(100),
        conversation_id
    );

    match nexora.chat(message, conversation_id).await {
        Ok(response) => Json(json!({
            "success": true,
            "response": response,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Chat failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn analyze_code(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let code = payload.get("code").and_then(|v| v.as_str()).unwrap_or("");

    if let Err((_status, err_json)) = validate_prompt(code) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let language = payload
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    info!(
        "Analyzing code: {} chars, language: {}",
        code.len(),
        language
    );

    match nexora.analyze_code(code, language).await {
        Ok(analysis) => Json(json!({
            "success": true,
            "analysis": analysis,
            "metadata": {
                "code_length": code.len(),
                "language": language,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }
        })),
        Err(e) => {
            error!("Code analysis failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn generate_code(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let description = payload
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if let Err((_status, err_json)) = validate_prompt(description) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let language = payload
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("rust");

    info!(
        "Generating code: description='{}', language='{}'",
        description, language
    );

    match nexora.generate_code(description, language).await {
        Ok(code) => Json(json!({
            "success": true,
            "generated_code": code,
            "metadata": {
                "description": description,
                "language": language,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }
        })),
        Err(e) => {
            error!("Code generation failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn get_config() -> Json<Value> {
    Json(json!({
        "server": {
            "version": env!("CARGO_PKG_VERSION"),
            "features": ["text_generation", "code_analysis", "chat", "health_check", "metrics"],
            "endpoints": [
                "/health", "/health/detailed", "/metrics",
                "/info", "/info/performance", "/info/memory",
                "/process", "/generate", "/chat",
                "/code/analyze", "/code/generate", "/config"
            ]
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

pub async fn update_config(
    headers: HeaderMap,
    Extension(_nexora): Extension<Arc<NexoraAI>>,
    api_key: Option<Extension<crate::server::router::ApiKey>>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    info!("Processing configuration update request");

    // Require authentication for config changes
    validate_admin(&headers, api_key.as_ref().map(|e| &e.0))?;

    let config_result = validate_and_process_config(&payload);

    match config_result {
        Ok(updated_config) => {
            info!("Configuration updated successfully");
            Ok(Json(json!({
                "success": true,
                "message": "Configuration updated successfully",
                "updated_fields": updated_config.updated_fields,
                "timestamp": chrono::Utc::now().to_rfc3339()
            })))
        }
        Err(e) => {
            error!("Configuration update failed: {}", e);
            Ok(Json(json!({
                "success": false,
                "message": format!("Configuration update failed: {}", e),
                "timestamp": chrono::Utc::now().to_rfc3339()
            })))
        }
    }
}

fn validate_and_process_config(payload: &Value) -> Result<ConfigUpdateResult, anyhow::Error> {
    let mut updated_fields = Vec::new();

    if let Some(server_config) = payload.get("server") {
        if let Some(host) = server_config.get("host").and_then(|v| v.as_str()) {
            if !host.is_empty() {
                updated_fields.push(format!("server.host: {}", host));
            }
        }

        if let Some(port) = server_config.get("port").and_then(|v| v.as_u64()) {
            if port > 0 && port <= 65535 {
                updated_fields.push(format!("server.port: {}", port));
            } else {
                return Err(anyhow::anyhow!("Invalid port number: {}", port));
            }
        }

        if let Some(max_connections) = server_config
            .get("max_connections")
            .and_then(|v| v.as_u64())
        {
            if max_connections > 0 && max_connections <= 10000 {
                updated_fields.push(format!("server.max_connections: {}", max_connections));
            }
        }
    }

    if let Some(api_config) = payload.get("api") {
        if let Some(timeout) = api_config.get("timeout_seconds").and_then(|v| v.as_u64()) {
            if timeout >= 1 && timeout <= 300 {
                updated_fields.push(format!("api.timeout_seconds: {}", timeout));
            }
        }

        if let Some(rate_limit) = api_config.get("rate_limit") {
            if let Some(enabled) = rate_limit.get("enabled").and_then(|v| v.as_bool()) {
                updated_fields.push(format!("api.rate_limit.enabled: {}", enabled));
            }

            if let Some(requests_per_minute) = rate_limit
                .get("requests_per_minute")
                .and_then(|v| v.as_u64())
            {
                if requests_per_minute > 0 && requests_per_minute <= 1000 {
                    updated_fields.push(format!(
                        "api.rate_limit.requests_per_minute: {}",
                        requests_per_minute
                    ));
                }
            }
        }
    }

    if let Some(model_config) = payload.get("models") {
        if let Some(default_model) = model_config.get("default").and_then(|v| v.as_str()) {
            if !default_model.is_empty() {
                updated_fields.push(format!("models.default: {}", default_model));
            }
        }

        if let Some(max_tokens) = model_config.get("max_tokens").and_then(|v| v.as_u64()) {
            if max_tokens > 0 && max_tokens <= 8192 {
                updated_fields.push(format!("models.max_tokens: {}", max_tokens));
            }
        }

        if let Some(temperature) = model_config.get("temperature").and_then(|v| v.as_f64()) {
            if temperature >= 0.0 && temperature <= 2.0 {
                updated_fields.push(format!("models.temperature: {}", temperature));
            } else {
                return Err(anyhow::anyhow!("Temperature must be between 0.0 and 2.0"));
            }
        }
    }

    if let Some(logging_config) = payload.get("logging") {
        if let Some(level) = logging_config.get("level").and_then(|v| v.as_str()) {
            match level {
                "debug" | "info" | "warn" | "error" => {
                    updated_fields.push(format!("logging.level: {}", level));
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid log level: {}", level));
                }
            }
        }

        if let Some(enabled) = logging_config.get("file_enabled").and_then(|v| v.as_bool()) {
            updated_fields.push(format!("logging.file_enabled: {}", enabled));
        }
    }

    if updated_fields.is_empty() {
        return Err(anyhow::anyhow!("No valid configuration fields provided"));
    }

    Ok(ConfigUpdateResult {
        updated_fields,
        timestamp: chrono::Utc::now(),
    })
}

#[derive(Debug)]
struct ConfigUpdateResult {
    updated_fields: Vec<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn get_train_metrics(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    let metrics = nexora.train_metrics.read().await;
    Json(metrics.clone())
}

pub async fn post_train_metrics(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let mut metrics = nexora.train_metrics.write().await;
    *metrics = payload.clone();
    Json(json!({ "success": true }))
}

/// Validate that the request has admin-level authentication.
/// Checks NEXORA_ADMIN_KEY env var match, then falls back to
/// the ApiKey extension from auth middleware (set after successful auth).
fn validate_admin(
    headers: &HeaderMap,
    api_key_opt: Option<&crate::server::router::ApiKey>,
) -> Result<(), (StatusCode, Json<Value>)> {
    // Check NEXORA_ADMIN_KEY env var
    if let Ok(admin_key) = std::env::var("NEXORA_ADMIN_KEY") {
        let provided = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .or_else(|| {
                headers
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
            });
        if provided == Some(&admin_key) {
            return Ok(());
        }
        return Err((
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Forbidden: invalid or missing admin key"})),
        ));
    }

    // No NEXORA_ADMIN_KEY: check if request came through auth middleware
    if api_key_opt.is_some() {
        return Ok(());
    }

    #[cfg(not(feature = "server-auth"))]
    {
        tracing::warn!("No NEXORA_ADMIN_KEY and no server-auth feature; config changes allowed without auth");
        return Ok(());
    }

    #[cfg(feature = "server-auth")]
    {
        Err((
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Forbidden: admin authentication required"})),
        ))
    }
}

/// GET /api/tiers — show loaded tiers, VRAM usage, hit counts
pub async fn tier_status(
    Extension(nexora): Extension<Arc<NexoraAI>>,
) -> Json<Value> {
    let loaded_tiers: Vec<String> = nexora.tier_router().loaded_tiers()
        .iter()
        .map(|t| format!("{:?}", t))
        .collect();
    let hit_counts: Vec<Value> = nexora.tier_router().hit_counts()
        .iter()
        .map(|(t, c)| json!({ "tier": format!("{:?}", t), "hits": c }))
        .collect();

    Json(json!({
        "loaded_tiers": loaded_tiers,
        "vram_used_mb": nexora.tier_router().total_vram_used_mb(),
        "vram_budget_mb": nexora.tier_router().config().vram_budget_mb,
        "eviction_threshold": nexora.tier_router().config().eviction_threshold,
        "max_tiers_in_vram": nexora.tier_router().config().max_tiers_in_vram,
        "tier_hit_counts": hit_counts,
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

pub async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

pub async fn static_files(
    Path(path): Path<String>,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    let base_path = std::path::Path::new("static")
        .canonicalize()
        .map_err(|_| axum::http::StatusCode::FORBIDDEN)?;
    let file_path = base_path
        .join(&path)
        .canonicalize()
        .map_err(|_| axum::http::StatusCode::FORBIDDEN)?;

    if !file_path.starts_with(&base_path) {
        return Err(axum::http::StatusCode::FORBIDDEN);
    }

    let content = match tokio::fs::read(&file_path).await {
        Ok(c) => c,
        Err(_) => return Err(axum::http::StatusCode::NOT_FOUND),
    };

    let content_type = match path.split('.').last() {
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("html") => "text/html",
        _ => "text/plain",
    };

    Ok(axum::http::Response::builder()
        .status(200)
        .header("Content-Type", content_type)
        .body(axum::body::Body::from(content))
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?)
}
