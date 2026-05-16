use std::sync::Arc;
use axum::{Json, Extension, response::Html, extract::Path, response::IntoResponse};
use serde_json::{json, Value};
use tracing::{info, error};
use std::sync::OnceLock;
use nexora_monitoring::MetricsCollector;

use crate::NexoraAI;

static METRICS: OnceLock<Arc<MetricsCollector>> = OnceLock::new();

pub fn init_metrics() -> Arc<MetricsCollector> {
    let collector = Arc::new(MetricsCollector::new());
    METRICS.set(collector.clone()).ok();
    collector
}

pub fn metrics_collector() -> Option<&'static Arc<MetricsCollector>> {
    METRICS.get()
}

pub async fn health_check(
    Extension(nexora): Extension<Arc<NexoraAI>>
) -> Json<Value> {
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

pub async fn detailed_health_check(
    Extension(nexora): Extension<Arc<NexoraAI>>
) -> Json<Value> {
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
    match metrics_collector() {
        Some(m) => {
            let body = m.gather_prometheus();
            axum::http::Response::builder()
                .header("Content-Type", "text/plain; charset=utf-8")
                .body(axum::body::Body::from(body))
                .unwrap()
        }
        None => axum::http::Response::builder()
            .header("Content-Type", "text/plain; charset=utf-8")
            .body(axum::body::Body::from("# metrics disabled"))
            .unwrap(),
    }
}

pub async fn system_info(
    Extension(nexora): Extension<Arc<NexoraAI>>
) -> impl IntoResponse {
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

pub async fn performance_metrics(
    Extension(nexora): Extension<Arc<NexoraAI>>
) -> impl IntoResponse {
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

pub async fn memory_stats(
    Extension(nexora): Extension<Arc<NexoraAI>>
) -> impl IntoResponse {
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
    let input = payload.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    info!("Processing request: {}", input);

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
    let prompt = payload.get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let max_tokens = payload.get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as usize;

    let temperature = payload.get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    info!("Generating text: prompt='{}', max_tokens={}, temperature={}", 
          prompt, max_tokens, temperature);

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

pub async fn chat(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let message = payload.get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let conversation_id = payload.get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    info!("Chat message: {} (conversation_id: {:?})", message, conversation_id);

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
    let code = payload.get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let language = payload.get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    info!("Analyzing code: {} chars, language: {}", code.len(), language);

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
    let description = payload.get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let language = payload.get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("rust");

    info!("Generating code: description='{}', language='{}'", description, language);

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

pub async fn update_config(Json(payload): Json<Value>) -> Json<Value> {
    info!("Processing configuration update request");

    let config_result = validate_and_process_config(&payload);

    match config_result {
        Ok(updated_config) => {
            info!("Configuration updated successfully");
            Json(json!({
                "success": true,
                "message": "Configuration updated successfully",
                "updated_fields": updated_config.updated_fields,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        },
        Err(e) => {
            error!("Configuration update failed: {}", e);
            Json(json!({
                "success": false,
                "message": format!("Configuration update failed: {}", e),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
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

        if let Some(max_connections) = server_config.get("max_connections").and_then(|v| v.as_u64()) {
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

            if let Some(requests_per_minute) = rate_limit.get("requests_per_minute").and_then(|v| v.as_u64()) {
                if requests_per_minute > 0 && requests_per_minute <= 1000 {
                    updated_fields.push(format!("api.rate_limit.requests_per_minute: {}", requests_per_minute));
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
                },
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
    #[allow(dead_code)]
    timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

pub async fn static_files(Path(path): Path<String>) -> Result<axum::response::Response, axum::http::StatusCode> {
    let base_path = std::path::Path::new("static");
    let file_path = base_path.join(&path);

    if file_path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(axum::http::StatusCode::FORBIDDEN);
    }

    let content = match std::fs::read(&file_path) {
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
