use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Extension, Json,
};
use std::sync::Arc;
use std::time::Instant;
use serde::Serialize;
use tracing::warn;

use crate::NexoraAI;
use super::ApiKey;

#[derive(Serialize)]
struct RateLimitError {
    error: String,
    retry_after_secs: u64,
}

/// Middleware that records all requests to Telemetry and enforces rate limits.
/// Must be placed AFTER auth middleware so `ApiKey` is available.
pub async fn telemetry_middleware_layer(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Extension(api_key): Extension<ApiKey>,
    req: Request,
    next: Next,
) -> Result<Response, axum::response::Response> {
    let start = Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let key_str = api_key.0.clone();

    if nexora.config.server.enable_auth {
        let result = nexora.telemetry.rate_limiter.check(&key_str).await;
        if !result.is_allowed() {
            let retry = match result {
                nexora_telemetry::rate_limiter::RateLimitResult::Denied { retry_after } => retry_after,
                _ => 1,
            };
            warn!("Rate limit exceeded for key {} on {}", &key_str[..key_str.len().min(8)], path);
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(RateLimitError {
                    error: "Rate limit exceeded".into(),
                    retry_after_secs: retry,
                }),
            ).into_response());
        }
    }

    let response = next.run(req).await;
    let latency = start.elapsed().as_millis() as u64;
    let status = response.status();

    nexora.telemetry.recorder.record_request(
        method.as_str(),
        &path,
        status.as_u16(),
        latency,
        Some(key_str),
        None,
    ).await;

    Ok(response)
}
