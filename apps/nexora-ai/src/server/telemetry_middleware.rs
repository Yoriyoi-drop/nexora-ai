use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
    Extension,
};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use crate::NexoraAI;
use crate::server::router::ApiKey;

pub async fn telemetry_middleware_layer(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Extension(api_key): Extension<ApiKey>,
    req: Request,
    next: Next,
) -> Result<Response, axum::response::Response> {
    let start = Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    let response = next.run(req).await;
    let latency = start.elapsed().as_millis() as u64;
    let status = response.status();

    info!(
        method = %method,
        path = %path,
        status = status.as_u16(),
        latency_ms = latency,
        api_key = %api_key.0.get(..8.min(api_key.0.len())).unwrap_or("?"),
        "telemetry: request"
    );

    Ok(response)
}
