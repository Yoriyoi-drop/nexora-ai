use std::sync::Arc;
use anyhow::Result;
use axum::{Router, routing::get, routing::post, Extension};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use crate::NexoraAI;
use super::config::ServerConfig;
use super::handlers::*;

pub async fn create_router(
    nexora: Arc<NexoraAI>,
    config: &ServerConfig,
) -> Result<Router> {
    let _ = init_metrics();

    let mut app = Router::new()
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health_check))
        .route("/metrics", get(metrics_handler))

        .route("/info", get(system_info))
        .route("/info/performance", get(performance_metrics))
        .route("/info/memory", get(memory_stats))

        .route("/process", post(process_request))
        .route("/generate", post(generate_text))
        .route("/chat", post(chat))

        .route("/code/analyze", post(analyze_code))
        .route("/code/generate", post(generate_code))

        .route("/config", get(get_config))
        .route("/config", post(update_config))

        .route("/", get(index))
        .route("/static/*path", get(static_files))

        .layer(Extension(nexora));

    if config.enable_cors {
        app = add_cors_layer(app, config)?;
    }

    app = app.layer(TraceLayer::new_for_http());

    info!("Router configured with 13 endpoints");
    Ok(app)
}

fn add_cors_layer(mut app: Router, config: &ServerConfig) -> Result<Router> {
    let origins: Result<Vec<_>, _> = config.cors_origins.clone()
        .into_iter()
        .map(|origin| origin.parse().map_err(|e| anyhow::anyhow!("Invalid origin '{}': {}", origin, e)))
        .collect();

    let methods: Result<Vec<_>, _> = vec!["GET", "POST", "PUT", "DELETE"]
        .into_iter()
        .map(|method| method.parse().map_err(|e| anyhow::anyhow!("Invalid method '{}': {}", method, e)))
        .collect();

    let headers: Result<Vec<_>, _> = vec!["Content-Type", "Authorization"]
        .into_iter()
        .map(|header| header.parse().map_err(|e| anyhow::anyhow!("Invalid header '{}': {}", header, e)))
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(origins.map_err(|e| anyhow::anyhow!("Failed to parse CORS origins: {}", e))?)
        .allow_methods(methods.map_err(|e| anyhow::anyhow!("Failed to parse CORS methods: {}", e))?)
        .allow_headers(headers.map_err(|e| anyhow::anyhow!("Failed to parse CORS headers: {}", e))?);

    app = app.layer(cors);
    Ok(app)
}
