use std::sync::Arc;
use anyhow::Result;
use axum::{Router, routing::get, routing::post, Extension};
use axum::http::{Method, HeaderName};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use crate::NexoraAI;
use crate::config::server::ServerConfig;
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

        .route("/train/metrics", get(get_train_metrics))
        .route("/train/metrics", post(post_train_metrics))

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

    info!("Router configured with 15 endpoints");
    Ok(app)
}

fn add_cors_layer(mut app: Router, config: &ServerConfig) -> Result<Router> {
    let is_wildcard = config.cors_origins.iter().any(|o| o == "*");

    let cors = if is_wildcard {
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any)
    } else {
        let origins: Vec<_> = config.cors_origins.iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        let methods: Vec<Method> = vec!["GET", "POST", "PUT", "DELETE"]
            .into_iter().filter_map(|m| m.parse().ok()).collect();
        let headers: Vec<HeaderName> = vec!["Content-Type", "Authorization"]
            .into_iter().filter_map(|h| h.parse().ok()).collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(methods)
            .allow_headers(headers)
    };

    app = app.layer(cors);
    Ok(app)
}
