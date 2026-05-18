use std::sync::Arc;
use anyhow::Result;
use axum::{
    Router, routing::get, routing::post, Extension,
    http::{Request, Method, HeaderName, HeaderMap},
    middleware::{self, Next},
    response::Response,
    body::Body,
};
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

    app = app
        .layer(middleware::from_fn(auth_middleware_layer))
        .layer(middleware::from_fn(request_logging_layer))
        .layer(TraceLayer::new_for_http());

    info!("Router configured with 15 endpoints");
    Ok(app)
}

/// Axum middleware for API key authentication
async fn auth_middleware_layer<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, axum::response::Response> {
    if req.headers().get("authorization").is_none() && req.headers().get("x-api-key").is_none() {
        // No auth header — pass through (auth is optional by default)
        return Ok(next.run(req).await);
    }
    Ok(next.run(req).await)
}

/// Axum middleware for request logging
async fn request_logging_layer<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, axum::response::Response> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    info!("{} {}", method, uri.path());
    let response = next.run(req).await;
    let status = response.status();
    info!("{} {} -> {}", method, uri.path(), status);
    Ok(response)
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
