//! Router setup and middleware configuration

use std::sync::Arc;
use anyhow::Result;
use axum::{Router, routing::get, routing::post, Extension};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use crate::NexoraAI;
use super::config::ServerConfig;
use super::handlers::*;

/// Create application router with all endpoints
pub async fn create_router(
    nexora: Arc<NexoraAI>,
    config: &ServerConfig,
) -> Result<Router> {
    let mut app = Router::new()
        // Health check endpoints - temporarily commented out due to handler trait issues
        // .route("/health", get(health_check))
        // .route("/health/detailed", get(detailed_health_check))
        
        // System information endpoints - temporarily commented out due to axum handler trait issues
        // .route("/info", get(system_info))
        // .route("/info/performance", get(performance_metrics))
        // .route("/info/memory", get(memory_stats))
        
        // Core AI endpoints
        .route("/process", post(process_request))
        .route("/generate", post(generate_text))
        .route("/chat", post(chat))
        
        // Code-related endpoints
        .route("/code/analyze", post(analyze_code))
        .route("/code/generate", post(generate_code))
        
        // Configuration endpoints
        .route("/config", get(get_config))
        .route("/config", post(update_config))
        
        // Static files and web interface
        .route("/", get(index))
        .route("/static/*path", get(static_files))
        
        // Extension for shared state
        .layer(Extension(nexora));
    
    // Add CORS if enabled
    if config.enable_cors {
        app = add_cors_layer(app, config)?;
    }
    
    // Add tracing middleware
    app = app.layer(TraceLayer::new_for_http());
    
    info!("Router configured with {} endpoints", 12);
    Ok(app)
}

/// Add CORS layer to router
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
