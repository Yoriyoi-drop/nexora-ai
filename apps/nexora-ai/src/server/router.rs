use anyhow::Result;
use axum::{
    extract::Request,
    http::{HeaderName, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Extension, Json, Router,
};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use super::agent_handlers::*;
use super::handlers::*;
use crate::config::server::ServerConfig;
use crate::security::{SecurityConfig, SecurityValidator};
use crate::NexoraAI;

static SECURITY: std::sync::OnceLock<SecurityValidator> = std::sync::OnceLock::new();
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);
static REQUEST_WINDOW_START: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn init_security_validator() -> &'static SecurityValidator {
    SECURITY.get_or_init(|| SecurityValidator::new(SecurityConfig::default()))
}

/// Newtype for API key extracted from auth middleware.
#[derive(Clone)]
pub struct ApiKey(pub String);

pub async fn create_router(nexora: Arc<NexoraAI>, config: &ServerConfig) -> Result<Router> {
    if let Err(e) = init_metrics() {
        tracing::warn!("Failed to initialize metrics collector: {}", e);
    }
    crate::metrics::BackgroundMetricsCollector::new(15).start();

    let valid_keys: Arc<HashSet<String>> = Arc::new(config.api_keys.iter().cloned().collect());
    let enable_auth = config.enable_auth;
    let rate_limiter: Option<RateLimiter> =
        config.rate_limit_rpm.filter(|&r| r > 0).map(RateLimiter::new);

    // Initialize global auth system for JWT / user auth (server-auth feature gate)
    #[cfg(feature = "server-auth")]
    let auth_system: Option<Arc<crate::auth::AuthSystem>> = {
        let auth = crate::api::client::init_global_auth("nexora-jwt-secret-change-me");
        Some(Arc::new(crate::auth::AuthSystem::new("nexora-jwt-secret-change-me")))
    };
    #[cfg(not(feature = "server-auth"))]
    let _auth_system: Option<Arc<crate::auth::AuthSystem>> = None;

    let mut app = Router::new()
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health_check))
        .route("/health/gpu", get(gpu_health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/info", get(system_info))
        .route("/info/performance", get(performance_metrics))
        .route("/info/memory", get(memory_stats))
        .route("/train/metrics", get(get_train_metrics))
        .route("/train/metrics", post(post_train_metrics))
        .route("/process", post(process_request))
        .route("/generate", post(generate_text))
        .route("/generate/stream", post(generate_text_stream))
        .route("/chat", post(chat))
        .route("/code/analyze", post(analyze_code))
        .route("/code/generate", post(generate_code))
        .route("/config", get(get_config))
        .route("/config", post(update_config))
        // Agent endpoints
        .route("/api/agents", get(list_agents))
        .route("/api/plans", post(create_plan))
        .route("/api/plans", get(list_plans))
        .route("/api/plans/:plan_id", get(get_plan))
        .route("/api/plans/dispatch", post(dispatch_plan))
        .route("/api/generate/agent", post(generate_via_agent))
        .route("/", get(index))
        .route("/static/*path", get(static_files))
        .layer(Extension(nexora));

    // Shard collective: POST /shard/reduce for HTTP-based all-reduce
    app = super::shard_collective::add_reduce_route(app);

    #[cfg(feature = "server-auth")]
    {
        use super::auth_handlers::*;
        app = app
            .route("/auth/register", post(register_handler))
            .route("/auth/login", post(login_handler))
            .route("/auth/profile", get(get_profile_handler))
            .route("/auth/profile", post(update_profile_handler))
            .route("/auth/keys", get(list_api_keys_handler))
            .route("/auth/keys", post(create_api_key_handler))
            .route("/auth/keys/:key_id", delete(revoke_api_key_handler))
            .route("/auth/keys/:key_id/rotate", post(rotate_api_key_handler));
        if let Some(auth) = auth_system {
            app = app.layer(Extension(auth));
        }
    }

    if config.enable_cors {
        app = add_cors_layer(app, config)?;
    }

    app = app
        .layer(Extension(valid_keys))
        .layer(Extension(enable_auth))
        .layer(middleware::from_fn(auth_middleware_layer));

    if let Some(limiter) = rate_limiter {
        app = app
            .layer(Extension(limiter))
            .layer(middleware::from_fn(rate_limit_layer));
    }

    app = app
        .layer(middleware::from_fn(request_logging_layer))
        .layer(TraceLayer::new_for_http());

    info!("Router configured with 21 endpoints");
    Ok(app)
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Axum middleware for API key authentication
/// Public routes (health, auth endpoints) always allowed.
/// When `enable_auth` is true, all non-public routes require a valid API key.
async fn auth_middleware_layer(
    Extension(valid_keys): Extension<Arc<HashSet<String>>>,
    Extension(enable_auth): Extension<bool>,
    req: Request,
    next: Next,
) -> Result<Response, axum::response::Response> {
    // Public routes always allowed (health checks, auth registration/login)
    let path = req.uri().path();
    if path.starts_with("/health")
        || path == "/auth/register"
        || path == "/auth/login"
    {
        return Ok(next.run(req).await);
    }

    // Auth disabled → dev mode, allow all non-public routes too
    if !enable_auth {
        return Ok(next.run(req).await);
    }

    // Extract header values as owned strings to avoid borrowing req
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok().map(|s| s.to_string()));
    let x_api_key = req
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok().map(|s| s.to_string()));

    let api_key_str = auth_header
        .as_deref()
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| x_api_key.as_deref())
        .map(|s| s.to_string());

    // Check both statically configured keys and AuthSystem (if available)
    let is_authorized = api_key_str.as_deref().map_or(false, |key| {
        if valid_keys.contains(key) {
            return true;
        }
        if let Some(auth) = crate::api::client::get_global_auth() {
            return tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(auth.authenticate_api_key(key))
            }).is_some();
        }
        false
    });

    if is_authorized {
        // Insert ApiKey into request extensions so auth handlers can access it
        let mut req = req;
        if let Some(ref key) = api_key_str {
            req.extensions_mut().insert(ApiKey(key.clone()));
        }
        Ok(next.run(req).await)
    } else {
        let resp = ErrorResponse {
            error: "Unauthorized: invalid or missing API key".into(),
        };
        Err((StatusCode::UNAUTHORIZED, Json(resp)).into_response())
    }
}

/// Per-IP sliding-window rate limiter state.
#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<RateLimiterInner>>,
    rpm: u64,
    window_secs: u64,
}

struct RateLimiterInner {
    /// Map client_ip → Vec of request timestamps within the current window.
    windows: HashMap<String, Vec<Instant>>,
}

impl RateLimiter {
    pub fn new(rpm: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RateLimiterInner {
                windows: HashMap::new(),
            })),
            rpm,
            window_secs: 60,
        }
    }

    /// Check and record a request. Returns `true` if allowed, `false` if rate-limited.
    pub fn check(&self, client_ip: &str) -> bool {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return true, // allow on lock poison
        };
        let now = Instant::now();
        let window = inner.windows.entry(client_ip.to_string()).or_default();
        // Prune timestamps outside the window
        window.retain(|t| now.duration_since(*t).as_secs() < self.window_secs);
        if window.len() >= self.rpm as usize {
            return false;
        }
        window.push(now);
        true
    }
}

/// Axum middleware for per-IP rate limiting + global rate limit validation.
/// Placed after auth middleware so we can fall back to the client IP.
async fn rate_limit_layer(
    req: Request,
    next: Next,
) -> Result<Response, axum::response::Response> {
    let client_ip = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next().map(|s| s.trim().to_string()))
        .or_else(|| {
            req.extensions()
                .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                .map(|ci| ci.0.ip().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Per-IP rate limit check
    if let Some(limiter) = req.extensions().get::<RateLimiter>() {
        if !limiter.check(&client_ip) {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": "Rate limit exceeded. Try again later.",
                    "retry_after_secs": 60u64,
                })),
            )
                .into_response());
        }
    }

    // Global rate limit check via SecurityValidator (wired from UNWIRED)
    let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let window_start = REQUEST_WINDOW_START.load(Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if window_start == 0 || now - window_start > 60 {
        // Reset window every 60 seconds
        REQUEST_WINDOW_START.store(now, Ordering::Relaxed);
        REQUEST_COUNTER.store(1, Ordering::Relaxed);
    }
    let validator = init_security_validator();
    if validator.check_rate_limit(counter as u32, 1).is_err() {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "Global rate limit exceeded. Try again later.",
                "retry_after_secs": 60u64,
            })),
        )
            .into_response());
    }

    Ok(next.run(req).await)
}

/// Axum middleware for request logging
async fn request_logging_layer(
    req: Request,
    next: Next,
) -> Result<Response, axum::response::Response> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    info!("{} {}", method, uri.path());
    let response = next.run(req).await;
    let status = response.status();
    info!("{} {} -> {}", method, uri.path(), status);
    Ok(response)
}

/// Returns true if the server has an auth system configured with users or API keys
fn config_has_auth_system() -> bool {
    #[cfg(feature = "server-auth")]
    {
        if let Some(auth) = crate::api::client::get_global_auth() {
            return true;
        }
    }
    false
}

fn add_cors_layer(mut app: Router, config: &ServerConfig) -> Result<Router> {
    let is_wildcard = config.cors_origins.iter().any(|o| o == "*");

    let cors = if is_wildcard {
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any)
    } else {
        let origins: Vec<_> = config
            .cors_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        let methods: Vec<Method> = vec!["GET", "POST", "PUT", "DELETE"]
            .into_iter()
            .filter_map(|m| m.parse().ok())
            .collect();
        let headers: Vec<HeaderName> = vec!["Content-Type", "Authorization"]
            .into_iter()
            .filter_map(|h| h.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(methods)
            .allow_headers(headers)
    };

    app = app.layer(cors);
    Ok(app)
}
