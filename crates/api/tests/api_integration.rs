use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use axum::{
    Router, routing::get, extract::{State, Request}, response::Json,
};
use serde_json::Value;
use tower_http::cors::{CorsLayer, Any};
use nexora_api::{ApiConfig, ApiStatistics};

#[derive(Clone)]
struct TestAppState {
    config: ApiConfig,
    statistics: Arc<tokio::sync::RwLock<ApiStatistics>>,
    start_time: std::time::Instant,
}

async fn health_handler(State(state): State<TestAppState>) -> Json<Value> {
    Json(serde_json::json!({
        "healthy": true,
        "uptime_seconds": state.start_time.elapsed().as_secs(),
    }))
}

async fn metrics_handler() -> Json<Value> {
    Json(serde_json::json!({
        "requests_total": 0,
        "requests_per_second": 0.0,
    }))
}

async fn echo_handler(req: Request) -> Json<Value> {
    Json(serde_json::json!({
        "method": req.method().as_str(),
        "uri": req.uri().to_string(),
    }))
}

async fn stats_handler(State(state): State<TestAppState>) -> Json<Value> {
    let stats = state.statistics.read().await;
    Json(serde_json::json!({
        "total_requests": stats.total_requests,
        "active_connections": stats.active_connections,
    }))
}

fn build_test_router(config: ApiConfig) -> Router {
    let state = TestAppState {
        statistics: Arc::new(tokio::sync::RwLock::new(ApiStatistics::default())),
        config,
        start_time: std::time::Instant::now(),
    };

    let mut app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/echo", get(echo_handler))
        .route("/api/v1/status", get(stats_handler))
        .with_state(state);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers(Any);

    app = app.layer(cors);
    app
}

async fn spawn_test_server() -> (String, tokio::task::JoinHandle<()>) {
    let config = ApiConfig::default();
    let router = build_test_router(config.clone());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}", addr);

    let handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    (url, handle)
}

#[tokio::test]
async fn test_health_endpoint() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/health", url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["healthy"], true);
    assert!(body["uptime_seconds"].as_u64().is_some());
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/metrics", url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert!(body["requests_total"].as_u64().is_some());
}

#[tokio::test]
async fn test_echo_endpoint() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/echo?foo=bar", url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["method"], "GET");
    assert!(body["uri"].as_str().unwrap().contains("/echo"));
}

#[tokio::test]
async fn test_status_endpoint() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/api/v1/status", url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert!(body["total_requests"].as_u64().is_some());
    assert!(body["active_connections"].as_u64().is_some());
}

#[tokio::test]
async fn test_404_unknown_endpoint() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/unknown/path", url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_cors_headers() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/health", url))
        .header("Origin", "https://example.com")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let cors_origin = resp.headers().get("access-control-allow-origin");
    assert!(cors_origin.is_some(), "CORS header should be present");
}

#[tokio::test]
async fn test_concurrent_requests() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();
    use futures::future::join_all;

    let requests: Vec<_> = (0..10).map(|i| {
        client.get(format!("{}/health", url))
            .header("x-request-id", format!("test-{}", i))
            .send()
    }).collect();

    let results = join_all(requests).await;
    for result in results {
        let resp = result.unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
    }
}

#[tokio::test]
async fn test_server_response_time() {
    let (url, _handle) = spawn_test_server().await;
    let client = reqwest::Client::new();

    let start = std::time::Instant::now();
    let resp = client.get(format!("{}/health", url)).send().await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert!(elapsed.as_millis() < 5000, "Response should be under 5s");
}
