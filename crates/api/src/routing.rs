//! API Routing — slimmed down, dead code removed.
//! Axum's built-in Router handles all routing.
//! These types are kept for API compatibility.

use serde::{Serialize, Deserialize};

/// Rate limit configuration for routes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub max_requests: u32,
    pub window_seconds: u64,
    pub burst_size: Option<u32>,
}

/// Route metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetadata {
    pub summary: String,
    pub description: String,
    pub tags: Vec<String>,
    pub parameters: Vec<RouteParameter>,
    pub responses: Vec<RouteResponse>,
    pub deprecated: bool,
}

/// Route parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteParameter {
    pub name: String,
    pub param_type: String,
    pub required: bool,
    pub description: String,
}

/// Route response definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResponse {
    pub status_code: u16,
    pub description: String,
    pub content_type: String,
}
