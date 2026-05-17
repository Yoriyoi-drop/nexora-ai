//! API Routing - Rust implementation
//! 
//! Request routing and path management

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

use crate::ApiHandler;

/// Router for managing API routes and handlers
pub struct Router {
    routes: HashMap<String, Route>,
    middleware_stack: Vec<String>,
    default_handler: Option<Arc<dyn ApiHandler>>,
}

/// Individual route definition
#[derive(Clone)]
pub struct Route {
    pub path: String,
    pub method: HttpMethod,
    pub handler: Arc<dyn ApiHandler>,
    pub middleware: Vec<String>,
    pub rate_limit: Option<RateLimit>,
    pub timeout_ms: Option<u64>,
    pub metadata: RouteMetadata,
}

/// HTTP methods
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

impl HttpMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::GET => "GET",
            HttpMethod::POST => "POST",
            HttpMethod::PUT => "PUT",
            HttpMethod::DELETE => "DELETE",
            HttpMethod::PATCH => "PATCH",
            HttpMethod::HEAD => "HEAD",
            HttpMethod::OPTIONS => "OPTIONS",
        }
    }
}

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
    pub description: Option<String>,
    pub version: String,
    pub deprecated: bool,
    pub tags: Vec<String>,
    pub parameters: Vec<RouteParameter>,
    pub responses: Vec<RouteResponse>,
}

/// Route parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteParameter {
    pub name: String,
    pub param_type: String,
    pub required: bool,
    pub description: Option<String>,
}

/// Route response definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResponse {
    pub status_code: u16,
    pub description: String,
    pub schema: Option<serde_json::Value>,
}

/// Path matcher for route resolution
#[derive(Debug)]
pub struct PathMatcher {
    _patterns: Vec<PathPattern>,
}

/// Path pattern for matching
#[derive(Debug, Clone)]
pub struct PathPattern {
    pub pattern: String,
    pub segments: Vec<PathSegment>,
    pub param_names: Vec<String>,
}

/// Path segment
#[derive(Debug, Clone)]
pub enum PathSegment {
    Literal(String),
    Parameter(String),
    Wildcard,
}

/// Route resolution result
pub struct RouteMatch {
    pub route: Route,
    pub params: HashMap<String, String>,
    pub score: f64,
}

impl Router {
    /// Create new router
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            middleware_stack: Vec::new(),
            default_handler: None,
        }
    }
    
    /// Add route to router
    pub fn add_route(&mut self, route: Route) {
        let route_key = format!("{} {}", route.method.as_str(), route.path);
        self.routes.insert(route_key, route);
    }
    
    /// Add simple route
    pub fn route(&mut self, method: HttpMethod, path: String, handler: Arc<dyn ApiHandler>) -> &mut Self {
        let route = Route {
            path: path.clone(),
            method,
            handler,
            middleware: self.middleware_stack.clone(),
            rate_limit: None,
            timeout_ms: None,
            metadata: RouteMetadata {
                description: None,
                version: "1.0".to_string(),
                deprecated: false,
                tags: Vec::new(),
                parameters: Vec::new(),
                responses: Vec::new(),
            },
        };
        
        self.add_route(route);
        self
    }
    
    /// Add GET route
    pub fn get(&mut self, path: String, handler: Arc<dyn ApiHandler>) -> &mut Self {
        self.route(HttpMethod::GET, path, handler)
    }
    
    /// Add POST route
    pub fn post(&mut self, path: String, handler: Arc<dyn ApiHandler>) -> &mut Self {
        self.route(HttpMethod::POST, path, handler)
    }
    
    /// Add PUT route
    pub fn put(&mut self, path: String, handler: Arc<dyn ApiHandler>) -> &mut Self {
        self.route(HttpMethod::PUT, path, handler)
    }
    
    /// Add DELETE route
    pub fn delete(&mut self, path: String, handler: Arc<dyn ApiHandler>) -> &mut Self {
        self.route(HttpMethod::DELETE, path, handler)
    }
    
    /// Add PATCH route
    pub fn patch(&mut self, path: String, handler: Arc<dyn ApiHandler>) -> &mut Self {
        self.route(HttpMethod::PATCH, path, handler)
    }
    
    /// Add middleware to stack
    pub fn middleware(&mut self, middleware_name: String) -> &mut Self {
        self.middleware_stack.push(middleware_name);
        self
    }
    
    /// Set default handler
    pub fn default_handler(&mut self, handler: Arc<dyn ApiHandler>) {
        self.default_handler = Some(handler);
    }
    
    /// Resolve route for request
    pub fn resolve_route(&self, method: &str, path: &str) -> Option<RouteMatch> {
        let route_key = format!("{} {}", method, path);
        
        // Exact match first
        if let Some(route) = self.routes.get(&route_key) {
            return Some(RouteMatch {
                route: route.clone(),
                params: HashMap::new(),
                score: 1.0,
            });
        }
        
        // Pattern matching for parameterized routes
        let mut best_match: Option<RouteMatch> = None;
        
        for (_route_key, route) in &self.routes {
            if route.method.as_str() != method {
                continue;
            }
            
            if let Some(params) = self.match_path(&route.path, path) {
                let score = self.calculate_match_score(&route.path, path);
                
                if let Some(ref best) = best_match {
                    if score > best.score {
                        best_match = Some(RouteMatch {
                            route: route.clone(),
                            params,
                            score,
                        });
                    }
                } else {
                    best_match = Some(RouteMatch {
                        route: route.clone(),
                        params,
                        score,
                    });
                }
            }
        }
        
        best_match
    }
    
    /// Match path pattern
    fn match_path(&self, pattern: &str, path: &str) -> Option<HashMap<String, String>> {
        let pattern_segments: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
        let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        
        if pattern_segments.len() != path_segments.len() {
            return None;
        }
        
        let mut params = HashMap::new();
        
        for (pattern_seg, path_seg) in pattern_segments.iter().zip(path_segments.iter()) {
            if pattern_seg.starts_with('{') && pattern_seg.ends_with('}') {
                let param_name = &pattern_seg[1..pattern_seg.len()-1];
                params.insert(param_name.to_string(), path_seg.to_string());
            } else if pattern_seg != path_seg {
                return None;
            }
        }
        
        Some(params)
    }
    
    /// Calculate match score
    fn calculate_match_score(&self, pattern: &str, path: &str) -> f64 {
        let pattern_segments: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
        let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        
        if pattern_segments.len() != path_segments.len() {
            return 0.0;
        }
        
        let mut exact_matches = 0;
        
        for (pattern_seg, path_seg) in pattern_segments.iter().zip(path_segments.iter()) {
            if pattern_seg.starts_with('{') && pattern_seg.ends_with('}') {
            } else if pattern_seg == path_seg {
                exact_matches += 1;
            } else {
                return 0.0; // Mismatch
            }
        }
        
        // Score based on exact matches (higher is better)
        exact_matches as f64 / pattern_segments.len() as f64
    }
    
    /// Get all routes
    pub fn get_routes(&self) -> Vec<&Route> {
        self.routes.values().collect()
    }
    
    /// Get routes by method
    pub fn get_routes_by_method(&self, method: &str) -> Vec<&Route> {
        self.routes.values()
            .filter(|route| route.method.as_str() == method)
            .collect()
    }
    
    /// Get routes by tag
    pub fn get_routes_by_tag(&self, tag: &str) -> Vec<&Route> {
        self.routes.values()
            .filter(|route| route.metadata.tags.contains(&tag.to_string()))
            .collect()
    }
    
    /// Generate OpenAPI specification
    pub fn generate_openapi(&self) -> serde_json::Value {
        let mut paths = serde_json::Map::new();
        
        for route in self.routes.values() {
            let path_item = paths.entry(route.path.clone()).or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            
            let operation = serde_json::json!({
                "operationId": format!("{}_{}", route.method.as_str().to_lowercase(), 
                    route.path.trim_start_matches('/').replace('/', "_")),
                "description": route.metadata.description,
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Bad request",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                },
                "tags": route.metadata.tags,
                "parameters": route.metadata.parameters,
                "deprecated": route.metadata.deprecated
            });
            
            if let serde_json::Value::Object(ref mut map) = path_item {
                map.insert(route.method.as_str().to_lowercase(), operation);
            }
        }
        
        serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": "Nexora AI API",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "High-performance AI API server"
            },
            "paths": paths,
            "components": {
                "schemas": {
                    "Error": {
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean"
                            },
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {
                                        "type": "string"
                                    },
                                    "message": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    }
    
    /// Validate routes
    pub fn validate_routes(&self) -> Vec<String> {
        let mut errors = Vec::new();
        
        for (route_key, route) in &self.routes {
            // Check for duplicate parameters
            let mut param_names = Vec::new();
            for segment in route.path.split('/').filter(|s| !s.is_empty()) {
                if segment.starts_with('{') && segment.ends_with('}') {
                    let param_name = &segment[1..segment.len()-1];
                    if param_names.contains(&param_name.to_string()) {
                        errors.push(format!("Duplicate parameter '{}' in route {}", param_name, route_key));
                    }
                    param_names.push(param_name.to_string());
                }
            }
            
            // Check for invalid characters in path
            if route.path.contains("//") {
                errors.push(format!("Invalid path format in route {}: contains double slash", route_key));
            }
            
            // Check if handler exists
            if route.handler.name().is_empty() {
                errors.push(format!("Route {} has invalid handler", route_key));
            }
        }
        
        errors
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

/// Route builder for fluent API
pub struct RouteBuilder {
    method: HttpMethod,
    path: String,
    handler: Option<Arc<dyn ApiHandler>>,
    middleware: Vec<String>,
    rate_limit: Option<RateLimit>,
    timeout_ms: Option<u64>,
    metadata: RouteMetadata,
}

impl RouteBuilder {
    /// Create new route builder
    pub fn new(method: HttpMethod, path: String) -> Self {
        Self {
            method,
            path,
            handler: None,
            middleware: Vec::new(),
            rate_limit: None,
            timeout_ms: None,
            metadata: RouteMetadata {
                description: None,
                version: "1.0".to_string(),
                deprecated: false,
                tags: Vec::new(),
                parameters: Vec::new(),
                responses: Vec::new(),
            },
        }
    }
    
    /// Set handler
    pub fn handler(mut self, handler: Arc<dyn ApiHandler>) -> Self {
        self.handler = Some(handler);
        self
    }
    
    /// Add middleware
    pub fn middleware(mut self, middleware: String) -> Self {
        self.middleware.push(middleware);
        self
    }
    
    /// Set rate limit
    pub fn rate_limit(mut self, max_requests: u32, window_seconds: u64) -> Self {
        self.rate_limit = Some(RateLimit {
            max_requests,
            window_seconds,
            burst_size: None,
        });
        self
    }
    
    /// Set timeout
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.metadata.description = Some(description);
        self
    }
    
    /// Add tag
    pub fn tag(mut self, tag: String) -> Self {
        self.metadata.tags.push(tag);
        self
    }
    
    /// Mark as deprecated
    pub fn deprecated(mut self) -> Self {
        self.metadata.deprecated = true;
        self
    }
    
    /// Add parameter
    pub fn parameter(mut self, name: String, param_type: String, required: bool) -> Self {
        self.metadata.parameters.push(RouteParameter {
            name,
            param_type,
            required,
            description: None,
        });
        self
    }
    
    /// Build route
    pub fn build(self) -> Result<Route> {
        let handler = self.handler.ok_or_else(|| anyhow::anyhow!("Handler is required"))?;
        
        Ok(Route {
            path: self.path,
            method: self.method,
            handler,
            middleware: self.middleware,
            rate_limit: self.rate_limit,
            timeout_ms: self.timeout_ms,
            metadata: self.metadata,
        })
    }
}

/// Utility function to create common routes
pub fn create_health_routes() -> Vec<Route> {
    vec![
        RouteBuilder::new(HttpMethod::GET, "/health".to_string())
            .description("Basic health check".to_string())
            .tag("health".to_string())
            .build()
            .expect("valid route configuration"),
        RouteBuilder::new(HttpMethod::GET, "/health/detailed".to_string())
            .description("Detailed health check with system metrics".to_string())
            .tag("health".to_string())
            .build()
            .expect("valid route configuration"),
    ]
}

pub fn create_metrics_routes() -> Vec<Route> {
    vec![
        RouteBuilder::new(HttpMethod::GET, "/metrics".to_string())
            .description("Get current metrics".to_string())
            .tag("metrics".to_string())
            .build()
            .expect("valid route configuration"),
        RouteBuilder::new(HttpMethod::GET, "/metrics/routes".to_string())
            .description("Get route-specific metrics".to_string())
            .tag("metrics".to_string())
            .build()
            .expect("valid route configuration"),
    ]
}

pub fn create_system_routes() -> Vec<Route> {
    vec![
        RouteBuilder::new(HttpMethod::GET, "/system/info".to_string())
            .description("Get system information".to_string())
            .tag("system".to_string())
            .build()
            .expect("valid route configuration"),
        RouteBuilder::new(HttpMethod::GET, "/system/stats".to_string())
            .description("Get system statistics".to_string())
            .tag("system".to_string())
            .build()
            .expect("valid route configuration"),
    ]
}
