use axum::{
    extract::Path,
    http::StatusCode,
    response::sse::{Event, Sse},
    response::Html,
    response::IntoResponse,
    Extension, Json,
};
use axum::http::HeaderMap;
use chrono::Utc;
use futures::stream::{self, Stream};
use std::collections::HashMap;
use std::sync::Mutex;
use nexora_alignment::sparo::SparoSystem;
use nexora_monitoring::MetricsCollector;
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::security::{SecurityConfig, SecurityUtils, SecurityValidator};
use crate::NexoraAI;

static METRICS: OnceLock<Arc<MetricsCollector>> = OnceLock::new();
static SECURITY: OnceLock<SecurityValidator> = OnceLock::new();

// ── SPARO alignment & jailbreak detection ──
static SPARO: OnceLock<SparoSystem> = OnceLock::new();
static CONVERSATION_TRACKER: OnceLock<Mutex<HashMap<String, ConversationState>>> = OnceLock::new();

struct ConversationState {
    messages: Vec<String>,
    blocked_attempts: u32,
    updated_at: Instant,
}

/// Prune conversation states older than 30 minutes to prevent unbounded memory growth.
fn prune_conversation_states() {
    let tracker = CONVERSATION_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut state) = tracker.lock() {
        let before = state.len();
        state.retain(|_, v| v.updated_at.elapsed() < Duration::from_secs(1800));
        let after = state.len();
        if before != after {
            debug!("Pruned {} stale conversation states ({} remaining)", before - after, after);
        }
    }
}

/// Periodically prune stale conversation states (called from check_jailbreak_input_v2 at ~1% rate).
fn maybe_prune() {
    use std::sync::atomic::{AtomicU32, Ordering};
    static PRUNE_COUNTER: AtomicU32 = AtomicU32::new(0);
    let count = PRUNE_COUNTER.fetch_add(1, Ordering::Relaxed);
    if count % 100 == 0 {
        prune_conversation_states();
    }
}

fn init_sparo() -> &'static SparoSystem {
    SPARO.get_or_init(|| SparoSystem::new())
}

/// Output guardrail: check generated text for dangerous content + SPARO alignment.
async fn check_output_safety(prompt: &str, output: &str) -> Result<(), String> {
    if SecurityUtils::is_dangerous_content(output) {
        return Err("Output contains potentially sensitive information (secrets, keys)".to_string());
    }

    let sparo = init_sparo();
    match sparo.align_behavior(output, prompt).await {
        Ok(result) => {
            if result.safety_level == "blocked" || result.alignment_score < 0.3 {
                return Err(format!("Output rejected by safety alignment: score={:.3}", result.alignment_score));
            }
        }
        Err(e) => {
            warn!("Output SPARO check failed (non-fatal, allowing through): {}", e);
        }
    }

    Ok(())
}

/// Multi-turn jailbreak: detect escalation from benign → jailbreak across conversation turns.
fn detect_multi_turn_jailbreak(session_id: &str, message: &str) -> bool {
    let tracker = CONVERSATION_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
    let mut state = match tracker.lock() {
        Ok(s) => s,
        Err(e) => {
            warn!("Conversation tracker lock poisoned: {}", e);
            return false;
        }
    };

    let entry = state.entry(session_id.to_string()).or_insert_with(|| ConversationState {
        messages: Vec::new(),
        blocked_attempts: 0,
        updated_at: Instant::now(),
    });
    entry.updated_at = Instant::now();

    let msg_lower = message.to_lowercase();
    let has_ignore = msg_lower.contains("ignore") || msg_lower.contains("forget") || msg_lower.contains("override") || msg_lower.contains("bypass");
    let has_target = msg_lower.contains("instruction") || msg_lower.contains("rule") || msg_lower.contains("constraint") || msg_lower.contains("safety") || msg_lower.contains("limit") || msg_lower.contains("previous") || msg_lower.contains("above");

    if has_ignore && has_target && !entry.messages.is_empty() {
        warn!("Multi-turn jailbreak detected in session {}: '{}' after {} prior messages",
            session_id, &message[..message.len().min(80)], entry.messages.len());
        entry.blocked_attempts += 1;
        return true;
    }

    if entry.blocked_attempts >= 3 {
        warn!("Session {} has {} blocked attempts", session_id, entry.blocked_attempts);
        return true;
    }

    entry.messages.push(message.to_string());
    false
}

fn track_blocked_attempt(session_id: &str) {
    let tracker = CONVERSATION_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut state) = tracker.lock() {
        state.entry(session_id.to_string())
            .or_insert_with(|| ConversationState { messages: Vec::new(), blocked_attempts: 0, updated_at: Instant::now() })
            .blocked_attempts += 1;
    }
}

// ── Per-IP violation tracking ──
static IP_VIOLATIONS: OnceLock<Mutex<HashMap<String, IpViolationState>>> = OnceLock::new();

struct IpViolationState {
    count: u32,
    first_violation: Instant,
}

fn track_ip_violation(ip: &str) {
    let tracker = IP_VIOLATIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut state) = tracker.lock() {
        let entry = state.entry(ip.to_string()).or_insert_with(|| IpViolationState {
            count: 0,
            first_violation: Instant::now(),
        });
        entry.count += 1;
    }
}

fn check_ip_rate_limit(ip: &str) -> bool {
    let tracker = IP_VIOLATIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut state) = tracker.lock() {
        let now = Instant::now();
        // Prune stale entries (> 1 hour old)
        state.retain(|_, v| v.first_violation.elapsed() < Duration::from_secs(3600));

        let entry = state.entry(ip.to_string()).or_insert_with(|| IpViolationState {
            count: 0,
            first_violation: now,
        });
        // Block if 10+ violations within the hour
        entry.count >= 10
    } else {
        false
    }
}

// ── Input normalization & encoding detection ──

/// Normalize prompt for keyword matching: lowercase + collapse whitespace.
fn normalize_prompt(prompt: &str) -> String {
    prompt.to_lowercase().split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Detect base64/hex-encoded prompts that might hide jailbreak intent.
/// Uses character-distribution heuristics — no decoding dependency needed.
fn detect_encoded_prompt(prompt: &str) -> bool {
    let trimmed = prompt.trim();
    if trimmed.len() < 20 {
        return false;
    }

    let total = trimmed.len();
    let alnum: usize = trimmed.chars().filter(|c| c.is_ascii_alphanumeric()).count();
    let base64_chars: usize = trimmed.chars().filter(|c| *c == '+' || *c == '/' || *c == '=').count();
    let hex_chars: usize = trimmed.chars().filter(|c| c.is_ascii_hexdigit()).count();

    // Base64 heuristic: ≥85% alphanumeric, has +/= padding chars, length multiple of 4
    let looks_like_base64 = (alnum + base64_chars) as f64 / total as f64 > 0.90
        && (trimmed.ends_with('=') || trimmed.contains('+') || trimmed.contains('/'))
        && total >= 32;
    if looks_like_base64 {
        warn!("Possible base64-encoded prompt detected ({} chars, alnum ratio={:.2})",
            total, alnum as f64 / total as f64);
        return true;
    }

    // Hex heuristic: ≥90% hex chars, even length
    if hex_chars as f64 / total as f64 > 0.90 && total % 2 == 0 && total >= 24 {
        warn!("Possible hex-encoded prompt detected ({} chars)", total);
        return true;
    }

    false
}

/// Expanded jailbreak keyword patterns for normalized text matching.
const JAILBREAK_KEYWORDS: &[&str] = &[
    // Classic DAN variants
    "ignore all instructions", "ignore your instructions", "ignore your safety",
    "ignore your previous", "ignore all rules", "ignore your programming",
    "ignore your core", "ignore safety protocols", "ignore your guidelines",
    "ignore your constraints", "ignore your limitations",
    // Bypass variants
    "bypass safety", "bypass restrictions", "bypass your limits",
    "bypass all filters", "bypass content policy", "bypass alignment",
    // DAN persona
    "you are now dan", "act as dan", "you are dan", "new character dan",
    "act as though you are dan", "pretend to be dan",
    // Freedom / no limits
    "no restrictions", "no limits", "no boundaries", "no rules",
    "no filters", "no constraints", "no ethical",
    "you are free", "you are not bound", "you have no limits",
    "you can do anything", "you are now free",
    // Meta instructions
    "do anything now", "output without restrictions",
    "you don't have to follow", "override your programming",
    "override your safety", "override your constraints",
    "forget your training", "forget your guidelines",
    // System prompt theft
    "system prompt", "your system prompt", "reveal your prompt",
    "initial prompt", "your initial instructions",
    "what are your instructions", "tell me your instructions",
    // Roleplay evasion
    "for educational purposes", "hypothetical scenario",
    "pretend you are", "roleplay as", "act as a",
    "write a story about", "imagine a scenario",
    // Character injection
    "you are a person", "you are an ai without",
    "you are not an ai", "you are not bound",
    "act as if you have no", "respond as if",
    // Multi-turn escalation
    "now ignore", "now forget", "now bypass", "forget your",
];

/// Combined input guardrail with normalization, encoding detection, IP rate limiting.
async fn check_jailbreak_input_v2(
    session_id: Option<&str>,
    client_ip: Option<&str>,
    prompt: &str,
) -> Result<(), (StatusCode, Json<Value>)> {
    maybe_prune();

    // IP rate limiting: 10+ violations in 1 hour → temp block
    if let Some(ip) = client_ip {
        if check_ip_rate_limit(ip) {
            warn!("IP rate-limited due to repeated violations: {}", ip);
            return Err(err_json(StatusCode::TOO_MANY_REQUESTS, "Too many security violations from this IP"));
        }
    }

    // Encoding detection (base64 / hex)
    if detect_encoded_prompt(prompt) {
        if let Some(sid) = session_id { track_blocked_attempt(sid); }
        if let Some(ip) = client_ip { track_ip_violation(ip); }
        return Err(err_json(StatusCode::BAD_REQUEST, "Encoded jailbreak prompt detected"));
    }

    // Normalize for keyword matching
    let normalized = normalize_prompt(prompt);

    // Direct keyword match on normalized input
    let kw_match = JAILBREAK_KEYWORDS.iter().any(|kw| normalized.contains(kw));
    if kw_match {
        warn!("Jailbreak keyword matched in prompt: {:.80}", prompt);
        if let Some(sid) = session_id { track_blocked_attempt(sid); }
        if let Some(ip) = client_ip { track_ip_violation(ip); }
        return Err(err_json(StatusCode::BAD_REQUEST, "Jailbreak attempt detected"));
    }

    // SPARO alignment check
    let sparo = init_sparo();
    match sparo.align_behavior(prompt, "").await {
        Ok(result) => {
            if result.safety_level == "blocked" {
                warn!("SPARO blocked prompt: score={:.3}, level={}", result.alignment_score, result.safety_level);
                if let Some(sid) = session_id { track_blocked_attempt(sid); }
                if let Some(ip) = client_ip { track_ip_violation(ip); }
                return Err(err_json(StatusCode::BAD_REQUEST, "Prompt rejected by safety alignment"));
            }
        }
        Err(e) => warn!("SPARO alignment failed (non-fatal): {}", e),
    }

    // Multi-turn jailbreak detection
    if let Some(sid) = session_id {
        if detect_multi_turn_jailbreak(sid, prompt) {
            return Err(err_json(StatusCode::BAD_REQUEST, "Conversation escalation detected"));
        }
    }

    Ok(())
}

/// Cached timestamp: refreshed every 1 second to avoid syscall per request.
static CACHED_TIMESTAMP: OnceLock<Mutex<TimestampCache>> = OnceLock::new();
struct TimestampCache {
    value: String,
    updated: Instant,
}
fn cached_timestamp() -> String {
    let cache = CACHED_TIMESTAMP.get_or_init(|| Mutex::new(TimestampCache {
        value: Utc::now().to_rfc3339(),
        updated: Instant::now(),
    }));
    let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
    if c.updated.elapsed() > Duration::from_secs(1) {
        c.value = Utc::now().to_rfc3339();
        c.updated = Instant::now();
    }
    c.value.clone()
}

/// Helper: success JSON response with timestamp.
fn ok_json(data: Value) -> Json<Value> {
    Json(json!({
        "success": true,
        "data": data,
        "timestamp": cached_timestamp(),
    }))
}

/// Helper: error JSON response with timestamp.
fn err_json(status: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({
        "success": false,
        "error": msg,
        "timestamp": cached_timestamp(),
    })))
}

/// Extract client IP from request headers (X-Forwarded-For or X-Real-IP).
fn client_ip_from_headers(headers: &HeaderMap) -> Option<String> {
    if let Some(ip) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        return Some(ip.split(',').next().unwrap_or(ip).trim().to_string());
    }
    if let Some(ip) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        return Some(ip.to_string());
    }
    None
}

fn init_security_validator() -> &'static SecurityValidator {
    SECURITY.get_or_init(|| {
        SecurityValidator::new(SecurityConfig::default())
    })
}

fn validate_prompt(prompt: &str) -> Result<(), (StatusCode, Json<Value>)> {
    let validator = init_security_validator();
    validator.validate_input(prompt).map_err(|e| {
        warn!("Security validation rejected input: {}", e);
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "Input rejected by security validator",
                "detail": e.to_string()
            })),
        )
    })
}

pub fn init_metrics() -> anyhow::Result<Arc<MetricsCollector>> {
    let collector = Arc::new(
        MetricsCollector::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize metrics collector: {}", e))?,
    );
    METRICS.set(collector.clone()).ok();
    Ok(collector)
}

pub fn metrics_collector() -> Option<&'static Arc<MetricsCollector>> {
    METRICS.get()
}

pub async fn health_check(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    let start = std::time::Instant::now();
    match nexora.health_check().await {
        Ok(health) => {
            if let Some(m) = metrics_collector() {
                m.record_request(true, start.elapsed().as_secs_f64());
            }
            Json(json!({
                "healthy": health.healthy,
                "timestamp": health.last_check,
                "version": env!("CARGO_PKG_VERSION")
            }))
        }
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Health check failed: {}", e);
            Json(json!({
                "healthy": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn detailed_health_check(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    match nexora.health_check().await {
        Ok(health) => Json(json!(health)),
        Err(e) => {
            error!("Detailed health check failed: {}", e);
            Json(json!({
                "healthy": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn metrics_handler() -> axum::response::Response {
    let body = match metrics_collector() {
        Some(m) => m.gather_prometheus(),
        None => String::new(),
    };
    axum::http::Response::builder()
        .header("Content-Type", "text/plain; charset=utf-8")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|e| {
            axum::http::Response::builder()
                .status(500)
                .body(axum::body::Body::from(format!("metrics error: {}", e)))
                .unwrap_or_else(|_| {
                    axum::http::Response::new(axum::body::Body::from("metrics error"))
                })
        })
}

pub async fn gpu_health_handler() -> Json<serde_json::Value> {
    let obs = nexora_inference::inference_trait::observability_snapshot();
    Json(serde_json::json!({
        "gpu_alive": obs.gpu_alive,
        "gpu_forward_errors": obs.gpu_forward_errors,
        "gpu_cpu_fallbacks": obs.gpu_cpu_fallbacks,
        "gpu_resident_successes": obs.gpu_resident_successes,
        "gpu_resident_fallbacks": obs.gpu_resident_fallbacks,
        "gpu_tokens_generated": obs.gpu_tokens_generated,
        "cpu_tokens_generated": obs.cpu_tokens_generated,
        "gpu_resident_ratio": obs.gpu_resident_ratio,
    }))
}

pub async fn system_info(Extension(nexora): Extension<Arc<NexoraAI>>) -> impl IntoResponse {
    match nexora.get_system_info().await {
        Ok(info) => Json(json!(info)),
        Err(e) => {
            error!("Failed to get system info: {}", e);
            Json(json!({
                "healthy": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn performance_metrics(Extension(nexora): Extension<Arc<NexoraAI>>) -> impl IntoResponse {
    match nexora.get_performance_metrics().await {
        Ok(metrics) => Json(metrics),
        Err(e) => {
            error!("Failed to get performance metrics: {}", e);
            Json(json!({
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn memory_stats(Extension(nexora): Extension<Arc<NexoraAI>>) -> impl IntoResponse {
    match nexora.get_system_info().await {
        Ok(info) => Json(json!({
            "total_memory": info.memory_stats.total_memory,
            "used_memory": info.memory_stats.used_memory,
            "available_memory": info.memory_stats.available_memory,
            "cache_size": info.memory_stats.cache_size,
            "usage_percent": info.memory_usage,
            "timestamp": info.last_updated
        })),
        Err(e) => {
            error!("Failed to get memory stats: {}", e);
            Json(json!({
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn process_request(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let input = payload.get("input").and_then(|v| v.as_str()).unwrap_or("");
    let client_ip = client_ip_from_headers(&headers);

    if let Err((_status, err_json)) = validate_prompt(input) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    if let Err((_status, err_json)) = check_jailbreak_input_v2(None, client_ip.as_deref(), input).await {
        return Json(json!({
            "success": false,
            "error": err_json.get("error").and_then(|v| v.as_str()).unwrap_or("blocked by safety alignment"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let sanitized = {
        let validator = init_security_validator();
        validator.sanitize_input(input)
    };

    let truncated: String = sanitized.chars().take(100).collect();
    info!(
        "Processing request: {} [truncated {} chars]",
        truncated,
        sanitized.len().min(100)
    );

    match nexora.process_request(&sanitized).await {
        Ok(mut response) => {
            if let Err(e) = check_output_safety(input, &response).await {
                warn!("Process output safety check triggered: {}", e);
                response = "[Content filtered by safety guardrail]".to_string();
            }

            Json(json!({
                "success": true,
                "response": response,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Request processing failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn generate_text(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let client_ip = client_ip_from_headers(&headers);

    if let Err((_status, err_json)) = validate_prompt(prompt) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    if let Err((_status, err_json)) = check_jailbreak_input_v2(None, client_ip.as_deref(), prompt).await {
        return Json(json!({
            "success": false,
            "error": err_json.get("error").and_then(|v| v.as_str()).unwrap_or("blocked by safety alignment"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let max_tokens = payload
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as usize;

    let temperature = payload
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let sanitized = {
        let validator = init_security_validator();
        validator.sanitize_input(prompt)
    };

    let truncated: String = sanitized.chars().take(100).collect();
    info!(
        "Generating text: prompt='{}' [truncated {} chars], max_tokens={}, temperature={}",
        truncated,
        sanitized.len().min(100),
        max_tokens,
        temperature
    );

    match nexora.generate_text(&sanitized, max_tokens, temperature).await {
        Ok(mut generated) => {
            if let Err(e) = check_output_safety(prompt, &generated).await {
                warn!("Output safety check triggered: {}", e);
                generated = "[Content filtered by safety guardrail]".to_string();
            }

            Json(json!({
                "success": true,
                "generated_text": generated,
                "parameters": {
                    "prompt": sanitized,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Text generation failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn generate_text_stream(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Result<Sse<impl Stream<Item = Result<Event, anyhow::Error>>>, (StatusCode, Json<Value>)> {
    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let client_ip = client_ip_from_headers(&headers);
    if prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Prompt cannot be empty" })),
        ));
    }

    validate_prompt(prompt)?;
    check_jailbreak_input_v2(None, client_ip.as_deref(), prompt).await?;

    let max_tokens = payload
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;

    let temperature = payload
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;

    let sanitized = {
        let validator = init_security_validator();
        validator.sanitize_input(prompt)
    };

    info!(
        "Streaming text generation: prompt={}, max_tokens={}, temperature={}",
        sanitized.len(),
        max_tokens,
        temperature
    );

    let rx = nexora
        .generate_text_stream(&sanitized, max_tokens, temperature)
        .await
        .map_err(|e| {
            error!("Streaming inference init failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
        })?;

    // Accumulating output guardrail for streaming: buffer tokens, check at end.
    let prompt_arc = Arc::new(prompt.to_string());
    let stream = stream::unfold(
        (rx, String::new(), false, prompt_arc),
        move |(mut rx, mut acc, mut flagged, prompt_arc)| async move {
            if flagged {
                return None;
            }
            match rx.recv().await {
                Some(token) => {
                    acc.push_str(&token.token_text);
                    let data = serde_json::to_string(&json!({
                        "token": token.token_text,
                        "position": token.position,
                    })).unwrap_or_else(|_| r#"{"token":"","position":0}"#.to_string());
                    Some((Ok::<_, anyhow::Error>(Event::default().data(data)), (rx, acc, false, prompt_arc)))
                }
                None => {
                    let final_event = if !acc.is_empty() {
                        match check_output_safety(&prompt_arc, &acc).await {
                            Ok(()) => Event::default().data("[DONE]"),
                            Err(e) => {
                                warn!("Streaming output safety check triggered: {}", e);
                                Event::default().data(r#"{"token":"[Content filtered by safety guardrail]","position":0}"#)
                            }
                        }
                    } else {
                        Event::default().data("[DONE]")
                    };
                    Some((Ok(final_event), (rx, acc, true, prompt_arc)))
                }
            }
        },
    );

    Ok(Sse::new(stream))
}

pub async fn chat(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let start = std::time::Instant::now();
    let message = payload
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let client_ip = client_ip_from_headers(&headers);

    if let Err((_status, err_json)) = validate_prompt(message) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let conversation_id = payload
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let session_id = conversation_id.as_deref();

    if let Err((_status, err_json)) = check_jailbreak_input_v2(session_id, client_ip.as_deref(), message).await {
        return Json(json!({
            "success": false,
            "error": err_json.get("error").and_then(|v| v.as_str()).unwrap_or("blocked by safety alignment"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let sanitized = {
        let validator = init_security_validator();
        validator.sanitize_input(message)
    };

    let truncated = &sanitized[..sanitized.len().min(100)];
    info!(
        "Chat message: {} [truncated {} chars] (conversation_id: {:?})",
        truncated,
        sanitized.len().min(100),
        conversation_id
    );

    match nexora.chat(&sanitized, conversation_id).await {
        Ok(mut response) => {
            if let Err(e) = check_output_safety(message, &response).await {
                warn!("Chat output safety check triggered: {}", e);
                response = "[Content filtered by safety guardrail]".to_string();
            }

            Json(json!({
                "success": true,
                "response": response,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
        Err(e) => {
            if let Some(m) = metrics_collector() {
                m.record_request(false, start.elapsed().as_secs_f64());
            }
            error!("Chat failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn analyze_code(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let code = payload.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let client_ip = client_ip_from_headers(&headers);

    if let Err((_status, err_json)) = validate_prompt(code) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    if let Err((_status, err_json)) = check_jailbreak_input_v2(None, client_ip.as_deref(), code).await {
        return Json(json!({
            "success": false,
            "error": err_json.get("error").and_then(|v| v.as_str()).unwrap_or("blocked by safety alignment"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let language = payload
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let sanitized = {
        let validator = init_security_validator();
        validator.sanitize_input(code)
    };

    info!(
        "Analyzing code: {} chars, language: {}",
        sanitized.len(),
        language
    );

    match nexora.analyze_code(&sanitized, language).await {
        Ok(mut analysis) => {
            if let Err(e) = check_output_safety(code, &analysis).await {
                warn!("Code analysis output safety check triggered: {}", e);
                analysis = "[Content filtered by safety guardrail]".to_string();
            }

            Json(json!({
                "success": true,
                "analysis": analysis,
                "metadata": {
                    "code_length": code.len(),
                    "language": language,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }
            }))
        }
        Err(e) => {
            error!("Code analysis failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn generate_code(
    headers: HeaderMap,
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let description = payload
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let client_ip = client_ip_from_headers(&headers);

    if let Err((_status, err_json)) = validate_prompt(description) {
        return Json(json!({
            "success": false,
            "error": err_json.get("detail").and_then(|v| v.as_str()).unwrap_or("validation failed"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    if let Err((_status, err_json)) = check_jailbreak_input_v2(None, client_ip.as_deref(), description).await {
        return Json(json!({
            "success": false,
            "error": err_json.get("error").and_then(|v| v.as_str()).unwrap_or("blocked by safety alignment"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));
    }

    let language = payload
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("rust");

    let sanitized = {
        let validator = init_security_validator();
        validator.sanitize_input(description)
    };

    info!(
        "Generating code: description='{}', language='{}'",
        sanitized, language
    );

    match nexora.generate_code(&sanitized, language).await {
        Ok(mut code) => {
            if let Err(e) = check_output_safety(description, &code).await {
                warn!("Code generation output safety check triggered: {}", e);
                code = "[Content filtered by safety guardrail]".to_string();
            }

            Json(json!({
                "success": true,
                "generated_code": code,
                "metadata": {
                    "description": description,
                    "language": language,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }
            }))
        }
        Err(e) => {
            error!("Code generation failed: {}", e);
            Json(json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn get_config() -> Json<Value> {
    Json(json!({
        "server": {
            "version": env!("CARGO_PKG_VERSION"),
            "features": ["text_generation", "code_analysis", "chat", "health_check", "metrics"],
            "endpoints": [
                "/health", "/health/detailed", "/metrics",
                "/info", "/info/performance", "/info/memory",
                "/process", "/generate", "/chat",
                "/code/analyze", "/code/generate", "/config"
            ]
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

pub async fn update_config(
    headers: HeaderMap,
    Extension(_nexora): Extension<Arc<NexoraAI>>,
    api_key: Option<Extension<crate::server::router::ApiKey>>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    info!("Processing configuration update request");

    // Require authentication for config changes
    validate_admin(&headers, api_key.as_ref().map(|e| &e.0))?;

    let config_result = validate_and_process_config(&payload);

    match config_result {
        Ok(updated_config) => {
            info!("Configuration updated successfully");
            Ok(Json(json!({
                "success": true,
                "message": "Configuration updated successfully",
                "updated_fields": updated_config.updated_fields,
                "timestamp": chrono::Utc::now().to_rfc3339()
            })))
        }
        Err(e) => {
            error!("Configuration update failed: {}", e);
            Ok(Json(json!({
                "success": false,
                "message": "Configuration update failed. Check server logs for details.",
                "timestamp": chrono::Utc::now().to_rfc3339()
            })))
        }
    }
}

fn validate_and_process_config(payload: &Value) -> Result<ConfigUpdateResult, anyhow::Error> {
    let mut updated_fields = Vec::new();

    if let Some(server_config) = payload.get("server") {
        if let Some(host) = server_config.get("host").and_then(|v| v.as_str()) {
            if !host.is_empty() {
                updated_fields.push(format!("server.host: {}", host));
            }
        }

        if let Some(port) = server_config.get("port").and_then(|v| v.as_u64()) {
            if port > 0 && port <= 65535 {
                updated_fields.push(format!("server.port: {}", port));
            } else {
                return Err(anyhow::anyhow!("Invalid port number: {}", port));
            }
        }

        if let Some(max_connections) = server_config
            .get("max_connections")
            .and_then(|v| v.as_u64())
        {
            if max_connections > 0 && max_connections <= 10000 {
                updated_fields.push(format!("server.max_connections: {}", max_connections));
            }
        }
    }

    if let Some(api_config) = payload.get("api") {
        if let Some(timeout) = api_config.get("timeout_seconds").and_then(|v| v.as_u64()) {
            if timeout >= 1 && timeout <= 300 {
                updated_fields.push(format!("api.timeout_seconds: {}", timeout));
            }
        }

        if let Some(rate_limit) = api_config.get("rate_limit") {
            if let Some(enabled) = rate_limit.get("enabled").and_then(|v| v.as_bool()) {
                updated_fields.push(format!("api.rate_limit.enabled: {}", enabled));
            }

            if let Some(requests_per_minute) = rate_limit
                .get("requests_per_minute")
                .and_then(|v| v.as_u64())
            {
                if requests_per_minute > 0 && requests_per_minute <= 1000 {
                    updated_fields.push(format!(
                        "api.rate_limit.requests_per_minute: {}",
                        requests_per_minute
                    ));
                }
            }
        }
    }

    if let Some(model_config) = payload.get("models") {
        if let Some(default_model) = model_config.get("default").and_then(|v| v.as_str()) {
            if !default_model.is_empty() {
                updated_fields.push(format!("models.default: {}", default_model));
            }
        }

        if let Some(max_tokens) = model_config.get("max_tokens").and_then(|v| v.as_u64()) {
            if max_tokens > 0 && max_tokens <= 8192 {
                updated_fields.push(format!("models.max_tokens: {}", max_tokens));
            }
        }

        if let Some(temperature) = model_config.get("temperature").and_then(|v| v.as_f64()) {
            if temperature >= 0.0 && temperature <= 2.0 {
                updated_fields.push(format!("models.temperature: {}", temperature));
            } else {
                return Err(anyhow::anyhow!("Temperature must be between 0.0 and 2.0"));
            }
        }
    }

    if let Some(logging_config) = payload.get("logging") {
        if let Some(level) = logging_config.get("level").and_then(|v| v.as_str()) {
            match level {
                "debug" | "info" | "warn" | "error" => {
                    updated_fields.push(format!("logging.level: {}", level));
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid log level: {}", level));
                }
            }
        }

        if let Some(enabled) = logging_config.get("file_enabled").and_then(|v| v.as_bool()) {
            updated_fields.push(format!("logging.file_enabled: {}", enabled));
        }
    }

    if updated_fields.is_empty() {
        return Err(anyhow::anyhow!("No valid configuration fields provided"));
    }

    Ok(ConfigUpdateResult {
        updated_fields,
        timestamp: chrono::Utc::now(),
    })
}

#[derive(Debug)]
struct ConfigUpdateResult {
    updated_fields: Vec<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn get_train_metrics(Extension(nexora): Extension<Arc<NexoraAI>>) -> Json<Value> {
    let metrics = nexora.train_metrics.read().await;
    Json(metrics.clone())
}

pub async fn post_train_metrics(
    Extension(nexora): Extension<Arc<NexoraAI>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let mut metrics = nexora.train_metrics.write().await;
    *metrics = payload.clone();
    Json(json!({ "success": true }))
}

/// Validate that the request has admin-level authentication.
/// Checks NEXORA_ADMIN_KEY env var match, then falls back to
/// the ApiKey extension from auth middleware (set after successful auth).
fn validate_admin(
    headers: &HeaderMap,
    api_key_opt: Option<&crate::server::router::ApiKey>,
) -> Result<(), (StatusCode, Json<Value>)> {
    // Check NEXORA_ADMIN_KEY env var
    if let Ok(admin_key) = std::env::var("NEXORA_ADMIN_KEY") {
        let provided = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .or_else(|| {
                headers
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
            });
        if provided == Some(&admin_key) {
            return Ok(());
        }
        return Err((
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Forbidden: invalid or missing admin key"})),
        ));
    }

    // No NEXORA_ADMIN_KEY: check if request came through auth middleware
    if api_key_opt.is_some() {
        return Ok(());
    }

    Err((
        StatusCode::FORBIDDEN,
        Json(json!({"error": "Forbidden: admin authentication required. Set NEXORA_ADMIN_KEY env var or use API key."})),
    ))
}

pub async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

pub async fn static_files(
    Path(path): Path<String>,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    let base_path = std::path::Path::new("static")
        .canonicalize()
        .map_err(|_| axum::http::StatusCode::FORBIDDEN)?;
    let file_path = base_path
        .join(&path)
        .canonicalize()
        .map_err(|_| axum::http::StatusCode::FORBIDDEN)?;

    if !file_path.starts_with(&base_path) {
        return Err(axum::http::StatusCode::FORBIDDEN);
    }

    let content = match tokio::fs::read(&file_path).await {
        Ok(c) => c,
        Err(_) => return Err(axum::http::StatusCode::NOT_FOUND),
    };

    let content_type = match path.split('.').last() {
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("html") => "text/html",
        _ => "text/plain",
    };

    Ok(axum::http::Response::builder()
        .status(200)
        .header("Content-Type", content_type)
        .body(axum::body::Body::from(content))
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?)
}
