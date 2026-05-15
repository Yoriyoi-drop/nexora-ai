//! Inference Session
//! 
//! Session management untuk inference requests.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::{debug, info};
use chrono::{DateTime, Utc};

use crate::{Result, InferenceError};

/// Configuration untuk inference session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum session duration (seconds)
    pub max_duration_seconds: u64,
    /// Maximum tokens per session
    pub max_tokens_per_session: u64,
    /// Enable session persistence
    pub enable_persistence: bool,
    /// Session timeout (seconds)
    pub timeout_seconds: u64,
    /// Enable session compression
    pub enable_compression: bool,
    /// Maximum context length
    pub max_context_length: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_duration_seconds: 3600, // 1 hour
            max_tokens_per_session: 10000,
            enable_persistence: false,
            timeout_seconds: 1800, // 30 minutes
            enable_compression: false,
            max_context_length: 4096,
        }
    }
}

/// Session state
#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    /// Session aktif
    Active,
    /// Session idle
    Idle,
    /// Session sedang diproses
    Processing,
    /// Session paused
    Paused,
    /// Session error
    Error(String),
    /// Session closed
    Closed,
    /// Session timeout
    Timeout,
}

/// Inference session
#[derive(Debug, Clone)]
pub struct InferenceSession {
    /// Session ID
    pub session_id: Uuid,
    /// User ID (optional)
    pub user_id: Option<Uuid>,
    /// Session configuration
    config: SessionConfig,
    /// Current session state
    state: Arc<RwLock<SessionState>>,
    /// Session metadata
    metadata: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// Session statistics
    stats: Arc<RwLock<SessionStats>>,
    /// Session history
    history: Arc<RwLock<Vec<SessionEntry>>>,
    /// Context cache
    context_cache: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// Created timestamp
    created_at: DateTime<Utc>,
    /// Last activity timestamp
    last_activity: Arc<RwLock<DateTime<Utc>>>,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Total requests in session
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Total processing time (ms)
    pub total_processing_time_ms: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Session duration (seconds)
    pub session_duration_seconds: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
}

/// Session entry (request/response pair)
#[derive(Debug, Clone)]
pub struct SessionEntry {
    /// Entry ID
    pub entry_id: Uuid,
    /// Request ID
    pub request_id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Request prompt
    pub prompt: String,
    /// Generated response
    pub response: String,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Processing time (ms)
    pub processing_time_ms: u64,
    /// Entry metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl InferenceSession {
    /// Create new session
    pub fn new(session_id: Uuid) -> Self {
        Self::with_config(session_id, SessionConfig::default())
    }

    /// Get session config
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.created_at
    }

    /// Default timeout seconds for stale session cleanup
    pub fn default_timeout_seconds() -> u64 {
        1800
    }
    
    /// Create session with configuration
    pub fn with_config(session_id: Uuid, config: SessionConfig) -> Self {
        let now = Utc::now();
        Self {
            session_id,
            user_id: None,
            config,
            state: Arc::new(RwLock::new(SessionState::Active)),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SessionStats::default())),
            history: Arc::new(RwLock::new(Vec::new())),
            context_cache: Arc::new(RwLock::new(HashMap::new())),
            created_at: now,
            last_activity: Arc::new(RwLock::new(now)),
        }
    }
    
    /// Set user ID
    pub fn with_user_id(mut self, user_id: Uuid) -> Self {
        self.user_id = Some(user_id);
        self
    }
    
    /// Get session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }
    
    /// Get user ID
    pub fn user_id(&self) -> Option<Uuid> {
        self.user_id
    }
    
    /// Get current state
    pub async fn get_state(&self) -> SessionState {
        self.state.read().await.clone()
    }
    
    /// Set session state
    pub async fn set_state(&self, new_state: SessionState) {
        let mut state = self.state.write().await;
        *state = new_state;
        self.update_last_activity().await;
    }
    
    /// Get session metadata
    pub async fn get_metadata(&self) -> HashMap<String, serde_json::Value> {
        self.metadata.read().await.clone()
    }
    
    /// Set metadata value
    pub async fn set_metadata(&self, key: String, value: serde_json::Value) {
        let mut metadata = self.metadata.write().await;
        metadata.insert(key, value);
        self.update_last_activity().await;
    }
    
    /// Get metadata value
    pub async fn get_metadata_value(&self, key: &str) -> Option<serde_json::Value> {
        let metadata = self.metadata.read().await;
        metadata.get(key).cloned()
    }
    
    /// Get session statistics
    pub async fn get_stats(&self) -> SessionStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update session duration
        let now = Utc::now();
        stats.session_duration_seconds = (now - self.created_at).num_seconds() as u64;
        
        stats
    }
    
    /// Add entry to session history
    pub async fn add_entry(&self, entry: SessionEntry) -> Result<()> {
        debug!("Adding entry to session {}: {}", self.session_id, entry.entry_id);
        
        // Check session limits
        self.check_session_limits(&entry).await?;
        
        // Add to history
        {
            let mut history = self.history.write().await;
            history.push(entry.clone());
            
            // Limit history size (keep last 1000 entries)
            if history.len() > 1000 {
                history.remove(0);
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.total_tokens_generated += entry.tokens_generated as u64;
            stats.total_processing_time_ms += entry.processing_time_ms;
            
            // Update average processing time
            if stats.total_requests > 0 {
                stats.avg_processing_time_ms = stats.total_processing_time_ms as f64 / stats.total_requests as f64;
            }
        }
        
        self.update_last_activity().await;
        
        debug!("Entry added successfully to session {}", self.session_id);
        Ok(())
    }
    
    /// Get session history
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<SessionEntry> {
        let history = self.history.read().await;
        match limit {
            Some(limit) => history.iter().rev().take(limit).cloned().collect(),
            None => history.iter().rev().cloned().collect(),
        }
    }
    
    /// Get context from cache
    pub async fn get_context(&self, key: &str) -> Option<serde_json::Value> {
        let cache = self.context_cache.read().await;
        cache.get(key).cloned()
    }
    
    /// Set context in cache
    pub async fn set_context(&self, key: String, value: serde_json::Value) {
        let mut cache = self.context_cache.write().await;
        cache.insert(key, value);
        self.update_last_activity().await;
    }
    
    /// Clear context cache
    pub async fn clear_context_cache(&self) {
        let mut cache = self.context_cache.write().await;
        cache.clear();
        self.update_last_activity().await;
    }
    
    /// Get last activity timestamp
    pub async fn get_last_activity(&self) -> DateTime<Utc> {
        *self.last_activity.read().await
    }
    
    /// Check if session is expired
    pub async fn is_expired(&self) -> bool {
        let last_activity = self.get_last_activity().await;
        let now = Utc::now();
        let duration_seconds = (now - last_activity).num_seconds() as u64;
        duration_seconds > self.config.timeout_seconds
    }
    
    /// Check if session duration exceeded
    pub async fn is_duration_exceeded(&self) -> bool {
        let now = Utc::now();
        let duration_seconds = (now - self.created_at).num_seconds() as u64;
        duration_seconds > self.config.max_duration_seconds
    }
    
    /// Check if token limit exceeded
    pub async fn is_token_limit_exceeded(&self) -> bool {
        let stats = self.stats.read().await;
        stats.total_tokens_generated > self.config.max_tokens_per_session
    }
    
    /// Validate session
    pub async fn validate(&self) -> Result<()> {
        // Check if session is expired
        if self.is_expired().await {
            self.set_state(SessionState::Timeout).await;
            return Err(InferenceError::InternalError("Session expired".to_string()));
        }
        
        // Check if duration exceeded
        if self.is_duration_exceeded().await {
            self.set_state(SessionState::Timeout).await;
            return Err(InferenceError::InternalError("Session duration exceeded".to_string()));
        }
        
        // Check if token limit exceeded
        if self.is_token_limit_exceeded().await {
            return Err(InferenceError::InternalError("Session token limit exceeded".to_string()));
        }
        
        // Check session state
        let state = self.get_state().await;
        match state {
            SessionState::Active | SessionState::Idle => Ok(()),
            SessionState::Processing => Ok(()), // Allow processing
            SessionState::Paused => Err(InferenceError::InternalError("Session is paused".to_string())),
            SessionState::Error(ref msg) => Err(InferenceError::InternalError(format!("Session error: {}", msg))),
            SessionState::Closed => Err(InferenceError::InternalError("Session is closed".to_string())),
            SessionState::Timeout => Err(InferenceError::InternalError("Session timeout".to_string())),
        }
    }
    
    /// Close session
    pub async fn close(&self) -> Result<()> {
        info!("Closing session: {}", self.session_id);
        
        self.set_state(SessionState::Closed).await;
        
        // Clear sensitive data
        self.clear_context_cache().await;
        
        info!("Session {} closed successfully", self.session_id);
        Ok(())
    }
    
    /// Reset session (clear history and cache)
    pub async fn reset(&self) -> Result<()> {
        info!("Resetting session: {}", self.session_id);
        
        // Clear history
        {
            let mut history = self.history.write().await;
            history.clear();
        }
        
        // Clear context cache
        self.clear_context_cache().await;
        
        // Reset statistics
        {
            let mut stats = self.stats.write().await;
            *stats = SessionStats::default();
        }
        
        // Reset state to active
        self.set_state(SessionState::Active).await;
        
        info!("Session {} reset successfully", self.session_id);
        Ok(())
    }
    
    /// Get session summary
    pub async fn get_summary(&self) -> SessionSummary {
        let stats = self.get_stats().await;
        let state = self.get_state().await;
        let metadata = self.get_metadata().await;
        let last_activity = self.get_last_activity().await;
        
        SessionSummary {
            session_id: self.session_id,
            user_id: self.user_id,
            state,
            created_at: self.created_at,
            last_activity,
            stats,
            metadata,
        }
    }
    
    /// Update last activity timestamp
    async fn update_last_activity(&self) {
        let mut last_activity = self.last_activity.write().await;
        *last_activity = Utc::now();
    }
    
    /// Check session limits
    async fn check_session_limits(&self, entry: &SessionEntry) -> Result<()> {
        // Check token limit
        let current_tokens = {
            let stats = self.stats.read().await;
            stats.total_tokens_generated
        };
        
        if current_tokens + entry.tokens_generated as u64 > self.config.max_tokens_per_session {
            return Err(InferenceError::InternalError("Session token limit would be exceeded".to_string()));
        }
        
        // Check context length limit
        let current_context_length = {
            let history = self.history.read().await;
            history.iter().map(|e| e.prompt.len() + e.response.len()).sum::<usize>()
        };
        
        let new_context_length = current_context_length + entry.prompt.len() + entry.response.len();
        if new_context_length > self.config.max_context_length {
            return Err(InferenceError::InternalError("Session context length limit would be exceeded".to_string()));
        }
        
        Ok(())
    }
}

/// Session summary
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: Uuid,
    pub user_id: Option<Uuid>,
    pub state: SessionState,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub stats: SessionStats,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SessionEntry {
    /// Create new session entry
    pub fn new(
        request_id: Uuid,
        prompt: String,
        response: String,
        tokens_generated: usize,
        processing_time_ms: u64,
    ) -> Self {
        Self {
            entry_id: Uuid::new_v4(),
            request_id,
            timestamp: Utc::now(),
            prompt,
            response,
            tokens_generated,
            processing_time_ms,
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}
