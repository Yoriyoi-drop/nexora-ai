//! Inference Agent
//! 
//! Agent untuk kontrol inference runtime.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info, warn, error};
use futures;

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};

/// Inference agent untuk mengontrol runtime inference
pub struct InferenceAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Inference engine reference
    inference_engine: Option<Arc<dyn InferenceEngine>>,
    /// Active inference sessions
    active_sessions: Arc<std::sync::Mutex<HashMap<Uuid, InferenceSession>>>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: InferenceAgentConfig,
}

/// Configuration untuk inference agent
#[derive(Debug, Clone)]
pub struct InferenceAgentConfig {
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Default timeout (seconds)
    pub default_timeout_seconds: u64,
    /// Enable streaming
    pub enable_streaming: bool,
    /// Session cleanup interval (seconds)
    pub cleanup_interval_seconds: u64,
    /// Maximum session age (seconds)
    pub max_session_age_seconds: u64,
}

/// Inference session information
#[derive(Debug, Clone)]
pub struct InferenceSession {
    /// Session ID
    pub session_id: Uuid,
    /// Model ID
    pub model_id: String,
    /// Session status
    pub status: InferenceSessionStatus,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last activity
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Tokens generated
    pub tokens_generated: u64,
    /// Processing time (ms)
    pub processing_time_ms: u64,
    /// Session metadata
    pub metadata: HashMap<String, Value>,
}

/// Inference session status
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum InferenceSessionStatus {
    Initializing,
    Ready,
    Running,
    Completed,
    Failed(String),
    Timeout,
    Cancelled,
}

/// Trait untuk inference engine
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Start inference session
    async fn start_session(&self, session_id: Uuid, model_id: &str, config: &Value) -> Result<()>;
    
    /// Generate tokens
    async fn generate_tokens(&self, session_id: Uuid, prompt: &str, max_tokens: u32) -> Result<String>;
    
    /// Stream tokens
    async fn stream_tokens(&self, session_id: Uuid, prompt: &str, max_tokens: u32) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send>>;
    
    /// Stop session
    async fn stop_session(&self, session_id: Uuid) -> Result<()>;
    
    /// Get session status
    async fn get_session_status(&self, session_id: Uuid) -> Result<InferenceSessionStatus>;
    
    /// Health check
    async fn health_check(&self) -> Result<bool>;
}

impl InferenceAgent {
    /// Create new inference agent
    pub fn new(config: InferenceAgentConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "InferenceAgent".to_string(),
            status: AgentStatus::Initializing,
            inference_engine: None,
            active_sessions: Arc::new(std::sync::Mutex::new(HashMap::new())),
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Set inference engine
    pub fn set_inference_engine(&mut self, engine: Arc<dyn InferenceEngine>) {
        self.inference_engine = Some(engine);
    }
    
    /// Start new inference session
    pub async fn start_inference_session(
        &self,
        model_id: String,
        config: Value,
    ) -> Result<Uuid> {
        debug!("Starting inference session for model: {}", model_id);
        
        // Check concurrent session limit
        {
            let sessions = self.active_sessions.lock().unwrap();
            if sessions.len() >= self.config.max_concurrent_sessions {
                return Err(AgentError::ProcessingError(
                    format!("Maximum concurrent sessions ({}) reached", self.config.max_concurrent_sessions)
                ));
            }
        }
        
        let session_id = Uuid::new_v4();
        let now = chrono::Utc::now();
        
        // Create session record
        let session = InferenceSession {
            session_id,
            model_id: model_id.clone(),
            status: InferenceSessionStatus::Initializing,
            created_at: now,
            last_activity: now,
            tokens_generated: 0,
            processing_time_ms: 0,
            metadata: HashMap::new(),
        };
        
        // Add to active sessions
        {
            let mut sessions = self.active_sessions.lock().unwrap();
            sessions.insert(session_id, session);
        }
        
        // Initialize session with inference engine
        if let Some(engine) = &self.inference_engine {
            match engine.start_session(session_id, &model_id, &config).await {
                Ok(_) => {
                    // Update session status
                    self.update_session_status(session_id, InferenceSessionStatus::Ready).await?;
                    info!("Inference session {} started successfully", session_id);
                    Ok(session_id)
                }
                Err(e) => {
                    // Clean up failed session
                    self.cleanup_session(session_id).await;
                    error!("Failed to start inference session: {}", e);
                    Err(e)
                }
            }
        } else {
            Err(AgentError::ProcessingError(
                "No inference engine configured".to_string()
            ))
        }
    }
    
    /// Generate text from session
    pub async fn generate_text(
        &self,
        session_id: Uuid,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String> {
        debug!("Generating text for session: {}", session_id);
        
        // Check session exists and is ready
        {
            let sessions = self.active_sessions.lock().unwrap();
            if let Some(session) = sessions.get(&session_id) {
                if session.status != InferenceSessionStatus::Ready && session.status != InferenceSessionStatus::Running {
                    return Err(AgentError::ProcessingError(
                        format!("Session {} is not ready for generation (status: {:?})", session_id, session.status)
                    ));
                }
            } else {
                return Err(AgentError::ProcessingError(
                    format!("Session {} not found", session_id)
                ));
            }
        }
        
        // Update session status to running
        self.update_session_status(session_id, InferenceSessionStatus::Running).await?;
        
        let start_time = std::time::Instant::now();
        
        // Generate with inference engine
        let result = if let Some(engine) = &self.inference_engine {
            engine.generate_tokens(session_id, prompt, max_tokens).await
        } else {
            Err(AgentError::ProcessingError(
                "No inference engine configured".to_string()
            ))
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        match result {
            Ok(text) => {
                // Update session stats
                self.update_session_stats(session_id, text.split_whitespace().count() as u64, processing_time).await?;
                self.update_session_status(session_id, InferenceSessionStatus::Completed).await?;
                
                debug!("Text generation completed for session: {}", session_id);
                Ok(text)
            }
            Err(e) => {
                self.update_session_status(session_id, InferenceSessionStatus::Failed(e.to_string())).await?;
                error!("Text generation failed for session {}: {}", session_id, e);
                Err(e)
            }
        }
    }
    
    /// Stream text from session
    pub async fn stream_text(
        &self,
        session_id: Uuid,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send>> {
        debug!("Starting text stream for session: {}", session_id);
        
        if !self.config.enable_streaming {
            return Err(AgentError::ProcessingError("Streaming is disabled".to_string()));
        }
        
        // Check session exists and is ready
        {
            let sessions = self.active_sessions.lock().unwrap();
            if let Some(session) = sessions.get(&session_id) {
                if session.status != InferenceSessionStatus::Ready && session.status != InferenceSessionStatus::Running {
                    return Err(AgentError::ProcessingError(
                        format!("Session {} is not ready for streaming", session_id)
                    ));
                }
            } else {
                return Err(AgentError::ProcessingError(
                    format!("Session {} not found", session_id)
                ));
            }
        }
        
        // Update session status to running
        self.update_session_status(session_id, InferenceSessionStatus::Running).await?;
        
        // Stream with inference engine
        if let Some(engine) = &self.inference_engine {
            engine.stream_tokens(session_id, prompt, max_tokens).await
        } else {
            Err(AgentError::ProcessingError(
                "No inference engine configured".to_string()
            ))
        }
    }
    
    /// Stop inference session
    pub async fn stop_inference_session(&self, session_id: Uuid) -> Result<()> {
        debug!("Stopping inference session: {}", session_id);
        
        // Stop with inference engine
        if let Some(engine) = &self.inference_engine {
            engine.stop_session(session_id).await?;
        }
        
        // Update session status
        self.update_session_status(session_id, InferenceSessionStatus::Cancelled).await?;
        
        // Clean up session
        self.cleanup_session(session_id).await;
        
        info!("Inference session {} stopped", session_id);
        Ok(())
    }
    
    /// Get session information
    pub async fn get_session_info(&self, session_id: Uuid) -> Result<Option<InferenceSession>> {
        let sessions = self.active_sessions.lock().unwrap();
        Ok(sessions.get(&session_id).cloned())
    }
    
    /// List active sessions
    pub async fn list_active_sessions(&self) -> Vec<InferenceSession> {
        let sessions = self.active_sessions.lock().unwrap();
        sessions.values().cloned().collect()
    }
    
    /// Cleanup old sessions
    pub async fn cleanup_old_sessions(&self) -> Result<usize> {
        let now = chrono::Utc::now();
        let mut sessions_to_remove = Vec::new();
        
        {
            let sessions = self.active_sessions.lock().unwrap();
            for (session_id, session) in sessions.iter() {
                let age_seconds = (now - session.created_at).num_seconds() as u64;
                if age_seconds > self.config.max_session_age_seconds {
                    sessions_to_remove.push(*session_id);
                }
            }
        }
        
        // Remove old sessions
        let removed_count = sessions_to_remove.len();
        for session_id in &sessions_to_remove {
            self.cleanup_session(*session_id).await;
        }
        
        Ok(removed_count)
    }
    
    /// Update session status
    async fn update_session_status(&self, session_id: Uuid, status: InferenceSessionStatus) -> Result<()> {
        let mut sessions = self.active_sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(&session_id) {
            session.status = status;
            session.last_activity = chrono::Utc::now();
            Ok(())
        } else {
            Err(AgentError::ProcessingError(format!("Session {} not found", session_id)))
        }
    }
    
    /// Update session statistics
    async fn update_session_stats(&self, session_id: Uuid, tokens_generated: u64, processing_time_ms: u64) -> Result<()> {
        let mut sessions = self.active_sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(&session_id) {
            session.tokens_generated += tokens_generated;
            session.processing_time_ms += processing_time_ms;
            session.last_activity = chrono::Utc::now();
            Ok(())
        } else {
            Err(AgentError::ProcessingError(format!("Session {} not found", session_id)))
        }
    }
    
    /// Clean up session
    async fn cleanup_session(&self, session_id: Uuid) {
        let mut sessions = self.active_sessions.lock().unwrap();
        sessions.remove(&session_id);
    }
    
    /// Get inference statistics
    pub fn get_inference_stats(&self) -> InferenceStats {
        let sessions = self.active_sessions.lock().unwrap();
        let total_tokens: u64 = sessions.values().map(|s| s.tokens_generated).sum();
        let total_processing_time: u64 = sessions.values().map(|s| s.processing_time_ms).sum();
        
        InferenceStats {
            active_sessions: sessions.len(),
            total_tokens_generated: total_tokens,
            total_processing_time_ms: total_processing_time,
            max_concurrent_sessions: self.config.max_concurrent_sessions,
            streaming_enabled: self.config.enable_streaming,
        }
    }
}

/// Inference statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceStats {
    pub active_sessions: usize,
    pub total_tokens_generated: u64,
    pub total_processing_time_ms: u64,
    pub max_concurrent_sessions: usize,
    pub streaming_enabled: bool,
}

#[async_trait]
impl Agent for InferenceAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "inference"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing InferenceAgent");
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("InferenceAgent received message: {}", message.message_type);
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("InferenceAgent processing request for session: {}", context.session_id);
        
        // Extract action from context
        let action = context.parameters.get("action").and_then(|v| v.as_str()).unwrap_or("status");
        
        let result = match action {
            "start_session" => {
                let model_id = context.parameters.get("model_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("model_id required".to_string()))?;
                
                let config = context.parameters.get("config").cloned().unwrap_or(Value::Object(serde_json::Map::new()));
                
                let session_id = self.start_inference_session(model_id.to_string(), config).await?;
                
                json!({
                    "action": "start_session",
                    "session_id": session_id,
                    "status": "started"
                })
            }
            
            "generate" => {
                let session_id_str = context.parameters.get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("session_id required".to_string()))?;
                
                let session_id = Uuid::parse_str(session_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid session_id".to_string()))?;
                
                let prompt = context.parameters.get("prompt")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("prompt required".to_string()))?;
                
                let max_tokens = context.parameters.get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(100) as u32;
                
                let text = self.generate_text(session_id, prompt, max_tokens).await?;
                
                let token_count = text.split_whitespace().count();
                json!({
                    "action": "generate",
                    "session_id": session_id,
                    "text": text,
                    "tokens": token_count
                })
            }
            
            "stop_session" => {
                let session_id_str = context.parameters.get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("session_id required".to_string()))?;
                
                let session_id = Uuid::parse_str(session_id_str)
                    .map_err(|_| AgentError::ProcessingError("Invalid session_id".to_string()))?;
                
                self.stop_inference_session(session_id).await?;
                
                json!({
                    "action": "stop_session",
                    "session_id": session_id,
                    "status": "stopped"
                })
            }
            
            "list_sessions" => {
                let sessions = self.list_active_sessions().await;
                let session_list: Vec<Value> = sessions.iter()
                    .map(|s| json!({
                        "session_id": s.session_id,
                        "model_id": s.model_id,
                        "status": s.status,
                        "created_at": s.created_at,
                        "tokens_generated": s.tokens_generated,
                        "processing_time_ms": s.processing_time_ms
                    }))
                    .collect();
                
                json!({
                    "action": "list_sessions",
                    "sessions": session_list,
                    "count": session_list.len()
                })
            }
            
            "stats" => {
                let stats = self.get_inference_stats();
                json!({
                    "action": "stats",
                    "stats": stats
                })
            }
            
            _ => {
                return Err(AgentError::ProcessingError(format!("Unknown action: {}", action)));
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.messages_processed - 1) as f64 + 
             processing_time as f64) / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();
        
        let response = AgentResponse::success(
            context.session_id,
            result,
            processing_time,
        );
        
        Ok(response)
    }
    
    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("InferenceAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down InferenceAgent");
        
        // Stop all active sessions
        let session_ids: Vec<Uuid> = {
            let sessions = self.active_sessions.lock().unwrap();
            sessions.keys().cloned().collect()
        };
        
        for session_id in session_ids {
            if let Err(e) = self.stop_inference_session(session_id).await {
                warn!("Failed to stop session {}: {}", session_id, e);
            }
        }
        
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check inference engine health
        if let Some(engine) = &self.inference_engine {
            engine.health_check().await
        } else {
            Err(AgentError::ProcessingError(
                "No inference engine configured".to_string()
            ))
        }
    }
    
    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
    
    fn get_config(&self) -> AgentConfig {
        self.config.clone().into()
    }
}

impl From<InferenceAgentConfig> for AgentConfig {
    fn from(config: InferenceAgentConfig) -> Self {
        AgentConfig {
            agent_id: "inference_agent".to_string(),
            agent_type: "inference".to_string(),
            max_concurrent_tasks: config.max_concurrent_sessions,
            timeout_seconds: config.default_timeout_seconds,
        }
    }
}

impl Default for InferenceAgentConfig {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 10,
            default_timeout_seconds: 60,
            enable_streaming: true,
            cleanup_interval_seconds: 300, // 5 minutes
            max_session_age_seconds: 3600, // 1 hour
        }
    }
}
