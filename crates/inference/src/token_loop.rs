//! Token Loop
//! 
//! Main token generation loop untuk inference.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::{debug, info, warn};
use chrono::{DateTime, Utc};

use crate::{
    Result, InferenceError, InferenceRequest, InferenceResponse, GeneratedToken,
    FinishReason
};
use crate::decoding::{DecodingStrategy, DecodingContext};
use crate::stop_conditions::{StopConditions, StopContext};
use crate::streaming::StreamingEngine;

/// Configuration untuk token loop
#[derive(Debug, Clone)]
pub struct TokenLoopConfig {
    /// Maximum tokens per generation
    pub max_tokens: u32,
    /// Enable streaming
    pub enable_streaming: bool,
    /// Streaming buffer size
    pub streaming_buffer_size: usize,
    /// Enable early stopping
    pub enable_early_stopping: bool,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f32,
    /// Enable token validation
    pub enable_token_validation: bool,
    /// Validation timeout (ms)
    pub validation_timeout_ms: u64,
    /// Loop iteration timeout (ms)
    pub loop_timeout_ms: u64,
}

impl Default for TokenLoopConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            enable_streaming: false,
            streaming_buffer_size: 10,
            enable_early_stopping: true,
            min_confidence_threshold: 0.1,
            enable_token_validation: true,
            validation_timeout_ms: 1000,
            loop_timeout_ms: 5000,
        }
    }
}

/// Token loop state
#[derive(Debug, Clone, PartialEq)]
pub enum TokenLoopState {
    /// Loop tidak diinisialisasi
    Uninitialized,
    /// Loop sedang diinisialisasi
    Initializing,
    /// Loop sedang berjalan
    Running,
    /// Loop sedang pause
    Paused,
    /// Loop selesai
    Completed,
    /// Loop error
    Error(String),
    /// Loop di-cancel
    Cancelled,
}

/// Token generation loop
pub struct TokenLoop {
    /// Configuration
    config: TokenLoopConfig,
    /// Decoding strategy
    decoding_strategy: Arc<dyn DecodingStrategy>,
    /// Stop conditions
    stop_conditions: Arc<StopConditions>,
    /// Streaming engine (optional)
    streaming_engine: Option<Arc<RwLock<StreamingEngine>>>,
    /// Loop state
    state: Arc<RwLock<TokenLoopState>>,
    /// Loop statistics
    stats: Arc<RwLock<TokenLoopStats>>,
    /// Active loops
    active_loops: Arc<RwLock<HashMap<Uuid, LoopInfo>>>,
}

/// Loop information
#[derive(Debug)]
struct LoopInfo {
    /// Loop ID
    loop_id: Uuid,
    /// Request ID
    request_id: Uuid,
    /// Start time
    start_time: DateTime<Utc>,
    /// Current token count
    token_count: usize,
    /// Generated tokens
    tokens: Vec<GeneratedToken>,
    /// Stream ID (if streaming)
    stream_id: Option<Uuid>,
    /// Loop metadata
    _metadata: HashMap<String, serde_json::Value>,
}

/// Token loop statistics
#[derive(Debug, Clone, Default)]
pub struct TokenLoopStats {
    /// Total loops created
    pub total_loops: u64,
    /// Active loops
    pub active_loops: usize,
    /// Completed loops
    pub completed_loops: u64,
    /// Cancelled loops
    pub cancelled_loops: u64,
    /// Failed loops
    pub failed_loops: u64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Average tokens per loop
    pub avg_tokens_per_loop: f64,
    /// Average loop duration (ms)
    pub avg_loop_duration_ms: f64,
    /// Peak concurrent loops
    pub peak_concurrent_loops: usize,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl TokenLoop {
    /// Create new token loop
    pub fn new(
        config: TokenLoopConfig,
        decoding_strategy: Arc<dyn DecodingStrategy>,
        stop_conditions: Arc<StopConditions>,
    ) -> Self {
        Self {
            config,
            decoding_strategy,
            stop_conditions,
            streaming_engine: None,
            state: Arc::new(RwLock::new(TokenLoopState::Uninitialized)),
            stats: Arc::new(RwLock::new(TokenLoopStats::default())),
            active_loops: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Set streaming engine
    pub fn with_streaming_engine(mut self, streaming_engine: Arc<RwLock<StreamingEngine>>) -> Self {
        self.streaming_engine = Some(streaming_engine);
        self
    }
    
    /// Initialize token loop
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing token loop");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = TokenLoopState::Initializing;
        }
        
        // Initialize streaming engine if provided
        if let Some(streaming_engine) = &self.streaming_engine {
            streaming_engine.write().await.initialize().await?;
        }
        
        // Update state to ready
        {
            let mut state = self.state.write().await;
            *state = TokenLoopState::Running;
        }
        
        info!("Token loop initialized successfully");
        Ok(())
    }
    
    /// Run token generation loop
    pub async fn run_loop(
        &self,
        request: &InferenceRequest,
        initial_logits: Vec<Vec<f32>>,
    ) -> Result<InferenceResponse> {
        debug!("Starting token generation loop for request: {}", request.request_id);
        
        let loop_id = Uuid::new_v4();
        let start_time = Utc::now();
        
        // Create loop info
        let loop_info = LoopInfo {
            loop_id,
            request_id: request.request_id,
            start_time,
            token_count: 0,
            tokens: Vec::new(),
            stream_id: None,
            _metadata: request.metadata.clone(),
        };
        
        // Add to active loops
        {
            let mut active_loops = self.active_loops.write().await;
            active_loops.insert(loop_id, loop_info);
            
            // Update statistics
            let mut stats = self.stats.write().await;
            stats.total_loops += 1;
            stats.active_loops = active_loops.len();
            stats.peak_concurrent_loops = stats.peak_concurrent_loops.max(stats.active_loops);
            stats.last_updated = Utc::now();
        }
        
        // Create stream if streaming enabled
        let stream_id = if self.config.enable_streaming {
            if let Some(streaming_engine) = &self.streaming_engine {
                let (sid, _rx) = streaming_engine.write().await.create_stream().await?;
                {
                    let mut active_loops = self.active_loops.write().await;
                    if let Some(info) = active_loops.get_mut(&loop_id) {
                        info.stream_id = Some(sid);
                    }
                }
                Some(sid)
            } else {
                None
            }
        } else {
            None
        };
        
        // Run main generation loop
        let result = self.run_generation_loop(
            loop_id,
            request,
            initial_logits,
            stream_id,
            start_time,
        ).await;
        
        // Clean up
        self.cleanup_loop(loop_id).await;
        
        result
    }
    
    /// Run main generation loop
    async fn run_generation_loop(
        &self,
        loop_id: Uuid,
        request: &InferenceRequest,
        initial_logits: Vec<Vec<f32>>,
        stream_id: Option<Uuid>,
        start_time: chrono::DateTime<Utc>,
    ) -> Result<InferenceResponse> {
        let mut tokens = Vec::with_capacity(initial_logits.len());
        let mut token_frequencies = HashMap::with_capacity(initial_logits.len());
        let mut finish_reason = FinishReason::Unknown;
        
        let mut decoding_context = DecodingContext::new(initial_logits[0].len());
        
        for (step, logits) in initial_logits.iter().enumerate() {
            // Check loop state
            {
                let state = self.state.read().await;
                match *state {
                    TokenLoopState::Running => {},
                    TokenLoopState::Cancelled => {
                        finish_reason = FinishReason::Cancelled;
                        break;
                    }
                    TokenLoopState::Error(ref msg) => {
                        return Err(InferenceError::InternalError(format!("Loop error: {}", msg)));
                    }
                    _ => {
                        return Err(InferenceError::InternalError("Loop not in running state".to_string()));
                    }
                }
            }
            
            // Check stop conditions
            let stop_context = StopContext {
                token_count: tokens.len(),
                start_time, // Use actual start time from loop creation
                current_time: Utc::now(),
                metadata: HashMap::new(),
            };
            
            if let Some(reason) = self.stop_conditions.should_stop(&tokens, &stop_context).await {
                finish_reason = match reason.as_str() {
                    "Maximum tokens reached" => FinishReason::MaxTokens,
                    "Stop sequence encountered" => FinishReason::StopSequence,
                    "End of sequence token encountered" => FinishReason::EndOfSequence,
                    "Time limit reached" => FinishReason::Timeout,
                    _ => FinishReason::Unknown,
                };
                break;
            }
            
            // Check token limit
            if tokens.len() >= request.max_tokens as usize {
                finish_reason = FinishReason::MaxTokens;
                break;
            }
            
            // Select next token
            let decoding_config = crate::decoding::DecodingConfig {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
                presence_penalty: request.presence_penalty,
                frequency_penalty: request.frequency_penalty,
                ..Default::default()
            };
            
            let token_selection = self.decoding_strategy.select_token(
                logits,
                &decoding_config,
                &decoding_context,
            )?;
            
            // Validate token if enabled
            if self.config.enable_token_validation {
                if !self.validate_token(&token_selection).await? {
                    warn!("Token validation failed for token {}", token_selection.token_id);
                    continue;
                }
            }
            
            // Create generated token and store in vec first
            tokens.push(GeneratedToken::new(
                token_selection.token_id,
                token_selection.token_text.clone(),
                token_selection.log_prob,
                step,
            ));
            let gen = tokens.last().unwrap();
            
            decoding_context.add_token(gen.clone());
            *token_frequencies.entry(gen.token_id).or_insert(0) += 1;
            
            if let Some(stream_id) = stream_id {
                if let Some(streaming_engine) = &self.streaming_engine {
                    streaming_engine.write().await.send_token(stream_id, gen.clone()).await?;
                }
            }
            
            {
                let mut active_loops = self.active_loops.write().await;
                if let Some(info) = active_loops.get_mut(&loop_id) {
                    info.token_count = tokens.len();
                    info.tokens.push(gen.clone());
                }
            }
            
            debug!("Generated token {} at step {}: {}", 
                   gen.token_id, step, gen.token_text);
        }
        
        // Create response
        let _generated_text = tokens.iter().map(|t| t.token_text.clone()).collect::<String>();
        let processing_time = (Utc::now() - start_time).num_milliseconds() as u64;
        
        let response = InferenceResponse::new(request.request_id)
            .with_finish_reason(finish_reason)
            .with_inference_time(processing_time);
        
        // Add tokens to response
        let mut final_response = response;
        for token in tokens {
            final_response.add_token(token);
        }
        
        Ok(final_response)
    }
    
    /// Cancel active loop
    pub async fn cancel_loop(&self, loop_id: Uuid) -> Result<bool> {
        debug!("Cancelling loop: {}", loop_id);
        
        let mut active_loops = self.active_loops.write().await;
        
        if let Some(info) = active_loops.get(&loop_id) {
            // Cancel stream if exists
            if let Some(stream_id) = info.stream_id {
                if let Some(streaming_engine) = &self.streaming_engine {
                    streaming_engine.write().await.cancel_stream(stream_id).await?;
                }
            }
            
            // Remove from active loops
            active_loops.remove(&loop_id);
            
            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.active_loops = active_loops.len();
                stats.cancelled_loops += 1;
            }
            
            debug!("Loop {} cancelled successfully", loop_id);
            Ok(true)
        } else {
            debug!("Loop {} not found for cancellation", loop_id);
            Ok(false)
        }
    }
    
    /// Cancel loop by request ID
    pub async fn cancel_loop_by_request(&self, request_id: Uuid) -> Result<bool> {
        let active_loops = self.active_loops.read().await;
        
        let target_loop_id = active_loops.iter()
            .find_map(|(loop_id, info)| {
                if info.request_id == request_id {
                    Some(*loop_id)
                } else {
                    None
                }
            });
        
        drop(active_loops);
        
        if let Some(loop_id) = target_loop_id {
            return self.cancel_loop(loop_id).await;
        }
        
        Ok(false)
    }
    
    /// Get loop status
    pub async fn get_loop_status(&self, loop_id: Uuid) -> Option<LoopStatus> {
        let active_loops = self.active_loops.read().await;
        
        if let Some(info) = active_loops.get(&loop_id) {
            let duration = (Utc::now() - info.start_time).num_milliseconds() as u64;
            
            Some(LoopStatus {
                loop_id: info.loop_id,
                request_id: info.request_id,
                token_count: info.token_count,
                duration_ms: duration,
                is_streaming: info.stream_id.is_some(),
                start_time: info.start_time,
            })
        } else {
            None
        }
    }
    
    /// Get loop statistics
    pub async fn get_stats(&self) -> TokenLoopStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update active loops count
        stats.active_loops = {
            let active_loops = self.active_loops.read().await;
            active_loops.len()
        };
        
        // Calculate average tokens per loop
        if stats.total_loops > 0 {
            stats.avg_tokens_per_loop = stats.total_tokens_generated as f64 / stats.total_loops as f64;
        }
        
        stats
    }
    
    /// Shutdown token loop
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down token loop");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = TokenLoopState::Cancelled;
        }
        
        // Cancel all active loops
        let loop_ids: Vec<Uuid> = {
            let active_loops = self.active_loops.read().await;
            active_loops.keys().cloned().collect()
        };
        
        for loop_id in loop_ids {
            if let Err(e) = self.cancel_loop(loop_id).await {
                warn!("Failed to cancel loop {}: {}", loop_id, e);
            }
        }
        
        // Shutdown streaming engine
        if let Some(streaming_engine) = &self.streaming_engine {
            streaming_engine.write().await.shutdown().await?;
        }
        
        info!("Token loop shutdown complete");
        Ok(())
    }
    
    /// Clean up loop
    async fn cleanup_loop(&self, loop_id: Uuid) {
        // Remove from active loops
        {
            let mut active_loops = self.active_loops.write().await;
            if let Some(info) = active_loops.remove(&loop_id) {
                // Update statistics
                let mut stats = self.stats.write().await;
                stats.active_loops = active_loops.len();
                stats.completed_loops += 1;
                stats.total_tokens_generated += info.token_count as u64;
            }
        }
    }
    
    /// Validate token
    async fn validate_token(&self, token: &crate::decoding::TokenSelection) -> Result<bool> {
        // Simple validation - check if token is reasonable
        if token.selection_prob < 0.0 || token.selection_prob > 1.0 {
            return Ok(false);
        }
        
        if !token.log_prob.is_finite() {
            return Ok(false);
        }
        
        if token.token_text.is_empty() {
            return Ok(false);
        }
        
        Ok(true)
    }
}

/// Loop status information
#[derive(Debug, Clone)]
pub struct LoopStatus {
    pub loop_id: Uuid,
    pub request_id: Uuid,
    pub token_count: usize,
    pub duration_ms: u64,
    pub is_streaming: bool,
    pub start_time: DateTime<Utc>,
}

impl Default for TokenLoop {
    fn default() -> Self {
        Self::new(
            TokenLoopConfig::default(),
            Arc::new(crate::decoding::GreedyDecoding),
            Arc::new(StopConditions::default()),
        )
    }
}
