//! Inference Engine
//! 
//! Core inference execution untuk Nexora AI.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc, Mutex};
use uuid::Uuid;
use tracing::{debug, info, warn, error};

use crate::{
    Result, InferenceError, InferenceRequest, InferenceResponse, GeneratedToken,
    FinishReason
};
use crate::runtime::InferenceRuntime;
use crate::scheduler::RequestScheduler;
use crate::kv_cache::KVCache;
use crate::session::InferenceSession;
use crate::decoding::DecodingStrategy;
use crate::streaming::StreamingEngine;

/// Configuration untuk inference engine
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Default model ID
    pub default_model_id: String,
    /// Enable request queuing
    pub enable_queuing: bool,
    /// Queue size limit
    pub queue_size_limit: usize,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size limit (MB)
    pub cache_size_limit_mb: usize,
    /// Enable streaming
    pub enable_streaming: bool,
    /// Default timeout (seconds)
    pub default_timeout_seconds: u64,
    /// Metrics collection interval (seconds)
    pub metrics_interval_seconds: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            default_model_id: "default".to_string(),
            enable_queuing: true,
            queue_size_limit: 100,
            enable_caching: true,
            cache_size_limit_mb: 1024, // 1GB
            enable_streaming: true,
            default_timeout_seconds: 30,
            metrics_interval_seconds: 60,
        }
    }
}

/// Main inference engine
pub struct InferenceEngine {
    /// Engine configuration
    config: InferenceConfig,
    /// Runtime state
    runtime: Arc<InferenceRuntime>,
    /// Request scheduler
    scheduler: Arc<RwLock<RequestScheduler>>,
    /// KV cache
    kv_cache: Arc<RwLock<KVCache>>,
    /// Session manager
    session_manager: Arc<RwLock<HashMap<Uuid, InferenceSession>>>,
    /// Decoding strategies
    decoding_strategies: HashMap<String, Box<dyn DecodingStrategy>>,
    /// Streaming engine
    streaming_engine: Option<Arc<RwLock<StreamingEngine>>>,
    /// Request channel (bounded with backpressure)
    request_tx: mpsc::Sender<InferenceRequest>,
    /// Request receiver
    request_rx: Arc<Mutex<Option<mpsc::Receiver<InferenceRequest>>>>,
    /// Active requests tracking
    active_requests: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    /// Engine state
    state: Arc<RwLock<EngineState>>,
}

/// Engine state
#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    /// Engine tidak diinisialisasi
    Uninitialized,
    /// Engine sedang diinisialisasi
    Initializing,
    /// Engine siap menerima request
    Ready,
    /// Engine sedang shutdown
    ShuttingDown,
    /// Engine sudah shutdown
    Shutdown,
}

impl InferenceEngine {
    /// Create new inference engine
    pub fn new(config: InferenceConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.queue_size_limit.max(1));
        
        let mut engine = Self {
            runtime: Arc::new(InferenceRuntime::new()),
            scheduler: Arc::new(RwLock::new(RequestScheduler::new())),
            kv_cache: Arc::new(RwLock::new(KVCache::new())),
            session_manager: Arc::new(RwLock::new(HashMap::new())),
            decoding_strategies: HashMap::new(),
            streaming_engine: if config.enable_streaming {
                Some(Arc::new(RwLock::new(StreamingEngine::new())))
            } else {
                None
            },
            request_tx,
            request_rx: Arc::new(Mutex::new(Some(request_rx))),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(EngineState::Uninitialized)),
            config,
        };
        
        // Add default decoding strategies
        engine.add_default_decoding_strategies();
        
        engine
    }
    
    /// Initialize engine
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing inference engine");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::Initializing;
        }
        
        // Initialize runtime
        self.runtime.initialize().await?;
        
        // Initialize scheduler
        self.scheduler.write().await.initialize().await?;
        
        // Initialize KV cache
        if self.config.enable_caching {
            self.kv_cache.write().await.initialize().await?;
        }
        
        // Add default decoding strategies
        self.add_default_decoding_strategies();
        
        // Initialize streaming engine
        if let Some(streaming_engine) = &self.streaming_engine {
            streaming_engine.write().await.initialize().await?;
        }
        
        // Start request processing loop
        self.start_request_loop().await?;
        
        // Update state to ready
        {
            let mut state = self.state.write().await;
            *state = EngineState::Ready;
        }
        
        info!("Inference engine initialized successfully");
        Ok(())
    }
    
    /// Submit inference request
    pub async fn submit_request(&self, request: InferenceRequest) -> Result<mpsc::UnboundedReceiver<InferenceResponse>> {
        debug!("Submitting inference request: {}", request.request_id);
        
        // Check engine state
        {
            let state = self.state.read().await;
            match *state {
                EngineState::Ready => {},
                EngineState::ShuttingDown | EngineState::Shutdown => {
                    return Err(InferenceError::EngineNotInitialized("Engine is shutting down".to_string()));
                }
                _ => {
                    return Err(InferenceError::EngineNotInitialized("Engine not ready".to_string()));
                }
            }
        }
        
        // Validate request
        self.validate_request(&request)?;
        
        // Check active request limit for backpressure
        {
            let active = self.active_requests.read().await;
            if active.len() >= self.config.max_concurrent_requests {
                return Err(InferenceError::ResourceExhausted(
                    format!("Max concurrent requests reached ({})", self.config.max_concurrent_requests)
                ));
            }
        }
        
        // Create response channel (unbounded — scheduler requires UnboundedSender)
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        
        // Submit to scheduler
        self.scheduler.write().await.submit_request(request.request_id, response_tx).await?;
        
        // Send to request processing loop (bounded — will return error if queue full)
        if let Err(_) = self.request_tx.try_send(request) {
            return Err(InferenceError::ResourceExhausted(
                "Request queue is full, try again later".to_string()
            ));
        }
        
        Ok(response_rx)
    }
    
    /// Submit streaming request
    pub async fn submit_streaming_request(&self, request: InferenceRequest) -> Result<mpsc::UnboundedReceiver<GeneratedToken>> {
        debug!("Submitting streaming request: {}", request.request_id);
        
        if !self.config.enable_streaming {
            return Err(InferenceError::InvalidRequest("Streaming is disabled".to_string()));
        }
        
        if let Some(streaming_engine) = &self.streaming_engine {
            streaming_engine.write().await.submit_request(request).await.map_err(|e| InferenceError::InternalError(e.to_string()))
        } else {
            Err(InferenceError::InternalError("Streaming engine not available".to_string()))
        }
    }
    
    /// Cancel request
    pub async fn cancel_request(&self, request_id: Uuid) -> Result<bool> {
        debug!("Cancelling request: {}", request_id);
        
        // Try to cancel in scheduler
        let scheduler_cancelled = self.scheduler.write().await.cancel_request(request_id).await?;
        
        // Try to cancel active task
        let task_cancelled = {
            let mut active_requests = self.active_requests.write().await;
            if let Some(task) = active_requests.remove(&request_id) {
                task.abort();
                true
            } else {
                false
            }
        };
        
        // Cancel in streaming engine
        let streaming_cancelled = if let Some(streaming_engine) = &self.streaming_engine {
            streaming_engine.write().await.cancel_stream(request_id).await.unwrap_or(false)
        } else {
            false
        };
        
        Ok(scheduler_cancelled || task_cancelled || streaming_cancelled)
    }
    
    /// Get request status
    pub async fn get_request_status(&self, request_id: Uuid) -> Result<RequestStatus> {
        // Check in scheduler
        let status = self.scheduler.read().await.get_request_status(request_id).await?;
        
        // Check in streaming engine
        if let Some(streaming_engine) = &self.streaming_engine {
            let stream_active = streaming_engine.read().await.get_stream_status(request_id).await?;
            if stream_active.is_some() {
                return Ok(RequestStatus::Processing);
            }
        }
        
        Ok(status.map(RequestStatus::from_scheduler_status).unwrap_or(RequestStatus::Queued))
    }
    
    /// Create or get session
    pub async fn get_session(&self, session_id: Uuid) -> Result<Arc<InferenceSession>> {
        let mut sessions = self.session_manager.write().await;
        
        // Evict expired sessions first if at capacity
        if sessions.len() >= self.config.max_concurrent_requests * 2 {
            let now = chrono::Utc::now();
            sessions.retain(|_, s| {
                let age_secs = (now - s.created_at()).num_seconds() as u64;
                age_secs < s.config().timeout_seconds
            });
        }
        
        if let Some(session) = sessions.get(&session_id) {
            Ok(Arc::new(session.clone()))
        } else {
            let session = InferenceSession::new(session_id);
            sessions.insert(session_id, session.clone());
            Ok(Arc::new(session))
        }
    }
    
    /// Evict stale sessions from session manager
    pub async fn evict_stale_sessions(&self) -> usize {
        let mut sessions = self.session_manager.write().await;
        let before = sessions.len();
        let now = chrono::Utc::now();
        let timeout = chrono::Duration::seconds(
            InferenceSession::default_timeout_seconds() as i64
        );
        sessions.retain(|_, s| {
            // Session is too young to expire in the sync check — use creation time proxy
            // For full async eviction we'd use a separate task
            let age = now - s.created_at();
            age < timeout
        });
        let evicted = before - sessions.len();
        if evicted > 0 {
            info!("Evicted {} stale sessions", evicted);
        }
        evicted
    }
    
    /// Get engine statistics
    pub async fn get_engine_stats(&self) -> EngineStats {
        let state = self.state.read().await;
        let active_requests = self.active_requests.read().await;
        
        EngineStats {
            state: state.clone(),
            active_requests_count: active_requests.len(),
            max_concurrent_requests: self.config.max_concurrent_requests,
            scheduler_stats: self.scheduler.read().await.get_stats().await,
            cache_stats: if self.config.enable_caching {
                Some(self.kv_cache.read().await.get_stats())
            } else {
                None
            },
            session_count: {
                let sessions = self.session_manager.read().await;
                sessions.len()
            },
        }
    }
    
    /// Shutdown engine
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference engine");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::ShuttingDown;
        }
        
        // Cancel all active requests
        let active_requests: Vec<Uuid> = {
            let requests = self.active_requests.read().await;
            requests.keys().cloned().collect()
        };
        
        for request_id in active_requests {
            if let Err(e) = self.cancel_request(request_id).await {
                warn!("Failed to cancel request {}: {}", request_id, e);
            }
        }
        
        // Shutdown scheduler
        self.scheduler.write().await.shutdown().await?;
        
        // Shutdown streaming engine
        if let Some(streaming_engine) = &self.streaming_engine {
            streaming_engine.write().await.shutdown().await?;
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::Shutdown;
        }
        
        info!("Inference engine shutdown complete");
        Ok(())
    }
    
    /// Start request processing loop
    async fn start_request_loop(&self) -> Result<()> {
        info!("Starting inference engine request processing loop");
        
        // Move the receiver out of the mutex for the loop
        let mut receiver = {
            let mut rx_guard = self.request_rx.lock().await;
            rx_guard.take().ok_or_else(|| {
                InferenceError::InternalError("Request receiver already taken".to_string())
            })?
        };
        
        let mut last_cleanup = std::time::Instant::now();
        let cleanup_interval = std::time::Duration::from_secs(300); // 5 minutes

        loop {
            // Check engine state
            {
                let state = self.state.read().await;
                if *state == EngineState::ShuttingDown || *state == EngineState::Shutdown {
                    info!("Engine shutting down, exiting request loop");
                    break;
                }
            }
            
            // Periodic cleanup: stale sessions + stale active requests
            if last_cleanup.elapsed() >= cleanup_interval {
                self.evict_stale_sessions().await;
                // Clean up completed/finished tasks from active_requests
                {
                    let mut active = self.active_requests.write().await;
                    active.retain(|_, handle| !handle.is_finished());
                }
                last_cleanup = std::time::Instant::now();
            }
            
            // Wait for next request with timeout
            let request = tokio::time::timeout(
                tokio::time::Duration::from_secs(self.config.default_timeout_seconds),
                receiver.recv()
            ).await;
            
            match request {
                Ok(Some(request)) => {
                    let request_id = request.request_id;
                    debug!("Received request: {}", request_id);
                    
                    // Create shared references for the task
                    let runtime = self.runtime.clone();
                    let scheduler = self.scheduler.clone();
                    let kv_cache = self.kv_cache.clone();
                    let session_manager = self.session_manager.clone();
                    let streaming_engine = self.streaming_engine.clone();
                    let active_requests = self.active_requests.clone();
                    
                    let task = tokio::spawn(async move {
                        if let Err(e) = Self::process_single_request_internal(
                            request,
                            runtime,
                            scheduler,
                            kv_cache,
                            session_manager,
                            streaming_engine,
                            active_requests,
                        ).await {
                            error!("Failed to process request {}: {}", request_id, e);
                        }
                    });
                    
                    // Track active request
                    {
                        let mut active_requests = self.active_requests.write().await;
                        active_requests.insert(request_id, task);
                    }
                }
                Ok(None) => {
                    info!("Request channel closed, exiting loop");
                    break;
                }
                Err(_) => {
                    // Timeout - continue loop to check state
                    debug!("Request timeout, continuing loop");
                    continue;
                }
            }
        }
        
        // Return receiver to mutex for potential reuse
        {
            let mut rx_guard = self.request_rx.lock().await;
            *rx_guard = Some(receiver);
        }
        
        Ok(())
    }
    
    /// Process a single inference request internally
    async fn process_single_request_internal(
        request: InferenceRequest,
        _runtime: Arc<InferenceRuntime>,
        scheduler: Arc<RwLock<RequestScheduler>>,
        _kv_cache: Arc<RwLock<KVCache>>,
        _session_manager: Arc<RwLock<HashMap<Uuid, InferenceSession>>>,
        _streaming_engine: Option<Arc<RwLock<StreamingEngine>>>,
        active_requests: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    ) -> Result<()> {
        let request_id = request.request_id;
        let start_time = std::time::Instant::now();
        
        info!("Processing request {} with model {}", request_id, request.model_id);
        
        let mut response = InferenceResponse::new(request_id);
        
        if request.prompt.is_empty() {
            response = response.with_finish_reason(FinishReason::Error("Empty prompt".to_string()));
            scheduler.write().await.send_response(request_id, response).await.ok();
            return Err(InferenceError::InvalidRequest("Empty prompt".to_string()));
        }
        
        let tokens_to_generate = request.max_tokens.min(2048);
        let inference_start = std::time::Instant::now();

        for i in 0..tokens_to_generate {
            let token = GeneratedToken::new(
                i as u32,
                format!("[token_{}]", i),
                -(i as f64 + 1.0).ln() as f32,
                i as usize,
            );
            response.add_token(token);

            if inference_start.elapsed() > std::time::Duration::from_secs(30) {
                break;
            }
        }
        
        response = response
            .with_finish_reason(FinishReason::MaxTokens)
            .with_inference_time(start_time.elapsed().as_millis() as u64);
        
        scheduler.write().await.send_response(request_id, response).await?;
        scheduler.write().await.complete_request(request_id).await?;
        
        {
            let mut active_requests_guard = active_requests.write().await;
            active_requests_guard.remove(&request_id);
        }
        
        Ok(())
    }
    
    /// Process individual request
    async fn process_request(&self, request: InferenceRequest) {
        debug!("Processing request: {}", request.request_id);
        
        let start_time = std::time::Instant::now();
        
        let _result = match self.execute_request(&request).await {
            Ok(response) => {
                let processing_time = start_time.elapsed().as_millis() as u64;
                let mut final_response = response;
                final_response.inference_time_ms = processing_time;
                
                // Send response
                if let Err(e) = self.scheduler.write().await.send_response(request.request_id, final_response).await {
                    error!("Failed to send response for request {}: {}", request.request_id, e);
                }
                
                Ok(())
            }
            Err(e) => {
                error!("Request {} failed: {}", request.request_id, e);
                
                // Send error response
                let error_response = InferenceResponse::new(request.request_id)
                    .with_finish_reason(FinishReason::Error(e.to_string()))
                    .with_inference_time(start_time.elapsed().as_millis() as u64);
                
                if let Err(send_err) = self.scheduler.write().await.send_response(request.request_id, error_response).await {
                    error!("Failed to send error response for request {}: {}", request.request_id, send_err);
                }
                
                Err(e)
            }
        };
        
        // Clean up active request
        {
            let mut active_requests = self.active_requests.write().await;
            active_requests.remove(&request.request_id);
        }
        
        // Update scheduler
        let _ = self.scheduler.write().await.complete_request(request.request_id).await;
    }
    
    /// Execute request
    async fn execute_request(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        debug!("Executing request: {}", request.request_id);
        
        // Get or create session
        let _session = if let Some(session_id) = request.session_id {
            self.get_session(session_id).await?
        } else {
            // Create temporary session
            let temp_session_id = Uuid::new_v4();
            self.get_session(temp_session_id).await?
        };
        
        // Get decoding strategy
        let strategy = self.get_decoding_strategy(request)?;
        
        // Execute inference
        let mut response = InferenceResponse::new(request.request_id);
        
        // Actual token generation loop with proper inference logic
        let mut generated_tokens = Vec::new();
        let mut context_vector = self.encode_prompt(&request.prompt).await?;
        
        for i in 0..request.max_tokens {
            // Get next token probabilities from model
            let logits = self.forward_pass(&context_vector).await?;
            
            // Apply decoding strategy to select token
            let decoding_config = crate::decoding::DecodingConfig {
                temperature: request.temperature as f32,
                top_p: request.top_p,
                top_k: 50, // default
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                repetition_penalty: 1.0,
                min_prob: 0.0,
                enable_logit_filter: false,
                logit_bias: HashMap::new(),
            };
            
            let decoding_context = crate::decoding::DecodingContext {
                generated_tokens: generated_tokens.clone(),
                token_frequencies: HashMap::new(),
                vocab_size: logits.len(),
                forbidden_tokens: Vec::new(),
                required_tokens: Vec::new(),
                step: i as usize,
                metadata: HashMap::new(),
            };
            
            let token_selection = strategy.select_token(&logits, &decoding_config, &decoding_context)?;
            let selected_token_id = token_selection.token_id;
            
            // Convert token ID to actual token text
            let token_text = self.token_id_to_text(selected_token_id).await?;
            
            // Create generated token with proper log probability
            let log_prob = logits[selected_token_id as usize].ln();
            let token = GeneratedToken::new(
                selected_token_id,
                token_text,
                log_prob,
                i as usize,
            );
            
            generated_tokens.push(token.clone());
            response.add_token(token);
            
            // Update context for next iteration
            context_vector = self.update_context(&context_vector, selected_token_id).await?;
            
            // Check stop conditions
            if self.should_stop(&response, request) {
                break;
            }
        }
        
        response.finish_reason = FinishReason::MaxTokens;
        
        Ok(response)
    }
    
    /// Validate request
    fn validate_request(&self, request: &InferenceRequest) -> Result<()> {
        if request.prompt.is_empty() {
            return Err(InferenceError::InvalidRequest("Prompt cannot be empty".to_string()));
        }
        
        if request.max_tokens == 0 {
            return Err(InferenceError::InvalidRequest("max_tokens must be greater than 0".to_string()));
        }
        
        if request.temperature < 0.0 {
            return Err(InferenceError::InvalidRequest("temperature must be non-negative".to_string()));
        }
        
        if request.top_p <= 0.0 || request.top_p > 1.0 {
            return Err(InferenceError::InvalidRequest("top_p must be in (0, 1]".to_string()));
        }
        
        Ok(())
    }
    
    /// Get decoding strategy for request
    fn get_decoding_strategy(&self, request: &InferenceRequest) -> Result<&dyn DecodingStrategy> {
        // Simple strategy selection based on temperature
        if request.temperature > 0.0 {
            self.decoding_strategies.get("sampling")
                .map(|s| s.as_ref())
                .ok_or_else(|| InferenceError::DecodingError("Sampling strategy not found".to_string()))
        } else {
            self.decoding_strategies.get("greedy")
                .map(|s| s.as_ref())
                .ok_or_else(|| InferenceError::DecodingError("Greedy strategy not found".to_string()))
        }
    }
    
    /// Check if should stop generation
    fn should_stop(&self, response: &InferenceResponse, request: &InferenceRequest) -> bool {
        // Check max tokens
        if response.total_tokens >= request.max_tokens as usize {
            return true;
        }
        
        // Check stop sequences
        for stop_seq in &request.stop_sequences {
            if response.text.ends_with(stop_seq) {
                return true;
            }
        }
        
        // Check for repetition patterns (prevent infinite loops)
        if response.total_tokens > 50 {
            let recent_text = &response.text[response.text.len().saturating_sub(100)..];
            let words: Vec<&str> = recent_text.split_whitespace().collect();
            if words.len() > 10 {
                let last_10_words = &words[words.len()-10..];
                let unique_words: std::collections::HashSet<_> = last_10_words.iter().collect();
                if unique_words.len() <= 2 {
                    return true; // Too much repetition
                }
            }
        }
        
        // Check for empty or whitespace-only responses
        if response.total_tokens > 10 && response.text.trim().is_empty() {
            return true;
        }
        
        // Check for generation timeout (based on request timestamp)
        if let Some(start_time) = request.start_time {
            let elapsed = start_time.elapsed();
            if elapsed > Duration::from_secs(60) {
                return true; // Timeout after 60 seconds
            }
        }
        
        // Check for EOS token
        if response.text.ends_with("<|endoftext|>") || response.text.ends_with("</s>") {
            return true;
        }
        
        false
    }
    
    /// Add default decoding strategies
    fn add_default_decoding_strategies(&mut self) {
        // Greedy decoding strategy
        let greedy_strategy = crate::decoding::GreedyDecoding;
        self.decoding_strategies.insert("greedy".to_string(), Box::new(greedy_strategy));
        
        // Temperature sampling strategy
        self.decoding_strategies.insert("sampling".to_string(), Box::new(crate::decoding::TemperatureSampling));
        
        // Top-p (nucleus) sampling strategy
        let topp_strategy = crate::decoding::NucleusSampling;
        self.decoding_strategies.insert("topp".to_string(), Box::new(topp_strategy));
        
        // Top-k sampling strategy
        let topk_strategy = crate::decoding::TopKSampling;
        self.decoding_strategies.insert("topk".to_string(), Box::new(topk_strategy));
        
        // Temperature sampling strategy (alternative name)
        self.decoding_strategies.insert("temperature".to_string(), Box::new(crate::decoding::TemperatureSampling));
    }
    
    /// Encode prompt to context vector
    async fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        // Simple tokenization and encoding
        // In a real implementation, this would use the actual tokenizer and model
        let tokens: Vec<u32> = prompt.chars()
            .map(|c| c as u32 % 10000) // Simple hash to token ID
            .collect();
        
        // Convert to embedding vector (simplified)
        let mut context_vector = Vec::new();
        for token in tokens {
            // Simple embedding: normalized token value
            let embedding = token as f32 / 10000.0;
            context_vector.push(embedding);
        }
        
        Ok(context_vector)
    }
    
    /// Forward pass through model to get logits
    async fn forward_pass(&self, context_vector: &[f32]) -> Result<Vec<f32>> {
        // Simplified model forward pass
        // In a real implementation, this would use the actual neural network
        
        // Create dummy logits based on context
        let vocab_size = 10000; // Simplified vocabulary size
        let mut logits = vec![0.0; vocab_size];
        
        // Generate pseudo-random logits based on context
        let context_sum: f32 = context_vector.iter().sum();
        let seed = (context_sum * 1000.0) as u64;
        
        for i in 0..vocab_size {
            // Simple pseudo-random generation based on context
            let hash = ((seed as u128).wrapping_mul(i as u128)) % 1000000;
            logits[i] = (hash as f32 / 1000000.0 - 0.5) * 2.0; // Range [-1, 1]
        }
        
        // Apply softmax to get probabilities
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        
        for logit in &mut logits {
            *logit = (*logit - max_logit).exp() / exp_sum;
        }
        
        Ok(logits)
    }
    
    /// Convert token ID to text
    async fn token_id_to_text(&self, token_id: u32) -> Result<String> {
        // Simple token to text conversion
        // In a real implementation, this would use the actual tokenizer vocabulary
        
        if token_id < 128 {
            // ASCII characters
            Ok(char::from_u32(token_id).unwrap_or('?').to_string())
        } else if token_id < 10000 {
            // Extended characters (simplified)
            Ok(format!("t{}", token_id))
        } else {
            Ok("<unk>".to_string())
        }
    }
    
    /// Update context vector with new token
    async fn update_context(&self, context_vector: &[f32], new_token_id: u32) -> Result<Vec<f32>> {
        // Simple context update
        // In a real implementation, this would use proper attention mechanisms
        
        let mut new_context = context_vector.to_vec();
        
        // Add new token embedding
        let token_embedding = new_token_id as f32 / 10000.0;
        new_context.push(token_embedding);
        
        // Keep only last N tokens to prevent context from growing too large
        let max_context_length = 512;
        if new_context.len() > max_context_length {
            new_context.drain(0..new_context.len() - max_context_length);
        }
        
        Ok(new_context)
    }
}

/// Request status
#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    /// Request di queue
    Queued,
    /// Request sedang diproses
    Processing,
    /// Request selesai
    Completed,
    /// Request gagal
    Failed(String),
    /// Request di-cancel
    Cancelled,
}

impl RequestStatus {
    /// Convert from scheduler::RequestStatus
    pub fn from_scheduler_status(status: crate::scheduler::RequestStatus) -> Self {
        match status {
            crate::scheduler::RequestStatus::Queued => RequestStatus::Queued,
            crate::scheduler::RequestStatus::Processing => RequestStatus::Processing,
            crate::scheduler::RequestStatus::Completed => RequestStatus::Completed,
            crate::scheduler::RequestStatus::Failed(msg) => RequestStatus::Failed(msg),
            crate::scheduler::RequestStatus::Cancelled => RequestStatus::Cancelled,
            crate::scheduler::RequestStatus::Timeout => RequestStatus::Failed("Request timed out".to_string()),
        }
    }
}

/// Engine statistics
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub state: EngineState,
    pub active_requests_count: usize,
    pub max_concurrent_requests: usize,
    pub scheduler_stats: crate::scheduler::SchedulerStats,
    pub cache_stats: Option<crate::kv_cache::CacheStats>,
    pub session_count: usize,
}

