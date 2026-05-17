//! Streaming
//! 
//! Real-time token output untuk inference.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use tracing::{debug, info, warn, error};
use chrono::{DateTime, Utc};

use crate::{Result, InferenceError, InferenceRequest, GeneratedToken};

/// Configuration untuk streaming
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for token stream
    pub buffer_size: usize,
    /// Maximum tokens per second (rate limiting)
    pub max_tokens_per_second: Option<f32>,
    /// Enable token buffering
    pub enable_buffering: bool,
    /// Buffer flush interval (ms)
    pub buffer_flush_interval_ms: u64,
    /// Enable compression for large streams
    pub enable_compression: bool,
    /// Compression threshold (tokens)
    pub compression_threshold: usize,
    /// Stream timeout (seconds)
    pub stream_timeout_seconds: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            max_tokens_per_second: None,
            enable_buffering: true,
            buffer_flush_interval_ms: 100,
            enable_compression: false,
            compression_threshold: 1000,
            stream_timeout_seconds: 300, // 5 minutes
            enable_metrics: true,
        }
    }
}

/// Token stream
pub struct TokenStream {
    /// Stream ID
    pub stream_id: Uuid,
    /// Request ID
    pub request_id: Uuid,
    /// Token receiver
    pub token_rx: mpsc::UnboundedReceiver<StreamToken>,
    /// Stream metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Stream token with metadata
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// Token data
    pub token: GeneratedToken,
    /// Is this the last token?
    pub is_last: bool,
    /// Stream position
    pub position: usize,
    /// Stream metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Streaming engine
pub struct StreamingEngine {
    /// Configuration
    config: StreamingConfig,
    /// Active streams
    active_streams: Arc<RwLock<HashMap<Uuid, StreamInfo>>>,
    /// Stream statistics
    stats: Arc<RwLock<StreamingStats>>,
    /// Engine state
    state: Arc<RwLock<EngineState>>,
}

/// Stream information
#[derive(Debug)]
struct StreamInfo {
    /// Stream ID
    _stream_id: Uuid,
    /// Request ID
    request_id: Uuid,
    /// Token sender
    token_tx: mpsc::UnboundedSender<StreamToken>,
    /// Stream configuration
    config: StreamingConfig,
    /// Stream statistics
    stats: StreamStats,
    /// Created timestamp
    created_at: DateTime<Utc>,
    /// Last activity timestamp
    last_activity: DateTime<Utc>,
    /// Token count
    token_count: usize,
    /// Is stream finished?
    finished: bool,
}

/// Stream statistics
#[derive(Debug, Clone, Default)]
struct StreamStats {
    /// Tokens sent
    pub tokens_sent: usize,
    /// Bytes sent
    pub bytes_sent: usize,
    /// Average tokens per second
    pub _avg_tokens_per_second: f64,
    /// Stream duration (ms)
    pub _duration_ms: u64,
    /// Buffer flushes
    pub _buffer_flushes: u64,
    /// Errors encountered
    pub _errors: u64,
}

/// Streaming statistics
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total streams created
    pub total_streams: u64,
    /// Active streams
    pub active_streams: usize,
    /// Total tokens streamed
    pub total_tokens_streamed: u64,
    /// Average stream duration (ms)
    pub avg_stream_duration_ms: f64,
    /// Peak concurrent streams
    pub peak_concurrent_streams: usize,
    /// Total errors
    pub total_errors: u64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Engine state
#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    /// Engine tidak diinisialisasi
    Uninitialized,
    /// Engine sedang diinisialisasi
    Initializing,
    /// Engine siap
    Ready,
    /// Engine sedang shutdown
    ShuttingDown,
    /// Engine sudah shutdown
    Shutdown,
}

impl StreamingEngine {
    /// Create new streaming engine
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }
    
    /// Create streaming engine with configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            config,
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StreamingStats::default())),
            state: Arc::new(RwLock::new(EngineState::Uninitialized)),
        }
    }
    
    /// Initialize streaming engine
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing streaming engine");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::Initializing;
        }
        
        // Start cleanup loop for expired streams
        self.start_cleanup_loop().await?;
        
        // Update state to ready
        {
            let mut state = self.state.write().await;
            *state = EngineState::Ready;
        }
        
        info!("Streaming engine initialized successfully");
        Ok(())
    }
    
    /// Create new stream for request
    pub async fn create_stream(&self, request: &InferenceRequest) -> Result<TokenStream> {
        debug!("Creating stream for request: {:?}", request.request_id);
        
        // Check engine state
        {
            let state = self.state.read().await;
            if *state != EngineState::Ready {
                return Err(InferenceError::InternalError("Streaming engine not ready".to_string()).into());
            }
        }
        
        let stream_id = Uuid::new_v4();
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        
        let stream_info = StreamInfo {
            _stream_id: stream_id,
            request_id: request.request_id.as_ref().and_then(|s| Uuid::parse_str(s).ok()).unwrap_or_else(Uuid::new_v4),
            token_tx,
            config: self.config.clone(),
            stats: StreamStats::default(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            token_count: 0,
            finished: false,
        };
        
        // Add to active streams
        {
            let mut streams = self.active_streams.write().await;
            streams.insert(stream_id, stream_info);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_streams += 1;
            stats.active_streams = {
                let streams = self.active_streams.read().await;
                streams.len()
            };
            stats.last_updated = Utc::now();
        }
        
        let token_stream = TokenStream {
            stream_id,
            request_id: request.request_id.as_ref().and_then(|s| Uuid::parse_str(s).ok()).unwrap_or_else(Uuid::new_v4),
            token_rx,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        };
        
        debug!("Stream {} created for request {:?}", stream_id, request.request_id);
        Ok(token_stream)
    }
    
    /// Submit streaming request
    pub async fn submit_request(&self, request: InferenceRequest) -> Result<mpsc::UnboundedReceiver<GeneratedToken>> {
        debug!("Submitting streaming request: {:?}", request.request_id);
        
        let is_streaming = request.parameters.get("streaming")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        if !is_streaming {
            return Err(InferenceError::InternalError("Request is not for streaming".to_string()).into());
        }
        
        let stream = self.create_stream(&request).await?;
        
        // Convert stream to simple token receiver
        let (simple_tx, simple_rx) = mpsc::unbounded_channel();
        
        // Start stream processing
        let engine = self.clone();
        tokio::spawn(async move {
            engine.process_stream(stream, simple_tx).await;
        });
        
        Ok(simple_rx)
    }
    
    /// Send token to stream
    pub async fn send_token(&self, stream_id: Uuid, token: GeneratedToken, is_last: bool) -> Result<()> {
        debug!("Sending token to stream {}: {}", stream_id, token.token_id);
        
        let mut streams = self.active_streams.write().await;
        if let Some(stream_info) = streams.get_mut(&stream_id) {
            if stream_info.finished {
                return Err(InferenceError::InternalError("Stream is already finished".to_string()).into());
            }
            
            // Apply rate limiting if configured
            if let Some(max_rate) = stream_info.config.max_tokens_per_second {
                if !self.check_rate_limit(stream_info, max_rate).await? {
                    return Err(InferenceError::InternalError("Rate limit exceeded".to_string()).into());
                }
            }
            
            let stream_token = StreamToken {
                token: token.clone(),
                is_last,
                position: stream_info.token_count,
                metadata: HashMap::new(),
            };
            
            // Send token
            if let Err(_) = stream_info.token_tx.send(stream_token) {
                // Stream receiver dropped, clean up
                streams.remove(&stream_id);
                return Err(InferenceError::InternalError("Stream receiver dropped".to_string()).into());
            }
            
            // Update stream info
            stream_info.token_count += 1;
            stream_info.last_activity = Utc::now();
            stream_info.stats.tokens_sent += 1;
            stream_info.stats.bytes_sent += token.text.len();
            
            if is_last {
                stream_info.finished = true;
            }
            
            // Update global statistics
            {
                let mut stats = self.stats.write().await;
                stats.total_tokens_streamed += 1;
            }
            
            debug!("Token sent successfully to stream {}", stream_id);
            Ok(())
        } else {
            Err(InferenceError::InternalError(format!("Stream {} not found", stream_id)).into())
        }
    }
    
    /// Get stream status
    pub async fn get_stream_status(&self, stream_id: Uuid) -> Result<Option<StreamStatus>> {
        let streams = self.active_streams.read().await;
        
        if let Some(stream_info) = streams.get(&stream_id) {
            let duration = (Utc::now() - stream_info.created_at).num_milliseconds() as u64;
            
            Ok(Some(StreamStatus {
                stream_id,
                request_id: stream_info.request_id,
                is_active: !stream_info.finished,
                token_count: stream_info.token_count,
                duration_ms: duration,
                tokens_per_second: if duration > 0 {
                    (stream_info.token_count as f64 * 1000.0) / duration as f64
                } else {
                    0.0
                },
                created_at: stream_info.created_at,
                last_activity: stream_info.last_activity,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Cancel stream
    pub async fn cancel_stream(&self, stream_id: Uuid) -> Result<bool> {
        debug!("Cancelling stream: {}", stream_id);
        
        let mut streams = self.active_streams.write().await;
        
        if let Some(stream_info) = streams.remove(&stream_id) {
            // Send completion signal
            let _ = stream_info.token_tx.send(StreamToken {
                token: GeneratedToken { token_id: 0, text: "[CANCELLED]".to_string(), logprob: 0.0, is_special: true },
                is_last: true,
                position: stream_info.token_count,
                metadata: HashMap::new(),
            });
            
            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.active_streams = streams.len();
                stats.total_errors += 1;
            }
            
            debug!("Stream {} cancelled successfully", stream_id);
            Ok(true)
        } else {
            debug!("Stream {} not found for cancellation", stream_id);
            Ok(false)
        }
    }
    
    /// Get streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update active streams count
        stats.active_streams = {
            let streams = self.active_streams.read().await;
            streams.len()
        };
        
        // Calculate average stream duration
        let streams = self.active_streams.read().await;
        if stats.total_streams > 0 {
            let total_duration: u64 = streams.values()
                .map(|s| (Utc::now() - s.created_at).num_milliseconds() as u64)
                .sum();
            stats.avg_stream_duration_ms = total_duration as f64 / stats.total_streams as f64;
        }
        
        stats
    }
    
    /// Shutdown streaming engine
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down streaming engine");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::ShuttingDown;
        }
        
        // Cancel all active streams
        let stream_ids: Vec<Uuid> = {
            let streams = self.active_streams.read().await;
            streams.keys().cloned().collect()
        };
        
        for stream_id in stream_ids {
            if let Err(e) = self.cancel_stream(stream_id).await {
                warn!("Failed to cancel stream {}: {}", stream_id, e);
            }
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::Shutdown;
        }
        
        info!("Streaming engine shutdown complete");
        Ok(())
    }
    
    /// Process stream and convert to simple token receiver
    async fn process_stream(&self, mut stream: TokenStream, simple_tx: mpsc::UnboundedSender<GeneratedToken>) {
        debug!("Processing stream: {}", stream.stream_id);
        
        while let Some(stream_token) = stream.token_rx.recv().await {
            if let Err(_) = simple_tx.send(stream_token.token.clone()) {
                debug!("Simple receiver dropped for stream {}", stream.stream_id);
                break;
            }
            
            if stream_token.is_last {
                debug!("Stream {} finished", stream.stream_id);
                break;
            }
        }
    }
    
    /// Start cleanup loop for expired streams
    async fn start_cleanup_loop(&self) -> Result<()> {
        let engine = self.clone();
        tokio::spawn(async move {
            engine.run_cleanup_loop().await;
        });
        Ok(())
    }
    
    /// Run cleanup loop
    async fn run_cleanup_loop(&self) {
        debug!("Starting stream cleanup loop");
        
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60)); // Check every minute
        
        loop {
            interval.tick().await;
            
            // Check engine state
            {
                let state = self.state.read().await;
                match *state {
                    EngineState::Ready => {},
                    EngineState::ShuttingDown | EngineState::Shutdown => break,
                    _ => continue,
                }
            }
            
            if let Err(e) = self.cleanup_expired_streams().await {
                error!("Error cleaning up expired streams: {}", e);
            }
        }
        
        debug!("Stream cleanup loop ended");
    }
    
    /// Clean up expired streams
    async fn cleanup_expired_streams(&self) -> Result<()> {
        let now = Utc::now();
        let mut expired_streams = Vec::new();
        
        {
            let streams = self.active_streams.read().await;
            for (stream_id, stream_info) in streams.iter() {
                let age_seconds = (now - stream_info.created_at).num_seconds() as u64;
                let idle_seconds = (now - stream_info.last_activity).num_seconds() as u64;
                
                if age_seconds > self.config.stream_timeout_seconds || 
                   (idle_seconds > 300 && stream_info.finished) { // 5 minutes idle for finished streams
                    expired_streams.push(*stream_id);
                }
            }
        }
        
        for stream_id in expired_streams {
            if let Err(e) = self.cancel_stream(stream_id).await {
                warn!("Failed to cleanup expired stream {}: {}", stream_id, e);
            }
        }
        
        Ok(())
    }
    
    /// Check rate limit for stream
    async fn check_rate_limit(&self, stream_info: &StreamInfo, max_rate: f32) -> Result<bool> {
        let now = Utc::now();
        let _time_window = 1.0; // 1 second window
        
        // Simple rate limiting check
        let tokens_in_window = stream_info.token_count;
        let elapsed_seconds = (now - stream_info.created_at).num_seconds() as f64;
        
        if elapsed_seconds > 0.0 {
            let current_rate = tokens_in_window as f64 / elapsed_seconds as f64;
            Ok(current_rate <= max_rate as f64)
        } else {
            Ok(true) // Allow first token
        }
    }
}

/// Stream status information
#[derive(Debug, Clone)]
pub struct StreamStatus {
    pub stream_id: Uuid,
    pub request_id: Uuid,
    pub is_active: bool,
    pub token_count: usize,
    pub duration_ms: u64,
    pub tokens_per_second: f64,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

impl Clone for StreamingEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_streams: Arc::clone(&self.active_streams),
            stats: Arc::clone(&self.stats),
            state: Arc::clone(&self.state),
        }
    }
}

impl Default for StreamingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for stream processing
pub mod utils {
    use super::*;
    
    /// Convert token stream to text stream
    pub async fn tokens_to_text(mut token_stream: TokenStream) -> Result<mpsc::UnboundedReceiver<String>> {
        let (text_tx, text_rx) = mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            let mut accumulated_text = String::new();
            
            while let Some(stream_token) = token_stream.token_rx.recv().await {
                accumulated_text.push_str(&stream_token.token.text);
                
                if let Err(_) = text_tx.send(accumulated_text.clone()) {
                    break;
                }
                
                if stream_token.is_last {
                    break;
                }
            }
        });
        
        Ok(text_rx)
    }
    
    /// Buffer tokens for batch sending
    pub async fn buffer_tokens(
        mut token_stream: TokenStream,
        buffer_size: usize,
        flush_interval_ms: u64,
    ) -> Result<mpsc::UnboundedReceiver<Vec<GeneratedToken>>> {
        let (buffered_tx, buffered_rx) = mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            let mut buffer = Vec::new();
            let mut last_flush = std::time::Instant::now();
            
            while let Some(stream_token) = token_stream.token_rx.recv().await {
                buffer.push(stream_token.token);
                
                let should_flush = buffer.len() >= buffer_size || 
                                 last_flush.elapsed().as_millis() as u128 >= flush_interval_ms as u128 ||
                                 stream_token.is_last;
                
                if should_flush && !buffer.is_empty() {
                    if let Err(_) = buffered_tx.send(buffer.clone()) {
                        break;
                    }
                    buffer.clear();
                    last_flush = std::time::Instant::now();
                }
                
                if stream_token.is_last {
                    break;
                }
            }
        });
        
        Ok(buffered_rx)
    }
}

/// Stream adapter for different output formats
pub mod adapters {
    use super::*;
    
    /// JSON stream adapter
    pub struct JsonStreamAdapter {
        inner: TokenStream,
    }
    
    impl JsonStreamAdapter {
        pub fn new(stream: TokenStream) -> Self {
            Self { inner: stream }
        }
        
        pub async fn receive_json(&mut self) -> Result<mpsc::UnboundedReceiver<serde_json::Value>> {
            let (json_tx, json_rx) = mpsc::unbounded_channel();
            
            while let Some(stream_token) = self.inner.token_rx.recv().await {
                let json_value = serde_json::json!({
                    "token": stream_token.token.text,
                    "token_id": stream_token.token.token_id,
                    "log_prob": stream_token.token.logprob,
                    "position": stream_token.position,
                    "is_last": stream_token.is_last
                });
                
                if let Err(_) = json_tx.send(json_value) {
                    break;
                }
                
                if stream_token.is_last {
                    break;
                }
            }
            
            Ok(json_rx)
        }
    }
    
    /// SSE (Server-Sent Events) stream adapter
    pub struct SseStreamAdapter {
        inner: TokenStream,
    }
    
    impl SseStreamAdapter {
        pub fn new(stream: TokenStream) -> Self {
            Self { inner: stream }
        }
        
        pub async fn receive_sse(&mut self) -> Result<mpsc::UnboundedReceiver<String>> {
            let (sse_tx, sse_rx) = mpsc::unbounded_channel();
            
            while let Some(stream_token) = self.inner.token_rx.recv().await {
                let sse_data = format!(
                    "data: {}\n\n",
                    serde_json::json!({
                        "token": stream_token.token.text,
                        "token_id": stream_token.token.token_id,
                        "log_prob": stream_token.token.logprob,
                        "position": stream_token.position,
                        "is_last": stream_token.is_last
                    })
                );
                
                if let Err(_) = sse_tx.send(sse_data) {
                    break;
                }
                
                if stream_token.is_last {
                    // Send final event
                    let _ = sse_tx.send("event: end\ndata: {}\n\n".to_string());
                    break;
                }
            }
            
            Ok(sse_rx)
        }
    }
}
