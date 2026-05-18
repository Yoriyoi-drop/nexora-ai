use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, RwLock};
use uuid::Uuid;
use tracing::{debug, error, info, warn};

use crate::kv_cache::KVCache;
use crate::runtime::InferenceRuntime;
use crate::scheduler::RequestScheduler;
use crate::session::InferenceSession;
use crate::streaming::StreamingEngine;
use crate::{
    FinishReason, GeneratedToken, InferenceError, InferenceRequest, InferenceResponse, Result,
};
use nexora_foundation::models::transformer::{CausalLM, TransformerConfig};
use nexora_tokenizer::BpeTokenizer;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_concurrent_requests: usize,
    pub default_model_id: String,
    pub enable_queuing: bool,
    pub queue_size_limit: usize,
    pub enable_caching: bool,
    pub cache_size_limit_mb: usize,
    pub enable_streaming: bool,
    pub default_timeout_seconds: u64,
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
            cache_size_limit_mb: 1024,
            enable_streaming: true,
            default_timeout_seconds: 30,
            metrics_interval_seconds: 60,
        }
    }
}

pub struct InferenceEngine {
    config: InferenceConfig,
    runtime: Arc<InferenceRuntime>,
    scheduler: Arc<RwLock<RequestScheduler>>,
    kv_cache: Arc<RwLock<KVCache>>,
    session_manager: Arc<RwLock<HashMap<Uuid, InferenceSession>>>,
    model: CausalLM,
    tokenizer: Option<Arc<parking_lot::RwLock<BpeTokenizer>>>,
    streaming_engine: Option<Arc<RwLock<StreamingEngine>>>,
    request_tx: mpsc::Sender<InferenceRequest>,
    request_rx: Arc<Mutex<Option<mpsc::Receiver<InferenceRequest>>>>,
    active_requests: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    state: Arc<RwLock<EngineState>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    Uninitialized,
    Initializing,
    Ready,
    ShuttingDown,
    Shutdown,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.queue_size_limit.max(1));
        let model_config = TransformerConfig::default();
        let model = CausalLM::new(model_config);
        info!(
            "CausalLM initialized with {} parameters",
            model.parameter_count()
        );

        Self {
            runtime: Arc::new(InferenceRuntime::new()),
            scheduler: Arc::new(RwLock::new(RequestScheduler::new())),
            kv_cache: Arc::new(RwLock::new(KVCache::new())),
            session_manager: Arc::new(RwLock::new(HashMap::new())),
            model,
            tokenizer: None,
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
        }
    }

    pub fn with_model(
        model: CausalLM,
        tokenizer: Option<Arc<parking_lot::RwLock<BpeTokenizer>>>,
        config: InferenceConfig,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.queue_size_limit.max(1));
        info!(
            "Initializing inference engine with loaded model ({} params)",
            model.parameter_count()
        );

        Self {
            runtime: Arc::new(InferenceRuntime::new()),
            scheduler: Arc::new(RwLock::new(RequestScheduler::new())),
            kv_cache: Arc::new(RwLock::new(KVCache::new())),
            session_manager: Arc::new(RwLock::new(HashMap::new())),
            model,
            tokenizer,
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
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        let state = self.state.read().await.clone();
        if state != EngineState::Uninitialized {
            return Err(InferenceError::EngineNotInitialized(
                "Engine already initialized".to_string(),
            ));
        }
        info!("Initializing inference engine");
        *self.state.write().await = EngineState::Initializing;

        self.runtime.initialize().await?;
        self.scheduler.write().await.initialize().await?;

        if self.config.enable_caching {
            self.kv_cache.write().await.initialize().await?;
        }
        if let Some(se) = &self.streaming_engine {
            se.write().await.initialize().await?;
        }

        self.start_request_loop().await?;
        *self.state.write().await = EngineState::Ready;
        info!("Inference engine initialized successfully");
        Ok(())
    }

    /// Submit a standard (non-streaming) inference request.
    /// Returns a Receiver that will receive exactly one response (buffer=1).
    pub async fn submit_request(
        &self,
        request: InferenceRequest,
    ) -> Result<mpsc::Receiver<InferenceResponse>> {
        let state = self.state.read().await.clone();
        match state {
            EngineState::Ready => {}
            EngineState::ShuttingDown | EngineState::Shutdown => {
                return Err(InferenceError::EngineNotInitialized(
                    "Engine is shutting down".to_string(),
                ));
            }
            _ => {
                return Err(InferenceError::EngineNotInitialized(
                    "Engine not ready".to_string(),
                ));
            }
        }

        self.validate_request(&request)?;

        {
            let active = self.active_requests.read().await;
            if active.len() >= self.config.max_concurrent_requests {
                return Err(InferenceError::ResourceExhausted(format!(
                    "Max concurrent requests ({})",
                    self.config.max_concurrent_requests
                )));
            }
        }

        let (response_tx, response_rx) = mpsc::channel(1);
        self.scheduler
            .write()
            .await
            .submit_request(request.request_id, response_tx)
            .await?;

        self.request_tx
            .try_send(request)
            .map_err(|_| InferenceError::ResourceExhausted("Queue full".to_string()))?;

        Ok(response_rx)
    }

    /// Submit a streaming request.
    /// Returns a bounded Receiver that yields GeneratedToken values (buffer=64, prevents runaway memory).
    pub async fn submit_streaming_request(
        &self,
        request: InferenceRequest,
    ) -> Result<mpsc::Receiver<GeneratedToken>> {
        if !self.config.enable_streaming {
            return Err(InferenceError::InvalidRequest(
                "Streaming disabled".to_string(),
            ));
        }
        let se = self
            .streaming_engine
            .as_ref()
            .ok_or_else(|| InferenceError::InternalError("No streaming engine".to_string()))?;

        let (stream_id, mut rx) = se.write().await.create_stream().await?;
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let _cfg = self.config.clone();
        let _scheduler = self.scheduler.clone();
        let se_clone = se.clone();
        let active = self.active_requests.clone();

        let request_id = request.request_id;
        let max_tokens = request.max_tokens;
        let temperature = request.temperature;
        let top_p = request.top_p;
        let top_k = request.top_k as usize;

        let task = tokio::spawn(async move {
            let prompt_ids: Vec<u32> = match &tokenizer {
                    Some(tok) => {
                        let t = tok.read();
                        t.encode(&request.prompt)
                    },
                    None => request.prompt.bytes().map(|b| b as u32).collect(),
            };

            let mut kv_state = model.reset_cache();
            let mut all_ids = prompt_ids.clone();
            let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
                temperature,
                top_k,
                top_p,
                ..Default::default()
            });

            let max_gen = max_tokens.min(2048) as usize;

            for pos in 0..max_gen {
                let input: &[u32] = if pos == 0 {
                    &prompt_ids
                } else {
                    core::slice::from_ref(all_ids.last().unwrap_or(&0))
                };

                let logits = model.forward(input, &mut kv_state);
                let logits_slice = logits.as_slice().unwrap_or(&[]);

                let token_id = match sampler.sample(logits_slice) {
                    Ok(idx) => idx as u32,
                    Err(e) => {
                        warn!("Sampler failed, error: {:?}, falling back to argmax", e);
                        logits
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                            .map(|(i, _)| i as u32)
                            .unwrap_or(0)
                    }
                };

                let token_text = match &tokenizer {
                    Some(tok) => {
                        let t = tok.read();
                        t.decode(&[token_id])
                    },
                    None => token_id_to_text_fallback(token_id),
                };

                let log_prob = logits
                    .get(token_id as usize)
                    .copied()
                    .unwrap_or(0.0)
                    .ln();
                let token = GeneratedToken::new(token_id, token_text, log_prob, pos);

                let is_last = pos == max_gen - 1 || token_id == 0;
                if let Err(e) = se_clone
                    .write()
                    .await
                    .push_tokens(stream_id, vec![token], is_last)
                    .await
                {
                    warn!("Stream push failed: {}", e);
                    break;
                }

                all_ids.push(token_id);
                if is_last {
                    break;
                }
            }

            {
                let mut a = active.write().await;
                a.remove(&request_id);
            }
        });

        {
            let mut a = self.active_requests.write().await;
            a.insert(request_id, task);
        }

        Ok(rx)
    }

    #[tracing::instrument(skip_all, fields(request_id = %request.request_id, prompt_len = request.prompt.len()))]
    pub async fn generate_internal(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();
        let mut response = InferenceResponse::new(request.request_id);

        if request.prompt.is_empty() {
            return Ok(response
                .with_finish_reason(FinishReason::Error("Empty prompt".to_string()))
                .with_inference_time(start.elapsed().as_millis() as u64));
        }

        let prompt_ids = self.encode_prompt(&request.prompt);
        let mut kv_state = self.model.reset_cache();
        let mut all_ids = prompt_ids.clone();
        let max_gen = request.max_tokens.min(2048) as usize;

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
            temperature: request.temperature,
            top_k: request.top_k as usize,
            top_p: request.top_p,
            ..Default::default()
        });

        for pos in 0..max_gen {
            let input: &[u32] = if pos == 0 {
                &prompt_ids
            } else {
                core::slice::from_ref(all_ids.last().unwrap_or(&0))
            };

            let logits = self.model.forward(input, &mut kv_state);
            let logits_slice = logits.as_slice().unwrap_or(&[]);

            let token_id = match sampler.sample(logits_slice) {
                Ok(idx) => idx as u32,
                Err(e) => {
                    warn!("Sampler failed in greedy path: {:?}, falling back to argmax", e);
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                },
            };

            let token_text = self.token_id_to_text(token_id);
            let log_prob = logits
                .get(token_id as usize)
                .copied()
                .unwrap_or(0.0)
                .ln();
            let token = GeneratedToken::new(token_id, token_text, log_prob, pos);
            response.add_token(token);
            all_ids.push(token_id);

            if token_id == 0 || start.elapsed() > Duration::from_secs(60) {
                break;
            }
        }

        if start.elapsed() > Duration::from_secs(60) {
            response.finish_reason = FinishReason::Timeout;
        } else {
            response.finish_reason = FinishReason::MaxTokens;
        }
        response.inference_time_ms = start.elapsed().as_millis() as u64;
        Ok(response)
    }

    pub async fn cancel_request(&self, request_id: Uuid) -> Result<bool> {
        let sched = self
            .scheduler
            .write()
            .await
            .cancel_request(request_id)
            .await?;
        let task = {
            let mut active = self.active_requests.write().await;
            active
                .remove(&request_id)
                .map(|h| {
                    h.abort();
                    true
                })
                .unwrap_or(false)
        };
        let stream = if let Some(se) = &self.streaming_engine {
            se.write().await.cancel_stream(request_id).await.unwrap_or_else(|e| {
                warn!("Failed to cancel stream {}: {}", request_id, e);
                false
            })
        } else {
            false
        };
        Ok(sched || task || stream)
    }

    pub async fn get_request_status(
        &self,
        request_id: Uuid,
    ) -> Result<RequestStatus> {
        let status = self
            .scheduler
            .read()
            .await
            .get_request_status(request_id)
            .await?;
        Ok(status
            .map(RequestStatus::from_scheduler_status)
            .unwrap_or(RequestStatus::Queued))
    }

    pub async fn get_session(
        &self,
        session_id: Uuid,
    ) -> Result<Arc<InferenceSession>> {
        let mut sessions = self.session_manager.write().await;
        if sessions.len() >= self.config.max_concurrent_requests * 2 {
            let now = chrono::Utc::now();
            sessions.retain(|_, s| {
                let age = (now - s.created_at()).num_seconds() as u64;
                age < s.config().timeout_seconds
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

    pub async fn evict_stale_sessions(&self) -> usize {
        let mut sessions = self.session_manager.write().await;
        let before = sessions.len();
        let now = chrono::Utc::now();
        let timeout = chrono::Duration::seconds(InferenceSession::default_timeout_seconds() as i64);
        sessions.retain(|_, s| (now - s.created_at()) < timeout);
        let evicted = before - sessions.len();
        if evicted > 0 {
            info!("Evicted {} stale sessions", evicted);
        }
        evicted
    }

    pub async fn get_engine_stats(&self) -> EngineStats {
        let state = self.state.read().await.clone();
        let active = self.active_requests.read().await.len();
        let sched_stats = self.scheduler.read().await.get_stats().await;
        let cache_stats = if self.config.enable_caching {
            Some(self.kv_cache.read().await.get_stats())
        } else {
            None
        };
        let session_count = self.session_manager.read().await.len();
        EngineStats {
            state,
            active_requests_count: active,
            max_concurrent_requests: self.config.max_concurrent_requests,
            scheduler_stats: sched_stats,
            cache_stats,
            session_count,
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference engine");
        *self.state.write().await = EngineState::ShuttingDown;

        let ids: Vec<Uuid> = self.active_requests.read().await.keys().copied().collect();
        for id in ids {
            if let Err(e) = self.cancel_request(id).await {
                warn!("Failed to cancel request {} during shutdown: {}", id, e);
            }
        }
        self.scheduler.write().await.shutdown().await?;
        if let Some(se) = &self.streaming_engine {
            se.write().await.shutdown().await?;
        }
        *self.state.write().await = EngineState::Shutdown;
        info!("Inference engine shutdown complete");
        Ok(())
    }

    // --- private ---

    async fn start_request_loop(&self) -> Result<()> {
        info!("Starting request processing loop");
        let mut rx = self
            .request_rx
            .lock()
            .await
            .take()
            .ok_or_else(|| InferenceError::InternalError("Receiver already taken".to_string()))?;

        let mut last_cleanup = std::time::Instant::now();

        loop {
            if matches!(
                *self.state.read().await,
                EngineState::ShuttingDown | EngineState::Shutdown
            ) {
                break;
            }

            if last_cleanup.elapsed() >= Duration::from_secs(300) {
                self.evict_stale_sessions().await;
                {
                    let mut active = self.active_requests.write().await;
                    active.retain(|_, h| !h.is_finished());
                }
                last_cleanup = std::time::Instant::now();
            }

            // Try to pop a batch first
            if let Some(batch) = self.scheduler.read().await.pop_batch().await {
                let engine = InferenceEngineHandle {
                    scheduler: self.scheduler.clone(),
                    model: self.model.clone(),
                    tokenizer: self.tokenizer.clone(),
                    state: self.state.clone(),
                };
                let task = tokio::spawn(async move {
                    engine.process_batch(batch).await;
                });
                // Track under a synthetic batch ID
                let bid = Uuid::new_v4();
                self.active_requests.write().await.insert(bid, task);
                continue;
            }

            // No batch ready, wait for next request
            let request = tokio::time::timeout(
                Duration::from_millis(50),
                rx.recv(),
            )
            .await;

            match request {
                Ok(Some(req)) => {
                    let _rid = req.request_id;
                    // Add to batch collector
                    self.scheduler.write().await.add_to_batch_collector(&req).await;

                    // Try to form a batch with the newly arrived request
                    if let Some(batch) = self.scheduler.read().await.pop_batch().await {
                        let engine = InferenceEngineHandle {
                            scheduler: self.scheduler.clone(),
                            model: self.model.clone(),
                            tokenizer: self.tokenizer.clone(),
                            state: self.state.clone(),
                        };
                        let task = tokio::spawn(async move {
                            engine.process_batch(batch).await;
                        });
                        let bid = Uuid::new_v4();
                        self.active_requests.write().await.insert(bid, task);
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    warn!("Batch scheduler error: {:?}, flushing timed-out batches", e);
                    if let Some(batch) = self.scheduler.read().await.pop_batch().await {
                        let engine = InferenceEngineHandle {
                            scheduler: self.scheduler.clone(),
                            model: self.model.clone(),
                            tokenizer: self.tokenizer.clone(),
                            state: self.state.clone(),
                        };
                        let task = tokio::spawn(async move {
                            engine.process_batch(batch).await;
                        });
                        let bid = Uuid::new_v4();
                        self.active_requests.write().await.insert(bid, task);
                    }
                    continue;
                }
            }
        }
        Ok(())
    }

    fn validate_request(&self, request: &InferenceRequest) -> Result<()> {
        if request.prompt.is_empty() {
            return Err(InferenceError::InvalidRequest(
                "Prompt cannot be empty".to_string(),
            ));
        }
        if request.max_tokens == 0 {
            return Err(InferenceError::InvalidRequest(
                "max_tokens must be > 0".to_string(),
            ));
        }
        if request.temperature < 0.0 {
            return Err(InferenceError::InvalidRequest(
                "temperature must be >= 0".to_string(),
            ));
        }
        Ok(())
    }

    fn encode_prompt(&self, prompt: &str) -> Vec<u32> {
        match &self.tokenizer {
            Some(tok) => {
                let t = tok.read();
                t.encode(prompt)
            },
            None => prompt.bytes().map(|b| b as u32).collect(),
        }
    }

    fn token_id_to_text(&self, token_id: u32) -> String {
        match &self.tokenizer {
            Some(tok) => {
                let guard = tok.read();
                guard.decode(&[token_id])
            }
            None => token_id_to_text_fallback(token_id),
        }
    }
}

fn token_id_to_text_fallback(token_id: u32) -> String {
    if token_id < 256 {
        char::from_u32(token_id).unwrap_or('?').to_string()
    } else {
        format!("[{}]", token_id)
    }
}

/// Handle used inside spawned tasks to avoid borrowing self
struct InferenceEngineHandle {
    scheduler: Arc<RwLock<RequestScheduler>>,
    model: CausalLM,
    tokenizer: Option<Arc<parking_lot::RwLock<BpeTokenizer>>>,
    state: Arc<RwLock<EngineState>>,
}

impl InferenceEngineHandle {

    /// Process a batch of requests.
    /// Each request is processed sequentially through the model (single-sequence forward).
    /// Results are fanned out to individual response channels.
    pub async fn process_batch(&self, batch: crate::batching::Batch) {
        let batch_size = batch.requests.len();
        debug!("Processing batch {} with {} requests", batch.batch_id, batch_size);

        for breq in &batch.requests {
            if self.is_shutdown().await {
                break;
            }

            let start = std::time::Instant::now();
            let mut response = InferenceResponse::new(breq.request_id);

            if breq.prompt.is_empty() {
                let err_resp = response
                    .with_finish_reason(FinishReason::Error("Empty prompt".to_string()))
                    .with_inference_time(start.elapsed().as_millis() as u64);
                if let Err(e) = self.scheduler.write().await.send_response(breq.request_id, err_resp).await {
                    warn!("Failed to send empty-prompt error to {}: {}", breq.request_id, e);
                }
                continue;
            }

            let prompt_ids: Vec<u32> = match &self.tokenizer {
                Some(tok) => {
                    let t = tok.read();
                    t.encode(&breq.prompt)
                },
                None => breq.prompt.bytes().map(|b| b as u32).collect(),
            };

            let mut kv_state = self.model.reset_cache();
            let mut all_ids = prompt_ids.clone();
            let max_gen = breq.max_tokens.min(2048) as usize;

            let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
                temperature: breq.temperature,
                top_k: breq.top_k as usize,
                top_p: breq.top_p,
                ..Default::default()
            });

            for pos in 0..max_gen {
                let input = if pos == 0 {
                    all_ids.clone()
                } else {
                    vec![*all_ids.last().unwrap_or(&0)]
                };

                let logits = self.model.forward(&input, &mut kv_state);

                let token_id = match sampler.sample(&logits.to_vec()) {
                    Ok(idx) => idx as u32,
                    Err(e) => {
                        warn!("Sampler failed in speculative path: {:?}, falling back to argmax", e);
                        logits
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                            .map(|(i, _)| i as u32)
                            .unwrap_or(0)
                    },
                };

                let token_text: String = match &self.tokenizer {
                    Some(tok) => {
                        let t = tok.read();
                        t.decode(&[token_id])
                    },
                    None => token_id_to_text_fallback(token_id),
                };

                let log_prob = logits
                    .get(token_id as usize)
                    .copied()
                    .unwrap_or(0.0)
                    .ln();
                let token = GeneratedToken::new(token_id, token_text, log_prob, pos);
                response.add_token(token);
                all_ids.push(token_id);

                if token_id == 0 || start.elapsed() > Duration::from_secs(60) {
                    break;
                }
            }

            if start.elapsed() > Duration::from_secs(60) {
                response.finish_reason = FinishReason::Timeout;
            } else {
                response.finish_reason = FinishReason::MaxTokens;
            }
            response.inference_time_ms = start.elapsed().as_millis() as u64;

            if let Err(e) = self.scheduler.write().await.send_response(breq.request_id, response).await {
                error!("Failed to send batch response for {}: {}", breq.request_id, e);
            }
        }

        if let Err(e) = self.scheduler.write().await.complete_batch(&batch).await {
            error!("Failed to complete batch {}: {}", batch.batch_id, e);
        }
        debug!("Batch {} completed", batch.batch_id);
    }

    async fn is_shutdown(&self) -> bool {
        matches!(
            *self.state.read().await,
            EngineState::ShuttingDown | EngineState::Shutdown
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Queued,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
}

impl RequestStatus {
    pub fn from_scheduler_status(status: crate::scheduler::RequestStatus) -> Self {
        match status {
            crate::scheduler::RequestStatus::Queued => RequestStatus::Queued,
            crate::scheduler::RequestStatus::Processing => RequestStatus::Processing,
            crate::scheduler::RequestStatus::Completed => RequestStatus::Completed,
            crate::scheduler::RequestStatus::Failed(msg) => RequestStatus::Failed(msg),
            crate::scheduler::RequestStatus::Cancelled => RequestStatus::Cancelled,
            crate::scheduler::RequestStatus::Timeout => {
                RequestStatus::Failed("Timed out".to_string())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineStats {
    pub state: EngineState,
    pub active_requests_count: usize,
    pub max_concurrent_requests: usize,
    pub scheduler_stats: crate::scheduler::SchedulerStats,
    pub cache_stats: Option<crate::kv_cache::CacheStats>,
    pub session_count: usize,
}
