use futures::FutureExt;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::inference_trait::ModelForward;
use crate::kv_cache::KVCache;
use crate::prefix_cache::PrefixCache;
use crate::runtime::InferenceRuntime;
use crate::scheduler::RequestScheduler;
use crate::session::InferenceSession;
use crate::streaming::StreamingEngine;
use crate::{
    FinishReason, GeneratedToken, InferenceError, InferenceRequest, InferenceResponse, Result,
};
use nexora_tokenizer::BpeTokenizer;
use nexora_transformer::{CausalLM, KVCacheProvider, TransformerConfig};
#[cfg(feature = "gpu")]
use nexora_transformer::GpuKVCache;

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
    pub use_gpu: bool,
    #[cfg(feature = "gpu")]
    pub use_gpu_cache: bool,
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
            use_gpu: true,
            #[cfg(feature = "gpu")]
            use_gpu_cache: true,
        }
    }
}

pub struct InferenceEngine {
    config: InferenceConfig,
    runtime: Arc<InferenceRuntime>,
    scheduler: Arc<RwLock<RequestScheduler>>,
    kv_cache: Arc<RwLock<KVCache>>,
    session_manager: Arc<RwLock<HashMap<Uuid, InferenceSession>>>,
    model: Arc<CausalLM>,
    tokenizer: Option<Arc<parking_lot::RwLock<BpeTokenizer>>>,
    streaming_engine: Option<Arc<RwLock<StreamingEngine>>>,
    prefix_cache: Arc<PrefixCache>,
    request_tx: mpsc::Sender<InferenceRequest>,
    request_rx: Arc<Mutex<Option<mpsc::Receiver<InferenceRequest>>>>,
    active_requests: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    state: Arc<RwLock<EngineState>>,
    #[cfg(feature = "gpu")]
    gpu_cache: Option<GpuKVCache>,
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
        let model = Arc::new(CausalLM::new(model_config));
        info!(
            "CausalLM initialized with {} parameters",
            model.parameter_count()
        );

        #[cfg(feature = "gpu")]
        let gpu_cache = if config.use_gpu_cache {
            let num_layers = model.config.num_layers;
            let n_kv_heads = model.config.num_kv_heads;
            let head_dim = model.config.head_dim();
            let max_seq = model.config.max_seq_len;
            Some(GpuKVCache::new(num_layers, n_kv_heads, head_dim, max_seq))
        } else {
            None
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_cache = ();

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
            prefix_cache: Arc::new(PrefixCache::default()),
            request_tx,
            request_rx: Arc::new(Mutex::new(Some(request_rx))),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(EngineState::Uninitialized)),
            config,
            #[cfg(feature = "gpu")]
            gpu_cache,
        }
    }

    pub fn with_model(
        model: Arc<CausalLM>,
        tokenizer: Option<Arc<parking_lot::RwLock<BpeTokenizer>>>,
        config: InferenceConfig,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.queue_size_limit.max(1));
        info!(
            "Initializing inference engine with loaded model ({} params)",
            model.parameter_count()
        );

        #[cfg(feature = "gpu")]
        let gpu_cache = if config.use_gpu_cache {
            let num_layers = model.config.num_layers;
            let n_kv_heads = model.config.num_kv_heads;
            let head_dim = model.config.head_dim();
            let max_seq = model.config.max_seq_len;
            Some(GpuKVCache::new(num_layers, n_kv_heads, head_dim, max_seq))
        } else {
            None
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_cache = ();

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
            prefix_cache: Arc::new(PrefixCache::default()),
            request_tx,
            request_rx: Arc::new(Mutex::new(Some(request_rx))),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(EngineState::Uninitialized)),
            config,
            #[cfg(feature = "gpu")]
            gpu_cache,
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
    ) -> Result<mpsc::Receiver<Arc<GeneratedToken>>> {
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
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let use_gpu = self.config.use_gpu;
        #[cfg(feature = "gpu")]
        let use_gpu_cache = self.config.use_gpu_cache;
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
                }
                None => request.prompt.bytes().map(|b| b as u32).collect(),
            };

            let cpu_cache = model.reset_cache();
            #[cfg(feature = "gpu")]
            let mut kv_state: Box<dyn KVCacheProvider> = if use_gpu_cache {
                let num_layers = model.config.num_layers;
                let n_kv_heads = model.config.num_kv_heads;
                let head_dim = model.config.head_dim();
                let max_seq = model.config.max_seq_len;
                Box::new(GpuKVCache::new(num_layers, n_kv_heads, head_dim, max_seq))
            } else {
                Box::new(cpu_cache)
            };
            #[cfg(not(feature = "gpu"))]
            let mut kv_state = Box::new(cpu_cache);
            let mut all_ids = prompt_ids.clone();
            let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
                temperature,
                top_k,
                top_p,
                ..Default::default()
            });
            sampler.set_use_gpu(use_gpu);

            let max_gen = max_tokens.min(2048) as usize;

            let generation_timeout = Duration::from_secs(60);
            let gen_result = tokio::time::timeout(generation_timeout, async {
                for pos in 0..max_gen {
                    let input: &[u32] = if pos == 0 {
                        &prompt_ids
                    } else {
                        core::slice::from_ref(all_ids.last().unwrap_or(&0))
                    };

                    let logits = ModelForward::forward(model.as_ref(), input, &mut *kv_state);
                    let logits_slice = logits.as_slice().unwrap_or(&[]);

                    let token_id = match sampler.sample(logits_slice) {
                        Ok(idx) => idx as u32,
                        Err(e) => {
                            warn!("Sampler failed, error: {:?}, falling back to argmax", e);
                            match logits
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            {
                                Some((i, _)) => i as u32,
                                None => {
                                    warn!("Empty logits in streaming, using default token 1");
                                    1
                                }
                            }
                        }
                    };

                    let token_text = match &tokenizer {
                        Some(tok) => {
                            let t = tok.read();
                            t.decode(&[token_id])
                        }
                        None => token_id_to_text_fallback(token_id),
                    };

                    let log_prob = logits.get(token_id as usize).copied().map(|x| (x.max(1e-10)).ln()).unwrap_or(0.0);
                    let token = GeneratedToken::new(token_id, token_text, log_prob, pos);

                    let is_last = pos == max_gen - 1 || token_id == 0;
                    if let Err(e) = se_clone
                        .write()
                        .await
                        .push_tokens(stream_id, vec![Arc::new(token)], is_last)
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
            }).await;

            match gen_result {
                Ok(()) => {}
                Err(_elapsed) => {
                    warn!("Streaming generation timed out after 60s for request {}", request_id);
                    let _ = se_clone.write().await.push_tokens(stream_id, vec![], true).await;
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
    pub async fn generate_internal(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();
        let mut response = InferenceResponse::new(request.request_id);

        if request.prompt.is_empty() {
            return Ok(response
                .with_finish_reason(FinishReason::Error("Empty prompt".to_string()))
                .with_inference_time(start.elapsed().as_millis() as u64));
        }

        let prompt_ids = self.encode_prompt(&request.prompt);
        let max_gen = request.max_tokens.min(2048) as usize;

        // Check prefix cache (informational hit/miss tracking)
        let prefix_match = self.prefix_cache.match_prefix(&prompt_ids).await;
        if prefix_match.prefix_len > 0 {
            debug!(
                "Prefix cache hit: {}/{} tokens matched",
                prefix_match.prefix_len,
                prompt_ids.len()
            );
        }

        // Use GPU-resident KV cache when configured
        let mut cpu_cache = self.model.reset_cache();
        #[cfg(feature = "gpu")]
        let mut kv_state: Box<dyn KVCacheProvider> = if self.config.use_gpu_cache {
            let num_layers = self.model.config.num_layers;
            let n_kv_heads = self.model.config.num_kv_heads;
            let head_dim = self.model.config.head_dim();
            let max_seq = self.model.config.max_seq_len;
            Box::new(GpuKVCache::new(num_layers, n_kv_heads, head_dim, max_seq))
        } else {
            Box::new(cpu_cache)
        };
        #[cfg(not(feature = "gpu"))]
        let mut kv_state = Box::new(cpu_cache);

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
            temperature: request.temperature,
            top_k: request.top_k as usize,
            top_p: request.top_p,
            ..Default::default()
        });
        sampler.set_use_gpu(self.config.use_gpu);

        // Run CPU-bound generation loop on blocking thread to avoid async runtime starvation
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let prompt_ids_for_loop = prompt_ids.clone();
        let (tokens, last_logits, timed_out) = tokio::task::spawn_blocking(move || {
            run_generation_loop(
                &model,
                &prompt_ids_for_loop,
                max_gen,
                &mut sampler,
                &mut *kv_state,
                tokenizer.as_ref(),
            )
        })
        .await
        .unwrap_or_else(|e| {
            tracing::error!("Generation loop panicked: {:?}", e);
            (Vec::<GeneratedToken>::new(), Vec::<f32>::new(), true)
        });

        for token in tokens {
            response.add_token(token);
        }

        if timed_out {
            response.finish_reason = FinishReason::Timeout;
        } else {
            response.finish_reason = FinishReason::MaxTokens;
        }
        response.inference_time_ms = start.elapsed().as_millis() as u64;

        // Cache generated sequence + last logits for future prefix reuse
        let generated_ids: Vec<u32> = response.tokens.iter().map(|t| t.token_id).collect();
        let mut full_seq = prompt_ids.clone();
        full_seq.extend(&generated_ids);
        if !full_seq.is_empty() && !last_logits.is_empty() {
            self.prefix_cache.insert(&full_seq, last_logits).await;
        }

        debug!(
            "Prefix cache stats: hit_rate={:.3}",
            self.prefix_cache.hit_rate()
        );

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
            se.write()
                .await
                .cancel_stream(request_id)
                .await
                .unwrap_or_else(|e| {
                    warn!("Failed to cancel stream {}: {}", request_id, e);
                    false
                })
        } else {
            false
        };
        Ok(sched || task || stream)
    }

    pub async fn get_request_status(&self, request_id: Uuid) -> Result<RequestStatus> {
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

    pub async fn get_session(&self, session_id: Uuid) -> Result<Arc<InferenceSession>> {
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

    /// Spawn a batch processing task with proper panic recovery and state tracking.
    async fn spawn_batch_processor(&self, batch: crate::batching::Batch) {
        let engine = InferenceEngineHandle {
            scheduler: self.scheduler.clone(),
            model: Arc::clone(&self.model),
            tokenizer: self.tokenizer.clone(),
            state: self.state.clone(),
            use_gpu: self.config.use_gpu,
            #[cfg(feature = "gpu")]
            use_gpu_cache: self.config.use_gpu_cache,
        };
        Self::spawn_batch_processor_inner(&engine, &self.active_requests, batch).await;
    }

    // --- private ---

    /// Start the request processing loop as a background task.
    /// This does NOT block — the infinite loop runs in a spawned tokio task.
    async fn start_request_loop(&self) -> Result<()> {
        info!("Starting request processing loop");
        let mut rx =
            self.request_rx.lock().await.take().ok_or_else(|| {
                InferenceError::InternalError("Receiver already taken".to_string())
            })?;

        let state = self.state.clone();
        let scheduler = self.scheduler.clone();
        let active_requests = self.active_requests.clone();
        let session_manager = self.session_manager.clone();

        let engine = InferenceEngineHandle {
            scheduler: self.scheduler.clone(),
            model: Arc::clone(&self.model),
            tokenizer: self.tokenizer.clone(),
            state: self.state.clone(),
            use_gpu: self.config.use_gpu,
            #[cfg(feature = "gpu")]
            use_gpu_cache: self.config.use_gpu_cache,
        };

        tokio::spawn(async move {
            let mut last_cleanup = std::time::Instant::now();

            loop {
                if matches!(
                    *state.read().await,
                    EngineState::ShuttingDown | EngineState::Shutdown
                ) {
                    break;
                }

                if last_cleanup.elapsed() >= Duration::from_secs(300) {
                    // Evict stale sessions
                    {
                        let mut sessions = session_manager.write().await;
                        let before = sessions.len();
                        let now = chrono::Utc::now();
                        let timeout = chrono::Duration::seconds(
                            InferenceSession::default_timeout_seconds() as i64,
                        );
                        sessions.retain(|_, s| (now - s.created_at()) < timeout);
                        let evicted = before - sessions.len();
                        if evicted > 0 {
                            info!("Evicted {} stale sessions", evicted);
                        }
                    }
                    {
                        let mut active = active_requests.write().await;
                        active.retain(|_, h| !h.is_finished());
                    }
                    last_cleanup = std::time::Instant::now();
                }

                // Try to pop a batch first
                if let Some(batch) = scheduler.read().await.pop_batch().await {
                    Self::spawn_batch_processor_inner(&engine, &active_requests, batch).await;
                    continue;
                }

                // No batch ready, wait for next request
                let request = tokio::time::timeout(Duration::from_millis(50), rx.recv()).await;

                match request {
                    Ok(Some(req)) => {
                        scheduler
                            .write()
                            .await
                            .add_to_batch_collector(&req)
                            .await;

                        if let Some(batch) = scheduler.read().await.pop_batch().await {
                            Self::spawn_batch_processor_inner(&engine, &active_requests, batch).await;
                        }
                    }
                    Ok(None) => break,
                    Err(_) => {
                        if let Some(batch) = scheduler.read().await.pop_batch().await {
                            Self::spawn_batch_processor_inner(&engine, &active_requests, batch).await;
                        }
                        continue;
                    }
                }
            }

            info!("Request processing loop ended");
        });

        Ok(())
    }

    /// Spawn a batch processor task (standalone helper used by the spawned request loop).
    async fn spawn_batch_processor_inner(
        engine: &InferenceEngineHandle,
        active_requests: &Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
        batch: crate::batching::Batch,
    ) {
        let bid = batch.batch_id;
        let engine = InferenceEngineHandle {
            scheduler: engine.scheduler.clone(),
            model: Arc::clone(&engine.model),
            tokenizer: engine.tokenizer.clone(),
            state: engine.state.clone(),
            use_gpu: engine.use_gpu,
            #[cfg(feature = "gpu")]
            use_gpu_cache: engine.use_gpu_cache,
        };
        let active = active_requests.clone();
        let task = tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(engine.process_batch(batch))
                .catch_unwind()
                .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error!("Batch {} processing failed: {:?}", bid, e);
                    if let Err(e2) = engine.scheduler.write().await.complete_batch_id(bid).await {
                        warn!("Failed to complete batch {} after error: {}", bid, e2);
                    }
                }
                Err(panic) => {
                    error!("Batch {} processing panicked: {:?}", bid, panic);
                    if let Err(e) = engine.scheduler.write().await.complete_batch_id(bid).await {
                        warn!("Failed to complete batch {} after panic: {}", bid, e);
                    }
                }
            }
        });
        active.write().await.insert(bid, task);
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
            }
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
        char::from_u32(token_id).map_or_else(
            || format!("[{}]", token_id),
            |c| c.to_string(),
        )
    } else {
        format!("[{}]", token_id)
    }
}

/// Core generation loop shared by single-request and batch-request paths.
/// Returns generated tokens and whether a timeout occurred.
fn run_generation_loop(
    model: &CausalLM,
    prompt_ids: &[u32],
    max_gen: usize,
    sampler: &mut crate::sampler::Sampler,
    kv_state: &mut dyn KVCacheProvider,
    tokenizer: Option<&Arc<parking_lot::RwLock<BpeTokenizer>>>,
) -> (Vec<GeneratedToken>, Vec<f32>, bool) {
    let start = std::time::Instant::now();
    let mut all_ids = prompt_ids.to_vec();
    let mut tokens = Vec::with_capacity(max_gen);
    let mut last_logits = Vec::new();

    for pos in 0..max_gen {
        let input: &[u32] = if pos == 0 {
            prompt_ids
        } else {
            core::slice::from_ref(all_ids.last().unwrap_or(&0))
        };

        let logits = ModelForward::forward(model, input, kv_state);
        let logits_slice = logits.as_slice().unwrap_or(&[]);
        last_logits = logits_slice.to_vec();

        let token_id = match sampler.sample(logits_slice) {
            Ok(idx) => idx as u32,
            Err(e) => {
                warn!("Sampler failed in generation loop: {:?}, falling back to argmax", e);
                match logits.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)) {
                    Some((i, _)) => i as u32,
                    None => {
                        warn!("Empty logits, using default token 1");
                        1
                    }
                }
            }
        };

        let token_text: String = match tokenizer {
            Some(tok) => {
                let t = tok.read();
                t.decode(&[token_id])
            }
            None => token_id_to_text_fallback(token_id),
        };

        let log_prob = logits.get(token_id as usize).copied().map(|x| (x.max(1e-10)).ln()).unwrap_or(0.0);
        tokens.push(GeneratedToken::new(token_id, token_text, log_prob, pos));
        all_ids.push(token_id);

        if token_id == 0 || start.elapsed() > Duration::from_secs(60) {
            let timed_out = start.elapsed() > Duration::from_secs(60);
            return (tokens, last_logits, timed_out);
        }
    }

    (tokens, last_logits, false)
}

/// Handle used inside spawned tasks to avoid borrowing self
struct InferenceEngineHandle {
    scheduler: Arc<RwLock<RequestScheduler>>,
    model: Arc<CausalLM>,
    tokenizer: Option<Arc<parking_lot::RwLock<BpeTokenizer>>>,
    state: Arc<RwLock<EngineState>>,
    use_gpu: bool,
    #[cfg(feature = "gpu")]
    use_gpu_cache: bool,
}

impl InferenceEngineHandle {
    /// Process a batch of requests in parallel using tokio::spawn.
    /// Each request runs in its own task and results are fanned out individually.
    pub async fn process_batch(&self, batch: crate::batching::Batch) -> Result<()> {
        let batch_size = batch.requests.len();
        if batch_size == 0 {
            return Ok(());
        }
        debug!(
            "Processing batch {} with {} requests",
            batch.batch_id, batch_size
        );

        if self.is_shutdown().await {
            self.scheduler.write().await.complete_batch(&batch).await
                .map_err(|e| warn!("Failed to complete batch {} during shutdown: {}", batch.batch_id, e))
                .ok();
            return Ok(());
        }

        let batch_id = batch.batch_id;
        let mut handles = Vec::with_capacity(batch_size);

        for breq in batch.requests {
            if self.is_shutdown().await {
                break;
            }

            let scheduler = self.scheduler.clone();
            let model = self.model.clone();
            let tokenizer = self.tokenizer.clone();
            let use_gpu = self.use_gpu;
            #[cfg(feature = "gpu")]
            let use_gpu_cache = self.use_gpu_cache;

            let handle = tokio::spawn(async move {
                let start = std::time::Instant::now();
                let mut response = InferenceResponse::new(breq.request_id);

                if breq.prompt.is_empty() {
                    let err_resp = response
                        .with_finish_reason(FinishReason::Error("Empty prompt".to_string()))
                        .with_inference_time(start.elapsed().as_millis() as u64);
                    if let Err(e) = scheduler
                        .write()
                        .await
                        .send_response(breq.request_id, err_resp)
                        .await
                    {
                        warn!(
                            "Failed to send empty-prompt error to {}: {}",
                            breq.request_id, e
                        );
                    }
                    return;
                }

                let prompt_ids: Vec<u32> = match &tokenizer {
                    Some(tok) => {
                        let t = tok.read();
                        t.encode(&breq.prompt)
                    }
                    None => breq.prompt.bytes().map(|b| b as u32).collect(),
                };

                let cpu_cache = model.reset_cache();
                #[cfg(feature = "gpu")]
                let mut kv_state: Box<dyn KVCacheProvider> = if use_gpu_cache {
                    let num_layers = model.config.num_layers;
                    let n_kv_heads = model.config.num_kv_heads;
                    let head_dim = model.config.head_dim();
                    let max_seq = model.config.max_seq_len;
                    Box::new(GpuKVCache::new(num_layers, n_kv_heads, head_dim, max_seq))
                } else {
                    Box::new(cpu_cache)
                };
                #[cfg(not(feature = "gpu"))]
                let mut kv_state = Box::new(cpu_cache);
                let max_gen = breq.max_tokens.min(2048) as usize;

                let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
                    temperature: breq.temperature,
                    top_k: breq.top_k as usize,
                    top_p: breq.top_p,
                    ..Default::default()
                });
                sampler.set_use_gpu(use_gpu);

                let model_c = Arc::clone(&model);
                let prompt_ids_c = prompt_ids.clone();
                let (tokens, _last_logits, timed_out) = tokio::task::spawn_blocking(move || {
                    run_generation_loop(
                        &model_c,
                        &prompt_ids_c,
                        max_gen,
                        &mut sampler,
                        &mut *kv_state,
                        tokenizer.as_ref(),
                    )
                })
                .await
                .unwrap_or_else(|e| {
                    tracing::error!("Batch generation loop panicked: {:?}", e);
                    (Vec::<GeneratedToken>::new(), Vec::<f32>::new(), true)
                });

                for token in tokens {
                    response.add_token(token);
                }

                if timed_out {
                    response.finish_reason = FinishReason::Timeout;
                } else {
                    response.finish_reason = FinishReason::MaxTokens;
                }
                response.inference_time_ms = start.elapsed().as_millis() as u64;

                if let Err(e) = scheduler
                    .write()
                    .await
                    .send_response(breq.request_id, response)
                    .await
                {
                    error!(
                        "Failed to send batch response for {}: {}",
                        breq.request_id, e
                    );
                }
            });
            handles.push(handle);
        }

        // Wait for all parallel tasks to complete
        for handle in handles {
            if let Err(e) = handle.await {
                error!("Parallel task in batch {} panicked: {:?}", batch_id, e);
            }
        }

        if let Err(e) = self.scheduler.write().await.complete_batch_id(batch_id).await {
            error!("Failed to complete batch {}: {}", batch_id, e);
        }
        debug!("Batch {} completed", batch_id);
        Ok(())
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
