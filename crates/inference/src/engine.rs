use futures::FutureExt;
use ndarray::Array1;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, info_span, warn, Instrument};
use uuid::Uuid;

use crate::continuous_batching::ContinuousBatchingConfig;
use crate::distributed::DistributedRouter;
use crate::inference_trait::ModelForward;
use crate::kv_cache::KVCache;
use crate::paged_cache::{PagedCacheConfig, PagedKVCache};
use crate::paged_provider::PagedKVCacheProvider;
use std::sync::Mutex as StdMutex;
use crate::prefix_cache::PrefixCache;
use crate::runtime::InferenceRuntime;
use crate::scheduler::RequestScheduler;
use crate::session::InferenceSession;
use crate::streaming::StreamingEngine;
use crate::{
    FinishReason, GeneratedToken, InferenceError, InferenceRequest, InferenceResponse, Result,
    SessionEntry,
};
use nexora_common::retry::RetryConfig;
use nexora_runtime::cluster::{ClusterConfig, NodeRegistry};
use nexora_memory::MemoryManager;
use nexora_tokenizer::BpeTokenizer;
#[cfg(feature = "gpu")]
use nexora_transformer::GpuKVCache;
use nexora_transformer::{CausalLM, KVCacheEntry, KVCacheProvider, TransformerConfig};

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
    pub generation_timeout_seconds: u64,
    pub cleanup_interval_seconds: u64,
    pub drain_timeout_seconds: u64,
    pub recv_timeout_millis: u64,
    pub metrics_interval_seconds: u64,
    pub use_gpu: bool,
    #[cfg(feature = "gpu")]
    pub use_gpu_cache: bool,
    #[cfg(feature = "gpu")]
    pub use_gpu_resident: bool,
    pub use_continuous_batching: bool,
    pub use_continuous_batching_config: Option<ContinuousBatchingConfig>,
    pub use_paged_cache: bool,
    pub paged_cache_f16: bool,
    pub paged_block_size: usize,
    pub paged_max_blocks: usize,
    pub checkpoint_path: Option<String>,
    pub enable_distributed: bool,
    pub distributed_listen_address: String,
    pub distributed_seed_nodes: Vec<String>,
    pub cluster_config: Option<ClusterConfig>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 32,
            default_model_id: "default-model".to_string(),
            enable_queuing: true,
            queue_size_limit: 1000,
            enable_caching: true,
            cache_size_limit_mb: 1024,
            enable_streaming: true,
            default_timeout_seconds: 30,
            generation_timeout_seconds: 60,
            cleanup_interval_seconds: 30,
            drain_timeout_seconds: 15,
            recv_timeout_millis: 50,
            metrics_interval_seconds: 60,
            use_gpu: true,
            #[cfg(feature = "gpu")]
            use_gpu_cache: true,
            #[cfg(feature = "gpu")]
            use_gpu_resident: true,
            use_continuous_batching: true,
            use_continuous_batching_config: None,
            use_paged_cache: true,
            paged_cache_f16: true,
            paged_block_size: 0,
            paged_max_blocks: 0,
            checkpoint_path: None,
            enable_distributed: false,
            distributed_listen_address: "127.0.0.1:8080".to_string(),
            distributed_seed_nodes: Vec::new(),
            cluster_config: None,
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
    tokenizer: Option<Arc<StdMutex<BpeTokenizer>>>,
    streaming_engine: Option<Arc<RwLock<StreamingEngine>>>,
    prefix_cache: Arc<PrefixCache>,
    shared_paged: Option<Arc<StdMutex<PagedKVCache>>>,
    request_tx: mpsc::Sender<InferenceRequest>,
    request_rx: Arc<Mutex<Option<mpsc::Receiver<InferenceRequest>>>>,
    active_requests: Arc<RwLock<HashMap<Uuid, tokio::task::JoinHandle<()>>>>,
    state: Arc<RwLock<EngineState>>,
    model_ready: bool,
    memory: Option<Arc<Mutex<MemoryManager>>>,
    distributed: Option<Arc<DistributedRouter>>,
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
        let (model, model_ready) = if let Some(ref ckpt_path) = config.checkpoint_path {
            match CausalLM::from_checkpoint(model_config.clone(), ckpt_path) {
                Ok(m) => {
                    info!("Loaded checkpoint from {}", ckpt_path);
                    (Arc::new(m), true)
                }
                Err(e) => {
                    error!("Failed to load checkpoint from {}: {}", ckpt_path, e);
                    warn!("Falling back to random weights — model is uninitialized for inference until checkpoint is loaded");
                    warn!("InferenceEngine::new() creates uninitialized model — use with_model() instead");
                    (Arc::new(CausalLM::new(model_config.clone())), true)
                }
            }
        } else {
            warn!("No checkpoint path configured — model initialized with random weights");
            warn!("InferenceEngine::new() creates uninitialized model — use with_model() instead");
            (Arc::new(CausalLM::new(model_config.clone())), true)
        };

        info!(
            "CausalLM initialized with {} parameters",
            model.parameter_count()
        );

        let shared_paged = Self::maybe_init_paged_cache(&config, Some(&model));

        Self {
            runtime: Arc::new(InferenceRuntime::new()),
            scheduler: Arc::new(RwLock::new(RequestScheduler::new(
                config.max_concurrent_requests,
            ))),
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
            shared_paged,
            request_tx,
            request_rx: Arc::new(Mutex::new(Some(request_rx))),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(EngineState::Uninitialized)),
            config,
            model_ready,
            memory: None,
            distributed: None,
        }
    }

    pub fn with_memory(mut self, memory: MemoryManager) -> Self {
        self.memory = Some(Arc::new(Mutex::new(memory)));
        self
    }

    pub fn with_model(
        model: Arc<CausalLM>,
        tokenizer: Option<Arc<StdMutex<BpeTokenizer>>>,
        config: InferenceConfig,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.queue_size_limit.max(1));
        info!(
            "Initializing inference engine with loaded model ({} params)",
            model.parameter_count()
        );

        let shared_paged = Self::maybe_init_paged_cache(&config, Some(&model));

        Self {
            runtime: Arc::new(InferenceRuntime::new()),
            scheduler: Arc::new(RwLock::new(RequestScheduler::new(
                config.max_concurrent_requests,
            ))),
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
            shared_paged,
            request_tx,
            request_rx: Arc::new(Mutex::new(Some(request_rx))),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(EngineState::Uninitialized)),
            config,
            model_ready: true,
            memory: None,
            distributed: None,
        }
    }

    pub fn with_distributed(
        mut self,
        registry: Arc<NodeRegistry>,
        node_id: Uuid,
    ) -> Self {
        let shared_secret = self.config.cluster_config.as_ref()
            .and_then(|c| c.shared_secret.clone());
        let tls_enabled = self.config.cluster_config.as_ref()
            .map(|c| c.tls_enabled)
            .unwrap_or(false);
        self.distributed = Some(Arc::new(DistributedRouter::new_with_auth(
            registry, node_id, shared_secret, tls_enabled,
        )));
        self
    }

    fn maybe_init_paged_cache(
        config: &InferenceConfig,
        model: Option<&Arc<CausalLM>>,
    ) -> Option<Arc<StdMutex<PagedKVCache>>> {
        if !config.use_paged_cache {
            return None;
        }
        if let Some(model) = model {
            let c = &model.config;
            let block_size = if config.paged_block_size > 0 {
                config.paged_block_size
            } else {
                PagedCacheConfig::suggest_block_size(c.head_dim(), c.num_kv_heads)
            };
            let max_blocks = if config.paged_max_blocks > 0 {
                config.paged_max_blocks
            } else {
                ((c.max_seq_len / block_size).max(64) * 4).min(65536)
            };
            let pc_config = PagedCacheConfig {
                block_size,
                max_blocks,
                num_layers: c.num_layers,
                num_kv_heads: c.num_kv_heads,
                head_dim: c.head_dim(),
                max_seq_len: c.max_seq_len,
                f16_storage: config.paged_cache_f16,
            };
            info!(
                "Initializing PagedKVCache: block_size={}, max_blocks={}, layers={}",
                block_size, max_blocks, c.num_layers
            );
            Some(Arc::new(StdMutex::new(PagedKVCache::new(pc_config))))
        } else {
            warn!("Paged cache requested but model not loaded yet — deferred to first cache creation");
            Some(Arc::new(StdMutex::new(PagedKVCache::new(PagedCacheConfig {
                num_layers: 1,
                block_size: 16,
                max_blocks: 64,
                num_kv_heads: 1,
                head_dim: 64,
                max_seq_len: 2048,
                f16_storage: config.paged_cache_f16,
            }))))
        }
    }

    /// Generate a unique sequence ID for paged cache registration.
    fn next_seq_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Create the appropriate KVCacheProvider based on engine config.
    fn make_kv_provider(
        model: &CausalLM,
        use_gpu_cache: bool,
        shared_paged: &Option<Arc<StdMutex<PagedKVCache>>>,
    ) -> Box<dyn KVCacheProvider> {
        // Paged cache takes priority: block-based, copy-on-write, prefix sharing
        if let Some(ref paged) = shared_paged {
            let seq_id = Self::next_seq_id();
            let num_layers = model.config.num_layers;
            info!(
                "Creating PagedKVCacheProvider seq_id={}, layers={}",
                seq_id, num_layers
            );
            return Box::new(PagedKVCacheProvider::new_shared(
                seq_id,
                paged.clone(),
                num_layers,
            ));
        }

        // GPU-resident cache
        #[cfg(feature = "gpu")]
        if use_gpu_cache {
            let num_layers = model.config.num_layers;
            let n_kv_heads = model.config.num_kv_heads;
            let head_dim = model.config.head_dim();
            let max_seq = model.config.max_seq_len;
            return Box::new(GpuKVCache::new(num_layers, n_kv_heads, head_dim, max_seq));
        }

        // Default: flat CPU cache
        Box::new(model.reset_cache())
    }

    pub async fn initialize(&mut self) -> Result<()> {
        if *self.state.read().await != EngineState::Uninitialized {
            return Err(InferenceError::EngineNotInitialized(
                "Engine already initialized".to_string(),
            ));
        }
        if !self.model_ready {
            return Err(InferenceError::ModelNotLoaded(
                "No checkpoint loaded — use with_model() or set checkpoint_path in InferenceConfig"
                    .to_string(),
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
        let span = info_span!(
            "submit_request",
            request_id = %request.request_id,
            max_tokens = request.max_tokens,
            temperature = request.temperature,
        );
        let _guard = span.enter();
        match *self.state.read().await {
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

        if let Some(ref remote) = self.distributed {
            if let Some(peer) = remote.best_remote_node().await {
                let resp = remote.route_remote(&peer, &request).await?;
                let (response_tx, response_rx) = mpsc::channel(1);
                let _ = response_tx.send(resp).await;
                return Ok(response_rx);
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
        let span = info_span!(
            "submit_streaming_request",
            request_id = %request.request_id,
            max_tokens = request.max_tokens,
            temperature = request.temperature,
        );
        let _guard = span.enter();
        if !self.config.enable_streaming {
            return Err(InferenceError::InvalidRequest(
                "Streaming disabled".to_string(),
            ));
        }

        if let Some(ref remote) = self.distributed {
            if let Some(peer) = remote.best_remote_node().await {
                return remote.route_remote_streaming(&peer, &request).await;
            }
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
        let shared_paged = self.shared_paged.clone();
        let _scheduler = self.scheduler.clone();
        let se_clone = se.clone();
        let active = self.active_requests.clone();

        let request_id = request.request_id;
        let prompt = request.prompt;
        let max_tokens = request.max_tokens;
        let temperature = request.temperature;
        let top_p = request.top_p;
        let top_k = request.top_k as usize;
        let generation_timeout = Duration::from_secs(self.config.generation_timeout_seconds);

        // Memory-based RAG for streaming path (no clone: move prompt ownership)
        let augmented_prompt = if let Some(memory) = &self.memory {
            let mem = memory.lock().await;
            match mem.search(&prompt).await {
                Ok(results) if !results.is_empty() => {
                    use std::fmt::Write;
                    let mut context = String::new();
                    for (i, r) in results.iter().take(3).enumerate() {
                        if i > 0 {
                            context.push('\n');
                        }
                        let _ = write!(context, "- {}", r.value);
                    }
                    format!("Relevant context:\n{}\n\n---\n\n{}", context, prompt)
                }
                _ => prompt,
            }
        } else {
            prompt
        };
        let memory_clone = self.memory.clone();

        let task = tokio::spawn(async move {
            let prompt_ids: Vec<u32> = match &tokenizer {
                Some(tok) => {
                let t = tok.lock().unwrap();
                t.encode(&augmented_prompt)
                }
                None => {
                    warn!(
                        "No tokenizer configured for streaming request {}",
                        request_id
                    );
                    return;
                }
            };

            let mut kv_state = Self::make_kv_provider(&model, use_gpu_cache, &shared_paged);
            let max_gen = max_tokens.min(2048) as usize;
            let mut generated_ids: Vec<u32> = Vec::with_capacity(max_gen);
            let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
                temperature,
                top_k,
                top_p,
                ..Default::default()
            });
            sampler.set_use_gpu(use_gpu);

            let gen_result = tokio::time::timeout(generation_timeout, async {
                let base_seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                for pos in 0..max_gen {
                    let input: &[u32] = if pos == 0 {
                        &prompt_ids
                    } else {
                        core::slice::from_ref(generated_ids.last().unwrap_or(&0))
                    };

                    // Try GPU-native path first — no 128KB logits readback per token
                    let (token_id, log_prob) = if let Some(tok) = ModelForward::forward_sample_gpu(
                        model.as_ref(),
                        input,
                        &mut *kv_state,
                        temperature,
                        top_k,
                        top_p,
                        base_seed.wrapping_add(pos as u64),
                    ) {
                        (tok, 0.0f32)
                    } else {
                        let logits = ModelForward::forward(model.as_ref(), input, &mut *kv_state);
                        let logits_slice = logits.as_slice().unwrap_or(&[]);

                        let tok = match sampler.sample(logits_slice) {
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
                        let lp = logits
                            .get(tok as usize)
                            .copied()
                            .map(|x| (x.max(1e-10)).ln())
                            .unwrap_or(0.0);
                        (tok, lp)
                    };

                    let token_text = match &tokenizer {
                        Some(tok) => {
                        let t = tok.lock().unwrap();
                        t.decode(&[token_id])
                        }
                        None => token_id_to_text_fallback(token_id),
                    };
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

                    generated_ids.push(token_id);
                    if is_last {
                        break;
                    }
                }
            })
            .await;

            match gen_result {
                Ok(()) => {}
                Err(_elapsed) => {
                    warn!(
                        "Streaming generation timed out after 60s for request {}",
                        request_id
                    );
                    let _ = se_clone
                        .write()
                        .await
                        .push_tokens(stream_id, vec![], true)
                        .await;
                }
            }

            // Store interaction in memory for future RAG
            if let Some(ref memory) = memory_clone {
                if !generated_ids.is_empty() {
                    let response_text = match &tokenizer {
                        Some(tok) => {
                        let t = tok.lock().unwrap();
                        t.decode(&generated_ids)
                        }
                        None => format!("[{} tokens generated]", generated_ids.len()),
                    };
                    let _ = memory
                        .lock()
                        .await
                        .store_interaction(&augmented_prompt, &response_text)
                        .await;
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

    /// Generate using continuous batching — multiple sequences share batched forward passes.
    /// Uses batched model forward (`forward_batched`) to process multiple sequences in
    /// a single kernel launch. Only activates with `use_continuous_batching: true`.
    /// Falls back to per-request generation for single requests.
    pub async fn generate_continuous_batched(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Vec<InferenceResponse> {
        if requests.is_empty() {
            return Vec::new();
        }
        let _span = info_span!("generate_continuous_batched", batch_size = requests.len()).entered();
        if requests.len() == 1 || !self.config.use_continuous_batching {
            return match requests.into_iter().next() {
                Some(req) => {
                    let request_id = req.request_id;
                    match self.generate_internal(req).await {
                        Ok(r) => vec![r],
                        Err(e) => {
                            let mut err_resp = InferenceResponse::new(request_id);
                            err_resp.finish_reason = FinishReason::Error(e.to_string());
                            vec![err_resp]
                        }
                    }
                }
                None => Vec::new(),
            };
        }

        let start = std::time::Instant::now();
        let max_steps = requests
            .iter()
            .map(|r| r.max_tokens as usize)
            .max()
            .unwrap_or(1024);
        let use_gpu = self.config.use_gpu;

        // Prepare per-sequence state
        struct SeqState {
            request_id: uuid::Uuid,
            prompt: Vec<u32>,
            generated: Vec<u32>,
            kv_cache: Vec<nexora_transformer::KVCacheEntry>,
            prompt_pos: usize,
            temperature: f32,
            top_k: usize,
            top_p: f32,
            finished: bool,
            finish_reason: FinishReason,
        }

        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let mut sequences: Vec<SeqState> = Vec::with_capacity(requests.len());

        for req in &requests {
            let prompt_ids = self
                .encode_prompt(&req.prompt)
                .await
                .unwrap_or_else(|_| req.input_tokens.clone());
            let cache = model.reset_cache();
            sequences.push(SeqState {
                request_id: req.request_id,
                prompt: prompt_ids,
                generated: Vec::new(),
                kv_cache: cache.entries,
                prompt_pos: 0,
                temperature: req.temperature,
                top_k: req.top_k as usize,
                top_p: req.top_p,
                finished: false,
                finish_reason: FinishReason::Unknown,
            });
        }

        // BATTER STEP LOOP
        let completed = tokio::task::spawn_blocking(move || {
            let mut sampler =
                crate::sampler::Sampler::new(crate::sampler::SamplingConfig::default());
            sampler.set_use_gpu(use_gpu);

            for _step in 0..max_steps {
                // Collect ready sequences
                let mut active_indices: Vec<usize> = Vec::new();
                let mut tokens: Vec<u32> = Vec::new();
                let mut seq_caches: Vec<nexora_transformer::CpuKVCache> = Vec::new();

                for (i, seq) in sequences.iter_mut().enumerate() {
                    if seq.finished {
                        continue;
                    }
                    let input = if seq.prompt_pos < seq.prompt.len() {
                        seq.prompt[seq.prompt_pos]
                    } else {
                        seq.generated.last().copied().unwrap_or(0)
                    };
                    active_indices.push(i);
                    tokens.push(input);
                    let cache = std::mem::replace(&mut seq.kv_cache, Vec::new());
                    seq_caches.push(nexora_transformer::CpuKVCache { entries: cache });
                }

                if active_indices.is_empty() {
                    break;
                }

                // Batched forward — prefer GPU batched matmul for throughput
                let all_logits: Vec<Array1<f32>> = if seq_caches.len() > 1 {
                    #[cfg(feature = "gpu")]
                    if use_gpu {
                        let mut vec_caches: Vec<Vec<nexora_transformer::KVCacheEntry>> = seq_caches
                            .iter_mut()
                            .map(|c| std::mem::take(&mut c.entries))
                            .collect();
                        match model.forward_gpu_batched(&tokens, &mut vec_caches) {
                            Ok(logits) => {
                                for (i, entries) in vec_caches.into_iter().enumerate() {
                                    seq_caches[i].entries = entries;
                                }
                                logits
                            }
                            Err(e) => {
                                crate::inference_trait::GPU_CPU_FALLBACKS.fetch_add(
                                    seq_caches.len() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                tracing::warn!(
                                    "Batched GPU forward failed: {e}, falling back to per-sequence"
                                );
                                for (i, entries) in vec_caches.into_iter().enumerate() {
                                    seq_caches[i].entries = entries;
                                }
                                tokens
                                    .iter()
                                    .zip(seq_caches.iter_mut())
                                    .filter_map(|(&tok, cache)| model.forward(&[tok], cache).ok())
                                    .collect()
                            }
                        }
                    } else {
                        tokens
                            .iter()
                            .zip(seq_caches.iter_mut())
                            .filter_map(|(&tok, cache)| model.forward(&[tok], cache).ok())
                            .collect()
                    }
                    #[cfg(not(feature = "gpu"))]
                    tokens
                        .iter()
                        .zip(seq_caches.iter_mut())
                        .filter_map(|(&tok, cache)| model.forward(&[tok], cache).ok())
                        .collect()
                } else {
                    tokens
                        .iter()
                        .zip(seq_caches.iter_mut())
                        .filter_map(|(&tok, cache)| model.forward(&[tok], cache).ok())
                        .collect()
                };

                // Restore caches and sample
                for (batch_idx, &seq_idx) in active_indices.iter().enumerate() {
                    let seq = &mut sequences[seq_idx];
                    seq.kv_cache = std::mem::take(&mut seq_caches[batch_idx].entries);

                    if batch_idx >= all_logits.len() {
                        continue;
                    }
                    let logits_arr = &all_logits[batch_idx];
                    let logits_slice: &[f32] = logits_arr.as_slice().unwrap_or(&[]);
                    if logits_slice.is_empty() {
                        seq.finished = true;
                        seq.finish_reason = FinishReason::Error("Empty logits from forward".into());
                        continue;
                    }

                    let token_id = match sampler.sample(logits_slice) {
                        Ok(idx) => idx as u32,
                        Err(_) => logits_slice
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(i, _)| i as u32)
                            .unwrap_or(0),
                    };

                    seq.generated.push(token_id);
                    if seq.prompt_pos < seq.prompt.len() {
                        seq.prompt_pos += 1;
                    }

                    // Check completion
                    if token_id == 0 || token_id == 2 {
                        seq.finished = true;
                        seq.finish_reason = FinishReason::EndOfSequence;
                    } else if seq.generated.len() >= max_steps {
                        seq.finished = true;
                        seq.finish_reason = FinishReason::MaxTokens;
                    }
                }
            }

            // Build responses
            let mut results = Vec::with_capacity(sequences.len());
            for mut seq in sequences {
                let text: String = seq
                    .generated
                    .iter()
                    .map(|&id| {
                        if let Some(ref tok) = tokenizer {
                            let t = tok.lock().unwrap();
                            t.decode(&[id])
                        } else {
                            format!("[{}]", id)
                        }
                    })
                    .collect();
                let tokens: Vec<GeneratedToken> = seq
                    .generated
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| GeneratedToken::new(id, format!("[{}]", id), 0.0, i))
                    .collect();
                results.push(InferenceResponse {
                    request_id: seq.request_id,
                    tokens,
                    text,
                    finish_reason: seq.finish_reason,
                    total_tokens: seq.generated.len(),
                    inference_time_ms: 0,
                    metadata: std::collections::HashMap::new(),
                });
            }
            results
        })
        .await
        .unwrap_or_else(|e| {
            tracing::error!("Continuous batching panicked: {:?}", e);
            Vec::new()
        });

        let elapsed = start.elapsed().as_millis() as u64;
        completed
            .into_iter()
            .map(|mut resp| {
                resp.inference_time_ms = elapsed;
                resp
            })
            .collect()
    }

    /// Generate tokens for a single request.
    #[tracing::instrument(skip_all, fields(request_id = %request.request_id, prompt_len = request.prompt.len()))]
    pub async fn generate_internal(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();
        let mut response = InferenceResponse::new(request.request_id);

        // Session tracking: get or create session if session_id provided
        let session = if let Some(sid) = request.session_id {
            Some(self.get_session(sid).await?)
        } else {
            None
        };

        if request.prompt.is_empty() {
            if let Some(ref s) = session {
                let _ = s
                    .add_entry(SessionEntry {
                        entry_id: Uuid::new_v4(),
                        request_id: request.request_id,
                        timestamp: chrono::Utc::now(),
                        prompt: String::new(),
                        response: String::new(),
                        tokens_generated: 0,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                        metadata: HashMap::new(),
                    })
                    .await;
            }
            return Ok(response
                .with_finish_reason(FinishReason::Error("Empty prompt".to_string()))
                .with_inference_time(start.elapsed().as_millis() as u64));
        }

        // Memory-based RAG: enrich prompt with relevant past context (no clone: move prompt)
        let prompt = request.prompt;
        let original_prompt = prompt.clone();
        let augmented_prompt = if let Some(memory) = &self.memory {
            let mem = memory.lock().await;
            match mem.search(&prompt).await {
                Ok(results) if !results.is_empty() => {
                    use std::fmt::Write;
                    let mut context = String::new();
                    for (i, r) in results.iter().take(3).enumerate() {
                        if i > 0 {
                            context.push('\n');
                        }
                        let _ = write!(context, "- {}", r.value);
                    }
                    let augmented = format!("Relevant context:\n{}\n\n---\n\n{}", context, prompt);
                    debug!(
                        "Memory RAG: prepended {} context items to prompt",
                        results.iter().take(3).count()
                    );
                    augmented
                }
                _ => prompt,
            }
        } else {
            prompt
        };

        let prompt_ids = self.encode_prompt(&augmented_prompt).await?;
        let max_gen = request.max_tokens.min(2048) as usize;

        // Check prefix cache for warm-start opportunity
        let prefix_match = self.prefix_cache.match_prefix(&prompt_ids).await;
        let prefix_len = prefix_match.prefix_len;
        let prefix_logits = prefix_match.cached_value;
        let prefix_kv_cache = prefix_match.cached_kv_cache;
        if let Some(ref s) = session {
            if prefix_len > 0 {
                s.record_cache_hit().await;
            } else {
                s.record_cache_miss().await;
            }
        }
        if prefix_len > 0 {
            debug!(
                "Prefix cache hit: {}/{} tokens matched, skipping prefill",
                prefix_len,
                prompt_ids.len()
            );
        }

        // Use paged/GPU/CPU KV cache based on config
        let mut kv_state = Self::make_kv_provider(
            &self.model,
            self.config.use_gpu_cache,
            &self.shared_paged,
        );

        // Restore KV cache from prefix match if available — avoids recomputing K/V for prefix tokens
        if !prefix_kv_cache.is_empty() {
            if let Some(cpu_entries) = kv_state.as_cpu_entries() {
                *cpu_entries = prefix_kv_cache;
                debug!("Restored KV cache for {} matched prefix tokens", prefix_len);
            }
        }

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplingConfig {
            temperature: request.temperature,
            top_k: request.top_k as usize,
            top_p: request.top_p,
            ..Default::default()
        });
        sampler.set_use_gpu(self.config.use_gpu);

        // GPU-resident generation path: logits stay on GPU, only 4 bytes/token readback.
        // Falls back to per-token loop if GPU-resident returns error or is disabled.
        #[cfg(feature = "gpu")]
        if self.config.use_gpu_resident {
            let model = Arc::clone(&self.model);
            let prompt_ids_for_loop = prompt_ids.clone();
            let result = tokio::task::spawn_blocking(move || {
                model.generate_with_gpu_keep_gpu(
                    &prompt_ids_for_loop,
                    max_gen,
                    request.temperature,
                    request.top_k as usize,
                    request.top_p,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64,
                )
            })
            .await;

            match result {
                Ok((out_tokens, _out_cache)) if !out_tokens.is_empty() => {
                    crate::inference_trait::GPU_RESIDENT_SUCCESSES
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    crate::inference_trait::GPU_TOKENS_GENERATED.fetch_add(
                        out_tokens.len() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let mut finish_reason = FinishReason::Unknown;
                    for (i, &token_id) in out_tokens.iter().enumerate() {
                        let token_text: String = match &self.tokenizer {
                    Some(tok) => {
                        let t = tok.lock().unwrap();
                        t.decode(&[token_id])
                    }
                            None => token_id_to_text_fallback(token_id),
                        };
                        let log_prob = 0.0;
                        response.add_token(GeneratedToken::new(token_id, token_text, log_prob, i));
                        if token_id == 0 || token_id == 2 {
                            finish_reason = FinishReason::EndOfSequence;
                            break;
                        }
                    }
                    if finish_reason == FinishReason::Unknown {
                        finish_reason = FinishReason::MaxTokens;
                    }
                    response.finish_reason = finish_reason;
                    response.inference_time_ms = start.elapsed().as_millis() as u64;
                    return Ok(response);
                }
                Ok((_, _)) => {
                    crate::inference_trait::GPU_RESIDENT_FALLBACKS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    tracing::warn!(
                        "GPU-resident generation returned empty, falling back to per-token loop"
                    );
                }
                Err(e) => {
                    crate::inference_trait::GPU_RESIDENT_FALLBACKS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    tracing::error!("GPU-resident generation panicked: {:?}, falling back", e);
                }
            }
        }

        // Run CPU-bound generation loop on blocking thread to avoid async runtime starvation
        let model = Arc::clone(&self.model);
        let tokenizer = self.tokenizer.clone();
        let prompt_ids_for_loop = prompt_ids.clone();
        let generation_timeout = Duration::from_secs(self.config.generation_timeout_seconds);
        let include_kv_cache = true;
        let (tokens, last_logits, timed_out, prefix_kv_cache) =
            tokio::task::spawn_blocking(move || {
                run_generation_loop(
                    &model,
                    &prompt_ids_for_loop,
                    max_gen,
                    &mut sampler,
                    &mut *kv_state,
                    tokenizer.as_ref(),
                    prefix_len,
                    prefix_logits,
                    generation_timeout,
                    include_kv_cache,
                )
            })
            .await
            .map_err(|e| {
                tracing::error!("Generation loop panicked: {:?}", e);
                InferenceError::InternalError(format!("Generation loop panicked: {:?}", e))
            })?;

        for token in tokens {
            response.add_token(token);
        }

        if timed_out {
            response.finish_reason = FinishReason::Timeout;
        } else {
            response.finish_reason = FinishReason::MaxTokens;
        }
        response.inference_time_ms = start.elapsed().as_millis() as u64;

        // Cache generated sequence + last logits + KV cache for future prefix reuse
        let generated_ids: Vec<u32> = response.tokens.iter().map(|t| t.token_id).collect();
        let mut full_seq = prompt_ids;
        full_seq.extend(&generated_ids);
        if !full_seq.is_empty() && !last_logits.is_empty() {
            let kvcache = if prefix_kv_cache.is_empty() {
                None
            } else {
                Some(prefix_kv_cache)
            };
            self.prefix_cache
                .insert_with_kvcache(&full_seq, last_logits, kvcache)
                .await;
        }

        debug!(
            "Prefix cache stats: hit_rate={:.3}",
            self.prefix_cache.hit_rate()
        );

        // Store interaction in memory for future RAG
        if let Some(memory) = &self.memory {
            let mem = memory.lock().await;
            let response_text = response.text.clone();
            if !response_text.is_empty() {
                let _ = mem
                    .store_interaction(&augmented_prompt, &response_text)
                    .await;
            }
        }

        // Finalize inference time before session entry
        let inference_time = start.elapsed().as_millis() as u64;
        let response = response.with_inference_time(inference_time);

        // Record session entry for this inference request
        if let Some(ref s) = session {
            let tokens_generated = response.tokens.len();
            let _ = s
                .add_entry(SessionEntry {
                    entry_id: Uuid::new_v4(),
                    request_id: request.request_id,
                    timestamp: chrono::Utc::now(),
                    prompt: original_prompt.clone(),
                    response: response.text.clone(),
                    tokens_generated,
                    processing_time_ms: inference_time,
                    metadata: HashMap::new(),
                })
                .await;
        }

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

    /// Create a new inference session for tracking generation state.
    /// Called from generate_internal when a session_id is provided.
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
        let active = self.active_requests.read().await.len();
        let sched_stats = self.scheduler.read().await.get_stats().await;
        let cache_stats = if self.config.enable_caching {
            Some(self.kv_cache.read().await.get_stats())
        } else {
            None
        };
        let session_count = self.session_manager.read().await.len();
        EngineStats {
            state: self.state.read().await.clone(),
            active_requests_count: active,
            max_concurrent_requests: self.config.max_concurrent_requests,
            scheduler_stats: sched_stats,
            cache_stats,
            session_count,
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference engine (drain timeout: {}s)", self.config.drain_timeout_seconds);
        *self.state.write().await = EngineState::ShuttingDown;

        let drain_duration = Duration::from_secs(self.config.drain_timeout_seconds);
        let drain_start = std::time::Instant::now();
        let total_active = self.active_requests.read().await.len();

        {
            let mut active = self.active_requests.write().await;
            active.retain(|_, h| !h.is_finished());
            let in_flight = active.len();
            if in_flight > 0 {
                info!("Draining {} in-flight requests (timeout: {}s)", in_flight, self.config.drain_timeout_seconds);
                let remaining = drain_duration.saturating_sub(drain_start.elapsed());
                let handles: Vec<_> = active.drain().map(|(_, h)| h).collect();
                drop(active);
                for h in handles {
                    if tokio::time::timeout(remaining, h).await.is_err() {
                        warn!("Request handle timed out during drain");
                    }
                }
            } else {
                info!("No in-flight requests to drain");
            }
        }

        self.scheduler.write().await.shutdown().await?;
        if let Some(se) = &self.streaming_engine {
            se.write().await.shutdown().await?;
        }
        *self.state.write().await = EngineState::Shutdown;
        info!("Inference engine shutdown complete (drained {}/{} active)", total_active - self.active_requests.read().await.len(), total_active);
        Ok(())
    }

    /// Spawn a batch processing task with proper panic recovery and state tracking.
    async fn spawn_batch_processor(&self, batch: crate::batching::Batch) {
        let batch_semaphore = Arc::new(Semaphore::new(BATCH_CONCURRENCY_LIMIT));
        let engine = InferenceEngineHandle {
            scheduler: self.scheduler.clone(),
            model: Arc::clone(&self.model),
            tokenizer: self.tokenizer.clone(),
            state: self.state.clone(),
            use_gpu: self.config.use_gpu,
            #[cfg(feature = "gpu")]
            use_gpu_cache: self.config.use_gpu_cache,
            shared_paged: self.shared_paged.clone(),
            prefix_cache: self.prefix_cache.clone(),
            batch_semaphore,
            generation_timeout: Duration::from_secs(self.config.generation_timeout_seconds),
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
                InferenceError::EngineNotInitialized(
                    "Request receiver already consumed — engine already initialized".to_string(),
                )
            })?;

        let state = self.state.clone();
        let scheduler = self.scheduler.clone();
        let active_requests = self.active_requests.clone();
        let active_handle = self.active_requests.clone();
        let session_manager = self.session_manager.clone();

        let engine = InferenceEngineHandle {
            scheduler: self.scheduler.clone(),
            model: Arc::clone(&self.model),
            tokenizer: self.tokenizer.clone(),
            state: self.state.clone(),
            use_gpu: self.config.use_gpu,
            #[cfg(feature = "gpu")]
            use_gpu_cache: self.config.use_gpu_cache,
            shared_paged: self.shared_paged.clone(),
            prefix_cache: self.prefix_cache.clone(),
            batch_semaphore: Arc::new(Semaphore::new(BATCH_CONCURRENCY_LIMIT)),
            generation_timeout: Duration::from_secs(self.config.generation_timeout_seconds),
        };

        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);
        let recv_timeout = Duration::from_millis(self.config.recv_timeout_millis);

        let handle = tokio::spawn(async move {
            let mut last_cleanup = std::time::Instant::now();

            loop {
                if matches!(
                    *state.read().await,
                    EngineState::ShuttingDown | EngineState::Shutdown
                ) {
                    break;
                }

                if last_cleanup.elapsed() >= cleanup_interval {
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
                let request = tokio::time::timeout(recv_timeout, rx.recv()).await;

                match request {
                    Ok(Some(req)) => {
                        scheduler.write().await.add_to_batch_collector(&req).await;

                        if let Some(batch) = scheduler.read().await.pop_batch().await {
                            Self::spawn_batch_processor_inner(&engine, &active_requests, batch)
                                .await;
                        }
                    }
                    Ok(None) => break,
                    Err(elapsed) => {
                        tracing::debug!("Request poll timeout: {}", elapsed);
                        if let Some(batch) = scheduler.read().await.pop_batch().await {
                            Self::spawn_batch_processor_inner(&engine, &active_requests, batch)
                                .await;
                        }
                        continue;
                    }
                }
            }

            info!("Request processing loop ended");
        });

        // Track the main request loop handle for graceful shutdown
        {
            let mut ah = active_handle.write().await;
            ah.retain(|_, h| !h.is_finished());
            ah.insert(Uuid::nil(), handle);
        }

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
            shared_paged: engine.shared_paged.clone(),
            prefix_cache: engine.prefix_cache.clone(),
            batch_semaphore: engine.batch_semaphore.clone(),
            generation_timeout: engine.generation_timeout,
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

    async fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        match &self.tokenizer {
            Some(tok) => {
                let t = tok.lock().unwrap();
                Ok(t.encode(prompt))
            }
            None => Err(InferenceError::InvalidRequest(
                "No tokenizer configured".to_string(),
            )),
        }
    }

    async fn token_id_to_text(&self, token_id: u32) -> String {
        match &self.tokenizer {
            Some(tok) => {
                let guard = tok.lock().unwrap();
                guard.decode(&[token_id])
            }
            None => token_id_to_text_fallback(token_id),
        }
    }
}

fn token_id_to_text_fallback(token_id: u32) -> String {
    if token_id < 256 {
        char::from_u32(token_id).map_or_else(|| format!("[{}]", token_id), |c| c.to_string())
    } else {
        format!("[{}]", token_id)
    }
}

/// Core generation loop shared by single-request and batch-request paths.
/// If `prefix_len > 0`, skips prefill for the cached prefix tokens and
/// uses `prefix_logits` as the starting logits (warm-start).
/// Returns generated tokens and whether a timeout occurred.
fn run_generation_loop(
    model: &CausalLM,
    prompt_ids: &[u32],
    max_gen: usize,
    sampler: &mut crate::sampler::Sampler,
    kv_state: &mut dyn KVCacheProvider,
    tokenizer: Option<&Arc<StdMutex<BpeTokenizer>>>,
    prefix_len: usize,
    prefix_logits: Vec<f32>,
    generation_timeout: Duration,
    include_kv_cache: bool,
) -> (Vec<GeneratedToken>, Vec<f32>, bool, Vec<KVCacheEntry>) {
    let start = std::time::Instant::now();
    let mut all_ids = Vec::with_capacity(prompt_ids.len() + max_gen);
    all_ids.extend_from_slice(prompt_ids);
    let mut tokens = Vec::with_capacity(max_gen);
    let vocab_size = model.config.vocab_size;
    let mut prefix_logits = prefix_logits;
    let use_prefix = prefix_len > 0 && !prefix_logits.is_empty();
    let mut last_logits: Vec<f32> = if use_prefix {
        prefix_logits.clone()
    } else {
        Vec::with_capacity(vocab_size)
    };

    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    for pos in 0..max_gen {
        let token_id: u32;
        let log_prob: f32;
        let mut used_gpu = false;

        if pos == 0 && use_prefix {
            if prefix_len >= prompt_ids.len() {
                let arr = Array1::from_vec(std::mem::take(&mut prefix_logits));
                let logits_slice = arr.as_slice().unwrap_or(&[]);
                let tok = sampler.sample(logits_slice).unwrap_or(0) as u32;
                let lp = logits_slice
                    .get(tok as usize)
                    .copied()
                    .map(|x| (x.max(1e-10)).ln())
                    .unwrap_or(0.0);
                token_id = tok;
                log_prob = lp;
                if logits_slice.len() == vocab_size && last_logits.len() == vocab_size {
                    last_logits.copy_from_slice(logits_slice);
                } else {
                    last_logits = logits_slice.to_vec();
                }
            } else {
                let input = &prompt_ids[prefix_len..];
                if let Some(tok) = ModelForward::forward_sample_gpu(
                    model,
                    input,
                    kv_state,
                    sampler.config().temperature,
                    sampler.config().top_k,
                    sampler.config().top_p,
                    base_seed.wrapping_add(pos as u64),
                ) {
                    token_id = tok;
                    log_prob = 0.0;
                    used_gpu = true;
                } else {
                    let arr = ModelForward::forward(model, input, kv_state);
                    let logits_slice = arr.as_slice().unwrap_or(&[]);
                    let tok = sampler.sample(logits_slice).unwrap_or(0) as u32;
                    let lp = logits_slice
                        .get(tok as usize)
                        .copied()
                        .map(|x| (x.max(1e-10)).ln())
                        .unwrap_or(0.0);
                    token_id = tok;
                    log_prob = lp;
                    if logits_slice.len() == vocab_size && last_logits.len() == vocab_size {
                        last_logits.copy_from_slice(logits_slice);
                    } else {
                        last_logits = logits_slice.to_vec();
                    }
                }
            }
        } else {
            let input = if pos == 0 {
                prompt_ids
            } else {
                core::slice::from_ref(all_ids.last().unwrap_or(&0))
            };
            if let Some(tok) = ModelForward::forward_sample_gpu(
                model,
                input,
                kv_state,
                sampler.config().temperature,
                sampler.config().top_k,
                sampler.config().top_p,
                base_seed.wrapping_add(pos as u64),
            ) {
                token_id = tok;
                log_prob = 0.0;
                used_gpu = true;
            } else {
                let arr = ModelForward::forward(model, input, kv_state);
                let logits_slice = arr.as_slice().unwrap_or(&[]);
                let tok = sampler.sample(logits_slice).unwrap_or(0) as u32;
                let lp = logits_slice
                    .get(tok as usize)
                    .copied()
                    .map(|x| (x.max(1e-10)).ln())
                    .unwrap_or(0.0);
                token_id = tok;
                log_prob = lp;
                if logits_slice.len() == vocab_size && last_logits.len() == vocab_size {
                    last_logits.copy_from_slice(logits_slice);
                } else {
                    last_logits = logits_slice.to_vec();
                }
            }
        }

        let token_text: String = match tokenizer {
            Some(tok) => {
                let t = tok.lock().unwrap();
                t.decode(&[token_id])
            }
            None => token_id_to_text_fallback(token_id),
        };

        tokens.push(GeneratedToken::new(token_id, token_text, log_prob, pos));
        if used_gpu {
            crate::inference_trait::GPU_TOKENS_GENERATED
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            crate::inference_trait::CPU_TOKENS_GENERATED
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        all_ids.push(token_id);

        if token_id == 0 || start.elapsed() > generation_timeout {
            let timed_out = start.elapsed() > generation_timeout;
            let kvcache = if include_kv_cache {
                kv_state
                    .as_cpu_entries()
                    .map(|e| e.clone())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            return (tokens, last_logits, timed_out, kvcache);
        }
    }

    let kvcache = if include_kv_cache {
        kv_state
            .as_cpu_entries()
            .map(|e| e.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    (tokens, last_logits, false, kvcache)
}

const BATCH_CONCURRENCY_LIMIT: usize = 64;

/// Handle used inside spawned tasks to avoid borrowing self
struct InferenceEngineHandle {
    scheduler: Arc<RwLock<RequestScheduler>>,
    model: Arc<CausalLM>,
    tokenizer: Option<Arc<StdMutex<BpeTokenizer>>>,
    state: Arc<RwLock<EngineState>>,
    use_gpu: bool,
    #[cfg(feature = "gpu")]
    use_gpu_cache: bool,
    shared_paged: Option<Arc<StdMutex<PagedKVCache>>>,
    prefix_cache: Arc<PrefixCache>,
    batch_semaphore: Arc<Semaphore>,
    generation_timeout: Duration,
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
            if let Err(e) = self.scheduler.write().await.complete_batch(&batch).await {
                warn!(
                    "Failed to complete batch {} during shutdown: {}",
                    batch.batch_id, e
                );
            }
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
            let prefix_cache = self.prefix_cache.clone();
            let use_gpu = self.use_gpu;
            #[cfg(feature = "gpu")]
            let use_gpu_cache = self.use_gpu_cache;
            let shared_paged = self.shared_paged.clone();
            let generation_timeout = self.generation_timeout;
            let _permit = match self.batch_semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => break,
            };

            let handle = tokio::spawn(async move {
                let __permit = _permit;
                let start = std::time::Instant::now();
                let mut response = InferenceResponse::new(breq.request_id);

                if breq.prompt.is_empty() {
                    let err_resp = response
                        .with_finish_reason(FinishReason::Error("Empty prompt".to_string()))
                        .with_inference_time(start.elapsed().as_millis() as u64);
                    if let Err(e) = scheduler
                        .read()
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
                        let t = tok.lock().unwrap();
                        t.encode(&breq.prompt)
                    }
                    None => {
                        warn!(
                            "No tokenizer configured for batch request {}",
                            breq.request_id
                        );
                        let err_resp = response
                            .with_finish_reason(FinishReason::Error(
                                "No tokenizer configured".to_string(),
                            ))
                            .with_inference_time(start.elapsed().as_millis() as u64);
                        if let Err(e) = scheduler
                            .read()
                            .await
                            .send_response(breq.request_id, err_resp)
                            .await
                        {
                            warn!(
                                "Failed to send tokenizer error to {}: {}",
                                breq.request_id, e
                            );
                        }
                        return;
                    }
                };

                // Check prefix cache for warm-start opportunity
                let prefix_match = prefix_cache.match_prefix(&prompt_ids).await;
                let prefix_len = prefix_match.prefix_len;
                let prefix_logits = prefix_match.cached_value;
                let prefix_kv_cache = prefix_match.cached_kv_cache;
                if prefix_len > 0 {
                    debug!(
                        "Prefix cache hit: {}/{} tokens matched, skipping prefill",
                        prefix_len,
                        prompt_ids.len()
                    );
                }

                let mut kv_state = InferenceEngine::make_kv_provider(&model, use_gpu_cache, &shared_paged);

                // Restore KV cache from prefix match if available
                if !prefix_kv_cache.is_empty() {
                    if let Some(cpu_entries) = kv_state.as_cpu_entries() {
                        *cpu_entries = prefix_kv_cache;
                        debug!("Restored KV cache for {} matched prefix tokens", prefix_len);
                    }
                }
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
                let gen_result = tokio::task::spawn_blocking(move || {
                    run_generation_loop(
                        &model_c,
                        &prompt_ids_c,
                        max_gen,
                        &mut sampler,
                        &mut *kv_state,
                        tokenizer.as_ref(),
                        prefix_len,
                        prefix_logits,
                        generation_timeout,
                        false,
                    )
                })
                .await;

                let (tokens, _last_logits, timed_out, _batch_kv_cache) = match gen_result {
                    Ok(r) => r,
                    Err(e) => {
                        error!("Batch generation loop panicked: {:?}", e);
                        let err_resp = response
                            .with_finish_reason(FinishReason::Error(format!(
                                "Generation panicked: {:?}",
                                e
                            )))
                            .with_inference_time(start.elapsed().as_millis() as u64);
                        if let Err(e) = scheduler
                            .read()
                            .await
                            .send_response(breq.request_id, err_resp)
                            .await
                        {
                            warn!("Failed to send panic error response: {}", e);
                        }
                        return;
                    }
                };

                for token in tokens {
                    response.add_token(token);
                }

                if timed_out {
                    response.finish_reason = FinishReason::Timeout;
                } else {
                    response.finish_reason = FinishReason::MaxTokens;
                }
                response.inference_time_ms = start.elapsed().as_millis() as u64;

                let rid = breq.request_id;
                let response = Arc::new(response);
                if let Err(e) = RetryConfig::default()
                    .retry(|| {
                        let scheduler = scheduler.clone();
                        let response = Arc::clone(&response);
                        async move {
                            scheduler
                                .read()
                                .await
                                .send_response(rid, (*response).clone())
                                .await
                        }
                    })
                    .await
                {
                    error!("Failed to send batch response for {}: {}", rid, e);
                }
                // Gempa 17: cleanup active request tracking after response sent
                scheduler.write().await.complete_request(rid).await;
            });
            handles.push(handle);
        }

        // Wait for all parallel tasks to complete
        for handle in handles {
            if let Err(e) = handle.await {
                error!("Parallel task in batch {} panicked: {:?}", batch_id, e);
            }
        }

        if let Err(e) = self
            .scheduler
            .write()
            .await
            .complete_batch_id(batch_id)
            .await
        {
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
