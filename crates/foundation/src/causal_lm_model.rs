use async_trait::async_trait;
use ndarray::Array1;
use rand::seq::SliceRandom;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn};
use unicode_segmentation::UnicodeSegmentation;

use nexora_training::{Trainer, TrainerConfig};
use nexora_transformer::{CausalLM, TransformerConfig};

use crate::echo_net_injector::EchoNetInjector;
use crate::shared::base_model::ValidationResult;
use crate::shared::{
    CapabilityVector, FinishReason, GenerationMetadata, InputData, ModelMeta, ModelStatistics,
    ModelTier, NxrInput, NxrModel, NxrModelError, NxrModelId, NxrModelResult, NxrOutput,
    NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage, StreamChunkData, TokenOutput,
};

/// Byte + BPE tokenizer: bytes 0-255 as base (BOS=1, EOS=2), plus optional learned merge rules for IDs ≥ 256.
#[derive(Clone)]
pub struct MiniTokenizer {
    vocab_size: usize,
    bpe_token_to_id: HashMap<String, u32>,
    bpe_id_to_token: HashMap<u32, String>,
    merges: Vec<(String, String)>,
}

impl MiniTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            bpe_token_to_id: HashMap::new(),
            bpe_id_to_token: HashMap::new(),
            merges: Vec::new(),
        }
    }

    /// Byte-level encode (backward compatible): [BOS=1, byte0, byte1, ..., EOS=2]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![1u32];
        for &b in text.as_bytes() {
            let id = b as u32;
            if (id as usize) < self.vocab_size {
                ids.push(id);
            }
        }
        ids.push(2);
        ids
    }

    /// Decode: reconstruct string from token IDs.
    /// Byte IDs (0-255) are converted directly.
    /// BPE token IDs (≥256) are looked up in the BPE vocab.
    /// BOS(1) and EOS(2) are skipped in output.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in token_ids {
            if id == 1 || id == 2 {
                continue;
            }
            if id < 256 {
                result.push(char::from(id as u8));
            } else if let Some(s) = self.bpe_id_to_token.get(&id) {
                result.push_str(s);
            }
        }
        result
    }

    /// Train BPE merges on provided texts. This extends the vocab with merged tokens
    /// that get IDs ≥ 256.
    pub fn train_bpe(&mut self, texts: &[String], num_merges: usize) {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for text in texts {
            let graphemes: Vec<String> = text.graphemes(true).map(|g| g.to_string()).collect();
            for pair in graphemes.windows(2) {
                *pair_counts
                    .entry((pair[0].clone(), pair[1].clone()))
                    .or_default() += 1;
            }
        }
        let max_merges = num_merges.min(self.vocab_size.saturating_sub(256));
        for _ in 0..max_merges {
            let best = pair_counts.iter().max_by_key(|&(_, &c)| c).map(|(k, _)| k.clone());
            let Some((left, right)) = best else { break };
            let merged = format!("{}{}", left, right);
            let id = 256u32 + self.merges.len() as u32;
            if id >= self.vocab_size as u32 {
                break;
            }
            self.bpe_token_to_id.insert(merged.clone(), id);
            self.bpe_id_to_token.insert(id, merged.clone());
            self.merges.push((left.clone(), right.clone()));
            pair_counts.remove(&(left, right));
        }
    }

    /// BPE-aware tokenization: applies learned merge rules then encodes.
    /// Falls back to byte-level for unknown character sequences.
    pub fn bpe_tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![1u32];
        let mut tokens: Vec<String> = text.graphemes(true).map(|g| g.to_string()).collect();
        if !self.merges.is_empty() {
            loop {
                let mut best_i = None;
                for i in 0..tokens.len().saturating_sub(1) {
                    let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                    if self.bpe_token_to_id.contains_key(&merged) {
                        best_i = Some(i);
                        break;
                    }
                }
                match best_i {
                    Some(i) => {
                        let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                        tokens[i] = merged;
                        tokens.remove(i + 1);
                    }
                    None => break,
                }
            }
        }
        for token in &tokens {
            if let Some(&id) = self.bpe_token_to_id.get(token.as_str()) {
                ids.push(id);
            } else {
                for &b in token.as_bytes() {
                    if (b as usize) < self.vocab_size {
                        ids.push(b as u32);
                    }
                }
            }
        }
        ids.push(2);
        ids
    }
}

/// Configuration for EchoNet injection into the transformer pipeline.
#[derive(Debug, Clone)]
pub struct EchoNetInjectionConfig {
    /// Which layer index to inject after (e.g. 2 = after layer 2).
    pub inject_after_layer: usize,
    /// APSS phase separation strength.
    pub phase_separation_strength: f32,
    /// Number of recent tokens to keep in the sliding window for APSS.
    pub max_window: usize,
    /// Amplitude modulation factor: h' = h * (1 + alpha * tanh(phase_delta)).
    pub alpha: f32,
}

impl Default for EchoNetInjectionConfig {
    fn default() -> Self {
        Self {
            inject_after_layer: 2,
            phase_separation_strength: 0.1,
            max_window: 64,
            alpha: 0.01,
        }
    }
}

pub struct CausalLmModel {
    meta: ModelMeta,
    capabilities: CapabilityVector,
    config: Value,
    model: Arc<RwLock<Option<CausalLM>>>,
    tokenizer: Arc<RwLock<Option<MiniTokenizer>>>,
    transformer_config: Arc<RwLock<TransformerConfig>>,
    initialized: Arc<RwLock<bool>>,
    statistics: Arc<RwLock<ModelStatistics>>,
    use_gpu: AtomicBool,
    echo_net_config: Option<EchoNetInjectionConfig>,
    sedc_enabled: bool,
    sedc_report: Arc<RwLock<Option<String>>>,
}

impl CausalLmModel {
    pub fn new(model_id: NxrModelId, transformer_config: TransformerConfig) -> Self {
        let meta = ModelMeta::new(
            model_id,
            ModelTier::Core,
            "0.1.0".to_string(),
            model_id.fullname().to_string(),
        );
        let capabilities = CapabilityVector::new(model_id);
        let cfg = serde_json::json!({
            "transformer_config": {
                "vocab_size": transformer_config.vocab_size,
                "hidden_size": transformer_config.hidden_size,
                "num_heads": transformer_config.num_heads,
                "num_kv_heads": transformer_config.num_kv_heads,
                "num_layers": transformer_config.num_layers,
                "max_seq_len": transformer_config.max_seq_len,
            }
        });

        Self {
            meta,
            capabilities,
            config: cfg,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            transformer_config: Arc::new(RwLock::new(transformer_config)),
            initialized: Arc::new(RwLock::new(false)),
            statistics: Arc::new(RwLock::new(ModelStatistics::default())),
            use_gpu: AtomicBool::new(true),
            echo_net_config: None,
            sedc_enabled: false,
            sedc_report: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_echo_net(mut self, config: EchoNetInjectionConfig) -> Self {
        self.echo_net_config = Some(config);
        self
    }

    pub fn with_sedc(mut self) -> Self {
        self.sedc_enabled = true;
        self
    }

    pub async fn load_model(&self) -> NxrModelResult<()> {
        let tc = self.transformer_config.read().await.clone();
        let mut model = CausalLM::new(tc.clone());

        // Attach EchoNet injector if configured
        if let Some(echo_cfg) = &self.echo_net_config {
            let injector = EchoNetInjector::new(
                tc.hidden_size,
                echo_cfg.phase_separation_strength,
                echo_cfg.max_window,
                echo_cfg.alpha,
            )
            .map_err(|e| NxrModelError::Internal(format!("EchoNet injector: {}", e)))?;

            model.injectors.push((
                echo_cfg.inject_after_layer,
                std::sync::Mutex::new(Box::new(injector)),
            ));

            info!(
                "EchoNet APSS injector attached after layer {}",
                echo_cfg.inject_after_layer
            );
        }

        // Run SEDC compression if enabled
        if self.sedc_enabled {
            match model.compress_sedc_default() {
                Ok(Some(report)) => {
                    let report_str = serde_json::to_string_pretty(&report).unwrap_or_default();
                    *self.sedc_report.write().await = Some(report_str);
                    info!("SEDC compression stored in model ✓");
                }
                Ok(None) => {
                    info!("SEDC compression skipped");
                }
                Err(e) => {
                    warn!("SEDC compression failed: {}", e);
                }
            }
        }

        *self.model.write().await = Some(model);
        *self.initialized.write().await = true;
        info!("CausalLM model loaded: {} params", tc.parameter_count());
        Ok(())
    }

    pub async fn load_tokenizer(&self, tokenizer: MiniTokenizer) {
        *self.tokenizer.write().await = Some(tokenizer);
    }

    pub async fn param_count(&self) -> usize {
        let tc = self.transformer_config.read().await;
        tc.parameter_count()
    }

    pub async fn save_checkpoint(&self, path: &str) -> NxrModelResult<()> {
        let model = self.model.read().await;
        let model_ref = model
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        use nexora_transformer::TrainableCausalLM;
        TrainableCausalLM::from_inference(model_ref)
            .save_checkpoint(path)
            .map_err(|e| NxrModelError::Inference(format!("Save failed: {}", e)))?;
        info!("Checkpoint saved to {}", path);
        Ok(())
    }

    pub async fn load_checkpoint(&self, path: &str) -> NxrModelResult<()> {
        let mut model = self.model.write().await;
        let model_ref = model
            .as_mut()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        use nexora_transformer::TrainableCausalLM;
        TrainableCausalLM::load_checkpoint(model_ref, path)
            .map_err(|e| NxrModelError::Inference(format!("Load failed: {}", e)))?;
        info!("Checkpoint loaded from {}", path);
        Ok(())
    }

    pub fn set_use_gpu(&self, enabled: bool) {
        self.use_gpu.store(enabled, Ordering::Relaxed);
    }

    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> NxrModelResult<String> {
        self.generate_text_with_gpu(prompt, max_tokens, temperature, self.use_gpu.load(Ordering::Relaxed))
            .await
    }

    pub async fn generate_text_with_gpu(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        use_gpu: bool,
    ) -> NxrModelResult<String> {
        let tok = {
            let tokenizer = self.tokenizer.read().await;
            tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?
                .clone()
        };
        let input_ids = tok.encode(prompt);

        let model_arc = self.model.clone();
        let (output_ids, _) = tokio::task::spawn_blocking(move || {
            let guard = model_arc.blocking_read();
            let m = guard.as_ref().ok_or_else(|| {
                NxrModelError::NotInitialized("Model not loaded".to_string())
            })?;
            Ok::<_, NxrModelError>(m.generate_with_gpu(
                &input_ids,
                max_tokens,
                temperature,
                50,
                use_gpu,
            ))
        })
        .await
        .map_err(|e| NxrModelError::Internal(e.to_string()))??;

        let tokenizer = self.tokenizer.read().await;
        let tok = tokenizer
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
        let text = tok.decode(&output_ids);
        Ok(text)
    }

    pub async fn train_on_data(
        &self,
        data: &[String],
        cfg: TrainerConfig,
        val_data: Option<&[String]>,
    ) -> NxrModelResult<TrainingReport> {
        let seq_length = cfg.seq_length;
        let batch_size = cfg.batch_size;
        let max_steps = cfg.max_steps;

        let (train_tokens, val_tokens) = {
            let tokenizer = self.tokenizer.read().await;
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;

            let train_tokens: Vec<Vec<u32>> = data
                .iter()
                .filter_map(|t| {
                    let ids = tok.encode(t);
                    if ids.len() >= 2 {
                        Some(ids)
                    } else {
                        None
                    }
                })
                .collect();

            let val_tokens: Vec<Vec<u32>> = match val_data {
                Some(vd) => vd
                    .iter()
                    .filter_map(|t| {
                        let ids = tok.encode(t);
                        if ids.len() >= 2 {
                            Some(ids)
                        } else {
                            None
                        }
                    })
                    .collect(),
                None => Vec::new(),
            };

            (train_tokens, val_tokens)
        };

        let total_tokens: usize = train_tokens.iter().map(|t| t.len()).sum();
        info!(
            "Training data: {} sequences, {} tokens",
            train_tokens.len(),
            total_tokens
        );

        let mut model = self.model.write().await;
        let causal_lm = model
            .take()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        drop(model);

        let mut trainer_cfg = cfg;
        trainer_cfg.save_every = 100;
        trainer_cfg.report_every = 1;
        trainer_cfg.val_every_steps = (max_steps / 5).max(1);

        let output_base = trainer_cfg.save_path.clone();

        let mut trainer = Trainer::with_model(causal_lm, trainer_cfg);
        trainer.prepare();
        trainer.try_restore_optimizer();

        let start = Instant::now();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut best_val_loss = f64::MAX;
        let mut early_stop_counter = 0;

        while trainer.step < max_steps {
            let mut epoch: Vec<&Vec<u32>> = train_tokens.iter().collect();
            epoch.shuffle(&mut rng);

            for sample in &epoch {
                if trainer.step >= max_steps {
                    break;
                }
                for chunk in sample.chunks(seq_length + 1) {
                    if chunk.len() < 2 {
                        continue;
                    }
                    let (input, target) = trainer.prepare_batch(chunk);
                    if input.is_empty() {
                        continue;
                    }
                    if let Some(loss) = trainer.train_batch(&input, &target) {
                        let step = trainer.step;
                        if step % trainer.config.report_every == 0 {
                            let lr = trainer.optimizer.as_ref().map(|o| o.lr).unwrap_or(0.0);
                            info!("  Step {} | loss: {:.4} | lr: {:.2e}", step, loss, lr);
                        }
                        if let Some(ref base) = output_base {
                            if step % 1000 == 0 && step > 0 {
                                let save_path = format!("{}.safetensors", base);
                                if let Err(e) = trainer.save(&save_path) {
                                    warn!("Failed to save checkpoint at step {}: {}", step, e);
                                } else {
                                    info!("  Checkpoint overwritten at step {}", step);
                                }
                            }
                        }
                    }
                    if trainer.step >= max_steps {
                        break;
                    }
                }
            }

            if !val_tokens.is_empty()
                && trainer.step % trainer.config.val_every_steps == 0
                && trainer.step > 0
            {
                let val_metrics = trainer.evaluate_loss(&val_tokens, seq_length);
                info!("  Validation | step {} | loss: {:.4} | perplexity: {:.2}", trainer.step, val_metrics.avg_loss, val_metrics.perplexity);
                if val_metrics.avg_loss < best_val_loss {
                    best_val_loss = val_metrics.avg_loss;
                    early_stop_counter = 0;
                } else {
                    early_stop_counter += 1;
                    if early_stop_counter >= 3 && trainer.step > max_steps / 2 {
                        info!("  Early stopping triggered at step {}", trainer.step);
                        break;
                    }
                }
            }
        }

        if output_base.is_some() {
            trainer.save_checkpoint();
        }

        trainer.sync_weights();
        let trained_model = trainer.model.clone();
        let final_loss = if trainer.step > 0 {
            trainer.avg_loss()
        } else {
            0.0
        };
        let elapsed = start.elapsed();

        let val_loss = if !val_tokens.is_empty() {
            Some(trainer.evaluate_loss(&val_tokens, seq_length))
        } else {
            None
        };

        *self.model.write().await = Some(trained_model);

        Ok(TrainingReport {
            steps: trainer.step,
            final_loss: final_loss as f64,
            total_tokens: trainer.total_tokens,
            duration_secs: elapsed.as_secs_f64(),
            val_loss: val_loss.as_ref().map(|m| m.avg_loss),
            val_perplexity: val_loss.as_ref().map(|m| m.perplexity),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub steps: usize,
    pub final_loss: f64,
    pub total_tokens: usize,
    pub duration_secs: f64,
    pub val_loss: Option<f64>,
    pub val_perplexity: Option<f64>,
}

#[async_trait]
impl NxrModel for CausalLmModel {
    type Config = Value;
    type Metrics = Value;
    type State = Value;

    fn identity(&self) -> &ModelMeta {
        &self.meta
    }

    fn capabilities(&self) -> &CapabilityVector {
        &self.capabilities
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    async fn state(&self) -> NxrModelResult<Self::State> {
        let initialized = *self.initialized.read().await;
        Ok(serde_json::json!({
            "initialized": initialized,
            "has_model": self.model.read().await.is_some(),
            "has_tokenizer": self.tokenizer.read().await.is_some(),
        }))
    }

    async fn initialize(&mut self, config: Self::Config) -> NxrModelResult<()> {
        if let Some(tc) = config.get("transformer_config") {
            if let (Some(vs), Some(hs), Some(nh), Some(nl), Some(ms)) = (
                tc.get("vocab_size").and_then(|v| v.as_u64()),
                tc.get("hidden_size").and_then(|v| v.as_u64()),
                tc.get("num_heads").and_then(|v| v.as_u64()),
                tc.get("num_layers").and_then(|v| v.as_u64()),
                tc.get("max_seq_len").and_then(|v| v.as_u64()),
            ) {
                let nkv = tc
                    .get("num_kv_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(nh / 4);
                let int_size = tc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(hs * 4);
                let new_tc = TransformerConfig {
                    vocab_size: vs as usize,
                    hidden_size: hs as usize,
                    num_heads: nh as usize,
                    num_kv_heads: nkv as usize,
                    num_layers: nl as usize,
                    max_seq_len: ms as usize,
                    intermediate_size: int_size as usize,
                    ..Default::default()
                };
                *self.transformer_config.write().await = new_tc;
            }
        }
        self.config = config;
        self.load_model().await?;
        Ok(())
    }

    async fn reset(&self) -> NxrModelResult<()> {
        *self.model.write().await = None;
        *self.tokenizer.write().await = None;
        *self.initialized.write().await = false;
        Ok(())
    }

    async fn metrics(&self) -> NxrModelResult<Self::Metrics> {
        let stats = self.statistics.read().await.clone();
        Ok(serde_json::to_value(stats).unwrap_or_default())
    }

    async fn infer(&self, input: &NxrInput) -> NxrModelResult<NxrOutput> {
        let tok = {
            let tokenizer = self.tokenizer.read().await;
            tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?
                .clone()
        };

        let text = match &input.data {
            InputData::Text(t) => t.clone(),
            InputData::Tokens(t) => {
                let decoded = tok.decode(t);
                let elapsed = std::time::Instant::now();
                let total = t.len() as u64;
                let dur = elapsed.elapsed().as_millis() as u64;
                let tp_s = if dur > 0 { total as f32 / (dur as f32 / 1000.0) } else { 0.0 };
                return Ok(NxrOutput {
                    id: uuid::Uuid::new_v4(),
                    input_id: input.id,
                    timestamp: chrono::Utc::now(),
                    data: OutputData::Tokens(
                        t.iter()
                            .enumerate()
                            .map(|(i, &tid)| {
                                let t = tok.decode(&[tid]);
                                TokenOutput {
                                    token_id: tid,
                                    text: t,
                                    log_prob: 0.0,
                                    position: i,
                                }
                            })
                            .collect(),
                    ),
                    metadata: GenerationMetadata {
                        finish_reason: FinishReason::MaxTokens,
                        total_tokens: t.len(),
                        generation_time_ms: dur,
                        model_version: "0.1.0".to_string(),
                        seed: None,
                        extras: Default::default(),
                    },
                    performance: PerformanceMetrics {
                        tokens_per_second: tp_s,
                        memory_usage_gb: 0.0,
                        gpu_utilization: None,
                        cpu_utilization: 0.0,
                        network_usage_mbps: None,
                    },
                });
            }
            _ => {
                return Err(NxrModelError::Inference(
                    "Unsupported input type".to_string(),
                ))
            }
        };

        let max_tokens = input
            .parameters
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        let temperature = input
            .parameters
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        let top_k = input
            .parameters
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let input_ids = tok.encode(&text);
        let start = std::time::Instant::now();

        let model_arc = self.model.clone();
        let use_gpu = self.use_gpu.load(Ordering::Relaxed);
        let (output_ids, _cache) = tokio::task::spawn_blocking(move || {
            let guard = model_arc.blocking_read();
            let m = guard.as_ref().ok_or_else(|| {
                NxrModelError::NotInitialized("Model was reset before generation".to_string())
            })?;
            Ok::<_, NxrModelError>(m.generate_with_gpu(
                &input_ids,
                max_tokens,
                temperature,
                top_k,
                use_gpu,
            ))
        })
        .await
        .map_err(|e| NxrModelError::Internal(e.to_string()))??;

        let elapsed = start.elapsed().as_millis() as u64;
        let generated_text = tok.decode(&output_ids);

        let mut stats = self.statistics.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        stats.total_tokens_generated += output_ids.len() as u64;

        let tokens_per_second = if elapsed > 0 {
            output_ids.len() as f32 / (elapsed as f32 / 1000.0)
        } else {
            0.0
        };

        Ok(NxrOutput {
            id: uuid::Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: OutputData::Text(generated_text),
            metadata: GenerationMetadata {
                finish_reason: FinishReason::MaxTokens,
                total_tokens: output_ids.len(),
                generation_time_ms: elapsed,
                model_version: "0.1.0".to_string(),
                seed: None,
                extras: Default::default(),
            },
            performance: PerformanceMetrics {
                tokens_per_second,
                memory_usage_gb: 0.0,
                gpu_utilization: None,
                cpu_utilization: 0.0,
                network_usage_mbps: None,
            },
        })
    }

    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> NxrModelResult<()> {
        let text = match &input.data {
            InputData::Text(t) => t.clone(),
            InputData::Tokens(t) => {
                let decoded = {
                    let tokenizer = self.tokenizer.read().await;
                    let tok = tokenizer
                        .as_ref()
                        .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
                    tok.decode(t)
                };
                callback(NxrStreamChunk {
                    id: uuid::Uuid::new_v4(),
                    input_id: input.id,
                    timestamp: chrono::Utc::now(),
                    data: StreamChunkData::TextDelta(decoded),
                    is_final: true,
                });
                return Ok(());
            }
            _ => {
                return Err(NxrModelError::Inference(
                    "Unsupported input type".to_string(),
                ))
            }
        };

        let max_tokens = input
            .parameters
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;
        let temperature = input
            .parameters
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        let top_k = input
            .parameters
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let (input_ids, _tok) = {
            let tokenizer = self.tokenizer.read().await;
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
            (tok.encode(&text), ())
        };
        let use_gpu = self.use_gpu.load(Ordering::Relaxed);

        let model_arc = self.model.clone();
        let (mut cache, mut last_id) = tokio::task::spawn_blocking(move || {
            let guard = model_arc.blocking_read();
            let m = guard.as_ref().ok_or_else(|| {
                NxrModelError::NotInitialized("Model not loaded".to_string())
            })?;
            let mut c = m.reset_cache();
            for &tid in &input_ids {
                let _ = m.forward(&[tid], &mut c);
            }
            let lid = *input_ids.last().unwrap_or(&0);
            Ok::<_, NxrModelError>((c, lid))
        })
        .await
        .map_err(|e| NxrModelError::Internal(e.to_string()))??;

        let tokenizer = self.tokenizer.read().await;
        let tok = tokenizer
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
        for _ in 0..max_tokens {
            let model = self.model.read().await;
            let m = model
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
            let vs = m.config.vocab_size;
            let logits = m
                .forward(&[last_id], &mut cache)
                .unwrap_or_else(|_| Array1::zeros(vs));
            drop(model);
            let next_id = nexora_transformer::sample_token(&logits, temperature, top_k);
            let token_text = tok.decode(&[next_id]);
            callback(NxrStreamChunk {
                id: uuid::Uuid::new_v4(),
                input_id: input.id,
                timestamp: chrono::Utc::now(),
                data: StreamChunkData::TextDelta(token_text),
                is_final: false,
            });
            if next_id == 0 || next_id == 2 {
                break;
            }
            last_id = next_id;
        }
        callback(NxrStreamChunk {
            id: uuid::Uuid::new_v4(),
            input_id: input.id,
            timestamp: chrono::Utc::now(),
            data: StreamChunkData::Status("done".to_string()),
            is_final: true,
        });
        Ok(())
    }

    async fn update_config(&mut self, config: Self::Config) -> NxrModelResult<()> {
        self.config = config;
        Ok(())
    }

    async fn validate(&self) -> NxrModelResult<ValidationResult> {
        let initialized = *self.initialized.read().await;
        let has_model = self.model.read().await.is_some();
        let has_tok = self.tokenizer.read().await.is_some();
        let mut errors = Vec::new();
        if !initialized {
            errors.push("Not initialized".to_string());
        }
        if !has_model {
            errors.push("Model not loaded".to_string());
        }
        if !has_tok {
            errors.push("Tokenizer not loaded".to_string());
        }
        Ok(ValidationResult {
            is_valid: initialized && has_model && has_tok,
            errors,
            warnings: vec![],
            score: if initialized && has_model && has_tok {
                1.0
            } else {
                0.0
            },
        })
    }

    async fn statistics(&self) -> NxrModelResult<ModelStatistics> {
        Ok(self.statistics.read().await.clone())
    }

    async fn is_ready(&self) -> bool {
        let (init, has_m, has_t) = (
            *self.initialized.read().await,
            self.model.read().await.is_some(),
            self.tokenizer.read().await.is_some(),
        );
        init && has_m && has_t
    }

    async fn resource_usage(&self) -> NxrModelResult<ResourceUsage> {
        let memory_gb = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<f64>().ok())
                    .map(|kb| (kb / 1_048_576.0) as f32)
            })
            .unwrap_or(0.0);

        let cpu_percent = std::fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|stat| {
                let parts: Vec<&str> = stat.split_whitespace().collect();
                if parts.len() >= 15 {
                    let utime: f64 = parts[13].parse().unwrap_or(0.0);
                    let stime: f64 = parts[14].parse().unwrap_or(0.0);
                    let total_ticks = utime + stime;
                    let uptime = std::fs::read_to_string("/proc/uptime")
                        .ok()
                        .and_then(|u| u.split_whitespace().next()?.parse::<f64>().ok())
                        .unwrap_or(1.0);
                    let hertz = 100.0;
                    Some(((total_ticks / hertz / uptime) * 100.0) as f32)
                } else {
                    None
                }
            })
            .unwrap_or(0.0);

        let use_gpu = self.use_gpu.load(Ordering::Relaxed);
        // Iterate over multiple GPU status files (multi-GPU support).
        // Returns the average utilization across all available GPUs.
        let gpu_percent = if use_gpu {
            let total: f32 = (0..8)
                .filter_map(|i| {
                    let path = format!("/proc/driver/nvidia/gpus/{i}/status");
                    std::fs::read_to_string(&path).ok().and_then(|s| {
                        s.lines()
                            .find(|l| l.contains("GPU Utilization"))
                            .and_then(|l| {
                                l.split_whitespace()
                                    .find(|w| w.ends_with('%'))
                                    .and_then(|w| w.trim_end_matches('%').parse::<f32>().ok())
                            })
                    })
                })
                .sum();
            let count = (0..8)
                .filter(|i| {
                    let path = format!("/proc/driver/nvidia/gpus/{i}/status");
                    std::path::Path::new(&path).exists()
                })
                .count() as f32;
            if count > 0.0 {
                Some(total / count)
            } else {
                None
            }
        } else {
            None
        };

        Ok(ResourceUsage {
            memory_gb,
            cpu_percent,
            gpu_percent,
            gpu_memory_gb: None,
            disk_gb: 0.0,
            network_mbps: 0.0,
            active_connections: 0,
            queue_size: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::NxrModelId;

    // ── MiniTokenizer tests ──

    #[test]
    fn test_mini_tokenizer_encode_has_bos_eos() {
        let tok = MiniTokenizer::new(512);
        let ids = tok.encode("hello");
        assert_eq!(ids.first(), Some(&1), "BOS token must be 1");
        assert_eq!(ids.last(), Some(&2), "EOS token must be 2");
        assert!(ids.len() >= 3, "Should have BOS + content + EOS");
    }

    #[test]
    fn test_mini_tokenizer_encode_decode_roundtrip() {
        let tok = MiniTokenizer::new(512);
        let text = "Hello, World!";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(
            text.as_bytes(),
            &decoded.as_bytes()[1..decoded.len() - 1],
            "Roundtrip should preserve text (minus BOS/EOS)"
        );
    }

    #[test]
    fn test_mini_tokenizer_empty_input() {
        let tok = MiniTokenizer::new(512);
        let ids = tok.encode("");
        assert_eq!(ids, vec![1, 2], "Empty input should give [BOS, EOS]");
    }

    #[test]
    fn test_mini_tokenizer_all_bytes_map_correctly() {
        let tok = MiniTokenizer::new(512);
        let text = "abcdefghijklmnopqrstuvwxyz";
        let ids = tok.encode(text);
        let content: Vec<u32> = ids[1..ids.len() - 1].to_vec();
        for (i, &c) in text.as_bytes().iter().enumerate() {
            assert_eq!(
                content[i], c as u32,
                "Char '{:?}' should map to {}",
                c as char, c
            );
        }
    }

    #[test]
    fn test_mini_tokenizer_decode_skips_special_tokens() {
        let tok = MiniTokenizer::new(512);
        let ids = vec![104, 101, 108, 108, 111];
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello", "Should decode simple byte tokens");
    }

    #[test]
    fn test_mini_tokenizer_decode_filters_above_255() {
        let tok = MiniTokenizer::new(512);
        let ids = vec![104, 101, 108, 108, 111, 256, 300];
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello", "Should skip tokens >255");
    }

    #[test]
    fn test_mini_tokenizer_vocab_size_respected() {
        let tok = MiniTokenizer::new(100);
        let ids = tok.encode("abc\u{ff}");
        let content: Vec<u32> = ids[1..ids.len() - 1].to_vec();
        assert_eq!(
            content,
            vec![97, 98, 99],
            "Byte 255 should be skipped (>= vocab_size)"
        );
    }

    // ── CausalLmModel tests ──

    #[tokio::test]
    async fn test_causal_lm_default_state() {
        let model = CausalLmModel::new(NxrModelId::Omnis, TransformerConfig::default());
        let state = model.state().await.unwrap();
        assert!(!state["initialized"].as_bool().unwrap());
        assert!(!state["has_model"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_causal_lm_not_ready_before_init() {
        let model = CausalLmModel::new(NxrModelId::Omnis, TransformerConfig::default());
        assert!(!model.is_ready().await);
    }

    #[tokio::test]
    async fn test_causal_lm_ready_after_init() {
        let cfg = TransformerConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 64,
            intermediate_size: 64,
            ..Default::default()
        };
        let mut model = CausalLmModel::new(NxrModelId::Omnis, cfg.clone());
        let tok = MiniTokenizer::new(64);
        model.load_tokenizer(tok).await;
        model
            .initialize(serde_json::json!({
                "transformer_config": {
                    "vocab_size": 64, "hidden_size": 32,
                    "num_heads": 4, "num_kv_heads": 2,
                    "num_layers": 2, "max_seq_len": 64,
                }
            }))
            .await
            .unwrap();
        assert!(model.is_ready().await);
    }

    #[tokio::test]
    async fn test_causal_lm_reset() {
        let cfg = TransformerConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 64,
            intermediate_size: 64,
            ..Default::default()
        };
        let mut model = CausalLmModel::new(NxrModelId::Omnis, cfg.clone());
        let tok = MiniTokenizer::new(64);
        model.load_tokenizer(tok).await;
        model
            .initialize(serde_json::json!({
                "transformer_config": {
                    "vocab_size": 64, "hidden_size": 32,
                    "num_heads": 4, "num_kv_heads": 2,
                    "num_layers": 2, "max_seq_len": 64,
                }
            }))
            .await
            .unwrap();
        model.reset().await.unwrap();
        assert!(!model.is_ready().await);
    }

    #[tokio::test]
    async fn test_causal_lm_validate_initial_state() {
        let model = CausalLmModel::new(NxrModelId::Omnis, TransformerConfig::default());
        let result = model.validate().await.unwrap();
        assert!(!result.is_valid);
        assert!(result.errors.len() >= 2);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_causal_lm_identity() {
        let model = CausalLmModel::new(NxrModelId::Omnis, TransformerConfig::default());
        assert_eq!(model.identity().id, NxrModelId::Omnis);
    }

    #[test]
    fn test_causal_lm_config_contains_hidden_size() {
        let cfg = TransformerConfig {
            vocab_size: 128,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 128,
            intermediate_size: 128,
            ..Default::default()
        };
        let model = CausalLmModel::new(NxrModelId::Omnis, cfg);
        let config = model.config();
        assert_eq!(
            config["transformer_config"]["hidden_size"].as_u64(),
            Some(64)
        );
    }
}
