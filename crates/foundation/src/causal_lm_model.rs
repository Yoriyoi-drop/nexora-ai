use async_trait::async_trait;
use rand::seq::SliceRandom;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn};

use nexora_training::{Trainer, TrainerConfig};
use nexora_transformer::{CausalLM, TransformerConfig};

use crate::shared::base_model::ValidationResult;
use crate::shared::{
    CapabilityVector, FinishReason, GenerationMetadata, InputData, ModelMeta, ModelStatistics,
    ModelTier, NxrInput, NxrModel, NxrModelError, NxrModelId, NxrModelResult, NxrOutput,
    NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage, StreamChunkData, TokenOutput,
};

/// Byte-level tokenizer for MVP: maps bytes 0-255 to token IDs directly.
pub struct MiniTokenizer {
    vocab_size: usize,
}

impl MiniTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

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

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .filter_map(|&id| if id < 256 { Some(id as u8) } else { None })
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
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
    use_gpu: Arc<RwLock<bool>>,
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
            use_gpu: Arc::new(RwLock::new(true)),
        }
    }

    pub async fn load_model(&self) -> NxrModelResult<()> {
        let tc = self.transformer_config.read().await.clone();
        let model = CausalLM::new(tc.clone());
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

    pub async fn set_use_gpu(&self, enabled: bool) {
        *self.use_gpu.write().await = enabled;
    }

    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> NxrModelResult<String> {
        self.generate_text_with_gpu(prompt, max_tokens, temperature, false)
            .await
    }

    pub async fn generate_text_with_gpu(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        use_gpu: bool,
    ) -> NxrModelResult<String> {
        let model = self.model.read().await;
        let model_ref = model
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        let tokenizer = self.tokenizer.read().await;
        let tok_ref = tokenizer
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
        let input_ids = tok_ref.encode(prompt);
        let (output_ids, _) =
            model_ref.generate_with_gpu(&input_ids, max_tokens, temperature, 50, use_gpu);
        let text = tok_ref.decode(&output_ids);
        Ok(text)
    }

    pub async fn train_on_data(
        &self,
        data: &[String],
        cfg: TrainerConfig,
        val_data: Option<&[String]>,
    ) -> NxrModelResult<TrainingReport> {
        let seq_length = cfg.seq_length;
        let _batch_size = cfg.batch_size;
        let max_steps = cfg.max_steps;

        let tokenizer = self.tokenizer.read().await;
        let tok_ref = tokenizer
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;

        let train_tokens: Vec<Vec<u32>> = data
            .iter()
            .filter_map(|t| {
                let ids = tok_ref.encode(t);
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
                    let ids = tok_ref.encode(t);
                    if ids.len() >= 2 {
                        Some(ids)
                    } else {
                        None
                    }
                })
                .collect(),
            None => Vec::new(),
        };
        drop(tokenizer);

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
                let _val_metrics = trainer.evaluate_loss(&val_tokens, seq_length);
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
        let model = self.model.read().await;
        let _model_ref = model
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        drop(model);

        let tokenizer = self.tokenizer.read().await;
        let tok_ref = tokenizer
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;

        let text = match &input.data {
            InputData::Text(t) => t.clone(),
            InputData::Tokens(t) => {
                let _decoded = tok_ref.decode(t);
                return Ok(NxrOutput {
                    id: uuid::Uuid::new_v4(),
                    input_id: input.id,
                    timestamp: chrono::Utc::now(),
                    data: OutputData::Tokens(
                        t.iter()
                            .enumerate()
                            .map(|(i, &tid)| {
                                let t = tok_ref.decode(&[tid]);
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
                        generation_time_ms: 0,
                        model_version: "0.1.0".to_string(),
                        seed: None,
                        extras: Default::default(),
                    },
                    performance: PerformanceMetrics {
                        tokens_per_second: 0.0,
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

        let input_ids = tok_ref.encode(&text);
        let start = std::time::Instant::now();

        let model = self.model.read().await;
        let model_ref = model.as_ref().ok_or_else(|| {
            NxrModelError::NotInitialized("Model was reset before generation".to_string())
        })?;

        let use_gpu = *self.use_gpu.read().await;
        let (output_ids, _cache) =
            model_ref.generate_with_gpu(&input_ids, max_tokens, temperature, top_k, use_gpu);

        let elapsed = start.elapsed().as_millis() as u64;
        let generated_text = tok_ref.decode(&output_ids);

        let mut stats = self.statistics.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        stats.total_tokens_generated += output_ids.len() as u64;

        let tokens_per_second = if elapsed > 0 {
            output_ids.len() as f32 / (elapsed as f32 / 1000.0)
        } else {
            0.0
        };

        let _token_outputs: Vec<TokenOutput> = output_ids
            .iter()
            .enumerate()
            .map(|(i, &tid)| {
                let t = tok_ref.decode(&[tid]);
                TokenOutput {
                    token_id: tid,
                    text: t,
                    log_prob: 0.0,
                    position: input_ids.len() + i,
                }
            })
            .collect();

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
        let result = self.infer(input).await?;
        if let OutputData::Text(text) = &result.data {
            callback(NxrStreamChunk {
                id: uuid::Uuid::new_v4(),
                input_id: input.id,
                timestamp: chrono::Utc::now(),
                data: StreamChunkData::TextDelta(text.clone()),
                is_final: true,
            });
        }
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
        Ok(ResourceUsage {
            memory_gb: 0.0,
            cpu_percent: 0.0,
            gpu_percent: None,
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
