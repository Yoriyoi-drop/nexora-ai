pub mod tokenizer;
pub mod types;

pub use tokenizer::MiniTokenizer;
pub use types::{EchoNetInjectionConfig, TrainingReport};

use async_trait::async_trait;
use ndarray::Array1;
use rand::seq::SliceRandom;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn};

use nexora_erp::ERPConfig;
use nexora_training::{Trainer, TrainerConfig};
use nexora_transformer::{CausalLM, TransformerConfig};

use crate::echo_net_injector::EchoNetInjector;
use crate::model_agent_manager::global_model_agents;
use crate::shared::base_model::ValidationResult;
use crate::shared::{
    CapabilityVector, FinishReason, GenerationMetadata, InputData, ModelMeta, ModelStatistics,
    ModelTier, NxrInput, NxrModel, NxrModelError, NxrModelId, NxrModelResult, NxrOutput,
    NxrStreamChunk, OutputData, PerformanceMetrics, ResourceUsage, StreamChunkData, TokenOutput,
};

pub struct CausalLmModel {
    meta: ModelMeta,
    capabilities: CapabilityVector,
    config: Value,
    model: Arc<RwLock<Option<Arc<CausalLM>>>>,
    tokenizer: Arc<RwLock<Option<MiniTokenizer>>>,
    transformer_config: Arc<RwLock<TransformerConfig>>,
    initialized: Arc<RwLock<bool>>,
    statistics: Arc<RwLock<ModelStatistics>>,
    use_gpu: AtomicBool,
    use_half_precision: AtomicBool,
    echo_net_config: Option<EchoNetInjectionConfig>,
    sedc_enabled: bool,
    sedc_report: Arc<RwLock<Option<String>>>,
    erp_config: ERPConfig,
}

impl CausalLmModel {
    pub fn new(model_id: NxrModelId, transformer_config: TransformerConfig) -> Self {
        let model_tier = model_id.tier();
        let meta = ModelMeta::new(
            model_id,
            model_tier,
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
            use_half_precision: AtomicBool::new(false),
            echo_net_config: None,
            sedc_enabled: false,
            sedc_report: Arc::new(RwLock::new(None)),
            erp_config: ERPConfig::default(),
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

    pub fn with_erp_config(mut self, config: ERPConfig) -> Self {
        self.erp_config = config;
        self
    }

    pub fn with_half_precision(mut self) -> Self {
        self.use_half_precision = AtomicBool::new(true);
        self
    }

    /// Unload model dari memory (set ke None) — digunakan oleh ActiveStandbyScheduler.
    pub async fn unload_model(&self) {
        *self.model.write().await = None;
        *self.initialized.write().await = false;
        tracing::info!("CausalLM model unloaded from memory");
    }

    pub async fn load_model(&self) -> NxrModelResult<()> {
        let model_id = self.meta.id;

        // Resolve dari SingleBackboneRegistry — SEMUA model share Arc<CausalLM> yang sama.
        // Weight di-load sekali untuk semua 10 model crates.
        let backbone = match nexora_transformer::resolve_single_backbone() {
            Ok(b) => {
                info!(
                    "Using shared single backbone for {:?}",
                    model_id
                );
                b
            }
            Err(e) => {
                let tc = self.transformer_config.read().await;
                warn!(
                    "Single backbone resolve failed ({}), creating independent model",
                    e
                );
                Arc::new(CausalLM::new((*tc).clone()))
            }
        };

        // Jika model perlu tambahan (EchoNet injector, SEDC), clone backbone.
        // Jika tidak, pakai Arc langsung -> zero copy sharing.
        let needs_modification = self.echo_net_config.is_some() || self.sedc_enabled;
        let model: Arc<CausalLM> = if needs_modification {
            let mut model = (*backbone).clone();
            if self.use_half_precision.load(Ordering::Relaxed) {
                model = model.with_half_precision();
            }

            if let Some(echo_cfg) = &self.echo_net_config {
                let tc = self.transformer_config.read().await;
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

            if self.sedc_enabled {
                match model.compress_sedc_default() {
                    Ok(Some(report)) => {
                        let report_str =
                            serde_json::to_string_pretty(&report).unwrap_or_default();
                        *self.sedc_report.write().await = Some(report_str);
                        info!("SEDC compression stored in model ✓");
                    }
                    Ok(None) => {
                        info!("SEDC compression skipped (no GPU context)");
                    }
                    Err(e) => {
                        warn!("SEDC compression failed: {}", e);
                    }
                }
            }

            // ERP Layer 2: resonance pruning on SEDC-compressed weights
            // Berjalan setelah SEDC agar ERP menganalisis weights yang sudah
            // dikompresi secara numerik — tidak bentrok karena ERP bekerja
            // di level neuron grouping, bukan modifikasi numerik.
            let mut erp = nexora_erp::ERPEngine::new(self.erp_config.clone());
            let weights = model.collect_weights_for_sedc().0;
            if !weights.is_empty() {
                match erp.apply_pruning(&weights) {
                    Ok(layers) => {
                        info!(
                            "ERP resonance pruning applied on SEDC-compressed model: {} layers ({} groups)",
                            layers.len(),
                            layers.iter().map(|l| l.resonance_representations.len()).sum::<usize>(),
                        );
                    }
                    Err(e) => {
                        warn!("ERP resonance pruning skipped (non-fatal): {}", e);
                    }
                }
            }

            Arc::new(model)
        } else {
            // Zero-copy path: langsung pakai Arc dari registry
            backbone.clone()
        };

        *self.model.write().await = Some(model);
        *self.initialized.write().await = true;
        info!("CausalLM model loaded: {} params", backbone.config.parameter_count());
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
        TrainableCausalLM::load_checkpoint(Arc::make_mut(model_ref), path)
            .map_err(|e| NxrModelError::Inference(format!("Load failed: {}", e)))?;
        info!("Checkpoint loaded from {}", path);
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub async fn load_checkpoint_gpu(&self, path: &str) -> NxrModelResult<()> {
        
        let tc = self.transformer_config.read().await.clone();
        let gpu_model = CausalLM::from_checkpoint_gpu(tc, path)
            .map_err(|e| NxrModelError::Inference(format!("GPU checkpoint load failed: {}", e)))?;
        *self.model.write().await = Some(Arc::new(gpu_model));
        info!("GPU checkpoint loaded from {}", path);
        Ok(())
    }

    pub async fn get_model_arc(&self) -> Option<Arc<CausalLM>> {
        self.model.read().await.as_ref().cloned()
    }

    pub fn set_use_gpu(&self, enabled: bool) {
        self.use_gpu.store(enabled, Ordering::Relaxed);
    }

    pub fn set_use_half_precision(&self, enabled: bool) {
        self.use_half_precision.store(enabled, Ordering::Relaxed);
    }

    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> NxrModelResult<String> {
        self.generate_text_with_gpu(
            prompt,
            max_tokens,
            temperature,
            self.use_gpu.load(Ordering::Relaxed),
        )
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

        let guard = self.model.read().await;
        let m = guard
            .as_ref()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        let (output_ids, _) = m.generate_with_gpu(&input_ids, max_tokens, temperature, 50, use_gpu);

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
        let max_steps = cfg.max_steps;

        let val_tokens: Vec<Vec<u32>> = {
            let tokenizer = self.tokenizer.read().await;
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
            match val_data {
                Some(vd) => vd
                    .iter()
                    .filter_map(|t| {
                        let ids = tok.encode(t);
                        if ids.len() >= 2 { Some(ids) } else { None }
                    })
                    .collect(),
                None => Vec::new(),
            }
        };

        let estimated_tokens: usize = {
            let tokenizer = self.tokenizer.read().await;
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
            data.iter()
                .filter_map(|t| {
                    let ids = tok.encode(t);
                    if ids.len() >= 2 { Some(ids.len()) } else { None }
                })
                .sum()
        };
        info!(
            "Training data: {} sequences, ~{} tokens (streaming tokenization)",
            data.len(),
            estimated_tokens
        );

        let mut model = self.model.write().await;
        let causal_lm = model
            .take()
            .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?;
        let causal_lm = Arc::try_unwrap(causal_lm)
            .unwrap_or_else(|arc| {
                info!("Multiple Arc references to model existed during train_on_data — cloning (weights may not transfer)");
                (*arc).clone()
            });
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
            let mut epoch: Vec<Vec<u32>> = {
                let tokenizer = self.tokenizer.read().await;
                let tok = tokenizer
                    .as_ref()
                    .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
                let mut batch: Vec<Vec<u32>> = Vec::with_capacity(data.len().min(10000));
                for text in data {
                    let ids = tok.encode(text);
                    if ids.len() >= 2 {
                        batch.push(ids);
                    }
                }
                batch
            };
            epoch.shuffle(&mut rng);

            for sample in &epoch {
                if trainer.step >= max_steps {
                    break;
                }
                let mut chunks: Vec<(&[u32], &[u32])> = Vec::new();
                chunks.extend(sample.chunks(seq_length + 1).filter_map(|c| {
                    if c.len() < 2 { return None; }
                    let seq = c.len().min(seq_length + 1);
                    if seq < 2 { return None; }
                    let input = &c[..seq - 1];
                    let target = &c[1..seq];
                    Some((input, target))
                }));
                if !chunks.is_empty() {
                    if let Some(loss) = trainer.train_batch_multi(&chunks) {
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
                }
            }

            if !val_tokens.is_empty()
                && trainer.step % trainer.config.val_every_steps == 0
                && trainer.step > 0
            {
                let val_metrics = trainer.evaluate_loss(&val_tokens, seq_length);
                info!(
                    "  Validation | step {} | loss: {:.4} | perplexity: {:.2}",
                    trainer.step, val_metrics.avg_loss, val_metrics.perplexity
                );
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
        let trained_model = trainer.take_model();
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

        *self.model.write().await = Some(Arc::new(trained_model));

        Ok(TrainingReport {
            steps: trainer.step,
            final_loss: final_loss as f64,
            total_tokens: trainer.total_tokens,
            duration_secs: elapsed.as_secs_f64(),
            val_loss: val_loss.as_ref().map(|m| m.avg_loss),
            val_perplexity: val_loss.as_ref().map(|m| m.perplexity),
        })
    }

    pub async fn run_agent_preprocessing(&self, input: &str) -> String {
        let model_id = self.meta.id;
        match model_id {
            NxrModelId::Axiom => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.axiom.logic_core().evaluate(input).await {
                        Ok(report) => format!("[Axiom analysis: {}] {}", report, input),
                        Err(_) => input.to_string(),
                    }
                } else {
                    input.to_string()
                }
            }
            NxrModelId::Cipher => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.cipher.security_guardian().analyze(input).await {
                        Ok(report) => format!("[Cipher safety: {}] {}", report, input),
                        Err(_) => input.to_string(),
                    }
                } else {
                    input.to_string()
                }
            }
            NxrModelId::Spectra => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.spectra.analyze(input).await {
                        Ok(annotation) => format!("{} {}", annotation, input),
                        Err(_) => input.to_string(),
                    }
                } else {
                    input.to_string()
                }
            }
            NxrModelId::Aether => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.aether.analyze_emotion(input).await {
                        Ok(annotation) => format!("{} {}", annotation, input),
                        Err(_) => input.to_string(),
                    }
                } else {
                    input.to_string()
                }
            }
            NxrModelId::Genesis => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.genesis.analyze_creation(input).await {
                        Ok(annotation) => format!("{} {}", annotation, input),
                        Err(_) => input.to_string(),
                    }
                } else {
                    input.to_string()
                }
            }
            NxrModelId::Kronos => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.kronos.analyze_temporal(input).await {
                        Ok(annotation) => format!("{} {}", annotation, input),
                        Err(_) => input.to_string(),
                    }
                } else {
                    input.to_string()
                }
            }
            _ => input.to_string(),
        }
    }

    pub async fn run_agent_postprocessing(&self, output: &str) -> String {
        let model_id = self.meta.id;
        match model_id {
            NxrModelId::Axiom => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.axiom.truth_validator().validate(output).await {
                        Ok(report) => format!("[Axiom validated: {}] {}", report, output),
                        Err(_) => output.to_string(),
                    }
                } else {
                    output.to_string()
                }
            }
            NxrModelId::Cipher => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.cipher.encryption_master().sanitize(output).await {
                        Ok(report) => report,
                        Err(_) => output.to_string(),
                    }
                } else {
                    output.to_string()
                }
            }
            NxrModelId::Nexum => {
                if let Some(mgr) = global_model_agents() {
                    match mgr.nexum.evaluate_alignment(output).await {
                        Ok(annotation) => format!("{}\n\n{}", output, annotation),
                        Err(_) => output.to_string(),
                    }
                } else {
                    output.to_string()
                }
            }
            NxrModelId::Vortex => {
                format!("{}\n\n[Vortex Analysis Pending]", output)
            }
            _ => output.to_string(),
        }
    }
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
        #[cfg(not(feature = "legacy-fastpath"))]
        {
            return Err(NxrModelError::NotInitialized(
                "Legacy inference path disabled. Use inference engine via generate_text() or chat().".to_string()
            ));
        }

        #[cfg(feature = "legacy-fastpath")]
        {
            return self.infer_legacy(input).await;
        }
    }

    async fn infer_stream(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> NxrModelResult<()> {
        #[cfg(not(feature = "legacy-fastpath"))]
        {
            return Err(NxrModelError::NotInitialized(
                "Legacy streaming inference disabled. Use inference engine via generate_text_stream().".to_string()
            ));
        }

        #[cfg(feature = "legacy-fastpath")]
        {
            return self.infer_stream_legacy(input, callback).await;
        }
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
        let memory_gb = match std::fs::read_to_string("/proc/self/status") {
            Ok(status) => {
                if let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:")) {
                    if let Some(v) = line.split_whitespace().nth(1) {
                        match v.parse::<f64>() {
                            Ok(kb) => (kb / 1_048_576.0) as f32,
                            Err(e) => {
                                tracing::warn!("Failed to parse VmRSS value '{}': {}", v, e);
                                0.0
                            }
                        }
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            Err(e) => {
                tracing::warn!("Failed to read /proc/self/status: {}", e);
                0.0
            }
        };

        let cpu_percent = match std::fs::read_to_string("/proc/self/stat") {
            Ok(stat) => {
                let parts: Vec<&str> = stat.split_whitespace().collect();
                if parts.len() >= 15 {
                    let utime: f64 = parts[13].parse().unwrap_or(0.0);
                    let stime: f64 = parts[14].parse().unwrap_or(0.0);
                    let total_ticks = utime + stime;
                    let uptime = match std::fs::read_to_string("/proc/uptime") {
                        Ok(u) => match u.split_whitespace().next() {
                            Some(val) => match val.parse::<f64>() {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::warn!("Failed to parse uptime '{}': {}", val, e);
                                    1.0
                                }
                            },
                            None => {
                                tracing::warn!("Empty /proc/uptime");
                                1.0
                            }
                        },
                        Err(e) => {
                            tracing::warn!("Failed to read /proc/uptime: {}", e);
                            1.0
                        }
                    };
                    let hertz = 100.0;
                    ((total_ticks / hertz / uptime) * 100.0) as f32
                } else {
                    0.0
                }
            }
            Err(e) => {
                tracing::warn!("Failed to read /proc/self/stat: {}", e);
                0.0
            }
        };

        let use_gpu = self.use_gpu.load(Ordering::Relaxed);
        let gpu_percent = if use_gpu {
            let total: f32 = (0..8)
                .filter_map(|i| {
                    let path = format!("/proc/driver/nvidia/gpus/{i}/status");
                    match std::fs::read_to_string(&path) {
                        Ok(s) => s
                            .lines()
                            .find(|l| l.contains("GPU Utilization"))
                            .and_then(|l| {
                                l.split_whitespace().find(|w| w.ends_with('%')).and_then(
                                    |w| match w.trim_end_matches('%').parse::<f32>() {
                                        Ok(val) => Some(val),
                                        Err(e) => {
                                            tracing::warn!(
                                                "Failed to parse GPU utilization '{}': {}",
                                                w.trim_end_matches('%'),
                                                e
                                            );
                                            None
                                        }
                                    },
                                )
                            }),
                        Err(e) => {
                            tracing::warn!("Failed to read GPU status {}: {}", path, e);
                            None
                        }
                    }
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

#[cfg(feature = "legacy-fastpath")]
impl CausalLmModel {
    /// Initialize with empty model (no random block weights).
    /// Used for ActiveStandby: model is registered but weights are loaded
    /// on demand from checkpoint via `load_lazy_blocks()`.
    pub async fn initialize_empty(&self) -> NxrModelResult<()> {
        let tc = self.transformer_config.read().await.clone();
        let mut model = CausalLM::new_empty(tc);

        if let Some(echo_cfg) = &self.echo_net_config {
            let injector = EchoNetInjector::new(
                model.config.hidden_size,
                echo_cfg.phase_separation_strength,
                echo_cfg.max_window,
                echo_cfg.alpha,
            )
            .map_err(|e| NxrModelError::Internal(format!("EchoNet injector: {}", e)))?;

            model.injectors.push((
                echo_cfg.inject_after_layer,
                std::sync::Mutex::new(Box::new(injector)),
            ));
        }

        *self.model.write().await = Some(Arc::new(model));
        *self.initialized.write().await = true;
        info!("CausalLM empty model registered (standby — weights loaded on demand)");
        Ok(())
    }

    async fn infer_legacy(&self, input: &NxrInput) -> NxrModelResult<NxrOutput> {
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
                let _decoded = tok.decode(t);
                let elapsed = std::time::Instant::now();
                let total = t.len() as u64;
                let dur = elapsed.elapsed().as_millis() as u64;
                let tp_s = if dur > 0 {
                    total as f32 / (dur as f32 / 1000.0)
                } else {
                    0.0
                };
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

        let use_gpu = self.use_gpu.load(Ordering::Relaxed);
        let guard = self.model.read().await;
        let m = guard.as_ref().ok_or_else(|| {
            NxrModelError::NotInitialized("Model was reset before generation".to_string())
        })?;
        let (output_ids, _cache) =
            m.generate_with_gpu(&input_ids, max_tokens, temperature, top_k, use_gpu);

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

    async fn infer_stream_legacy(
        &self,
        input: &NxrInput,
        callback: Arc<dyn Fn(NxrStreamChunk) + Send + Sync>,
    ) -> NxrModelResult<()> {
        let text = match &input.data {
            InputData::Text(t) => t.clone(),
            InputData::Tokens(t) => {
                let decoded = {
                    let tokenizer = self.tokenizer.read().await;
                    let tok = tokenizer.as_ref().ok_or_else(|| {
                        NxrModelError::NotInitialized("Tokenizer not loaded".to_string())
                    })?;
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

        let (input_ids, tok) = {
            let tokenizer = self.tokenizer.read().await;
            let tok = tokenizer
                .as_ref()
                .ok_or_else(|| NxrModelError::NotInitialized("Tokenizer not loaded".to_string()))?;
            (tok.encode(&text), tok.clone())
        };
        let model_arc = {
            let guard = self.model.read().await;
            guard
                .clone()
                .ok_or_else(|| NxrModelError::NotInitialized("Model not loaded".to_string()))?
        };
        let callback = callback.clone();
        let input_id = input.id;

        tokio::task::spawn_blocking(move || {
            let mut cache = model_arc.reset_cache();
            for &tid in &input_ids {
                if let Err(e) = model_arc.forward(&[tid], &mut cache) {
                    tracing::warn!("Prefill token {} failed: {}", tid, e);
                }
            }
            let mut last_id = *input_ids.last().unwrap_or(&0);

            for _ in 0..max_tokens {
                let vs = model_arc.config.vocab_size;
                let logits = match model_arc.forward(&[last_id], &mut cache) {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!("Generation step forward failed: {}", e);
                        Array1::zeros(vs)
                    }
                };
                let next_id = nexora_transformer::sample_token(&logits, temperature, top_k);
                let token_text = tok.decode(&[next_id]);
                callback(NxrStreamChunk {
                    id: uuid::Uuid::new_v4(),
                    input_id,
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
                input_id,
                timestamp: chrono::Utc::now(),
                data: StreamChunkData::Status("done".to_string()),
                is_final: true,
            });
        })
        .await
        .map_err(|e| NxrModelError::Inference(format!("Stream generation task failed: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::NxrModelId;

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
