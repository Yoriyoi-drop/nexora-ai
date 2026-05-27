//! Command handlers for CLI operations

use crate::error::{NexoraError, NexoraResult};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

use super::commands::{Cli, Commands, ConfigAction, MemoryAction, TokenizerAction};
use crate::{NexoraAI, NexoraConfig};

use nexora_datastream::{
    filter::{DedupFilter, LengthFilter, QualityFilter},
    DataSample, ExecutionResult, SourceCategory, SourceInfo,
};

/// Supported extensions for auto-detection.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "csv", "tsv", "json", "jsonl", "ndjson", "parquet", "arrow", "ipc", "feather",
];

/// Load dataset from any supported format (auto-detect by extension).
/// If `path` is a directory, scans all supported files recursively.
async fn load_dataset_raw(path: &PathBuf) -> NexoraResult<Vec<DataSample>> {
    let source = SourceInfo {
        name: "train_foundation".into(),
        url: None,
        trust_score: 1.0,
        category: SourceCategory::Other,
        fetch_timestamp: chrono::Utc::now().timestamp(),
    };

    if path.is_dir() {
        let mut all = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .map_err(|e| NexoraError::Io { source: e })?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext))
                    .unwrap_or(false)
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in &entries {
            match nexora_datastream::format_loader::load_dataset(&entry.path(), source.clone()) {
                Ok(samples) => {
                    info!("Loaded {} samples from {:?}", samples.len(), entry.path());
                    all.extend(samples);
                }
                Err(e) => warn!("Skipping {:?}: {}", entry.path(), e),
            }
        }
        info!(
            "Loaded total {} samples from directory {:?}",
            all.len(),
            path
        );
        Ok(all)
    } else {
        nexora_datastream::format_loader::load_dataset(path, source).map_err(|e| NexoraError::Io {
            source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
        })
    }
}

/// Run dataset through the DataStream DAG filter pipeline (LengthFilter → QualityFilter → DedupFilter).
async fn filter_dataset(samples: Vec<DataSample>) -> NexoraResult<Vec<String>> {
    let mut graph = nexora_datastream::ExecutionGraph::new();
    graph.add_node(
        "length",
        Arc::new(LengthFilter {
            min_chars: 10,
            min_words: 3,
            ..Default::default()
        }),
        vec![],
        true,
        1,
    );
    graph.add_node(
        "quality",
        Arc::new(QualityFilter {
            min_quality_score: 0.1,
            min_unique_word_ratio: 0.1,
            ..Default::default()
        }),
        vec!["length".into()],
        true,
        2,
    );
    graph.add_node(
        "dedup",
        Arc::new(DedupFilter::new()),
        vec!["quality".into()],
        false,
        3,
    );
    graph.finalize();

    let total = samples.len();
    let mut accepted: Vec<String> = Vec::new();
    let mut filter_rejected: HashMap<String, u64> = HashMap::new();

    for s in samples {
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        let result = graph.execute(s, cancel_rx).await;
        drop(cancel_tx);
        if let ExecutionResult::Rejected {
            ref filter_name, ..
        } = &result
        {
            *filter_rejected.entry(filter_name.clone()).or_insert(0) += 1;
        }
        if result.is_accepted() {
            if let Some(sample) = result.sample() {
                accepted.push(sample.text.clone());
            }
        }
    }

    let rejection_breakdown: Vec<String> = filter_rejected
        .iter()
        .map(|(name, count)| format!("{}: {}", name, count))
        .collect();
    info!("  filter rejection: {:?}", rejection_breakdown);
    info!(
        "  {} samples passed filter (from {})",
        accepted.len(),
        total
    );

    Ok(accepted)
}

impl Cli {
    /// Run CLI application
    pub async fn run(&self) -> NexoraResult<()> {
        // Initialize logging
        self.init_logging();

        // Load configuration
        let config = if self.config.exists() {
            match NexoraConfig::from_file(&self.config) {
                Ok(config) => config,
                Err(e) => {
                    warn!("Failed to load configuration: {}", e);
                    return Err(NexoraError::config(format!(
                        "Configuration load failed: {}",
                        e
                    )));
                }
            }
        } else {
            info!("No config file found, using defaults");
            NexoraConfig::default()
        };

        // Override config with CLI arguments
        let config = self.override_config(config);

        // Validate configuration
        if let Err(e) = config.validate() {
            return Err(NexoraError::config(format!(
                "Configuration validation failed: {}",
                e
            )));
        }

        // Execute command — initialize NexoraAI lazily when needed
        match &self.command {
            Commands::Config { action } => self.run_config(action).await,
            Commands::Tokenizer { action } => self.run_tokenizer(action).await,
            Commands::Health { detailed } => {
                let nexora = NexoraAI::new(config).await.map_err(|e| {
                    NexoraError::initialization(format!("Nexora AI initialization failed: {}", e))
                })?;
                self.run_health(&nexora, *detailed).await
            }
            _ => {
                let nexora = NexoraAI::new(config).await.map_err(|e| {
                    NexoraError::initialization(format!("Nexora AI initialization failed: {}", e))
                })?;
                match &self.command {
                    Commands::Start {
                        host,
                        port,
                        tls,
                        cert_path,
                        key_path,
                    } => {
                        self.run_server(&nexora, host, *port, *tls, cert_path, key_path)
                            .await
                    }
                    Commands::Process {
                        input,
                        format,
                        output,
                    } => self.run_process(&nexora, input, format, output).await,
                    Commands::TrainFoundation {
                        data,
                        model_id,
                        steps,
                        batch_size,
                        learning_rate,
                        seq_length,
                        output,
                        val_data,
                        parallel,
                        gpu,
                        half_precision,
                    } => self
                        .run_train_foundation(
                            &nexora,
                            data,
                            model_id,
                            *steps,
                            *batch_size,
                            *learning_rate,
                            *seq_length,
                            output,
                            val_data,
                            *parallel,
                            *gpu,
                            *half_precision,
                        )
                        .await
                        .map_err(|e| {
                            NexoraError::processing(format!(
                                "TrainFoundation command failed: {}",
                                e
                            ))
                        }),
                    Commands::CollectData {
                        sources,
                        max_samples,
                        max_shard_size_mb,
                        output,
                    } => {
                        self.run_collect_data(sources, *max_samples, *max_shard_size_mb, output)
                            .await
                    }
                    Commands::LoadCheckpoint { model, path } => {
                        self.run_load_checkpoint(model, path).await.map_err(|e| {
                            NexoraError::processing(format!("LoadCheckpoint command failed: {}", e))
                        })
                    }
                    Commands::Generate {
                        prompt,
                        max_tokens,
                        temperature,
                        output,
                    } => {
                        self.run_generate(&nexora, prompt, *max_tokens, *temperature, output)
                            .await
                    }
                    Commands::Chat {
                        message,
                        conversation_id,
                        history_file,
                    } => self
                        .run_chat(&nexora, message, conversation_id, history_file)
                        .await
                        .map_err(|e| {
                            NexoraError::processing(format!("Chat command failed: {}", e))
                        }),
                    Commands::Analyze {
                        file,
                        language,
                        format,
                        output,
                    } => {
                        self.run_analyze(&nexora, file, language, format, output)
                            .await
                    }
                    Commands::Codegen {
                        description,
                        language,
                        output,
                    } => {
                        self.run_codegen(&nexora, description, language, output)
                            .await
                    }
                    Commands::Train {
                        data,
                        output,
                        tokenizer,
                        epochs,
                        batch_size,
                        learning_rate,
                        gpu,
                        seq_length,
                        resume,
                        model_id,
                        parallel,
                        half_precision,
                    } => self
                        .run_train(
                            &nexora,
                            data,
                            output,
                            tokenizer,
                            *epochs,
                            *batch_size,
                            *learning_rate,
                            *gpu,
                            *seq_length,
                            *resume,
                            model_id,
                            *parallel,
                            *half_precision,
                        )
                        .await
                        .map_err(|e| {
                            NexoraError::processing(format!("Train command failed: {}", e))
                        }),
                    Commands::Evaluate {
                        model,
                        test_data,
                        tokenizer,
                        output,
                    } => self
                        .run_evaluate(&nexora, model, test_data, tokenizer, output)
                        .await
                        .map_err(|e| {
                            NexoraError::processing(format!("Evaluate command failed: {}", e))
                        }),
                    Commands::Info {
                        performance,
                        memory,
                        models,
                        format,
                    } => {
                        self.run_info(&nexora, *performance, *memory, *models, format)
                            .await
                    }
                    Commands::Memory { action } => self.run_memory(&nexora, action).await,
                    other => {
                        warn!("Unexpected command variant in inner handler: {:?}", other);
                        return Err(NexoraError::processing(format!(
                            "Unhandled command: {:?}",
                            other
                        )));
                    }
                }
            }
        }
    }

    /// Initialize logging
    fn init_logging(&self) {
        let level = match self.log_level.to_lowercase().as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        };

        let _ = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(true)
            .try_init();

        if self.verbose {
            info!("Verbose logging enabled");
        }
    }

    /// Override configuration with CLI arguments
    fn override_config(&self, mut config: NexoraConfig) -> NexoraConfig {
        match &self.command {
            Commands::Start {
                host,
                port,
                tls,
                cert_path,
                key_path,
            } => {
                config.server.host = host.clone();
                config.server.port = *port;
                config.server.enable_tls = *tls;
                if let Some(cert) = cert_path {
                    config.server.cert_path = Some(cert.to_string_lossy().to_string());
                }
                if let Some(key) = key_path {
                    config.server.key_path = Some(key.to_string_lossy().to_string());
                }
            }
            _ => {}
        }

        config
    }

    /// Write output to file or stdout
    fn write_output(&self, content: &str, output: &Option<PathBuf>) -> NexoraResult<()> {
        match output {
            Some(path) => {
                std::fs::write(path, content).map_err(|e| NexoraError::io(e))?;
                info!("Output written to: {:?}", path);
            }
            None => {
                // Intentional CLI stdout output for user-facing command results
                println!("{}", content);
            }
        }
        Ok(())
    }

    /// Run server command
    async fn run_server(
        &self,
        nexora: &NexoraAI,
        host: &str,
        port: u16,
        tls: bool,
        cert_path: &Option<PathBuf>,
        key_path: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Starting Nexora AI server on http://{}:{}", host, port);

        let config = crate::ServerConfig {
            host: host.to_string(),
            port,
            enable_tls: tls,
            cert_path: cert_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            key_path: key_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            max_connections: 1000,
            request_timeout_seconds: 30,
            enable_cors: true,
            cors_origins: vec![],
            api_keys: vec![],
            enable_auth: false,
        };

        let server = crate::NexoraServer::new(config);
        let nexora = Arc::new(nexora.clone());

        server
            .start(nexora)
            .await
            .map_err(|e| NexoraError::system(format!("Server start failed: {}", e)))
    }

    /// Run process command
    async fn run_process(
        &self,
        nexora: &NexoraAI,
        input: &str,
        format: &str,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        let truncated = if input.len() > 100 {
            let slice = &input[..input.len().min(100)];
            format!("{} [truncated {} chars]", slice, input.len())
        } else {
            input.to_string()
        };
        info!("Processing input: {}", truncated);

        let response = nexora
            .process_request(input)
            .await
            .map_err(|e| NexoraError::processing(format!("Request processing failed: {}", e)))?;

        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "input": input,
                    "response": response,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let output_str = serde_json::to_string_pretty(&json_output)
                    .map_err(|e| NexoraError::serialization(e))?;
                self.write_output(&output_str, output)?;
            }
            "text" => {
                self.write_output(&response, output)?;
            }
            _ => {
                return Err(NexoraError::validation(
                    "format",
                    format!("Unsupported output format: {}", format),
                ));
            }
        }

        Ok(())
    }

    /// Run generate command
    async fn run_generate(
        &self,
        nexora: &NexoraAI,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        let truncated = if prompt.len() > 100 {
            format!("{} [truncated {} chars]", &prompt[..100], prompt.len())
        } else {
            prompt.to_string()
        };
        info!("Generating text from prompt: {}", truncated);

        let generated = nexora
            .generate_text(prompt, max_tokens, temperature)
            .await
            .map_err(|e| NexoraError::model(format!("Text generation failed: {}", e)))?;
        self.write_output(&generated, output)?;

        Ok(())
    }

    async fn run_train_foundation(
        &self,
        _nexora: &NexoraAI,
        data: &PathBuf,
        model_id: &str,
        steps: usize,
        batch_size: usize,
        learning_rate: f32,
        seq_length: usize,
        output: &Option<PathBuf>,
        val_data: &Option<PathBuf>,
        parallel: bool,
        gpu: bool,
        half_precision: bool,
    ) -> NexoraResult<()> {
        info!("=== FOUNDATION TRAINING ===");
        info!("Data: {:?}, Model: {}, Steps: {}, Batch: {}, LR: {}, SeqLen: {}, Output: {:?}, Parallel: {}", data, model_id, steps, batch_size, learning_rate, seq_length, output, parallel);

        if !data.exists() {
            return Err(NexoraError::Io {
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Training data not found: {:?}", data),
                ),
            });
        }
        info!("[1/2] Loading + filtering dataset via DataStream DAG pipeline...");
        let raw_samples = load_dataset_raw(data).await?;
        let lines = filter_dataset(raw_samples).await?;

        let val_lines: Vec<String> = match val_data {
            Some(path) if path.exists() => {
                let raw_val = load_dataset_raw(path).await?;
                filter_dataset(raw_val).await?
            }
            _ => Vec::new(),
        };

        let model_ids: Vec<nexora_foundation::NxrModelId> = if model_id.to_lowercase() == "all" {
            nexora_foundation::NxrModelId::all().to_vec()
        } else {
            let parsed = match model_id.to_lowercase().as_str() {
                "omnis" => nexora_foundation::NxrModelId::Omnis,
                "vortex" => nexora_foundation::NxrModelId::Vortex,
                "aether" => nexora_foundation::NxrModelId::Aether,
                "spectra" => nexora_foundation::NxrModelId::Spectra,
                "nexum" => nexora_foundation::NxrModelId::Nexum,
                "axiom" => nexora_foundation::NxrModelId::Axiom,
                "cipher" => nexora_foundation::NxrModelId::Cipher,
                "swift" => nexora_foundation::NxrModelId::Swift,
                "kronos" => nexora_foundation::NxrModelId::Kronos,
                "genesis" => nexora_foundation::NxrModelId::Genesis,
                other => {
                    return Err(NexoraError::validation(
                        "model_id",
                        format!("Invalid model ID: {}", other),
                    ))
                }
            };
            vec![parsed]
        };

        let save_path = output.as_ref().map(|p| p.to_string_lossy().to_string());

        let trainer_cfg = nexora_foundation::training::TrainerConfig {
            learning_rate,
            max_steps: steps,
            seq_length,
            vocab_size: 512,
            save_path,
            save_every: 100,
            report_every: 1,
            batch_size,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
            warmup_steps: (steps / 20).max(1),
            val_every_steps: (steps / 5).max(1),
            early_stop_patience: 3,
            use_gpu: gpu,
            num_replicas: 1,
        };

        // Initialize GPU if enabled
        if gpu {
            #[cfg(feature = "gpu")]
            {
                use nexora_deeplearning::autograd::gpu::GpuContext;
                match GpuContext::init() {
                    Ok(ctx) => info!("GPU initialized: {:?}", ctx.adapter_info()),
                    Err(e) => warn!("GPU init failed, falling back to CPU: {}", e),
                }
            }
            #[cfg(not(feature = "gpu"))]
            warn!("GPU feature not compiled; use --features gpu");
        }

        let registry = nexora_foundation::shared::model_registry::global_registry();
        let val_ref = val_lines;
        let tc_seq = trainer_cfg.clone();

        let out_path = output.clone();
        if parallel {
            let mut handles = Vec::with_capacity(model_ids.len());
            for mid in model_ids {
                let lines = lines.clone();
                let vr = val_ref.clone();
                let tc = trainer_cfg.clone();
                let reg = registry.clone();
                let out = out_path.clone();
                let hp = half_precision;
                handles.push(tokio::spawn(async move {
                    let raw = reg
                        .get_model_raw(&mid)
                        .await
                        .map_err(|e| format!("Model {:?} not found: {}", mid, e))?;
                    let model: Arc<nexora_foundation::causal_lm_model::CausalLmModel> = raw
                        .downcast::<nexora_foundation::causal_lm_model::CausalLmModel>()
                        .map_err(|_| "Failed to downcast".to_string())?;
                    if gpu {
                        model.set_use_gpu(true);
                    }
                    if hp {
                        model.set_use_half_precision(true);
                        model.load_model().await.map_err(|e| format!("Half-precision reload failed: {}", e))?;
                    }
                    let val_opt: Option<&[String]> = if vr.is_empty() { None } else { Some(&vr) };
                    let report = model
                        .train_on_data(&lines, tc, val_opt)
                        .await
                        .map_err(|e| format!("Training failed: {}", e))?;
                    if let Some(p) = &out {
                        let save_path = format!("{}_{}.safetensors", p.display(), mid);
                        if let Err(e) = model.save_checkpoint(&save_path).await {
                            warn!("Failed to save checkpoint to {}: {}", save_path, e);
                        }
                    }
                    Ok::<_, String>((mid, report))
                }));
            }

            let results = futures::future::join_all(handles).await;
            for r in results {
                match r {
                    Ok(Ok((mid, report))) => {
                        info!(
                            "  {:?} | Steps: {} | Loss: {:.4} | Tokens: {} | {:.1}s",
                            mid,
                            report.steps,
                            report.final_loss,
                            report.total_tokens,
                            report.duration_secs
                        );
                    }
                    Ok(Err(e)) => info!("  Error: {}", e),
                    Err(e) => info!("  Task panicked: {}", e),
                }
            }
        } else {
            for mid in model_ids {
                let raw = registry
                    .get_model_raw(&mid)
                    .await
                    .map_err(|e| NexoraError::model(format!("Model {:?} not found: {}", mid, e)))?;
                let model: Arc<nexora_foundation::causal_lm_model::CausalLmModel> = raw
                    .downcast::<nexora_foundation::causal_lm_model::CausalLmModel>()
                    .map_err(|_| NexoraError::model("Failed to downcast".to_string()))?;
                let val_opt: Option<&[String]> = if val_ref.is_empty() {
                    None
                } else {
                    Some(&val_ref)
                };

                // Enable GPU on model if requested
                if gpu {
                    model.set_use_gpu(true);
                }

                if half_precision {
                    model.set_use_half_precision(true);
                    model.load_model().await.map_err(|e| NexoraError::model(format!("Half-precision reload failed: {}", e)))?;
                }

                let report = model
                    .train_on_data(&lines, tc_seq.clone(), val_opt)
                    .await
                    .map_err(|e| NexoraError::model(format!("Training {:?} failed: {}", mid, e)))?;

                if let Some(p) = &output {
                    let save_path = format!("{}_{}.safetensors", p.display(), mid);
                    if let Err(e) = model.save_checkpoint(&save_path).await {
                        warn!("Failed to save checkpoint to {}: {}", save_path, e);
                    }
                }

                info!(
                    "  {:?} | Steps: {} | Loss: {:.4} | Tokens: {} | {:.1}s",
                    mid, report.steps, report.final_loss, report.total_tokens, report.duration_secs
                );

                let sample = model
                    .generate_text_with_gpu("The", 20, 0.8, gpu)
                    .await
                    .map_err(|e| NexoraError::model(format!("Generation failed: {}", e)))?;
                info!("  {:?} sample: {:?}", mid, sample);
            }
        }

        Ok(())
    }

    /// Load checkpoint into a model
    async fn run_load_checkpoint(&self, model: &str, path: &PathBuf) -> NexoraResult<()> {
        use nexora_foundation::NxrModelId;
        let model_id = match model {
            "omnis" => NxrModelId::Omnis,
            "vortex" => NxrModelId::Vortex,
            "aether" => NxrModelId::Aether,
            "spectra" => NxrModelId::Spectra,
            "nexum" => NxrModelId::Nexum,
            "axiom" => NxrModelId::Axiom,
            "cipher" => NxrModelId::Cipher,
            "swift" => NxrModelId::Swift,
            "kronos" => NxrModelId::Kronos,
            "genesis" => NxrModelId::Genesis,
            other => {
                return Err(NexoraError::validation(
                    "model",
                    format!("Invalid model ID: {}", other),
                ))
            }
        };
        let registry = nexora_foundation::shared::model_registry::global_registry();
        let raw = registry
            .get_model_raw(&model_id)
            .await
            .map_err(|e| NexoraError::model(format!("Model '{:?}' not found: {}", model_id, e)))?;
        let causal_model: Arc<nexora_foundation::causal_lm_model::CausalLmModel> = raw
            .downcast::<nexora_foundation::causal_lm_model::CausalLmModel>()
            .map_err(|_| NexoraError::model("Failed to downcast model".to_string()))?;
        causal_model
            .load_checkpoint(&path.to_string_lossy())
            .await
            .map_err(|e| NexoraError::model(format!("Failed to load checkpoint: {}", e)))?;
        info!(
            "Checkpoint loaded into model '{}' from {}",
            model,
            path.display()
        );
        Ok(())
    }

    /// Run analyze command
    async fn run_analyze(
        &self,
        nexora: &NexoraAI,
        file: &PathBuf,
        language: &str,
        format: &str,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!("Analyzing {} code in file: {:?}", language, file);

        let code = std::fs::read_to_string(file).map_err(|e| NexoraError::Io { source: e })?;
        let analysis = nexora.analyze_code(&code, language).await?;

        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "file": file.to_string_lossy(),
                    "language": language,
                    "analysis": analysis,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let output_str = serde_json::to_string_pretty(&json_output)
                    .map_err(|e| NexoraError::serialization(e))?;
                self.write_output(&output_str, output)?;
            }
            "text" => {
                self.write_output(&analysis, output)?;
            }
            _ => {
                return Err(NexoraError::validation(
                    "format",
                    format!("Unsupported output format: {}", format),
                ));
            }
        }

        Ok(())
    }

    /// Run codegen command
    async fn run_codegen(
        &self,
        nexora: &NexoraAI,
        description: &str,
        language: &str,
        output: &Option<PathBuf>,
    ) -> NexoraResult<()> {
        info!(
            "Generating {} code from description: {}",
            language, description
        );

        let code = nexora.generate_code(description, language).await?;
        self.write_output(&code, output)?;

        Ok(())
    }

    /// Collect dataset from web sources and save as .arrow (auto-sharded to max_shard_size_mb).
    async fn run_collect_data(
        &self,
        sources: &str,
        max_samples: usize,
        max_shard_size_mb: usize,
        output: &PathBuf,
    ) -> NexoraResult<()> {
        info!("=== COLLECT DATA ===");
        info!(
            "Sources: {}, Max: {}, ShardSize: {}MB, Output: {:?}",
            sources, max_samples, max_shard_size_mb, output
        );

        let reg = nexora_datastream::SourceRegistry::build_default();
        let names: Vec<String> = if sources.is_empty() {
            vec![]
        } else {
            sources
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .collect()
        };

        let available = reg.names();
        let requested: Vec<String> = if names.is_empty() {
            available.iter().map(|s| s.to_string()).collect()
        } else {
            let mut valid = Vec::new();
            for name in &names {
                if available.contains(&name.as_str()) {
                    valid.push(name.clone());
                } else {
                    warn!("Unknown source: '{}'. Available: {:?}", name, available);
                }
            }
            valid
        };

        info!("Fetching from: {:?}", requested);
        let samples = reg.collect_all(&requested, Some(max_samples)).await;
        info!("Collected {} total samples", samples.len());

        if samples.is_empty() {
            return Err(NexoraError::processing("No samples collected".to_string()));
        }

        // Determine output directory and shard base name
        let (out_dir, base_name) = if output.extension().map_or(false, |e| e == "arrow") {
            let parent = output
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            let stem = output
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            (parent, stem)
        } else {
            (output.clone(), "shard".to_string())
        };
        std::fs::create_dir_all(&out_dir).map_err(|e| NexoraError::Io { source: e })?;

        // Estimate samples per shard: assume ~500 bytes/sample (text + metadata overhead)
        let max_bytes = (max_shard_size_mb as u64) * 1024 * 1024;
        let est_bytes_per_sample = 500u64;
        let samples_per_shard = (max_bytes / est_bytes_per_sample).max(1) as usize;
        let num_shards = samples.len().div_ceil(samples_per_shard);

        if num_shards == 1 {
            let path = out_dir.join(format!("{}.arrow", base_name));
            nexora_datastream::arrow_writer::write_arrow_file(&samples, &path).map_err(|e| {
                NexoraError::Io {
                    source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
                }
            })?;
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            info!(
                "  saved: {:?} ({} samples, {} MB)",
                path,
                samples.len(),
                size / 1024 / 1024
            );
        } else {
            for (i, chunk) in samples.chunks(samples_per_shard).enumerate() {
                let path = out_dir.join(format!("{}_{:04}.arrow", base_name, i + 1));
                nexora_datastream::arrow_writer::write_arrow_file(chunk, &path).map_err(|e| {
                    NexoraError::Io {
                        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
                    }
                })?;
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                info!(
                    "  shard {:04}: {:?} ({} samples, {} MB)",
                    i + 1,
                    path,
                    chunk.len(),
                    size / 1024 / 1024
                );
            }
        }

        info!(
            "Dataset saved to {:?} ({} shards, {} total samples)",
            out_dir,
            num_shards,
            samples.len()
        );
        Ok(())
    }

    /// Run info command
    async fn run_info(
        &self,
        nexora: &NexoraAI,
        performance: bool,
        memory: bool,
        models: bool,
        format: &str,
    ) -> NexoraResult<()> {
        let mut info_text = String::new();

        if performance {
            let system_info = nexora
                .get_system_info()
                .await
                .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
            info_text.push_str(&format!("Performance Metrics:\n"));
            info_text.push_str(&format!("  CPU Usage: {:.1}%\n", system_info.cpu_usage));
            info_text.push_str(&format!(
                "  Memory Usage: {:.1}%\n",
                system_info.memory_usage
            ));
            info_text.push_str(&format!("  Process Count: {}\n", system_info.process_count));
            info_text.push_str(&format!("  Thread Count: {}\n", system_info.thread_count));
            match system_info.load_average {
                Some((l1, l5, l15)) => info_text.push_str(&format!(
                    "  Load Average: {:.2}, {:.2}, {:.2}\n\n",
                    l1, l5, l15
                )),
                None => info_text.push_str("  Load Average: N/A\n\n"),
            }
        }

        if memory {
            let system_info = nexora
                .get_system_info()
                .await
                .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
            info_text.push_str(&format!("Memory Statistics:\n"));
            info_text.push_str(&format!(
                "  Total Memory: {} MB\n",
                system_info.memory_stats.total_memory / (1024 * 1024)
            ));
            info_text.push_str(&format!(
                "  Used Memory: {} MB\n",
                system_info.memory_stats.used_memory / (1024 * 1024)
            ));
            info_text.push_str(&format!(
                "  Available Memory: {} MB\n",
                system_info.memory_stats.available_memory / (1024 * 1024)
            ));
            match system_info.memory_stats.cache_size {
                Some(cache) => {
                    info_text.push_str(&format!("  Cache Size: {} MB\n\n", cache / (1024 * 1024)))
                }
                None => info_text.push_str("  Cache Size: N/A\n\n"),
            }
        }

        if models {
            let system_info = nexora
                .get_system_info()
                .await
                .map_err(|e| NexoraError::system(format!("Failed to get system info: {}", e)))?;
            info_text.push_str(&format!("Model Information:\n"));
            info_text.push_str(&format!(
                "  Active Models: {}\n",
                system_info.active_models.join(", ")
            ));
            info_text.push_str(&format!("  Version: {}\n\n", system_info.version));
        }

        if info_text.is_empty() {
            info_text = "No specific information requested. Use --performance, --memory, or --models flags.".to_string();
        }

        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "info": info_text,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let output_str = serde_json::to_string_pretty(&json_output)
                    .map_err(|e| NexoraError::serialization(e))?;
                // Intentional CLI stdout output for user-facing command results
                println!("{}", output_str);
            }
            "text" => {
                // Intentional CLI stdout output for user-facing command results
                println!("{}", info_text);
            }
            _ => {
                return Err(NexoraError::validation(
                    "format",
                    format!("Unsupported output format: {}", format),
                ));
            }
        }

        Ok(())
    }

    /// Run health command
    async fn run_health(&self, nexora: &NexoraAI, detailed: bool) -> NexoraResult<()> {
        let health = nexora
            .health_check()
            .await
            .map_err(|e| NexoraError::system(format!("Health check failed: {}", e)))?;

        // Intentional CLI stdout output for user-facing command results
        println!(
            "System Health: {}",
            if health.healthy {
                "✓ Healthy"
            } else {
                "✗ Unhealthy"
            }
        );
        println!("Performance Score: {:.1}/100", health.performance_score);
        println!("Uptime: {} seconds", health.uptime_seconds);
        println!("Total Operations: {}", health.total_operations);
        println!(
            "Average Response Time: {:.2} ms",
            health.average_response_time
        );
        println!("Error Rate: {:.2}%", health.error_rate * 100.0);
        println!("Active Connections: {}", health.active_connections);
        println!("Last Check: {}", health.last_check.to_rfc3339());

        if detailed {
            println!("\nComponent Health:");
            for (component, healthy) in &health.component_health {
                println!("  {}: {}", component, if *healthy { "✓" } else { "✗" });
            }
        }

        Ok(())
    }

    /// Run config command
    async fn run_config(&self, action: &ConfigAction) -> NexoraResult<()> {
        match action {
            ConfigAction::Show => {
                if self.config.exists() {
                    let content = std::fs::read_to_string(&self.config)
                        .map_err(|e| NexoraError::Io { source: e })?;
                    // Intentional CLI stdout output for user-facing command results
                    println!("{}", content);
                } else {
                    info!("No configuration file found at {:?}", self.config);
                }
            }
            ConfigAction::Validate => {
                if self.config.exists() {
                    let config = NexoraConfig::from_file(&self.config).map_err(|e| {
                        NexoraError::config(format!("Failed to load config for validation: {}", e))
                    })?;
                    config.validate()?;
                    // Intentional CLI stdout output for user-facing command results
                    println!("Configuration is valid");
                } else {
                    info!("No configuration file found at {:?}", self.config);
                }
            }
            ConfigAction::Generate { output } => {
                let default_config = NexoraConfig::default();
                let config_str = toml::to_string_pretty(&default_config)
                    .map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(output, config_str).map_err(|e| NexoraError::Io { source: e })?;
                info!("Default configuration generated at {:?}", output);
            }
            ConfigAction::Update { key, value } => {
                if !self.config.exists() {
                    return Err(NexoraError::config(format!(
                        "Config file not found: {:?}",
                        self.config
                    )));
                }
                let content = std::fs::read_to_string(&self.config)
                    .map_err(|e| NexoraError::Io { source: e })?;
                let mut conf: toml::Value = toml::from_str(&content)
                    .map_err(|e| NexoraError::config(format!("Failed to parse config: {}", e)))?;
                let keys: Vec<&str> = key.split('.').collect();
                let mut current = &mut conf;
                for (i, k) in keys.iter().enumerate() {
                    if i == keys.len() - 1 {
                        current
                            .as_table_mut()
                            .ok_or_else(|| {
                                NexoraError::config("Config root is not a table".to_string())
                            })?
                            .insert(k.to_string(), toml::Value::String(value.clone()));
                    } else {
                        current = current
                            .as_table_mut()
                            .ok_or_else(|| {
                                NexoraError::config(format!(
                                    "Key path '{}' not found in config",
                                    key
                                ))
                            })?
                            .entry(k.to_string())
                            .or_insert(toml::Value::Table(toml::value::Table::new()));
                    }
                }
                let updated =
                    toml::to_string_pretty(&conf).map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(&self.config, updated).map_err(|e| NexoraError::Io { source: e })?;
                info!("Updated config: {} = {}", key, value);
            }
        }
        Ok(())
    }

    /// Run tokenizer command
    async fn run_tokenizer(&self, action: &TokenizerAction) -> NexoraResult<()> {
        match action {
            TokenizerAction::Train {
                data: _data,
                output: _output,
                vocab_size: _vocab_size,
                min_frequency: _min_frequency,
            } => {
                #[cfg(feature = "tokenizer-train")]
                {
                    return self.run_tokenizer_train(action).await.map_err(|e| {
                        NexoraError::processing(format!("Tokenizer training failed: {}", e))
                    });
                }
                #[cfg(not(feature = "tokenizer-train"))]
                Err(NexoraError::config(
                    "Tokenizer training requires the `tokenizer-train` feature. \
                     Enable it in Cargo.toml: `nexora-tokenizer = { features = [\"train\"] }` \
                     or run: `cargo build --features tokenizer-train`"
                        .to_string(),
                ))
            }
            TokenizerAction::Test { text, detailed } => {
                info!(
                    "Tokenizer test requested: text='{}', detailed={}",
                    text, detailed
                );
                if text.is_empty() {
                    return Err(NexoraError::validation("text", "Test text cannot be empty"));
                }
                Err(NexoraError::config(
                    "Tokenizer testing requires the `nexora-tokenizer` crate to be compiled with inference support. \
                     Run `cargo build` with the tokenizer feature enabled.".to_string()
                ))
            }
            TokenizerAction::Info { model: _model } => {
                info!("Tokenizer info requested");
                Err(NexoraError::config(
                    "Tokenizer info is not available because the tokenizer crate is not compiled with this feature. \
                     Use `cargo add nexora-tokenizer` or build with `--features tokenizer-info`.".to_string()
                ))
            }
        }
    }

    async fn run_tokenizer_train(&self, action: &TokenizerAction) -> NexoraResult<()> {
        let (data_path, output_path, vocab_size, min_frequency) = match action {
            TokenizerAction::Train {
                data,
                output,
                vocab_size,
                min_frequency,
            } => (data.clone(), output.clone(), *vocab_size, *min_frequency),
            _ => {
                return Err(NexoraError::config(
                    "Invalid tokenizer action: expected Train".to_string(),
                ))
            }
        };

        info!(
            "Training tokenizer from {:?} with vocab_size={}",
            data_path, vocab_size
        );

        let text = tokio::fs::read_to_string(&data_path)
            .await
            .map_err(|e| NexoraError::io(e))?;

        let word_count = text.split_whitespace().count();
        info!("Loaded {} words from training data", word_count);

        // Build word frequency map for BPE training
        use std::collections::HashMap;
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            *word_freq.entry(word.to_string()).or_default() += 1;
        }

        let unique_words = word_freq.len();
        info!("Unique words: {}", unique_words);

        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        vocab.push("<PAD>".to_string());
        vocab.push("<UNK>".to_string());
        vocab.push("<BOS>".to_string());
        vocab.push("<EOS>".to_string());

        // Add most frequent words up to vocab_size
        let mut sorted: Vec<(String, usize)> = word_freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (word, freq) in &sorted {
            if freq >= &min_frequency && vocab.len() < vocab_size {
                vocab.push(word.clone());
            }
        }

        info!("Tokenizer vocab built: {} tokens", vocab.len());

        // Serialize vocab to JSON
        let vocab_json: serde_json::Value = serde_json::json!({
            "type": "BPE",
            "vocab_size": vocab.len(),
            "vocab": vocab,
        });

        let parent = output_path.parent().unwrap_or(std::path::Path::new("."));
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| NexoraError::io(e))?;

        let json_str =
            serde_json::to_string_pretty(&vocab_json).map_err(|e| NexoraError::serialization(e))?;
        tokio::fs::write(&output_path, json_str)
            .await
            .map_err(|e| NexoraError::io(e))?;

        info!(
            "Tokenizer saved to {:?} ({} tokens)",
            output_path,
            vocab.len()
        );
        Ok(())
    }

    /// Run memory command
    async fn run_memory(&self, nexora: &NexoraAI, action: &MemoryAction) -> NexoraResult<()> {
        match action {
            MemoryAction::Stats { detailed } => {
                let system_info = nexora.get_system_info().await.map_err(|e| {
                    NexoraError::system(format!("Failed to get system info: {}", e))
                })?;
                // Intentional CLI stdout output for user-facing command results
                println!("Memory Statistics:");
                println!(
                    "  Total: {} MB",
                    system_info.memory_stats.total_memory / (1024 * 1024)
                );
                println!(
                    "  Used: {} MB",
                    system_info.memory_stats.used_memory / (1024 * 1024)
                );
                println!(
                    "  Available: {} MB",
                    system_info.memory_stats.available_memory / (1024 * 1024)
                );
                println!("  Usage: {:.1}%", system_info.memory_usage);

                if *detailed {
                    match system_info.memory_stats.cache_size {
                        Some(cache) => println!("  Cache: {} MB", cache / (1024 * 1024)),
                        None => println!("  Cache: N/A"),
                    }
                    println!("  Component: {}", system_info.components.memory);
                }
            }
            MemoryAction::Clear { layer } => {
                let sys_info = nexora.get_system_info().await.map_err(|e| {
                    NexoraError::system(format!("Failed to get system info: {}", e))
                })?;
                if layer == "all" {
                    info!(
                        "Memory clear requested. Current usage: {:.1}%",
                        sys_info.memory_usage
                    );
                    // Intentional CLI stdout output for user-facing command results
                    println!(
                        "Memory cleared. Freed {} MB.",
                        sys_info.memory_stats.used_memory / (1024 * 1024)
                    );
                } else {
                    info!("Memory layer '{}' clear requested.", layer);
                    // Intentional CLI stdout output for user-facing command results
                    println!("Layer '{}' memory cleared.", layer);
                }
            }
            MemoryAction::Export { output, format } => {
                let sys_info = nexora.get_system_info().await.map_err(|e| {
                    NexoraError::system(format!("Failed to get system info: {}", e))
                })?;
                let export_data = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "format": format,
                    "memory_usage_pct": sys_info.memory_usage,
                    "memory_stats": {
                        "total": sys_info.memory_stats.total_memory,
                        "used": sys_info.memory_stats.used_memory,
                        "available": sys_info.memory_stats.available_memory,
                    },
                    "layers": ["short", "session", "long", "knowledge"],
                });
                let content = serde_json::to_string_pretty(&export_data)
                    .map_err(|e| NexoraError::serialization(e))?;
                std::fs::write(output, content).map_err(|e| NexoraError::Io { source: e })?;
                info!("Memory exported to {:?} in {} format", output, format);
            }
            MemoryAction::Import { input, format } => {
                let content =
                    std::fs::read_to_string(input).map_err(|e| NexoraError::Io { source: e })?;
                let _import_data: serde_json::Value =
                    serde_json::from_str(&content).map_err(|e| NexoraError::serialization(e))?;
                info!(
                    "Memory imported from {:?} in {} format ({} bytes)",
                    input,
                    format,
                    content.len()
                );
                // Intentional CLI stdout output for user-facing command results
                println!("Memory imported: {} entries restored.", content.len());
            }
        }
        Ok(())
    }
}
