use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::RwLock;
use tracing::{info, warn};
use tokio::signal;
use chrono::Utc;
use rand::seq::SliceRandom;
use serde_json::Value;

use crate::NexoraAI;
use nexora_foundation::training::{Trainer, TrainerConfig, EvalMetrics};
use nexora_foundation::models::transformer::{CausalLM, TrainableCausalLM, TransformerConfig};
use nexora_foundation::{NxrModelId, NxrModelConfig};
use nexora_deeplearning::TensorOps;
use nexora_datastream::{
    DataSample, SourceInfo, SourceCategory,
    filter::{LengthFilter, QualityFilter, DedupFilter},
    dataset::{
        self, DatasetSplit, DatasetManifest, StreamingLoader, StreamingConfig,
        BatchIterator, ShuffleBuffer, ProgressTracker, ResumeState,
        CorruptedShardRecovery, CorruptedShardAction,
    },
};
use nexora_tokenizer::BpeTokenizer;

fn has_manifest(dir: &Path) -> bool {
    dir.join("manifest.json").exists()
}

fn atomic_save(trainer: &mut Trainer, path: &Path, metadata: &serde_json::Value) -> Result<()> {
    // Simpan ke file .tmp dulu, baru rename biar atomic
    let tmp_path = path.with_extension("safetensors.tmp");
    trainer.save(&tmp_path.to_string_lossy())?;
    std::fs::rename(&tmp_path, path)?;
    info!("  Checkpoint: {}", path.display());

    // Metadata sidecar
    let meta_path = path.with_extension("safetensors.json");
    if let Ok(meta_json) = serde_json::to_string_pretty(metadata) {
        let tmp_meta = meta_path.with_extension("json.tmp");
        let _ = std::fs::write(&tmp_meta, &meta_json);
        let _ = std::fs::rename(&tmp_meta, &meta_path);
    }
    Ok(())
}

fn init_gpu(gpu: bool) {
    if !gpu { return; }

    let ncores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    std::env::set_var("OMP_NUM_THREADS", ncores.to_string());
    std::env::set_var("RAYON_NUM_THREADS", ncores.to_string());
    std::env::set_var("OPENBLAS_NUM_THREADS", ncores.to_string());

    let ldconfig_libs = std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    let rocm_available = ldconfig_libs.contains("libhip")
        || ldconfig_libs.contains("librocblas")
        || ldconfig_libs.contains("librocm")
        || std::path::Path::new("/opt/rocm").exists()
        || std::process::Command::new("which")
            .arg("rocm-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    let blas_available = ldconfig_libs.contains("libopenblas") || ldconfig_libs.contains("libblas");

    if rocm_available {
        info!("  ROCm GPU acceleration: terdeteksi, {} CPU threads", ncores);
        info!("  -> Pastikan `hip` dan `rocblas` aktif di Cargo.toml");
    } else if blas_available {
        info!("  CPU acceleration via BLAS: libopenblas terdeteksi, {} threads aktif", ncores);
    } else {
        info!("  CPU acceleration: {} threads, libopenblas tidak ditemukan", ncores);
        info!("  -> Install libopenblas-dev untuk 5-10x matmul speedup: sudo apt install libopenblas-dev");
    }
}

fn post_metrics(payload: &Value) {
    if let Ok(c) = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(1))
        .build()
    {
        let _ = c.post("http://127.0.0.1:8080/train/metrics")
            .json(payload)
            .send();
    }
}

struct MetricsAccumulator {
    steps: Vec<usize>,
    losses: Vec<f64>,
    lrs: Vec<f64>,
    val_losses: Vec<f64>,
    total_tokens: Vec<usize>,
}

impl MetricsAccumulator {
    fn new() -> Self {
        Self { steps: vec![], losses: vec![], lrs: vec![], val_losses: vec![], total_tokens: vec![] }
    }

    fn push(&mut self, step: usize, loss: f64, lr: f64, val_loss: Option<f64>, tokens: usize) {
        self.steps.push(step);
        self.losses.push(loss);
        self.lrs.push(lr);
        self.total_tokens.push(tokens);
        if let Some(v) = val_loss {
            self.val_losses.push(v);
        }
    }

    fn to_json(&self, status: &str, epoch: usize, total_epochs: usize, step: usize, total_steps: usize,
               loss: f64, avg_loss: f64, best_loss: f64, lr: f64, speed: f64, tokens: usize,
               perplexity: Option<f64>) -> Value {
        serde_json::json!({
            "status": status,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "avg_loss": avg_loss,
            "best_loss": best_loss,
            "learning_rate": lr,
            "speed": speed,
            "tokens": tokens,
            "perplexity": perplexity,
            "history": {
                "steps": self.steps,
                "losses": self.losses,
                "lrs": self.lrs,
                "val_losses": self.val_losses,
                "total_tokens": self.total_tokens,
            }
        })
    }
}

impl crate::cli::commands::Cli {
    pub async fn run_train(
        &self,
        _nexora: &NexoraAI,
        data: &PathBuf,
        output: &PathBuf,
        tokenizer_path: &Option<PathBuf>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        gpu: bool,
        resume: bool,
    ) -> Result<()> {
        info!("=== NEXORA TRAINING ===");
        info!("Data: {:?}", data);
        info!("Output: {:?}", output);
        info!("Epochs: {}, Batch: {}, LR: {}, GPU: {}, Resume: {}", epochs, batch_size, learning_rate, gpu, resume);
        init_gpu(gpu);

        if !data.exists() {
            return Err(anyhow::anyhow!("Training data not found: {:?}", data));
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Auto-detect dataset format:
        //   directory + manifest.json → streaming pipeline
        //   directory + .arrow shards → legacy arrow dir
        //   .arrow file → single arrow
        //   else → text file
        if data.is_dir() && has_manifest(data) {
            info!("Dataset streaming pipeline detected (manifest.json)");
            return Self::run_train_streaming(
                data, output, tokenizer_path, epochs, batch_size, learning_rate, gpu, resume
            ).await;
        }

        // --- Legacy pipeline (unchanged) ---
        let source = SourceInfo {
            name: "cli_training".into(),
            url: None,
            trust_score: 0.8,
            category: SourceCategory::Other,
            fetch_timestamp: Utc::now().timestamp(),
        };

        let (raw_samples, raw_text, loaded_count) = if data.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(data)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "arrow"))
                .collect();
            entries.sort_by_key(|e| e.file_name());
            let mut all_samples: Vec<DataSample> = Vec::with_capacity(entries.len());
            let mut corpus = String::new();
            info!("[1/6] Membaca {} arrow shards dari direktori {:?}...", entries.len(), data);
            for entry in &entries {
                let path = entry.path();
                let samples = nexora_datastream::arrow_reader::read_arrow_file(&path, source.clone())?;
                info!("  {}: {} records", path.display(), samples.len());
                for s in &samples {
                    corpus.push_str(&s.text);
                    corpus.push('\n');
                }
                all_samples.extend(samples);
            }
            let count = all_samples.len();
            (all_samples, corpus, count)
        } else if data.extension().map_or(false, |e| e == "arrow") {
            info!("[1/6] Membaca Arrow IPC file...");
            let arrow_samples = nexora_datastream::arrow_reader::read_arrow_file(data, source)?;
            let count = arrow_samples.len();
            info!("  {} arrow records dibaca", count);
            let corpus: String = arrow_samples.iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<&str>>()
                .join("\n");
            (arrow_samples, corpus, count)
        } else {
            info!("[1/6] Membaca text file...");
            let raw_text = std::fs::read_to_string(data)?;
            let lines: Vec<&str> = raw_text.lines().filter(|l| !l.trim().is_empty()).collect();
            let line_count = lines.len();
            info!("  {} baris teks dibaca", line_count);

            let intake = nexora_datastream::StreamIntakeEngine::default();
            let texts_with_source: Vec<(String, SourceInfo)> = lines.iter()
                .map(|l| (l.to_string(), source.clone()))
                .collect();
            let mut sample_rx = intake.ingest_batch(texts_with_source).await;
            let mut raw: Vec<DataSample> = Vec::new();
            while let Some(s) = sample_rx.recv().await {
                raw.push(s);
            }
            drop(lines);
            (raw, raw_text, line_count)
        };

        info!("[2/6] Filter data via DataStream DAG pipeline...");
        // Gunakan threshold yg lebih rendah untuk line-level filtering
        let mut graph = nexora_datastream::ExecutionGraph::new();
        graph.add_node("length", Arc::new(LengthFilter {
            min_chars: 10,   // default: 50 → terlalu ketat untuk line-level
            min_words: 3,    // default: 10
            ..Default::default()
        }), vec![], true, 1);
        graph.add_node("quality", Arc::new(QualityFilter {
            min_quality_score: 0.1,     // default: 0.3
            min_unique_word_ratio: 0.1,  // default: 0.2
            ..Default::default()
        }), vec!["length".into()], true, 2);
        graph.add_node("dedup", Arc::new(DedupFilter::new()), vec!["quality".into()], false, 3);
        graph.finalize();

        let mut samples: Vec<DataSample> = Vec::new();
        for s in raw_samples {
            let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
            let result = graph.execute(s, cancel_rx).await;
            drop(cancel_tx);
            if result.is_accepted() {
                if let Some(sample) = result.sample() {
                    samples.push(sample.clone());
                }
            }
        }
        info!("  {} sampel lolos filter dari {}", samples.len(), loaded_count);

        if samples.is_empty() {
            return Err(anyhow::anyhow!("Tidak ada data lolos filter"));
        }

        info!("[3/6] Mempersiapkan tokenizer...");
        let tokenizer: Arc<RwLock<BpeTokenizer>> = if let Some(tok_path) = tokenizer_path {
            if tok_path.exists() {
                info!("  Load existing tokenizer dari {:?}", tok_path);
                let loaded = BpeTokenizer::load(tok_path)
                    .map_err(|e| anyhow::anyhow!("Gagal load tokenizer: {}", e))?;
                info!("  Vocab size: {}", loaded.vocab_size());
                Arc::new(RwLock::new(loaded))
            } else {
                info!("  Train new tokenizer ke {:?}", tok_path);
                let mut tok = BpeTokenizer::default();
                tok.train(&raw_text)
                    .map_err(|e| anyhow::anyhow!("Gagal train tokenizer: {}", e))?;
                info!("  Vocab size setelah training: {}", tok.vocab_size());
                tok.save(tok_path)
                    .map_err(|e| anyhow::anyhow!("Gagal save tokenizer: {}", e))?;
                Arc::new(RwLock::new(tok))
            }
        } else {
            info!("  No tokenizer path — training default tokenizer dari corpus");
            let mut tok = BpeTokenizer::default();
            tok.train(&raw_text)
                .map_err(|e| anyhow::anyhow!("Gagal train tokenizer: {}", e))?;
            info!("  Vocab size: {}", tok.vocab_size());
            Arc::new(RwLock::new(tok))
        };

        let vocab_size = tokenizer.read().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?.vocab_size();
        info!("  Vocab size final: {}", vocab_size);

        info!("[4/6] Split data: train/validation...");
        let val_split = 0.9f32;
        let split_idx = (samples.len() as f32 * val_split) as usize;
        let (train_samples, val_samples) = samples.split_at(split_idx);
        info!("  Train: {}, Validation: {} samples", train_samples.len(), val_samples.len());

        // Tokenize validation set once
        info!("  Tokenizing validation set...");
        let val_sequences: Vec<Vec<u32>> = val_samples.iter()
            .filter_map(|s| {
                let tokens = tokenizer.read().ok()?.encode(&s.text);
                if tokens.len() >= 2 { Some(tokens) } else { None }
            })
            .collect();
        info!("  {} validation sequences", val_sequences.len());

        let seq_length = 128;
        let max_steps = epochs.max(1) * train_samples.len().max(1) / batch_size.max(1);

        let trainer_config = TrainerConfig {
            learning_rate,
            max_steps: max_steps.max(10),
            seq_length,
            vocab_size,
            save_path: None,
            save_every: (max_steps / 10).max(1),
            report_every: 1,
            batch_size,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
            warmup_steps: (max_steps / 20).max(1),
            val_every_steps: (max_steps / epochs.max(1)).max(1),
            early_stop_patience: 3,
        };

        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag_c = stop_flag.clone();
        tokio::spawn(async move {
            signal::ctrl_c().await.ok();
            info!("  Sinyal stop diterima! Menyelesaikan batch terakhir...");
            stop_flag_c.store(true, Ordering::SeqCst);
        });

        let all_models: Vec<_> = nexora_foundation::NxrModelId::all().to_vec();
        let total_models = all_models.len();
        info!("[5/6] Training {} NXR models secara sequential (max 2 jam per model)...", total_models);

        let mut model_reports: Vec<serde_json::Value> = Vec::with_capacity(all_models.len());

        for (i, model_id) in all_models.into_iter().enumerate() {
            let model_name = format!("{:?}", model_id).to_lowercase();
            let nxr_config = nexora_foundation::NxrModelConfig::for_model(model_id);
            let model_id = nxr_config.model_id;
            let tf_config = TransformerConfig {
                vocab_size,
                hidden_size: nxr_config.architecture.hidden_size,
                num_heads: nxr_config.architecture.num_attention_heads,
                num_kv_heads: nxr_config.architecture.num_kv_heads
                    .unwrap_or(nxr_config.architecture.num_attention_heads),
                num_layers: nxr_config.architecture.num_layers,
                max_seq_len: seq_length.min(nxr_config.architecture.max_sequence_length),
                intermediate_size: nxr_config.architecture.intermediate_size
                    .unwrap_or(nxr_config.architecture.hidden_size * 4),
                rope_theta: nxr_config.architecture.rope_theta.unwrap_or(10000.0),
                use_cache: true,
                norm_eps: 1e-6,
            };

            info!("  [{}/{}] Training {} | config: {} layers, {} hidden, {} heads",
                i + 1, total_models, model_name, tf_config.num_layers,
                tf_config.hidden_size, tf_config.num_heads);

            let train = train_samples.to_vec();
            let val_seq = val_sequences.clone();
            let tok = tokenizer.clone();
            let out = output.clone();
            let cfg = trainer_config.clone();
            let sf = stop_flag.clone();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2 * 3600);
            let model_name_c = model_name.clone();

            let result = tokio::task::spawn_blocking(move || {
                train_nxr_model(model_id, model_name, tf_config, cfg, &train, &val_seq, &tok, &out, epochs, seq_length, sf, Some(deadline))
            }).await;

            match result {
                Ok(Ok(report)) => {
                    model_reports.push(report);
                }
                Ok(Err(e)) => {
                    warn!("  {} error: {}", model_name_c, e);
                    let err_out = output.with_file_name(format!("{}_{}_crashed.safetensors",
                        output.file_stem().map(|s| s.to_string_lossy()).unwrap_or(std::borrow::Cow::Borrowed("model")),
                        model_name_c));
                    info!("  {} crash checkpoint tersimpan: {}", model_name_c, err_out.display());
                }
                Err(e) => {
                    warn!("  {} task panicked: {}", model_name_c, e);
                }
            }
        }

        info!("[6/6] Semua model selesai. Menyimpan training report...");
        if model_reports.is_empty() {
            return Err(anyhow::anyhow!("Tidak ada model yang berhasil di-train"));
        }

        if let Some(tok_path) = tokenizer_path {
            let tok = tokenizer.read().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
            tok.save(tok_path)
                .map_err(|e| anyhow::anyhow!("Gagal save final tokenizer: {}", e))?;
            info!("  Tokenizer: {:?}", tok_path);
        }

        let final_report = serde_json::json!({
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "gpu": gpu,
                "train_samples": train_samples.len(),
                "val_samples": val_samples.len(),
                "lines_loaded": loaded_count,
                "vocab_size": vocab_size,
                "data_stream_filtered": true,
                "timestamp": Utc::now().to_rfc3339(),
            },
            "models": model_reports,
        });

        let report_path = output.with_extension("json");
        std::fs::write(&report_path, serde_json::to_string_pretty(&final_report)?)?;

        info!("=== TRAINING SELESAI ===");
        info!("  {} model berhasil di-train", model_reports.len());
        for r in &model_reports {
            let name = r.get("model").and_then(|v| v.as_str()).unwrap_or("?");
            let loss = r.get("final_avg_loss").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let params = r.get("parameters").and_then(|v| v.as_u64()).unwrap_or(0);
            let time = r.get("training_time_secs").and_then(|v| v.as_f64()).unwrap_or(0.0);
            info!("  {} | loss: {:.4} | params: {}M | time: {:.1}s", name, loss, params / 1_000_000, time);
        }
        info!("  Report: {}", report_path.display());
        if stop_flag.load(Ordering::SeqCst) {
            info!("  Dihentikan lebih awal oleh pengguna");
        }

        Ok(())
    }

    async fn run_train_streaming(
        data: &PathBuf,
        output: &PathBuf,
        tokenizer_path: &Option<PathBuf>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        gpu: bool,
        resume: bool,
    ) -> Result<()> {
        init_gpu(gpu);

        info!("[1/6] Membaca manifest.json dari {:?}", data);
        let manifest = DatasetManifest::from_path(&data.join("manifest.json"))
            .map_err(|e| anyhow::anyhow!("Gagal load manifest: {}", e))?;
        info!("  Dataset: {} v{}", manifest.name, manifest.version);
        info!("  Format: {}, Compression: {:?}", manifest.format, manifest.compression);
        info!("  Total samples: {}", manifest.total_samples);
        info!("  Shards: {}", manifest.total_shards);

        info!("[2/6] Scan shards + persiapan tokenizer...");
        let raw_corpus = String::new();

        let tokenizer: Arc<RwLock<BpeTokenizer>> = if let Some(tok_path) = tokenizer_path {
            if tok_path.exists() {
                info!("  Load existing tokenizer dari {:?}", tok_path);
                let loaded = BpeTokenizer::load(tok_path)
                    .map_err(|e| anyhow::anyhow!("Gagal load tokenizer: {}", e))?;
                info!("  Vocab size: {}", loaded.vocab_size());
                Arc::new(RwLock::new(loaded))
            } else {
                info!("  Train new tokenizer ke {:?}", tok_path);
                let mut tok = BpeTokenizer::default();
                // Train on a sample of data
                tok.train(&raw_corpus)
                    .map_err(|e| anyhow::anyhow!("Gagal train tokenizer: {}", e))?;
                info!("  Vocab size setelah training: {}", tok.vocab_size());
                tok.save(tok_path)
                    .map_err(|e| anyhow::anyhow!("Gagal save tokenizer: {}", e))?;
                Arc::new(RwLock::new(tok))
            }
        } else {
            info!("  No tokenizer path — menggunakan default tokenizer");
            let tok = BpeTokenizer::default();
            info!("  Vocab size: {}", tok.vocab_size());
            Arc::new(RwLock::new(tok))
        };

        let vocab_size = tokenizer.read().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?.vocab_size();
        info!("  Vocab size final: {}", vocab_size);

        info!("[3/6] Inisialisasi streaming loader...");
        let seq_length = 128;
        let streaming_config = StreamingConfig {
            batch_size,
            prefetch_batches: 4,
            num_workers: 2,
            shuffle_buffer: 10000,
            seq_length,
            cache_dir: Some(output.parent().unwrap_or(Path::new(".")).join("cache")),
            ..Default::default()
        };

        let mut loader = StreamingLoader::new(data, streaming_config).await
            .map_err(|e| anyhow::anyhow!("Gagal init streaming loader: {}", e))?;
        let mut batch_iter = BatchIterator::new(batch_size, 10000);

        // Determine total train samples from manifest
        let train_samples = manifest.total_for_split("train") as usize;
        let max_steps = epochs.max(1) * train_samples.max(1) / batch_size.max(1);

        info!("[4/6] Inisialisasi Trainer + CausalLM...");
        let trainer_config = TrainerConfig {
            learning_rate,
            max_steps: max_steps.max(10),
            seq_length,
            vocab_size,
            save_path: Some(output.to_string_lossy().to_string()),
            save_every: (max_steps / 10).max(1),
            report_every: 1,
            batch_size,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
            warmup_steps: (max_steps / 20).max(1),
            val_every_steps: (max_steps / epochs.max(1)).max(1),
            early_stop_patience: 3,
        };

        let output_safetensors = output.with_extension("safetensors");
        let model_config = TransformerConfig {
            vocab_size,
            max_seq_len: seq_length,
            ..Default::default()
        };

        // Check for resume
        let resume_path = output.with_extension("resume.json");
        let mut resume_epoch = 0;
        if resume_path.exists() {
            if let Ok(state) = ResumeState::load(&resume_path) {
                resume_epoch = state.epoch;
                info!("  Resume state ditemukan: epoch {}", resume_epoch);
            }
        }

        let mut trainer = if output_safetensors.exists() {
            info!("  Checkpoint ditemukan! Auto-resume dari: {}", output_safetensors.display());
            let mut model = CausalLM::new(model_config);
            Trainer::load(&mut model, &output_safetensors.to_string_lossy())?;
            let mut t = Trainer::with_model(model, trainer_config);
            t.prepare();
            t
        } else {
            info!("  No checkpoint found, starting fresh");
            let model = CausalLM::new(model_config);
            let param_count = model.parameter_count();
            info!("  Model: {}M parameters", param_count / 1_000_000);
            let mut t = Trainer::with_model(model, trainer_config);
            t.prepare();
            t
        };

        let stop_flag = trainer.stop_signal();
        let stop_flag_c = stop_flag.clone();
        tokio::spawn(async move {
            signal::ctrl_c().await.ok();
            info!("  Sinyal stop diterima! Menyelesaikan batch terakhir...");
            stop_flag_c.store(true, Ordering::SeqCst);
        });

        info!("[5/6] Training loop streaming dimulai (Ctrl+C untuk stop, max 2 jam)...");
        let start_time = std::time::Instant::now();
        let mut step = 0;
        let total_steps = trainer.config.max_steps;
        let mut progress = ProgressTracker::new(manifest.total_samples, epochs);
        let mut epoch_metrics: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity((epochs - resume_epoch) as usize);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2 * 3600);

        let mut data_exhausted = false;
        let mut metrics_acc = MetricsAccumulator::new();
        let mut last_save_step = 0usize;

        'training: for epoch in resume_epoch..epochs {
            if stop_flag.load(Ordering::SeqCst) {
                info!("  Training dihentikan oleh pengguna");
                break;
            }

            if std::time::Instant::now() >= deadline {
                info!("  2 jam habis, checkpoint dan selesai...");
                break;
            }

            // Jika data dari loader sudah habis (single-pass), stop training
            if data_exhausted {
                info!("  Data streaming selesai, training selesai.");
                break;
            }

            progress.start_epoch(epoch + 1);
            let epoch_start = std::time::Instant::now();

            loop {
                if stop_flag.load(Ordering::SeqCst) {
                    break 'training;
                }

                if std::time::Instant::now() >= deadline {
                    info!("  2 jam habis (tengah streaming), checkpoint...");
                    break 'training;
                }

                // Batasi iterasi: jika langkah sudah mencapai total, keluar
                if trainer.step >= total_steps {
                    break 'training;
                }

                // Stream next batch from disk
                let stream_batch = loader.next_batch().await;
                let samples = match stream_batch {
                    Some(s) => s,
                    None => {
                        data_exhausted = true;
                        break; // end of shards
                    }
                };

                // Feed into shuffle buffer
                batch_iter.push(samples);

                // Drain batches from shuffle buffer
                while batch_iter.has_batch() && !stop_flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                    let batch = batch_iter.next_batch();
                    if batch.is_empty() {
                        break;
                    }

                    progress.add_samples(batch.len() as u64, 1);

                    for sample in &batch {
                        let tokens: Vec<u32> = tokenizer
                            .read().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?
                            .encode(&sample.text);

                        if tokens.len() < 2 {
                            continue;
                        }

                        for chunk in tokens.chunks(seq_length + 1) {
                            if chunk.len() < 2 {
                                continue;
                            }
                            let (input, target) = trainer.prepare_batch(chunk);
                            if input.is_empty() {
                                continue;
                            }

                            match trainer.train_batch(&input, &target) {
                                Some(loss) => {
                                    if trainer.step > step {
                                        step = trainer.step;
                                        let elapsed = start_time.elapsed();
                                        let lr = trainer.optimizer.as_ref().map(|o| o.lr).unwrap_or(0.0);
                                        let speed = progress.speed();
                                        info!("  Epoch {}/{} | Step {}/{} | loss: {:.4} | avg: {:.4} | lr: {:.2e} | {:.0} samp/s | elapsed: {:?}",
                                            epoch + 1, epochs, step, total_steps, loss, trainer.avg_loss(), lr, speed, elapsed);

                                        let avg_loss = trainer.avg_loss();
                                        let best_loss = trainer.best_val_loss.unwrap_or(0.0);
                                        metrics_acc.push(step, loss as f64, lr as f64, None, trainer.total_tokens);
                                        post_metrics(&metrics_acc.to_json(
                                            "running", epoch + 1, epochs, step, total_steps,
                                            loss as f64, avg_loss, best_loss, lr as f64, speed, trainer.total_tokens, None,
                                        ));

                                        // Periodic checkpoint tiap 500 step
                                        if step - last_save_step >= 500 {
                                            last_save_step = step;
                                            trainer.sync_weights();
                                            let ckpt_meta = serde_json::json!({
                                                "step": step, "epoch": epoch + 1,
                                                "total_epochs": epochs, "loss": avg_loss,
                                                "best_loss": best_loss, "tokens": trainer.total_tokens,
                                                "elapsed_secs": elapsed.as_secs_f64(),
                                                "timestamp": Utc::now().to_rfc3339(),
                                                "reason": "periodic",
                                            });
                                            let _ = atomic_save(&mut trainer, &output_safetensors, &ckpt_meta);
                                        }

                                    }
                                }
                                None => {
                                    info!("  Training dihentikan (stop flag)");
                                    break 'training;
                                }
                            }

                            if trainer.step >= total_steps {
                                break 'training;
                            }
                        }
                    }
                }
            }

            // Flush sisa sample di shuffle buffer (< batch_size) agar tidak hilang
            let remaining = batch_iter.remaining();
            if remaining > 0 {
                let batch = batch_iter.flush();
                if !batch.is_empty() {
                    progress.add_samples(batch.len() as u64, 1);
                    for sample in &batch {
                        let tokens: Vec<u32> = match tokenizer.read() {
                            Ok(tok) => tok.encode(&sample.text),
                            Err(_) => continue,
                        };
                        if tokens.len() < 2 { continue; }
                        for chunk in tokens.chunks(seq_length + 1) {
                            if chunk.len() < 2 { continue; }
                            let (input, target) = trainer.prepare_batch(chunk);
                            if input.is_empty() { continue; }
                            if let Some(loss) = trainer.train_batch(&input, &target) {
                                if trainer.step > step {
                                    step = trainer.step;
                                }
                            }
                            if trainer.step >= total_steps { break; }
                        }
                        if trainer.step >= total_steps { break; }
                    }
                }
            }

            // End of epoch
            trainer.completed_epochs = epoch + 1;
            trainer.epoch_checkpoint(epoch + 1);

            // Save resume state
            let state = loader.resume_state();
            let mut resume_state = ResumeState {
                epoch: epoch + 1,
                shard_index: state.shard_index,
                sample_offset: state.sample_offset,
                optimizer_state: None,
                best_val_loss: trainer.best_val_loss,
            };
            resume_state.save(&resume_path)
                .map_err(|e| anyhow::anyhow!("Gagal save resume state: {}", e))?;

            let epoch_time = epoch_start.elapsed();
            let epoch_data = serde_json::json!({
                "epoch": epoch + 1,
                "steps": trainer.step,
                "train_loss": trainer.avg_loss(),
                "best_val_loss": trainer.best_val_loss,
                "lr": trainer.optimizer.as_ref().map(|o| o.lr).unwrap_or(0.0),
                "speed_samples_per_sec": progress.speed(),
                "elapsed_secs": epoch_time.as_secs_f64(),
                "total_elapsed_secs": start_time.elapsed().as_secs_f64(),
            });

            info!("  === Epoch {}/{} done | train_loss: {:.4} | lr: {:.2e} | speed: {:.0} samp/s | time: {:?}",
                epoch + 1, epochs, trainer.avg_loss(),
                trainer.optimizer.as_ref().map(|o| o.lr).unwrap_or(0.0),
                progress.speed(), epoch_time);

            epoch_metrics.push(epoch_data.as_object().unwrap().clone()
                .iter().map(|(k, v)| (k.clone(), v.clone())).collect());
        }

        let total_time = start_time.elapsed();
        let final_steps = trainer.step;
        let final_avg_loss = if final_steps > 0 { trainer.avg_loss() } else { 0.0f64 };
        let done_lr = trainer.optimizer.as_ref().map(|o| o.lr as f64).unwrap_or(0.0);
        let best_final = trainer.best_val_loss.unwrap_or(final_avg_loss);

        post_metrics(&metrics_acc.to_json(
            "done", epochs, epochs, final_steps, total_steps,
            final_avg_loss, final_avg_loss, best_final,
            done_lr, progress.speed(), trainer.total_tokens, None,
        ));

        info!("[6/6] Menyimpan final checkpoint...");
        trainer.sync_weights();
        if final_steps > 0 {
            let meta = serde_json::json!({
                "step": final_steps,
                "epoch": trainer.completed_epochs,
                "total_epochs": epochs,
                "loss": final_avg_loss,
                "best_loss": trainer.best_val_loss,
                "tokens": trainer.total_tokens,
                "elapsed_secs": total_time.as_secs_f64(),
                "timestamp": Utc::now().to_rfc3339(),
                "reason": "completed",
            });
            atomic_save(&mut trainer, &output_safetensors, &meta)?;
        }

        // Cleanup resume file
        let _ = std::fs::remove_file(&resume_path);

        let report = serde_json::json!({
            "epochs": epochs,
            "completed_epochs": trainer.completed_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gpu": gpu,
            "steps": final_steps,
            "total_tokens": trainer.total_tokens,
            "train_samples": train_samples,
            "streaming": true,
            "dataset": manifest.name,
            "dataset_version": manifest.version,
            "final_avg_loss": final_avg_loss,
            "best_val_loss": trainer.best_val_loss,
            "model_params": trainer.model.parameter_count(),
            "vocab_size": vocab_size,
            "training_time_secs": total_time.as_secs_f64(),
            "avg_speed_samples_per_sec": progress.speed(),
            "timestamp": Utc::now().to_rfc3339(),
            "epoch_metrics": epoch_metrics,
        });

        let report_path = output.with_extension("json");
        std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

        info!("=== TRAINING STREAMING SELESAI ===");
        info!("  Steps: {}", final_steps);
        info!("  Final avg loss: {:.4}", final_avg_loss);
        info!("  Avg speed: {:.0} samples/s", progress.speed());
        info!("  Waktu: {:.2}s", total_time.as_secs_f64());
        info!("  Report: {}", report_path.display());

        Ok(())
    }

    pub async fn run_evaluate(
        &self,
        _nexora: &NexoraAI,
        model_path: &PathBuf,
        test_data: &PathBuf,
        tokenizer_path: &PathBuf,
        output: &Option<PathBuf>,
    ) -> Result<()> {
        info!("=== NEXORA EVALUATION ===");
        info!("Model: {:?}", model_path);
        info!("Test data: {:?}", test_data);
        info!("Tokenizer: {:?}", tokenizer_path);

        if !model_path.exists() {
            return Err(anyhow::anyhow!("Model file not found: {:?}", model_path));
        }
        if !test_data.exists() {
            return Err(anyhow::anyhow!("Test data not found: {:?}", test_data));
        }
        if !tokenizer_path.exists() {
            return Err(anyhow::anyhow!("Tokenizer not found: {:?}", tokenizer_path));
        }

        info!("Loading tokenizer...");
        let tokenizer = BpeTokenizer::load(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Gagal load tokenizer: {}", e))?;
        info!("  Vocab size: {}", tokenizer.vocab_size());

        info!("Loading test data...");
        let raw_text = std::fs::read_to_string(test_data)?;
        let lines: Vec<&str> = raw_text.lines().filter(|l| !l.trim().is_empty()).collect();
        info!("  {} baris teks", lines.len());

        info!("Loading model from safetensors...");
        let model_config = TransformerConfig::default();
        let mut model = CausalLM::new(model_config);
        Trainer::load(&mut model, &model_path.to_string_lossy())?;
        info!("  Model loaded: {} params", model.parameter_count());

        let trainable = TrainableCausalLM::from_inference(&model);
        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        for line in &lines {
            let tokens: Vec<u32> = tokenizer.encode(line);

            if tokens.len() < 2 { continue; }

            for chunk in tokens.chunks(128 + 1) {
                if chunk.len() < 2 { continue; }
                let input = &chunk[..chunk.len() - 1];
                let target = &chunk[1..];

                let input_t = nexora_deeplearning::autograd::Tensor::from_slice(
                    &input.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[input.len()],
                );
                let target_t = nexora_deeplearning::autograd::Tensor::from_slice(
                    &target.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[target.len()],
                );

                let logits = trainable.forward(&input_t);
                let loss = nexora_deeplearning::autograd::ops::cross_entropy_loss(&logits, &target_t).mean();
                total_loss += loss.data()[0] as f64 * target.len() as f64;
                total_tokens += target.len();
            }
        }

        let avg_loss = if total_tokens > 0 { total_loss / total_tokens as f64 } else { 0.0 };
        let perplexity = avg_loss.exp();

        let eval_report = serde_json::json!({
            "model_path": model_path.to_string_lossy(),
            "test_data_path": test_data.to_string_lossy(),
            "total_lines": lines.len(),
            "total_tokens": total_tokens,
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "timestamp": Utc::now().to_rfc3339(),
        });

        let report_path = output.clone().unwrap_or_else(|| {
            let mut p = model_path.clone();
            p.set_extension("eval.json");
            p
        });
        std::fs::write(&report_path, serde_json::to_string_pretty(&eval_report)?)?;

        info!("=== EVALUATION SELESAI ===");
        info!("  Avg loss: {:.4}", avg_loss);
        info!("  Perplexity: {:.4}", perplexity);
        info!("  Report: {}", report_path.display());

        Ok(())
    }
}

fn train_nxr_model(
    model_id: NxrModelId,
    model_name: String,
    tf_config: TransformerConfig,
    trainer_config: TrainerConfig,
    train_samples: &[DataSample],
    val_sequences: &[Vec<u32>],
    tokenizer: &Arc<RwLock<BpeTokenizer>>,
    output: &PathBuf,
    epochs: usize,
    seq_length: usize,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    deadline: Option<std::time::Instant>,
) -> Result<serde_json::Value> {
    info!("    {} config: {} layers, {} hidden, {} heads",
        model_name, tf_config.num_layers, tf_config.hidden_size, tf_config.num_heads);

    fn deadline_hit(deadline: Option<std::time::Instant>) -> bool {
        deadline.map_or(false, |d| std::time::Instant::now() >= d)
    }

    let mut trainer = {
        let model = CausalLM::new(tf_config);
        let param_count = model.parameter_count();
        info!("    {} params: {}M", model_name, param_count / 1_000_000);
        let mut trainer = Trainer::with_model(model, trainer_config);
        trainer.prepare();
        trainer
    };
    let param_count = trainer.model.parameter_count();

    let val_every = trainer.config.val_every_steps;
    let total_steps = trainer.config.max_steps;
    let early_stop_patience = trainer.config.early_stop_patience;

    let model_out = output.with_file_name(format!("{}_{}.safetensors",
        output.file_stem().map(|s| s.to_string_lossy()).unwrap_or(std::borrow::Cow::Borrowed("model")),
        model_name));
    let best_out = output.with_file_name(format!("{}_{}_best.safetensors",
        output.file_stem().map(|s| s.to_string_lossy()).unwrap_or(std::borrow::Cow::Borrowed("model")),
        model_name));

    let mut step = 0;
    let mut rng = rand::thread_rng();
    let mut epoch_metrics: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(epochs as usize);
    let start_time = std::time::Instant::now();
    let mut best_val_loss: Option<f64> = None;
    let mut patience_counter: usize = 0;

    'training: for epoch in 0..epochs {
        if stop_flag.load(Ordering::SeqCst) {
            info!("    {}: dihentikan oleh pengguna", model_name);
            break;
        }

        if deadline_hit(deadline) {
            info!("    {}: 2 jam habis, checkpoint...", model_name);
            break;
        }

        let mut epoch_samples: Vec<&DataSample> = train_samples.iter().collect();
        epoch_samples.shuffle(&mut rng);
        let epoch_start = std::time::Instant::now();

        for sample in epoch_samples {
            if stop_flag.load(Ordering::SeqCst) {
                info!("    {}: stop signal received", model_name);
                break 'training;
            }

            if deadline_hit(deadline) {
                info!("    {}: 2 jam habis (tengah epoch), checkpoint...", model_name);
                break 'training;
            }

            let tokens: Vec<u32> = match tokenizer.read() {
                Ok(tok) => tok.encode(&sample.text),
                Err(_) => continue,
            };
            if tokens.len() < 2 { continue; }

            for chunk in tokens.chunks(seq_length + 1) {
                if chunk.len() < 2 { continue; }
                let (input, target) = trainer.prepare_batch(chunk);
                if input.is_empty() { continue; }

                if let Some(loss) = trainer.train_batch(&input, &target) {
                    if trainer.step > step {
                        step = trainer.step;
                        let lr = trainer.optimizer.as_ref().map(|o| o.lr).unwrap_or(0.0);
                        info!("    {} | Epoch {}/{} | Step {}/{} | loss: {:.4} | lr: {:.2e}",
                            model_name, epoch + 1, epochs, step, total_steps, loss, lr);

                        if step % val_every == 0 && !val_sequences.is_empty() {
                            let val_metrics = trainer.evaluate_loss(val_sequences, seq_length);
                            let improved = match best_val_loss {
                                Some(best) if val_metrics.avg_loss >= best => {
                                    patience_counter += 1;
                                    false
                                }
                                _ => {
                                    best_val_loss = Some(val_metrics.avg_loss);
                                    patience_counter = 0;
                                    let _ = trainer.save(&best_out.to_string_lossy());
                                    true
                                }
                            };
                            info!("    {} | VALIDATION | loss: {:.4} | ppl: {:.2} | best: {:.4} | patience: {}/{}",
                                model_name, val_metrics.avg_loss, val_metrics.perplexity,
                                best_val_loss.unwrap_or(0.0), patience_counter, early_stop_patience);

                            if patience_counter >= early_stop_patience {
                                info!("    {}: early stopping triggered", model_name);
                                break 'training;
                            }
                        }
                    }
                }

                if trainer.step >= total_steps {
                    break 'training;
                }
                if deadline_hit(deadline) {
                    info!("    {}: 2 jam habis (mid-chunk), checkpoint...", model_name);
                    break 'training;
                }
            }
        }

        trainer.completed_epochs = epoch + 1;
        let epoch_time = epoch_start.elapsed();
        let val_info = if !val_sequences.is_empty() {
            let m = trainer.evaluate_loss(val_sequences, seq_length);
            Some((m.avg_loss, m.perplexity))
        } else {
            None
        };

        epoch_metrics.push(HashMap::from([
            ("epoch".into(), serde_json::Value::from(epoch + 1)),
            ("steps".into(), serde_json::Value::from(trainer.step)),
            ("train_loss".into(), serde_json::Value::from(trainer.avg_loss())),
            ("val_loss".into(), val_info.map(|(l, _)| serde_json::Value::from(l)).unwrap_or(serde_json::Value::Null)),
            ("val_ppl".into(), val_info.map(|(_, p)| serde_json::Value::from(p)).unwrap_or(serde_json::Value::Null)),
            ("best_val_loss".into(), best_val_loss.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null)),
            ("elapsed_secs".into(), serde_json::Value::from(epoch_time.as_secs_f64())),
        ]));

        info!("    {} | Epoch {}/{} done | loss: {:.4} | time: {:?}",
            model_name, epoch + 1, epochs, trainer.avg_loss(), epoch_time);

        if trainer.step >= total_steps { break; }
    }

    let total_time = start_time.elapsed();
    let final_steps = trainer.step;
    let final_avg_loss = if final_steps > 0 { trainer.avg_loss() } else { 0.0 };

    trainer.sync_weights();
    if final_steps > 0 {
        trainer.save(&model_out.to_string_lossy())?;
        info!("    {} checkpoint: {}", model_name, model_out.display());

        // Save metadata sidecar for foundation type loading
        let meta = serde_json::json!({
            "model_id": format!("{:?}", model_id),
            "model_name": model_name,
            "framework": "nexora",
            "checkpoint_format": "safetensors",
            "transformer_config": {
                "hidden_size": trainer.model.config.hidden_size,
                "num_layers": trainer.model.config.num_layers,
                "num_heads": trainer.model.config.num_heads,
                "num_kv_heads": trainer.model.config.num_kv_heads,
                "intermediate_size": trainer.model.config.intermediate_size,
                "max_seq_len": trainer.model.config.max_seq_len,
                "vocab_size": trainer.model.config.vocab_size,
            },
            "training": {
                "steps": final_steps,
                "final_avg_loss": final_avg_loss,
                "best_val_loss": best_val_loss,
            },
            "load_hint": format!("foundation::Nxr{}Model::from_checkpoint(\"{}\")",
                format!("{:?}", model_id),
                model_out.file_name().map(|s| s.to_string_lossy()).unwrap_or_default()),
        });
        let meta_path = model_out.with_extension("safetensors.json");
        if let Ok(meta_json) = serde_json::to_string_pretty(&meta) {
            let _ = std::fs::write(&meta_path, &meta_json);
        }
    }

    let report = serde_json::json!({
        "model": model_name,
        "model_id": format!("{:?}", model_id),
        "parameters": param_count,
        "config": {
            "hidden_size": trainer.model.config.hidden_size,
            "num_layers": trainer.model.config.num_layers,
            "num_heads": trainer.model.config.num_heads,
            "num_kv_heads": trainer.model.config.num_kv_heads,
            "intermediate_size": trainer.model.config.intermediate_size,
            "max_seq_len": trainer.model.config.max_seq_len,
        },
        "training": {
            "epochs_completed": trainer.completed_epochs,
            "steps": final_steps,
            "final_avg_loss": final_avg_loss,
            "best_val_loss": best_val_loss,
            "total_tokens": trainer.total_tokens,
            "training_time_secs": total_time.as_secs_f64(),
        },
        "checkpoint": model_out.to_string_lossy().to_string(),
        "epoch_metrics": epoch_metrics,
    });

    Ok(report)
}
