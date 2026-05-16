use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::RwLock;
use tracing::{info, warn};
use tokio::signal;
use chrono::Utc;

use crate::NexoraAI;
use nexora_foundation::training::{Trainer, TrainerConfig};
use nexora_foundation::models::transformer::{CausalLM, TrainableCausalLM, TransformerConfig};
use nexora_deeplearning::TensorOps;
use nexora_datastream::{
    DataSample, SourceInfo, SourceCategory,
    filter::{LengthFilter, QualityFilter, DedupFilter},
};
use nexora_tokenizer::BpeTokenizer;

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
    ) -> Result<()> {
        info!("=== NEXORA TRAINING ===");
        info!("Data: {:?}", data);
        info!("Output: {:?}", output);
        info!("Epochs: {}, Batch: {}, LR: {}, GPU: {}", epochs, batch_size, learning_rate, gpu);

        if !data.exists() {
            return Err(anyhow::anyhow!("Training data not found: {:?}", data));
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!("[1/6] Membaca data training...");
        let raw_text = std::fs::read_to_string(data)?;
        let lines: Vec<&str> = raw_text.lines().filter(|l| !l.trim().is_empty()).collect();
        info!("  {} baris teks dibaca", lines.len());

        info!("[2/6] Mempersiapkan tokenizer...");
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

        info!("[3/6] Inisialisasi DataStream pipeline (filter)...");
        let source = SourceInfo {
            name: "cli_training".into(),
            url: None,
            trust_score: 0.8,
            category: SourceCategory::Other,
            fetch_timestamp: Utc::now().timestamp(),
        };

        let mut graph = nexora_datastream::ExecutionGraph::new();
        graph.add_node("length", Arc::new(LengthFilter::default()), vec![], true, 1);
        graph.add_node("quality", Arc::new(QualityFilter::default()), vec!["length".into()], true, 2);
        graph.add_node("dedup", Arc::new(DedupFilter::new()), vec!["quality".into()], false, 3);
        graph.finalize();

        let intake = nexora_datastream::StreamIntakeEngine::default();
        let texts_with_source: Vec<(String, SourceInfo)> = lines.iter()
            .map(|l| (l.to_string(), source.clone()))
            .collect();
        let mut sample_rx = intake.ingest_batch(texts_with_source).await;

        let mut samples: Vec<DataSample> = Vec::new();
        while let Some(s) = sample_rx.recv().await {
            let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
            let result = graph.execute(s, cancel_rx).await;
            drop(cancel_tx);
            if result.is_accepted() {
                if let Some(sample) = result.sample() {
                    samples.push(sample.clone());
                }
            }
        }
        info!("  {} sampel lolos filter dari {}", samples.len(), lines.len());

        if samples.is_empty() {
            return Err(anyhow::anyhow!("Tidak ada data lolos filter"));
        }

        info!("[4/6] Inisialisasi Trainer + CausalLM...");
        let seq_length = 128;
        let max_steps = epochs.max(1) * samples.len().max(1) / batch_size.max(1);

        let trainer_config = TrainerConfig {
            learning_rate,
            max_steps: max_steps.max(10),
            seq_length,
            vocab_size,
            save_path: Some(output.to_string_lossy().to_string()),
            save_every: (max_steps / 10).max(1),
            report_every: 1,
        };

        let model_config = TransformerConfig {
            vocab_size,
            max_seq_len: seq_length,
            ..Default::default()
        };
        let model = CausalLM::new(model_config);
        let param_count = model.parameter_count();
        info!("  Model: {}M parameters", param_count / 1_000_000);

        let mut trainer = Trainer::with_model(model, trainer_config);
        trainer.prepare();
        let stop_flag = trainer.stop_signal();
        info!("  Trainer siap: {} steps, {} seq_length", trainer.config.max_steps, trainer.config.seq_length);

        let stop_flag_c = stop_flag.clone();
        tokio::spawn(async move {
            signal::ctrl_c().await.ok();
            info!("  Sinyal stop diterima! Menyelesaikan batch terakhir...");
            stop_flag_c.store(true, Ordering::SeqCst);
        });

        info!("[5/6] Training loop dimulai (Ctrl+C untuk stop)...");
        let start_time = std::time::Instant::now();
        let mut step = 0;
        let total_steps = trainer.config.max_steps;

        'training: for sample in samples.iter().cycle() {
            if stop_flag.load(Ordering::SeqCst) {
                info!("  Training dihentikan oleh pengguna");
                break;
            }

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
                        step += 1;
                        if step % trainer.config.report_every == 0 || step == 1 {
                            let elapsed = start_time.elapsed();
                            info!("  Step {}/{} | loss: {:.4} | avg: {:.4} | elapsed: {:?}",
                                step, total_steps, loss, trainer.avg_loss(), elapsed);
                        }
                    }
                    None => {
                        info!("  Training dihentikan (stop flag)");
                        break 'training;
                    }
                }

                if stop_flag.load(Ordering::SeqCst) {
                    break 'training;
                }

                if step >= total_steps {
                    break 'training;
                }
            }
        }

        let total_time = start_time.elapsed();
        let final_steps = step;
        let final_avg_loss = if final_steps > 0 { trainer.avg_loss() } else { 0.0 };

        info!("[6/6] Menyimpan final checkpoint + tokenizer...");
        trainer.sync_weights();
        if final_steps > 0 {
            let safetensors_path = output.with_extension("safetensors");
            let save_path_str = safetensors_path.to_string_lossy().to_string();
            trainer.save(&save_path_str)?;
            info!("  Checkpoint: {}", safetensors_path.display());
        } else {
            warn!("  Tidak ada checkpoint disimpan (0 steps)");
        }

        if let Some(tok_path) = tokenizer_path {
            let tok = tokenizer.read().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
            tok.save(tok_path)
                .map_err(|e| anyhow::anyhow!("Gagal save final tokenizer: {}", e))?;
            info!("  Tokenizer: {:?}", tok_path);
        }

        let report = serde_json::json!({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gpu": gpu,
            "steps": final_steps,
            "samples_filtered": samples.len(),
            "lines_loaded": lines.len(),
            "final_avg_loss": final_avg_loss,
            "model_params": param_count,
            "vocab_size": vocab_size,
            "stopped_early": stop_flag.load(Ordering::SeqCst),
            "training_time_secs": total_time.as_secs_f64(),
            "timestamp": Utc::now().to_rfc3339(),
            "data_stream_filtered": true,
        });

        let report_path = output.with_extension("json");
        std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

        info!("=== TRAINING SELESAI ===");
        info!("  Steps: {}", final_steps);
        info!("  Final avg loss: {:.4}", final_avg_loss);
        info!("  Waktu: {:.2}s", total_time.as_secs_f64());
        if stop_flag.load(Ordering::SeqCst) {
            info!("  Dihentikan lebih awal oleh pengguna");
        }
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
