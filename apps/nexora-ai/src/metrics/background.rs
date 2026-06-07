use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::debug;

use crate::server::handlers::metrics_collector;

struct TrainingSnapshot {
    loss: f64,
    learning_rate: f64,
    grad_norm: f64,
}

static TRAINING: once_cell::sync::Lazy<Mutex<TrainingSnapshot>> =
    once_cell::sync::Lazy::new(|| {
        Mutex::new(TrainingSnapshot {
            loss: 0.0,
            learning_rate: 0.0,
            grad_norm: 0.0,
        })
    });

/// Update training metrics from CLI or training subsystem.
/// Thread-safe: can be called from any tokio task.
pub fn update_training_metrics(loss: f64, learning_rate: f64, grad_norm: f64) {
    if let Ok(mut t) = TRAINING.lock() {
        t.loss = loss;
        t.learning_rate = learning_rate;
        t.grad_norm = grad_norm;
    }
}

/// Background metrics collector that periodically syncs GPU, system, and cache metrics
/// into the Prometheus monitoring registry.
///
/// This bridges:
/// - `observability_snapshot()` GPU atomic counters + prefix cache stats → Prometheus counters/gauges
/// - NVML GPU memory readings → Prometheus gauges
pub struct BackgroundMetricsCollector {
    interval: Duration,
    stop_signal: Option<tokio::sync::oneshot::Sender<()>>,
}

impl BackgroundMetricsCollector {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            stop_signal: None,
        }
    }

    /// Start the background polling task.
    pub fn start(&mut self) {
        let interval = self.interval;
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        self.stop_signal = Some(tx);

        // Track token throughput across collection intervals
        let last_tokens = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let last_time = std::sync::Arc::new(std::sync::Mutex::new(Instant::now()));
        let last_tokens_clone = last_tokens.clone();
        let last_time_clone = last_time.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut rx => {
                        tracing::info!("Background metrics collector stopped");
                        break;
                    }
                    _ = tokio::time::sleep(interval) => {
                        Self::collect_once(
                            &last_tokens_clone,
                            &last_time_clone,
                        ).await;
                    }
                }
            }
        });
    }

    /// Stop the background polling task.
    pub fn stop(&mut self) {
        if let Some(tx) = self.stop_signal.take() {
            let _ = tx.send(());
        }
    }

    async fn collect_once(
        last_tokens: &std::sync::Arc<std::sync::atomic::AtomicU64>,
        last_time: &std::sync::Arc<std::sync::Mutex<Instant>>,
    ) {
        let collector = match metrics_collector() {
            Some(c) => c,
            None => return,
        };

        let obs = nexora_inference::inference_trait::observability_snapshot();
        collector.set_gpu_forward_errors(obs.gpu_forward_errors);
        collector.set_gpu_cpu_fallbacks(obs.gpu_cpu_fallbacks);
        collector.set_gpu_resident_successes(obs.gpu_resident_successes);
        collector.set_gpu_resident_fallbacks(obs.gpu_resident_fallbacks);
        collector.set_gpu_tokens(obs.gpu_tokens_generated);
        collector.set_cpu_tokens(obs.cpu_tokens_generated);
        collector.set_gpu_alive(obs.gpu_alive);

        collector.set_cache_hit_ratio(obs.prefix_cache_hit_ratio);

        let total_tokens = obs.gpu_tokens_generated + obs.cpu_tokens_generated;
        if total_tokens > 0 {
            collector.set_throughput_tokens(total_tokens);
        }

        // Token/sec throughput (moving window)
        let now = Instant::now();
        let prev_tokens = last_tokens.swap(total_tokens, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut prev_time) = last_time.lock() {
            let elapsed = now.duration_since(*prev_time).as_secs_f64();
            if elapsed > 0.0 && total_tokens >= prev_tokens {
                let tps = (total_tokens - prev_tokens) as f64 / elapsed;
                collector.set_tokens_per_sec(tps);
            }
            *prev_time = now;
        }

        // Phase 6b metrics
        collector.set_kv_cache_pressure(obs.kv_cache_pressure);
        collector.set_scheduler_queue_depth(obs.scheduler_queue_depth);
        collector.set_batching_efficiency_pct(obs.batching_efficiency_pct);
        collector.set_gpu_utilization_pct(obs.gpu_utilization_pct);
        collector.set_pcie_read_bytes(obs.pcie_read_bytes);
        collector.set_pcie_write_bytes(obs.pcie_write_bytes);
        collector.set_kv_internal_frag_ratio(obs.kv_internal_frag_ratio);
        collector.set_kv_external_frag_ratio(obs.kv_external_frag_ratio);

        // Memory fragmentation estimate from pool alloc/dealloc ratio
        let allocs = obs.pool_allocs;
        let deallocs = obs.pool_deallocs;
        let fragmentation = if allocs + deallocs > 0 {
            (deallocs as f64 / (allocs + deallocs) as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        collector.set_memory_fragmentation(fragmentation);

        // Training metrics
        if let Ok(t) = TRAINING.lock() {
            collector.set_training_loss(t.loss);
            collector.set_training_learning_rate(t.learning_rate);
            collector.set_training_grad_norm(t.grad_norm);
        }

        let math_fallbacks = nexora_deeplearning::autograd::ops::math::gpu_math_fallback_count();
        collector.set_gpu_math_fallbacks(math_fallbacks);

        match nexora_inference::read_gpu_memory().await {
            Ok((used_bytes, used_pct)) => {
                collector.set_gpu_memory_bytes(used_bytes);
                collector.set_gpu_memory_percent(used_pct);
            }
            Err(e) => debug!("GPU memory read failed: {}", e),
        }
    }
}
