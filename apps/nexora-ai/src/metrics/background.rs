use std::time::Duration;
use tracing::debug;

use crate::server::handlers::metrics_collector;

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

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut rx => {
                        tracing::info!("Background metrics collector stopped");
                        break;
                    }
                    _ = tokio::time::sleep(interval) => {
                        Self::collect_once().await;
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

    async fn collect_once() {
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
