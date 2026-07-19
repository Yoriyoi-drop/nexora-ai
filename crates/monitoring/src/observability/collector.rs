use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::broadcast;

use super::metrics::ObservabilityMetrics;

pub struct MetricCounters {
    pub tokens_generated: AtomicU64,
    pub total_latency_ms: AtomicU64,
    pub request_count: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_evictions: AtomicU64,
    pub tool_calls_total: AtomicU64,
    pub tool_calls_failed: AtomicU64,
    pub tool_call_duration_ms: AtomicU64,
    pub total_cost_micro: AtomicU64,
    pub total_saved_micro: AtomicU64,
    pub hallucination_count: AtomicU64,
    pub agent_restarts: AtomicU64,
    pub retry_count: AtomicU64,
    pub failure_count: AtomicU64,
}

impl MetricCounters {
    pub fn new() -> Self {
        Self {
            tokens_generated: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_evictions: AtomicU64::new(0),
            tool_calls_total: AtomicU64::new(0),
            tool_calls_failed: AtomicU64::new(0),
            tool_call_duration_ms: AtomicU64::new(0),
            total_cost_micro: AtomicU64::new(0),
            total_saved_micro: AtomicU64::new(0),
            hallucination_count: AtomicU64::new(0),
            agent_restarts: AtomicU64::new(0),
            retry_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
        }
    }
}

impl Default for MetricCounters {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ObservabilityCollector {
    counters: Arc<MetricCounters>,
    latencies: RwLock<Vec<u64>>,
    max_latency_samples: usize,
    start_time: Instant,
    running: Arc<AtomicBool>,
    metrics_tx: broadcast::Sender<ObservabilityMetrics>,
    system: RwLock<System>,
}

impl ObservabilityCollector {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(64);
        let sys = System::new();
        Self {
            counters: Arc::new(MetricCounters::new()),
            latencies: RwLock::new(Vec::with_capacity(10000)),
            max_latency_samples: 10000,
            start_time: Instant::now(),
            running: Arc::new(AtomicBool::new(false)),
            metrics_tx: tx,
            system: RwLock::new(sys),
        }
    }

    pub fn counters(&self) -> &Arc<MetricCounters> {
        &self.counters
    }

    pub fn record_token_generated(&self) {
        self.counters.tokens_generated.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tokens(&self, count: u64) {
        self.counters.tokens_generated.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_latency(&self, latency_ms: u64) {
        self.counters.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
        self.counters.request_count.fetch_add(1, Ordering::Relaxed);
        let mut latencies = self.latencies.write();
        if latencies.len() >= self.max_latency_samples {
            latencies.remove(0);
        }
        latencies.push(latency_ms);
    }

    pub fn record_cache_hit(&self) {
        self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_eviction(&self) {
        self.counters.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tool_call(&self, success: bool, duration_ms: u64) {
        self.counters.tool_calls_total.fetch_add(1, Ordering::Relaxed);
        self.counters.tool_call_duration_ms.fetch_add(duration_ms, Ordering::Relaxed);
        if !success {
            self.counters.tool_calls_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_cost(&self, cost_usd: f64, saved_usd: f64) {
        self.counters.total_cost_micro.fetch_add((cost_usd * 1_000_000.0) as u64, Ordering::Relaxed);
        self.counters.total_saved_micro.fetch_add((saved_usd * 1_000_000.0) as u64, Ordering::Relaxed);
    }

    pub fn record_hallucination(&self) {
        self.counters.hallucination_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.counters.failure_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_retry(&self) {
        self.counters.retry_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_agent_restart(&self) {
        self.counters.agent_restarts.fetch_add(1, Ordering::Relaxed);
    }

    fn refresh_system_info(&self) -> (f64, f64, f64) {
        let mut sys = self.system.write();
        sys.refresh_cpu();
        sys.refresh_memory();
        let cpu = sys.global_cpu_info().cpu_usage() as f64;
        let used = sys.used_memory() as f64 / (1024.0 * 1024.0);
        let total = sys.total_memory() as f64 / (1024.0 * 1024.0);
        (cpu, used, total)
    }

    pub fn snapshot(&self) -> ObservabilityMetrics {
        let counters = &self.counters;
        let request_count = counters.request_count.load(Ordering::Relaxed);
        let hits = counters.cache_hits.load(Ordering::Relaxed);
        let misses = counters.cache_misses.load(Ordering::Relaxed);
        let total_cache = hits + misses;
        let failures = counters.failure_count.load(Ordering::Relaxed);
        let total_ops = request_count + failures;
        let elapsed_secs = self.start_time.elapsed().as_secs_f64().max(1.0);
        let tool_calls_total = counters.tool_calls_total.load(Ordering::Relaxed);

        let (avg, p95, p99) = self.compute_latency_percentiles();
        let (cpu, ram_used, ram_total) = self.refresh_system_info();

        ObservabilityMetrics {
            cpu_util_pct: cpu,
            gpu_util_pct: 0.0,
            gpu_memory_used_mb: 0.0,
            ram_used_mb: ram_used,
            ram_total_mb: ram_total,
            tokens_generated: counters.tokens_generated.load(Ordering::Relaxed),
            tokens_per_sec: counters.tokens_generated.load(Ordering::Relaxed) as f64 / elapsed_secs,
            avg_latency_ms: avg,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            queue_depth: 0,
            queue_wait_ms: 0.0,
            requests_in_flight: 0,
            cache_hit_rate: if total_cache > 0 { hits as f64 / total_cache as f64 } else { 0.0 },
            cache_hits: hits,
            cache_misses: misses,
            cache_evictions: counters.cache_evictions.load(Ordering::Relaxed),
            tool_calls_total,
            tool_calls_failed: counters.tool_calls_failed.load(Ordering::Relaxed),
            tool_call_avg_ms: if tool_calls_total > 0 {
                counters.tool_call_duration_ms.load(Ordering::Relaxed) as f64 / tool_calls_total as f64
            } else {
                0.0
            },
            total_cost_usd: counters.total_cost_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            estimated_savings_usd: counters.total_saved_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            cost_per_request: if request_count > 0 {
                counters.total_cost_micro.load(Ordering::Relaxed) as f64 / request_count as f64 / 1_000_000.0
            } else {
                0.0
            },
            hallucination_rate: if request_count > 0 {
                counters.hallucination_count.load(Ordering::Relaxed) as f64 / request_count as f64
            } else {
                0.0
            },
            hallucination_count: counters.hallucination_count.load(Ordering::Relaxed),
            active_agents: 0,
            idle_agents: 0,
            failed_agents: 0,
            agent_restarts: counters.agent_restarts.load(Ordering::Relaxed),
            retry_count: counters.retry_count.load(Ordering::Relaxed),
            failure_count: failures,
            success_rate: if total_ops > 0 {
                (request_count as f64 / total_ops as f64) * 100.0
            } else {
                100.0
            },
        }
    }

    fn compute_latency_percentiles(&self) -> (f64, f64, f64) {
        let latencies = self.latencies.read();
        let len = latencies.len();
        if len == 0 {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted = latencies.clone();
        sorted.sort_unstable();
        let avg = sorted.iter().sum::<u64>() as f64 / len as f64;
        let p95_idx = ((len as f64) * 0.95).ceil() as usize - 1;
        let p99_idx = ((len as f64) * 0.99).ceil() as usize - 1;
        let p95 = sorted.get(p95_idx.min(len - 1)).copied().unwrap_or(0) as f64;
        let p99 = sorted.get(p99_idx.min(len - 1)).copied().unwrap_or(0) as f64;
        (avg, p95, p99)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ObservabilityMetrics> {
        self.metrics_tx.subscribe()
    }

    pub fn start_background_collection(&self, interval_secs: u64) {
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let tx = self.metrics_tx.clone();
        let collector = self.counters.clone();
        let start = self.start_time;

        tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                if !running.load(Ordering::Relaxed) {
                    break;
                }

                let elapsed_secs = start.elapsed().as_secs_f64().max(1.0);
                let req_count = collector.request_count.load(Ordering::Relaxed);
                let failures = collector.failure_count.load(Ordering::Relaxed);
                let total_ops = req_count + failures;
                let hits = collector.cache_hits.load(Ordering::Relaxed);
                let misses = collector.cache_misses.load(Ordering::Relaxed);
                let total_cache = hits + misses;
                let tool_calls_total = collector.tool_calls_total.load(Ordering::Relaxed);

                let metrics = ObservabilityMetrics {
                    cpu_util_pct: 0.0,
                    gpu_util_pct: 0.0,
                    gpu_memory_used_mb: 0.0,
                    ram_used_mb: 0.0,
                    ram_total_mb: 0.0,
                    tokens_generated: collector.tokens_generated.load(Ordering::Relaxed),
                    tokens_per_sec: collector.tokens_generated.load(Ordering::Relaxed) as f64 / elapsed_secs,
                    avg_latency_ms: 0.0,
                    p95_latency_ms: 0.0,
                    p99_latency_ms: 0.0,
                    queue_depth: 0,
                    queue_wait_ms: 0.0,
                    requests_in_flight: 0,
                    cache_hit_rate: if total_cache > 0 { hits as f64 / total_cache as f64 } else { 0.0 },
                    cache_hits: hits,
                    cache_misses: misses,
                    cache_evictions: collector.cache_evictions.load(Ordering::Relaxed),
                    tool_calls_total,
                    tool_calls_failed: collector.tool_calls_failed.load(Ordering::Relaxed),
                    tool_call_avg_ms: if tool_calls_total > 0 {
                        collector.tool_call_duration_ms.load(Ordering::Relaxed) as f64 / tool_calls_total as f64
                    } else {
                        0.0
                    },
                    total_cost_usd: collector.total_cost_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                    estimated_savings_usd: collector.total_saved_micro.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                    cost_per_request: if req_count > 0 {
                        collector.total_cost_micro.load(Ordering::Relaxed) as f64 / req_count as f64 / 1_000_000.0
                    } else {
                        0.0
                    },
                    hallucination_rate: if req_count > 0 {
                        collector.hallucination_count.load(Ordering::Relaxed) as f64 / req_count as f64
                    } else {
                        0.0
                    },
                    hallucination_count: collector.hallucination_count.load(Ordering::Relaxed),
                    active_agents: 0,
                    idle_agents: 0,
                    failed_agents: 0,
                    agent_restarts: collector.agent_restarts.load(Ordering::Relaxed),
                    retry_count: collector.retry_count.load(Ordering::Relaxed),
                    failure_count: failures,
                    success_rate: if total_ops > 0 {
                        (req_count as f64 / total_ops as f64) * 100.0
                    } else {
                        100.0
                    },
                };
                let _ = tx.send(metrics);
            }
        });
    }

    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

impl Default for ObservabilityCollector {
    fn default() -> Self {
        Self::new()
    }
}
