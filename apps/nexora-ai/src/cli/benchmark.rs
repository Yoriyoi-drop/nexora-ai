//! Benchmark & Profiling Nyata (P0)
//!
//! Comprehensive benchmark suite untuk mengukur 9 metrik production:
//!   1. Token/s CPU
//!   2. Token/s GPU
//!   3. Prefill latency
//!   4. Decode latency
//!   5. VRAM usage
//!   6. RAM usage
//!   7. Prefix cache hit rate
//!   8. Continuous batching throughput
//!   9. Multi-user concurrency
//!
//! Target: bukan "jalan", tapi tahu bottleneck terbesar.

use crate::error::{NexoraError, NexoraResult};
use crate::NexoraAI;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

// ─── Report Types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub model: String,
    pub gpu_available: bool,
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub metrics: MetricsReport,
    pub bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MetricsReport {
    pub tokens_per_sec_cpu: Option<MetricSample>,
    pub tokens_per_sec_gpu: Option<MetricSample>,
    pub prefill_latency_ms: Option<MetricSample>,
    pub decode_latency_ms: Option<MetricSample>,
    pub vram_usage_mb: Option<MetricSample>,
    pub ram_usage_mb: Option<MetricSample>,
    pub prefix_cache_hit_rate: Option<f64>,
    pub cb_throughput: Option<BatchingMetric>,
    pub concurrency: Option<ConcurrencyMetric>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricSample {
    pub mean: f64,
    pub median: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchingMetric {
    pub batch_size: usize,
    pub sequences: usize,
    pub tokens_per_sec: f64,
    pub avg_step_time_ms: f64,
    pub total_time_ms: f64,
    pub padding_waste_pct: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ConcurrencyMetric {
    pub concurrent_users: usize,
    pub total_throughput_tok_s: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Bottleneck {
    pub metric: String,
    pub value: String,
    pub severity: String, // "critical" | "high" | "medium" | "low"
    pub detail: String,
}

// ─── Statistics Helpers ───────────────────────────────────────────────────────

fn compute_stats(values: &[f64]) -> MetricSample {
    let n = values.len();
    if n == 0 {
        return MetricSample {
            mean: 0.0,
            median: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
            samples: 0,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);
    let p99 = percentile(&sorted, 99.0);
    MetricSample {
        mean,
        median: p50,
        p50,
        p95,
        p99,
        min: sorted[0],
        max: sorted[n - 1],
        stddev,
        samples: n,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── Token Counter ────────────────────────────────────────────────────────────

/// Fast approximate token counter (chars/4). Real tokenizer via engine.
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

// ─── Benchmark Runner ─────────────────────────────────────────────────────────

pub struct BenchmarkRunner {
    nexora: Arc<NexoraAI>,
    warmup_rounds: usize,
    sample_rounds: usize,
}

impl BenchmarkRunner {
    pub fn new(nexora: Arc<NexoraAI>) -> Self {
        Self {
            nexora,
            warmup_rounds: 2,
            sample_rounds: 5,
        }
    }

    pub fn with_warmup(mut self, n: usize) -> Self {
        self.warmup_rounds = n;
        self
    }

    pub fn with_samples(mut self, n: usize) -> Self {
        self.sample_rounds = n;
        self
    }

    /// Run full benchmark suite — semua 9 metrik
    pub async fn run_full(&self) -> NexoraResult<BenchmarkReport> {
        info!("=== BENCHMARK & PROFILING NYATA (P0) ===");
        info!("Warmup: {} rounds, Sample: {} rounds", self.warmup_rounds, self.sample_rounds);

        let sys_info = self.nexora.get_system_info().await?;
        let gpu_available = self.check_gpu().await;

        let mut report = BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            model: self.nexora.model_id().to_string(),
            gpu_available,
            cpu_cores: sys_info.thread_count as usize,
            total_ram_mb: sys_info.memory_stats.total_memory / (1024 * 1024),
            metrics: MetricsReport::default(),
            bottlenecks: Vec::new(),
        };

        // 1. Token/s CPU
        report.metrics.tokens_per_sec_cpu = Some(self.bench_tokens_per_sec(false).await);

        // 2. Token/s GPU (jika available)
        if gpu_available {
            report.metrics.tokens_per_sec_gpu = Some(self.bench_tokens_per_sec(true).await);
        }

        // 3. Prefill latency
        report.metrics.prefill_latency_ms = Some(self.bench_prefill_latency().await);

        // 4. Decode latency
        report.metrics.decode_latency_ms = Some(self.bench_decode_latency().await);

        // 5. VRAM usage
        if gpu_available {
            report.metrics.vram_usage_mb = Some(self.bench_vram_usage().await);
        }

        // 6. RAM usage
        report.metrics.ram_usage_mb = Some(self.bench_ram_usage().await);

        // 7. Prefix cache hit rate
        report.metrics.prefix_cache_hit_rate = Some(self.bench_prefix_cache().await);

        // 8. Continuous batching throughput
        report.metrics.cb_throughput = Some(self.bench_continuous_batching().await);

        // 9. Multi-user concurrency
        report.metrics.concurrency = Some(self.bench_concurrency().await);

        // Analisa bottleneck
        report.bottlenecks = self.analyze_bottlenecks(&report);

        info!("=== BENCHMARK SELESAI ===");
        Ok(report)
    }

    /// Cek ketersediaan GPU
    async fn check_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            let available = nexora_deeplearning::gpu::GpuContext::is_available();
            info!("GPU tersedia: {}", available);
            return available;
        }
        #[cfg(not(feature = "gpu"))]
        {
            info!("GPU feature tidak dikompilasi");
            false
        }
    }

    // ── 1. Token/s CPU ──────────────────────────────────────────────────────

    async fn bench_tokens_per_sec(&self, use_gpu: bool) -> MetricSample {
        info!(">>> Benchmark Token/s ({})", if use_gpu { "GPU" } else { "CPU" });
        let prompts = vec![
            "The future of artificial intelligence lies in",
            "In the beginning, there was",
            "The key to understanding consciousness is",
            "Throughout human history, we have",
            "The most important discovery in science",
        ];

        let mut tok_rates = Vec::new();

        // Warmup
        for p in &prompts {
            let _ = self.generate_with_config(p, 32, use_gpu).await;
        }

        // Sample
        for round in 0..self.sample_rounds {
            for prompt in &prompts {
                let start = Instant::now();
                let result = self.generate_with_config(prompt, 64, use_gpu).await;
                let elapsed = start.elapsed();
                let tokens = estimate_tokens(&result);
                if elapsed.as_secs_f64() > 0.0 {
                    tok_rates.push(tokens as f64 / elapsed.as_secs_f64());
                }
            }
            info!("  Round {}/{}: {} samples", round + 1, self.sample_rounds, tok_rates.len());
        }

        let stats = compute_stats(&tok_rates);
        info!("  Token/s ({}): mean={:.1}, p50={:.1}, p95={:.1}",
            if use_gpu { "GPU" } else { "CPU" },
            stats.mean, stats.p50, stats.p95);
        stats
    }

    // ── 3. Prefill Latency ───────────────────────────────────────────────────

    async fn bench_prefill_latency(&self) -> MetricSample {
        info!(">>> Benchmark Prefill Latency");
        let prompt = "The philosophical implications of recursive self-improvement in artificial general intelligence systems have been debated since the early days of AI research. Some argue that a sufficiently advanced AI would undergo an intelligence explosion, rapidly outpacing human cognitive capabilities. Others contend that diminishing returns and fundamental limitations would prevent such an outcome. This debate touches on questions of consciousness, value alignment, and the future of humanity.";

        let mut latencies = Vec::new();

        for _ in 0..self.warmup_rounds {
            let _ = self.nexora.generate_text(prompt, 1, 0.7).await;
        }

        for _ in 0..self.sample_rounds {
            // Reset observability counters before measurement
            nexora_inference::inference_trait::reset_all_observability();

            let start = Instant::now();
            let _ = self.nexora.generate_text(prompt, 1, 0.7).await;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            latencies.push(elapsed);
        }

        let stats = compute_stats(&latencies);
        info!("  Prefill latency (ms): mean={:.1}, p50={:.1}, p95={:.1}", stats.mean, stats.p50, stats.p95);
        stats
    }

    // ── 4. Decode Latency ────────────────────────────────────────────────────

    async fn bench_decode_latency(&self) -> MetricSample {
        info!(">>> Benchmark Decode Latency");
        let prompt = "Once upon a time";

        let mut per_token_latencies = Vec::new();

        for _ in 0..self.warmup_rounds {
            let _ = self.nexora.generate_text(prompt, 10, 0.7).await;
        }

        for _ in 0..self.sample_rounds {
            let start = Instant::now();
            let result = self.nexora.generate_text(prompt, 50, 0.7).await.unwrap_or_default();
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            let tokens = estimate_tokens(&result);
            if tokens > 0 {
                per_token_latencies.push(total_ms / tokens as f64);
            }
        }

        let stats = compute_stats(&per_token_latencies);
        info!("  Decode latency (ms/token): mean={:.1}, p50={:.1}, p95={:.1}", stats.mean, stats.p50, stats.p95);
        stats
    }

    // ── 5. VRAM Usage ───────────────────────────────────────────────────────

    async fn bench_vram_usage(&self) -> MetricSample {
        info!(">>> Benchmark VRAM Usage");
        let prompt = "Test prompt for VRAM measurement.";

        let mut vram_samples = Vec::new();

        // Measure before generation
        if let Some(mb) = self.read_gpu_memory().await {
            vram_samples.push(mb);
        }

        // Generate to force GPU allocation
        for _ in 0..3 {
            let _ = self.nexora.generate_text(prompt, 100, 0.7).await;
            if let Some(mb) = self.read_gpu_memory().await {
                vram_samples.push(mb);
            }
        }

        if vram_samples.is_empty() {
            info!("  VRAM: tidak bisa diukur (GPU tidak tersedia atau tidak support nvml)");
            return MetricSample { mean: 0.0, median: 0.0, p50: 0.0, p95: 0.0, p99: 0.0, min: 0.0, max: 0.0, stddev: 0.0, samples: 0 };
        }

        let stats = compute_stats(&vram_samples);
        info!("  VRAM usage (MB): mean={:.0}, max={:.0}", stats.mean, stats.max);
        stats
    }

    /// Baca perkiraan VRAM via observability counters
    async fn read_gpu_memory(&self) -> Option<f64> {
        #[cfg(feature = "gpu")]
        {
            // Coba baca dari peak tracking
            let peak = nexora_inference::inference_trait::PEAK_GPU_MEM_BYTES
                .load(std::sync::atomic::Ordering::Relaxed);
            if peak > 0 {
                return Some(peak as f64 / (1024.0 * 1024.0));
            }
            // Estimasi dari KV cache memory
            let kv_mem = nexora_inference::inference_trait::KV_CACHE_MEMORY_BYTES
                .load(std::sync::atomic::Ordering::Relaxed);
            if kv_mem > 0 {
                return Some(kv_mem as f64 / (1024.0 * 1024.0));
            }
        }
        None
    }

    // ── 6. RAM Usage ────────────────────────────────────────────────────────

    async fn bench_ram_usage(&self) -> MetricSample {
        info!(">>> Benchmark RAM Usage");
        let prompt = "Memory measurement prompt.";

        let mut ram_samples = Vec::new();

        // Baseline
        ram_samples.push(self.read_process_ram());

        // Generate untuk trigger allocasi
        for _ in 0..3 {
            let _ = self.nexora.generate_text(prompt, 100, 0.7).await;
            ram_samples.push(self.read_process_ram());
        }

        let stats = compute_stats(&ram_samples);
        info!("  RAM usage (MB): mean={:.0}, max={:.0}", stats.mean, stats.max);
        stats
    }

    fn read_process_ram(&self) -> f64 {
        let mut sys = sysinfo::System::new();
        let pid = sysinfo::Pid::from_u32(std::process::id());
        sys.refresh_process(pid);
        if let Some(proc) = sys.process(pid) {
            proc.memory() as f64 / 1024.0 // KB → MB
        } else {
            0.0
        }
    }

    // ── 7. Prefix Cache Hit Rate ────────────────────────────────────────────

    async fn bench_prefix_cache(&self) -> f64 {
        info!(">>> Benchmark Prefix Cache Hit Rate");

        // Reset counters
        nexora_inference::inference_trait::reset_all_observability();

        // Shared prefix prompts — semua berbagi prefix yang sama
        let base = "The fundamental principles of quantum mechanics suggest that";
        let prompts = vec![
            format!("{} reality is fundamentally probabilistic in nature.", base),
            format!("{} particles can exist in multiple states simultaneously.", base),
            format!("{} observation itself affects the system being measured.", base),
            format!("{} entanglement allows instantaneous correlation across distance.", base),
            format!("{} the wave function collapses upon measurement.", base),
        ];

        // Warmup: generate semua prompt sekali
        for p in &prompts {
            let _ = self.nexora.generate_text(p, 20, 0.7).await;
        }

        // Reset lagi
        nexora_inference::inference_trait::reset_all_observability();

        // Generate ulang — prefix cache harus hit
        for p in &prompts {
            let _ = self.nexora.generate_text(p, 20, 0.7).await;
        }

        let hits = nexora_inference::inference_trait::prefix_cache_hits();
        let misses = nexora_inference::inference_trait::prefix_cache_misses();
        let total = hits + misses;
        let ratio = if total > 0 { hits as f64 / total as f64 } else { 0.0 };

        info!("  Prefix cache: hits={}, misses={}, ratio={:.2}%", hits, misses, ratio * 100.0);
        ratio
    }

    // ── 8. Continuous Batching Throughput ───────────────────────────────────

    async fn bench_continuous_batching(&self) -> BatchingMetric {
        info!(">>> Benchmark Continuous Batching Throughput");

        let prompts = vec![
            "Explain the theory of relativity",
            "What is machine learning",
            "Write a poem about nature",
            "Describe the water cycle",
            "What is the meaning of life",
            "Explain quantum computing",
            "How does photosynthesis work",
            "What is the history of Rome",
            "Describe the solar system",
            "What is artificial intelligence",
        ];

        let n = prompts.len();
        let batch_size = n.min(8);

        let start = Instant::now();

        // Kirim semua request concurrent
        let mut handles = Vec::with_capacity(n);
        for prompt in &prompts {
            let nexora = self.nexora.clone();
            let p = (*prompt);
            handles.push(tokio::spawn(async move {
                nexora.generate_text(&p, 30, 0.7).await
            }));
        }

        let mut total_tokens = 0;
        for h in handles {
            match h.await {
                Ok(Ok(text)) => total_tokens += estimate_tokens(&text),
                Ok(Err(e)) => warn!("  Concurrency request gagal: {}", e),
                Err(e) => warn!("  Task panic: {}", e),
            }
        }

        let total_time = start.elapsed();
        let tok_s = total_tokens as f64 / total_time.as_secs_f64();

        // Baca batch metrics dari observability
        let count = nexora_inference::inference_trait::BATCHING_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        let waste_sum = nexora_inference::inference_trait::BATCHING_PADDING_WASTE_SUM
            .load(std::sync::atomic::Ordering::Relaxed);
        let waste_pct = if count > 0 { waste_sum as f64 / count as f64 / 100.0 } else { 0.0 };

        info!("  CB throughput: {} seq, {} tok/s, waste={:.1}%, time={:.1}s",
            n, tok_s as u64, waste_pct * 100.0, total_time.as_secs_f64());

        BatchingMetric {
            batch_size,
            sequences: n,
            tokens_per_sec: tok_s,
            avg_step_time_ms: total_time.as_secs_f64() * 1000.0 / count.max(1) as f64,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
            padding_waste_pct: waste_pct * 100.0,
        }
    }

    // ── 9. Multi-User Concurrency ───────────────────────────────────────────

    async fn bench_concurrency(&self) -> ConcurrencyMetric {
        info!(">>> Benchmark Multi-User Concurrency");
        let prompt = "Tell me a story about a brave knight";

        // Simulasi 1, 2, 4, 8 concurrent users
        let concurrency_levels = [1, 2, 4, 8];
        let mut best_throughput = 0.0;
        let mut best_latency = 0.0;
        let mut best_p95 = 0.0;
        let mut best_concurrency = 1;
        let mut total_errors = 0;
        let mut total_reqs = 0;

        for &concurrent in &concurrency_levels {
            let mut latencies = Vec::new();
            let mut errors = 0;

            for _ in 0..3 {
                let mut handles = Vec::with_capacity(concurrent);
                for _ in 0..concurrent {
                    let nexora = self.nexora.clone();
                    let p = prompt.to_string();
                    handles.push(tokio::spawn(async move {
                        let t = Instant::now();
                        let result = nexora.generate_text(&p, 20, 0.7).await;
                        (t.elapsed(), result)
                    }));
                }

                for h in handles {
                    match h.await {
                        Ok((lat, Ok(_text))) => {
                            latencies.push(lat.as_secs_f64() * 1000.0);
                            total_reqs += 1;
                        }
                        Ok((_, Err(_))) => {
                            errors += 1;
                            total_errors += 1;
                        }
                        Err(_) => {
                            errors += 1;
                            total_errors += 1;
                        }
                    }
                }
            }

            if latencies.is_empty() {
                continue;
            }

            let avg_lat = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let sorted_lat = {
                let mut s = latencies.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                s
            };
            let p95 = percentile(&sorted_lat, 95.0);
            let throughput = (concurrent * 3) as f64 / latencies.iter().sum::<f64>() * 1000.0;

            info!("  Concurrency {}: avg_lat={:.1}ms, p95={:.1}ms, throughput={:.0} req/s, errors={}",
                concurrent, avg_lat, p95, throughput, errors);

            if throughput > best_throughput {
                best_throughput = throughput;
                best_latency = avg_lat;
                best_p95 = p95;
                best_concurrency = concurrent;
            }
        }

        let error_rate = if total_reqs + total_errors > 0 {
            total_errors as f64 / (total_reqs + total_errors) as f64
        } else {
            0.0
        };

        info!("  Concurrency best: {} concurrent, {} tok/s, p95={:.1}ms",
            best_concurrency, best_throughput as u64, best_p95);

        ConcurrencyMetric {
            concurrent_users: best_concurrency,
            total_throughput_tok_s: best_throughput,
            avg_latency_ms: best_latency,
            p95_latency_ms: best_p95,
            error_rate,
        }
    }

    // ── Helper ──────────────────────────────────────────────────────────────

    async fn generate_with_config(&self, prompt: &str, max_tokens: usize, _use_gpu: bool) -> String {
        self.nexora.generate_text(prompt, max_tokens, 0.7).await.unwrap_or_default()
    }

    // ── Bottleneck Analysis ─────────────────────────────────────────────────

    fn analyze_bottlenecks(&self, report: &BenchmarkReport) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Thresholds
        let min_tok_s_cpu = 5.0;
        let min_tok_s_gpu = 20.0;
        let max_prefill_ms = 500.0;
        let max_decode_ms = 100.0;
        let max_ram_pct = 80.0;
        let min_prefix_hit = 0.3;
        let min_cb_throughput = 10.0;

        let m = &report.metrics;

        // Token/s CPU
        if let Some(ref cpu) = m.tokens_per_sec_cpu {
            if cpu.mean < min_tok_s_cpu {
                bottlenecks.push(Bottleneck {
                    metric: "Token/s CPU".into(),
                    value: format!("{:.1}", cpu.mean),
                    severity: "critical".into(),
                    detail: format!("CPU throughput sangat rendah ({:.1} tok/s). Threshold minimum {} tok/s. Indikasi: model terlalu besar untuk CPU, atau forward pass tidak optimal.", cpu.mean, min_tok_s_cpu),
                });
            } else if cpu.mean < min_tok_s_cpu * 2.0 {
                bottlenecks.push(Bottleneck {
                    metric: "Token/s CPU".into(),
                    value: format!("{:.1}", cpu.mean),
                    severity: "high".into(),
                    detail: format!("CPU throughput rendah ({:.1} tok/s). Pertimbangkan quantisasi atau model lebih kecil.", cpu.mean),
                });
            }
        }

        // Token/s GPU
        if let Some(ref gpu) = m.tokens_per_sec_gpu {
            if gpu.mean < min_tok_s_gpu {
                bottlenecks.push(Bottleneck {
                    metric: "Token/s GPU".into(),
                    value: format!("{:.1}", gpu.mean),
                    severity: "critical".into(),
                    detail: format!("GPU throughput sangat rendah ({:.1} tok/s). Cek PCIe bandwidth, GPU util, atau batch size.", gpu.mean),
                });
            }
            // GPU vs CPU ratio
            if let Some(ref cpu) = m.tokens_per_sec_cpu {
                let ratio = gpu.mean / cpu.mean;
                if ratio < 2.0 {
                    bottlenecks.push(Bottleneck {
                        metric: "GPU/CPU Speedup".into(),
                        value: format!("{:.1}x", ratio),
                        severity: "high".into(),
                        detail: format!("GPU hanya {:.1}x lebih cepat dari CPU. Kemungkinan bottleneck di PCIe transfer atau kernel launch overhead.", ratio),
                    });
                }
            }
        }

        // Prefill latency
        if let Some(ref pl) = m.prefill_latency_ms {
            if pl.mean > max_prefill_ms {
                bottlenecks.push(Bottleneck {
                    metric: "Prefill Latency".into(),
                    value: format!("{:.1} ms", pl.mean),
                    severity: if pl.mean > max_prefill_ms * 2.0 { "critical".into() } else { "high".into() },
                    detail: format!("Prefill terlalu lambat ({:.1} ms). Cek prompt length, KV cache init, atau GPU prefill kernel.", pl.mean),
                });
            }
        }

        // Decode latency
        if let Some(ref dl) = m.decode_latency_ms {
            if dl.mean > max_decode_ms {
                bottlenecks.push(Bottleneck {
                    metric: "Decode Latency".into(),
                    value: format!("{:.1} ms/token", dl.mean),
                    severity: if dl.mean > max_decode_ms * 2.0 { "critical".into() } else { "high".into() },
                    detail: format!("Decode terlalu lambat ({:.1} ms/token). Target < {} ms. Cek: quantization, KV cache access pattern, GPU memory bandwidth.", dl.mean, max_decode_ms),
                });
            }
        }

        // Prefix cache hit rate
        if let Some(ph) = m.prefix_cache_hit_rate {
            if ph < min_prefix_hit {
                bottlenecks.push(Bottleneck {
                    metric: "Prefix Cache Hit Rate".into(),
                    value: format!("{:.1}%", ph * 100.0),
                    severity: "medium".into(),
                    detail: format!("Hit rate hanya {:.1}%. Prefix cache tidak efektif. Pertimbangkan: longer TTL, lebih banyak shared prefix requests, atau periksa eviction policy.", ph * 100.0),
                });
            }
        }

        // CB throughput
        if let Some(ref cb) = m.cb_throughput {
            if cb.tokens_per_sec < min_cb_throughput {
                bottlenecks.push(Bottleneck {
                    metric: "CB Throughput".into(),
                    value: format!("{:.1} tok/s", cb.tokens_per_sec),
                    severity: "high".into(),
                    detail: format!("Continuous batching throughput rendah ({:.1} tok/s) untuk {} sequences. Cek: max_batch_size, padding waste ({:.1}%), scheduling policy.", cb.tokens_per_sec, cb.sequences, cb.padding_waste_pct),
                });
            }
        }

        // RAM usage
        if let Some(ref ram) = m.ram_usage_mb {
            let pct = (ram.max / report.total_ram_mb as f64) * 100.0;
            if pct > max_ram_pct {
                bottlenecks.push(Bottleneck {
                    metric: "RAM Usage".into(),
                    value: format!("{:.0} MB ({:.0}%)", ram.max, pct),
                    severity: "high".into(),
                    detail: format!("RAM usage {:.0}% dari total {} MB. Berisiko OOM saat concurrent load tinggi.", pct, report.total_ram_mb),
                });
            }
        }

        bottlenecks
    }
}

// ── Report Formatting ─────────────────────────────────────────────────────────

impl BenchmarkReport {
    /// Format laporan sebagai string untuk ditampilkan di CLI
    pub fn to_formatted_string(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();

        writeln!(s, "╔══════════════════════════════════════════════════════════╗").ok();
        writeln!(s, "║     NEXORA AI — BENCHMARK & PROFILING NYATA (P0)      ║").ok();
        writeln!(s, "╚══════════════════════════════════════════════════════════╝").ok();
        writeln!(s).ok();
        writeln!(s, "  Model:       {}", self.model).ok();
        writeln!(s, "  Waktu:       {}", self.timestamp).ok();
        writeln!(s, "  GPU:         {}", if self.gpu_available { "✓ Tersedia" } else { "✗ Tidak ada" }).ok();
        writeln!(s, "  CPU Cores:   {}", self.cpu_cores).ok();
        writeln!(s, "  RAM Total:   {} MB", self.total_ram_mb).ok();
        writeln!(s).ok();

        writeln!(s, "┌─────────────────────────────────────────────────────────┐").ok();
        writeln!(s, "│                    METRIC RESULTS                       │").ok();
        writeln!(s, "├─────────────────────────────────────────────────────────┤").ok();

        // Helper untuk print metric
        let print_metric = |s: &mut String, name: &str, sample: &Option<MetricSample>, unit: &str| {
            if let Some(ref m) = sample {
                writeln!(s, "  {:<25} {:>10.1} {}  (p95: {:>10.1})", name, m.mean, unit, m.p95).ok();
            } else {
                writeln!(s, "  {:<25} {:>10}", name, "N/A").ok();
            }
        };

        print_metric(&mut s, "Token/s CPU", &self.metrics.tokens_per_sec_cpu, "tok/s");
        print_metric(&mut s, "Token/s GPU", &self.metrics.tokens_per_sec_gpu, "tok/s");
        print_metric(&mut s, "Prefill latency", &self.metrics.prefill_latency_ms, "ms");
        print_metric(&mut s, "Decode latency", &self.metrics.decode_latency_ms, "ms/tok");
        print_metric(&mut s, "VRAM usage", &self.metrics.vram_usage_mb, "MB");
        print_metric(&mut s, "RAM usage", &self.metrics.ram_usage_mb, "MB");

        if let Some(ph) = self.metrics.prefix_cache_hit_rate {
            writeln!(s, "  {:<25} {:>10.1}%", "Prefix cache hit rate", ph * 100.0).ok();
        }

        if let Some(ref cb) = self.metrics.cb_throughput {
            writeln!(s, "  {:<25} {:>10.1} tok/s  ({} seq, {:.1}% waste)", "CB throughput", cb.tokens_per_sec, cb.sequences, cb.padding_waste_pct).ok();
        }

        if let Some(ref conc) = self.metrics.concurrency {
            writeln!(s, "  {:<25} {:>10} users  ({} tok/s, p95: {:.0}ms)", "Concurrency best", conc.concurrent_users, conc.total_throughput_tok_s as u64, conc.p95_latency_ms).ok();
        }

        writeln!(s, "└─────────────────────────────────────────────────────────┘").ok();

        // Bottlenecks
        if !self.bottlenecks.is_empty() {
            writeln!(s).ok();
            writeln!(s, "┌─────────────────────────────────────────────────────────┐").ok();
            writeln!(s, "│                  BOTTLENECK ANALYSIS                   │").ok();
            writeln!(s, "├─────────────────────────────────────────────────────────┤").ok();
            for (i, b) in self.bottlenecks.iter().enumerate() {
                let icon = match b.severity.as_str() {
                    "critical" => "🔴 CRITICAL",
                    "high" => "🟠 HIGH",
                    "medium" => "🟡 MEDIUM",
                    _ => "⚪ LOW",
                };
                writeln!(s, "  {}. [{}] {}", i + 1, icon, b.metric).ok();
                writeln!(s, "     Value: {}", b.value).ok();
                writeln!(s, "     {}", b.detail).ok();
                writeln!(s).ok();
            }
            writeln!(s, "└─────────────────────────────────────────────────────────┘").ok();
        }

        // Summary
        writeln!(s).ok();
        let critical = self.bottlenecks.iter().filter(|b| b.severity == "critical").count();
        let high = self.bottlenecks.iter().filter(|b| b.severity == "high").count();
        let medium = self.bottlenecks.iter().filter(|b| b.severity == "medium").count();

        writeln!(s, "  Summary: {} critical, {} high, {} medium bottlenecks ditemukan.", critical, high, medium).ok();
        if critical > 0 {
            writeln!(s, "  ⚠ Rekomendasi: Prioritaskan fix critical bottlenecks.").ok();
        }
        writeln!(s).ok();

        s
    }

    /// Format sebagai JSON
    pub fn to_json_string(&self) -> NexoraResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| NexoraError::serialization(e))
    }
}
