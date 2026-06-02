//! Benchmark types and harness for Nexora.
//!
//! Provides `MetricSample`, `BenchmarkReport`, `BaselineReport`,
//! and related types used by the CLI benchmark runner and criterion benches.
//! Pure data types — no runtime dependencies on model crates.

/// Statistical summary of a measured metric.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

impl MetricSample {
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        Self {
            mean,
            median,
            p50: percentile(&sorted, 50.0),
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
            min: sorted[0],
            max: sorted[n - 1],
            stddev: variance.sqrt(),
            samples: n,
        }
    }
}

impl Default for MetricSample {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
            samples: 0,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let k = (p / 100.0) * (n - 1) as f64;
    let f = k.floor() as usize;
    let c = k.ceil() as usize;
    if f == c {
        sorted[f]
    } else {
        sorted[f] + (k - f as f64) * (sorted[c] - sorted[f])
    }
}

/// Continuous batching throughput metric.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchingMetric {
    pub batch_size: usize,
    pub sequences: usize,
    pub tokens_per_sec: f64,
    pub avg_step_time_ms: f64,
    pub total_time_ms: f64,
    pub padding_waste_pct: f64,
}

/// Multi-user concurrency metric.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConcurrencyMetric {
    pub num_clients: usize,
    pub total_requests: usize,
    pub total_time_ms: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub throughput_req_per_sec: f64,
    pub errors: usize,
}

/// Identified bottleneck with severity.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Bottleneck {
    pub component: String,
    pub severity: BottleneckSeverity,
    pub detail: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum BottleneckSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Comprehensive benchmark report for a single run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub model: String,
    pub gpu_available: bool,
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub metrics: MetricsReport,
    pub bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone, serde::Serialize)]
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

/// Baseline report for tracking performance over time.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BaselineReport {
    pub timestamp: String,
    pub system: SystemBaseline,
    pub inference: Option<MetricSample>,
    pub training: Option<TrainingBaseline>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemBaseline {
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub ram_usage_mb: Option<u64>,
    pub vram_usage_mb: Option<u64>,
    pub os_info: String,
    pub rust_version: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingBaseline {
    pub token_per_sec: f64,
    pub final_perplexity: f64,
    pub best_loss: f64,
    pub total_steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_sample_from_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = MetricSample::from_samples(&samples);
        assert!((m.mean - 3.0).abs() < 0.01);
        assert!((m.median - 3.0).abs() < 0.01);
        assert!((m.min - 1.0).abs() < 0.01);
        assert!((m.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_metric_sample_empty() {
        let m = MetricSample::from_samples(&[]);
        assert_eq!(m.samples, 0);
    }

    #[test]
    fn test_metric_sample_single() {
        let m = MetricSample::from_samples(&[42.0]);
        assert!((m.mean - 42.0).abs() < 0.01);
        assert_eq!(m.samples, 1);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = percentile(&data, 50.0);
        let p95 = percentile(&data, 95.0);
        let p99 = percentile(&data, 99.0);
        assert!((p50 - 5.5).abs() < 0.1);
        assert!((p95 - 9.55).abs() < 0.1);
        assert!((p99 - 9.91).abs() < 0.1);
    }
}
