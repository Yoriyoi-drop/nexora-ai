use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Registry, TextEncoder};

pub struct MetricsCollector {
    registry: Registry,
    pub request_counter: Counter,
    pub request_failures: Counter,
    pub request_latency: Histogram,
    pub active_connections: Gauge,
    pub memory_usage: Gauge,
    pub cpu_usage: Gauge,
    pub queue_depth: Gauge,
    pub gpu_forward_errors: Counter,
    pub gpu_cpu_fallbacks: Counter,
    pub gpu_resident_successes: Counter,
    pub gpu_resident_fallbacks: Counter,
    pub gpu_tokens: Counter,
    pub cpu_tokens: Counter,
    pub gpu_alive: Gauge,
    pub gpu_memory_bytes: Gauge,
    pub gpu_memory_percent: Gauge,
    pub cache_hit_ratio: Gauge,
    pub gpu_math_fallbacks: Counter,
    pub inference_queue_depth: Gauge,
    pub throughput_tokens: Counter,
}

impl MetricsCollector {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        let request_counter = Counter::new("nexora_requests_total", "Total requests")?;
        let request_failures = Counter::new("nexora_request_failures_total", "Failed requests")?;
        let request_latency = Histogram::with_opts(
            HistogramOpts::new(
                "nexora_request_latency_seconds",
                "Request latency (seconds)",
            )
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
        )?;
        let active_connections = Gauge::new("nexora_active_connections", "Active connections")?;
        let memory_usage = Gauge::new("nexora_memory_usage_bytes", "Memory usage in bytes")?;
        let cpu_usage = Gauge::new("nexora_cpu_usage_ratio", "CPU usage (0-1)")?;
        let queue_depth = Gauge::new("nexora_queue_depth", "Current queue depth")?;
        let gpu_forward_errors = Counter::new(
            "nexora_gpu_forward_errors_total",
            "Total GPU forward failures",
        )?;
        let gpu_cpu_fallbacks = Counter::new(
            "nexora_gpu_cpu_fallbacks_total",
            "Total GPU→CPU fallback events",
        )?;
        let gpu_resident_successes = Counter::new(
            "nexora_gpu_resident_successes_total",
            "GPU-resident generation successes",
        )?;
        let gpu_resident_fallbacks = Counter::new(
            "nexora_gpu_resident_fallbacks_total",
            "GPU-resident generation fallbacks",
        )?;
        let gpu_tokens = Counter::new(
            "nexora_gpu_tokens_total",
            "Total tokens generated via GPU path",
        )?;
        let cpu_tokens = Counter::new(
            "nexora_cpu_tokens_total",
            "Total tokens generated via CPU path",
        )?;
        let gpu_alive = Gauge::new("nexora_gpu_alive", "GPU device health (1=alive, 0=dead)")?;
        let gpu_memory_bytes = Gauge::new("nexora_gpu_memory_bytes", "GPU memory usage in bytes")?;
        let gpu_memory_percent =
            Gauge::new("nexora_gpu_memory_percent", "GPU memory usage percent")?;
        let cache_hit_ratio = Gauge::new("nexora_cache_hit_ratio", "Prefix cache hit ratio (0-1)")?;
        let gpu_math_fallbacks = Counter::new(
            "nexora_gpu_math_fallbacks_total",
            "Total GPU math fallback events",
        )?;
        let inference_queue_depth = Gauge::new(
            "nexora_inference_queue_depth",
            "Inference engine queue depth",
        )?;
        let throughput_tokens = Counter::new(
            "nexora_throughput_tokens_total",
            "Total tokens generated (all paths)",
        )?;

        registry.register(Box::new(request_counter.clone()))?;
        registry.register(Box::new(request_failures.clone()))?;
        registry.register(Box::new(request_latency.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(memory_usage.clone()))?;
        registry.register(Box::new(cpu_usage.clone()))?;
        registry.register(Box::new(queue_depth.clone()))?;
        registry.register(Box::new(gpu_forward_errors.clone()))?;
        registry.register(Box::new(gpu_cpu_fallbacks.clone()))?;
        registry.register(Box::new(gpu_resident_successes.clone()))?;
        registry.register(Box::new(gpu_resident_fallbacks.clone()))?;
        registry.register(Box::new(gpu_tokens.clone()))?;
        registry.register(Box::new(cpu_tokens.clone()))?;
        registry.register(Box::new(gpu_alive.clone()))?;
        registry.register(Box::new(gpu_memory_bytes.clone()))?;
        registry.register(Box::new(gpu_memory_percent.clone()))?;
        registry.register(Box::new(cache_hit_ratio.clone()))?;
        registry.register(Box::new(gpu_math_fallbacks.clone()))?;
        registry.register(Box::new(inference_queue_depth.clone()))?;
        registry.register(Box::new(throughput_tokens.clone()))?;

        Ok(Self {
            registry,
            request_counter,
            request_failures,
            request_latency,
            active_connections,
            memory_usage,
            cpu_usage,
            queue_depth,
            gpu_forward_errors,
            gpu_cpu_fallbacks,
            gpu_resident_successes,
            gpu_resident_fallbacks,
            gpu_tokens,
            cpu_tokens,
            gpu_alive,
            gpu_memory_bytes,
            gpu_memory_percent,
            cache_hit_ratio,
            gpu_math_fallbacks,
            inference_queue_depth,
            throughput_tokens,
        })
    }

    pub fn record_request(&self, success: bool, latency_secs: f64) {
        self.request_counter.inc();
        self.request_latency.observe(latency_secs);
        if !success {
            self.request_failures.inc();
        }
    }

    pub fn set_active_connections(&self, count: usize) {
        self.active_connections.set(count as f64);
    }

    pub fn set_memory_usage(&self, bytes: f64) {
        self.memory_usage.set(bytes);
    }

    pub fn set_cpu_usage(&self, ratio: f64) {
        self.cpu_usage.set(ratio);
    }

    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.set(depth as f64);
    }

    pub fn set_gpu_forward_errors(&self, count: u64) {
        self.gpu_forward_errors.inc_by(count as f64);
    }

    pub fn set_gpu_cpu_fallbacks(&self, count: u64) {
        self.gpu_cpu_fallbacks.inc_by(count as f64);
    }

    pub fn set_gpu_resident_successes(&self, count: u64) {
        self.gpu_resident_successes.inc_by(count as f64);
    }

    pub fn set_gpu_resident_fallbacks(&self, count: u64) {
        self.gpu_resident_fallbacks.inc_by(count as f64);
    }

    pub fn set_gpu_tokens(&self, count: u64) {
        self.gpu_tokens.inc_by(count as f64);
    }

    pub fn set_cpu_tokens(&self, count: u64) {
        self.cpu_tokens.inc_by(count as f64);
    }

    pub fn set_gpu_alive(&self, alive: bool) {
        self.gpu_alive.set(if alive { 1.0 } else { 0.0 });
    }

    pub fn set_gpu_memory_bytes(&self, bytes: u64) {
        self.gpu_memory_bytes.set(bytes as f64);
    }

    pub fn set_gpu_memory_percent(&self, percent: f64) {
        self.gpu_memory_percent.set(percent);
    }

    pub fn set_cache_hit_ratio(&self, ratio: f64) {
        self.cache_hit_ratio.set(ratio);
    }

    pub fn set_gpu_math_fallbacks(&self, count: u64) {
        self.gpu_math_fallbacks.inc_by(count as f64);
    }

    pub fn set_inference_queue_depth(&self, depth: usize) {
        self.inference_queue_depth.set(depth as f64);
    }

    pub fn set_throughput_tokens(&self, count: u64) {
        self.throughput_tokens.inc_by(count as f64);
    }

    pub fn gather_prometheus(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
            return format!("# error encoding metrics: {}", e);
        }
        String::from_utf8(buffer).unwrap_or_default()
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new().expect("MetricsCollector::default() failed to register prometheus metrics")
    }
}

impl std::fmt::Debug for MetricsCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsCollector").finish()
    }
}
