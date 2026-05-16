use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, Registry, TextEncoder, Encoder,
};

pub struct MetricsCollector {
    registry: Registry,
    pub request_counter: Counter,
    pub request_failures: Counter,
    pub request_latency: Histogram,
    pub active_connections: Gauge,
    pub memory_usage: Gauge,
    pub cpu_usage: Gauge,
    pub queue_depth: Gauge,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let registry = Registry::new();

        let request_counter = Counter::new("nexora_requests_total", "Total requests")
            .expect("metric registered");
        let request_failures = Counter::new("nexora_request_failures_total", "Failed requests")
            .expect("metric registered");
        let request_latency = Histogram::with_opts(
            HistogramOpts::new("nexora_request_latency_seconds", "Request latency (seconds)")
                .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        )
        .expect("metric registered");
        let active_connections =
            Gauge::new("nexora_active_connections", "Active connections")
                .expect("metric registered");
        let memory_usage =
            Gauge::new("nexora_memory_usage_bytes", "Memory usage in bytes")
                .expect("metric registered");
        let cpu_usage = Gauge::new("nexora_cpu_usage_ratio", "CPU usage (0-1)")
            .expect("metric registered");
        let queue_depth =
            Gauge::new("nexora_queue_depth", "Current queue depth")
                .expect("metric registered");

        registry
            .register(Box::new(request_counter.clone()))
            .expect("registered");
        registry
            .register(Box::new(request_failures.clone()))
            .expect("registered");
        registry
            .register(Box::new(request_latency.clone()))
            .expect("registered");
        registry
            .register(Box::new(active_connections.clone()))
            .expect("registered");
        registry
            .register(Box::new(memory_usage.clone()))
            .expect("registered");
        registry
            .register(Box::new(cpu_usage.clone()))
            .expect("registered");
        registry
            .register(Box::new(queue_depth.clone()))
            .expect("registered");

        Self {
            registry,
            request_counter,
            request_failures,
            request_latency,
            active_connections,
            memory_usage,
            cpu_usage,
            queue_depth,
        }
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
        Self::new()
    }
}

impl std::fmt::Debug for MetricsCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricsCollector").finish()
    }
}
