use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ObservabilityMetrics {
    pub cpu_util_pct: f64,
    pub gpu_util_pct: f64,
    pub gpu_memory_used_mb: f64,
    pub ram_used_mb: f64,
    pub ram_total_mb: f64,
    pub tokens_generated: u64,
    pub tokens_per_sec: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub queue_depth: usize,
    pub queue_wait_ms: f64,
    pub requests_in_flight: usize,
    pub cache_hit_rate: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub tool_calls_total: u64,
    pub tool_calls_failed: u64,
    pub tool_call_avg_ms: f64,
    pub total_cost_usd: f64,
    pub estimated_savings_usd: f64,
    pub cost_per_request: f64,
    pub hallucination_rate: f64,
    pub hallucination_count: u64,
    pub active_agents: usize,
    pub idle_agents: usize,
    pub failed_agents: usize,
    pub agent_restarts: u64,
    pub retry_count: u64,
    pub failure_count: u64,
    pub success_rate: f64,
}

impl ObservabilityMetrics {
    pub fn new() -> Self {
        Self {
            cpu_util_pct: 0.0,
            gpu_util_pct: 0.0,
            gpu_memory_used_mb: 0.0,
            ram_used_mb: 0.0,
            ram_total_mb: 0.0,
            tokens_generated: 0,
            tokens_per_sec: 0.0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            queue_depth: 0,
            queue_wait_ms: 0.0,
            requests_in_flight: 0,
            cache_hit_rate: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            cache_evictions: 0,
            tool_calls_total: 0,
            tool_calls_failed: 0,
            tool_call_avg_ms: 0.0,
            total_cost_usd: 0.0,
            estimated_savings_usd: 0.0,
            cost_per_request: 0.0,
            hallucination_rate: 0.0,
            hallucination_count: 0,
            active_agents: 0,
            idle_agents: 0,
            failed_agents: 0,
            agent_restarts: 0,
            retry_count: 0,
            failure_count: 0,
            success_rate: 100.0,
        }
    }
}

impl Default for ObservabilityMetrics {
    fn default() -> Self {
        Self::new()
    }
}
