/// Agent Health Dashboard — visualisasi real-time untuk observability
///
/// Ini adalah struktur data yang bisa dirender oleh TUI atau web dashboard.
/// Data dikumpulkan dari ObservabilityCollector dan disajikan sebagai JSON.
use serde::Serialize;
use std::collections::HashMap;

use super::metrics::ObservabilityMetrics;

/// Dashboard data — siap untuk dirender
#[derive(Debug, Serialize)]
pub struct DashboardData {
    pub timestamp: String,
    pub system: SystemPanel,
    pub performance: PerformancePanel,
    pub cache: CachePanel,
    pub agents: AgentPanel,
    pub cost: CostPanel,
    pub reliability: ReliabilityPanel,
}

#[derive(Debug, Serialize)]
pub struct SystemPanel {
    pub cpu_util: f64,
    pub gpu_util: f64,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub gpu_mem_used_gb: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize)]
pub struct PerformancePanel {
    pub tokens_per_sec: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub queue_depth: usize,
    pub requests_in_flight: usize,
    pub tokens_generated: u64,
}

#[derive(Debug, Serialize)]
pub struct CachePanel {
    pub hit_rate: f64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub layers: HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
pub struct AgentPanel {
    pub active: usize,
    pub idle: usize,
    pub failed: usize,
    pub total: usize,
    pub restarts: u64,
    pub plans_completed: u64,
    pub tasks_completed: u64,
}

#[derive(Debug, Serialize)]
pub struct CostPanel {
    pub total_cost_usd: f64,
    pub estimated_savings_usd: f64,
    pub cost_per_request: f64,
    pub savings_pct: f64,
    pub large_model_pct: f64,
}

#[derive(Debug, Serialize)]
pub struct ReliabilityPanel {
    pub success_rate: f64,
    pub failure_count: u64,
    pub retry_count: u64,
    pub hallucination_rate: f64,
    pub hallucination_count: u64,
    pub failure_tree: Vec<FailureNode>,
}

#[derive(Debug, Serialize)]
pub struct FailureNode {
    pub error: String,
    pub count: u64,
    pub children: Vec<FailureNode>,
}

impl DashboardData {
    pub fn from_metrics(metrics: &ObservabilityMetrics, uptime_secs: u64) -> Self {
        let total_cost = metrics.total_cost_usd;
        let savings = metrics.estimated_savings_usd;
        let savings_pct = if total_cost + savings > 0.0 {
            (savings / (total_cost + savings)) * 100.0
        } else {
            0.0
        };

        let _total_requests = metrics.cache_hits + metrics.cache_misses;

        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            system: SystemPanel {
                cpu_util: metrics.cpu_util_pct,
                gpu_util: metrics.gpu_util_pct,
                ram_used_gb: metrics.ram_used_mb / 1024.0,
                ram_total_gb: metrics.ram_total_mb / 1024.0,
                gpu_mem_used_gb: metrics.gpu_memory_used_mb / 1024.0,
                uptime_seconds: uptime_secs,
            },
            performance: PerformancePanel {
                tokens_per_sec: metrics.tokens_per_sec,
                avg_latency_ms: metrics.avg_latency_ms,
                p95_latency_ms: metrics.p95_latency_ms,
                p99_latency_ms: metrics.p99_latency_ms,
                queue_depth: metrics.queue_depth,
                requests_in_flight: metrics.requests_in_flight,
                tokens_generated: metrics.tokens_generated,
            },
            cache: CachePanel {
                hit_rate: metrics.cache_hit_rate,
                hits: metrics.cache_hits,
                misses: metrics.cache_misses,
                evictions: metrics.cache_evictions,
                layers: HashMap::new(),
            },
            agents: AgentPanel {
                active: metrics.active_agents,
                idle: metrics.idle_agents,
                failed: metrics.failed_agents,
                total: metrics.active_agents + metrics.idle_agents + metrics.failed_agents,
                restarts: metrics.agent_restarts,
                plans_completed: 0,
                tasks_completed: 0,
            },
            cost: CostPanel {
                total_cost_usd: total_cost,
                estimated_savings_usd: savings,
                cost_per_request: metrics.cost_per_request,
                savings_pct,
                large_model_pct: 0.0,
            },
            reliability: ReliabilityPanel {
                success_rate: metrics.success_rate,
                failure_count: metrics.failure_count,
                retry_count: metrics.retry_count,
                hallucination_rate: metrics.hallucination_rate,
                hallucination_count: metrics.hallucination_count,
                failure_tree: vec![],
            },
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}
