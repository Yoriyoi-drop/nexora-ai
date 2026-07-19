use std::sync::Arc;

use super::collector::ObservabilityCollector;

/// Report metrics dalam berbagai format
pub struct MetricsReporter {
    collector: Arc<ObservabilityCollector>,
}

impl MetricsReporter {
    pub fn new(collector: Arc<ObservabilityCollector>) -> Self {
        Self { collector }
    }

    /// Snapshot sebagai JSON
    pub fn as_json(&self) -> String {
        let metrics = self.collector.snapshot();
        serde_json::to_string_pretty(&metrics).unwrap_or_default()
    }

    /// Snapshot sebagai key=value (prometheus-friendly)
    pub fn as_prometheus(&self) -> String {
        let m = self.collector.snapshot();
        let mut output = String::new();
        output.push_str(&format!("# HELP nexora_cpu_util CPU utilization\n"));
        output.push_str(&format!("# TYPE nexora_cpu_util gauge\n"));
        output.push_str(&format!("nexora_cpu_util {}\n", m.cpu_util_pct));

        output.push_str(&format!("# HELP nexora_gpu_util GPU utilization\n"));
        output.push_str(&format!("nexora_gpu_util {}\n", m.gpu_util_pct));

        output.push_str(&format!("# HELP nexora_tokens_generated Total tokens generated\n"));
        output.push_str(&format!("# TYPE nexora_tokens_generated counter\n"));
        output.push_str(&format!("nexora_tokens_generated {}\n", m.tokens_generated));

        output.push_str(&format!("# HELP nexora_latency_ms Request latency\n"));
        output.push_str(&format!("nexora_latency_ms{{quantile=\"avg\"}} {}\n", m.avg_latency_ms));
        output.push_str(&format!("nexora_latency_ms{{quantile=\"p95\"}} {}\n", m.p95_latency_ms));
        output.push_str(&format!("nexora_latency_ms{{quantile=\"p99\"}} {}\n", m.p99_latency_ms));

        output.push_str(&format!("# HELP nexora_cache_hit_rate Cache hit rate\n"));
        output.push_str(&format!("nexora_cache_hit_rate {}\n", m.cache_hit_rate));

        output.push_str(&format!("# HELP nexora_tool_calls_total Total tool calls\n"));
        output.push_str(&format!("nexora_tool_calls_total {}\n", m.tool_calls_total));

        output.push_str(&format!("# HELP nexora_cost_usd Total cost\n"));
        output.push_str(&format!("nexora_cost_usd {}\n", m.total_cost_usd));

        output.push_str(&format!("# HELP nexora_success_rate Request success rate\n"));
        output.push_str(&format!("nexora_success_rate {}\n", m.success_rate));

        output.push_str(&format!("# HELP nexora_hallucination_rate Hallucination rate\n"));
        output.push_str(&format!("nexora_hallucination_rate {}\n", m.hallucination_rate));

        output
    }

    /// Failure tree sebagai JSON
    pub fn failure_tree(&self, failures: &[FailureRecord]) -> String {
        serde_json::to_string_pretty(failures).unwrap_or_default()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FailureRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub agent_id: Option<String>,
    pub error_type: String,
    pub error_message: String,
    pub retry_attempts: u32,
    pub resolved: bool,
}
