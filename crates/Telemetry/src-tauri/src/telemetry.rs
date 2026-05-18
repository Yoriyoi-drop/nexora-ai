use serde::{Deserialize, Serialize};

/// Standard event format for the event pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub timestamp: u64,
    pub source: String,
    pub event_type: String,
    pub value: f64,
    pub severity: String,
    pub label: String,
    pub metadata: Option<serde_json::Value>,
}

impl TelemetryEvent {
    pub fn new(source: &str, event_type: &str, value: f64, severity: &str, label: &str) -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            source: source.to_string(),
            event_type: event_type.to_string(),
            value,
            severity: severity.to_string(),
            label: label.to_string(),
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// System telemetry - full system info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTelemetry {
    pub cpu_usage: f64,
    pub cpu_per_core: Vec<f64>,
    pub cpu_cores: usize,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub ram_percent: f64,
    pub disk_used_gb: f64,
    pub disk_total_gb: f64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub processes: usize,
    pub uptime_secs: u64,
    pub gpu_usage: Option<f64>,
    pub gpu_temp: Option<f64>,
    pub gpu_vram_used_gb: Option<f64>,
    pub gpu_vram_total_gb: Option<f64>,
}

/// Inference telemetry from nexora-inference crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTelemetry {
    pub model_name: String,
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub context_length: u32,
    pub batch_size: u32,
    pub cache_hit_rate: f64,
    pub cache_total_entries: u64,
    pub cache_memory_bytes: u64,
    pub active_requests: u32,
    pub total_requests: u64,
    pub error_rate: f64,
    pub speculative_acceptance_rate: Option<f64>,
    pub kv_cache_usage_pct: Option<f64>,
}

/// Agent telemetry from nexora-agent crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTelemetry {
    pub agent_id: String,
    pub agent_name: String,
    pub status: String,
    pub role: String,
    pub entropy: f64,
    pub reasoning_depth: u32,
    pub tokens_consumed: u64,
    pub tool_calls: u32,
    pub memory_hits: u32,
    pub memory_usage: f64,
    pub uptime_seconds: u64,
    pub hallucinations: u32,
    pub loop_detected: bool,
    pub cluster: String,
    pub cpu_load: f64,
    pub gpu_load: f64,
    pub thoughts_pending: u32,
}

/// Memory telemetry from nexora-memory crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTelemetry {
    pub node_id: String,
    pub label: String,
    pub memory_type: String,
    pub strength: f64,
    pub stability: f64,
    pub access_count: u64,
    pub last_access: u64,
    pub connections: Vec<String>,
    pub cluster: String,
}

/// Memory system summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    pub total_nodes: u64,
    pub episodic_count: u64,
    pub semantic_count: u64,
    pub procedural_count: u64,
    pub working_count: u64,
    pub avg_strength: f64,
    pub avg_stability: f64,
    pub fragmentation_index: f64,
    pub high_strength_nodes: u64,
    pub unstable_nodes: u64,
}

/// Pipeline telemetry from nexora-datastream crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTelemetry {
    pub pipeline_name: String,
    pub samples_loaded: u64,
    pub samples_filtered: u64,
    pub samples_accepted: u64,
    pub total_latency_ms: f64,
    pub throughput_per_sec: f64,
    pub filter_breakdown: Vec<FilterBreakdown>,
    pub backpressure_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterBreakdown {
    pub filter_name: String,
    pub rejected: u64,
    pub percent: f64,
}

/// Hallucination telemetry from nexora-hallucination crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationTelemetry {
    pub total_checked: u64,
    pub total_blocked: u64,
    pub total_flagged: u64,
    pub hallucination_rate: f64,
    pub risk_score_avg: f64,
    pub pre_gen_blocked: u64,
    pub in_gen_corrected: u64,
    pub post_gen_flagged: u64,
    pub top_risk_sources: Vec<(String, f64)>,
}

/// Training telemetry from nexora-ai training pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTelemetry {
    pub is_training: bool,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_step: u64,
    pub total_steps: u64,
    pub loss: Option<f64>,
    pub learning_rate: Option<f64>,
    pub grad_norm: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub samples_processed: u64,
    pub time_elapsed_secs: f64,
    pub estimated_time_remaining_secs: Option<f64>,
}

/// Model telemetry from nexora-foundation crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTelemetry {
    pub model_id: String,
    pub model_name: String,
    pub model_type: String,
    pub provider: String,
    pub status: String,
    pub gpu_memory_mb: f64,
    pub gpu_memory_total_mb: f64,
    pub throughput_tokens_per_sec: f64,
    pub latency_ms: f64,
    pub accuracy: f64,
    pub active_agents: u32,
    pub quantization: String,
    pub temperature: f64,
}

/// Token flow telemetry for TokenFlowView
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFlowTelemetry {
    pub source: String,
    pub target: String,
    pub volume: f64,
    pub efficiency: f64,
    pub is_bottleneck: bool,
    pub latency_ms: f64,
}

/// Comprehensive telemetry snapshot - all data in one call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub timestamp: u64,
    pub system: Option<SystemTelemetry>,
    pub ai_health: Option<AiHealthTelemetry>,
    pub inference: Option<InferenceTelemetry>,
    pub agents: Vec<AgentTelemetry>,
    pub memory_nodes: Vec<MemoryTelemetry>,
    pub memory_summary: Option<MemorySummary>,
    pub pipelines: Vec<PipelineTelemetry>,
    pub hallucinations: Option<HallucinationTelemetry>,
    pub training: Option<TrainingTelemetry>,
    pub models: Vec<ModelTelemetry>,
    pub token_flows: Vec<TokenFlowTelemetry>,
}

/// AI health from /health/detailed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiHealthTelemetry {
    pub healthy: bool,
    pub uptime_seconds: f64,
    pub version: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub requests_per_second: f64,
    pub active_connections: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub thread_count: usize,
    pub error_rate_percent: f64,
    pub active_models: Vec<String>,
    pub component_health: Vec<ComponentHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: String,
    pub message: String,
}

/// Response wrapper for memory telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResponse {
    pub nodes: Vec<MemoryTelemetry>,
    pub summary: Option<MemorySummary>,
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub source: String,
    pub metric: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub severity: String,
    pub enabled: bool,
    pub cooldown_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    RateIncrease,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub rule_id: String,
    pub name: String,
    pub message: String,
    pub severity: String,
    pub value: f64,
    pub threshold: f64,
    pub timestamp: u64,
    pub acknowledged: bool,
}

// === HTTP Client Helpers for fetching telemetry from nexora-ai ===

#[derive(Debug, Clone)]
pub struct TelemetryClient {
    client: reqwest::Client,
    base_url: String,
}

impl TelemetryClient {
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn fetch_health(&self) -> Result<AiHealthTelemetry, String> {
        #[derive(serde::Deserialize)]
        struct HealthResp { healthy: bool, uptime_seconds: serde_json::Value, version: String }
        #[derive(serde::Deserialize)]
        struct DetailedResp {
            healthy: bool, status: Option<String>,
            statistics: Option<serde_json::Value>,
            system: Option<serde_json::Value>,
            uptime_seconds: Option<serde_json::Value>,
            components: Option<Vec<ComponentHealth>>,
        }

        let health: HealthResp = self.fetch("/health").await?;
        let detailed: Option<DetailedResp> = self.fetch_opt("/health/detailed").await;
        let metrics_val: Option<serde_json::Value> = self.fetch_opt("/metrics").await;

        let stats = detailed.as_ref().and_then(|d| d.statistics.as_ref());
        let sys = detailed.as_ref().and_then(|d| d.system.as_ref());

        let uptime_seconds = detailed.as_ref()
            .and_then(|d| d.uptime_seconds.as_ref())
            .map(j2f)
            .unwrap_or_else(|| j2f(&health.uptime_seconds));

        let active_models: Vec<String> = self.fetch_opt::<serde_json::Value>("/system/stats").await
            .and_then(|v| v.get("active_models").and_then(|arr| arr.as_array().map(|a| a.clone())))
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let component_health = detailed.as_ref()
            .and_then(|d| d.components.as_ref())
            .cloned()
            .unwrap_or_default();

        Ok(AiHealthTelemetry {
            healthy: health.healthy,
            uptime_seconds,
            version: health.version,
            total_requests: stats.and_then(|s| j2u64(s.get("total_requests"))).unwrap_or(0),
            successful_requests: stats.and_then(|s| j2u64(s.get("successful_requests"))).unwrap_or(0),
            failed_requests: stats.and_then(|s| j2u64(s.get("failed_requests"))).unwrap_or(0),
            average_response_time_ms: stats.and_then(|s| j2f64(s.get("average_response_time_ms"))).unwrap_or(0.0),
            requests_per_second: stats.and_then(|s| j2f64(s.get("requests_per_second"))).unwrap_or(0.0),
            active_connections: stats.and_then(|s| j2u(s.get("active_connections"))).unwrap_or(0),
            memory_usage_mb: sys.and_then(|s| j2f64(s.get("memory_usage_mb"))).unwrap_or(0.0),
            cpu_usage_percent: sys.and_then(|s| j2f64(s.get("cpu_usage_percent"))).unwrap_or(0.0),
            thread_count: sys.and_then(|s| j2u(s.get("thread_count"))).unwrap_or(0),
            error_rate_percent: metrics_val.as_ref()
                .and_then(|m| j2f64(m.get("error_rate_percent"))
                    .or_else(|| j2f64(m.get("error_rate"))))
                .unwrap_or(0.0),
            active_models,
            component_health,
        })
    }

    pub async fn fetch_inference(&self) -> Result<InferenceTelemetry, String> {
        #[derive(serde::Deserialize)]
        struct InfResp {
            model_name: Option<String>,
            tokens_per_second: Option<f64>,
            latency_ms: Option<f64>,
            latency_p50_ms: Option<f64>,
            latency_p99_ms: Option<f64>,
            context_length: Option<u32>,
            batch_size: Option<u32>,
            cache_hit_rate: Option<f64>,
            cache_total_entries: Option<u64>,
            cache_memory_bytes: Option<u64>,
            active_requests: Option<u32>,
            total_requests: Option<u64>,
            error_rate: Option<f64>,
            speculative_acceptance_rate: Option<f64>,
            kv_cache_usage_pct: Option<f64>,
        }

        let resp: Option<InfResp> = self.fetch_opt("/telemetry/inference").await;
        let r = resp.unwrap_or(InfResp {
            model_name: None, tokens_per_second: None, latency_ms: None,
            latency_p50_ms: None, latency_p99_ms: None, context_length: None,
            batch_size: None, cache_hit_rate: None, cache_total_entries: None,
            cache_memory_bytes: None, active_requests: None, total_requests: None,
            error_rate: None, speculative_acceptance_rate: None, kv_cache_usage_pct: None,
        });

        Ok(InferenceTelemetry {
            model_name: r.model_name.unwrap_or_else(|| "unknown".into()),
            tokens_per_second: r.tokens_per_second.unwrap_or(0.0),
            latency_ms: r.latency_ms.unwrap_or(0.0),
            latency_p50_ms: r.latency_p50_ms.unwrap_or(0.0),
            latency_p99_ms: r.latency_p99_ms.unwrap_or(0.0),
            context_length: r.context_length.unwrap_or(0),
            batch_size: r.batch_size.unwrap_or(1),
            cache_hit_rate: r.cache_hit_rate.unwrap_or(0.0),
            cache_total_entries: r.cache_total_entries.unwrap_or(0),
            cache_memory_bytes: r.cache_memory_bytes.unwrap_or(0),
            active_requests: r.active_requests.unwrap_or(0),
            total_requests: r.total_requests.unwrap_or(0),
            error_rate: r.error_rate.unwrap_or(0.0),
            speculative_acceptance_rate: r.speculative_acceptance_rate,
            kv_cache_usage_pct: r.kv_cache_usage_pct,
        })
    }

    pub async fn fetch_agents(&self) -> Result<Vec<AgentTelemetry>, String> {
        let resp: Option<Vec<AgentTelemetry>> = self.fetch_opt("/telemetry/agents").await;
        Ok(resp.unwrap_or_default())
    }

    pub async fn fetch_memory(&self) -> Result<(Vec<MemoryTelemetry>, Option<MemorySummary>), String> {
        #[derive(serde::Deserialize)]
        struct MemResp {
            nodes: Vec<MemoryTelemetry>,
            summary: Option<MemorySummary>,
        }
        let resp: Option<MemResp> = self.fetch_opt("/telemetry/memory").await;
        Ok(resp.map(|r| (r.nodes, r.summary)).unwrap_or_default())
    }

    pub async fn fetch_pipelines(&self) -> Result<Vec<PipelineTelemetry>, String> {
        let resp: Option<Vec<PipelineTelemetry>> = self.fetch_opt("/telemetry/pipeline").await;
        Ok(resp.unwrap_or_default())
    }

    pub async fn fetch_hallucinations(&self) -> Result<HallucinationTelemetry, String> {
        #[derive(serde::Deserialize)]
        struct HalResp {
            total_checked: Option<u64>, total_blocked: Option<u64>,
            total_flagged: Option<u64>, hallucination_rate: Option<f64>,
            risk_score_avg: Option<f64>, pre_gen_blocked: Option<u64>,
            in_gen_corrected: Option<u64>, post_gen_flagged: Option<u64>,
            top_risk_sources: Option<Vec<(String, f64)>>,
        }
        let resp: Option<HalResp> = self.fetch_opt("/telemetry/hallucination").await;
        let r = resp.unwrap_or(HalResp {
            total_checked: None, total_blocked: None, total_flagged: None,
            hallucination_rate: None, risk_score_avg: None, pre_gen_blocked: None,
            in_gen_corrected: None, post_gen_flagged: None, top_risk_sources: None,
        });
        Ok(HallucinationTelemetry {
            total_checked: r.total_checked.unwrap_or(0),
            total_blocked: r.total_blocked.unwrap_or(0),
            total_flagged: r.total_flagged.unwrap_or(0),
            hallucination_rate: r.hallucination_rate.unwrap_or(0.0),
            risk_score_avg: r.risk_score_avg.unwrap_or(0.0),
            pre_gen_blocked: r.pre_gen_blocked.unwrap_or(0),
            in_gen_corrected: r.in_gen_corrected.unwrap_or(0),
            post_gen_flagged: r.post_gen_flagged.unwrap_or(0),
            top_risk_sources: r.top_risk_sources.unwrap_or_default(),
        })
    }

    pub async fn fetch_training(&self) -> Result<TrainingTelemetry, String> {
        #[derive(serde::Deserialize)]
        struct TrResp {
            is_training: Option<bool>, current_epoch: Option<u32>,
            total_epochs: Option<u32>, current_step: Option<u64>,
            total_steps: Option<u64>, loss: Option<f64>,
            learning_rate: Option<f64>, grad_norm: Option<f64>,
            tokens_per_second: Option<f64>, samples_processed: Option<u64>,
            time_elapsed_secs: Option<f64>, estimated_time_remaining_secs: Option<f64>,
        }
        let resp: Option<TrResp> = self.fetch_opt("/telemetry/training").await;
        let r = resp.unwrap_or(TrResp {
            is_training: None, current_epoch: None, total_epochs: None,
            current_step: None, total_steps: None, loss: None,
            learning_rate: None, grad_norm: None, tokens_per_second: None,
            samples_processed: None, time_elapsed_secs: None,
            estimated_time_remaining_secs: None,
        });
        Ok(TrainingTelemetry {
            is_training: r.is_training.unwrap_or(false),
            current_epoch: r.current_epoch.unwrap_or(0),
            total_epochs: r.total_epochs.unwrap_or(0),
            current_step: r.current_step.unwrap_or(0),
            total_steps: r.total_steps.unwrap_or(0),
            loss: r.loss,
            learning_rate: r.learning_rate,
            grad_norm: r.grad_norm,
            tokens_per_second: r.tokens_per_second,
            samples_processed: r.samples_processed.unwrap_or(0),
            time_elapsed_secs: r.time_elapsed_secs.unwrap_or(0.0),
            estimated_time_remaining_secs: r.estimated_time_remaining_secs,
        })
    }

    pub async fn fetch_models(&self) -> Result<Vec<ModelTelemetry>, String> {
        let resp: Option<Vec<ModelTelemetry>> = self.fetch_opt("/telemetry/models").await;
        Ok(resp.unwrap_or_default())
    }

    pub async fn fetch_token_flows(&self) -> Result<Vec<TokenFlowTelemetry>, String> {
        let resp: Option<Vec<TokenFlowTelemetry>> = self.fetch_opt("/telemetry/token-flows").await;
        Ok(resp.unwrap_or_default())
    }

    pub async fn fetch_snapshot(&self) -> Result<TelemetrySnapshot, String> {
        let health = self.fetch_health().await.ok();
        let inference = self.fetch_inference().await.ok();
        let agents = self.fetch_agents().await.unwrap_or_default();
        let (memory_nodes, memory_summary) = self.fetch_memory().await.unwrap_or_default();
        let pipelines = self.fetch_pipelines().await.unwrap_or_default();
        let hallucinations = self.fetch_hallucinations().await.ok();
        let training = self.fetch_training().await.ok();
        let models = self.fetch_models().await.unwrap_or_default();
        let token_flows = self.fetch_token_flows().await.unwrap_or_default();

        Ok(TelemetrySnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
            system: None,
            ai_health: health,
            inference,
            agents,
            memory_nodes,
            memory_summary,
            pipelines,
            hallucinations,
            training,
            models,
            token_flows,
        })
    }

    async fn fetch<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let url = format!("{}{}", self.base_url, path);
        self.client.get(&url).send().await
            .map_err(|e| format!("HTTP {}: {}", path, e))?
            .json::<T>().await
            .map_err(|e| format!("JSON {}: {}", path, e))
    }

    async fn fetch_opt<T: serde::de::DeserializeOwned>(&self, path: &str) -> Option<T> {
        let url = format!("{}{}", self.base_url, path);
        self.client.get(&url).send().await.ok()?
            .json::<T>().await.ok()
    }
}

fn j2f(v: &serde_json::Value) -> f64 {
    v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)).or_else(|| v.as_i64().map(|i| i as f64)).unwrap_or(0.0)
}

fn j2f64(v: Option<&serde_json::Value>) -> Option<f64> {
    v.and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)).or_else(|| v.as_i64().map(|i| i as f64)))
}

fn j2u64(v: Option<&serde_json::Value>) -> Option<u64> {
    v.and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)).or_else(|| v.as_f64().map(|f| f as u64)))
}

fn j2u(v: Option<&serde_json::Value>) -> Option<usize> {
    v.and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)).or_else(|| v.as_f64().map(|f| f as u64))).map(|u| u as usize)
}
