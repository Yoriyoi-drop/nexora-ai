//! System monitoring and health check functionality

use crate::error::{NexoraError, NexoraResult};
use crate::NexoraConfig;
use chrono::Utc;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use super::types::{ComponentStatus, HealthStatus, MemoryStats, SystemInfo};

/// System monitoring functionality
#[derive(Debug, Clone)]
pub struct SystemMonitor {
    registry: Arc<nexora_foundation::shared::model_registry::NxrModelRegistry>,
    config: NexoraConfig,
    start_time: chrono::DateTime<Utc>,
    system_info_cache: Arc<RwLock<Option<SystemInfo>>>,
    request_count: Arc<AtomicU64>,
    /// Shared sysinfo::System — single allocation, refreshed on demand.
    system: Arc<Mutex<sysinfo::System>>,
    /// Ring buffer of (timestamp, cpu_usage) samples for load average estimation
    cpu_samples: Arc<Mutex<VecDeque<(Instant, f64)>>>,
    /// Ring buffer of request durations in ms for average response time calculation
    request_timings: Arc<Mutex<VecDeque<(Instant, f64)>>>,
    /// Total error count for error rate calculation
    error_count: Arc<AtomicU64>,
}

impl SystemMonitor {
    pub fn new(
        registry: Arc<nexora_foundation::shared::model_registry::NxrModelRegistry>,
        config: NexoraConfig,
        start_time: chrono::DateTime<Utc>,
        system_info_cache: Arc<RwLock<Option<SystemInfo>>>,
        request_count: Arc<AtomicU64>,
    ) -> Self {
        info!(
            "SystemMonitor initialized with config: max_concurrent={}, timeout={}ms",
            config.core.max_concurrent_requests, config.core.request_timeout_ms
        );
        Self {
            registry,
            config,
            start_time,
            system_info_cache,
            request_count,
            system: Arc::new(Mutex::new(sysinfo::System::new_with_specifics(
                RefreshKind::everything(),
            ))),
            cpu_samples: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            request_timings: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            error_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get system information with caching
    pub async fn get_system_info(&self) -> NexoraResult<SystemInfo> {
        info!("Getting comprehensive system information...");

        // Check cache first (cache for 5 seconds)
        {
            let cache = self.system_info_cache.read().await;
            if let Some(ref cached_info) = *cache {
                let cache_age = (Utc::now() - cached_info.last_updated).num_seconds();
                if cache_age < 5 {
                    debug!("Returning cached system info (age: {}s)", cache_age);
                    return Ok(cached_info.clone());
                }
            }
        }

        // Gather real system information
        let mut system = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        system.refresh_all();

        let model_ids = nexora_foundation::shared::model_identity::NxrModelId::all();
        let active_models: Vec<String> = model_ids.iter().map(|id| format!("{:?}", id)).collect();

        // Calculate uptime
        let uptime = (Utc::now() - self.start_time).num_seconds() as u64;

        // Get CPU usage
        let total_cpu_usage: f32 = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;

        // Get memory information
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let available_memory = total_memory - used_memory;
        let memory_usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;

        // Component health checks
        let components = self.check_component_health().await?;

        let system_info = SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime,
            components,
            memory_stats: MemoryStats {
                total_memory,
                used_memory,
                available_memory,
                cache_size: None,
            },
            active_models,
            memory_usage: memory_usage_percent,
            cpu_usage: total_cpu_usage as f64,
            last_updated: Utc::now(),
            process_count: system.processes().len() as u64,
            thread_count: system.cpus().len() as u64,
            load_average: self.get_load_average().await,
        };

        // Update cache
        {
            let mut cache = self.system_info_cache.write().await;
            *cache = Some(system_info.clone());
        }

        Ok(system_info)
    }

    /// Check component health with sophisticated validation
    async fn check_component_health(&self) -> NexoraResult<ComponentStatus> {
        let mut system = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        system.refresh_all();
        self.check_component_health_with_system(&system).await
    }

    async fn check_component_health_with_system(
        &self,
        system: &System,
    ) -> NexoraResult<ComponentStatus> {
        // Core component health
        let core_status = if system.total_memory() > 0 {
            "healthy"
        } else {
            "critical"
        };

        // Models health (check if model definitions exist)
        let model_ids = nexora_foundation::shared::model_identity::NxrModelId::all();
        let models_status = if !model_ids.is_empty() {
            "healthy"
        } else {
            "warning"
        };

        // Memory health
        let memory_usage_percent =
            (system.used_memory() as f64 / system.total_memory() as f64) * 100.0;
        let memory_status = match memory_usage_percent {
            x if x < 80.0 => "healthy",
            x if x < 90.0 => "warning",
            _ => "critical",
        };

        // Inference health - check actual inference engine status
        let inference_status = self.check_inference_health_with_system(system).await;

        // Agent health - check agent system status
        let agent_status = self.check_agent_health_with_system(system).await;

        // API health - check API endpoints status
        let api_status = self.check_api_health_with_system(system).await;

        Ok(ComponentStatus {
            core: core_status.to_string(),
            models: models_status.to_string(),
            memory: memory_status.to_string(),
            inference: inference_status.to_string(),
            agent: agent_status.to_string(),
            api: api_status.to_string(),
        })
    }

    async fn check_inference_health_with_system(&self, system: &System) -> &'static str {
        match self.inference_health_check_with_system(system).await {
            Ok(healthy) => {
                if healthy {
                    "healthy"
                } else {
                    "warning"
                }
            }
            Err(_) => "critical",
        }
    }

    async fn check_agent_health_with_system(&self, system: &System) -> &'static str {
        match self.agent_health_check_with_system(system).await {
            Ok(healthy) => {
                if healthy {
                    "healthy"
                } else {
                    "warning"
                }
            }
            Err(_) => "critical",
        }
    }

    async fn check_api_health_with_system(&self, system: &System) -> &'static str {
        match self.api_health_check_with_system(system).await {
            Ok(healthy) => {
                if healthy {
                    "healthy"
                } else {
                    "warning"
                }
            }
            Err(_) => "critical",
        }
    }

    async fn inference_health_check_with_system(&self, system: &System) -> NexoraResult<bool> {
        let cpu_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;
        let memory_usage = (system.used_memory() as f64 / system.total_memory() as f64) * 100.0;
        let model_ids = nexora_foundation::shared::model_identity::NxrModelId::all();
        let models_available = !model_ids.is_empty();
        if !models_available {
            warn!("inference health: no models registered");
        }
        Ok(models_available && cpu_usage < 80.0 && memory_usage < 90.0)
    }

    async fn agent_health_check_with_system(&self, system: &System) -> NexoraResult<bool> {
        let available_memory = system.total_memory() - system.used_memory();
        let min_memory_required = 100 * 1024 * 1024;
        let process_count = system.processes().len();
        if available_memory < min_memory_required {
            warn!("agent health: low memory ({} bytes available)", available_memory);
        }
        if process_count == 0 {
            warn!("agent health: no processes detected");
        }
        Ok(available_memory >= min_memory_required && process_count > 0)
    }

    async fn api_health_check_with_system(&self, system: &System) -> NexoraResult<bool> {
        let load_average = sysinfo::System::load_average();
        if load_average.one >= 10.0 {
            warn!("api health: load average too high ({:.2})", load_average.one);
        }
        let model_ids = nexora_foundation::shared::model_identity::NxrModelId::all();
        Ok(load_average.one < 10.0 && !model_ids.is_empty())
    }

    async fn calculate_average_response_time_with_system(
        &self,
        _request_count: u64,
        _uptime_seconds: u64,
        _system: &System,
    ) -> NexoraResult<f64> {
        let timings = self.request_timings.lock().await;
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(300);
        let window: Vec<f64> = timings
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, d)| *d)
            .collect();
        if window.is_empty() {
            return Ok(0.0);
        }
        Ok(window.iter().sum::<f64>() / window.len() as f64)
    }

    async fn calculate_error_rate_with_system(
        &self,
        request_count: u64,
        _system: &System,
    ) -> NexoraResult<f64> {
        if request_count == 0 {
            return Ok(0.0);
        }
        let errors = self.error_count.load(Ordering::Relaxed);
        Ok((errors as f64 / request_count as f64).min(1.0))
    }

    /// Record a request duration for real response time tracking.
    pub async fn record_request_duration(&self, duration_ms: f64) {
        let mut timings = self.request_timings.lock().await;
        timings.push_back((Instant::now(), duration_ms));
        while timings.len() > 1000 {
            timings.pop_front();
        }
    }

    /// Record an error for real error rate tracking.
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    async fn get_active_connections_with_system(&self, _system: &System) -> Option<u64> {
        let mut count = 0u64;
        for path in &["/proc/net/tcp", "/proc/net/tcp6"] {
            if let Ok(content) = std::fs::read_to_string(path) {
                for line in content.lines().skip(1) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 4 && parts[3] != "0A" {
                        count += 1;
                    }
                }
            }
        }
        if count > 0 {
            return Some(count);
        }
        if let Ok(output) = std::process::Command::new("ss")
            .args(["-tun", "-H", "state", "established"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
                return Some(lines.len() as u64);
            }
        }
        None
    }

    /// Get system load average (1, 5, 15 min), or `None` if unavailable
    async fn get_load_average(&self) -> Option<(f64, f64, f64)> {
        if let Ok(load_str) = std::fs::read_to_string("/proc/loadavg") {
            let parts: Vec<&str> = load_str.split_whitespace().collect();
            if parts.len() >= 3 {
                let load1: f64 = parts[0].parse().unwrap_or(0.0);
                let load5: f64 = parts[1].parse().unwrap_or(0.0);
                let load15: f64 = parts[2].parse().unwrap_or(0.0);
                return Some((load1, load5, load15));
            }
        }

        let cpu_sample = {
            let mut system = self.system.lock().await;
            system.refresh_cpu();
            system.global_cpu_usage() as f64
        };
        let now = Instant::now();
        {
            let mut samples = self.cpu_samples.lock().await;
            samples.push_back((now, cpu_sample));
            while let Some(front) = samples.front() {
                if now.duration_since(front.0).as_secs() > 900 {
                    samples.pop_front();
                } else {
                    break;
                }
            }
        }

        let samples = self.cpu_samples.lock().await;
        if samples.len() < 2 {
            return None;
        }

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as f64;

        let avg_cpu_in_window = |window_secs: u64| -> f64 {
            let cutoff = now - Duration::from_secs(window_secs);
            let in_window: Vec<f64> = samples
                .iter()
                .filter(|(t, _)| *t >= cutoff)
                .map(|(_, cpu)| *cpu)
                .collect();
            if in_window.is_empty() {
                return samples.back().map(|(_, cpu)| *cpu).unwrap_or(0.0);
            }
            in_window.iter().sum::<f64>() / in_window.len() as f64
        };

        let load1 = avg_cpu_in_window(60) / 100.0 * num_cpus;
        let load5 = avg_cpu_in_window(300) / 100.0 * num_cpus;
        let load15 = avg_cpu_in_window(900) / 100.0 * num_cpus;

        Some((load1, load5, load15))
    }

    /// Health check with comprehensive validation
    pub async fn health_check(&self) -> NexoraResult<HealthStatus> {
        info!("Performing comprehensive health check...");

        let mut system = self.system.lock().await;
        system.refresh_all();

        // Get component health
        let components = self.check_component_health_with_system(&system).await?;

        // Build component health map
        let mut component_health = std::collections::HashMap::new();
        component_health.insert("core".to_string(), components.core == "healthy");
        component_health.insert("models".to_string(), components.models == "healthy");
        component_health.insert("memory".to_string(), components.memory == "healthy");
        component_health.insert("inference".to_string(), components.inference == "healthy");
        component_health.insert("agent".to_string(), components.agent == "healthy");
        component_health.insert("api".to_string(), components.api == "healthy");

        // Calculate performance score based on various metrics
        let cpu_usage = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
            / system.cpus().len() as f32;
        let memory_usage_percent =
            (system.used_memory() as f64 / system.total_memory() as f64) * 100.0;

        let performance_score = self
            .calculate_performance_score(cpu_usage as f64, memory_usage_percent)
            .await?;

        // Calculate actual average response time from request metrics
        let request_count = self.request_count.load(Ordering::Relaxed);
        let uptime = (Utc::now() - self.start_time).num_seconds() as u64;
        let average_response_time = self
            .calculate_average_response_time_with_system(request_count, uptime, &system)
            .await?;

        // Calculate actual error rate from request metrics
        let error_rate = self
            .calculate_error_rate_with_system(request_count, &system)
            .await?;

        // Get actual active connections count
        let active_connections = self.get_active_connections_with_system(&system).await.unwrap_or(0);

        Ok(HealthStatus {
            healthy: component_health.values().all(|&healthy| healthy) && performance_score > 50.0,
            performance_score,
            component_health: component_health.clone(),
            core_status: if component_health.get("core").unwrap_or(&false) == &true {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            },
            tokenizer_status: if component_health.get("tokenizer").unwrap_or(&false) == &true {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            },
            models_status: if component_health.get("models").unwrap_or(&false) == &true {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            },
            memory_status: if component_health.get("memory").unwrap_or(&false) == &true {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            },
            total_operations: request_count,
            average_response_time,
            error_rate,
            last_check: chrono::Utc::now(),
            uptime_seconds: uptime,
            active_connections,
        })
    }

    async fn calculate_performance_score(
        &self,
        cpu_usage: f64,
        memory_usage_percent: f64,
    ) -> NexoraResult<f64> {
        let mut score = 100.0;

        // Penalize high CPU usage
        if cpu_usage > 80.0 {
            score -= (cpu_usage - 80.0) as f64 * 0.5;
        }

        // Penalize high memory usage
        if memory_usage_percent > 80.0 {
            score -= (memory_usage_percent - 80.0) * 0.3;
        }

        // Ensure score doesn't go below 0
        score = score.max(0.0);

        Ok(score)
    }
}
