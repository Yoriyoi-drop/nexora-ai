//! Inference Runtime
//!
//! Runtime state management untuk inference engine.

use chrono::{DateTime, Utc};
use procfs::process::Process as ProcProcess;
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{InferenceError, Result};

/// Runtime state untuk inference engine
pub struct InferenceRuntime {
    /// Runtime configuration
    config: RuntimeConfig,
    /// Current runtime state
    state: Arc<RwLock<RuntimeState>>,
    /// Resource usage tracking
    resource_usage: Arc<RwLock<ResourceUsage>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Runtime events
    events: Arc<RwLock<Vec<RuntimeEvent>>>,
    /// System monitor
    system: Arc<RwLock<System>>,
    /// Process ID for monitoring
    pid: usize,
    /// Tracked background task handles for shutdown/cancellation
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Maximum GPU memory usage (MB)
    pub max_gpu_memory_mb: usize,
    /// CPU affinity settings
    pub cpu_affinity: Option<Vec<usize>>,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Metrics collection interval (seconds)
    pub metrics_interval_seconds: u64,
    /// Resource monitoring interval (seconds)
    pub resource_monitor_interval_seconds: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 8192,      // 8GB
            max_gpu_memory_mb: 16384, // 16GB
            cpu_affinity: None,
            thread_pool_size: num_cpus::get(),
            enable_profiling: false,
            metrics_interval_seconds: 10,
            resource_monitor_interval_seconds: 5,
        }
    }
}

/// Runtime state
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeState {
    /// Runtime tidak diinisialisasi
    Uninitialized,
    /// Runtime sedang diinisialisasi
    Initializing,
    /// Runtime siap
    Ready,
    /// Runtime sedang busy
    Busy,
    /// Runtime sedang shutdown
    ShuttingDown,
    /// Runtime error
    Error(String),
    /// Runtime sudah shutdown
    Shutdown,
}

/// Resource usage information
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// GPU memory usage (bytes) - jika available
    pub gpu_memory_usage_bytes: Option<u64>,
    /// GPU memory usage percentage - jika available
    pub gpu_memory_usage_percent: Option<f64>,
    /// Active threads count
    pub active_threads: usize,
    /// Open file descriptors count
    pub open_files: usize,
    /// Network I/O (bytes)
    pub network_io_bytes: u64,
    /// Disk I/O (bytes)
    pub disk_io_bytes: u64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average request latency (ms)
    pub avg_latency_ms: f64,
    /// P95 latency (ms)
    pub p95_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Throughput (requests/second)
    pub throughput_rps: f64,
    /// Tokens generated per second
    pub tokens_per_second: f64,
    /// Model load time (ms)
    pub model_load_time_ms: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate
    pub error_rate: f64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Runtime event
#[derive(Debug, Clone)]
pub struct RuntimeEvent {
    /// Event ID
    pub event_id: Uuid,
    /// Event type
    pub event_type: RuntimeEventType,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event message
    pub message: String,
    /// Event metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Runtime event types
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeEventType {
    /// Runtime initialized
    Initialized,
    /// Runtime started
    Started,
    /// Runtime stopped
    Stopped,
    /// Runtime error
    Error,
    /// Resource warning
    ResourceWarning,
    /// Resource critical
    ResourceCritical,
    /// Performance degradation
    PerformanceDegradation,
    /// Model loaded
    ModelLoaded,
    /// Model unloaded
    ModelUnloaded,
    /// Configuration changed
    ConfigurationChanged,
    /// Custom event
    Custom(String),
}

impl InferenceRuntime {
    /// Create new runtime
    pub fn new() -> Self {
        Self::with_config(RuntimeConfig::default())
    }

    /// Create runtime with configuration
    pub fn with_config(config: RuntimeConfig) -> Self {
        let pid = std::process::id() as usize;

        Self {
            config,
            state: Arc::new(RwLock::new(RuntimeState::Uninitialized)),
            resource_usage: Arc::new(RwLock::new(ResourceUsage::default())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            events: Arc::new(RwLock::new(Vec::new())),
            system: Arc::new(RwLock::new(System::new())),
            pid,
            background_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Initialize runtime
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing inference runtime");

        // Update state
        {
            let mut state = self.state.write().await;
            *state = RuntimeState::Initializing;
        }

        // Emit initialization event
        self.emit_event(
            RuntimeEventType::Initialized,
            "Runtime initialization started".to_string(),
        )
        .await;

        // Initialize resource monitoring
        self.initialize_resource_monitoring().await?;

        // Initialize performance tracking
        self.initialize_performance_tracking().await?;

        // Set CPU affinity if specified
        if let Some(ref affinity) = self.config.cpu_affinity {
            self.set_cpu_affinity(affinity).await?;
        }

        // Update state to ready
        {
            let mut state = self.state.write().await;
            *state = RuntimeState::Ready;
        }

        // Emit ready event
        self.emit_event(RuntimeEventType::Started, "Runtime ready".to_string())
            .await;

        info!("Inference runtime initialized successfully");
        Ok(())
    }

    /// Get current runtime state
    pub async fn get_state(&self) -> RuntimeState {
        self.state.read().await.clone()
    }

    /// Update runtime state
    pub async fn set_state(&self, new_state: RuntimeState) {
        let old_state = {
            let mut state = self.state.write().await;
            let old_state = state.clone();
            *state = new_state.clone();
            old_state
        };

        self.emit_event(
            RuntimeEventType::ConfigurationChanged,
            format!("State changed from {:?} to {:?}", old_state, new_state),
        )
        .await;
    }

    /// Get resource usage
    pub async fn get_resource_usage(&self) -> ResourceUsage {
        self.resource_usage.read().await.clone()
    }

    /// Update resource usage
    pub async fn update_resource_usage(&self) -> Result<()> {
        // Collect all async data first without holding any locks
        let cpu_usage = self.get_cpu_usage().await?;
        let mem_bytes = self.get_memory_usage().await?;
        let gpu_result = self.get_gpu_memory_usage().await.ok();
        let active_threads = self.get_active_thread_count().await?;
        let open_files = self.get_open_file_count().await?;
        let network_io = self.get_network_io_bytes().await?;
        let disk_io = self.get_disk_io_bytes().await?;

        // Write collected data under a single lock scope
        let usage_snapshot = {
            let mut usage = self.resource_usage.write().await;
            usage.cpu_usage_percent = cpu_usage;
            usage.memory_usage_bytes = mem_bytes;
            usage.memory_usage_percent =
                (mem_bytes as f64 / (self.config.max_memory_mb as f64 * 1024.0 * 1024.0)) * 100.0;
            if let Some((gpu_bytes, gpu_percent)) = gpu_result {
                usage.gpu_memory_usage_bytes = Some(gpu_bytes);
                usage.gpu_memory_usage_percent = Some(gpu_percent);
            }
            usage.active_threads = active_threads;
            usage.open_files = open_files;
            usage.network_io_bytes = network_io;
            usage.disk_io_bytes = disk_io;
            usage.last_updated = Utc::now();
            usage.clone()
        };

        // Check for resource warnings (lock already dropped)
        self.check_resource_warnings(&usage_snapshot).await;

        Ok(())
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Update performance metrics
    pub async fn update_performance_metrics(
        &self,
        request_count: u64,
        success_count: u64,
        failed_count: u64,
        total_latency_ms: u64,
        tokens_generated: u64,
    ) -> Result<()> {
        let mut metrics = self.performance_metrics.write().await;

        // Update request counts
        metrics.total_requests += request_count;
        metrics.successful_requests += success_count;
        metrics.failed_requests += failed_count;

        // Update latency metrics
        if request_count > 0 {
            let avg_latency = total_latency_ms as f64 / request_count as f64;
            metrics.avg_latency_ms = (metrics.avg_latency_ms
                * (metrics.total_requests - request_count) as f64
                + avg_latency * request_count as f64)
                / metrics.total_requests as f64;
        }

        // Update throughput
        let now = Utc::now();
        let duration = match now
            .signed_duration_since(metrics.last_updated)
            .to_std()
        {
            Ok(d) => Some(d),
            Err(e) => {
                warn!("Failed to compute duration since last metrics update: {}", e);
                None
            }
        };
        if let Some(duration) = duration {
            let duration_seconds = duration.as_secs_f64();
            if duration_seconds > 0.0 {
                metrics.throughput_rps = request_count as f64 / duration_seconds;
                metrics.tokens_per_second = tokens_generated as f64 / duration_seconds;
            }
        }

        // Update error rate
        if metrics.total_requests > 0 {
            metrics.error_rate = metrics.failed_requests as f64 / metrics.total_requests as f64;
        }

        metrics.last_updated = now;

        // Check for performance issues
        self.check_performance_issues(&metrics).await;

        Ok(())
    }

    /// Get runtime events
    pub async fn get_events(&self, limit: Option<usize>) -> Vec<RuntimeEvent> {
        let events = self.events.read().await;
        match limit {
            Some(limit) => events.iter().rev().take(limit).cloned().collect(),
            None => events.clone(),
        }
    }

    /// Clear old events
    pub async fn clear_old_events(&self, max_age_hours: u64) -> Result<usize> {
        let cutoff_time = Utc::now() - chrono::Duration::hours(max_age_hours as i64);
        let mut events = self.events.write().await;
        let initial_count = events.len();

        events.retain(|event| event.timestamp > cutoff_time);

        let removed_count = initial_count - events.len();
        debug!("Cleared {} old runtime events", removed_count);

        Ok(removed_count)
    }

    /// Shutdown runtime
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference runtime");

        // Update state
        {
            let mut state = self.state.write().await;
            *state = RuntimeState::ShuttingDown;
        }

        // Emit shutdown event
        self.emit_event(
            RuntimeEventType::Stopped,
            "Runtime shutdown initiated".to_string(),
        )
        .await;

        // Final resource update
        if let Err(e) = self.update_resource_usage().await {
            warn!("Final resource update failed during shutdown: {}", e);
        }

        // Cancel tracked background tasks
        if let Ok(mut tasks) = self.background_tasks.lock() {
            for h in tasks.drain(..) {
                h.abort();
            }
        }

        // Update state to shutdown
        {
            let mut state = self.state.write().await;
            *state = RuntimeState::Shutdown;
        }

        info!("Inference runtime shutdown complete");
        Ok(())
    }

    /// Initialize resource monitoring
    async fn initialize_resource_monitoring(&self) -> Result<()> {
        let runtime = self.clone();
        let tasks = Arc::clone(&self.background_tasks);
        let handle = tokio::spawn(async move {
            let fut = std::panic::AssertUnwindSafe(async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
                    runtime.config.resource_monitor_interval_seconds,
                ));

                loop {
                    interval.tick().await;

                    let state = runtime.get_state().await;
                    match state {
                        RuntimeState::Ready | RuntimeState::Busy => {
                            if let Err(e) = runtime.update_resource_usage().await {
                                warn!("Failed to update resource usage: {}", e);
                            }
                        }
                        RuntimeState::ShuttingDown | RuntimeState::Shutdown => break,
                        _ => continue,
                    }
                }
            });
            if let Err(e) = futures::future::FutureExt::catch_unwind(fut).await {
                error!("Resource monitoring loop panicked: {:?}", e);
            }
        });
        if let Ok(mut tasks) = tasks.lock() {
            tasks.push(handle);
        }

        Ok(())
    }

    /// Initialize performance tracking
    async fn initialize_performance_tracking(&self) -> Result<()> {
        let runtime = self.clone();
        let tasks = Arc::clone(&self.background_tasks);
        let handle = tokio::spawn(async move {
            let fut = std::panic::AssertUnwindSafe(async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
                    runtime.config.metrics_interval_seconds,
                ));

                let mut prev_total_requests: u64 = 0;
                let mut prev_tokens: u64 = 0;
                loop {
                    interval.tick().await;

                    let state = runtime.get_state().await;
                    match state {
                        RuntimeState::Ready | RuntimeState::Busy => {
                            let metrics = runtime.performance_metrics.read().await;
                            let total = metrics.total_requests;
                            let tokens = metrics.tokens_per_second as u64;
                            let new_requests = total.saturating_sub(prev_total_requests);
                            let new_tokens = tokens.saturating_sub(prev_tokens);
                            drop(metrics);

                            if new_requests > 0 || new_tokens > 0 {
                                debug!(
                                    "Perf tracking: +{} requests, +{} tokens in interval",
                                    new_requests, new_tokens
                                );
                            }

                            // Aggregate interval metrics and reset counters
                            if let Err(e) = runtime.aggregate_and_reset_counters(new_requests, new_tokens).await {
                                warn!("Failed to aggregate and reset counters: {}", e);
                            }

                            prev_total_requests = total;
                            prev_tokens = tokens;
                        }
                        RuntimeState::ShuttingDown | RuntimeState::Shutdown => break,
                        _ => continue,
                    }
                }
            });
            if let Err(e) = futures::future::FutureExt::catch_unwind(fut).await {
                error!("Performance tracking loop panicked: {:?}", e);
            }
        });
        if let Ok(mut tasks) = tasks.lock() {
            tasks.push(handle);
        }

        Ok(())
    }

    /// Aggregate interval metrics and reset counters for accurate long-term tracking
    async fn aggregate_and_reset_counters(&self, interval_requests: u64, interval_tokens: u64) -> Result<()> {
        let mut metrics = self.performance_metrics.write().await;

        // Store interval-based metrics (could be stored in a separate history)
        let interval_throughput = if interval_requests > 0 {
            interval_requests as f64 / self.config.metrics_interval_seconds as f64
        } else {
            0.0
        };

        let interval_tokens_per_sec = if interval_tokens > 0 {
            interval_tokens as f64 / self.config.metrics_interval_seconds as f64
        } else {
            0.0
        };

        debug!(
            "Interval metrics: {} req/s, {} tokens/s",
            interval_throughput, interval_tokens_per_sec
        );

        // Reset cumulative counters to prevent overflow and maintain accuracy
        // Note: We keep total_requests for overall stats but reset interval-based calculations
        // In a production system, you might want to store these in a time-series database

        // For now, we just log the interval metrics. The cumulative counters
        // continue to grow for overall statistics, but throughput is calculated
        // from interval deltas (as implemented above)

        Ok(())
    }

    /// Emit runtime event
    async fn emit_event(&self, event_type: RuntimeEventType, message: String) {
        let event = RuntimeEvent {
            event_id: Uuid::new_v4(),
            event_type,
            timestamp: Utc::now(),
            message,
            metadata: HashMap::new(),
        };

        let mut events = self.events.write().await;
        events.push(event);

        // Keep only last 1000 events
        if events.len() > 1000 {
            events.remove(0);
        }
    }

    /// Check resource warnings
    async fn check_resource_warnings(&self, usage: &ResourceUsage) {
        // Check memory usage
        if usage.memory_usage_percent > 80.0 {
            self.emit_event(
                RuntimeEventType::ResourceWarning,
                format!("High memory usage: {:.1}%", usage.memory_usage_percent),
            )
            .await;
        }

        if usage.memory_usage_percent > 95.0 {
            self.emit_event(
                RuntimeEventType::ResourceCritical,
                format!("Critical memory usage: {:.1}%", usage.memory_usage_percent),
            )
            .await;
        }

        // Check GPU memory if available
        if let Some(gpu_percent) = usage.gpu_memory_usage_percent {
            if gpu_percent > 80.0 {
                self.emit_event(
                    RuntimeEventType::ResourceWarning,
                    format!("High GPU memory usage: {:.1}%", gpu_percent),
                )
                .await;
            }

            if gpu_percent > 95.0 {
                self.emit_event(
                    RuntimeEventType::ResourceCritical,
                    format!("Critical GPU memory usage: {:.1}%", gpu_percent),
                )
                .await;
            }
        }

        // Check CPU usage
        if usage.cpu_usage_percent > 90.0 {
            self.emit_event(
                RuntimeEventType::ResourceWarning,
                format!("High CPU usage: {:.1}%", usage.cpu_usage_percent),
            )
            .await;
        }
    }

    /// Check performance issues
    async fn check_performance_issues(&self, metrics: &PerformanceMetrics) {
        // Check error rate
        if metrics.error_rate > 0.1 {
            // 10% error rate
            self.emit_event(
                RuntimeEventType::PerformanceDegradation,
                format!("High error rate: {:.2}%", metrics.error_rate * 100.0),
            )
            .await;
        }

        // Check latency
        if metrics.avg_latency_ms > 5000.0 {
            // 5 seconds
            self.emit_event(
                RuntimeEventType::PerformanceDegradation,
                format!("High average latency: {:.1}ms", metrics.avg_latency_ms),
            )
            .await;
        }

        // Check throughput
        if metrics.throughput_rps < 1.0 && metrics.total_requests > 100 {
            self.emit_event(
                RuntimeEventType::PerformanceDegradation,
                format!("Low throughput: {:.2} RPS", metrics.throughput_rps),
            )
            .await;
        }
    }

    // Resource monitoring helper methods
    async fn get_cpu_usage(&self) -> Result<f64> {
        let system_clone = Arc::clone(&self.system);
        tokio::task::spawn_blocking(move || -> Result<f64> {
            let mut system = system_clone.blocking_write();
            system.refresh_cpu();
            let total_cpu_usage: f64 = system.cpus().iter().map(|cpu| cpu.cpu_usage() as f64).sum();
            if system.cpus().is_empty() {
                return Ok(0.0);
            }
            Ok(total_cpu_usage / system.cpus().len() as f64)
        })
        .await
        .map_err(|e| InferenceError::InternalError(format!("CPU usage read failed: {}", e)))?
    }

    async fn get_memory_usage(&self) -> Result<u64> {
        let system_clone = Arc::clone(&self.system);
        let pid = self.pid;
        tokio::task::spawn_blocking(move || -> Result<u64> {
            let mut system = system_clone.blocking_write();
            system.refresh_memory();
            if let Some(process) = system.process(pid.into()) {
                Ok(process.memory())
            } else {
                Ok(system.used_memory())
            }
        })
        .await
        .map_err(|e| InferenceError::InternalError(format!("Memory usage read failed: {}", e)))?
    }

    async fn get_gpu_memory_usage(&self) -> Result<(u64, f64)> {
        tokio::task::spawn_blocking(|| {
            match nvml_wrapper::Nvml::init() {
                Ok(nvml) => match nvml.device_count() {
                    Ok(count) if count > 0 => {
                        if let Ok(device) = nvml.device_by_index(0) {
                            if let Ok(memory_info) = device.memory_info() {
                                let total_memory = memory_info.total;
                                let used_memory = memory_info.used;
                                let usage_percent =
                                    (used_memory as f64 / total_memory as f64) * 100.0;
                                return Ok((used_memory, usage_percent));
                            }
                        }
                    }
                    _ => {
                        debug!("No NVIDIA GPU detected");
                    }
                },
                Err(e) => {
                    debug!("NVML initialization failed: {}", e);
                }
            }
            Ok((0, 0.0))
        })
        .await
        .map_err(|e| {
            InferenceError::InternalError(format!("GPU memory read failed: {}", e))
        })?
    }

    async fn get_active_thread_count(&self) -> Result<usize> {
        tokio::task::spawn_blocking(|| {
            if let Ok(process) = ProcProcess::myself() {
                if let Ok(stat) = process.stat() {
                    return Ok(stat.num_threads as usize);
                }
            }
            Ok(num_cpus::get())
        })
        .await
        .map_err(|e| {
            InferenceError::InternalError(format!("Thread count read failed: {}", e))
        })?
    }

    async fn get_open_file_count(&self) -> Result<usize> {
        tokio::task::spawn_blocking(|| {
            match std::fs::read_dir("/proc/self/fd") {
                Ok(entries) => Ok(entries.count()),
                Err(_) => {
                    debug!("Cannot read /proc/self/fd, using fallback");
                    Ok(0)
                }
            }
        })
        .await
        .map_err(|e| {
            InferenceError::InternalError(format!("Open file count read failed: {}", e))
        })?
    }

    async fn get_network_io_bytes(&self) -> Result<u64> {
        // Try to get network I/O from /proc/net/dev
        let content = tokio::task::spawn_blocking(|| {
            std::fs::read_to_string("/proc/net/dev")
        }).await.map_err(|e| InferenceError::InternalError(format!("spawn_blocking: {}", e)))?;
        match content {
            Ok(content) => {
                let mut total_bytes = 0u64;
                for line in content.lines().skip(2) {
                    // Skip header lines
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 9 {
                        if let (Ok(recv_bytes), Ok(transmit_bytes)) =
                            (parts[1].parse::<u64>(), parts[9].parse::<u64>())
                        {
                            total_bytes += recv_bytes + transmit_bytes;
                        }
                    }
                }
                Ok(total_bytes)
            }
            Err(_) => {
                debug!("Cannot read /proc/net/dev, using fallback");
                Ok(0)
            }
        }
    }

    async fn get_disk_io_bytes(&self) -> Result<u64> {
        tokio::task::spawn_blocking(|| {
            match ProcProcess::myself() {
                Ok(process) => {
                    if let Ok(io) = process.io() {
                        Ok(io.read_bytes + io.write_bytes)
                    } else {
                        Ok(0)
                    }
                }
                Err(_) => {
                    debug!("Cannot read disk I/O stats");
                    Ok(0)
                }
            }
        })
        .await
        .map_err(|e| {
            InferenceError::InternalError(format!("Disk I/O read failed: {}", e))
        })?
    }

    async fn set_cpu_affinity(&self, affinity: &[usize]) -> Result<()> {
        // CPU affinity setting requires platform-specific implementation
        // For now, we'll log the request and return success
        info!("Setting CPU affinity to cores: {:?}", affinity);

        #[cfg(target_os = "linux")]
        {
            // SAFETY: `cpu_set_t` is a C struct that can be safely zero-initialized.
            // `MaybeUninit::zeroed().assume_init()` is valid because `cpu_set_t` is
            // a plain-old-data type with no invalid bit patterns.
            let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
            for &cpu in affinity {
                if cpu < 64 {
                    // SAFETY: `CPU_SET` is a libc macro that only touches the given
                    // `cpu_set_t` and `cpu < 64` ensures we stay within bounds.
                    unsafe {
                        libc::CPU_SET(cpu, &mut cpu_set);
                    }
                }
            }

            // SAFETY: `sched_setaffinity` is the standard Linux syscall for setting
            // CPU affinity. `cpu_set` is fully initialized via `CPU_SET` calls above.
            // Passing 0 as pid targets the calling thread. The kernel validates the
            // CPU mask size and returns -1 on error, which we handle below.
            let result = unsafe {
                libc::sched_setaffinity(
                    0,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &cpu_set as *const libc::cpu_set_t,
                )
            };

            if result == 0 {
                info!("CPU affinity set successfully");
                Ok(())
            } else {
                warn!("Failed to set CPU affinity: {}", result);
                Err(InferenceError::InternalError(
                    "Failed to set CPU affinity".to_string(),
                ))
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            warn!("CPU affinity not supported on this platform");
            Ok(())
        }
    }
}

impl Clone for InferenceRuntime {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            resource_usage: Arc::clone(&self.resource_usage),
            performance_metrics: Arc::clone(&self.performance_metrics),
            events: Arc::clone(&self.events),
            system: Arc::clone(&self.system),
            pid: self.pid,
            background_tasks: Arc::clone(&self.background_tasks),
        }
    }
}
