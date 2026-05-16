//! Performance monitoring utilities untuk Nexora

use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::RwLock;
use tracing::debug;
use serde::{Serialize, Deserialize};

#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics: Arc<RwLock<PerformanceMetrics>>,
    config: PerformanceConfig,
}

#[derive(Debug, Clone)]
pub struct CpuTimes {
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub total: u64,
}

#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub max_samples: usize,
    pub enable_detailed_tracking: bool,
    pub enable_memory_tracking: bool,
    pub enable_cpu_tracking: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_samples: 1000,
            enable_detailed_tracking: true,
            enable_memory_tracking: true,
            enable_cpu_tracking: true,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceMetrics {
    operation_times: HashMap<String, VecDeque<Duration>>,
    operation_counts: HashMap<String, u64>,
    error_counts: HashMap<String, u64>,
    memory_usage: VecDeque<MemorySnapshot>,
    cpu_usage: VecDeque<CpuSnapshot>,
    custom_metrics: HashMap<String, VecDeque<f64>>,
    start_time: Instant,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            operation_times: HashMap::new(),
            operation_counts: HashMap::new(),
            error_counts: HashMap::new(),
            memory_usage: VecDeque::new(),
            cpu_usage: VecDeque::new(),
            custom_metrics: HashMap::new(),
            start_time: Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: u64, // Changed from Instant to u64 for serializability
    pub memory_usage_mb: f64,
    pub memory_usage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSnapshot {
    pub timestamp: u64, // Changed from Instant to u64 for serializability
    pub cpu_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_operations: u64,
    pub average_operation_time_ms: u64, // Changed from Duration to u64 for serializability
    pub operation_breakdown: HashMap<String, OperationStats>,
    pub error_rate: f64,
    pub memory_trend: Vec<MemorySnapshot>,
    pub cpu_trend: Vec<CpuSnapshot>,
    pub uptime_ms: u64, // Changed from Duration to u64 for serializability
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub count: u64,
    pub average_time_ms: u64, // Changed from Duration to u64 for serializability
    pub min_time_ms: u64, // Changed from Duration to u64 for serializability
    pub max_time_ms: u64, // Changed from Duration to u64 for serializability
    pub error_count: u64,
    pub error_rate: f64,
}

impl PerformanceMonitor {
    pub fn new(cache_size: usize) -> Self {
        let config = PerformanceConfig {
            max_samples: cache_size,
            ..Default::default()
        };
        
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            config,
        }
    }
    
    pub fn with_config(config: PerformanceConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            config,
        }
    }
    
    /// Start timing an operation
    pub async fn start_timer(&self, operation: &str) -> OperationTimer {
        OperationTimer::new(operation.to_string(), self.metrics.clone(), self.config.enable_detailed_tracking)
    }
    
    /// Record operation completion
    pub async fn record_operation(&self, operation: &str, duration: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        
        // Record operation time
        let times = metrics.operation_times.entry(operation.to_string()).or_insert_with(VecDeque::new);
        times.push_back(duration);
        
        // Keep only recent samples
        while times.len() > self.config.max_samples {
            times.pop_front();
        }
        
        // Record operation count
        *metrics.operation_counts.entry(operation.to_string()).or_insert(0) += 1;
        
        // Record error if applicable
        if !success {
            *metrics.error_counts.entry(operation.to_string()).or_insert(0) += 1;
        }
    }
    
    /// Record custom metric
    pub async fn record_metric(&self, metric_name: &str, value: f64) {
        let mut metrics = self.metrics.write().await;
        let values = metrics.custom_metrics.entry(metric_name.to_string()).or_insert_with(VecDeque::new);
        values.push_back(value);
        
        // Keep only recent samples
        while values.len() > self.config.max_samples {
            values.pop_front();
        }
    }
    
    /// Record memory usage
    pub async fn record_memory_usage(&self, memory_mb: f64) {
        if !self.config.enable_memory_tracking {
            return;
        }
        
        let mut metrics = self.metrics.write().await;
        let snapshot = MemorySnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
            memory_usage_mb : memory_mb,
            memory_usage_bytes: (memory_mb * 1024.0 * 1024.0) as u64,
        };
        
        metrics.memory_usage.push_back(snapshot);
        
        // Keep only recent samples
        while metrics.memory_usage.len() > self.config.max_samples {
            metrics.memory_usage.pop_front();
        }
    }
    
    /// Record CPU usage
    pub async fn record_cpu_usage(&self, cpu_percent: f64) {
        if !self.config.enable_cpu_tracking {
            return;
        }
        
        let mut metrics = self.metrics.write().await;
        let snapshot = CpuSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
            cpu_usage_percent: cpu_percent,
        };
        
        metrics.cpu_usage.push_back(snapshot);
        
        // Keep only recent samples
        while metrics.cpu_usage.len() > self.config.max_samples {
            metrics.cpu_usage.pop_front();
        }
    }
    
    /// Get current memory usage in MB
    pub async fn get_memory_usage() -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            use std::io::Read;
            
            // Read memory info from /proc/self/status
            let mut status_file = fs::File::open("/proc/self/status")
                .map_err(|e| anyhow::anyhow!("Failed to open /proc/self/status: {}", e))?;
            
            let mut contents = String::new();
            status_file.read_to_string(&mut contents)
                .map_err(|e| anyhow::anyhow!("Failed to read /proc/self/status: {}", e))?;
            
            // Parse VmRSS (Resident Set Size) from status
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb_usage: f64 = parts[1].parse()
                            .map_err(|e| anyhow::anyhow!("Failed to parse memory usage: {}", e))?;
                        return Ok(kb_usage / 1024.0); // Convert KB to MB
                    }
                }
            }
            
            Err(anyhow::anyhow!("VmRSS not found in /proc/self/status"))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems - use a simple estimation
            // In production, you'd use platform-specific APIs like:
            // - Windows: GetProcessMemoryInfo
            // - macOS: task_info
            Ok(64.0) // Default 64MB fallback
        }
    }
    
    /// Get current CPU usage as percentage (0-100)
    pub async fn get_cpu_usage() -> Result<f64> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            use std::io::Read;
            use std::time::{Duration, Instant};
            
            let _start_time = Instant::now();
            
            // Read initial CPU stats
            let mut stat_file = fs::File::open("/proc/stat")
                .map_err(|e| anyhow::anyhow!("Failed to open /proc/stat: {}", e))?;
            
            let mut stat_contents = String::new();
            stat_file.read_to_string(&mut stat_contents)
                .map_err(|e| anyhow::anyhow!("Failed to read /proc/stat: {}", e))?;
            
            let initial_cpu_times = Self::parse_cpu_times(&stat_contents)?;
            
            // Wait a short duration
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            // Read final CPU stats
            let mut stat_file = fs::File::open("/proc/stat")
                .map_err(|e| anyhow::anyhow!("Failed to open /proc/stat: {}", e))?;
            
            let mut stat_contents = String::new();
            stat_file.read_to_string(&mut stat_contents)
                .map_err(|e| anyhow::anyhow!("Failed to read /proc/stat: {}", e))?;
            
            let final_cpu_times = Self::parse_cpu_times(&stat_contents)?;
            
            // Calculate CPU usage percentage
            let total_diff = final_cpu_times.total - initial_cpu_times.total;
            let idle_diff = final_cpu_times.idle - initial_cpu_times.idle;
            
            if total_diff > 0 {
                let usage = ((total_diff - idle_diff) as f64 / total_diff as f64) * 100.0;
                Ok(usage)
            } else {
                Ok(0.0)
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems
            // In production, you'd use platform-specific APIs like:
            // - Windows: GetProcessTimes
            // - macOS: host_processor_info
            Ok(5.0) // Default 5% fallback
        }
    }
    
    #[cfg(target_os = "linux")]
    fn parse_cpu_times(stat_contents: &str) -> Result<CpuTimes> {
        let first_line = stat_contents.lines().next()
            .ok_or_else(|| anyhow::anyhow!("No CPU stats found"))?;
        
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 5 || !parts[0].starts_with("cpu") {
            return Err(anyhow::anyhow!("Invalid CPU stats format"));
        }
        
        let user: u64 = parts[1].parse()
            .map_err(|e| anyhow::anyhow!("Failed to parse user time: {}", e))?;
        let nice: u64 = parts[2].parse()
            .map_err(|e| anyhow::anyhow!("Failed to parse nice time: {}", e))?;
        let system: u64 = parts[3].parse()
            .map_err(|e| anyhow::anyhow!("Failed to parse system time: {}", e))?;
        let idle: u64 = parts[4].parse()
            .map_err(|e| anyhow::anyhow!("Failed to parse idle time: {}", e))?;
        
        let total = user + nice + system + idle;
        
        Ok(CpuTimes { user, nice, system, idle, total })
    }
    
    /// Generate performance report
    pub async fn generate_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().await;
        let uptime = metrics.start_time.elapsed();
        
        // Calculate total operations
        let total_operations: u64 = metrics.operation_counts.values().sum();
        
        // Calculate average operation time
        let total_time: Duration = metrics.operation_times.values()
            .flatten()
            .sum();
        let average_operation_time = if total_operations > 0 {
            total_time / total_operations as u32
        } else {
            Duration::ZERO
        };
        
        // Calculate operation breakdown
        let mut operation_breakdown = HashMap::new();
        for (operation, times) in &metrics.operation_times {
            let count = *metrics.operation_counts.get(operation).unwrap_or(&0);
            let error_count = *metrics.error_counts.get(operation).unwrap_or(&0);
            
            if !times.is_empty() {
                let total_time: Duration = times.iter().sum();
                let avg_time = total_time / times.len() as u32;
                let min_time = *times.iter().min().expect("times is non-empty");
                let max_time = *times.iter().max().expect("times is non-empty");
                let error_rate = if count > 0 {
                    error_count as f64 / count as f64
                } else {
                    0.0
                };
                
                operation_breakdown.insert(operation.clone(), OperationStats {
                    count,
                    average_time_ms: avg_time.as_millis() as u64,
                    min_time_ms: min_time.as_millis() as u64,
                    max_time_ms: max_time.as_millis() as u64,
                    error_count,
                    error_rate,
                });
            }
        }
        
        // Calculate overall error rate
        let total_errors: u64 = metrics.error_counts.values().sum();
        let error_rate = if total_operations > 0 {
            total_errors as f64 / total_operations as f64
        } else {
            0.0
        };
        
        PerformanceReport {
            total_operations,
            average_operation_time_ms: average_operation_time.as_millis() as u64,
            operation_breakdown,
            error_rate,
            memory_trend: metrics.memory_usage.iter().cloned().collect(),
            cpu_trend: metrics.cpu_usage.iter().cloned().collect(),
            uptime_ms: uptime.as_millis() as u64,
        }
    }
    
    /// Get operation statistics
    pub async fn get_operation_stats(&self, operation: &str) -> Option<OperationStats> {
        let metrics = self.metrics.read().await;
        
        let times = metrics.operation_times.get(operation)?;
        let count = *metrics.operation_counts.get(operation).unwrap_or(&0);
        let error_count = *metrics.error_counts.get(operation).unwrap_or(&0);
        
        if times.is_empty() {
            return None;
        }
        
        let total_time: Duration = times.iter().sum();
        let avg_time = total_time / times.len() as u32;
        let min_time = *times.iter().min().expect("times is non-empty");
        let max_time = *times.iter().max().expect("times is non-empty");
        let error_rate = if count > 0 {
            error_count as f64 / count as f64
        } else {
            0.0
        };
        
        Some(OperationStats {
            count,
            average_time_ms: avg_time.as_millis() as u64,
            min_time_ms: min_time.as_millis() as u64,
            max_time_ms: max_time.as_millis() as u64,
            error_count,
            error_rate,
        })
    }
    
    /// Get custom metric values
    pub async fn get_metric_values(&self, metric_name: &str) -> Option<Vec<f64>> {
        let metrics = self.metrics.read().await;
        metrics.custom_metrics.get(metric_name).map(|values| values.iter().copied().collect())
    }
    
    /// Get all operation names
    pub async fn get_operation_names(&self) -> Vec<String> {
        let metrics = self.metrics.read().await;
        metrics.operation_counts.keys().cloned().collect()
    }
    
    /// Get all custom metric names
    pub async fn get_metric_names(&self) -> Vec<String> {
        let metrics = self.metrics.read().await;
        metrics.custom_metrics.keys().cloned().collect()
    }
    
    /// Clear all metrics
    pub async fn clear_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = PerformanceMetrics::default();
    }
    
    /// Clear metrics for specific operation
    pub async fn clear_operation_metrics(&self, operation: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.operation_times.remove(operation);
        metrics.operation_counts.remove(operation);
        metrics.error_counts.remove(operation);
    }
    
    /// Clear custom metric
    pub async fn clear_custom_metric(&self, metric_name: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.custom_metrics.remove(metric_name);
    }
    
    /// Export metrics to JSON
    pub async fn export_metrics(&self) -> Result<String> {
        let report = self.generate_report().await;
        serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow::anyhow!("Failed to export metrics: {}", e))
    }
    
    /// Check if performance is healthy
    pub async fn is_healthy(&self) -> bool {
        let report = self.generate_report().await;
        
        // Simple health check logic
        if report.error_rate > 0.1 {
            return false; // Error rate too high
        }
        
        if report.average_operation_time_ms > 10000 {
            return false; // Operations too slow (more than 10 seconds)
        }
        
        true
    }
    
    /// Get performance score (0-100)
    pub async fn get_performance_score(&self) -> f64 {
        let report = self.generate_report().await;
        
        let mut score = 100.0;
        
        // Penalize high error rate
        score -= report.error_rate * 100.0;
        
        // Penalize slow operations
        let avg_seconds = report.average_operation_time_ms as f64 / 1000.0;
        if avg_seconds > 1.0 {
            score -= (avg_seconds - 1.0) * 10.0;
        }
        
        score.max(0.0).min(100.0)
    }
}

/// Operation timer for measuring execution time
pub struct OperationTimer {
    operation: String,
    start_time: Instant,
    metrics: Arc<RwLock<PerformanceMetrics>>,
    enable_detailed_tracking: bool,
}

impl OperationTimer {
    fn new(operation: String, metrics: Arc<RwLock<PerformanceMetrics>>, enable_detailed_tracking: bool) -> Self {
        Self {
            operation,
            start_time: Instant::now(),
            metrics,
            enable_detailed_tracking,
        }
    }
    
    /// Finish timing and record the operation
    pub async fn finish(self, success: bool) -> Duration {
        let duration = self.start_time.elapsed();
        
        let mut metrics = self.metrics.write().await;
        
        // Record operation time
        let times = metrics.operation_times.entry(self.operation.clone()).or_insert_with(VecDeque::new);
        times.push_back(duration);
        
        // Record operation count
        *metrics.operation_counts.entry(self.operation.clone()).or_insert(0) += 1;
        
        // Record error if applicable
        if !success {
            *metrics.error_counts.entry(self.operation.clone()).or_insert(0) += 1;
        }
        
        if self.enable_detailed_tracking {
            debug!("Operation '{}' completed in {:?} (success: {})", self.operation, duration, success);
        }
        
        duration
    }
    
    /// Finish timing with success
    pub async fn finish_success(self) -> Duration {
        self.finish(true).await
    }
    
    /// Finish timing with failure
    pub async fn finish_failure(self) -> Duration {
        self.finish(false).await
    }
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        // Auto-finish with success if not explicitly finished
        let duration = self.start_time.elapsed();
        
        if self.enable_detailed_tracking {
            debug!("Operation '{}' auto-finished in {:?} (assumed success)", self.operation, duration);
        }
        
        // Note: This is a simplified approach. In production, you might want to handle this differently
        // since we can't easily use async in Drop.
    }
}

/// Performance benchmarking utilities
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Benchmark a function
    pub async fn benchmark<F, Fut, T>(name: &str, iterations: usize, function: F) -> BenchmarkResult
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = T>,
    {
        let mut durations = Vec::with_capacity(iterations);
        let mut results = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            let result = function().await;
            let duration = start.elapsed();
            
            durations.push(duration);
            results.push(result);
        }
        
        let total_time: Duration = durations.iter().sum();
        let average_time = total_time / iterations as u32;
        let min_time = *durations.iter().min().expect("durations is non-empty");
        let max_time = *durations.iter().max().expect("durations is non-empty");
        
        // Calculate standard deviation
        let mean = average_time.as_secs_f64();
        let variance: f64 = durations.iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>() / iterations as f64;
        let std_dev = variance.sqrt();
        
        BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_time,
            average_time,
            min_time,
            max_time,
            standard_deviation: Duration::from_secs_f64(std_dev),
            throughput: iterations as f64 / total_time.as_secs_f64(),
        }
    }
    
    /// Compare two functions
    pub async fn compare<F1, Fut1, T1, F2, Fut2, T2>(
        name1: &str,
        name2: &str,
        iterations: usize,
        function1: F1,
        function2: F2,
    ) -> ComparisonResult
    where
        F1: Fn() -> Fut1 + Clone,
        Fut1: std::future::Future<Output = T1>,
        F2: Fn() -> Fut2 + Clone,
        Fut2: std::future::Future<Output = T2>,
    {
        let result1 = Self::benchmark(name1, iterations, function1).await;
        let result2 = Self::benchmark(name2, iterations, function2).await;
        
        let speedup = result1.average_time.as_secs_f64() / result2.average_time.as_secs_f64();
        
        ComparisonResult {
            result1,
            result2,
            speedup,
            winner: if speedup > 1.0 { name2.to_string() } else { name1.to_string() },
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub standard_deviation: Duration,
    pub throughput: f64,
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub result1: BenchmarkResult,
    pub result2: BenchmarkResult,
    pub speedup: f64,
    pub winner: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new(100);
        
        // Test operation timing
        let timer = monitor.start_timer("test_operation").await;
        sleep(Duration::from_millis(10)).await;
        timer.finish_success().await;
        
        // Test custom metrics
        monitor.record_metric("test_metric", 42.0).await;
        
        // Test operation stats
        let stats = monitor.get_operation_stats("test_operation").await;
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().count, 1);
        
        // Test metric values
        let values = monitor.get_metric_values("test_metric").await;
        assert!(values.is_some());
        assert_eq!(values.unwrap()[0], 42.0);
        
        // Test report generation
        let report = monitor.generate_report().await;
        assert_eq!(report.total_operations, 1);
        assert!(report.operation_breakdown.contains_key("test_operation"));
        
        // Test health check
        assert!(monitor.is_healthy().await);
        
        // Test performance score
        let score = monitor.get_performance_score().await;
        assert!(score > 0.0 && score <= 100.0);
    }
    
    #[tokio::test]
    async fn test_benchmark() {
        async fn test_function() -> String {
            sleep(Duration::from_millis(1)).await;
            "result".to_string()
        }
        
        let result = BenchmarkUtils::benchmark("test", 5, test_function).await;
        assert_eq!(result.name, "test");
        assert_eq!(result.iterations, 5);
        assert!(result.average_time > Duration::ZERO);
        assert!(result.throughput > 0.0);
    }
    
    #[tokio::test]
    async fn test_comparison() {
        async fn fast_function() -> String {
            sleep(Duration::from_millis(1)).await;
            "fast".to_string()
        }
        
        async fn slow_function() -> String {
            sleep(Duration::from_millis(2)).await;
            "slow".to_string()
        }
        
        let comparison = BenchmarkUtils::compare("fast", "slow", 3, fast_function, slow_function).await;
        assert_eq!(comparison.result1.name, "fast");
        assert_eq!(comparison.result2.name, "slow");
        assert!(comparison.speedup > 1.0);
        assert_eq!(comparison.winner, "fast");
    }
}
