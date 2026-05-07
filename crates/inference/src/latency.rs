//! Latency Tracking
//! 
//! Latency measurement dan analysis untuk inference.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::{debug, info};
use chrono::{DateTime, Utc};

use crate::Result;

/// Latency tracker untuk inference
pub struct LatencyTracker {
    /// Configuration
    config: LatencyConfig,
    /// Latency measurements
    measurements: Arc<RwLock<VecDeque<LatencyMeasurement>>>,
    /// Aggregated statistics
    stats: Arc<RwLock<LatencyStats>>,
    /// Percentile calculations
    percentiles: Arc<RwLock<PercentileStats>>,
}

/// Configuration untuk latency tracking
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    /// Maximum measurements to keep
    pub max_measurements: usize,
    /// Percentiles to calculate
    pub percentiles: Vec<f32>,
    /// Enable automatic cleanup
    pub enable_cleanup: bool,
    /// Cleanup interval (measurements)
    pub cleanup_interval: usize,
    /// Enable outlier detection
    pub enable_outlier_detection: bool,
    /// Outlier threshold (standard deviations)
    pub outlier_threshold: f32,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            max_measurements: 10000,
            percentiles: vec![50.0, 90.0, 95.0, 99.0],
            enable_cleanup: true,
            cleanup_interval: 1000,
            enable_outlier_detection: true,
            outlier_threshold: 2.0,
        }
    }
}

/// Individual latency measurement
#[derive(Debug, Clone)]
pub struct LatencyMeasurement {
    /// Measurement ID
    pub id: Uuid,
    /// Request ID
    pub request_id: Uuid,
    /// Total latency (ms)
    pub total_latency_ms: u64,
    /// Queue time (ms)
    pub queue_time_ms: u64,
    /// Processing time (ms)
    pub processing_time_ms: u64,
    /// Token generation time (ms)
    pub token_generation_time_ms: u64,
    /// Post-processing time (ms)
    pub post_processing_time_ms: u64,
    /// Token count
    pub token_count: usize,
    /// Model ID
    pub model_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LatencyMeasurement {
    /// Create new measurement
    pub fn new(request_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            request_id,
            total_latency_ms: 0,
            queue_time_ms: 0,
            processing_time_ms: 0,
            token_generation_time_ms: 0,
            post_processing_time_ms: 0,
            token_count: 0,
            model_id: "default".to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        if self.token_generation_time_ms > 0 {
            (self.token_count as f64 * 1000.0) / self.token_generation_time_ms as f64
        } else {
            0.0
        }
    }
    
    /// Check if this is an outlier
    pub fn is_outlier(&self, avg_latency: f64, std_dev: f64, threshold: f32) -> bool {
        let z_score = ((self.total_latency_ms as f64 - avg_latency) / std_dev).abs();
        z_score > threshold as f64
    }
}

/// Aggregated latency statistics
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    /// Total measurements
    pub total_measurements: u64,
    /// Average total latency (ms)
    pub avg_total_latency_ms: f64,
    /// Average queue time (ms)
    pub avg_queue_time_ms: f64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Average token generation time (ms)
    pub avg_token_generation_time_ms: f64,
    /// Average post-processing time (ms)
    pub avg_post_processing_time_ms: f64,
    /// Minimum latency (ms)
    pub min_latency_ms: u64,
    /// Maximum latency (ms)
    pub max_latency_ms: u64,
    /// Standard deviation (ms)
    pub std_deviation_ms: f64,
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Outlier count
    pub outlier_count: u64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Percentile statistics
#[derive(Debug, Clone, Default)]
pub struct PercentileStats {
    /// Percentile values (ms) - keys are percentile * 1000 to avoid f32 HashMap issues
    pub percentiles: HashMap<u32, f64>,
    /// Last calculated timestamp
    pub last_calculated: DateTime<Utc>,
}

impl LatencyTracker {
    /// Create new latency tracker
    pub fn new() -> Self {
        Self::with_config(LatencyConfig::default())
    }
    
    /// Create latency tracker with configuration
    pub fn with_config(config: LatencyConfig) -> Self {
        Self {
            config,
            measurements: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(LatencyStats::default())),
            percentiles: Arc::new(RwLock::new(PercentileStats::default())),
        }
    }
    
    /// Add latency measurement
    pub async fn add_measurement(&self, measurement: LatencyMeasurement) -> Result<()> {
        debug!("Adding latency measurement for request: {}", measurement.request_id);
        
        // Add to measurements
        {
            let mut measurements = self.measurements.write().await;
            measurements.push_back(measurement.clone());
            
            // Maintain size limit
            if measurements.len() > self.config.max_measurements {
                measurements.pop_front();
            }
            
            // Trigger cleanup if needed
            if self.config.enable_cleanup && 
               measurements.len() % self.config.cleanup_interval == 0 {
                drop(measurements);
                self.cleanup_old_measurements().await?;
            }
        }
        
        // Update statistics
        self.update_statistics().await?;
        
        Ok(())
    }
    
    /// Get current statistics
    pub async fn get_stats(&self) -> LatencyStats {
        self.stats.read().await.clone()
    }
    
    /// Get percentile statistics
    pub async fn get_percentiles(&self) -> PercentileStats {
        self.percentiles.read().await.clone()
    }
    
    /// Get measurements for time range
    pub async fn get_measurements_in_range(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<LatencyMeasurement> {
        let measurements = self.measurements.read().await;
        measurements.iter()
            .filter(|m| m.timestamp >= start_time && m.timestamp <= end_time)
            .cloned()
            .collect()
    }
    
    /// Get recent measurements
    pub async fn get_recent_measurements(&self, count: usize) -> Vec<LatencyMeasurement> {
        let measurements = self.measurements.read().await;
        measurements.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    /// Calculate latency percentiles
    pub async fn calculate_percentiles(&self) -> Result<PercentileStats> {
        let measurements = self.measurements.read().await;
        
        if measurements.is_empty() {
            return Ok(PercentileStats::default());
        }
        
        // Extract latencies
        let mut latencies: Vec<f64> = measurements.iter()
            .map(|m| m.total_latency_ms as f64)
            .collect();
        
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut percentile_stats = HashMap::new();
        
        for &percentile in &self.config.percentiles {
            let index = ((percentile as f64 / 100.0) * latencies.len() as f64) as usize;
            let index = index.min(latencies.len().saturating_sub(1));
            // Convert f32 to integer key (multiply by 1000 to preserve 3 decimal places)
            let percentile_key = (percentile * 1000.0) as u32;
            percentile_stats.insert(percentile_key, latencies[index]);
        }
        
        let stats = PercentileStats {
            percentiles: percentile_stats,
            last_calculated: Utc::now(),
        };
        
        // Update cached percentiles
        {
            let mut cached_percentiles = self.percentiles.write().await;
            *cached_percentiles = stats.clone();
        }
        
        Ok(stats)
    }
    
    /// Detect outliers in recent measurements
    pub async fn detect_outliers(&self) -> Result<Vec<LatencyMeasurement>> {
        if !self.config.enable_outlier_detection {
            return Ok(Vec::new());
        }
        
        let measurements = self.measurements.read().await;
        let stats = self.stats.read().await;
        
        if measurements.len() < 10 {
            return Ok(Vec::new()); // Not enough data
        }
        
        let mut outliers = Vec::new();
        
        for measurement in measurements.iter() {
            if measurement.is_outlier(
                stats.avg_total_latency_ms,
                stats.std_deviation_ms,
                self.config.outlier_threshold,
            ) {
                outliers.push(measurement.clone());
            }
        }
        
        Ok(outliers)
    }
    
    /// Get latency breakdown by component
    pub async fn get_component_breakdown(&self) -> ComponentBreakdown {
        let measurements = self.measurements.read().await;
        
        if measurements.is_empty() {
            return ComponentBreakdown::default();
        }
        
        let _total_count = measurements.len() as f64;
        
        let breakdown = ComponentBreakdown {
            queue_percentage: (measurements.iter().map(|m| m.queue_time_ms as f64).sum::<f64>() / 
                              measurements.iter().map(|m| m.total_latency_ms as f64).sum::<f64>()) * 100.0,
            processing_percentage: (measurements.iter().map(|m| m.processing_time_ms as f64).sum::<f64>() / 
                                  measurements.iter().map(|m| m.total_latency_ms as f64).sum::<f64>()) * 100.0,
            token_generation_percentage: (measurements.iter().map(|m| m.token_generation_time_ms as f64).sum::<f64>() / 
                                       measurements.iter().map(|m| m.total_latency_ms as f64).sum::<f64>()) * 100.0,
            post_processing_percentage: (measurements.iter().map(|m| m.post_processing_time_ms as f64).sum::<f64>() / 
                                      measurements.iter().map(|m| m.total_latency_ms as f64).sum::<f64>()) * 100.0,
        };
        
        breakdown
    }
    
    /// Clear all measurements
    pub async fn clear(&self) -> Result<()> {
        info!("Clearing all latency measurements");
        
        {
            let mut measurements = self.measurements.write().await;
            measurements.clear();
        }
        
        {
            let mut stats = self.stats.write().await;
            *stats = LatencyStats::default();
        }
        
        {
            let mut percentiles = self.percentiles.write().await;
            *percentiles = PercentileStats::default();
        }
        
        Ok(())
    }
    
    /// Update aggregated statistics
    async fn update_statistics(&self) -> Result<()> {
        let measurements = self.measurements.read().await;
        
        if measurements.is_empty() {
            return Ok(());
        }
        
        let count = measurements.len();
        let total_latency: f64 = measurements.iter().map(|m| m.total_latency_ms as f64).sum();
        let queue_time: f64 = measurements.iter().map(|m| m.queue_time_ms as f64).sum();
        let processing_time: f64 = measurements.iter().map(|m| m.processing_time_ms as f64).sum();
        let token_gen_time: f64 = measurements.iter().map(|m| m.token_generation_time_ms as f64).sum();
        let post_proc_time: f64 = measurements.iter().map(|m| m.post_processing_time_ms as f64).sum();
        
        let avg_total = total_latency / count as f64;
        let avg_queue = queue_time / count as f64;
        let avg_processing = processing_time / count as f64;
        let avg_token_gen = token_gen_time / count as f64;
        let avg_post_proc = post_proc_time / count as f64;
        
        // Calculate min/max
        let min_latency = measurements.iter().map(|m| m.total_latency_ms).min().unwrap_or(0);
        let max_latency = measurements.iter().map(|m| m.total_latency_ms).max().unwrap_or(0);
        
        // Calculate standard deviation
        let variance: f64 = measurements.iter()
            .map(|m| (m.total_latency_ms as f64 - avg_total).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        // Calculate average tokens per second
        let avg_tokens_per_sec: f64 = measurements.iter()
            .map(|m| m.tokens_per_second())
            .sum::<f64>() / count as f64;
        
        // Count outliers
        let outlier_count = measurements.iter()
            .filter(|m| m.is_outlier(avg_total, std_dev, self.config.outlier_threshold))
            .count() as u64;
        
        let stats = LatencyStats {
            total_measurements: count as u64,
            avg_total_latency_ms: avg_total,
            avg_queue_time_ms: avg_queue,
            avg_processing_time_ms: avg_processing,
            avg_token_generation_time_ms: avg_token_gen,
            avg_post_processing_time_ms: avg_post_proc,
            min_latency_ms: min_latency,
            max_latency_ms: max_latency,
            std_deviation_ms: std_dev,
            avg_tokens_per_second: avg_tokens_per_sec,
            outlier_count,
            last_updated: Utc::now(),
        };
        
        {
            let mut cached_stats = self.stats.write().await;
            *cached_stats = stats;
        }
        
        Ok(())
    }
    
    /// Clean up old measurements
    async fn cleanup_old_measurements(&self) -> Result<()> {
        let mut measurements = self.measurements.write().await;
        let initial_count = measurements.len();
        
        // Remove measurements older than 24 hours
        let cutoff_time = Utc::now() - chrono::Duration::hours(24);
        measurements.retain(|m| m.timestamp > cutoff_time);
        
        let removed_count = initial_count - measurements.len();
        if removed_count > 0 {
            debug!("Cleaned up {} old latency measurements", removed_count);
        }
        
        Ok(())
    }
}

/// Component breakdown statistics
#[derive(Debug, Clone, Default)]
pub struct ComponentBreakdown {
    pub queue_percentage: f64,
    pub processing_percentage: f64,
    pub token_generation_percentage: f64,
    pub post_processing_percentage: f64,
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for latency analysis
pub mod analysis {
    use super::*;
    
    /// Calculate moving average
    pub fn moving_average(measurements: &[LatencyMeasurement], window_size: usize) -> Vec<f64> {
        if measurements.is_empty() || window_size == 0 {
            return Vec::new();
        }
        
        let mut averages = Vec::new();
        
        for i in window_size..=measurements.len() {
            let window = &measurements[i - window_size..i];
            let avg: f64 = window.iter().map(|m| m.total_latency_ms as f64).sum::<f64>() / window.len() as f64;
            averages.push(avg);
        }
        
        averages
    }
    
    /// Detect latency trends
    pub fn detect_trend(measurements: &[LatencyMeasurement]) -> TrendDirection {
        if measurements.len() < 2 {
            return TrendDirection::Stable;
        }
        
        let first_half = &measurements[..measurements.len() / 2];
        let second_half = &measurements[measurements.len() / 2..];
        
        let first_avg: f64 = first_half.iter().map(|m| m.total_latency_ms as f64).sum::<f64>() / first_half.len() as f64;
        let second_avg: f64 = second_half.iter().map(|m| m.total_latency_ms as f64).sum::<f64>() / second_half.len() as f64;
        
        let change_percent = ((second_avg - first_avg) / first_avg) * 100.0;
        
        match change_percent {
            x if x > 5.0 => TrendDirection::Increasing,
            x if x < -5.0 => TrendDirection::Decreasing,
            _ => TrendDirection::Stable,
        }
    }
    
    /// Calculate latency distribution
    pub fn calculate_distribution(measurements: &[LatencyMeasurement], bins: usize) -> Vec<(f64, f64)> {
        if measurements.is_empty() || bins == 0 {
            return Vec::new();
        }
        
        let latencies: Vec<f64> = measurements.iter().map(|m| m.total_latency_ms as f64).collect();
        let min_latency = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_latency = latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let bin_width = (max_latency - min_latency) / bins as f64;
        
        let mut distribution = Vec::new();
        
        for i in 0..bins {
            let bin_start = min_latency + (i as f64 * bin_width);
            let bin_end = bin_start + bin_width;
            
            let count = latencies.iter()
                .filter(|&&latency| latency >= bin_start && latency < bin_end)
                .count();
            
            let frequency = count as f64 / latencies.len() as f64;
            distribution.push((bin_start, frequency));
        }
        
        distribution
    }
    
    /// Compare latency between two periods
    pub fn compare_periods(
        period1: &[LatencyMeasurement],
        period2: &[LatencyMeasurement],
    ) -> LatencyComparison {
        let avg1: f64 = period1.iter().map(|m| m.total_latency_ms as f64).sum::<f64>() / period1.len() as f64;
        let avg2: f64 = period2.iter().map(|m| m.total_latency_ms as f64).sum::<f64>() / period2.len() as f64;
        
        let change_percent = ((avg2 - avg1) / avg1) * 100.0;
        
        LatencyComparison {
            period1_avg: avg1,
            period2_avg: avg2,
            change_percent,
            improvement: change_percent < 0.0,
        }
    }
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Latency comparison result
#[derive(Debug, Clone)]
pub struct LatencyComparison {
    pub period1_avg: f64,
    pub period2_avg: f64,
    pub change_percent: f64,
    pub improvement: bool,
}
