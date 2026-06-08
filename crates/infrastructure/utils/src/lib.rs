//! Nexora Utils - Utility functions
//!
//! Module ini menyediakan fungsi-fungsi utility untuk Nexora AI system

pub mod crypto;
pub mod file_utils;
pub mod math;
pub mod performance;
pub mod simd_ops;
pub mod string;
pub mod text_processing;
pub mod time;
pub mod validation;

pub use crypto::CryptoUtils;
pub use file_utils::FileUtils;
pub use math::MathUtils;
pub use performance::PerformanceMonitor;
pub use string::StringUtils;
pub use text_processing::TextProcessor;
pub use time::TimeUtils;
pub use validation::ValidationUtils;

use anyhow::Result;
use std::sync::Arc;
use tracing::debug;

/// Central utility manager untuk Nexora AI
#[derive(Debug)]
pub struct UtilsManager {
    performance_monitor: Arc<PerformanceMonitor>,
    #[allow(dead_code)]
    config: UtilsConfig,
}

#[derive(Debug, Clone)]
pub struct UtilsConfig {
    pub enable_performance_monitoring: bool,
    pub enable_logging: bool,
    pub max_log_entries: usize,
    pub cache_size: usize,
}

impl Default for UtilsConfig {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            enable_logging: true,
            max_log_entries: 10000,
            cache_size: 1000,
        }
    }
}

impl UtilsManager {
    pub fn new(config: UtilsConfig) -> Self {
        let performance_monitor = Arc::new(PerformanceMonitor::new(config.cache_size));

        Self {
            performance_monitor,
            config,
        }
    }

    pub fn default() -> Self {
        Self::new(UtilsConfig::default())
    }

    /// Get performance monitor
    pub fn performance_monitor(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    /// Validate input data
    pub fn validate_input(&self, input: &str) -> Result<()> {
        ValidationUtils::validate_text(input)
    }

    /// Process text dengan preprocessing
    pub fn preprocess_text(&self, text: &str) -> Result<String> {
        TextProcessor::preprocess(text)
    }

    /// Generate secure hash
    pub fn generate_hash(&self, data: &[u8]) -> Result<String> {
        CryptoUtils::hash(data)
    }

    /// Format timestamp
    pub fn format_timestamp(&self, timestamp: u64) -> Result<String> {
        TimeUtils::format_timestamp(timestamp)
    }

    /// Calculate similarity score
    pub fn calculate_similarity(&self, text1: &str, text2: &str) -> Result<f64> {
        StringUtils::calculate_similarity(text1, text2)
    }

    /// Compute dot product between two vectors (SIMD-accelerated)
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        simd_ops::SimdVectorOps::dot_product(a, b)
    }

    /// Compute cosine similarity between two vectors (SIMD-accelerated)
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        simd_ops::SimdVectorOps::cosine_similarity(a, b)
    }

    /// Compute string similarity (SIMD-accelerated for large inputs)
    pub fn string_similarity(&self, a: &str, b: &str) -> f32 {
        simd_ops::SimdTextOps::string_similarity(a, b)
    }

    /// Get system info
    pub async fn get_system_info(&self) -> Result<SystemInfo> {
        let memory_usage = crate::performance::PerformanceMonitor::get_memory_usage()
            .await
            .unwrap_or_else(|e| {
                debug!("Failed to get memory usage: {}", e);
                0.0
            });
        let cpu_usage = crate::performance::PerformanceMonitor::get_cpu_usage()
            .await
            .unwrap_or_else(|e| {
                debug!("Failed to get CPU usage: {}", e);
                0.0
            });

        Ok(SystemInfo {
            timestamp: TimeUtils::current_timestamp(),
            memory_usage,
            cpu_usage,
        })
    }
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub timestamp: u64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
}

impl Default for UtilsManager {
    fn default() -> Self {
        Self::new(UtilsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utils_manager() {
        let manager = UtilsManager::default();

        // Test validation
        let result = manager.validate_input("Hello world");
        assert!(result.is_ok());

        // Test text processing
        let processed = manager.preprocess_text("Hello, world!").unwrap();
        assert_eq!(processed, "hello world");

        // Test hash generation
        let hash = manager.generate_hash(b"test").unwrap();
        assert!(!hash.is_empty());

        // Test similarity
        let similarity = manager.calculate_similarity("hello", "hello").unwrap();
        assert_eq!(similarity, 1.0);
    }
}
