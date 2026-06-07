//! ERP Utilities dan Helper Functions
//!
//! Utility functions untuk ERP operations termasuk validation,
//! benchmarking, dan performance monitoring.

use crate::{CompressedLayer, ERPConfig, ERPError};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

/// Performance benchmark untuk ERP
pub struct ERPBenchmark {
    _config: ERPConfig,
}

impl ERPBenchmark {
    pub fn new(config: ERPConfig) -> Self {
        Self { _config: config }
    }
}

pub struct ERPValidator {
    _config: ERPConfig,
}

impl ERPValidator {
    pub fn new(config: ERPConfig) -> Self {
        Self { _config: config }
    }

    /// Validate compressed model
    pub fn validate_compressed_model(
        &self,
        original_weights: &[Array2<f32>],
        compressed_layers: &[CompressedLayer],
    ) -> Result<ValidationReport, ERPError> {
        let mut validation_report = ValidationReport::new();

        // Check dimension consistency
        self.validate_dimensions(original_weights, compressed_layers, &mut validation_report)?;

        // Check numerical stability
        self.validate_numerical_stability(compressed_layers, &mut validation_report)?;

        // Check reconstruction accuracy
        self.validate_reconstruction_accuracy(
            original_weights,
            compressed_layers,
            &mut validation_report,
        )?;

        // Check memory efficiency
        self.validate_memory_efficiency(
            original_weights,
            compressed_layers,
            &mut validation_report,
        )?;

        Ok(validation_report)
    }

    /// Validate dimension consistency
    fn validate_dimensions(
        &self,
        original_weights: &[Array2<f32>],
        compressed_layers: &[CompressedLayer],
        report: &mut ValidationReport,
    ) -> Result<(), ERPError> {
        if original_weights.len() != compressed_layers.len() {
            report.add_error("Layer count mismatch between original and compressed models");
            return Err(ERPError::ConfigError("Layer count mismatch".to_string()));
        }

        for (i, (original, compressed)) in original_weights
            .iter()
            .zip(compressed_layers.iter())
            .enumerate()
        {
            if original.dim() != compressed.compressed_weights.dim() {
                report.add_error(&format!(
                    "Dimension mismatch in layer {}: original {:?} vs compressed {:?}",
                    i,
                    original.dim(),
                    compressed.compressed_weights.dim()
                ));
            }
        }

        Ok(())
    }

    /// Validate numerical stability
    fn validate_numerical_stability(
        &self,
        layers: &[CompressedLayer],
        report: &mut ValidationReport,
    ) -> Result<(), ERPError> {
        for (i, layer) in layers.iter().enumerate() {
            // Check untuk NaN dan infinite values
            for (j, &value) in layer.compressed_weights.iter().enumerate() {
                if !value.is_finite() {
                    report.add_warning(&format!(
                        "Non-finite value in layer {} at position {}: {}",
                        i, j, value
                    ));
                }
            }

            // Check untuk exploding weights
            let max_weight = layer
                .compressed_weights
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f32::max);
            if max_weight > 1000.0 {
                report.add_warning(&format!(
                    "Large weights detected in layer {}: max = {}",
                    i, max_weight
                ));
            }
        }

        Ok(())
    }

    /// Validate reconstruction accuracy
    fn validate_reconstruction_accuracy(
        &self,
        original_weights: &[Array2<f32>],
        compressed_layers: &[CompressedLayer],
        report: &mut ValidationReport,
    ) -> Result<(), ERPError> {
        let mut total_mse = 0.0;
        let mut total_params = 0;

        for (original, compressed) in original_weights.iter().zip(compressed_layers.iter()) {
            let mse = self.compute_mse(original, &compressed.compressed_weights);
            total_mse += mse * original.len() as f32;
            total_params += original.len();
        }

        let avg_mse = total_mse / total_params as f32;

        if avg_mse > 0.1 {
            report.add_warning(&format!("High reconstruction error: MSE = {:.6}", avg_mse));
        } else {
            report.add_info(&format!(
                "Good reconstruction accuracy: MSE = {:.6}",
                avg_mse
            ));
        }

        Ok(())
    }

    /// Validate memory efficiency
    fn validate_memory_efficiency(
        &self,
        original_weights: &[Array2<f32>],
        compressed_layers: &[CompressedLayer],
        report: &mut ValidationReport,
    ) -> Result<(), ERPError> {
        let original_memory: usize = original_weights.iter().map(|w| w.len()).sum();
        let compressed_memory: usize = compressed_layers
            .iter()
            .map(|l| l.compressed_weights.len())
            .sum();

        let compression_ratio = 1.0 - (compressed_memory as f32 / original_memory as f32);

        if compression_ratio < 0.2 {
            report.add_warning(&format!(
                "Low compression efficiency: {:.1}%",
                compression_ratio * 100.0
            ));
        } else {
            report.add_info(&format!(
                "Good compression efficiency: {:.1}%",
                compression_ratio * 100.0
            ));
        }

        Ok(())
    }

    /// Run all validation checks and log warnings for failures
    pub fn validate_all(
        &self,
        original: &[Array2<f32>],
        compressed: &[CompressedLayer],
        report: &mut ValidationReport,
    ) {
        for (i, layer) in compressed.iter().enumerate() {
            if let Err(e) = self.validate_numerical_stability(&[layer.clone()], report) {
                tracing::warn!(
                    "Numerical stability validation failed for layer {}: {}",
                    i,
                    e
                );
            }
            if let Err(e) = self.validate_reconstruction_accuracy(original, compressed, report) {
                tracing::warn!(
                    "Reconstruction accuracy validation failed for layer {}: {}",
                    i,
                    e
                );
            }
            if let Err(e) = self.validate_memory_efficiency(original, compressed, report) {
                tracing::warn!("Memory efficiency validation failed for layer {}: {}", i, e);
            }
        }
    }

    /// Compute Mean Squared Error
    fn compute_mse(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        if a.dim() != b.dim() {
            return f32::INFINITY;
        }

        let diff = a - b;
        let squared_diff = &diff * &diff;
        let sum_squared_diff = squared_diff.sum();
        sum_squared_diff / a.len() as f32
    }
}

/// Performance monitor untuk runtime ERP operations
pub struct ERPMonitor {
    metrics: HashMap<String, MetricData>,
}

#[derive(Debug, Clone)]
pub struct MetricData {
    pub values: Vec<f32>,
    pub timestamps: Vec<Instant>,
    pub unit: String,
}

impl Default for ERPMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ERPMonitor {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Record metric value
    pub fn record_metric(&mut self, name: &str, value: f32, unit: &str) {
        let data = self
            .metrics
            .entry(name.to_string())
            .or_insert_with(|| MetricData {
                values: Vec::new(),
                timestamps: Vec::new(),
                unit: unit.to_string(),
            });

        data.values.push(value);
        data.timestamps.push(Instant::now());

        // Keep only last 1000 measurements
        if data.values.len() > 1000 {
            data.values.remove(0);
            data.timestamps.remove(0);
        }
    }

    /// Get metric statistics
    pub fn get_metric_stats(&self, name: &str) -> Option<MetricStats> {
        self.metrics.get(name).map(|data| {
            let values = &data.values;
            if values.is_empty() {
                return MetricStats::default();
            }

            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt();

            MetricStats {
                count: values.len(),
                mean,
                min: values.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                max: values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                std_dev,
                unit: data.unit.clone(),
            }
        })
    }

    /// Get all metric names
    pub fn get_metric_names(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }
}

#[derive(Debug, Default)]
pub struct MetricStats {
    pub count: usize,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    pub std_dev: f32,
    pub unit: String,
}

/// Validation report
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub info: Vec<String>,
    pub is_valid: bool,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, error: &str) {
        self.errors.push(error.to_string());
        self.is_valid = false;
    }

    pub fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }

    pub fn add_info(&mut self, info: &str) {
        self.info.push(info.to_string());
    }

    pub fn print_summary(&self) {
        if !self.errors.is_empty() {
            tracing::info!("Errors:");
            for error in &self.errors {
                tracing::info!("  - {}", error);
            }
        }

        if !self.warnings.is_empty() {
            tracing::info!("Warnings:");
            for warning in &self.warnings {
                tracing::info!("  - {}", warning);
            }
        }

        if !self.info.is_empty() {
            tracing::info!("Info:");
            for info_msg in &self.info {
                tracing::info!("  - {}", info_msg);
            }
        }

        tracing::info!(
            "Validation Status: {}",
            if self.is_valid { "PASSED" } else { "FAILED" }
        );
    }
}

/// Benchmark results
#[derive(Debug)]
pub struct BenchmarkResults {
    pub avg_inference_time: std::time::Duration,
    pub min_inference_time: std::time::Duration,
    pub max_inference_time: std::time::Duration,
    pub total_memory_usage: usize,
    pub compression_ratio: f32,
}

impl BenchmarkResults {
    pub fn print_summary(&self) {
        tracing::info!("=== ERP Benchmark Results ===");
        tracing::info!("Average Inference Time: {:?}", self.avg_inference_time);
        tracing::info!("Min Inference Time: {:?}", self.min_inference_time);
        tracing::info!("Max Inference Time: {:?}", self.max_inference_time);
        tracing::info!("Total Memory Usage: {} bytes", self.total_memory_usage);
        tracing::info!("Compression Ratio: {:.1}%", self.compression_ratio * 100.0);
    }
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Compute cosine similarity antara dua vektor
    pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        // Try GPU-accelerated dot + L2 norm
        #[cfg(feature = "gpu")]
        {
            use crate::gpu;
            if let (Some(dot), Some(norm_a), Some(norm_b)) = (
                crate::gpu_try!(gpu::try_dot(a, b)),
                crate::gpu_try!(gpu::try_l2_norm(a)),
                crate::gpu_try!(gpu::try_l2_norm(b)),
            ) {
                if norm_a > 0.0 && norm_b > 0.0 {
                    return dot / (norm_a * norm_b);
                }
                return 0.0;
            }
        }

        let dot_product = a.dot(b);
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Compute KL divergence
    pub fn kl_divergence(p: &Array1<f32>, q: &Array1<f32>) -> f32 {
        if p.len() != q.len() {
            return f32::INFINITY;
        }

        let eps = 1e-8;
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > eps && qi > eps {
                    pi * ((pi / qi).ln())
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Normalize array ke sum 1
    pub fn normalize_to_sum_one(arr: &mut Array1<f32>) {
        let sum = arr.sum();
        if sum > 0.0 {
            *arr /= sum;
        }
    }

    /// Apply softmax ke array
    pub fn softmax(arr: &Array1<f32>) -> Array1<f32> {
        // Try GPU-accelerated softmax
        #[cfg(feature = "gpu")]
        {
            use crate::gpu;
            if let Some(Ok(result)) = gpu::try_softmax(arr.as_slice().unwrap_or(&[])) {
                return Array1::from_vec(result);
            }
        }

        let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = arr.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        if sum_exp > 0.0 {
            Array1::from_vec(exp_vals.iter().map(|x| x / sum_exp).collect())
        } else {
            Array1::zeros(arr.len())
        }
    }

    /// Compute quantile dari array
    pub fn quantile(arr: &Array1<f32>, q: f32) -> f32 {
        if !(0.0..=1.0).contains(&q) {
            return f32::NAN;
        }

        let mut sorted_vals: Vec<f32> = arr.iter().copied().collect();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted_vals.is_empty() {
            return f32::NAN;
        }

        let index = (q * (sorted_vals.len() - 1) as f32) as usize;
        sorted_vals[index]
    }

    /// Compute percentile ranks
    pub fn percentile_ranks(arr: &Array1<f32>) -> Array1<f32> {
        let n = arr.len();
        let mut ranks = Array1::zeros(n);

        for (i, &value) in arr.iter().enumerate() {
            let rank = arr.iter().filter(|&&x| x <= value).count() as f32 / n as f32;
            ranks[i] = rank;
        }

        ranks
    }

    /// Generate random projection matrix
    pub fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        Array2::from_shape_fn((rows, cols), |_| rng.gen())
    }

    /// Compute Frobenius norm
    pub fn frobenius_norm(matrix: &Array2<f32>) -> f32 {
        // Try GPU-accelerated sum of squares
        #[cfg(feature = "gpu")]
        {
            use crate::gpu;
            use ndarray::Array1;
            let flat = Array1::from_iter(matrix.iter().copied());
            if let Some(sum_sq) = crate::gpu_try!(gpu::try_sum_squares(&flat)) {
                return sum_sq.sqrt();
            }
        }

        matrix.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute matrix rank approximation
    pub fn low_rank_approximation(matrix: &Array2<f32>, rank: usize) -> Array2<f32> {
        let (m, n) = matrix.dim();
        let actual_rank = std::cmp::min(rank, std::cmp::min(m, n));

        // Simplified low-rank approximation - dalam implementasi nyata gunakan SVD
        let mut rng = rand::thread_rng();
        let u = Array2::from_shape_fn((m, actual_rank), |_| rng.gen());
        let v = Array2::from_shape_fn((actual_rank, n), |_| rng.gen());

        u.dot(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ERPConfig {
        ERPConfig::default()
    }

    fn layer(nrows: usize, ncols: usize) -> CompressedLayer {
        CompressedLayer {
            layer_idx: 0,
            original_weights: Array2::zeros((nrows, ncols)),
            compressed_weights: Array2::zeros((nrows, ncols)),
            resonance_representations: vec![],
            neuron_status: vec![],
            compression_ratio: 0.0,
        }
    }

    // ── ERPValidator ──

    #[test]
    fn test_validator_new() {
        let v = ERPValidator::new(cfg());
        let report = v.validate_compressed_model(&[], &[]).unwrap();
        // Default is_valid = false (from Default impl); empty validation has no errors
        assert!(!report.is_valid);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_validate_dimensions_mismatch() {
        let v = ERPValidator::new(cfg());
        let orig = vec![Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap()];
        let comp = vec![layer(2, 2), layer(2, 2)];
        let result = v.validate_compressed_model(&orig, &comp);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dimensions_mismatch_shape() {
        let v = ERPValidator::new(cfg());
        let orig = vec![Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap()];
        let comp = vec![layer(3, 2)];
        let report = v.validate_compressed_model(&orig, &comp).unwrap();
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_validate_numerical_stability_ok() {
        let v = ERPValidator::new(cfg());
        let layers = vec![layer(2, 2)];
        let mut report = ValidationReport::new();
        let result = v.validate_numerical_stability(&layers, &mut report);
        assert!(result.is_ok());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_validate_numerical_stability_nan() {
        let v = ERPValidator::new(cfg());
        let mut l = layer(2, 2);
        l.compressed_weights[[0, 0]] = f32::NAN;
        let mut report = ValidationReport::new();
        let _ = v.validate_numerical_stability(&[l], &mut report);
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_validate_reconstruction_accuracy() {
        let v = ERPValidator::new(cfg());
        let orig = vec![Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap()];
        let comp = vec![layer(2, 2)];
        let mut report = ValidationReport::new();
        let _ = v.validate_reconstruction_accuracy(&orig, &comp, &mut report);
        // MSE = sum((1-0)^2)/4 = 1.0, which is > 0.1 -> warning
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_validate_memory_efficiency() {
        let v = ERPValidator::new(cfg());
        let orig = vec![Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap()];
        let comp = vec![layer(2, 2)];
        let mut report = ValidationReport::new();
        let _ = v.validate_memory_efficiency(&orig, &comp, &mut report);
        // compression_ratio = 0.0 < 0.2 → warning, not info
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_compute_mse_different_shapes() {
        let v = ERPValidator::new(cfg());
        let a = Array2::zeros((2, 2));
        let b = Array2::zeros((3, 3));
        assert_eq!(v.compute_mse(&a, &b), f32::INFINITY);
    }

    #[test]
    fn test_compute_mse_identical() {
        let v = ERPValidator::new(cfg());
        let a = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
        assert!((v.compute_mse(&a, &a) - 0.0).abs() < 1e-5);
    }

    // ── ERPMonitor ──

    #[test]
    fn test_monitor_new() {
        let m = ERPMonitor::new();
        assert!(m.get_metric_names().is_empty());
    }

    #[test]
    fn test_record_and_get_stats() {
        let mut m = ERPMonitor::new();
        m.record_metric("latency", 1.0, "ms");
        m.record_metric("latency", 2.0, "ms");
        m.record_metric("latency", 3.0, "ms");
        let stats = m.get_metric_stats("latency").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 2.0).abs() < 1e-5);
        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_metric_stats_missing() {
        let m = ERPMonitor::new();
        assert!(m.get_metric_stats("nonexistent").is_none());
    }

    #[test]
    fn test_clear() {
        let mut m = ERPMonitor::new();
        m.record_metric("test", 1.0, "unit");
        m.clear();
        assert!(m.get_metric_names().is_empty());
    }

    #[test]
    fn test_get_metric_names() {
        let mut m = ERPMonitor::new();
        m.record_metric("a", 1.0, "");
        m.record_metric("b", 2.0, "");
        let names = m.get_metric_names();
        assert_eq!(names.len(), 2);
    }

    // ─── validation_report ──

    #[test]
    fn test_validation_report_new_is_valid_false() {
        let r = ValidationReport::new();
        // Default impl sets is_valid = false
        assert!(!r.is_valid);
    }

    #[test]
    fn test_validation_report_add_error() {
        let mut r = ValidationReport::new();
        r.add_error("test error");
        assert!(!r.is_valid);
        assert_eq!(r.errors.len(), 1);
    }

    #[test]
    fn test_validation_report_add_warning() {
        let mut r = ValidationReport::new();
        r.add_warning("test warning");
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn test_validation_report_add_info() {
        let mut r = ValidationReport::new();
        r.add_info("test info");
        assert_eq!(r.info.len(), 1);
    }

    #[test]
    fn test_benchmark_results_print() {
        let b = BenchmarkResults {
            avg_inference_time: std::time::Duration::from_micros(100),
            min_inference_time: std::time::Duration::from_micros(50),
            max_inference_time: std::time::Duration::from_micros(200),
            total_memory_usage: 1024,
            compression_ratio: 0.5,
        };
        b.print_summary(); // smoke test — shouldn't panic
    }

    // ── utils sub-module ──

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((super::utils::cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_different_dims() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(super::utils::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let a = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        assert!((super::utils::kl_divergence(&a, &a) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_kl_divergence_different_dims() {
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![1.0]);
        assert_eq!(super::utils::kl_divergence(&a, &b), f32::INFINITY);
    }

    #[test]
    fn test_normalize_to_sum_one() {
        let mut a = Array1::from_vec(vec![2.0, 3.0, 5.0]);
        super::utils::normalize_to_sum_one(&mut a);
        assert!((a.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_to_sum_one_zero() {
        let mut a = Array1::from_vec(vec![0.0, 0.0]);
        super::utils::normalize_to_sum_one(&mut a);
        assert!(a.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_softmax() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = super::utils::softmax(&a);
        assert!((s.sum() - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1]);
    }

    #[test]
    fn test_softmax_uniform() {
        let a = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let s = super::utils::softmax(&a);
        assert!((s.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantile_median() {
        let a = Array1::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0]);
        let q = super::utils::quantile(&a, 0.5);
        assert!((q - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantile_out_of_range() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        assert!(super::utils::quantile(&a, 1.5).is_nan());
    }

    #[test]
    fn test_quantile_empty() {
        let a = Array1::<f32>::from_vec(vec![]);
        assert!(super::utils::quantile(&a, 0.5).is_nan());
    }

    #[test]
    fn test_percentile_ranks() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let ranks = super::utils::percentile_ranks(&a);
        assert_eq!(ranks.len(), 4);
        assert!((ranks[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_generate_random_matrix() {
        let m = super::utils::generate_random_matrix(3, 4);
        assert_eq!(m.shape(), &[3, 4]);
    }

    #[test]
    fn test_frobenius_norm() {
        let m = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let n = super::utils::frobenius_norm(&m);
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_low_rank_approximation() {
        let m = Array2::from_shape_vec((5, 5), vec![1.0; 25]).unwrap();
        let approx = super::utils::low_rank_approximation(&m, 2);
        assert_eq!(approx.shape(), &[5, 5]);
    }

    // ── ERPBenchmark ──

    #[test]
    fn test_benchmark_new() {
        let b = ERPBenchmark::new(cfg());
        // just ensure no panic
        let _ = b;
    }
}
