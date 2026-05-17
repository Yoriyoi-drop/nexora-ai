//! ERP Utilities dan Helper Functions
//! 
//! Utility functions untuk ERP operations termasuk validation,
//! benchmarking, dan performance monitoring.

use crate::erp::{ERPConfig, ERPError, CompressedLayer};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::{Rng, SeedableRng};
use rand_distr::Standard;
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
    pub fn validate_compressed_model(&self, original_weights: &[Array2<f32>], compressed_layers: &[CompressedLayer]) -> Result<ValidationReport, ERPError> {
        let mut validation_report = ValidationReport::new();

        // Check dimension consistency
        self.validate_dimensions(original_weights, compressed_layers, &mut validation_report)?;
        
        // Check numerical stability
        self.validate_numerical_stability(compressed_layers, &mut validation_report)?;
        
        // Check reconstruction accuracy
        self.validate_reconstruction_accuracy(original_weights, compressed_layers, &mut validation_report)?;
        
        // Check memory efficiency
        self.validate_memory_efficiency(original_weights, compressed_layers, &mut validation_report)?;

        Ok(validation_report)
    }

    /// Validate dimension consistency
    fn validate_dimensions(&self, original_weights: &[Array2<f32>], compressed_layers: &[CompressedLayer], report: &mut ValidationReport) -> Result<(), ERPError> {
        if original_weights.len() != compressed_layers.len() {
            report.add_error("Layer count mismatch between original and compressed models");
            return Err(ERPError::ConfigError("Layer count mismatch".to_string()));
        }

        for (i, (original, compressed)) in original_weights.iter().zip(compressed_layers.iter()).enumerate() {
            if original.dim() != compressed.compressed_weights.dim() {
                report.add_error(&format!("Dimension mismatch in layer {}: original {:?} vs compressed {:?}", 
                    i, original.dim(), compressed.compressed_weights.dim()));
            }
        }

        Ok(())
    }

    /// Validate numerical stability
    fn validate_numerical_stability(&self, layers: &[CompressedLayer], report: &mut ValidationReport) -> Result<(), ERPError> {
        for (i, layer) in layers.iter().enumerate() {
            // Check untuk NaN dan infinite values
            for (j, &value) in layer.compressed_weights.iter().enumerate() {
                if !value.is_finite() {
                    report.add_warning(&format!("Non-finite value in layer {} at position {}: {}", i, j, value));
                }
            }

            // Check untuk exploding weights
            let max_weight = layer.compressed_weights.iter().map(|&x| x.abs()).fold(0.0, f32::max);
            if max_weight > 1000.0 {
                report.add_warning(&format!("Large weights detected in layer {}: max = {}", i, max_weight));
            }
        }

        Ok(())
    }

    /// Validate reconstruction accuracy
    fn validate_reconstruction_accuracy(&self, original_weights: &[Array2<f32>], compressed_layers: &[CompressedLayer], report: &mut ValidationReport) -> Result<(), ERPError> {
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
            report.add_info(&format!("Good reconstruction accuracy: MSE = {:.6}", avg_mse));
        }

        Ok(())
    }

    /// Validate memory efficiency
    fn validate_memory_efficiency(&self, original_weights: &[Array2<f32>], compressed_layers: &[CompressedLayer], report: &mut ValidationReport) -> Result<(), ERPError> {
        let original_memory: usize = original_weights.iter().map(|w| w.len()).sum();
        let compressed_memory: usize = compressed_layers.iter().map(|l| l.compressed_weights.len()).sum();
        
        let compression_ratio = 1.0 - (compressed_memory as f32 / original_memory as f32);
        
        if compression_ratio < 0.2 {
            report.add_warning(&format!("Low compression efficiency: {:.1}%", compression_ratio * 100.0));
        } else {
            report.add_info(&format!("Good compression efficiency: {:.1}%", compression_ratio * 100.0));
        }

        Ok(())
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

impl ERPMonitor {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Record metric value
    pub fn record_metric(&mut self, name: &str, value: f32, unit: &str) {
        let data = self.metrics.entry(name.to_string()).or_insert_with(|| MetricData {
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
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
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
            println!("Errors:");
            for error in &self.errors {
                println!("  - {}", error);
            }
        }

        if !self.warnings.is_empty() {
            println!("Warnings:");
            for warning in &self.warnings {
                println!("  - {}", warning);
            }
        }

        if !self.info.is_empty() {
            println!("Info:");
            for info_msg in &self.info {
                println!("  - {}", info_msg);
            }
        }

        println!("Validation Status: {}", if self.is_valid { "PASSED" } else { "FAILED" });
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
        println!("=== ERP Benchmark Results ===");
        println!("Average Inference Time: {:?}", self.avg_inference_time);
        println!("Min Inference Time: {:?}", self.min_inference_time);
        println!("Max Inference Time: {:?}", self.max_inference_time);
        println!("Total Memory Usage: {} bytes", self.total_memory_usage);
        println!("Compression Ratio: {:.1}%", self.compression_ratio * 100.0);
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
        p.iter().zip(q.iter())
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
        if q < 0.0 || q > 1.0 {
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
