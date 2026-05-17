//! Recursive Holographic Compression (RHC)
//!
//! Block 4 dari ECHO-Net Ω
//!
//! Inti scalability dengan hierarchical memory compression:
//! - Level 0: token detail
//! - Level 1: frasa
//! - Level 2: paragraf  
//! - Level 3: konsep
//! - Level 4: abstraksi global
//!
//! Compression operator:
//! C(H) = FFT-Pool(H) + ResSummary(H)
//!
//! Fitur:
//! - Semantic zooming
//! - Hierarchical memory
//! - Abstraction reasoning

use crate::{DLResult, DeepLearningError};
use crate::echo_net::utils::{HolographicFFT, MemoryCompressor, Complex};
use ndarray::{Array2, Array1};

/// Compression level configuration
#[derive(Debug, Clone)]
pub struct CompressionLevel {
    pub level: usize,
    pub compression_ratio: f32,
    pub window_size: usize,
    pub description: String,
    pub target_features: usize,
}

impl CompressionLevel {
    pub fn new(level: usize, ratio: f32, window_size: usize, description: &str, target_features: usize) -> Self {
        Self {
            level,
            compression_ratio: ratio,
            window_size,
            description: description.to_string(),
            target_features,
        }
    }
}

/// Recursive Holographic Compression implementation
#[derive(Debug, Clone)]
pub struct RecursiveHolographicCompression {
    // Compression levels
    levels: Vec<CompressionLevel>,
    num_levels: usize,
    
    // Compressed memory hierarchy
    compressed_memory: Vec<Array2<f32>>,
    memory_shapes: Vec<(usize, usize)>,
    
    // Compression parameters
    min_compression_ratio: f32,
    max_compression_ratio: f32,
    adaptive_threshold: f32,
    
    // FFT pooling parameters
    pool_size: usize,
    frequency_cutoff: f32,
    
    // Summary statistics
    level_statistics: Vec<CompressionStatistics>,
    
    // Feature extraction
    feature_extractors: Vec<Array2<f32>>,
    
    // Compression history
    compression_history: Vec<CompressionEvent>,
    
    // Adaptive compression
    importance_scores: Vec<Vec<f32>>,
    compression_budget: f32,
}

impl RecursiveHolographicCompression {
    /// Create new Recursive Holographic Compression
    pub fn new(
        levels: Vec<CompressionLevel>,
        pool_size: usize,
        frequency_cutoff: f32,
        compression_budget: f32,
    ) -> DLResult<Self> {
        let num_levels = levels.len();
        
        // Initialize compressed memory for each level
        let mut compressed_memory = Vec::with_capacity(levels.len());
        let mut memory_shapes = Vec::with_capacity(levels.len());
        let mut feature_extractors = Vec::with_capacity(levels.len());
        
        for level in &levels {
            // Estimate memory size based on compression ratio
            let base_size = 1024; // Base memory size
            let compressed_size = (base_size as f32 * level.compression_ratio) as usize;
            let memory = Array2::zeros((compressed_size, level.target_features));
            
            compressed_memory.push(memory);
            memory_shapes.push((compressed_size, level.target_features));
            
            // Initialize feature extractor
            let extractor = Array2::zeros((base_size, level.target_features));
            feature_extractors.push(extractor);
        }
        
        // Initialize statistics
        let level_statistics = vec![CompressionStatistics::default(); num_levels];
        
        Ok(Self {
            levels,
            num_levels,
            compressed_memory,
            memory_shapes,
            min_compression_ratio: 0.1,
            max_compression_ratio: 0.9,
            adaptive_threshold: 0.5,
            pool_size,
            frequency_cutoff,
            level_statistics,
            feature_extractors,
            compression_history: Vec::new(),
            importance_scores: vec![Vec::new(); num_levels],
            compression_budget,
        })
    }
    
    /// Forward pass - compress input holographic data
    pub fn forward(&mut self, input: &Array2<f32>, timestamp: usize) -> DLResult<Vec<Array2<f32>>> {
        let mut compressed_levels = Vec::with_capacity(self.num_levels);
        let mut current_data = input.clone();
        
        // Compress through each level
        for level_idx in 0..self.num_levels {
            let level = &self.levels[level_idx];
            
            // Apply compression at this level
            let compressed = self.compress_level(&current_data, level_idx, timestamp)?;
            compressed_levels.push(compressed.clone());
            
            // Update memory for this level
            self.update_level_memory(level_idx, &compressed)?;
            
            // Prepare for next level
            current_data = compressed;
        }
        
        // Update statistics
        self.update_compression_statistics(input, &compressed_levels, timestamp)?;
        
        Ok(compressed_levels)
    }
    
    /// Compress data at specific level
    fn compress_level(&mut self, input: &Array2<f32>, level_idx: usize, timestamp: usize) -> DLResult<Array2<f32>> {
        let level = &self.levels[level_idx];
        
        // Apply FFT pooling
        let fft_pooled = self.apply_fft_pooling(input, level)?;
        
        // Generate summary statistics
        let summary = self.generate_summary_statistics(input, level)?;
        
        // Combine FFT pooled data with summary
        let combined = self.combine_compression_results(&fft_pooled, &summary, level)?;
        
        // Apply adaptive compression
        let adaptive_compressed = self.apply_adaptive_compression(&combined, level_idx)?;
        
        // Extract features
        let features = self.extract_features(&adaptive_compressed, level_idx)?;
        
        // Record compression event
        self.record_compression_event(level_idx, input.shape(), features.shape(), timestamp);
        
        Ok(features)
    }
    
    /// Apply FFT pooling compression
    fn apply_fft_pooling(&self, input: &Array2<f32>, level: &CompressionLevel) -> DLResult<Array2<f32>> {
        // Apply FFT to input
        let fft_result = HolographicFFT::fft_2d(input)?;
        
        // Pool low-frequency components
        let mut pooled = Array2::zeros((
            (fft_result.nrows() + self.pool_size - 1) / self.pool_size,
            (fft_result.ncols() + self.pool_size - 1) / self.pool_size,
        ));
        
        for i in 0..pooled.nrows() {
            for j in 0..pooled.ncols() {
                let mut sum = Complex::new(0.0, 0.0);
                let mut count = 0;
                
                for di in 0..self.pool_size {
                    for dj in 0..self.pool_size {
                        let row = i * self.pool_size + di;
                        let col = j * self.pool_size + dj;
                        
                        if row < fft_result.nrows() && col < fft_result.ncols() {
                            // Apply frequency cutoff
                            let freq = ((row * row + col * col) as f32).sqrt();
                            if freq <= self.frequency_cutoff {
                                sum = sum + fft_result[[row, col]];
                                count += 1;
                            }
                        }
                    }
                }
                
                if count > 0 {
                    let magnitude = sum.magnitude() / count as f32;
                    pooled[[i, j]] = magnitude;
                }
            }
        }
        
        Ok(pooled)
    }
    
    /// Generate summary statistics
    fn generate_summary_statistics(&self, input: &Array2<f32>, level: &CompressionLevel) -> DLResult<Array1<f32>> {
        // Use MemoryCompressor for summary generation
        let summary = MemoryCompressor::generate_summary(input);
        
        // Adapt summary size to level requirements
        let mut adapted_summary = Array1::zeros(level.target_features);
        
        for i in 0..level.target_features.min(summary.len()) {
            adapted_summary[i] = summary[i];
        }
        
        Ok(adapted_summary)
    }
    
    /// Combine FFT pooled data with summary
    fn combine_compression_results(&self, fft_pooled: &Array2<f32>, summary: &Array1<f32>, level: &CompressionLevel) -> DLResult<Array2<f32>> {
        let (rows, cols) = fft_pooled.dim();
        let mut combined = Array2::zeros((rows, cols + summary.len()));
        
        // Copy FFT pooled data
        for i in 0..rows {
            for j in 0..cols {
                combined[[i, j]] = fft_pooled[[i, j]];
            }
        }
        
        // Add summary as additional columns
        for i in 0..rows.min(summary.len()) {
            for j in 0..summary.len() {
                combined[[i, cols + j]] = summary[j];
            }
        }
        
        Ok(combined)
    }
    
    /// Apply adaptive compression based on importance scores
    fn apply_adaptive_compression(&mut self, input: &Array2<f32>, level_idx: usize) -> DLResult<Array2<f32>> {
        let level = &self.levels[level_idx];
        let compression_ratio = level.compression_ratio;
        
        // Calculate importance scores
        let importance = self.calculate_importance_scores(input, level_idx)?;
        let importance_vec = importance.to_vec();
        self.importance_scores[level_idx] = importance_vec;
        
        // Apply adaptive compression
        let target_size = ((input.nrows() as f32 * compression_ratio) as usize).max(1);
        let mut compressed = Array2::zeros((target_size, input.ncols()));
        
        // Sort by importance
        let mut indexed_importance: Vec<(usize, f32)> = importance.iter().enumerate().map(|(i, &score)| (i, score)).collect();
        indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Keep most important rows
        for i in 0..target_size.min(indexed_importance.len()) {
            let original_row = indexed_importance[i].0;
            for j in 0..input.ncols() {
                compressed[[i, j]] = input[[original_row, j]];
            }
        }
        
        Ok(compressed)
    }
    
    /// Calculate importance scores for compression
    fn calculate_importance_scores(&mut self, input: &Array2<f32>, level_idx: usize) -> DLResult<Array1<f32>> {
        let mut importance = Array1::zeros(input.nrows());
        
        for i in 0..input.nrows() {
            let row = input.row(i);
            
            // Calculate importance based on multiple factors
            let energy: f32 = row.iter().map(|&x| x * x).sum();
            let entropy = self.calculate_row_entropy(&row);
            let variance = self.calculate_row_variance(&row);
            
            // Combined importance score
            importance[i] = 0.4 * energy + 0.3 * entropy + 0.3 * variance;
        }
        
        // Normalize importance scores
        let max_importance = importance.iter().fold(0.0f32, |acc, &x| acc.max(x));
        if max_importance > 0.0 {
            importance.mapv_inplace(|x| x / max_importance);
        }
        
        Ok(importance)
    }
    
    /// Calculate entropy of a row
    fn calculate_row_entropy(&self, row: &ndarray::ArrayView1<f32>) -> f32 {
        let total: f32 = row.iter().map(|&x| x.abs()).sum();
        if total == 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &val in row.iter() {
            let abs_val = val.abs();
            if abs_val > 0.0 {
                let p = abs_val / total;
                entropy -= p * p.ln();
            }
        }
        
        entropy
    }
    
    /// Calculate variance of a row
    fn calculate_row_variance(&self, row: &ndarray::ArrayView1<f32>) -> f32 {
        let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
        let variance: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
        variance
    }
    
    /// Extract features at specific level
    fn extract_features(&self, input: &Array2<f32>, level_idx: usize) -> DLResult<Array2<f32>> {
        let extractor = &self.feature_extractors[level_idx];
        
        // Apply feature extraction (matrix multiplication)
        let features = input.dot(extractor);
        
        Ok(features)
    }
    
    /// Update memory for specific level
    fn update_level_memory(&mut self, level_idx: usize, data: &Array2<f32>) -> DLResult<()> {
        let memory = &mut self.compressed_memory[level_idx];
        let (memory_rows, memory_cols) = memory.dim();
        let (data_rows, data_cols) = data.dim();
        
        // Ensure compatible dimensions
        if data_cols != memory_cols {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![memory_rows, memory_cols],
                actual: vec![data_rows, data_cols],
            });
        }
        
        // Circular buffer update
        let start_pos = (self.level_statistics[level_idx].total_writes % memory_rows as u64) as usize;
        
        for i in 0..data_rows.min(memory_rows) {
            let memory_row = (start_pos + i) % memory_rows;
            for j in 0..memory_cols {
                memory[[memory_row, j]] = data[[i, j]];
            }
        }
        
        Ok(())
    }
    
    /// Update compression statistics
    fn update_compression_statistics(&mut self, input: &Array2<f32>, compressed_levels: &[Array2<f32>], timestamp: usize) -> DLResult<()> {
        // Calculate quality scores first to avoid borrow conflicts
        let quality_scores: Vec<DLResult<f32>> = compressed_levels.iter()
            .map(|compressed| self.calculate_compression_quality(input, compressed))
            .collect();
        
        for (level_idx, compressed) in compressed_levels.iter().enumerate() {
            let stats = &mut self.level_statistics[level_idx];
            
            // Update basic statistics
            stats.total_writes += 1;
            stats.last_update = timestamp;
            
            // Calculate compression ratio
            let input_size = input.len() as f32;
            let compressed_size = compressed.len() as f32;
            let actual_ratio = compressed_size / input_size;
            stats.current_compression_ratio = actual_ratio;
            
            // Update memory utilization
            let memory = &self.compressed_memory[level_idx];
            let active_cells = memory.iter().filter(|&&x| x.abs() > 1e-6).count();
            stats.memory_utilization = active_cells as f32 / memory.len() as f32;
            
            // Set quality score
            if let Ok(quality_score) = quality_scores[level_idx] {
                stats.quality_score = quality_score;
            }
        }
        
        Ok(())
    }
    
    /// Calculate compression quality metrics
    fn calculate_compression_quality(&self, original: &Array2<f32>, compressed: &Array2<f32>) -> DLResult<f32> {
        // Simple quality metric based on energy preservation
        let original_energy: f32 = original.iter().map(|&x| x * x).sum();
        let compressed_energy: f32 = compressed.iter().map(|&x| x * x).sum();
        
        if original_energy == 0.0 {
            return Ok(1.0);
        }
        
        let energy_ratio = compressed_energy / original_energy;
        
        // Penalize excessive compression
        let compression_ratio = compressed.len() as f32 / original.len() as f32;
        let compression_penalty = if compression_ratio < 0.1 {
            0.5
        } else if compression_ratio < 0.2 {
            0.8
        } else {
            1.0
        };
        
        Ok(energy_ratio * compression_penalty)
    }
    
    /// Record compression event
    fn record_compression_event(&mut self, level_idx: usize, input_shape: &[usize], output_shape: &[usize], timestamp: usize) {
        let event = CompressionEvent {
            level: level_idx,
            timestamp,
            input_size: input_shape.iter().product(),
            output_size: output_shape.iter().product(),
            compression_ratio: output_shape.iter().product::<usize>() as f32 / input_shape.iter().product::<usize>() as f32,
        };
        
        self.compression_history.push(event);
        
        // Keep only recent history
        if self.compression_history.len() > 1000 {
            self.compression_history.remove(0);
        }
    }
    
    /// Get compressed memory at specific level
    pub fn get_level_memory(&self, level_idx: usize) -> DLResult<&Array2<f32>> {
        if level_idx >= self.num_levels {
            return Err(DeepLearningError::InvalidDimension { dim: level_idx });
        }
        
        Ok(&self.compressed_memory[level_idx])
    }
    
    /// Get number of compression levels
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }
    
    /// Get compression statistics
    pub fn get_level_statistics(&self, level_idx: usize) -> DLResult<&CompressionStatistics> {
        if level_idx >= self.num_levels {
            return Err(DeepLearningError::InvalidDimension { dim: level_idx });
        }
        
        Ok(&self.level_statistics[level_idx])
    }
    
    /// Get importance scores for level
    pub fn get_importance_scores(&self, level_idx: usize) -> DLResult<&[f32]> {
        if level_idx >= self.num_levels {
            return Err(DeepLearningError::InvalidDimension { dim: level_idx });
        }
        
        Ok(&self.importance_scores[level_idx])
    }
    
    /// Reset compression state
    pub fn reset(&mut self) -> DLResult<()> {
        for memory in &mut self.compressed_memory {
            memory.fill(0.0);
        }
        
        for stats in &mut self.level_statistics {
            *stats = CompressionStatistics::default();
        }
        
        for scores in &mut self.importance_scores {
            scores.clear();
        }
        
        self.compression_history.clear();
        
        Ok(())
    }
    
    /// Set adaptive compression threshold
    pub fn set_adaptive_threshold(&mut self, threshold: f32) {
        self.adaptive_threshold = threshold;
    }
    
    /// Set compression budget
    pub fn set_compression_budget(&mut self, budget: f32) {
        self.compression_budget = budget;
    }
}

/// Statistics for compression monitoring
#[derive(Debug, Clone, Default)]
pub struct CompressionStatistics {
    pub total_writes: u64,
    pub last_update: usize,
    pub current_compression_ratio: f32,
    pub memory_utilization: f32,
    pub quality_score: f32,
    pub average_latency: f32,
}

/// Compression event record
#[derive(Debug, Clone)]
pub struct CompressionEvent {
    pub level: usize,
    pub timestamp: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_rhc_creation() {
        let levels = vec![
            CompressionLevel::new(0, 0.8, 8, "Token level", 256),
            CompressionLevel::new(1, 0.6, 16, "Phrase level", 128),
            CompressionLevel::new(2, 0.4, 32, "Paragraph level", 64),
        ];
        
        let rhc = RecursiveHolographicCompression::new(levels, 4, 10.0, 0.5).unwrap();
        assert_eq!(rhc.num_levels, 3);
        assert_eq!(rhc.pool_size, 4);
    }
    
    #[test]
    fn test_compression_level_creation() {
        let level = CompressionLevel::new(0, 0.5, 16, "Test level", 128);
        assert_eq!(level.level, 0);
        assert_eq!(level.compression_ratio, 0.5);
        assert_eq!(level.window_size, 16);
        assert_eq!(level.description, "Test level");
        assert_eq!(level.target_features, 128);
    }
    
    #[test]
    fn test_importance_calculation() {
        let levels = vec![CompressionLevel::new(0, 0.5, 8, "Test", 64)];
        let mut rhc = RecursiveHolographicCompression::new(levels, 4, 10.0, 0.5).unwrap();
        
        let input = Array2::from_shape_vec((4, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
            5.0, 6.0, 7.0, 8.0,
            1.0, 1.0, 1.0, 1.0,
        ]).unwrap();
        
        let importance = rhc.calculate_importance_scores(&input, 0).unwrap();
        assert_eq!(importance.len(), 4);
        
        // Row with zeros should have lowest importance
        assert!(importance[1] < importance[0]);
        assert!(importance[1] < importance[2]);
    }
}
