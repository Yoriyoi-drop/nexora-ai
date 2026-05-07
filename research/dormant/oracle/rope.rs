//! Extended RoPE dengan Frekuensi Basis Dinamis
//! 
//! Implementasi Rotary Position Embedding yang diperluas dengan
//! frekuensi basis dinamis untuk mempertahankan koherensi posisional
//! lintas file dalam satu repositori kode.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Konfigurasi Extended RoPE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedRopeConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Base frequency for RoPE
    pub base: f32,
    /// Scaling factor for extended context
    pub scaling_factor: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dynamic frequency adjustment
    pub dynamic_frequency: bool,
    /// Cross-file awareness factor
    pub cross_file_factor: f32,
}

impl Default for ExtendedRopeConfig {
    fn default() -> Self {
        Self {
            d_model: 4096,
            n_heads: 32,
            base: 10000.0,
            scaling_factor: 1.0,
            max_seq_len: 32768,
            dynamic_frequency: bool,
            cross_file_factor: 0.1,
        }
    }
}

/// Extended Rotary Position Embedding
pub struct ExtendedRope {
    config: ExtendedRopeConfig,
    freqs_cos: Array2<f32>,
    freqs_sin: Array2<f32>,
    cross_file_embeddings: Array2<f32>,
}

impl ExtendedRope {
    pub fn new(config: ExtendedRopeConfig) -> Self {
        let head_dim = config.d_model / config.n_heads;
        let (freqs_cos, freqs_sin) = Self::compute_freqs(&config, head_dim);
        let cross_file_embeddings = Self::compute_cross_file_embeddings(&config);
        
        Self {
            config,
            freqs_cos,
            freqs_sin,
            cross_file_embeddings,
        }
    }
    
    /// Apply RoPE to query and key tensors
    pub fn apply_rotary_emb(&self, q: &Array3<f32>, k: &Array3<f32>, positions: &[usize]) -> Result<(Array3<f32>, Array3<f32>)> {
        let (batch_size, seq_len, head_dim) = (q.dim().0, q.dim().1, q.dim().2);
        
        // Get frequency embeddings for positions
        let (cos_emb, sin_emb) = self.get_position_embeddings(positions)?;
        
        // Apply rotary embedding to query
        let q_rotated = self.apply_rotary_to_tensor(q, &cos_emb, &sin_emb)?;
        
        // Apply rotary embedding to key
        let k_rotated = self.apply_rotary_to_tensor(k, &cos_emb, &sin_emb)?;
        
        Ok((q_rotated, k_rotated))
    }
    
    /// Apply cross-file awareness
    pub fn apply_cross_file_awareness(&self, x: &Array3<f32>, file_ids: &[usize]) -> Result<Array3<f32>> {
        let (batch_size, seq_len, d_model) = (x.dim().0, x.dim().1, x.dim().2);
        let mut output = x.clone();
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                let file_id = file_ids[i];
                if file_id < self.cross_file_embeddings.dim().0 {
                    let file_emb = self.cross_file_embeddings.slice(s![file_id, ..]);
                    let mut x_slice = output.slice_mut(s![b, i, ..]);
                    
                    // Apply cross-file embedding
                    for j in 0..d_model {
                        x_slice[[j]] += self.config.cross_file_factor * file_emb[[j]];
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Get position embeddings for given positions
    fn get_position_embeddings(&self, positions: &[usize]) -> Result<(Array2<f32>, Array2<f32>)> {
        let seq_len = positions.len();
        let head_dim = self.config.d_model / self.config.n_heads;
        
        let mut cos_emb = Array2::zeros((seq_len, head_dim));
        let mut sin_emb = Array2::zeros((seq_len, head_dim));
        
        for (i, &pos) in positions.iter().enumerate() {
            if pos < self.config.max_seq_len {
                let cos_row = self.freqs_cos.slice(s![pos, ..]);
                let sin_row = self.freqs_sin.slice(s![pos, ..]);
                
                let mut cos_slice = cos_emb.slice_mut(s![i, ..]);
                let mut sin_slice = sin_emb.slice_mut(s![i, ..]);
                
                cos_slice.assign(&cos_row);
                sin_slice.assign(&sin_row);
            }
        }
        
        Ok((cos_emb, sin_emb))
    }
    
    /// Apply rotary transformation to tensor
    fn apply_rotary_to_tensor(&self, x: &Array3<f32>, cos_emb: &Array2<f32>, sin_emb: &Array2<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, head_dim) = (x.dim().0, x.dim().1, x.dim().2);
        let mut output = Array3::zeros(x.dim());
        
        for b in 0..batch_size {
            for i in 0..seq_len {
                let cos_row = cos_emb.slice(s![i, ..]);
                let sin_row = sin_emb.slice(s![i, ..]);
                let x_row = x.slice(s![b, i, ..]);
                
                // Apply rotation: x_rot = x * cos + x_rotated * sin
                for j in 0..head_dim {
                    let x_j = x_row[[j]];
                    let x_j_rotated = if j % 2 == 0 && j + 1 < head_dim {
                        x_row[[j + 1]] * -1.0
                    } else if j % 2 == 1 && j - 1 >= 0 {
                        x_row[[j - 1]]
                    } else {
                        x_j
                    };
                    
                    output[[b, i, j]] = x_j * cos_row[[j]] + x_j_rotated * sin_row[[j]];
                }
            }
        }
        
        Ok(output)
    }
    
    /// Compute frequency embeddings
    fn compute_freqs(config: &ExtendedRopeConfig, head_dim: usize) -> (Array2<f32>, Array2<f32>) {
        let mut freqs_cos = Array2::zeros((config.max_seq_len, head_dim));
        let mut freqs_sin = Array2::zeros((config.max_seq_len, head_dim));
        
        for pos in 0..config.max_seq_len {
            for i in 0..head_dim {
                let freq = Self::compute_frequency(config, pos, i, head_dim);
                freqs_cos[[pos, i]] = freq.cos();
                freqs_sin[[pos, i]] = freq.sin();
            }
        }
        
        (freqs_cos, freqs_sin)
    }
    
    /// Compute frequency with dynamic adjustment
    fn compute_frequency(config: &ExtendedRopeConfig, pos: usize, i: usize, head_dim: usize) -> f32 {
        let base_freq = config.base;
        let scaling = config.scaling_factor;
        
        // Dynamic frequency adjustment based on position
        let dynamic_factor = if config.dynamic_frequency {
            let position_ratio = pos as f32 / config.max_seq_len as f32;
            1.0 + 0.5 * position_ratio // Increase frequency for later positions
        } else {
            1.0
        };
        
        // Original RoPE frequency computation
        let freq_index = i / 2;
        let theta = base_freq.powf(freq_index as f32 / head_dim as f32);
        let freq = pos as f32 * theta * scaling * dynamic_factor;
        
        freq
    }
    
    /// Compute cross-file embeddings
    fn compute_cross_file_embeddings(config: &ExtendedRopeConfig) -> Array2<f32> {
        let max_files = 100; // Maximum number of files to track
        let mut embeddings = Array2::zeros((max_files, config.d_model));
        
        for file_id in 0..max_files {
            for i in 0..config.d_model {
                // Simple deterministic embedding based on file ID
                embeddings[[file_id, i]] = ((file_id * (i + 1)) as f32).sin();
            }
        }
        
        embeddings
    }
    
    /// Update frequency scaling dynamically
    pub fn update_scaling(&mut self, new_scaling: f32) {
        self.config.scaling_factor = new_scaling;
        let head_dim = self.config.d_model / self.config.n_heads;
        let (freqs_cos, freqs_sin) = Self::compute_freqs(&self.config, head_dim);
        self.freqs_cos = freqs_cos;
        self.freqs_sin = freqs_sin;
    }
    
    /// Get current scaling factor
    pub fn get_scaling(&self) -> f32 {
        self.config.scaling_factor
    }
    
    /// Enable/disable dynamic frequency
    pub fn set_dynamic_frequency(&mut self, enabled: bool) {
        self.config.dynamic_frequency = enabled;
        let head_dim = self.config.d_model / self.config.n_heads;
        let (freqs_cos, freqs_sin) = Self::compute_freqs(&self.config, head_dim);
        self.freqs_cos = freqs_cos;
        self.freqs_sin = freqs_sin;
    }
}

/// Position-aware attention mask
pub struct PositionAwareMask {
    max_seq_len: usize,
    cross_file_mask: Array2<bool>,
}

impl PositionAwareMask {
    pub fn new(max_seq_len: usize) -> Self {
        let cross_file_mask = Array2::from_elem((max_seq_len, max_seq_len), true);
        
        Self {
            max_seq_len,
            cross_file_mask,
        }
    }
    
    /// Create attention mask with cross-file awareness
    pub fn create_mask(&self, seq_len: usize, file_ids: &[usize]) -> Array2<f32> {
        let mut mask = Array2::zeros((seq_len, seq_len));
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal mask
                if j > i {
                    mask[[i, j]] = f32::NEG_INFINITY;
                } else {
                    // Cross-file attention penalty
                    let file_i = file_ids[i];
                    let file_j = file_ids[j];
                    
                    if file_i != file_j {
                        // Apply small penalty for cross-file attention
                        mask[[i, j]] = -0.1;
                    } else {
                        mask[[i, j]] = 0.0;
                    }
                }
            }
        }
        
        mask
    }
    
    /// Update cross-file mask
    pub fn update_cross_file_mask(&mut self, file_i: usize, file_j: usize, allow: bool) {
        if file_i < self.max_seq_len && file_j < self.max_seq_len {
            self.cross_file_mask[[file_i, file_j]] = allow;
        }
    }
}

/// Dynamic frequency scheduler
pub struct FrequencyScheduler {
    initial_scaling: f32,
    final_scaling: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl FrequencyScheduler {
    pub fn new(initial_scaling: f32, final_scaling: f32, warmup_steps: usize) -> Self {
        Self {
            initial_scaling,
            final_scaling,
            warmup_steps,
            current_step: 0,
        }
    }
    
    /// Get current scaling factor
    pub fn get_scaling(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            self.initial_scaling + (self.final_scaling - self.initial_scaling) * progress
        } else {
            self.final_scaling
        }
    }
    
    /// Step the scheduler
    pub fn step(&mut self) {
        self.current_step += 1;
    }
    
    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Cross-file position tracker
pub struct CrossFilePositionTracker {
    file_positions: std::collections::HashMap<usize, Vec<usize>>,
    current_file: usize,
    position_counter: usize,
}

impl CrossFilePositionTracker {
    pub fn new() -> Self {
        Self {
            file_positions: std::collections::HashMap::new(),
            current_file: 0,
            position_counter: 0,
        }
    }
    
    /// Add a position for a file
    pub fn add_position(&mut self, file_id: usize) -> usize {
        if file_id != self.current_file {
            self.current_file = file_id;
            self.position_counter = 0;
        } else {
            self.position_counter += 1;
        }
        
        let positions = self.file_positions.entry(file_id).or_insert_with(Vec::new);
        positions.push(self.position_counter);
        
        self.position_counter
    }
    
    /// Get all positions for a file
    pub fn get_positions(&self, file_id: usize) -> Option<&Vec<usize>> {
        self.file_positions.get(&file_id)
    }
    
    /// Get relative position between two positions
    pub fn get_relative_position(&self, pos1: usize, pos2: usize) -> isize {
        (pos2 as isize) - (pos1 as isize)
    }
    
    /// Check if two positions are in the same file
    pub fn same_file(&self, pos1: usize, pos2: usize) -> bool {
        for (file_id, positions) in &self.file_positions {
            if positions.contains(&pos1) && positions.contains(&pos2) {
                return true;
            }
        }
        false
    }
    
    /// Reset tracker
    pub fn reset(&mut self) {
        self.file_positions.clear();
        self.current_file = 0;
        self.position_counter = 0;
    }
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Compute optimal scaling factor for sequence length
    pub fn compute_optimal_scaling(seq_len: usize, max_seq_len: usize) -> f32 {
        if seq_len <= max_seq_len {
            1.0
        } else {
            (max_seq_len as f32 / seq_len as f32).max(0.1)
        }
    }
    
    /// Analyze position distribution in code
    pub fn analyze_position_distribution(positions: &[usize]) -> PositionAnalysis {
        if positions.is_empty() {
            return PositionAnalysis::default();
        }
        
        let min_pos = *positions.iter().min().unwrap();
        let max_pos = *positions.iter().max().unwrap();
        let avg_pos = positions.iter().sum::<usize>() as f32 / positions.len() as f32;
        
        let mut gaps = Vec::new();
        for i in 1..positions.len() {
            gaps.push(positions[i] - positions[i - 1]);
        }
        
        let avg_gap = gaps.iter().sum::<usize>() as f32 / gaps.len() as f32;
        let max_gap = *gaps.iter().max().unwrap();
        
        PositionAnalysis {
            min_position: min_pos,
            max_position: max_pos,
            avg_position: avg_pos,
            avg_gap,
            max_gap,
            total_positions: positions.len(),
        }
    }
    
    /// Create file-aware position sequence
    pub fn create_file_aware_positions(code_blocks: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut positions = Vec::new();
        let mut file_ids = Vec::new();
        
        for &(file_id, block_len) in code_blocks {
            for i in 0..block_len {
                positions.push(i);
                file_ids.push(file_id);
            }
        }
        
        (positions, file_ids)
    }
    
    #[derive(Debug, Clone, Default)]
    pub struct PositionAnalysis {
        pub min_position: usize,
        pub max_position: usize,
        pub avg_position: f32,
        pub avg_gap: f32,
        pub max_gap: usize,
        pub total_positions: usize,
    }
    
    /// Validate RoPE configuration
    pub fn validate_rope_config(config: &ExtendedRopeConfig) -> Result<()> {
        if config.d_model % config.n_heads != 0 {
            return Err(anyhow::anyhow!("d_model must be divisible by n_heads"));
        }
        
        if config.base <= 0.0 {
            return Err(anyhow::anyhow!("base frequency must be positive"));
        }
        
        if config.scaling_factor <= 0.0 {
            return Err(anyhow::anyhow!("scaling factor must be positive"));
        }
        
        if config.max_seq_len == 0 {
            return Err(anyhow::anyhow!("max_seq_len must be positive"));
        }
        
        Ok(())
    }
    
    /// Estimate memory usage for RoPE
    pub fn estimate_rope_memory(config: &ExtendedRopeConfig) -> usize {
        let head_dim = config.d_model / config.n_heads;
        let freqs_memory = config.max_seq_len * head_dim * 2 * 4; // cos + sin, f32
        let cross_file_memory = 100 * config.d_model * 4; // 100 files max
        let mask_memory = config.max_seq_len * config.max_seq_len * 4; // attention mask
        
        freqs_memory + cross_file_memory + mask_memory
    }
}
