//! Vector Quantized VAE + FSQ (Finite Scalar Quantization) untuk CAFFEINE
//! 
//! VQ-VAE: EMA-based codebook (existing)
//! FSQ: Lookup-free quantization (SOTA — tidak ada codebook collapse)

use crate::caffeine::types::*;
use crate::caffeine::error::Result;

/// FSQ levels per dimension — fixed rounding boundaries
/// Default: [8, 6, 5] → 8×6×5 = 240 codes, 3 dimensi
pub const DEFAULT_FSQ_LEVELS: &[usize] = &[8, 6, 5];
/// High capacity FSQ levels: [8, 6, 5, 4] → 8×6×5×4 = 960 codes
pub const HIGH_CAPACITY_FSQ: &[usize] = &[8, 6, 5, 4];
/// Max capacity FSQ levels: [8, 6, 5, 4, 3] → 8×6×5×4×3 = 2880 codes
pub const MAX_CAPACITY_FSQ: &[usize] = &[8, 6, 5, 4, 3];

/// Finite Scalar Quantization (FSQ)
/// Tidak memerlukan codebook — langsung round ke nearest integer.
/// Tidak ada codebook collapse, deterministik, zero training overhead.
pub struct FSQuantizer {
    /// Levels per dimension (e.g., [8, 6, 5])
    levels: Vec<usize>,
    /// Projection dimension (before quantization)
    input_dim: usize,
    /// Number of codebooks (repeated quantization for higher capacity)
    num_codebooks: usize,
    /// Scaling factor (prevents saturation)
    scale: f32,
}

impl FSQuantizer {
    pub fn new(input_dim: usize, levels: &[usize], num_codebooks: usize) -> Self {
        let scale = (levels.iter().max().copied().unwrap_or(8) as f32) * 0.9;
        Self {
            levels: levels.to_vec(),
            input_dim,
            num_codebooks,
            scale,
        }
    }

    /// Quantize input embedding using FSQ
    /// 1. Project input ke low-dim space
    /// 2. Apply tanh untuk bounded range
    /// 3. Round ke nearest integer level
    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<f32>, Vec<usize>, f32)> {
        let codebook_dim = self.levels.len();
        let mut quantized_output = vec![0.0f32; input.len()];
        let mut token_ids = Vec::new();
        let mut total_rounding_error = 0.0f32;

        for book in 0..self.num_codebooks {
            let offset = book * codebook_dim;
            let (codes, ids, error) = self.quantize_single_codebook(input, offset)?;
            
            for (i, &val) in codes.iter().enumerate() {
                if i < quantized_output.len() {
                    quantized_output[i] += val / self.num_codebooks as f32;
                }
            }
            
            token_ids.push(ids);
            total_rounding_error += error;
        }

        Ok((quantized_output, token_ids, total_rounding_error))
    }

    /// Single codebook FSQ quantization
    fn quantize_single_codebook(&self, input: &[f32], offset: usize) -> Result<(Vec<f32>, usize, f32)> {
        let codebook_dim = self.levels.len();
        let mut codes = vec![0.0f32; codebook_dim];
        let mut rounding_error = 0.0f32;
        let mut id = 0usize;
        let mut _stride = 1usize;

        for d in 0..codebook_dim {
            let input_idx = (offset + d) % self.input_dim.max(1);
            let val = if input_idx < input.len() { input[input_idx] } else { 0.0 };
            
            // tanh untuk bounded range dalam [-scale, scale]
            let bounded = (val * self.scale * 0.1).tanh() * self.scale;
            let level = self.levels[d];
            let half_levels = (level - 1) as f32 / 2.0;
            
            // Map ke [0, level-1]
            let normalized = (bounded / self.scale * half_levels + half_levels)
                .clamp(0.0, (level - 1) as f32);
            let rounded = normalized.round();
            let rounded_idx = rounded as usize;
            
            codes[d] = (rounded - half_levels) / half_levels * self.scale;
            rounding_error += (normalized - rounded).powi(2);
            
            // Encode ke single integer ID (base-level encoding)
            id = id * level + rounded_idx.min(level - 1);
            _stride *= level;
        }

        Ok((codes, id, rounding_error))
    }

    /// Dequantize token IDs back to embeddings
    pub fn dequantize(&self, token_ids: &[usize]) -> Result<Vec<f32>> {
        let codebook_dim = self.levels.len();
        let mut output = vec![0.0f32; self.input_dim];

        for (book, &token_id) in token_ids.iter().enumerate() {
            let mut id = token_id;
            let mut codes = vec![0.0f32; codebook_dim];

            for d in (0..codebook_dim).rev() {
                let level = self.levels[d];
                let idx = id % level;
                id /= level;
                
                let half_levels = (level - 1) as f32 / 2.0;
                // Reverse the mapping
                codes[d] = (idx as f32 - half_levels) / half_levels * self.scale;
            }

            for (i, &val) in codes.iter().enumerate() {
                let output_idx = (book * codebook_dim + i) % self.input_dim;
                if output_idx < output.len() {
                    output[output_idx] += val / self.num_codebooks as f32;
                }
            }
        }

        Ok(output)
    }

    /// Get compression ratio
    pub fn get_compression_ratio(&self) -> f32 {
        let total_codes: usize = self.levels.iter().product();
        let bits_per_token = (total_codes as f32).log2();
        let original_bits = self.input_dim as f32 * 32.0; // FP32
        original_bits / bits_per_token
    }

    /// FSQ does not require training
    pub fn train_batch(&self, _inputs: &[Vec<f32>]) -> Result<TrainingMetrics> {
        Ok(TrainingMetrics {
            total_loss: 0.0,
            reconstruction_error: 0.0,
            commitment_loss: 0.0,
            codebook_entropy: (self.levels.iter().product::<usize>() as f32).log2(),
        })
    }
}

/// Vector Quantized VAE (existing) — EMA-based codebook
pub struct VectorQuantizedVAE {
    token_dim: usize,
    codebook_size: usize,
    num_codebooks: usize,
    commitment_weight: f32,
    codebooks: Vec<Vec<f32>>,
    usage_counts: Vec<Vec<usize>>,
}

impl VectorQuantizedVAE {
    /// Create new VQ-VAE
    pub fn new(token_dim: usize, codebook_size: usize, num_codebooks: usize, commitment_weight: f32) -> Result<Self> {
        let mut codebooks = Vec::new();
        let mut usage_counts = Vec::new();
        
        for _ in 0..num_codebooks {
            let codebook = Self::initialize_codebook(token_dim, codebook_size)?;
            codebooks.push(codebook);
            usage_counts.push(vec![0; codebook_size]);
        }
        
        Ok(Self {
            token_dim,
            codebook_size,
            num_codebooks,
            commitment_weight,
            codebooks,
            usage_counts,
        })
    }
    
    /// Quantize input embeddings
    pub fn quantize(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<usize>, f32)> {
        if input.len() != self.token_dim {
            return Err(crate::caffeine::error::CaffeineError::tokenizer(
                "Input dimension doesn't match token dimension"
            ));
        }
        
        let mut quantized_output = vec![0.0f32; input.len()];
        let mut token_ids = Vec::new();
        let mut total_loss = 0.0f32;
        
        // Quantize using each codebook
        for book_idx in 0..self.num_codebooks {
            let (best_code, token_id, loss) = self.find_closest_code(input, book_idx)?;
            
            // Add to quantized output
            for i in 0..self.token_dim {
                quantized_output[i] += best_code[i] / self.num_codebooks as f32;
            }
            
            token_ids.push(token_id);
            total_loss += loss;
            
            // Update usage count
            self.usage_counts[book_idx][token_id] += 1;
        }
        
        Ok((quantized_output, token_ids, total_loss))
    }
    
    /// Dequantize token IDs back to embeddings
    pub fn dequantize(&self, token_ids: &[usize]) -> Result<Vec<f32>> {
        if token_ids.len() != self.num_codebooks {
            return Err(crate::caffeine::error::CaffeineError::tokenizer(
                "Number of token IDs doesn't match number of codebooks"
            ));
        }
        
        let mut output = vec![0.0f32; self.token_dim];
        
        for (book_idx, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.codebook_size {
                return Err(crate::caffeine::error::CaffeineError::tokenizer(
                    "Token ID exceeds codebook size"
                ));
            }
            
            if book_idx < self.codebooks.len() && token_id < self.codebooks[book_idx].len() / self.token_dim {
                let code_start = token_id * self.token_dim;
                let code_end = code_start + self.token_dim;
                
                if code_end <= self.codebooks[book_idx].len() {
                    for i in 0..self.token_dim {
                        output[i] += self.codebooks[book_idx][code_start + i] / self.num_codebooks as f32;
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Find closest code in codebook
    fn find_closest_code(&self, input: &[f32], book_idx: usize) -> Result<(Vec<f32>, usize, f32)> {
        if book_idx >= self.codebooks.len() {
            return Err(crate::caffeine::error::CaffeineError::tokenizer(
                "Codebook index out of bounds"
            ));
        }
        
        let codebook = &self.codebooks[book_idx];
        let mut best_distance = f32::INFINITY;
        let mut best_code_idx = 0;
        let mut best_code = vec![0.0f32; self.token_dim];
        
        // Search through codebook
        for code_idx in 0..self.codebook_size {
            let code_start = code_idx * self.token_dim;
            let code_end = code_start + self.token_dim;
            
            if code_end > codebook.len() {
                continue;
            }
            
            let code = &codebook[code_start..code_end];
            let distance = Self::compute_euclidean_distance(input, code);
            
            if distance < best_distance {
                best_distance = distance;
                best_code_idx = code_idx;
                best_code = code.to_vec();
            }
        }
        
        // Compute commitment loss
        let commitment_loss = best_distance * self.commitment_weight;
        
        Ok((best_code, best_code_idx, commitment_loss))
    }
    
    /// Compute Euclidean distance
    fn compute_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len().min(b.len()) {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    /// Initialize codebook with random values
    fn initialize_codebook(token_dim: usize, codebook_size: usize) -> Result<Vec<f32>> {
        let mut codebook = vec![0.0f32; token_dim * codebook_size];
        
        for i in 0..codebook_size {
            for d in 0..token_dim {
                let idx = i * token_dim + d;
                // Initialize with small random values
                codebook[idx] = ((i * d) as f32 * 0.01).sin() * 0.1;
            }
        }
        
        Ok(codebook)
    }
    
    /// Update codebook using exponential moving average
    pub fn update_codebook(&mut self, input: &[f32], token_ids: &[usize], learning_rate: f32) -> Result<()> {
        if token_ids.len() != self.num_codebooks {
            return Err(crate::caffeine::error::CaffeineError::tokenizer(
                "Number of token IDs doesn't match number of codebooks"
            ));
        }
        
        for (book_idx, &token_id) in token_ids.iter().enumerate() {
            if book_idx >= self.codebooks.len() || token_id >= self.codebook_size {
                continue;
            }
            
            let code_start = token_id * self.token_dim;
            let code_end = code_start + self.token_dim;
            
            if code_end > self.codebooks[book_idx].len() {
                continue;
            }
            
            // Update code using EMA
            for d in 0..self.token_dim {
                let idx = code_start + d;
                let error = input[d] - self.codebooks[book_idx][idx];
                self.codebooks[book_idx][idx] += learning_rate * error;
            }
        }
        
        Ok(())
    }
    
    /// Get compression ratio
    pub fn get_compression_ratio(&self) -> f32 {
        // Compression ratio = original_size / compressed_size
        let original_size = self.token_dim as f32 * 4.0; // 4 bytes per float
        let compressed_size = (self.num_codebooks as f32 * (self.codebook_size as f32).log2() / 8.0); // bits to bytes
        original_size / compressed_size
    }
    
    /// Get codebook usage statistics
    pub fn get_usage_stats(&self) -> Vec<Vec<usize>> {
        self.usage_counts.clone()
    }
    
    /// Reset usage counts
    pub fn reset_usage_counts(&mut self) {
        for counts in &mut self.usage_counts {
            counts.fill(0);
        }
    }
    
    /// Get codebook entropy (measure of usage diversity)
    pub fn get_codebook_entropy(&self) -> f32 {
        let mut total_entropy = 0.0f32;
        
        for book_usage in &self.usage_counts {
            let total_usage: usize = book_usage.iter().sum();
            if total_usage == 0 {
                continue;
            }
            
            let mut entropy = 0.0f32;
            for &count in book_usage {
                if count > 0 {
                    let probability = count as f32 / total_usage as f32;
                    entropy -= probability * probability.log2();
                }
            }
            
            total_entropy += entropy;
        }
        
        total_entropy / self.num_codebooks as f32
    }
}

/// VQ-VAE training utilities
pub struct VQVAETrainer {
    learning_rate: f32,
    _decay: f32,
}

impl VQVAETrainer {
    /// Create new trainer
    pub fn new(learning_rate: f32, _decay: f32) -> Self {
        Self {
            learning_rate,
            _decay,
        }
    }
    
    /// Train VQ-VAE on batch of inputs
    pub fn train_batch(
        &mut self,
        vq_vae: &mut VectorQuantizedVAE,
        inputs: &[Vec<f32>],
    ) -> Result<TrainingMetrics> {
        let mut total_loss = 0.0f32;
        let mut total_reconstruction_error = 0.0f32;
        let mut total_commitment_loss = 0.0f32;
        
        for input in inputs {
            // Quantize input
            let (quantized, token_ids, commitment_loss) = vq_vae.quantize(input)?;
            
            // Compute reconstruction error
            let reconstruction_error = Self::compute_reconstruction_error(input, &quantized);
            
            // Update codebook
            vq_vae.update_codebook(input, &token_ids, self.learning_rate)?;
            
            total_loss += reconstruction_error + commitment_loss;
            total_reconstruction_error += reconstruction_error;
            total_commitment_loss += commitment_loss;
        }
        
        let batch_size = inputs.len() as f32;
        Ok(TrainingMetrics {
            total_loss: total_loss / batch_size,
            reconstruction_error: total_reconstruction_error / batch_size,
            commitment_loss: total_commitment_loss / batch_size,
            codebook_entropy: vq_vae.get_codebook_entropy(),
        })
    }
    
    /// Compute reconstruction error
    fn compute_reconstruction_error(original: &[f32], reconstructed: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..original.len().min(reconstructed.len()) {
            let diff = original[i] - reconstructed[i];
            sum += diff * diff;
        }
        sum / original.len() as f32
    }
}

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub total_loss: f32,
    pub reconstruction_error: f32,
    pub commitment_loss: f32,
    pub codebook_entropy: f32,
}
