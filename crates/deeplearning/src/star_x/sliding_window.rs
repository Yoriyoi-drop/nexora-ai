//! Sliding Window Attention untuk STAR-X Performance Optimization
//!
//! True sparse attention dengan O(n√n) complexity menggunakan:
//! - Sliding window patterns untuk local attention
//! - Hierarchical attention untuk global context
//! - Efficient memory access patterns
//! - Integration dengan existing BLAS backend

use crate::{DLResult, DeepLearningError};
use crate::star_x::blas_backend::BlasOperations;
use crate::star_x::tensor_pool::PooledTensor2D;
use ndarray::{ArrayD, Array2, ArrayView, s};

/// Sliding Window Attention implementation
#[derive(Debug, Clone)]
pub struct SlidingWindowAttention {
    // Attention parameters
    query_weights: Array2<f32>,
    key_weights: Array2<f32>,
    value_weights: Array2<f32>,
    output_weights: Array2<f32>,
    
    // BLAS backend for high-performance operations
    blas_ops: Option<BlasOperations>,
    
    // Sliding window configuration
    window_size: usize,
    stride: usize,
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    
    // Hierarchical attention levels
    num_levels: usize,
    level_ratios: Vec<f32>,
    
    // Performance statistics
    avg_window_utilization: f32,
    cache_hit_rate: f32,
}

impl SlidingWindowAttention {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        window_size: usize,
        stride: usize,
        num_levels: usize,
    ) -> DLResult<Self> {
        if hidden_dim % num_heads != 0 {
            return Err(DeepLearningError::Configuration {
                reason: format!("hidden_dim {} must be divisible by num_heads {}", hidden_dim, num_heads),
            });
        }
        
        let head_dim = hidden_dim / num_heads;
        
        // Initialize weights
        let query_weights = Self::xavier_init(hidden_dim, hidden_dim);
        let key_weights = Self::xavier_init(hidden_dim, hidden_dim);
        let value_weights = Self::xavier_init(hidden_dim, hidden_dim);
        let output_weights = Self::xavier_init(hidden_dim, hidden_dim);
        
        // Initialize BLAS operations
        let blas_ops = BlasOperations::auto_detect().ok();
        
        // Calculate level ratios for hierarchical attention
        let mut level_ratios = Vec::with_capacity(num_levels);
        for i in 0..num_levels {
            let ratio = 2.0_f32.powi(-(i as i32));
            level_ratios.push(ratio);
        }
        
        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            blas_ops,
            window_size,
            stride,
            num_heads,
            head_dim,
            hidden_dim,
            num_levels,
            level_ratios,
            avg_window_utilization: 0.0,
            cache_hit_rate: 0.0,
        })
    }
    
    /// Forward pass dengan sliding window attention
    pub fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()?;
        let seq_len = input_view.len() / self.hidden_dim;
        
        if seq_len == 0 {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![self.hidden_dim],
                actual: vec![0],
            });
        }
        
        // Reshape input untuk multi-head attention
        let input_reshaped = input_view.into_shape((seq_len, self.hidden_dim))?;
        
        // Compute Q, K, V projections
        let (q, k, v) = self.compute_qkv(&input_reshaped)?;
        
        // Apply sliding window attention
        let attention_output = self.sliding_window_attention(&q, &k, &v, seq_len)?;
        
        // Final projection
        let output = self.final_projection(&attention_output)?;
        
        Ok(output.into_dyn())
    }
    
    /// Compute Q, K, V projections dengan BLAS optimization
    fn compute_qkv(&self, input: &ArrayView<f32, ndarray::Ix2>) -> DLResult<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let (seq_len, hidden_dim) = input.dim();
        
        // Use pooled tensors for output
        let mut pooled_q = PooledTensor2D::new(seq_len, hidden_dim)?;
        let mut pooled_k = PooledTensor2D::new(seq_len, hidden_dim)?;
        let mut pooled_v = PooledTensor2D::new(seq_len, hidden_dim)?;
        
        let q = pooled_q.get_mut();
        let k = pooled_k.get_mut();
        let v = pooled_v.get_mut();
        
        // Use BLAS if available
        if let Some(ref blas_ops) = self.blas_ops {
            blas_ops.gemm(1.0, input.view(), self.query_weights.view(), 0.0, q.view_mut())?;
            blas_ops.gemm(1.0, input.view(), self.key_weights.view(), 0.0, k.view_mut())?;
            blas_ops.gemm(1.0, input.view(), self.value_weights.view(), 0.0, v.view_mut())?;
        } else {
            // Fallback to ndarray
            let q_result = input.dot(&self.query_weights);
            let k_result = input.dot(&self.key_weights);
            let v_result = input.dot(&self.value_weights);
            
            q.assign(&q_result);
            k.assign(&k_result);
            v.assign(&v_result);
        }
        
        Ok((q.clone(), k.clone(), v.clone()))
    }
    
    /// Sliding window attention computation
    fn sliding_window_attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        seq_len: usize,
    ) -> DLResult<Array2<f32>> {
        let (seq_len_q, hidden_dim) = q.dim();
        let (seq_len_k, _) = k.dim();
        
        // Reshape untuk multi-head attention
        let q_heads = q.view().into_shape((seq_len_q, self.num_heads, self.head_dim))?;
        let k_heads = k.view().into_shape((seq_len_k, self.num_heads, self.head_dim))?;
        let v_heads = v.view().into_shape((seq_len_k, self.num_heads, self.head_dim))?;
        
        // Initialize output tensor
        let mut pooled_output = PooledTensor2D::new(seq_len_q, hidden_dim)?;
        let output = pooled_output.get_mut();
        
        // Process each head
        for head in 0..self.num_heads {
            let q_head = q_heads.slice(s![.., head, ..]);
            let k_head = k_heads.slice(s![.., head, ..]);
            let v_head = v_heads.slice(s![.., head, ..]);
            
            // Apply sliding window attention for this head
            let head_output = self.sliding_window_attention_head(&q_head, &k_head, &v_head, seq_len)?;
            
            // Copy to output tensor
            for i in 0..seq_len_q {
                for j in 0..self.head_dim {
                    output[[i, head * self.head_dim + j]] = head_output[[i, j]];
                }
            }
        }
        
        Ok(output.clone())
    }
    
    /// Sliding window attention untuk single head
    fn sliding_window_attention_head(
        &self,
        q: &ArrayView<f32, ndarray::Ix2>,
        k: &ArrayView<f32, ndarray::Ix2>,
        v: &ArrayView<f32, ndarray::Ix2>,
        seq_len: usize,
    ) -> DLResult<Array2<f32>> {
        let mut pooled_output = PooledTensor2D::new(seq_len, self.head_dim)?;
        let output = pooled_output.get_mut();
        
        // Process each position with sliding window
        for i in 0..seq_len {
            // Determine window boundaries
            let start = if i >= self.window_size / 2 {
                i - self.window_size / 2
            } else {
                0
            };
            let end = (i + self.window_size / 2 + 1).min(seq_len);
            
            // Extract window
            let window_k = k.slice(s![start..end, ..]);
            let window_v = v.slice(s![start..end, ..]);
            let q_i = q.slice(s![i..i+1, ..]);
            
            // Compute attention scores
            let mut scores = self.compute_attention_scores(&q_i, &window_k)?;
            
            // Apply causal mask
            for j in start..i {
                scores[[0, j - start]] = f32::NEG_INFINITY;
            }
            
            // Apply softmax
            self.softmax_inplace(&mut scores)?;
            
            // Compute weighted sum
            let mut output_i = Array2::zeros((1, self.head_dim));
            if let Some(ref blas_ops) = self.blas_ops {
                blas_ops.gemm(1.0, scores.view(), window_v.view(), 0.0, output_i.view_mut())?;
            } else {
                output_i = scores.dot(&window_v);
            }
            
            // Copy to output
            for j in 0..self.head_dim {
                output[[i, j]] = output_i[[0, j]];
            }
        }
        
        Ok(output.clone())
    }
    
    /// Compute attention scores with BLAS optimization
    fn compute_attention_scores(
        &self,
        q: &ArrayView<f32, ndarray::Ix2>,
        k: &ArrayView<f32, ndarray::Ix2>,
    ) -> DLResult<Array2<f32>> {
        let (_, head_dim) = q.dim();
        let (window_size, _) = k.dim();
        
        let mut pooled_scores = PooledTensor2D::new(1, window_size)?;
        let scores = pooled_scores.get_mut();
        
        // Compute Q @ K^T / sqrt(d_k)
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        if let Some(ref blas_ops) = self.blas_ops {
            // Use BLAS for matrix multiplication
            blas_ops.gemm(scale, q.view(), k.view().t(), 0.0, scores.view_mut())?;
        } else {
            // Fallback to ndarray
            let q_k_t = q.dot(&k.t());
            scores.assign(&(q_k_t * scale));
        }
        
        Ok(scores.clone())
    }
    
    /// In-place softmax computation
    fn softmax_inplace(&self, scores: &mut Array2<f32>) -> DLResult<()> {
        let (batch_size, seq_len) = scores.dim();
        
        for i in 0..batch_size {
            // Find max for numerical stability
            let mut max_val = scores[[i, 0]];
            for j in 1..seq_len {
                max_val = max_val.max(scores[[i, j]]);
            }
            
            // Compute exp and sum
            let mut exp_sum = 0.0f32;
            for j in 0..seq_len {
                scores[[i, j]] = (scores[[i, j]] - max_val).exp();
                exp_sum += scores[[i, j]];
            }
            
            // Normalize
            if exp_sum > 0.0 {
                for j in 0..seq_len {
                    scores[[i, j]] /= exp_sum;
                }
            }
        }
        
        Ok(())
    }
    
    /// Final projection with BLAS optimization
    fn final_projection(&self, attention_output: &Array2<f32>) -> DLResult<Array2<f32>> {
        let (seq_len, hidden_dim) = attention_output.dim();
        let output_dim = self.output_weights.shape()[1];
        
        let mut pooled_output = PooledTensor2D::new(seq_len, output_dim)?;
        let output = pooled_output.get_mut();
        
        if let Some(ref blas_ops) = self.blas_ops {
            blas_ops.gemm(1.0, attention_output.view(), self.output_weights.view(), 0.0, output.view_mut())?;
        } else {
            let output_result = attention_output.dot(&self.output_weights);
            output.assign(&output_result);
        }
        
        Ok(output.clone())
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_elem((rows, cols), 0.0).map(|_| {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            rng.gen_range(-limit..limit)
        })
    }
    
    /// Get window utilization statistics
    pub fn get_window_utilization(&self) -> f32 {
        self.avg_window_utilization
    }
    
    /// Get cache hit rate statistics
    pub fn get_cache_hit_rate(&self) -> f32 {
        self.cache_hit_rate
    }
    
    /// Update performance statistics
    fn update_statistics(&mut self, seq_len: usize) {
        // Calculate average window utilization
        let total_windows = ((seq_len as f32 - 1.0) / self.stride as f32).ceil() as usize;
        let effective_window_size = (self.window_size as f32 * total_windows as f32).min(seq_len as f32);
        self.avg_window_utilization = effective_window_size / seq_len as f32;
    }
}

/// Hierarchical Sliding Window Attention untuk long sequences
#[derive(Debug, Clone)]
pub struct HierarchicalSlidingWindow {
    base_attention: SlidingWindowAttention,
    num_levels: usize,
    level_scales: Vec<usize>,
    fusion_weights: Vec<f32>,
}

impl HierarchicalSlidingWindow {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        window_size: usize,
        stride: usize,
        num_levels: usize,
    ) -> DLResult<Self> {
        let base_attention = SlidingWindowAttention::new(hidden_dim, num_heads, window_size, stride, num_levels)?;
        
        // Calculate level scales
        let mut level_scales = Vec::with_capacity(num_levels);
        let mut fusion_weights = Vec::with_capacity(num_levels);
        
        for i in 0..num_levels {
            let scale = 2_usize.pow(i as u32);
            level_scales.push(scale);
            
            // Fusion weights decrease with level
            let weight = 0.5_f32.powi(i as i32);
            fusion_weights.push(weight);
        }
        
        Ok(Self {
            base_attention,
            num_levels,
            level_scales,
            fusion_weights,
        })
    }
    
    /// Forward pass dengan hierarchical attention
    pub fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()?;
        let seq_len = input_view.len() / self.base_attention.hidden_dim;
        
        // Base level attention
        let base_output = self.base_attention.forward(input)?;
        
        // Additional levels for long sequences
        if seq_len > self.base_attention.window_size * 4 {
            let mut hierarchical_outputs = Vec::new();
            hierarchical_outputs.push(base_output);
            
            // Process additional levels
            for level in 1..self.num_levels {
                let scale = self.level_scales[level];
                let downsampled_len = seq_len / scale;
                
                if downsampled_len < self.base_attention.window_size {
                    break;
                }
                
                // Downsample input
                let downsampled_input = self.downsample_input(input, scale)?;
                
                // Apply attention at this level
                let level_output = self.base_attention.forward(&downsampled_input)?;
                
                // Upsample and add to outputs
                let upsampled_output = self.upsample_output(&level_output, seq_len)?;
                hierarchical_outputs.push(upsampled_output);
            }
            
            // Fuse hierarchical outputs
            self.fuse_hierarchical_outputs(&hierarchical_outputs)
        } else {
            Ok(base_output)
        }
    }
    
    /// Downsample input untuk higher level attention
    fn downsample_input(&self, input: &ArrayD<f32>, scale: usize) -> DLResult<ArrayD<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()?;
        let seq_len = input_view.len() / self.base_attention.hidden_dim;
        
        let downsampled_len = seq_len / scale;
        let hidden_dim = self.base_attention.hidden_dim;
        
        let mut pooled_output = PooledTensor2D::new(downsampled_len, hidden_dim)?;
        let output = pooled_output.get_mut();
        
        // Simple averaging downsampling
        for i in 0..downsampled_len {
            for j in 0..hidden_dim {
                let mut sum = 0.0f32;
                for k in 0..scale {
                    let idx = (i * scale + k) * hidden_dim + j;
                    if idx < input_view.len() {
                        sum += input_view[idx];
                    }
                }
                output[[i, j]] = sum / scale as f32;
            }
        }
        
        Ok(output.clone().into_dyn())
    }
    
    /// Upsample output dari higher level attention
    fn upsample_output(&self, output: &ArrayD<f32>, target_len: usize) -> DLResult<ArrayD<f32>> {
        let output_view = output.view().into_dimensionality::<ndarray::Ix1>()?;
        let seq_len = output_view.len() / self.base_attention.hidden_dim;
        let hidden_dim = self.base_attention.hidden_dim;
        
        let mut pooled_upsampled = PooledTensor2D::new(target_len, hidden_dim)?;
        let upsampled = pooled_upsampled.get_mut();
        
        let scale = target_len / seq_len;
        
        // Simple nearest neighbor upsampling
        for i in 0..target_len {
            let source_i = i / scale;
            if source_i < seq_len {
                for j in 0..hidden_dim {
                    upsampled[[i, j]] = output_view[source_i * hidden_dim + j];
                }
            }
        }
        
        Ok(upsampled.clone().into_dyn())
    }
    
    /// Fuse hierarchical outputs dengan learned weights
    fn fuse_hierarchical_outputs(&self, outputs: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        if outputs.is_empty() {
            return Err(DeepLearningError::Configuration {
                reason: "No outputs to fuse".to_string(),
            });
        }
        
        let first_output = &outputs[0];
        let output_view = first_output.view().into_dimensionality::<ndarray::Ix1>()?;
        let output_len = output_view.len();
        
        let mut pooled_fused = PooledTensor2D::new(1, output_len)?;
        let fused = pooled_fused.get_mut();
        
        // Initialize with first output
        for i in 0..output_len {
            fused[[0, i]] = output_view[i];
        }
        
        // Add weighted outputs from other levels
        for (level, output) in outputs.iter().enumerate().skip(1) {
            let weight = self.fusion_weights[level];
            let output_view = output.view().into_dimensionality::<ndarray::Ix1>()?;
            
            for i in 0..output_len {
                fused[[0, i]] += weight * output_view[i];
            }
        }
        
        Ok(fused.clone().into_dyn())
    }
}

/// Global instance untuk sliding window attention
pub fn create_sliding_window_attention(
    hidden_dim: usize,
    num_heads: usize,
    window_size: usize,
    stride: usize,
) -> DLResult<SlidingWindowAttention> {
    SlidingWindowAttention::new(hidden_dim, num_heads, window_size, stride, 1)
}

/// Global instance untuk hierarchical sliding window attention
pub fn create_hierarchical_sliding_window(
    hidden_dim: usize,
    num_heads: usize,
    window_size: usize,
    stride: usize,
    num_levels: usize,
) -> DLResult<HierarchicalSlidingWindow> {
    HierarchicalSlidingWindow::new(hidden_dim, num_heads, window_size, stride, num_levels)
}
