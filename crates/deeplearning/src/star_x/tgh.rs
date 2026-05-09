//! Temporal Gating Hierarchy (TGH)
//!
//! Multi-resolution temporal processing:
//! - Micro Gate: dependensi lokal (sintaks, morfologi, token adjacency)
//! - Meso Gate: dependensi chunk-level (frasa, klausa, sub-sequence)  
//! - Macro Gate: episodic memory eksternal

use crate::{DLResult, DeepLearningError};
use crate::star_x::core::HierarchicalGating;
use ndarray::{ArrayD, Array1, Array2};
use rand;

/// Temporal Gating Hierarchy implementation
#[derive(Debug, Clone)]
pub struct TemporalGatingHierarchy {
    // Gate parameters
    micro_weights: Array2<f32>,
    micro_bias: Array1<f32>,
    meso_weights: Array2<f32>,
    meso_bias: Array1<f32>,
    macro_weights: Array2<f32>,
    macro_bias: Array1<f32>,
    
    // Fusion parameters
    fusion_weights: Array1<f32>, // [alpha, beta, gamma]
    
    // Dimensions
    input_size: usize,
    hidden_size: usize,
    micro_size: usize,
    meso_size: usize,
    macro_size: usize,
    chunk_size: usize,
    
    // Chunk buffer untuk meso level
    chunk_buffer: Vec<ArrayD<f32>>,
    current_chunk_pos: usize,
}

impl TemporalGatingHierarchy {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        micro_size: usize,
        meso_size: usize,
        macro_size: usize,
        chunk_size: usize,
    ) -> DLResult<Self> {
        // Initialize weights dengan Xavier initialization
        // All gates expect concatenated input: [hidden_state + input] = [hidden_size + input_size]
        // All gates output hidden_size for consistent fusion
        let concat_input_size = hidden_size + input_size;
        
        let micro_weights = Self::xavier_init(concat_input_size, hidden_size);
        let micro_bias = Array1::zeros(hidden_size);
        
        let meso_weights = Self::xavier_init(concat_input_size, hidden_size);
        let meso_bias = Array1::zeros(hidden_size);
        
        let macro_weights = Self::xavier_init(concat_input_size, hidden_size);
        let macro_bias = Array1::zeros(hidden_size);
        
        // Fusion weights - initialize ke uniform distribution
        let fusion_weights = Array1::from_vec(vec![1.0/3.0, 1.0/3.0, 1.0/3.0]);
        
        Ok(Self {
            micro_weights,
            micro_bias,
            meso_weights,
            meso_bias,
            macro_weights,
            macro_bias,
            fusion_weights,
            input_size,
            hidden_size,
            micro_size,
            meso_size,
            macro_size,
            chunk_size,
            chunk_buffer: Vec::new(),
            current_chunk_pos: 0,
        })
    }
    
    /// Xavier initialization untuk weights
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            (rand::random::<f32>() * 2.0 - 1.0) * limit
        })
    }
    
    /// Process chunk untuk meso level
    fn process_chunk(&self, chunk: &[ArrayD<f32>]) -> DLResult<ArrayD<f32>> {
        if chunk.is_empty() {
            return Ok(ArrayD::zeros(vec![self.meso_size]));
        }
        
        // Pool chunk - simple average pooling
        let mut pooled = Array1::zeros(self.hidden_size);
        for state in chunk {
            let flat = state.as_slice().unwrap();
            for (i, &val) in flat.iter().enumerate().take(self.hidden_size) {
                pooled[i] += val;
            }
        }
        
        for val in pooled.iter_mut() {
            *val /= chunk.len() as f32;
        }
        
        Ok(pooled.into_dyn())
    }
    
    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Apply sigmoid ke array
    fn sigmoid_array(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        for val in arr.iter_mut() {
            *val = self.sigmoid(*val);
        }
        arr
    }
    
    /// Tanh activation
    fn tanh_array(&self, mut arr: ArrayD<f32>) -> ArrayD<f32> {
        for val in arr.iter_mut() {
            *val = val.tanh();
        }
        arr
    }
    
    /// Concatenate input dan hidden state
    fn concatenate(&self, input: &ArrayD<f32>, hidden: &ArrayD<f32>) -> ArrayD<f32> {
        let input_flat = input.as_slice().unwrap();
        let hidden_flat = hidden.as_slice().unwrap();
        
        let mut concatenated = Vec::with_capacity(input_flat.len() + hidden_flat.len());
        concatenated.extend_from_slice(hidden_flat); // Hidden state first
        concatenated.extend_from_slice(input_flat);  // Then input
        
        Array1::from_vec(concatenated).into_dyn()
    }
    
    /// Matrix multiplication
    fn matmul(&self, weights: &Array2<f32>, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().unwrap();
        if input_flat.len() != weights.shape()[0] {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![weights.shape()[0]],
                actual: vec![input_flat.len()],
            });
        }
        
        let mut output = vec![0.0; weights.shape()[1]];
        for (i, row) in weights.outer_iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                output[j] += input_flat[i] * weight;
            }
        }
        
        Ok(Array1::from_vec(output).into_dyn())
    }
    
    /// Add bias
    fn add_bias(&self, output: &mut ArrayD<f32>, bias: &Array1<f32>) {
        let output_flat = output.as_slice_mut().unwrap();
        for (i, &b) in bias.iter().enumerate().take(output_flat.len()) {
            output_flat[i] += b;
        }
    }
    
    /// Update fusion weights dengan softmax
    fn update_fusion_weights(&mut self, alpha: f32, beta: f32, gamma: f32) {
        let weights = vec![alpha, beta, gamma];
        let softmax_weights: Vec<f32> = weights.iter()
            .map(|&w| w.exp())
            .collect();
        
        let sum: f32 = softmax_weights.iter().sum();
        let normalized: Vec<f32> = softmax_weights.iter()
            .map(|&w| w / sum)
            .collect();
        
        self.fusion_weights = Array1::from_vec(normalized);
    }
    
    /// Get current fusion weights
    pub fn get_fusion_weights(&self) -> (f32, f32, f32) {
        let weights = self.fusion_weights.as_slice().unwrap();
        (weights[0], weights[1], weights[2])
    }
}

impl HierarchicalGating for TemporalGatingHierarchy {
    fn process_hierarchical(&self, 
        input: &ArrayD<f32>,
        hidden_state: &ArrayD<f32>,
        chunk_context: &ArrayD<f32>,
        episodic_memory: &ArrayD<f32>
    ) -> DLResult<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        
        // Micro Gate: local dependencies
        let micro_input = self.concatenate(input, hidden_state);
        let micro_linear = self.matmul(&self.micro_weights, &micro_input)?;
        let mut micro_output = micro_linear;
        self.add_bias(&mut micro_output, &self.micro_bias);
        
        // Apply sigmoid gate dan tanh activation
        let micro_gate = self.sigmoid_array(micro_output.clone());
        let micro_activation = self.tanh_array(micro_output);
        let micro_final = &micro_gate * &micro_activation;
        
        // Meso Gate: chunk-level dependencies  
        let meso_input = self.concatenate(input, chunk_context);
        let meso_linear = self.matmul(&self.meso_weights, &meso_input)?;
        let mut meso_output = meso_linear;
        self.add_bias(&mut meso_output, &self.meso_bias);
        
        let meso_gate = self.sigmoid_array(meso_output.clone());
        let meso_activation = self.tanh_array(meso_output);
        let meso_final = &meso_gate * &meso_activation;
        
        // Macro Gate: episodic memory dependencies
        let macro_input = self.concatenate(input, episodic_memory);
        let macro_linear = self.matmul(&self.macro_weights, &macro_input)?;
        let mut macro_output = macro_linear;
        self.add_bias(&mut macro_output, &self.macro_bias);
        
        let macro_gate = self.sigmoid_array(macro_output.clone());
        let macro_activation = self.tanh_array(macro_output);
        let macro_final = &macro_gate * &macro_activation;
        
        Ok((micro_final, meso_final, macro_final))
    }
    
    fn fuse_hierarchical(&self, 
        micro: &ArrayD<f32>,
        meso: &ArrayD<f32>, 
        macro_out: &ArrayD<f32>,
        weights: (f32, f32, f32)
    ) -> DLResult<ArrayD<f32>> {
        
        let (alpha, beta, gamma) = weights;
        
        // Ensure weights sum to 1
        let total = alpha + beta + gamma;
        let alpha_norm = alpha / total;
        let beta_norm = beta / total;
        let gamma_norm = gamma / total;
        
        // Weighted combination
        let mut fused = ArrayD::zeros(micro.shape());
        let micro_flat = micro.as_slice().unwrap();
        let meso_flat = meso.as_slice().unwrap();
        let macro_flat = macro_out.as_slice().unwrap();
        let fused_flat = fused.as_slice_mut().unwrap();
        
        for i in 0..fused_flat.len() {
            fused_flat[i] = alpha_norm * micro_flat[i] + 
                           beta_norm * meso_flat[i] + 
                           gamma_norm * macro_flat[i];
        }
        
        Ok(fused)
    }
}

/// Methods untuk chunk management
impl TemporalGatingHierarchy {
    /// Add hidden state ke chunk buffer
    pub fn add_to_chunk(&mut self, hidden_state: ArrayD<f32>) -> DLResult<()> {
        self.chunk_buffer.push(hidden_state);
        self.current_chunk_pos += 1;
        
        // Jika chunk buffer penuh, process dan reset
        if self.current_chunk_pos >= self.chunk_size {
            self.current_chunk_pos = 0;
        }
        
        Ok(())
    }
    
    /// Get current chunk context
    pub fn get_chunk_context(&self) -> DLResult<ArrayD<f32>> {
        if self.chunk_buffer.is_empty() {
            return Ok(ArrayD::zeros(vec![self.hidden_size]));
        }
        
        self.process_chunk(&self.chunk_buffer)
    }
    
    /// Reset chunk buffer
    pub fn reset_chunk(&mut self) {
        self.chunk_buffer.clear();
        self.current_chunk_pos = 0;
    }
    
    /// Get chunk statistics
    pub fn get_chunk_stats(&self) -> (usize, usize) {
        (self.chunk_buffer.len(), self.chunk_size)
    }
}

impl crate::traits::Forward for TemporalGatingHierarchy {
    type Input = ArrayD<f32>;
    type Output = (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>);
    
    fn forward(&self, input: &Self::Input) -> DLResult<Self::Output> {
        // Create dummy contexts for hierarchical processing
        let hidden_state = ArrayD::zeros(vec![self.hidden_size]);
        let chunk_context = ArrayD::zeros(vec![self.hidden_size]);
        let episodic_memory = ArrayD::zeros(vec![self.hidden_size]);
        
        self.process_hierarchical(input, &hidden_state, &chunk_context, &episodic_memory)
    }
}
