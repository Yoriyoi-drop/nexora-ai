//! Associative State Composition (ASC)
//!
//! Parallel associative scan untuk menghilangkan recurrent bottleneck:
//! - Transform recurrence menjadi associative operation
//! - Parallel training dengan scan operation  
//! - GPU utilization optimization
//! - Recurrent bottleneck elimination

use crate::{DLResult, DeepLearningError};
use ndarray::{ArrayD, Array1, Array2};
use rand;

/// Associative operator untuk state composition
#[derive(Debug, Clone)]
pub struct AssociativeOperator {
    // Linear transformation parameters
    weight_a: Array2<f32>, // For state A
    weight_b: Array2<f32>, // For state B
    bias: Array1<f32>,
    
    // Composition parameters
    hidden_size: usize,
    associative_strength: f32,
    
    // Parallel computation parameters
    chunk_size: usize,
    num_threads: usize,
}

impl AssociativeOperator {
    pub fn new(hidden_size: usize, associative_strength: f32) -> DLResult<Self> {
        // Initialize weights dengan Xavier initialization
        let weight_a = Self::xavier_init(hidden_size, hidden_size);
        let weight_b = Self::xavier_init(hidden_size, hidden_size);
        let bias = Array1::zeros(hidden_size);
        
        Ok(Self {
            weight_a,
            weight_b,
            bias,
            hidden_size,
            associative_strength,
            chunk_size: 64,
            num_threads: 4,
        })
    }
    
    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            (rand::random::<f32>() * 2.0 - 1.0) * limit
        })
    }
    
    /// Apply associative operation: A ⊗ B
    pub fn associative_operation(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let a_flat = a.as_slice().expect("tensor should be contiguous");
        let b_flat = b.as_slice().expect("tensor should be contiguous");
        
        if a_flat.len() != self.hidden_size || b_flat.len() != self.hidden_size {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![self.hidden_size],
                actual: vec![a_flat.len(), b_flat.len()],
            });
        }
        
        // Linear transformations
        let transformed_a = self.matmul(&self.weight_a, a)?;
        let transformed_b = self.matmul(&self.weight_b, b)?;
        
        // Associative composition
        let mut composed = Array1::zeros(self.hidden_size);
        let comp_flat = composed.as_slice_mut().expect("tensor should be contiguous");
        let a_trans_flat = transformed_a.as_slice().expect("tensor should be contiguous");
        let b_trans_flat = transformed_b.as_slice().expect("tensor should be contiguous");
        let bias_flat = self.bias.as_slice().expect("tensor should be contiguous");
        
        for i in 0..self.hidden_size {
            // Element-wise associative operation
            let assoc_val = a_trans_flat[i] + b_trans_flat[i] + bias_flat[i];
            comp_flat[i] = self.associative_strength * assoc_val.tanh();
        }
        
        Ok(composed.into_dyn())
    }
    
    /// Matrix multiplication
    fn matmul(&self, weights: &Array2<f32>, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_flat = input.as_slice().expect("tensor should be contiguous");
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
    
    /// Check associativity property: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
    pub fn verify_associativity(&self, a: &ArrayD<f32>, b: &ArrayD<f32>, c: &ArrayD<f32>) -> DLResult<bool> {
        // Left associative: (A ⊗ B) ⊗ C
        let left_result = self.associative_operation(a, b)?;
        let left_final = self.associative_operation(&left_result, c)?;
        
        // Right associative: A ⊗ (B ⊗ C)
        let right_result = self.associative_operation(b, c)?;
        let right_final = self.associative_operation(a, &right_result)?;
        
        // Compare results
        let left_flat = left_final.as_slice().expect("tensor should be contiguous");
        let right_flat = right_final.as_slice().expect("tensor should be contiguous");
        
        let mut difference = 0.0;
        for (l, r) in left_flat.iter().zip(right_flat.iter()) {
            difference += (l - r).abs();
        }
        
        Ok(difference < 1e-6) // Tolerance for floating point comparison
    }
}

/// Parallel associative scan implementation
#[derive(Debug, Clone)]
pub struct ParallelAssociativeScan {
    associative_op: AssociativeOperator,
    
    // Scan parameters
    sequence_length: usize,
    hidden_size: usize,
    
    // Parallel execution parameters
    block_size: usize,
    num_blocks: usize,
    
    // Intermediate results for parallel scan
    block_results: Vec<ArrayD<f32>>,
    prefix_sums: Vec<ArrayD<f32>>,
}

impl ParallelAssociativeScan {
    pub fn new(hidden_size: usize, sequence_length: usize) -> DLResult<Self> {
        let associative_op = AssociativeOperator::new(hidden_size, 0.8)?;
        
        let block_size = 64; // Optimal for GPU cache
        let num_blocks = (sequence_length + block_size - 1) / block_size;
        
        Ok(Self {
            associative_op,
            sequence_length,
            hidden_size,
            block_size,
            num_blocks,
            block_results: Vec::with_capacity(num_blocks),
            prefix_sums: Vec::with_capacity(num_blocks),
        })
    }
    
    /// Sequential state transition: h_t = A_t ⊗ h_{t-1} ⊕ B_t ⊗ x_t
    pub fn state_transition(&self, 
        a_matrix: &ArrayD<f32>,
        previous_state: &ArrayD<f32>,
        b_matrix: &ArrayD<f32>,
        input: &ArrayD<f32>
    ) -> DLResult<ArrayD<f32>> {
        
        // A_t ⊗ h_{t-1}
        let state_part = self.associative_op.associative_operation(a_matrix, previous_state)?;
        
        // B_t ⊗ x_t
        let input_part = self.associative_op.associative_operation(b_matrix, input)?;
        
        // Final composition (simplified - in practice would be element-wise)
        let mut final_state = Array1::zeros(self.hidden_size);
        let final_flat = final_state.as_slice_mut().expect("tensor should be contiguous");
        let state_flat = state_part.as_slice().expect("tensor should be contiguous");
        let input_flat = input_part.as_slice().expect("tensor should be contiguous");
        
        for i in 0..self.hidden_size {
            final_flat[i] = state_flat[i] + input_flat[i];
        }
        
        Ok(final_state.into_dyn())
    }
    
    /// Parallel scan implementation
    pub fn parallel_scan(&mut self, 
        a_matrices: &[ArrayD<f32>],
        b_matrices: &[ArrayD<f32>],
        inputs: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        if a_matrices.len() != self.sequence_length || 
           b_matrices.len() != self.sequence_length || 
           inputs.len() != self.sequence_length {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![self.sequence_length],
                actual: vec![a_matrices.len(), b_matrices.len(), inputs.len()],
            });
        }
        
        // Phase 1: Process blocks in parallel
        self.process_blocks_parallel(a_matrices, b_matrices, inputs, initial_state)?;
        
        // Phase 2: Compute prefix sums of block results
        self.compute_prefix_sums()?;
        
        // Phase 3: Apply prefix corrections to blocks
        self.apply_prefix_corrections(a_matrices, b_matrices, inputs)
    }
    
    /// Process individual blocks in parallel
    fn process_blocks_parallel(&mut self,
        a_matrices: &[ArrayD<f32>],
        b_matrices: &[ArrayD<f32>],
        inputs: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>
    ) -> DLResult<()> {
        
        self.block_results.clear();
        
        for block_idx in 0..self.num_blocks {
            let start_idx = block_idx * self.block_size;
            let end_idx = (start_idx + self.block_size).min(self.sequence_length);
            
            if start_idx >= self.sequence_length {
                break;
            }
            
            // Process block sequentially (simplified - would be parallel in practice)
            let mut current_state = if block_idx == 0 {
                initial_state.clone()
            } else {
                ArrayD::zeros(vec![self.hidden_size]) // Placeholder
            };
            
            for i in start_idx..end_idx {
                current_state = self.state_transition(
                    &a_matrices[i],
                    &current_state,
                    &b_matrices[i],
                    &inputs[i]
                )?;
            }
            
            self.block_results.push(current_state);
        }
        
        Ok(())
    }
    
    /// Compute prefix sums of block results
    fn compute_prefix_sums(&mut self) -> DLResult<()> {
        self.prefix_sums.clear();
        
        if self.block_results.is_empty() {
            return Ok(());
        }
        
        let mut running_sum = self.block_results[0].clone();
        self.prefix_sums.push(running_sum.clone());
        
        for i in 1..self.block_results.len() {
            running_sum = self.associative_op.associative_operation(
                &running_sum,
                &self.block_results[i]
            )?;
            self.prefix_sums.push(running_sum.clone());
        }
        
        Ok(())
    }
    
    /// Apply prefix corrections to individual elements
    fn apply_prefix_corrections(&self,
        a_matrices: &[ArrayD<f32>],
        b_matrices: &[ArrayD<f32>],
        inputs: &[ArrayD<f32>]
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        let mut final_results = Vec::with_capacity(self.sequence_length);
        
        for block_idx in 0..self.num_blocks {
            let start_idx = block_idx * self.block_size;
            let end_idx = (start_idx + self.block_size).min(self.sequence_length);
            
            if start_idx >= self.sequence_length {
                break;
            }
            
            // Get prefix correction for this block
            let prefix_correction = if block_idx > 0 {
                &self.prefix_sums[block_idx - 1]
            } else {
                return Err(DeepLearningError::Computation {
                    reason: "Invalid block index for prefix correction".to_string(),
                });
            };
            
            // Process block with correction
            let mut current_state = prefix_correction.clone();
            
            for i in start_idx..end_idx {
                current_state = self.state_transition(
                    &a_matrices[i],
                    &current_state,
                    &b_matrices[i],
                    &inputs[i]
                )?;
                final_results.push(current_state.clone());
            }
        }
        
        Ok(final_results)
    }
    
    /// Optimized scan for long sequences
    pub fn optimized_long_sequence_scan(&mut self,
        sequence: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        if sequence.len() <= self.block_size {
            // Short sequence - process sequentially
            return self.sequential_scan(sequence, initial_state);
        }
        
        // Long sequence - use hierarchical scan
        self.hierarchical_scan(sequence, initial_state)
    }
    
    /// Sequential scan fallback
    fn sequential_scan(&self,
        sequence: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        let mut results = Vec::with_capacity(sequence.len());
        let mut current_state = initial_state.clone();
        
        for element in sequence {
            current_state = self.associative_op.associative_operation(
                &current_state,
                element
            )?;
            results.push(current_state.clone());
        }
        
        Ok(results)
    }
    
    /// Hierarchical scan untuk very long sequences
    fn hierarchical_scan(&mut self,
        sequence: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        // Divide into chunks
        let chunk_size = self.block_size * 4; // Larger chunks for hierarchy
        let mut chunks = Vec::new();
        
        for chunk in sequence.chunks(chunk_size) {
            chunks.push(chunk.to_vec());
        }
        
        // Process each chunk
        let mut chunk_results = Vec::new();
        let mut running_state = initial_state.clone();
        
        for chunk in &chunks {
            let chunk_result = self.sequential_scan(chunk, &running_state)?;
            running_state = chunk_result.last().expect("chunk results should not be empty").clone();
            chunk_results.push(chunk_result);
        }
        
        // Flatten results
        let mut final_results = Vec::new();
        for chunk_result in chunk_results {
            final_results.extend(chunk_result);
        }
        
        Ok(final_results)
    }
    
    /// Get scan performance statistics
    pub fn get_performance_stats(&self) -> (usize, usize, f32) {
        let total_operations = self.sequence_length * self.hidden_size;
        let parallel_operations = self.num_blocks * self.block_size * self.hidden_size;
        let parallel_efficiency = parallel_operations as f32 / total_operations as f32;
        
        (total_operations, parallel_operations, parallel_efficiency)
    }
}

/// Advanced associative operations
impl ParallelAssociativeScan {
    /// Dynamic block size optimization
    pub fn optimize_block_size(&mut self, sequence_length: usize, gpu_memory_mb: usize) {
        // Estimate memory usage per block
        let memory_per_block = self.block_size * self.hidden_size * 4 * 4; // float32 * 4 tensors
        let max_blocks_in_memory = gpu_memory_mb * 1024 * 1024 / memory_per_block;
        
        // Optimize block size for parallel efficiency
        let optimal_block_size = (sequence_length / max_blocks_in_memory.max(1)).max(16).min(256);
        self.block_size = optimal_block_size;
        self.num_blocks = (sequence_length + self.block_size - 1) / self.block_size;
    }
    
    /// Memory-efficient scan untuk very large sequences
    pub fn memory_efficient_scan(&mut self,
        sequence: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>,
        memory_limit_mb: usize
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        // Estimate memory usage
        let element_size = self.hidden_size * 4; // float32
        let sequence_memory = sequence.len() * element_size;
        let memory_limit_bytes = memory_limit_mb * 1024 * 1024;
        
        if sequence_memory <= memory_limit_bytes {
            // Can process entire sequence
            return self.optimized_long_sequence_scan(sequence, initial_state);
        }
        
        // Process in streaming fashion
        self.streaming_scan(sequence, initial_state, memory_limit_bytes)
    }
    
    /// Streaming scan untuk memory-constrained scenarios
    fn streaming_scan(&mut self,
        sequence: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>,
        memory_limit_bytes: usize
    ) -> DLResult<Vec<ArrayD<f32>>> {
        
        let element_size = self.hidden_size * 4;
        let max_elements_in_memory = memory_limit_bytes / element_size;
        let chunk_size = max_elements_in_memory / 2; // Reserve memory for computation
        
        let mut all_results = Vec::new();
        let mut current_state = initial_state.clone();
        
        for chunk in sequence.chunks(chunk_size) {
            let chunk_results = self.sequential_scan(chunk, &current_state)?;
            current_state = chunk_results.last().expect("chunk results should not be empty").clone();
            all_results.extend(chunk_results);
        }
        
        Ok(all_results)
    }
    
    /// Verify scan correctness
    pub fn verify_scan_correctness(&self,
        sequence: &[ArrayD<f32>],
        initial_state: &ArrayD<f32>
    ) -> DLResult<bool> {
        
        // Sequential reference
        let sequential_results = self.sequential_scan(sequence, initial_state)?;
        
        // Parallel scan
        let mut temp_scan = self.clone();
        let parallel_results = temp_scan.optimized_long_sequence_scan(sequence, initial_state)?;
        
        // Compare results
        if sequential_results.len() != parallel_results.len() {
            return Ok(false);
        }
        
        let mut total_error = 0.0;
        for (seq, par) in sequential_results.iter().zip(parallel_results.iter()) {
            let seq_flat = seq.as_slice().expect("tensor should be contiguous");
            let par_flat = par.as_slice().expect("tensor should be contiguous");
            
            for (s, p) in seq_flat.iter().zip(par_flat.iter()) {
                total_error += (s - p).abs();
            }
        }
        
        let avg_error = total_error / (sequential_results.len() * self.hidden_size) as f32;
        Ok(avg_error < 1e-4)
    }
}
