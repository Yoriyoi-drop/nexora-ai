//! Fused Operations untuk STAR-X Performance Optimization
//!
//! Menggabungkan multiple tensor operations menjadi single kernel call
//! untuk mengurangi memory bandwidth dan cache overhead:
//! - Fused Linear + Activation
//! - Fused Attention + Softmax
//! - Fused Matrix Multiplication + Bias
//! - Fused Element-wise Operations

use crate::{DLResult, DeepLearningError};
use crate::star_x::tensor_pool::PooledTensor1D;
use ndarray::{ArrayD, Array1, Array2, ArrayView, ArrayViewMut, Zip, s};
use std::arch::x86_64::*;

/// Fused Linear + Activation operations
pub struct FusedLinearActivation {
    weights: Array2<f32>,
    bias: Array1<f32>,
    activation_type: ActivationType,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
}

impl FusedLinearActivation {
    pub fn new(
        weights: Array2<f32>,
        bias: Array1<f32>,
        activation_type: ActivationType,
    ) -> DLResult<Self> {
        if weights.shape()[1] != bias.shape()[0] {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![weights.shape()[1]],
                actual: vec![bias.shape()[0]],
            });
        }

        Ok(Self {
            weights,
            bias,
            activation_type,
        })
    }

    /// Fused forward pass: y = activation(Wx + b)
    pub fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()?;
        let input_size = input_view.len();
        
        if input_size != self.weights.shape()[0] {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![self.weights.shape()[0]],
                actual: vec![input_size],
            });
        }

        // Use pooled tensor untuk output
        let mut pooled_output = PooledTensor1D::new(self.weights.shape()[1])?;
        let output = pooled_output.get_mut();

        // Perform fused operation
        self.fused_linear_activation_impl(input_view, output.view_mut())?;

        Ok(output.clone().into_dyn())
    }

    #[target_feature(enable = "fma")]
    #[target_feature(enable = "avx2")]
    unsafe fn fused_linear_activation_impl_avx2(
        &self,
        input: ArrayView<f32, ndarray::Ix1>,
        mut output: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        let input_slice = input.as_slice().unwrap();
        let output_slice = output.as_slice_mut().unwrap();
        let weights_slice = self.weights.as_slice().unwrap();
        let bias_slice = self.bias.as_slice().unwrap();

        let output_dim = self.weights.shape()[1];
        let input_dim = self.weights.shape()[0];

        for out_idx in 0..output_dim {
            let mut sum = bias_slice[out_idx];
            
            // Vectorized matrix multiplication
            let mut i = 0;
            while i + 8 <= input_dim {
                let weights_vec = _mm256_loadu_ps(weights_slice.as_ptr().add(out_idx * input_dim + i));
                let input_vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                let product = _mm256_mul_ps(weights_vec, input_vec);
                sum += horizontal_sum_avx2(product);
                i += 8;
            }

            // Handle remaining elements
            for i in i..input_dim {
                sum += weights_slice[out_idx * input_dim + i] * input_slice[i];
            }

            // Apply activation
            output_slice[out_idx] = match self.activation_type {
                ActivationType::ReLU => if sum > 0.0 { sum } else { 0.0 },
                ActivationType::GELU => self.gelu_approx(sum),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                ActivationType::Tanh => sum.tanh(),
                ActivationType::Swish => sum * (1.0 / (1.0 + (-sum).exp())),
            };
        }

        Ok(())
    }

    fn fused_linear_activation_impl(
        &self,
        input: ArrayView<f32, ndarray::Ix1>,
        mut output: ArrayViewMut<f32, ndarray::Ix1>,
    ) -> DLResult<()> {
        // Check if AVX2 is available
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return self.fused_linear_activation_impl_avx2(input, output);
            }
        }

        // Fallback implementation
        let output_dim = self.weights.shape()[1];
        let input_dim = self.weights.shape()[0];

        for out_idx in 0..output_dim {
            let mut sum = self.bias[out_idx];
            
            for in_idx in 0..input_dim {
                sum += self.weights[[in_idx, out_idx]] * input[in_idx];
            }

            output[out_idx] = match self.activation_type {
                ActivationType::ReLU => if sum > 0.0 { sum } else { 0.0 },
                ActivationType::GELU => self.gelu_approx(sum),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                ActivationType::Tanh => sum.tanh(),
                ActivationType::Swish => sum * (1.0 / (1.0 + (-sum).exp())),
            };
        }

        Ok(())
    }

    #[inline]
    fn gelu_approx(&self, x: f32) -> f32 {
        // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = 0.7978845608_f32;
        let coeff = 0.044715_f32;
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x * x * x)).tanh())
    }
}

/// Fused Attention + Softmax operations
#[derive(Debug, Clone)]
pub struct FusedAttentionSoftmax {
    query_weights: Array2<f32>,
    key_weights: Array2<f32>,
    value_weights: Array2<f32>,
    output_weights: Array2<f32>,
    head_dim: usize,
    num_heads: usize,
}

impl FusedAttentionSoftmax {
    pub fn new(
        query_weights: Array2<f32>,
        key_weights: Array2<f32>,
        value_weights: Array2<f32>,
        output_weights: Array2<f32>,
        head_dim: usize,
        num_heads: usize,
    ) -> DLResult<Self> {
        // Validate dimensions
        let hidden_dim = query_weights.shape()[0];
        if key_weights.shape()[0] != hidden_dim || 
           value_weights.shape()[0] != hidden_dim ||
           output_weights.shape()[1] != hidden_dim {
            return Err(DeepLearningError::ShapeMismatch {
                expected: vec![hidden_dim],
                actual: vec![],
            });
        }

        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            head_dim,
            num_heads,
        })
    }

    /// Fused attention computation with integrated softmax
    pub fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let input_view = input.view().into_dimensionality::<ndarray::Ix1>()?;
        
        // Project to Q, K, V (fused)
        let (q, k, v) = self.fused_qkv_projection(input_view)?;
        
        // Compute attention scores with fused softmax
        let attention_output = self.fused_attention_softmax(&q, &k, &v)?;
        
        // Final projection
        let output = self.final_projection(&attention_output)?;
        
        Ok(output.into_dyn())
    }

    fn fused_qkv_projection(&self, input: ArrayView<f32, ndarray::Ix1>) -> DLResult<(Array1<f32>, Array1<f32>, Array1<f32>)> {
        let hidden_dim = input.len();
        let _head_dim = self.head_dim;
        let _num_heads = self.num_heads;
        
        // Use pooled tensors
        let mut pooled_q = PooledTensor1D::new(hidden_dim)?;
        let mut pooled_k = PooledTensor1D::new(hidden_dim)?;
        let mut pooled_v = PooledTensor1D::new(hidden_dim)?;
        
        let q = pooled_q.get_mut();
        let k = pooled_k.get_mut();
        let v = pooled_v.get_mut();

        // Fused QKV projection
        for i in 0..hidden_dim {
            q[i] = 0.0;
            k[i] = 0.0;
            v[i] = 0.0;
            
            for j in 0..hidden_dim {
                let input_val = input[j];
                q[i] += self.query_weights[[j, i]] * input_val;
                k[i] += self.key_weights[[j, i]] * input_val;
                v[i] += self.value_weights[[j, i]] * input_val;
            }
        }

        Ok((q.clone(), k.clone(), v.clone()))
    }

    fn fused_attention_softmax(
        &self,
        q: &Array1<f32>,
        k: &Array1<f32>,
        v: &Array1<f32>,
    ) -> DLResult<Array1<f32>> {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let hidden_dim = q.len();
        
        let mut pooled_output = PooledTensor1D::new(hidden_dim)?;
        let output = pooled_output.get_mut();

        // Process each head
        for head in 0..num_heads {
            let start_idx = head * head_dim;
            let end_idx = start_idx + head_dim;
            
            let q_head = q.slice(s![start_idx..end_idx]);
            let k_head = k.slice(s![start_idx..end_idx]);
            let v_head = v.slice(s![start_idx..end_idx]);
            
            // Compute attention scores (q @ k^T)
            let mut scores = vec![0.0f32; head_dim];
            for i in 0..head_dim {
                for j in 0..head_dim {
                    scores[i] += q_head[i] * k_head[j];
                }
                scores[i] /= (head_dim as f32).sqrt(); // Scale
            }

            // Fused softmax + weighted sum
            let mut output_head = vec![0.0f32; head_dim];
            
            // Compute softmax and apply to values in one pass
            let mut max_score = scores[0];
            for &score in &scores {
                max_score = max_score.max(score);
            }
            
            let mut exp_sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                exp_sum += *score;
            }
            
            // Apply softmax weights to values
            for i in 0..head_dim {
                let softmax_weight = scores[i] / exp_sum;
                for j in 0..head_dim {
                    output_head[j] += softmax_weight * v_head[j];
                }
            }
            
            // Copy to output
            for (i, &val) in output_head.iter().enumerate() {
                output[start_idx + i] = val;
            }
        }

        Ok(output.clone())
    }

    fn final_projection(&self, attention_output: &Array1<f32>) -> DLResult<Array1<f32>> {
        let hidden_dim = attention_output.len();
        let output_dim = self.output_weights.shape()[1];
        
        let mut pooled_output = PooledTensor1D::new(output_dim)?;
        let output = pooled_output.get_mut();

        // Matrix multiplication: attention @ output_weights
        for i in 0..output_dim {
            output[i] = 0.0;
            for j in 0..hidden_dim {
                output[i] += attention_output[j] * self.output_weights[[j, i]];
            }
        }

        Ok(output.clone())
    }
}

/// Fused Element-wise Operations
#[derive(Debug, Clone)]
pub struct FusedElementWise {
    operations: Vec<ElementWiseOp>,
}

#[derive(Debug, Clone)]
pub enum ElementWiseOp {
    Add(f32),
    Mul(f32),
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    Swish,
    Pow(f32),
}

impl FusedElementWise {
    pub fn new(operations: Vec<ElementWiseOp>) -> Self {
        Self { operations }
    }

    /// Apply fused element-wise operations
    pub fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let mut result = input.clone();
        
        for op in &self.operations {
            result = match op {
                ElementWiseOp::Add(val) => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| *out = inp + val);
                    output.clone().into_dyn()
                }
                ElementWiseOp::Mul(val) => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| *out = inp * val);
                    output.clone().into_dyn()
                }
                ElementWiseOp::Relu => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| *out = if inp > 0.0 { inp } else { 0.0 });
                    output.clone().into_dyn()
                }
                ElementWiseOp::Gelu => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| {
                            let x = inp;
                            let sqrt_2_over_pi = 0.7978845608_f32;
                            let coeff = 0.044715_f32;
                            *out = 0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x * x * x)).tanh());
                        });
                    output.clone().into_dyn()
                }
                ElementWiseOp::Sigmoid => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| *out = 1.0 / (1.0 + (-inp).exp()));
                    output.clone().into_dyn()
                }
                ElementWiseOp::Tanh => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| *out = inp.tanh());
                    output.clone().into_dyn()
                }
                ElementWiseOp::Swish => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| {
                            let x = inp;
                            *out = x * (1.0 / (1.0 + (-x).exp()));
                        });
                    output.clone().into_dyn()
                }
                ElementWiseOp::Pow(exp) => {
                    let mut pooled = PooledTensor1D::new(result.len())?;
                    let output = pooled.get_mut();
                    let result_flat = result.as_slice().unwrap();
                    let result_array = Array1::from_vec(result_flat.to_vec());
                    Zip::from(&mut output.view_mut())
                        .and(&result_array.view())
                        .for_each(|out, &inp| *out = inp.powf(*exp));
                    output.clone().into_dyn()
                }
            };
        }

        Ok(result)
    }
}

// Helper function for AVX2 horizontal sum
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Horizontal add within 256-bit vector
    let v128_hi = _mm256_extractf128_ps(v, 1);
    let v128_lo = _mm256_castps256_ps128(v);
    let v128_sum = _mm_add_ps(v128_lo, v128_hi);
    
    // Horizontal add within 128-bit vector
    let v64_hi = _mm_movehl_ps(v128_sum, v128_sum);
    let v64_sum = _mm_add_ps(v128_sum, v64_hi);
    
    let v32_hi = _mm_shuffle_ps(v64_sum, v64_sum, 0x1);
    let v32_sum = _mm_add_ss(v64_sum, v32_hi);
    
    _mm_cvtss_f32(v32_sum)
}
