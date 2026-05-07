//! Core Tensor Operations
//! 
//! Konsolidasi dari ATQS tensor_ops, CAFFEINE tensor_utils, dan math utilities.

use crate::tensor::{TensorError, TensorResult};
use ndarray::{Array, ArrayD, ArrayView, IxDyn, s};
use std::ops::{Add, Mul, Sub};

/// Basic tensor operations
pub struct TensorOps;

impl TensorOps {
    /// Element-wise addition
    pub fn add(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        if a.shape() != b.shape() {
            return Err(TensorError::ShapeMismatch(
                format!("Cannot add tensors with shapes {:?} and {:?}", a.shape(), b.shape())
            ));
        }
        Ok(a + b)
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        if a.shape() != b.shape() {
            return Err(TensorError::ShapeMismatch(
                format!("Cannot subtract tensors with shapes {:?} and {:?}", a.shape(), b.shape())
            ));
        }
        Ok(a - b)
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        if a.shape() != b.shape() {
            return Err(TensorError::ShapeMismatch(
                format!("Cannot multiply tensors with shapes {:?} and {:?}", a.shape(), b.shape())
            ));
        }
        Ok(a * b)
    }
    
    /// Scalar multiplication
    pub fn scalar_mul(&self, tensor: &ArrayD<f32>, scalar: f32) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x * scalar))
    }
    
    /// Scalar addition
    pub fn scalar_add(&self, tensor: &ArrayD<f32>, scalar: f32) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x + scalar))
    }
    
    /// Matrix multiplication (2D tensors)
    pub fn matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "Matrix multiplication requires 2D tensors".to_string()
            ));
        }
        
        let a_2d = a.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TensorError::DimensionError(format!("Cannot convert first tensor to 2D: {:?}", e)))?;
        let b_2d = b.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TensorError::DimensionError(format!("Cannot convert second tensor to 2D: {:?}", e)))?;
        
        if a_2d.shape()[1] != b_2d.shape()[0] {
            return Err(TensorError::ShapeMismatch(
                format!("Cannot multiply matrices with shapes {:?} and {:?}", a.shape(), b.shape())
            ));
        }
        
        let result = a_2d.dot(&b_2d);
        Ok(result.into_dyn())
    }
    
    /// Transpose tensor (2D)
    pub fn transpose(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        if tensor.ndim() != 2 {
            return Err(TensorError::InvalidOperation(
                "Transpose requires 2D tensor".to_string()
            ));
        }
        
        let tensor_2d = tensor.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TensorError::DimensionError(format!("Cannot convert to 2D: {:?}", e)))?;
        
        let transposed = tensor_2d.t();
        Ok(transposed.into_owned().into_dyn())
    }
    
    /// Sum all elements
    pub fn sum(&self, tensor: &ArrayD<f32>) -> f32 {
        tensor.sum()
    }
    
    /// Mean of all elements
    pub fn mean(&self, tensor: &ArrayD<f32>) -> f32 {
        if tensor.is_empty() {
            0.0
        } else {
            tensor.sum() / tensor.len() as f32
        }
    }
    
    /// Maximum element
    pub fn max(&self, tensor: &ArrayD<f32>) -> f32 {
        if tensor.is_empty() {
            0.0
        } else {
            tensor.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        }
    }
    
    /// Minimum element
    pub fn min(&self, tensor: &ArrayD<f32>) -> f32 {
        if tensor.is_empty() {
            0.0
        } else {
            tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b))
        }
    }
    
    /// Element-wise absolute value
    pub fn abs(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x.abs()))
    }
    
    /// Element-wise square
    pub fn square(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x * x))
    }
    
    /// Element-wise square root
    pub fn sqrt(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x.sqrt()))
    }
    
    /// Element-wise exponential
    pub fn exp(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x.exp()))
    }
    
    /// Element-wise natural logarithm
    pub fn log(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x.ln()))
    }
    
    /// Element-wise sigmoid
    pub fn sigmoid(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }
    
    /// Element-wise ReLU
    pub fn relu(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x.max(0.0)))
    }
    
    /// Element-wise tanh
    pub fn tanh(&self, tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        Ok(tensor.mapv(|x| x.tanh()))
    }
}

impl Default for TensorOps {
    fn default() -> Self {
        Self
    }
}
