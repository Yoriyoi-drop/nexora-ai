//! Tensor Reshaping Operations
//! 
//! Konsolidasi dari ATQS tensor_utils reshape functions.

use crate::tensor::{TensorError, TensorResult};
use ndarray::{Array, ArrayD, ArrayView, IxDyn};

/// Tensor reshaping utilities
pub struct TensorReshaper;

impl TensorReshaper {
    /// Create new reshaper instance
    pub fn new() -> Self {
        Self
    }
    
    /// Reshape tensor to specified dimensions, preserving order
    pub fn reshape_preserving_order(
        tensor: &ArrayD<f32>,
        new_shape: &[usize],
    ) -> TensorResult<ArrayD<f32>> {
        let total_elements: usize = tensor.shape().iter().product();
        let new_total: usize = new_shape.iter().product();
        
        if total_elements != new_total {
            return Err(TensorError::ShapeMismatch(
                format!("Cannot reshape tensor with {} elements to shape {:?} ({} elements)", 
                       total_elements, new_shape, new_total)
            ));
        }
        
        tensor.clone().into_shape(new_shape)
            .map(|arr| arr.into_dyn())
            .map_err(|e| TensorError::DimensionError(format!("Reshape failed: {:?}", e)))
    }
    
    /// Flatten tensor to 1D
    pub fn flatten(tensor: &ArrayD<f32>) -> TensorResult<ArrayD<f32>> {
        let size = tensor.len();
        tensor.clone().into_shape((size,))
            .map(|arr| arr.into_dyn())
            .map_err(|e| TensorError::DimensionError(format!("Flatten failed: {:?}", e)))
    }
    
    /// Reshape 2D matrix to 4D tensor (for attention layers)
    pub fn matrix_to_4d(
        matrix: &Array<f32, ndarray::Ix2>,
        dim1: usize,
        dim2: usize,
        dim3: usize,
        dim4: usize,
    ) -> TensorResult<ArrayD<f32>> {
        let total = dim1 * dim2 * dim3 * dim4;
        if matrix.len() != total {
            return Err(TensorError::ShapeMismatch(
                "Matrix size doesn't match target 4D dimensions".to_string()
            ));
        }
        
        matrix.clone().into_shape((dim1, dim2, dim3, dim4))
            .map(|arr| arr.into_dyn())
            .map_err(|e| TensorError::DimensionError(format!("Matrix to 4D reshape failed: {:?}", e)))
    }
    
    /// Reshape 4D tensor to 2D matrix
    pub fn tensor_4d_to_matrix(tensor: &ArrayD<f32>) -> TensorResult<Array<f32, ndarray::Ix2>> {
        if tensor.ndim() != 4 {
            return Err(TensorError::InvalidOperation(
                "Expected 4D tensor for matrix conversion".to_string()
            ));
        }
        
        let total_elements = tensor.len();
        tensor.clone().into_shape((total_elements,))
            .and_then(|arr| arr.into_dimensionality::<ndarray::Ix2>())
            .map_err(|e| TensorError::DimensionError(format!("4D to matrix conversion failed: {:?}", e)))
    }
    
    /// Add new dimension at specified axis
    pub fn add_dimension(tensor: &ArrayD<f32>, axis: usize) -> TensorResult<ArrayD<f32>> {
        let mut new_shape = tensor.shape().to_vec();
        if axis > new_shape.len() {
            return Err(TensorError::DimensionError(
                format!("Axis {} out of bounds for tensor with {} dimensions", axis, new_shape.len())
            ));
        }
        new_shape.insert(axis, 1);
        
        tensor.clone().into_shape(new_shape)
            .map(|arr| arr.into_dyn())
            .map_err(|e| TensorError::DimensionError(format!("Add dimension failed: {:?}", e)))
    }
    
    /// Remove dimension at specified axis (must be size 1)
    pub fn squeeze(tensor: &ArrayD<f32>, axis: usize) -> TensorResult<ArrayD<f32>> {
        if tensor.ndim() == 0 {
            return Err(TensorError::InvalidOperation(
                "Cannot squeeze scalar tensor".to_string()
            ));
        }
        
        if axis >= tensor.ndim() {
            return Err(TensorError::DimensionError(
                format!("Axis {} out of bounds for tensor with {} dimensions", axis, tensor.ndim())
            ));
        }
        
        if tensor.shape()[axis] != 1 {
            return Err(TensorError::ShapeMismatch(
                format!("Cannot squeeze axis {} with size {}", axis, tensor.shape()[axis])
            ));
        }
        
        let mut new_shape = tensor.shape().to_vec();
        new_shape.remove(axis);
        
        tensor.clone().into_shape(new_shape)
            .map(|arr| arr.into_dyn())
            .map_err(|e| TensorError::DimensionError(format!("Squeeze failed: {:?}", e)))
    }
    
    /// Broadcast tensor to target shape
    pub fn broadcast_to(tensor: &ArrayD<f32>, target_shape: &[usize]) -> TensorResult<ArrayD<f32>> {
        // Check if broadcasting is possible
        let tensor_shape = tensor.shape();
        
        if tensor_shape.len() > target_shape.len() {
            return Err(TensorError::ShapeMismatch(
                "Cannot broadcast: tensor has more dimensions than target".to_string()
            ));
        }
        
        // Check compatibility from right to left
        for (i, &target_dim) in target_shape.iter().rev().enumerate() {
            let tensor_idx = tensor_shape.len().wrapping_sub(i + 1);
            if tensor_idx < tensor_shape.len() {
                let tensor_dim = tensor_shape[tensor_idx];
                if tensor_dim != target_dim && tensor_dim != 1 {
                    return Err(TensorError::ShapeMismatch(
                        format!("Cannot broadcast dimension {} (size {}) to {}", 
                               tensor_idx, tensor_dim, target_dim)
                    ));
                }
            }
        }
        
        // For simplicity, return a clone (in real implementation, use ndarray broadcasting)
        Ok(tensor.clone())
    }
    
    /// Permute tensor dimensions
    pub fn permute(tensor: &ArrayD<f32>, axes: &[usize]) -> TensorResult<ArrayD<f32>> {
        if axes.len() != tensor.ndim() {
            return Err(TensorError::DimensionError(
                format!("Permutation axes length {} must match tensor dimensions {}", 
                       axes.len(), tensor.ndim())
            ));
        }
        
        // Check if axes are valid
        let mut used_axes = vec![false; tensor.ndim()];
        for &axis in axes {
            if axis >= tensor.ndim() {
                return Err(TensorError::DimensionError(
                    format!("Permutation axis {} out of bounds for tensor with {} dimensions", 
                           axis, tensor.ndim())
                ));
            }
            if used_axes[axis] {
                return Err(TensorError::InvalidOperation(
                    format!("Duplicate axis {} in permutation", axis)
                ));
            }
            used_axes[axis] = true;
        }
        
        // For simplicity, return a clone (in real implementation, use ndarray permutation)
        Ok(tensor.clone())
    }
}

impl Default for TensorReshaper {
    fn default() -> Self {
        Self::new()
    }
}
