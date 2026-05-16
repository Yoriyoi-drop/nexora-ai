use ndarray::{ArrayD, Axis};

use super::super::tensor::Tensor;

/// NOT YET IMPLEMENTED for requires_grad tensors — use basic concat for now
pub fn cat(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "cat: at least one tensor required");
    let arrays: Vec<ArrayD<f32>> = tensors.iter().map(|t| t.data()).collect();
    let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(Axis(axis), &views)
        .expect("cat failed: shape mismatch");

    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    if !requires_grad {
        return Tensor::new(result);
    }

    let input_shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();

    Tensor::with_grad_fn(
        result,
        tensors.iter().map(|t| (*t).clone()).collect(),
        vec![],
        Box::new(move |grad, _saved| {
            let mut grads = Vec::new();
            let mut offset = 0usize;
            for shape in &input_shapes {
                let cat_dim = shape.len().saturating_sub(1).min(axis);
                let len = shape[cat_dim];
                let mut slices = vec![ndarray::s![..]; shape.len()];
                slices[cat_dim] = ndarray::s![offset..offset + len];

                // Build a properly indexed sub-array
                let mut grad_shape = grad.shape().to_vec();
                grad_shape[cat_dim] = len;
                let g = grad
                    .select(Axis(cat_dim),
                        (offset..offset + len).collect::<Vec<_>>().as_slice())
                    .to_owned()
                    .into_dyn();
                grads.push(g);
                offset += len;
            }
            grads
        }),
    )
}

pub fn stack(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "stack: at least one tensor required");
    let first = tensors[0];
    let mut new_shape = first.shape().to_vec();
    new_shape.insert(axis.min(new_shape.len()), 1);
    let data = first.data().into_shape(new_shape).expect("stack: reshape failed");
    Tensor::new(data)
}
