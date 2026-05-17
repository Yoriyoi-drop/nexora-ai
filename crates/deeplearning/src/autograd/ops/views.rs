use ndarray::{ArrayD, Axis};

use super::super::tensor::Tensor;

/// Concatenate tensors along an axis.
/// Backward: each input gets its slice of the gradient.
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

    let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
    let input_tensors: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let saved_sizes = ArrayD::from_shape_vec(
        vec![dim_sizes.len()],
        dim_sizes.iter().map(|&x| x as f32).collect(),
    ).expect("shape data fits vector");

    Tensor::with_grad_fn(
        result,
        input_tensors,
        vec![saved_sizes],
        Box::new(move |grad, saved| {
            let sizes: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let mut offset = 0usize;
            let mut grads = Vec::with_capacity(sizes.len());
            for &len in &sizes {
                let indices: Vec<usize> = (offset..offset + len).collect();
                let g = grad.select(Axis(axis), &indices).to_owned().into_dyn();
                grads.push(g);
                offset += len;
            }
            grads
        }),
    )
}

/// Stack tensors along a new axis. No grad tracking (placeholder).
pub fn stack(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "stack: at least one tensor required");
    let first = tensors[0];
    let mut new_shape = first.shape().to_vec();
    new_shape.insert(axis.min(new_shape.len()), 1);
    let data = first.data().into_shape(new_shape).expect("stack: reshape failed");
    Tensor::new(data)
}
