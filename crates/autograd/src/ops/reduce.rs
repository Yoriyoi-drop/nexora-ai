use ndarray::ArrayD;

use super::super::tensor::Tensor;

pub fn sum(input: &Tensor) -> Tensor {
    let data = input.data();
    let result = ArrayD::from_elem(vec![1], data.sum());

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let orig_shape = data.shape().to_vec();
    let shape_saved = ArrayD::from_shape_vec(
        vec![orig_shape.len()],
        orig_shape.iter().map(|&x| x as f32).collect(),
    ).expect("shape data fits vector");

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![shape_saved],
        Box::new(|grad, saved| {
            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
            let grad_val = grad.iter().copied().next().unwrap_or(1.0);
            vec![ArrayD::from_elem(orig_shape, grad_val)]
        }),
    )
}

pub fn mean(input: &Tensor) -> Tensor {
    let data = input.data();
    let numel = data.len() as f32;
    let result = ArrayD::from_elem(vec![1], data.sum() / numel);

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let orig_shape = data.shape().to_vec();
    let shape_saved = ArrayD::from_shape_vec(
        vec![orig_shape.len()],
        orig_shape.iter().map(|&x| x as f32).collect(),
    ).expect("shape data fits vector");

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![shape_saved, ArrayD::from_elem(vec![1], numel)],
        Box::new(|grad, saved| {
            let shape_data: Vec<f32> = saved[0].iter().copied().collect();
            let orig_shape: Vec<usize> = shape_data.iter().map(|&x| x as usize).collect();
            let n = saved[1].iter().copied().next().unwrap_or(1.0);
            let grad_val = grad.iter().copied().next().unwrap_or(1.0);
            vec![ArrayD::from_elem(orig_shape, grad_val / n)]
        }),
    )
}
