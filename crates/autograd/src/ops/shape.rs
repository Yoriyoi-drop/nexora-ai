use ndarray::ArrayD;

use super::super::tensor::Tensor;

pub fn reshape(input: &Tensor, new_shape: &[usize]) -> Tensor {
    let data = input.data();
    let new_len: usize = new_shape.iter().product();
    assert_eq!(data.len(), new_len, "Reshape: total elements must match");

    let result = data.clone().into_shape(new_shape.to_vec())
        .expect("Reshape failed");

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
            vec![grad.clone().into_shape(orig_shape).expect("Reshape backward failed")]
        }),
    )
}

pub fn transpose(input: &Tensor) -> Tensor {
    let data = input.data();
    assert_eq!(data.ndim(), 2, "Transpose: input must be 2D");

    let result = data.view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("Transpose: must be 2D")
        .t()
        .to_owned()
        .into_dyn();

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![],
        Box::new(|grad, _| {
            let grad_mat = grad.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("grad must be 2D");
            vec![grad_mat.t().to_owned().into_dyn()]
        }),
    )
}
