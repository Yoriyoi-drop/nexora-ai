use ndarray::ArrayD;

use super::super::tensor::Tensor;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();

    let a_view = a_data.view();
    let b_view = b_data.view();

    let a_shape = a_view.shape().to_vec();
    let b_shape = b_view.shape().to_vec();

    assert_eq!(a_shape.len(), 2, "MatMul: a must be 2D");
    assert_eq!(b_shape.len(), 2, "MatMul: b must be 2D");
    assert_eq!(a_shape[1], b_shape[0], "MatMul: inner dims must match");

    let a_mat = a_view.into_dimensionality::<ndarray::Ix2>()
        .expect("MatMul: a must be 2D");
    let b_mat = b_view.into_dimensionality::<ndarray::Ix2>()
        .expect("MatMul: b must be 2D");

    let result = a_mat.dot(&b_mat).into_dyn();

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let saved = vec![a_data.clone(), b_data.clone()];

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        saved,
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let grad_arr = grad.clone();

            let grad_mat = grad_arr.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("grad must be 2D");

            let a_mat = a_val.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("a must be 2D");

            let b_mat = b_val.view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("b must be 2D");

            let da = grad_mat.dot(&b_mat.t()).into_dyn();
            let db = a_mat.t().dot(&grad_mat).into_dyn();

            vec![da, db]
        }),
    )
}
