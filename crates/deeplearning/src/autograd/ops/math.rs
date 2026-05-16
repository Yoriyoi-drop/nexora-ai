use ndarray::ArrayD;

use super::super::broadcast;
use super::super::tensor::Tensor;

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc + &b_bc;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap(),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap(),
        ],
        Box::new(|grad, saved| {
            let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
            let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
            vec![da, db]
        }),
    )
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc - &b_bc;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap(),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap(),
        ],
        Box::new(|grad, saved| {
            let a_shape: Vec<usize> = saved[0].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[1].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(grad, &a_shape);
            let db = broadcast::reduce_grad_for_shape(grad, &b_shape);
            vec![da, -db]
        }),
    )
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc * &b_bc;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            a_bc.clone(),
            b_bc.clone(),
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap(),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap(),
        ],
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let a_shape: Vec<usize> = saved[2].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[3].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(&(grad.clone() * b_val), &a_shape);
            let db = broadcast::reduce_grad_for_shape(&(grad.clone() * a_val), &b_shape);
            vec![da, db]
        }),
    )
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let (a_bc, b_bc, _) = broadcast::broadcast_arrays(&a_data, &b_data);
    let result = &a_bc / &b_bc;

    let requires_grad = a.requires_grad() || b.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let a_shape = a_data.shape().to_vec();
    let b_shape = b_data.shape().to_vec();

    Tensor::with_grad_fn(
        result,
        vec![a.clone(), b.clone()],
        vec![
            a_bc.clone(),
            b_bc.clone(),
            ArrayD::from_shape_vec(vec![a_shape.len()], a_shape.iter().map(|&x| x as f32).collect()).unwrap(),
            ArrayD::from_shape_vec(vec![b_shape.len()], b_shape.iter().map(|&x| x as f32).collect()).unwrap(),
        ],
        Box::new(|grad, saved| {
            let a_val = &saved[0];
            let b_val = &saved[1];
            let a_shape: Vec<usize> = saved[2].iter().map(|&x| x as usize).collect();
            let b_shape: Vec<usize> = saved[3].iter().map(|&x| x as usize).collect();
            let da = broadcast::reduce_grad_for_shape(&(grad.clone() / b_val), &a_shape);
            let db = broadcast::reduce_grad_for_shape(&(grad.clone() * (-a_val) / (b_val * b_val)), &b_shape);
            vec![da, db]
        }),
    )
}

pub fn neg(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let result = -&a_data;

    if !a.requires_grad() {
        return Tensor::new(result);
    }

    Tensor::with_grad_fn(
        result,
        vec![a.clone()],
        vec![],
        Box::new(|grad, _| vec![-grad.clone()]),
    )
}
