use ndarray::ArrayD;

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let a_padded: Vec<usize> = std::iter::repeat(1).take(max_len - a.len()).chain(a.iter().cloned()).collect();
    let b_padded: Vec<usize> = std::iter::repeat(1).take(max_len - b.len()).chain(b.iter().cloned()).collect();
    let mut result = Vec::with_capacity(max_len);

    for (ai, bi) in a_padded.iter().zip(b_padded.iter()) {
        if *ai == *bi {
            result.push(*ai);
        } else if *ai == 1 {
            result.push(*bi);
        } else if *bi == 1 {
            result.push(*ai);
        } else {
            panic!("Broadcast error: shapes {:?} and {:?} incompatible", a, b);
        }
    }
    result
}

pub fn broadcast_arrays(a: &ArrayD<f32>, b: &ArrayD<f32>) -> (ArrayD<f32>, ArrayD<f32>, Vec<usize>) {
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    if a_shape == b_shape {
        return (a.clone(), b.clone(), a_shape);
    }
    let target = broadcast_shapes(&a_shape, &b_shape);
    let a_bc = if a_shape != target {
        a.clone().broadcast(target.as_slice())
            .unwrap_or_else(|| panic!("Cannot broadcast {:?} to {:?}", a_shape, target))
            .to_owned()
    } else {
        a.clone()
    };
    let b_bc = if b_shape != target {
        b.clone().broadcast(target.as_slice())
            .unwrap_or_else(|| panic!("Cannot broadcast {:?} to {:?}", b_shape, target))
            .to_owned()
    } else {
        b.clone()
    };
    (a_bc, b_bc, target)
}

pub fn reduce_grad_for_shape(grad: &ArrayD<f32>, orig_shape: &[usize]) -> ArrayD<f32> {
    if grad.shape() == orig_shape {
        return grad.clone();
    }
    let grad_shape = grad.shape().to_vec();
    let padded: Vec<usize> = std::iter::repeat(1)
        .take(grad_shape.len().saturating_sub(orig_shape.len()))
        .chain(orig_shape.iter().cloned())
        .collect();

    let mut result = grad.clone();
    for axis in (0..padded.len()).rev() {
        if padded[axis] == 1 && grad_shape[axis] > 1 {
            result = result.sum_axis(ndarray::Axis(axis)).into_dyn();
        }
    }

    if result.shape() == orig_shape {
        return result;
    }

    match result.clone().into_shape(orig_shape.to_vec()) {
        Ok(r) => r,
        Err(_) => {
            let flat: Vec<f32> = result.iter().copied().collect();
            let flat_len = flat.len();
            if flat_len == orig_shape.iter().product::<usize>() {
                ArrayD::from_shape_vec(orig_shape.to_vec(), flat).expect("data length matches shape")
            } else {
                ArrayD::from_elem(orig_shape.to_vec(), flat[0])
            }
        }
    }
}
