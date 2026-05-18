use ndarray::ArrayD;
use tracing;

/// Broadcast error type
#[derive(Debug)]
pub struct BroadcastError(pub String);

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Broadcast error: {}", self.0)
    }
}

impl std::error::Error for BroadcastError {}

/// Compute broadcast shape for two shapes
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    match try_broadcast_shapes(a, b) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("{}. Falling back to shape a", e);
            a.to_vec()
        }
    }
}

fn try_broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, BroadcastError> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let idx_from_end = |shape: &[usize], offset: usize| {
            if offset < shape.len() {
                shape[shape.len() - 1 - offset]
            } else {
                1
            }
        };
        let ai = idx_from_end(a, i);
        let bi = idx_from_end(b, i);
        if ai == bi {
            result.push(ai);
        } else if ai == 1 {
            result.push(bi);
        } else if bi == 1 {
            result.push(ai);
        } else {
            return Err(BroadcastError(format!(
                "shapes {:?} and {:?} incompatible at dim {}", a, b, max_len - 1 - i
            )));
        }
    }
    result.reverse();
    Ok(result)
}

/// Broadcast two arrays to a common shape
pub fn broadcast_arrays(a: &ArrayD<f32>, b: &ArrayD<f32>) -> (ArrayD<f32>, ArrayD<f32>, Vec<usize>) {
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    if a_shape == b_shape {
        return (a.clone(), b.clone(), a_shape);
    }
    let target = broadcast_shapes(&a_shape, &b_shape);
    let a_bc = if a_shape != target {
        match a.clone().broadcast(target.as_slice()) {
            Some(view) => view.to_owned(),
            None => {
                tracing::warn!("Cannot broadcast {:?} to {:?}, returning a clone", a_shape, target);
                a.clone()
            }
        }
    } else {
        a.clone()
    };
    let b_bc = if b_shape != target {
        match b.clone().broadcast(target.as_slice()) {
            Some(view) => view.to_owned(),
            None => {
                tracing::warn!("Cannot broadcast {:?} to {:?}, returning b clone", b_shape, target);
                b.clone()
            }
        }
    } else {
        b.clone()
    };
    (a_bc, b_bc, target)
}

/// Reduce gradient to original shape
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
                ArrayD::from_shape_vec(orig_shape.to_vec(), flat)
                    .expect("shape vec should match flat length")
            } else {
                ArrayD::from_elem(orig_shape.to_vec(), flat[0])
            }
        }
    }
}
