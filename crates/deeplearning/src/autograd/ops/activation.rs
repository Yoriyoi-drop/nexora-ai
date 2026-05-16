use ndarray::ArrayD;

use super::super::tensor::Tensor;

pub fn relu(input: &Tensor) -> Tensor {
    let data = input.data();
    let result = data.mapv(|x| if x > 0.0 { x } else { 0.0 });

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = data.clone();

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let input_data = &saved[0];
            let mask = input_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            vec![grad.clone() * mask]
        }),
    )
}

pub fn gelu(input: &Tensor) -> Tensor {
    let data = input.data();
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;

    let result = data.mapv(|x| {
        let x3 = x * x * x;
        let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x3);
        0.5 * x * (1.0 + tanh_arg.tanh())
    });

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = data.clone();

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(move |grad, saved| {
            let x = &saved[0];
            let sqrt_2_over_pi = (2.0 / std::f64::consts::PI) as f32;

            let gelu_grad = x.mapv(|xv| {
                let x3 = xv * xv * xv;
                let tanh_arg = sqrt_2_over_pi * (xv + 0.044715 * x3);
                let t = tanh_arg.tanh();
                let sech2 = 1.0 - t * t;
                0.5 * (1.0 + t) + 0.5 * xv * sech2 * sqrt_2_over_pi * (1.0 + 0.134145 * xv * xv)
            });

            vec![grad.clone() * gelu_grad]
        }),
    )
}

pub fn sigmoid(input: &Tensor) -> Tensor {
    let data = input.data();
    let result = data.mapv(|x| 1.0 / (1.0 + (-x).exp()));

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = result.clone();

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let sig = &saved[0];
            let sig_grad = sig.mapv(|s| s * (1.0 - s));
            vec![grad.clone() * sig_grad]
        }),
    )
}

pub fn silu(input: &Tensor) -> Tensor {
    let data = input.data();
    let result = data.mapv(|x| x * (1.0 / (1.0 + (-x).exp())));

    if !input.requires_grad() {
        return Tensor::new(result);
    }

    let saved = data.clone();

    Tensor::with_grad_fn(
        result,
        vec![input.clone()],
        vec![saved],
        Box::new(|grad, saved| {
            let x = &saved[0];
            let sig = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            let silu_grad = &sig + x * &sig * (1.0 - &sig);
            vec![grad.clone() * silu_grad]
        }),
    )
}

pub fn swiglu(gate: &Tensor, x: &Tensor) -> Tensor {
    let gate_data = gate.data();
    let x_data = x.data();

    let sig = gate_data.mapv(|g| 1.0 / (1.0 + (-g).exp()));
    let silu_gate = &gate_data * &sig;
    let result = &silu_gate * &x_data;

    let requires_grad = gate.requires_grad() || x.requires_grad();
    if !requires_grad {
        return Tensor::new(result);
    }

    let saved = vec![gate_data.clone(), x_data.clone()];

    Tensor::with_grad_fn(
        result,
        vec![gate.clone(), x.clone()],
        saved,
        Box::new(|grad, saved| {
            let g = &saved[0];
            let x_val = &saved[1];

            let sig = g.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            let dsilu = &sig + g * &sig * (1.0 - &sig);

            let d_gate = grad.clone() * x_val * dsilu;
            let d_x = grad.clone() * &sig;
            vec![d_gate, d_x]
        }),
    )
}
