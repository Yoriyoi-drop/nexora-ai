pub mod tensor;
pub mod tape;
pub mod engine;
pub mod broadcast;
pub mod device;
pub mod ops;
pub mod mixed_precision;
pub mod data_parallel;
pub mod training_pipeline;

pub use tensor::Tensor;
pub use tape::clear_tape;
pub use device::{Device, Storage};
pub use mixed_precision::{DType, LossScaler};
pub use data_parallel::{GradientAccumulator, DataParallel, DataParallelConfig};
pub use training_pipeline::{TrainingLoop, TrainingLoopConfig, TrainingMetrics, Checkpoint, compute_grad_norm};
pub use ops::*;
use ndarray::ArrayD;

/// Tensor operations as methods for chain-style API
pub trait TensorOps {
    // Math
    fn add(&self, other: &Self) -> Tensor;
    fn sub(&self, other: &Self) -> Tensor;
    fn mul(&self, other: &Self) -> Tensor;
    fn div(&self, other: &Self) -> Tensor;
    fn exp(&self) -> Tensor;
    fn ln(&self) -> Tensor;
    fn powf(&self, exponent: f32) -> Tensor;
    fn sqrt(&self) -> Tensor;
    // Linear algebra
    fn matmul(&self, other: &Self) -> Tensor;
    // Reduce
    fn sum(&self) -> Tensor;
    fn mean(&self) -> Tensor;
    // Shape
    fn reshape(&self, shape: &[usize]) -> Tensor;
    fn transpose(&self) -> Tensor;
    // Activation
    fn relu(&self) -> Tensor;
    fn gelu(&self) -> Tensor;
    fn sigmoid(&self) -> Tensor;
    fn tanh(&self) -> Tensor;
    fn leaky_relu(&self, negative_slope: f32) -> Tensor;
    fn silu(&self) -> Tensor;
    // Neural network
    fn softmax(&self, axis: usize) -> Tensor;
    fn log_softmax(&self, axis: usize) -> Tensor;
    fn dropout(&self, rate: f32, training: bool) -> Tensor;
}

impl TensorOps for Tensor {
    fn add(&self, other: &Self) -> Tensor { ops::math::add(self, other) }
    fn sub(&self, other: &Self) -> Tensor { ops::math::sub(self, other) }
    fn mul(&self, other: &Self) -> Tensor { ops::math::mul(self, other) }
    fn div(&self, other: &Self) -> Tensor { ops::math::div(self, other) }
    fn exp(&self) -> Tensor { ops::math::exp(self) }
    fn ln(&self) -> Tensor { ops::math::ln(self) }
    fn powf(&self, exponent: f32) -> Tensor { ops::math::powf(self, exponent) }
    fn sqrt(&self) -> Tensor { ops::math::sqrt(self) }
    fn matmul(&self, other: &Self) -> Tensor { ops::matmul::matmul(self, other) }
    fn sum(&self) -> Tensor { ops::reduce::sum(self) }
    fn mean(&self) -> Tensor { ops::reduce::mean(self) }
    fn reshape(&self, shape: &[usize]) -> Tensor { ops::shape::reshape(self, shape) }
    fn transpose(&self) -> Tensor { ops::shape::transpose(self) }
    fn relu(&self) -> Tensor { ops::activation::relu(self) }
    fn gelu(&self) -> Tensor { ops::activation::gelu(self) }
    fn sigmoid(&self) -> Tensor { ops::activation::sigmoid(self) }
    fn tanh(&self) -> Tensor { ops::activation::tanh(self) }
    fn leaky_relu(&self, negative_slope: f32) -> Tensor { ops::activation::leaky_relu(self, negative_slope) }
    fn silu(&self) -> Tensor { ops::activation::silu(self) }
    fn softmax(&self, axis: usize) -> Tensor { ops::nn::softmax(self, axis) }
    fn log_softmax(&self, axis: usize) -> Tensor { ops::nn::log_softmax(self, axis) }
    fn dropout(&self, rate: f32, training: bool) -> Tensor { ops::nn::dropout(self, rate, training) }
}

/// Module system — like PyTorch nn.Module
pub trait Module {
    fn parameters(&self) -> Vec<Tensor>;
    fn forward(&self, input: &Tensor) -> Tensor;
    fn name(&self) -> &str { "Module" }
}

/// Linear layer (fully connected)
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, with_bias: bool) -> Self {
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(&[in_features, out_features], true)
            .mul(&Tensor::from_slice(&[scale], &[1]));
        let bias = if with_bias {
            Some(Tensor::zeros(&[out_features], true))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.matmul(&self.weight);
        if let Some(ref b) = self.bias {
            out = out.add(b);
        }
        out
    }

    fn name(&self) -> &str { "Linear" }
}

/// Simple MLP
pub struct MLP {
    pub layers: Vec<Box<dyn Module>>,
}

impl MLP {
    pub fn new(layer_sizes: &[usize], activation: &str) -> Self {
        let mut layers: Vec<Box<dyn Module>> = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Box::new(Linear::new(layer_sizes[i], layer_sizes[i + 1], true)));
        }
        Self { layers }
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut h = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.layers.len() - 1 {
                h = h.relu();
            }
        }
        h
    }

    fn name(&self) -> &str { "MLP" }
}

/// AdamW Optimizer (Adam with decoupled weight decay)
pub struct Adam {
    pub parameters: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: Option<f32>,
    pub(crate) step: usize,
    pub(crate) m: Vec<ArrayD<f32>>,
    pub(crate) v: Vec<ArrayD<f32>>,
}

impl Adam {
    pub fn new(parameters: Vec<Tensor>, lr: f32) -> Self {
        let m = parameters.iter().map(|p| ArrayD::zeros(p.shape().to_vec())).collect();
        let v = parameters.iter().map(|p| ArrayD::zeros(p.shape().to_vec())).collect();
        Self { parameters, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, max_grad_norm: None, step: 0, m, v }
    }

    pub fn zero_grad(&self) {
        for p in &self.parameters {
            p.zero_grad();
        }
    }

    pub fn set_weight_decay(&mut self, wd: f32) {
        self.weight_decay = wd;
    }

    pub fn set_max_grad_norm(&mut self, max_norm: Option<f32>) {
        self.max_grad_norm = max_norm;
    }

    fn clip_gradients(&self, max_norm: f32) {
        let mut total_norm_sq = 0.0f32;
        let mut has_nan = false;
        let grads: Vec<(usize, ArrayD<f32>)> = self.parameters.iter().enumerate()
            .filter_map(|(i, p)| p.grad().map(|g| (i, g)))
            .collect();

        for (_, ref g) in &grads {
            for &x in g.iter() {
                if !x.is_finite() {
                    has_nan = true;
                }
                total_norm_sq += x * x;
            }
        }

        if has_nan {
            eprintln!("[WARN] NaN/Inf detected in gradients — zeroing all gradients to prevent cascade corruption");
            for (i, _) in &grads {
                self.parameters[*i].zero_grad();
            }
            return;
        }

        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            for (i, mut g) in grads {
                g.mapv_inplace(|x| x * scale);
                self.parameters[i].set_grad(g);
            }
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
        let bias_corr1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_corr2 = 1.0 - self.beta2.powi(self.step as i32);

        if let Some(max_norm) = self.max_grad_norm {
            self.clip_gradients(max_norm);
        }

        for (i, p) in self.parameters.iter().enumerate() {
            if let Some(g) = p.grad() {
                self.m[i] = &self.m[i] * self.beta1 + &g * (1.0 - self.beta1);
                self.v[i] = &self.v[i] * self.beta2 + (&g * &g) * (1.0 - self.beta2);

                let m_hat = &self.m[i] / bias_corr1;
                let v_hat = &self.v[i] / bias_corr2;
                let denom = v_hat.mapv(|x| x.sqrt()) + self.eps;

                let lr_arr = ArrayD::from_elem(denom.shape().to_vec(), self.lr);
                let mut update = &m_hat * (&lr_arr / &denom);

                if self.weight_decay > 0.0 {
                    let decay = p.data().mapv(|x| x * self.lr * self.weight_decay);
                    update = update + &decay;
                }

                p.subtract_from_data(&update);
            }
        }
    }
}

/// SGD Optimizer
pub struct SGD {
    pub parameters: Vec<Tensor>,
    pub lr: f32,
    pub momentum: f32,
    velocities: Vec<Option<ArrayD<f32>>>,
}

impl SGD {
    pub fn new(parameters: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let velocities = parameters.iter().map(|_| None).collect();
        Self { parameters, lr, momentum, velocities }
    }

    pub fn zero_grad(&self) {
        for p in &self.parameters {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        for (i, p) in self.parameters.iter().enumerate() {
            if let Some(g) = p.grad() {
                let update = if self.momentum > 0.0 {
                    let vel = self.velocities[i].get_or_insert_with(|| ArrayD::zeros(g.shape().to_vec()));
                    let new_vel = &*vel * self.momentum + &g * self.lr;
                    *vel = new_vel.clone();
                    new_vel
                } else {
                    &g * self.lr
                };
                p.subtract_from_data(&update);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward() {
        let a = Tensor::randn(&[3], true);
        let b = Tensor::randn(&[3], true);
        let c = a.add(&b);
        let d = c.sum();
        d.backward();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::randn(&[3, 1], true);
        let b = Tensor::randn(&[1, 4], true);
        let c = a.add(&b);
        let d = c.sum();
        d.backward();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
        assert_eq!(a.grad().unwrap().shape(), &[3, 1]);
        assert_eq!(b.grad().unwrap().shape(), &[1, 4]);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::randn(&[2, 3], true);
        let b = Tensor::randn(&[3, 4], true);
        let c = a.matmul(&b);
        let d = c.sum();
        d.backward();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
        assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
        assert_eq!(b.grad().unwrap().shape(), &[3, 4]);
    }

    #[test]
    fn test_chain_backward() {
        let x = Tensor::randn(&[4, 4], true);
        let w = Tensor::randn(&[4, 4], true);
        let y = x.matmul(&w).gelu().mean();
        y.backward();
        assert!(x.grad().is_some());
        assert!(w.grad().is_some());
    }

    #[test]
    fn test_linear_layer() {
        let layer = Linear::new(10, 5, true);
        let x = Tensor::randn(&[3, 10], false);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[3, 5]);
    }

    #[test]
    fn test_linear_backward() {
        let layer = Linear::new(10, 5, true);
        let x = Tensor::randn(&[3, 10], false);
        let y = layer.forward(&x).sum();
        y.backward();
        for p in layer.parameters() {
            assert!(p.grad().is_some());
        }
    }

    #[test]
    fn test_sgd_step() {
        let layer = Linear::new(4, 2, true);
        let x = Tensor::randn(&[2, 4], false);
        let y = layer.forward(&x).sum();
        y.backward();

        let params = layer.parameters();
        let old_data: Vec<f32> = params[0].data().iter().copied().collect();

        let mut opt = SGD::new(params.clone(), 0.01, 0.0);
        opt.step();

        let new_data = params[0].data();
        let changed = old_data.iter().zip(new_data.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "SGD should update parameters");
    }

    #[test]
    fn test_softmax_backward() {
        let a = Tensor::randn(&[2, 4], true);
        let b = a.softmax(1);
        let c = b.sum();
        c.backward();
        assert!(a.grad().is_some());
        assert_eq!(a.grad().unwrap().shape(), &[2, 4]);
    }

    #[test]
    fn test_dropout_forward() {
        let a = Tensor::randn(&[100], false);
        let b = a.dropout(0.5, true);
        assert_eq!(b.shape(), &[100]);
    }

    #[test]
    fn test_requires_grad_false() {
        let a = Tensor::randn(&[3], false);
        let b = Tensor::randn(&[3], true);
        let c = a.add(&b).sum();
        c.backward();
        assert!(b.grad().is_some());
    }

    #[test]
    fn test_reshape_backward() {
        let a = Tensor::randn(&[2, 3, 4], true);
        let b = a.reshape(&[6, 4]).sum();
        b.backward();
        assert_eq!(a.grad().unwrap().shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_bce_backward() {
        let pred = Tensor::from_slice(&[0.9, 0.2, 0.7, 0.4], &[4]);
        pred.set_requires_grad(true);
        let target = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let loss = ops::nn::binary_cross_entropy(&pred, &target).mean();
        loss.backward();
        assert!(pred.grad().is_some());
        assert_eq!(pred.grad().unwrap().shape(), &[4]);
    }

    #[test]
    fn test_tanh_backward() {
        let a = Tensor::randn(&[4], true);
        let b = a.tanh().sum();
        b.backward();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_leaky_relu_backward() {
        let a = Tensor::randn(&[4], true);
        let b = a.leaky_relu(0.01).sum();
        b.backward();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_exp_backward() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        a.set_requires_grad(true);
        let b = a.exp().sum();
        b.backward();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_ln_backward() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        a.set_requires_grad(true);
        let b = a.ln().sum();
        b.backward();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_sqrt_backward() {
        let a = Tensor::from_slice(&[4.0, 9.0, 16.0], &[3]);
        a.set_requires_grad(true);
        let b = a.sqrt().sum();
        b.backward();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_powf_backward() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]);
        a.set_requires_grad(true);
        let b = a.powf(2.0).sum();
        b.backward();
        assert!(a.grad().is_some());
    }

    #[test]
    fn test_cross_entropy_backward() {
        let logits = Tensor::from_slice(&[2.0, 1.0, 0.1, 0.5, 2.5, 0.3], &[2, 3]);
        logits.set_requires_grad(true);
        let targets = Tensor::from_slice(&[0.0, 2.0], &[2]);
        let loss = cross_entropy_loss(&logits, &targets).mean();
        loss.backward();
        assert!(logits.grad().is_some());
        assert_eq!(logits.grad().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn test_adam_step() {
        let layer = Linear::new(4, 2, true);
        let x = Tensor::randn(&[2, 4], false);
        let y = layer.forward(&x).sum();
        y.backward();

        let params = layer.parameters();
        let old_data: Vec<f32> = params[0].data().iter().copied().collect();

        let mut opt = Adam::new(params.clone(), 0.001);
        opt.step();

        let new_data = params[0].data();
        let changed = old_data.iter().zip(new_data.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Adam should update parameters");
    }

    #[test]
    fn test_end_to_end_linear_regression() {
        // y = X @ w + noise
        let n = 100;
        let d = 5;
        let w_true: Vec<f32> = (0..d).map(|i| (i + 1) as f32 / 10.0).collect();
        let x_data: Vec<f32> = (0..n * d).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let y_data: Vec<f32> = (0..n).map(|i| {
            let pred: f32 = (0..d).map(|j| x_data[i * d + j] * w_true[j]).sum();
            pred + rand::random::<f32>() * 0.1
        }).collect();

        let x = Tensor::from_slice(&x_data, &[n, d]);
        let y = Tensor::from_slice(&y_data, &[n, 1]);
        let model = Linear::new(d, 1, true);

        let mut losses = Vec::with_capacity(50);
        let mut opt = SGD::new(model.parameters(), 0.01, 0.9);

        for _step in 0..50 {
            let pred = model.forward(&x);
            let diff = pred.sub(&y);
            let loss = diff.mul(&diff).mean();
            loss.backward();
            opt.step();
            opt.zero_grad();
            losses.push(loss.data()[0]);
        }

        assert!(losses[losses.len() - 1] < 0.5, "Loss should decrease during training: {:?}", losses);
    }

    #[test]
    fn test_end_to_end_classifier() {
        // Simple binary classifier with BCE
        let n = 50;
        let d = 4;
        let x_data: Vec<f32> = (0..n * d).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let y_data: Vec<f32> = (0..n).map(|i| {
            let sum: f32 = (0..d).map(|j| x_data[i * d + j]).sum();
            if sum > 0.0 { 1.0 } else { 0.0 }
        }).collect();

        let x = Tensor::from_slice(&x_data, &[n, d]);
        let y = Tensor::from_slice(&y_data, &[n, 1]);
        let model = Linear::new(d, 1, true);

        let mut losses = Vec::with_capacity(100);
        let mut opt = SGD::new(model.parameters(), 0.1, 0.9);

        for _step in 0..100 {
            let logits = model.forward(&x);
            let probs = logits.sigmoid();
            let loss = ops::nn::binary_cross_entropy(&probs, &y).mean();
            loss.backward();
            opt.step();
            opt.zero_grad();
            losses.push(loss.data()[0]);
        }

        assert!(losses[losses.len() - 1] < 0.5, "BCE loss should decrease: {:?}", losses);
    }

    #[test]
    fn test_full_training_step() {
        let w = Tensor::randn(&[4, 2], true);
        let x = Tensor::randn(&[3, 4], false);
        let y = x.matmul(&w).sigmoid().mean();
        y.backward();
        assert!(w.grad().is_some());
        assert_eq!(w.grad().unwrap().shape(), &[4, 2]);
    }
}
