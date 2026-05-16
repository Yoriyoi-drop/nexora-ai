pub mod tensor;
pub mod tape;
pub mod engine;
pub mod broadcast;
pub mod ops;

pub use tensor::Tensor;
pub use ops::*;
use ndarray::ArrayD;

/// Tensor operations as methods for chain-style API
pub trait TensorOps {
    // Math
    fn add(&self, other: &Self) -> Tensor;
    fn sub(&self, other: &Self) -> Tensor;
    fn mul(&self, other: &Self) -> Tensor;
    fn div(&self, other: &Self) -> Tensor;
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
    fn matmul(&self, other: &Self) -> Tensor { ops::matmul::matmul(self, other) }
    fn sum(&self) -> Tensor { ops::reduce::sum(self) }
    fn mean(&self) -> Tensor { ops::reduce::mean(self) }
    fn reshape(&self, shape: &[usize]) -> Tensor { ops::shape::reshape(self, shape) }
    fn transpose(&self) -> Tensor { ops::shape::transpose(self) }
    fn relu(&self) -> Tensor { ops::activation::relu(self) }
    fn gelu(&self) -> Tensor { ops::activation::gelu(self) }
    fn sigmoid(&self) -> Tensor { ops::activation::sigmoid(self) }
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
        let mut layers: Vec<Box<dyn Module>> = Vec::new();
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
                    *vel = vel.clone() * self.momentum + &g * self.lr;
                    vel.clone()
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
    fn test_full_training_step() {
        let w = Tensor::randn(&[4, 2], true);
        let x = Tensor::randn(&[3, 4], false);
        let y = x.matmul(&w).sigmoid().mean();
        y.backward();
        assert!(w.grad().is_some());
        assert_eq!(w.grad().unwrap().shape(), &[4, 2]);
    }
}
