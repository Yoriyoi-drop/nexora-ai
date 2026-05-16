use ndarray::ArrayD;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::tape;

static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorInner>>);

struct TensorInner {
    id: usize,
    data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    requires_grad: bool,
    grad_fn_idx: Option<usize>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Rc::new(RefCell::new(TensorInner {
            id,
            data,
            grad: None,
            requires_grad: false,
            grad_fn_idx: None,
        })))
    }

    pub fn set_requires_grad(&self, val: bool) {
        self.0.borrow_mut().requires_grad = val;
    }

    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    pub fn id(&self) -> usize {
        self.0.borrow().id
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.0.borrow().data.ndim()
    }

    pub fn numel(&self) -> usize {
        self.0.borrow().data.len()
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        let len: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..len).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let arr = ArrayD::from_shape_vec(shape.to_vec(), data)
            .expect("Failed to create tensor from shape");
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::zeros(shape.to_vec());
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::ones(shape.to_vec());
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let arr = ArrayD::from_shape_vec(shape.to_vec(), data.to_vec())
            .expect("Failed to create tensor from slice");
        Self::new(arr)
    }

    pub(crate) fn with_grad_fn(
        data: ArrayD<f32>,
        inputs: Vec<Tensor>,
        saved: Vec<ArrayD<f32>>,
        backward: Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>,
    ) -> Self {
        let grad_fn_idx = tape::register_grad_fn(inputs, saved, backward);
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Rc::new(RefCell::new(TensorInner {
            id,
            data,
            grad: None,
            requires_grad: true,
            grad_fn_idx: Some(grad_fn_idx),
        })))
    }

    pub(crate) fn accumulate_grad(&self, grad: &ArrayD<f32>) {
        let mut inner = self.0.borrow_mut();
        if let Some(ref mut existing) = inner.grad {
            *existing = existing.clone() + grad;
        } else {
            inner.grad = Some(grad.clone());
        }
    }

    pub(crate) fn get_grad_fn_idx(&self) -> Option<usize> {
        self.0.borrow().grad_fn_idx
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = None;
    }

    pub fn subtract_from_data(&self, delta: &ArrayD<f32>) {
        let mut inner = self.0.borrow_mut();
        let new_data = &inner.data - delta;
        inner.data = new_data;
    }

    pub fn backward(&self) {
        let shape = self.0.borrow().data.shape().to_vec();
        drop(self.0.borrow());
        let grad = ArrayD::ones(shape);
        self.accumulate_grad(&grad);
        super::engine::backward_engine(self);
    }
}
