use ndarray::ArrayD;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::tape;
use super::device::{Device, Storage};
use super::mixed_precision::DType;

static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorInner>>);

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.borrow();
        f.debug_struct("Tensor")
            .field("id", &inner.id)
            .field("shape", &inner.storage.shape())
            .field("dtype", &inner.dtype)
            .field("device", &inner.device)
            .field("requires_grad", &inner.requires_grad)
            .finish()
    }
}

struct TensorInner {
    id: usize,
    storage: Storage,
    device: Device,
    dtype: DType,
    grad: Option<ArrayD<f32>>,
    requires_grad: bool,
    grad_fn_idx: Option<usize>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Rc::new(RefCell::new(TensorInner {
            id,
            storage: Storage::Cpu(data),
            device: Device::Cpu,
            dtype: DType::F32,
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

    pub fn device(&self) -> Device {
        self.0.borrow().device.clone()
    }

    pub fn dtype(&self) -> DType {
        self.0.borrow().dtype
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().storage.shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.borrow().storage.ndim()
    }

    pub fn numel(&self) -> usize {
        self.0.borrow().storage.numel()
    }

    pub fn storage(&self) -> std::cell::Ref<'_, Storage> {
        std::cell::Ref::map(self.0.borrow(), |inner| &inner.storage)
    }

    pub fn data_ref(&self) -> std::cell::Ref<'_, ArrayD<f32>> {
        std::cell::Ref::map(self.0.borrow(), |inner| {
            inner.storage.as_cpu().expect("Tensor is on GPU; call to_device(Device::Cpu) first")
        })
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().storage.to_cpu()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    /// Move tensor to a specific device.
    pub fn to_device(&self, target: &Device) -> Self {
        let inner = self.0.borrow();
        if inner.device == *target {
            return self.clone();
        }
        match target {
            Device::Cpu => {
                let cpu_data = inner.storage.to_cpu();
                drop(inner);
                let t = Tensor::new(cpu_data);
                t.set_requires_grad(self.requires_grad());
                t
            }
            #[cfg(feature = "gpu")]
            Device::Gpu(_device_id) => {
                let cpu_data = inner.storage.to_cpu();
                let gpu_tensor = crate::gpu::GpuTensor::from_cpu(&cpu_data)
                    .expect("Failed to transfer tensor to GPU");
                let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                let t = Tensor(Rc::new(RefCell::new(TensorInner {
                    id,
                    storage: Storage::Gpu(gpu_tensor),
                    device: Device::Gpu(0),
                    dtype: DType::F32,
                    grad: inner.grad.clone(),
                    requires_grad: inner.requires_grad,
                    grad_fn_idx: None,
                })));
                t
            }
        }
    }

    /// Check if tensor is on GPU
    pub fn is_cuda(&self) -> bool {
        self.is_gpu()
    }

    /// Check if tensor is on GPU
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self.0.borrow().device, Device::Gpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
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
            storage: Storage::Cpu(data),
            device: Device::Cpu,
            dtype: DType::F32,
            grad: None,
            requires_grad: true,
            grad_fn_idx: Some(grad_fn_idx),
        })))
    }

    pub(crate) fn accumulate_grad(&self, grad: &ArrayD<f32>) {
        let mut inner = self.0.borrow_mut();
        if let Some(ref mut existing) = inner.grad {
            *existing += grad;
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

    pub fn set_data(&self, new_data: ArrayD<f32>) {
        let mut inner = self.0.borrow_mut();
        inner.storage = Storage::Cpu(new_data);
        inner.device = Device::Cpu;
    }

    pub fn set_grad(&self, grad: ArrayD<f32>) {
        self.0.borrow_mut().grad = Some(grad);
    }

    pub fn subtract_from_data(&self, delta: &ArrayD<f32>) {
        let mut inner = self.0.borrow_mut();
        let current = inner.storage.to_cpu();
        let new_data = &current - delta;
        inner.storage = Storage::Cpu(new_data);
        inner.device = Device::Cpu;
    }

    pub fn backward(&self) {
        let shape = self.0.borrow().storage.shape();
        drop(self.0.borrow());
        let grad = ArrayD::ones(shape);
        self.accumulate_grad(&grad);
        super::engine::backward_engine(self);
    }
}
