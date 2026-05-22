use ndarray::ArrayD;
use rand::Rng;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::device::{Device, Storage};
use super::mixed_precision::DType;
use super::tape;

static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Global flag: when set, Tensor::new / from_slice / zeros / ones create GPU-resident
/// tensors so that GPU ops (matmul, add, etc.) stay on GPU instead of falling back to CPU.
/// Defaults to true when gpu feature is compiled.
#[cfg(feature = "gpu")]
static GPU_AUTO_CREATE: AtomicBool = AtomicBool::new(true);
#[cfg(not(feature = "gpu"))]
static GPU_AUTO_CREATE: AtomicBool = AtomicBool::new(false);

/// Enable automatic GPU tensor creation for from_slice / zeros / ones.
pub fn set_gpu_auto_create(enabled: bool) {
    GPU_AUTO_CREATE.store(enabled, Ordering::SeqCst);
}

/// Check whether automatic GPU tensor creation is enabled.
pub fn is_gpu_auto_create() -> bool {
    GPU_AUTO_CREATE.load(Ordering::SeqCst)
}

pub fn next_tensor_id() -> usize {
    TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone)]
pub struct Tensor(Arc<Mutex<TensorInner>>);

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
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
    grad: Option<Storage>,
    requires_grad: bool,
    grad_fn_idx: Option<usize>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        #[cfg(feature = "gpu")]
        if is_gpu_auto_create() {
            if let Ok(_ctx) = crate::gpu::GpuContext::global() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&data) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    return Tensor(Arc::new(Mutex::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad: false,
                        grad_fn_idx: None,
                    })));
                }
            }
        }
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Arc::new(Mutex::new(TensorInner {
            id,
            storage: Storage::Cpu(data),
            device: Device::Cpu,
            dtype: DType::F32,
            grad: None,
            requires_grad: false,
            grad_fn_idx: None,
        })))
    }

    /// Create tensor from GPU tensor (no grad tracking)
    #[cfg(feature = "gpu")]
    pub fn from_gpu(gpu_tensor: crate::gpu::GpuTensor, id: usize, requires_grad: bool) -> Self {
        Self(Arc::new(Mutex::new(TensorInner {
            id,
            storage: Storage::Gpu(gpu_tensor),
            device: Device::Gpu(0),
            dtype: DType::F32,
            grad: None,
            requires_grad,
            grad_fn_idx: None,
        })))
    }

    pub fn set_requires_grad(&self, val: bool) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requires_grad = val;
    }

    pub fn requires_grad(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requires_grad
    }

    pub fn id(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).id
    }

    pub fn device(&self) -> Device {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .device
            .clone()
    }

    pub fn dtype(&self) -> DType {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).dtype
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .shape()
    }

    pub fn ndim(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .ndim()
    }

    pub fn numel(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .numel()
    }

    pub fn storage(&self) -> Storage {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .clone()
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .storage
            .to_cpu()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .grad
            .as_ref()
            .map(|g| g.to_cpu())
    }

    /// Returns the gradient storage directly (no CPU readback if on GPU).
    /// Use this in GPU training paths to avoid blocking GPU→CPU transfer.
    #[cfg(feature = "gpu")]
    pub fn grad_storage(&self) -> Option<Storage> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .grad
            .clone()
    }

    /// Move tensor to a specific device.
    pub fn to_device(&self, target: &Device) -> Self {
        let inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if inner.device == *target {
            return self.clone();
        }
        match target {
            Device::Cpu => {
                let cpu_data = inner.storage.to_cpu();
                let requires_grad = inner.requires_grad;
                drop(inner);
                let t = Tensor::new(cpu_data);
                t.set_requires_grad(requires_grad);
                t
            }
            #[cfg(feature = "gpu")]
            Device::Gpu(_device_id) => {
                let cpu_data = inner.storage.to_cpu();
                let gpu_tensor = crate::gpu::GpuTensor::from_cpu(&cpu_data)
                    .expect("Failed to transfer tensor to GPU");
                let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                let t = Tensor(Arc::new(Mutex::new(TensorInner {
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
            matches!(
                self.0.lock().unwrap_or_else(|e| e.into_inner()).device,
                Device::Gpu(_)
            )
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        let len: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..len / 2)
            .flat_map(|_| {
                let u1: f32 = rng.gen::<f32>().max(1e-38);
                let u2: f32 = rng.gen::<f32>().max(1e-38);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                [r * theta.cos(), r * theta.sin()]
            })
            .collect();
        let data = if len % 2 == 0 {
            data
        } else {
            let u: f32 = rng.gen::<f32>().max(1e-38);
            let v: f32 = rng.gen::<f32>().max(1e-38);
            let r = (-2.0 * u.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * v;
            let mut d = data;
            d.push(r * theta.cos());
            d
        };
        let arr = ArrayD::from_shape_vec(shape.to_vec(), data)
            .expect("Failed to create tensor from shape");
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::zeros(shape.to_vec());
        #[cfg(feature = "gpu")]
        if is_gpu_auto_create() {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&arr) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    let t = Tensor(Arc::new(Mutex::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad,
                        grad_fn_idx: None,
                    })));
                    return t;
                }
            }
        }
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::ones(shape.to_vec());
        #[cfg(feature = "gpu")]
        if is_gpu_auto_create() {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&arr) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    let t = Tensor(Arc::new(Mutex::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad,
                        grad_fn_idx: None,
                    })));
                    return t;
                }
            }
        }
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let arr = ArrayD::from_shape_vec(shape.to_vec(), data.to_vec())
            .expect("Failed to create tensor from slice");
        #[cfg(feature = "gpu")]
        if is_gpu_auto_create() {
            if let Ok(ctx) = crate::gpu::GpuContext::global() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&arr) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    return Tensor(Arc::new(Mutex::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad: false,
                        grad_fn_idx: None,
                    })));
                }
            }
        }
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
        Self(Arc::new(Mutex::new(TensorInner {
            id,
            storage: Storage::Cpu(data),
            device: Device::Cpu,
            dtype: DType::F32,
            grad: None,
            requires_grad: true,
            grad_fn_idx: Some(grad_fn_idx),
        })))
    }

    /// Create GPU tensor with GPU-native backward (Phase 4).
    /// Forward data stays on GPU. Backward runs on GPU via gpu_backward closure.
    /// Falls back to CPU backward if GPU backward fails.
    #[cfg(feature = "gpu")]
    pub(crate) fn from_gpu_with_grad_fn(
        gpu_tensor: crate::gpu::GpuTensor,
        inputs: Vec<Tensor>,
        saved: Vec<ArrayD<f32>>,
        saved_gpu: Vec<crate::gpu::GpuTensor>,
        backward: Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>,
        gpu_backward: Option<crate::tape::GpuBackwardFn>,
    ) -> Self {
        let grad_fn_idx =
            crate::tape::register_gpu_grad_fn(inputs, saved, saved_gpu, backward, gpu_backward);
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Arc::new(Mutex::new(TensorInner {
            id,
            storage: Storage::Gpu(gpu_tensor),
            device: Device::Gpu(0),
            dtype: DType::F32,
            grad: None,
            requires_grad: true,
            grad_fn_idx: Some(grad_fn_idx),
        })))
    }

    pub(crate) fn accumulate_grad(&self, grad: &ArrayD<f32>) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match inner.grad {
            Some(Storage::Cpu(ref mut arr)) => {
                *arr += grad;
            }
            #[cfg(feature = "gpu")]
            Some(Storage::Gpu(_)) => {
                let mut existing = inner.grad.take().unwrap().to_cpu();
                existing += grad;
                inner.grad = Some(Storage::Cpu(existing));
            }
            None => {
                inner.grad = Some(Storage::Cpu(grad.clone()));
            }
        }
    }

    /// Accumulate gradient from Storage (CPU or GPU).
    /// GPU+GPU accumulates entirely on GPU without CPU round-trip.
    #[cfg(feature = "gpu")]
    pub(crate) fn accumulate_grad_storage(&self, grad: &Storage) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match (&mut inner.grad, grad) {
            (Some(Storage::Cpu(existing)), Storage::Cpu(g)) => *existing += g,
            (Some(Storage::Cpu(existing)), Storage::Gpu(g)) => {
                let g_cpu = g.to_cpu();
                *existing += &g_cpu;
            }
            (Some(Storage::Gpu(existing)), Storage::Cpu(g)) => {
                let mut e = existing.to_cpu();
                e += g;
                inner.grad = Some(Storage::Cpu(e));
            }
            (Some(Storage::Gpu(existing)), Storage::Gpu(g)) => {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    let _ = ctx.add_inplace(existing, g);
                } else {
                    let mut e = existing.to_cpu();
                    e += &g.to_cpu();
                    inner.grad = Some(Storage::Cpu(e));
                }
            }
            (None, g) => {
                inner.grad = Some(g.clone());
            }
        }
    }

    pub(crate) fn get_grad_fn_idx(&self) -> Option<usize> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).grad_fn_idx
    }

    pub fn zero_grad(&self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).grad = None;
    }

    /// Zero grads on GPU when tensor storage is on GPU.
    #[cfg(feature = "gpu")]
    pub fn zero_grad_gpu(&self) -> Result<(), crate::gpu::GpuError> {
        use crate::gpu::GpuContext;
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.grad = None;
        if let Storage::Gpu(gpu_t) = &inner.storage {
            let ctx = GpuContext::global()?;
            ctx.fill_zero(gpu_t)?;
        }
        Ok(())
    }

    pub fn set_data(&self, new_data: ArrayD<f32>) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.storage = Storage::Cpu(new_data);
    }

    pub fn set_storage(&self, storage: Storage) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.storage = storage;
    }

    pub fn set_device(&self, device: Device) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.device = device;
    }

    pub fn set_grad(&self, grad: ArrayD<f32>) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .grad = Some(Storage::Cpu(grad));
    }

    /// Set gradient from Storage (CPU or GPU).
    #[cfg(feature = "gpu")]
    pub fn set_grad_storage(&self, grad: Storage) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .grad = Some(grad);
    }

    pub fn subtract_from_data(&self, delta: &ArrayD<f32>) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let current = inner.storage.to_cpu();
        let new_data = &current - delta;
        inner.storage = Storage::Cpu(new_data);
    }

    pub fn backward(&self) {
        let shape = {
            let inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
            inner.storage.shape()
        };
        let grad = ArrayD::ones(shape);
        self.accumulate_grad(&grad);
        super::engine::backward_engine(self);
        super::tape::clear_tape();
    }
}

/// Helper: extract CPU gradient from Option<Storage>, used by training pipeline.
pub fn grad_as_cpu(grad: &Option<Storage>) -> Option<ArrayD<f32>> {
    grad.as_ref().map(|s| s.to_cpu())
}
