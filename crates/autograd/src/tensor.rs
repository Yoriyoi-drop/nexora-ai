use ndarray::ArrayD;
use parking_lot::RwLock;
use rand::Rng;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{error, warn};

use super::device::{Device, Storage};
use super::mixed_precision::DType;
use super::tape;

static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// RAII guard providing zero-copy access to CPU tensor data.
/// Holds the read lock so the data cannot be mutated while in use.
pub struct DataRef<'a> {
    _guard: parking_lot::RwLockReadGuard<'a, TensorInner>,
    data: *const ArrayD<f32>,
}

impl<'a> Deref for DataRef<'a> {
    type Target = ArrayD<f32>;
    fn deref(&self) -> &ArrayD<f32> {
        // SAFETY: `self.data` is a raw pointer to the ArrayD<f32> inside the
        // `TensorInner` that `self._guard` (RwLockReadGuard) protects. The
        // guard ensures no mutable access to the TensorInner exists for the
        // lifetime of this DataRef, making the dereference safe.
        unsafe { &*self.data }
    }
}

/// Global flag: when set, Tensor::new / from_slice / zeros / ones create GPU-resident
/// tensors so that GPU ops (matmul, add, etc.) stay on GPU instead of falling back to CPU.
/// Defaults to true when gpu feature is compiled.
#[cfg(feature = "device-gpu")]
static GPU_AUTO_CREATE: AtomicBool = AtomicBool::new(true);
#[cfg(not(feature = "device-gpu"))]
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
pub struct Tensor(Arc<RwLock<TensorInner>>);

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.read();
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
        #[cfg(feature = "device-gpu")]
        if is_gpu_auto_create() {
            if let Ok(_ctx) = crate::gpu::GpuContext::global() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&data) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    return Tensor(Arc::new(RwLock::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t, data.shape().to_vec()),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad: false,
                        grad_fn_idx: None,
                    })));
                }
            }
        }
        #[cfg(feature = "device-cuda")]
        if is_gpu_auto_create() {
            if let Ok(rt) = crate::gpu::CudaRuntime::init(0) {
                let flat: Vec<f32> = data.iter().copied().collect();
                match rt.stream.clone_htod(&flat) {
                    Ok(buf) => {
                        let cuda_t = crate::gpu::cuda::CudaTensor {
                            shape: data.shape().to_vec(),
                            buffer: buf,
                            device_id: 0,
                        };
                        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                        return Tensor(Arc::new(RwLock::new(TensorInner {
                            id,
                            #[cfg(feature = "device-cuda")]
                            storage: Storage::Cuda(cuda_t, data.shape().to_vec()),
                            device: Device::Cuda(0),
                            dtype: DType::F32,
                            grad: None,
                            requires_grad: false,
                            grad_fn_idx: None,
                        })));
                    }
                    Err(e) => {
                        tracing::warn!("CUDA upload failed in Tensor::new: {e}");
                    }
                }
            }
        }
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Arc::new(RwLock::new(TensorInner {
            id,
            storage: Storage::Cpu(Arc::new(data)),
            device: Device::Cpu,
            dtype: DType::F32,
            grad: None,
            requires_grad: false,
            grad_fn_idx: None,
        })))
    }

    /// Create tensor from GPU tensor (no grad tracking)
    #[cfg(feature = "device-gpu")]
    pub fn from_gpu(gpu_tensor: crate::gpu::GpuTensor, id: usize, requires_grad: bool) -> Self {
        let shape = gpu_tensor.shape();
        Self(Arc::new(RwLock::new(TensorInner {
            id,
            storage: Storage::Gpu(gpu_tensor, shape),
            device: Device::Gpu(0),
            dtype: DType::F32,
            grad: None,
            requires_grad,
            grad_fn_idx: None,
        })))
    }

    /// Clone the GPU buffer handle without cloning Storage.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_buffer(&self) -> Option<wgpu::Buffer> {
        match &self.0.read().storage {
            Storage::Gpu(gpu, _) => Some(gpu.buffer_arc()),
            _ => None,
        }
    }

    /// Create tensor from CUDA tensor (no grad tracking)
    #[cfg(feature = "device-cuda")]
    pub fn from_cuda(
        cuda_tensor: crate::gpu::cuda::CudaTensor,
        id: usize,
        requires_grad: bool,
    ) -> Self {
        let device_id = cuda_tensor.device_id;
        let shape = cuda_tensor.shape.clone();
        Self(Arc::new(RwLock::new(TensorInner {
            id,
            #[cfg(feature = "device-cuda")]
            storage: Storage::Cuda(cuda_tensor, shape),
            device: Device::Cuda(device_id),
            dtype: DType::F32,
            grad: None,
            requires_grad,
            grad_fn_idx: None,
        })))
    }

    /// Create CUDA tensor with CPU backward (forward stays on CUDA, backward on CPU).
    #[cfg(feature = "device-cuda")]
    pub(crate) fn from_cuda_with_grad_fn(
        cuda_tensor: crate::gpu::cuda::CudaTensor,
        inputs: Vec<Tensor>,
        saved: Vec<ArrayD<f32>>,
        backward: Box<dyn FnOnce(&ArrayD<f32>, &[ArrayD<f32>]) -> Vec<ArrayD<f32>>>,
    ) -> Self {
        let device_id = cuda_tensor.device_id;
        let shape = cuda_tensor.shape.clone();
        let grad_fn_idx = crate::tape::register_grad_fn(inputs, saved, backward);
        let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(Arc::new(RwLock::new(TensorInner {
            id,
            #[cfg(feature = "device-cuda")]
            storage: Storage::Cuda(cuda_tensor, shape),
            device: Device::Cuda(device_id),
            dtype: DType::F32,
            grad: None,
            requires_grad: true,
            grad_fn_idx: Some(grad_fn_idx),
        })))
    }

    pub fn set_requires_grad(&self, val: bool) {
        self.0.write().requires_grad = val;
    }

    pub fn requires_grad(&self) -> bool {
        self.0.read().requires_grad
    }

    pub fn id(&self) -> usize {
        self.0.read().id
    }

    pub fn device(&self) -> Device {
        self.0.read().device.clone()
    }

    pub fn dtype(&self) -> DType {
        self.0.read().dtype
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.read().storage.shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.read().storage.ndim()
    }

    pub fn numel(&self) -> usize {
        self.0.read().storage.numel()
    }

    pub fn storage(&self) -> Storage {
        self.0.read().storage.clone()
    }

    /// Zero-copy access to CPU data via a guard that holds the read lock.
    /// Returns `Err` if tensor is on GPU/CUDA — use `data()` for GPU readback.
    pub fn data_ref(&self) -> Result<DataRef<'_>, &'static str> {
        let guard = self.0.read();
        let ptr: *const ArrayD<f32> = match &guard.storage {
            Storage::Cpu(arr) => arr.as_ref(),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(..) => {
                return Err("data_ref() on GPU tensor — use data() for GPU readback")
            }
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(..) => {
                return Err("data_ref() on CUDA tensor — use data() for GPU readback")
            }
        };
        Ok(DataRef {
            _guard: guard,
            data: ptr,
        })
    }

    /// Returns an `Arc<ArrayD<f32>>` — O(1) for CPU (refcount increment), O(N) GPU readback.
    pub fn data_arc(&self) -> Arc<ArrayD<f32>> {
        self.0.read().storage.data_arc()
    }

    /// Returns an owned copy of the tensor data (CPU clone or GPU readback).
    #[cfg(any(feature = "device-gpu", feature = "device-cuda"))]
    pub fn data(&self) -> ArrayD<f32> {
        let inner = self.0.read();
        match &inner.storage {
            Storage::Cpu(arr) => arr.as_ref().clone(),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(gpu, _) => {
                gpu.to_cpu().unwrap_or_else(|e| {
                    error!("Tensor::data GPU readback failed: {e}");
                    ArrayD::zeros(vec![0])
                })
            }
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(tensor, _) => match crate::gpu::CudaRuntime::global() {
                Ok(rt) => match tensor.to_cpu_vec(&rt.stream) {
                    Ok(cpu_buf) => match ndarray::ArrayD::from_shape_vec(tensor.shape(), cpu_buf)
                    {
                        Ok(arr) => arr,
                        Err(e) => {
                            error!("Tensor::data CUDA shape mismatch: {e}");
                            ArrayD::zeros(vec![0])
                        }
                    },
                    Err(e) => {
                        error!("Tensor::data CUDA readback failed: {e}");
                        ArrayD::zeros(vec![0])
                    }
                },
                Err(e) => {
                    error!("Tensor::data CUDA runtime not available: {e}");
                    ArrayD::zeros(vec![0])
                }
            },
        }
    }

    /// Returns an owned copy of the tensor data (CPU-only path).
    #[cfg(not(any(feature = "device-gpu", feature = "device-cuda")))]
    pub fn data(&self) -> ArrayD<f32> {
        self.0.read().storage.to_cpu()
    }

    #[cfg(any(feature = "device-gpu", feature = "device-cuda"))]
    pub fn grad(&self) -> Option<ArrayD<f32>> {
        match self.0.read().grad.as_ref() {
            Some(Storage::Cpu(arr)) => Some(arr.as_ref().clone()),
            #[cfg(feature = "device-gpu")]
            Some(Storage::Gpu(gpu, _)) => {
                gpu.to_cpu()
                    .map_err(|e| tracing::warn!("Tensor::grad GPU readback failed: {}", e))
                    .ok()
            }
            #[cfg(feature = "device-cuda")]
            Some(Storage::Cuda(tensor, _)) => match crate::gpu::CudaRuntime::global() {
                Ok(rt) => match tensor.to_cpu_vec(&rt.stream) {
                    Ok(cpu_buf) => {
                        match ndarray::ArrayD::from_shape_vec(tensor.shape(), cpu_buf) {
                            Ok(arr) => Some(arr),
                            Err(e) => {
                                tracing::warn!("Tensor::grad CUDA shape mismatch: {e}");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Tensor::grad CUDA readback failed: {e}");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("Tensor::grad CUDA runtime not available: {e}");
                    None
                }
            },
            None => None,
        }
    }

    #[cfg(not(any(feature = "device-gpu", feature = "device-cuda")))]
    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.read().grad.as_ref().map(|g| g.to_cpu())
    }

    /// Returns the gradient storage directly (no CPU readback if on GPU).
    /// Use this in GPU training paths to avoid blocking GPU→CPU transfer.
    #[cfg(feature = "device-gpu")]
    pub fn grad_storage(&self) -> Option<Storage> {
        self.0.read().grad.clone()
    }

    /// Move tensor to a specific device.
    pub fn to_device(&self, target: &Device) -> Self {
        let inner = self.0.read();
        if inner.device == *target {
            return self.clone();
        }
        #[cfg(any(feature = "device-gpu", feature = "device-cuda"))]
        let cpu_data = match &inner.storage {
            Storage::Cpu(arr) => arr.as_ref().clone(),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(gpu, _) => {
                gpu.to_cpu().unwrap_or_else(|e| {
                    error!("Tensor::to_device GPU readback failed: {e}, falling back to CPU");
                    ArrayD::zeros(vec![0])
                })
            }
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(tensor, _) => match crate::gpu::CudaRuntime::global() {
                Ok(rt) => match tensor.to_cpu_vec(&rt.stream) {
                    Ok(cpu_buf) => match ndarray::ArrayD::from_shape_vec(tensor.shape(), cpu_buf)
                    {
                        Ok(arr) => arr,
                        Err(e) => {
                            error!("Tensor::to_device CUDA shape mismatch: {e}");
                            ArrayD::zeros(vec![0])
                        }
                    },
                    Err(e) => {
                        error!(
                            "Tensor::to_device CUDA readback failed: {e}, falling back to CPU"
                        );
                        ArrayD::zeros(vec![0])
                    }
                },
                Err(e) => {
                    error!(
                        "Tensor::to_device CUDA runtime not available: {e}, falling back to CPU"
                    );
                    ArrayD::zeros(vec![0])
                }
            },
        };
        #[cfg(not(any(feature = "device-gpu", feature = "device-cuda")))]
        let cpu_data = inner.storage.to_cpu();
        let requires_grad = inner.requires_grad;
        let _grad = inner.grad.clone();
        let grad_fn_idx = inner.grad_fn_idx;
        drop(inner);
        match target {
            Device::Cpu => {
                let t = Tensor::new(cpu_data);
                t.set_requires_grad(requires_grad);
                if grad_fn_idx.is_some() {
                    t.0.write().grad_fn_idx = grad_fn_idx;
                }
                t
            }
            #[cfg(feature = "device-gpu")]
            Device::Gpu(_) => {
                tracing::warn!("to_device(Gpu) not supported in CPU-only build, staying on CPU");
                Tensor::new(cpu_data)
            }
            #[cfg(feature = "device-cuda")]
            Device::Cuda(_) => {
                tracing::warn!("to_device(Cuda) not supported in CPU-only build, staying on CPU");
                Tensor::new(cpu_data)
            }
        }
    }

    /// Check if tensor is on wgpu GPU
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "device-gpu")]
        {
            matches!(self.0.read().device, Device::Gpu(_))
        }
        #[cfg(not(feature = "device-gpu"))]
        {
            false
        }
    }

    /// Check if tensor is on CUDA GPU
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "device-cuda")]
        {
            matches!(self.0.read().device, Device::Cuda(_))
        }
        #[cfg(not(feature = "device-cuda"))]
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
        let data = if len.is_multiple_of(2) {
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
        let arr = match ArrayD::from_shape_vec(shape.to_vec(), data) {
            Ok(a) => a,
            Err(e) => {
                warn!("[tensor::randn] shape mismatch (unexpected): {e}");
                ArrayD::zeros(shape.to_vec())
            }
        };
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::zeros(shape.to_vec());
        #[cfg(feature = "device-gpu")]
        if is_gpu_auto_create()
            && crate::gpu::GpuContext::global().is_ok() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&arr) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    let t = Tensor(Arc::new(RwLock::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t, arr.shape().to_vec()),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad,
                        grad_fn_idx: None,
                    })));
                    return t;
                }
            }
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::ones(shape.to_vec());
        #[cfg(feature = "device-gpu")]
        if is_gpu_auto_create()
            && crate::gpu::GpuContext::global().is_ok() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&arr) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    let t = Tensor(Arc::new(RwLock::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t, arr.shape().to_vec()),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad,
                        grad_fn_idx: None,
                    })));
                    return t;
                }
            }
        let t = Self::new(arr);
        t.set_requires_grad(requires_grad);
        t
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let arr = ArrayD::from_shape_vec(shape.to_vec(), data.to_vec()).unwrap_or_else(|e| {
            error!(
                "from_slice: data.len()={} product(shape)={:?} did not match: {}",
                data.len(),
                shape,
                e
            );
            ArrayD::zeros(vec![0])
        });
        #[cfg(feature = "device-gpu")]
        if is_gpu_auto_create()
            && crate::gpu::GpuContext::global().is_ok() {
                if let Ok(gpu_t) = crate::gpu::GpuTensor::from_cpu(&arr) {
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::SeqCst);
                    return Tensor(Arc::new(RwLock::new(TensorInner {
                        id,
                        storage: Storage::Gpu(gpu_t, arr.shape().to_vec()),
                        device: Device::Gpu(0),
                        dtype: DType::F32,
                        grad: None,
                        requires_grad: false,
                        grad_fn_idx: None,
                    })));
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
        Self(Arc::new(RwLock::new(TensorInner {
            id,
            storage: Storage::Cpu(Arc::new(data)),
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
    #[cfg(feature = "device-gpu")]
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
        let shape = gpu_tensor.shape();
        Self(Arc::new(RwLock::new(TensorInner {
            id,
            storage: Storage::Gpu(gpu_tensor, shape),
            device: Device::Gpu(0),
            dtype: DType::F32,
            grad: None,
            requires_grad: true,
            grad_fn_idx: Some(grad_fn_idx),
        })))
    }

    pub(crate) fn accumulate_grad(&self, grad: &ArrayD<f32>) {
        let mut inner = self.0.write();
        match &mut inner.grad {
            Some(Storage::Cpu(arr)) => {
                let mut existing = arr.as_ref().clone();
                existing += grad;
                inner.grad = Some(Storage::Cpu(Arc::new(existing)));
            }
            #[cfg(feature = "device-gpu")]
            Some(Storage::Gpu(ref mut gpu, _)) => {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    match crate::gpu::GpuTensor::from_cpu(grad) {
                        Ok(g_gpu) => {
                            if ctx.add_inplace(gpu, &g_gpu).is_ok() {
                                return;
                            }
                        }
                        Err(e) => tracing::warn!("accumulate_grad GPU upload failed: {e}"),
                    }
                }
                inner.grad = Some(Storage::Cpu(Arc::new(grad.clone())));
            }
            #[cfg(feature = "device-cuda")]
            Some(Storage::Cuda(_, _)) => {
                inner.grad = Some(Storage::Cpu(Arc::new(grad.clone())));
            }
            None => {
                inner.grad = Some(Storage::Cpu(Arc::new(grad.clone())));
            }
        }
    }

    /// Accumulate gradient from Storage (CPU or GPU).
    /// GPU+GPU accumulates entirely on GPU without CPU round-trip.
    /// Returns error if GPU readback fails (caller can skip or propagate).
    #[cfg(feature = "device-gpu")]
    pub(crate) fn try_accumulate_grad_storage(
        &self,
        grad: &Storage,
    ) -> Result<(), crate::gpu::GpuError> {
        let mut inner = self.0.write();
        match (&mut inner.grad, grad) {
            (Some(Storage::Cpu(existing)), Storage::Cpu(g)) => {
                *Arc::make_mut(existing) += g.as_ref();
                Ok(())
            }
            (Some(Storage::Cpu(existing)), Storage::Gpu(g, _)) => {
                let g_cpu = g.to_cpu()?;
                *Arc::make_mut(existing) += &g_cpu;
                Ok(())
            }
            (Some(Storage::Gpu(existing, _)), Storage::Cpu(g)) => {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    match crate::gpu::GpuTensor::from_cpu(g) {
                        Ok(g_gpu) => {
                            return ctx.add_inplace(existing, &g_gpu)
                                .map_err(|_| crate::gpu::GpuError::Unsupported("add_inplace failed".into()));
                        }
                        Err(e) => tracing::warn!("try_accumulate_grad_storage GPU upload failed: {e}"),
                    }
                }
                let e = ArrayD::zeros(g.shape());
                inner.grad = Some(Storage::Cpu(Arc::new(e)));
                Ok(())
            }
            (Some(Storage::Gpu(existing, _)), Storage::Gpu(g, _)) => {
                if let Ok(ctx) = crate::gpu::GpuContext::global() {
                    return ctx.add_inplace(existing, g)
                        .map_err(|_| crate::gpu::GpuError::Unsupported("add_inplace failed".into()));
                }
                let e = ArrayD::zeros(g.shape());
                inner.grad = Some(Storage::Cpu(Arc::new(e)));
                Ok(())
            }
            #[cfg(feature = "device-cuda")]
            (Some(Storage::Cuda(_, _)), g) => {
                inner.grad = Some(g.clone());
                Ok(())
            }
            (None, g) => {
                inner.grad = Some(g.clone());
                Ok(())
            }
            #[cfg(feature = "device-cuda")]
            (&mut Some(Storage::Cpu(_)), Storage::Cuda(_, _)) | (&mut Some(Storage::Gpu(_, _)), Storage::Cuda(_, _)) => {
                tracing::warn!("try_accumulate_grad_storage: mixing CPU/GPU grad with CUDA grad, overwriting");
                inner.grad = Some(grad.clone());
                Ok(())
            }
        }
    }

    /// Legacy wrapper — logs warning on GPU readback failure.
    /// Prefer try_accumulate_grad_storage() for production code.
    #[cfg(feature = "device-gpu")]
    pub(crate) fn accumulate_grad_storage(&self, grad: &Storage) {
        if let Err(e) = self.try_accumulate_grad_storage(grad) {
            warn!("accumulate_grad_storage GPU readback/accumulation failed: {e}. Skipping.");
        }
    }

    pub(crate) fn get_grad_fn_idx(&self) -> Option<usize> {
        self.0.read().grad_fn_idx
    }

    pub fn zero_grad(&self) {
        self.0.write().grad = None;
    }

    /// Zero grads on GPU when tensor storage is on GPU.
    #[cfg(feature = "device-gpu")]
    pub fn zero_grad_gpu(&self) -> Result<(), crate::gpu::GpuError> {
        use crate::gpu::GpuContext;
        let mut inner = self.0.write();
        inner.grad = None;
        if let Storage::Gpu(gpu, _) = &inner.storage {
            let ctx = GpuContext::global()?;
            ctx.fill_zero(gpu)?;
        }
        Ok(())
    }

    pub fn set_data(&self, new_data: ArrayD<f32>) {
        let mut inner = self.0.write();
        inner.storage = Storage::Cpu(Arc::new(new_data));
    }

    pub fn set_storage(&self, storage: Storage) {
        let mut inner = self.0.write();
        inner.storage = storage;
    }

    pub fn set_device(&self, device: Device) {
        let mut inner = self.0.write();
        inner.device = device;
    }

    pub fn set_grad(&self, grad: ArrayD<f32>) {
        self.0.write().grad = Some(Storage::Cpu(Arc::new(grad)));
    }

    /// Set gradient from Storage (CPU or GPU).
    #[cfg(feature = "device-gpu")]
    pub fn set_grad_storage(&self, grad: Storage) {
        self.0.write().grad = Some(grad);
    }

    #[cfg(feature = "device-gpu")]
    pub fn subtract_from_data(&self, delta: &ArrayD<f32>) {
        let mut inner = self.0.write();
        let current = match &inner.storage {
            Storage::Cpu(arr) => arr.as_ref().clone(),
            Storage::Gpu(gpu, _) => {
                match gpu.to_cpu() {
                    Ok(cpu) => cpu,
                    Err(e) => {
                        tracing::error!("subtract_from_data: GPU readback failed: {e}. Skipping.");
                        return;
                    }
                }
            }
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => {
                tracing::error!("subtract_from_data: CUDA storage not supported in Gpu path. Skipping.");
                return;
            }
        };
        let new_data = &current - delta;
        inner.storage = Storage::Cpu(Arc::new(new_data));
    }

    #[cfg(not(feature = "device-gpu"))]
    pub fn subtract_from_data(&self, delta: &ArrayD<f32>) {
        let mut inner = self.0.write();
        let current = inner.storage.to_cpu();
        let new_data = &current - delta;
        inner.storage = Storage::Cpu(Arc::new(new_data));
    }

    pub fn backward(&self) {
        let shape = {
            let inner = self.0.read();
            inner.storage.shape()
        };
        let grad = ArrayD::ones(shape);
        self.accumulate_grad(&grad);
        super::engine::backward_engine(self);
        super::tape::clear_tape();
    }
}

/// Helper: extract CPU gradient from Option<Storage>, used by training pipeline.
#[cfg(feature = "device-gpu")]
pub fn grad_as_cpu(grad: &Option<Storage>) -> Option<ArrayD<f32>> {
    match grad.as_ref()? {
        Storage::Cpu(arr) => Some(arr.as_ref().clone()),
        Storage::Gpu(gpu, _) => {
            gpu.to_cpu()
                .map_err(|e| tracing::warn!("grad_as_cpu readback failed: {e}"))
                .ok()
        }
        #[cfg(feature = "device-cuda")]
        Storage::Cuda(_, _) => {
            tracing::warn!("grad_as_cpu: CUDA storage encountered, returning zeros");
            None
        }
    }
}

/// Helper: extract CPU gradient from Option<Storage>, used by training pipeline.
#[cfg(not(feature = "device-gpu"))]
pub fn grad_as_cpu(grad: &Option<Storage>) -> Option<ArrayD<f32>> {
    grad.as_ref().map(|s| s.to_cpu())
}
