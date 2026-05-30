use ndarray::ArrayD;
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::gpu::GpuError;

/// Physical device where tensor data resides
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// Cross-platform GPU (wgpu backend: NVIDIA, AMD, Intel, Apple Silicon)
    #[cfg(feature = "gpu")]
    Gpu(usize),
    /// NVIDIA CUDA GPU
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        #[cfg(feature = "gpu")]
        if crate::gpu::GpuContext::is_available() {
            return Device::Gpu(0);
        }
        #[cfg(feature = "cuda")]
        if let Ok(runtime) = crate::gpu::CudaRuntime::init(0) {
            let _ = runtime;
            return Device::Cuda(0);
        }
        Device::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => write!(f, "gpu:{}", id),
            #[cfg(feature = "cuda")]
            Device::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

/// Internal tensor storage — wraps concrete array types per device
/// CPU storage uses Arc for O(1) clone and zero-copy data access.
pub enum Storage {
    Cpu(Arc<ArrayD<f32>>),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuTensor),
    #[cfg(feature = "cuda")]
    Cuda(crate::gpu::cuda::CudaTensor),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(arr) => Storage::Cpu(Arc::clone(arr)),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => Storage::Gpu(t.clone()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => Storage::Cuda(t.clone()),
        }
    }
}

impl Storage {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Storage::Cpu(arr) => arr.shape().to_vec(),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.shape(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => t.shape(),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Storage::Cpu(arr) => arr.len(),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.numel(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => t.numel(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => Device::Gpu(t.device_id()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => Device::Cuda(t.device_id()),
        }
    }

    /// Extract CPU data — clones data. For zero-copy access, use as_cpu() or data_arc().
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub fn to_cpu(&self) -> Result<ArrayD<f32>, GpuError> {
        match self {
            Storage::Cpu(arr) => Ok(arr.as_ref().clone()),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.to_cpu(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => {
                let rt = crate::gpu::CudaRuntime::global()
                    .map_err(|e| GpuError::Device(format!("CUDA to_cpu: {e}")))?;
                let data = rt
                    .stream
                    .clone_dtoh_sync(t.buffer())
                    .map_err(|e| GpuError::Device(format!("CUDA to_cpu readback: {e}")))?;
                let arr = ArrayD::from_shape_vec(t.shape(), data)
                    .map_err(|e| GpuError::Device(format!("CUDA to_cpu shape: {e}")))?;
                Ok(arr)
            }
        }
    }

    /// Extract CPU data — clones data (CPU-only path).
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    pub fn to_cpu(&self) -> ArrayD<f32> {
        match self {
            Storage::Cpu(arr) => arr.as_ref().clone(),
        }
    }

    /// Return CPU reference if available
    pub fn as_cpu(&self) -> Option<&ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Some(arr.as_ref()),
            #[cfg(feature = "gpu")]
            Storage::Gpu(_) => None,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => None,
        }
    }

    pub fn as_cpu_mut(&mut self) -> Option<&mut ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Some(Arc::make_mut(arr)),
            #[cfg(feature = "gpu")]
            Storage::Gpu(_) => None,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => None,
        }
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub fn into_cpu(self) -> Result<ArrayD<f32>, GpuError> {
        match self {
            Storage::Cpu(arr) => Ok(Arc::unwrap_or_clone(arr)),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.to_cpu(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => {
                let rt = crate::gpu::CudaRuntime::global()
                    .map_err(|e| GpuError::Device(format!("CUDA into_cpu: {e}")))?;
                let data = rt
                    .stream
                    .clone_dtoh_sync(t.buffer())
                    .map_err(|e| GpuError::Device(format!("CUDA into_cpu readback: {e}")))?;
                let arr = ArrayD::from_shape_vec(t.shape(), data)
                    .map_err(|e| GpuError::Device(format!("CUDA into_cpu shape: {e}")))?;
                Ok(arr)
            }
        }
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    pub fn into_cpu(self) -> ArrayD<f32> {
        match self {
            Storage::Cpu(arr) => Arc::unwrap_or_clone(arr),
        }
    }

    /// Returns Arc<ArrayD<f32>> — O(1) for CPU, O(N) readback for GPU.
    /// Panics on GPU readback failure. Use try_data_arc() for fallback.
    pub fn data_arc(&self) -> Arc<ArrayD<f32>> {
        self.try_data_arc().expect("GPU/CUDA readback failed in data_arc")
    }

    /// Returns Arc<ArrayD<f32>> — O(1) for CPU, O(N) readback for GPU.
    /// Returns Ok for CPU, Err on GPU/CUDA readback failure.
    pub fn try_data_arc(&self) -> Result<Arc<ArrayD<f32>>, Box<dyn std::error::Error>> {
        match self {
            Storage::Cpu(arr) => Ok(Arc::clone(arr)),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => {
                let cpu = t.to_cpu().map_err(|e| {
                    format!("GPU readback failed in try_data_arc: {e}")
                })?;
                Ok(Arc::new(cpu))
            }
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => {
                let rt = crate::gpu::CudaRuntime::global()
                    .ok_or_else(|| "CUDA runtime not initialized for try_data_arc".to_string())?;
                let data = rt
                    .stream
                    .clone_dtoh_sync(t.buffer())
                    .map_err(|e| format!("CUDA readback failed in try_data_arc: {e}"))?;
                let arr = ArrayD::from_shape_vec(t.shape(), data)
                    .map_err(|e| format!("CUDA data_arc shape mismatch: {e}"))?;
                Ok(Arc::new(arr))
            }
            #[cfg(not(any(feature = "gpu", feature = "cuda")))]
            _ => unreachable!(),
        }
    }
}

impl From<ArrayD<f32>> for Storage {
    fn from(arr: ArrayD<f32>) -> Self {
        Storage::Cpu(Arc::new(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::TensorOps;

    #[test]
    fn test_device_creation() {
        let cpu = Device::Cpu;
        assert_eq!(cpu, Device::Cpu);
        assert_eq!(format!("{}", cpu), "cpu");
        #[cfg(feature = "gpu")]
        {
            let gpu0 = Device::Gpu(0);
            assert_eq!(format!("{}", gpu0), "gpu:0");
            assert_eq!(gpu0, Device::Gpu(0));
            assert_ne!(gpu0, Device::Gpu(1));
        }
    }

    #[test]
    fn test_device_default() {
        let d: Device = Default::default();
        assert_eq!(d, Device::Cpu);
    }

    #[test]
    fn test_storage_cpu_basics() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let storage: Storage = arr.clone().into();

        assert_eq!(storage.shape(), vec![2, 3]);
        assert_eq!(storage.numel(), 6);
        assert_eq!(storage.ndim(), 2);
        assert_eq!(storage.device(), Device::Cpu);
        #[cfg(feature = "gpu")]
        assert_eq!(storage.to_cpu().unwrap(), arr);
        #[cfg(not(feature = "gpu"))]
        assert_eq!(storage.to_cpu(), arr);
        assert!(storage.as_cpu().is_some());
    }

    #[test]
    fn test_storage_into_cpu() {
        let arr = ArrayD::zeros(vec![4]);
        let storage: Storage = arr.clone().into();
        #[cfg(feature = "gpu")]
        let recovered = storage.into_cpu().unwrap();
        #[cfg(not(feature = "gpu"))]
        let recovered = storage.into_cpu();
        assert_eq!(recovered, arr);
    }

    #[test]
    fn test_tensor_device_tracking() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(!t.is_cuda());
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.shape(), vec![3]);
        assert_eq!(t.numel(), 3);
    }

    #[test]
    fn test_tensor_to_device_cpu_roundtrip() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let t2 = t.to_device(&Device::Cpu);
        assert_eq!(t.data(), t2.data());
        assert_eq!(t2.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_backward_still_works() {
        let a = Tensor::randn(&[2, 3], true);
        let b = Tensor::randn(&[3, 4], true);
        let c = a.matmul(&b).sum();
        c.backward();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
        assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
        assert_eq!(b.grad().unwrap().shape(), &[3, 4]);
    }

    #[test]
    fn test_tensor_debug_format() {
        let t = Tensor::from_slice(&[1.0], &[1]);
        let debug_str = format!("{:?}", t);
        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("device: Cpu"));
        assert!(debug_str.contains("shape: [1]"));
    }
}
