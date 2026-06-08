use ndarray::ArrayD;
use std::fmt;
use std::sync::Arc;

/// Physical device where tensor data resides
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// Cross-platform GPU (wgpu backend: NVIDIA, AMD, Intel, Apple Silicon)
    #[cfg(feature = "device-gpu")]
    Gpu(usize),
    /// NVIDIA CUDA GPU
    #[cfg(feature = "device-cuda")]
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "device-gpu")]
            Device::Gpu(id) => write!(f, "gpu:{}", id),
            #[cfg(feature = "device-cuda")]
            Device::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

/// Convenience: check if a device variant exists
impl Device {
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "device-gpu")]
        if matches!(self, Device::Gpu(_)) {
            return true;
        }
        #[cfg(feature = "device-cuda")]
        if matches!(self, Device::Cuda(_)) {
            return true;
        }
        false
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
}

/// Internal tensor storage — wraps concrete array types per device
/// CPU storage uses Arc for O(1) clone and zero-copy data access.
pub enum Storage {
    Cpu(Arc<ArrayD<f32>>),
    #[cfg(feature = "device-gpu")]
    Gpu(crate::gpu::GpuTensor, Vec<usize>),
    #[cfg(feature = "device-cuda")]
    Cuda(crate::gpu::cuda::CudaTensor, Vec<usize>),
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(arr) => Storage::Cpu(Arc::clone(arr)),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(tensor, shape) => Storage::Gpu(tensor.clone(), shape.clone()),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(tensor, shape) => Storage::Cuda(tensor.clone(), shape.clone()),
        }
    }
}

impl Storage {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Storage::Cpu(arr) => arr.shape().to_vec(),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, shape) => shape.clone(),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, shape) => shape.clone(),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Storage::Cpu(arr) => arr.len(),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, shape) => shape.iter().product(),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, shape) => shape.iter().product(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => Device::Gpu(0),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => Device::Cuda(0),
        }
    }

    pub fn to_cpu(&self) -> ArrayD<f32> {
        match self {
            Storage::Cpu(arr) => arr.as_ref().clone(),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => {
                tracing::warn!("Storage::to_cpu() called on Gpu variant - returning empty array");
                ArrayD::zeros(vec![0])
            }
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => {
                tracing::warn!("Storage::to_cpu() called on Cuda variant - returning empty array");
                ArrayD::zeros(vec![0])
            }
        }
    }

    pub fn as_cpu(&self) -> Option<&ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Some(arr.as_ref()),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => None,
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => None,
        }
    }

    pub fn as_cpu_mut(&mut self) -> Option<&mut ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Some(Arc::make_mut(arr)),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => None,
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => None,
        }
    }

    pub fn into_cpu(self) -> ArrayD<f32> {
        match self {
            Storage::Cpu(arr) => Arc::unwrap_or_clone(arr),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => panic!("Storage::into_cpu() called on Gpu variant - use data() for GPU readback"),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => panic!("Storage::into_cpu() called on Cuda variant - use data() for GPU readback"),
        }
    }

    pub fn data_arc(&self) -> Arc<ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Arc::clone(arr),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => panic!("Storage::data_arc() called on Gpu variant - not supported"),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => panic!("Storage::data_arc() called on Cuda variant - not supported"),
        }
    }

    pub fn try_data_arc(&self) -> Result<Arc<ArrayD<f32>>, Box<dyn std::error::Error>> {
        match self {
            Storage::Cpu(arr) => Ok(Arc::clone(arr)),
            #[cfg(feature = "device-gpu")]
            Storage::Gpu(_, _) => Err("Storage::try_data_arc() called on Gpu variant - not supported".into()),
            #[cfg(feature = "device-cuda")]
            Storage::Cuda(_, _) => Err("Storage::try_data_arc() called on Cuda variant - not supported".into()),
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
        assert_eq!(storage.to_cpu(), arr);
        assert!(storage.as_cpu().is_some());
    }

    #[test]
    fn test_storage_into_cpu() {
        let arr = ArrayD::zeros(vec![4]);
        let storage: Storage = arr.clone().into();
        let recovered = storage.into_cpu();
        assert_eq!(recovered, arr);
    }

    #[test]
    fn test_tensor_device_tracking() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
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
