use ndarray::ArrayD;
use std::fmt;

/// Physical device where tensor data resides
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// Cross-platform GPU (wgpu backend: NVIDIA, AMD, Intel, Apple Silicon)
    #[cfg(feature = "gpu")]
    Gpu(usize),
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
            #[cfg(feature = "gpu")]
            Device::Gpu(id) => write!(f, "gpu:{}", id),
        }
    }
}

/// Internal tensor storage — wraps concrete array types per device
#[derive(Clone)]
pub enum Storage {
    Cpu(ArrayD<f32>),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuTensor),
}

impl Storage {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Storage::Cpu(arr) => arr.shape().to_vec(),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.shape(),
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Storage::Cpu(arr) => arr.len(),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.numel(),
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
        }
    }

    /// Extract CPU data — panics if on GPU without conversion
    pub fn to_cpu(&self) -> ArrayD<f32> {
        match self {
            Storage::Cpu(arr) => arr.clone(),
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.to_cpu(),
        }
    }

    /// Return CPU reference if available
    pub fn as_cpu(&self) -> Option<&ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Some(arr),
            #[cfg(feature = "gpu")]
            Storage::Gpu(_) => None,
        }
    }

    pub fn as_cpu_mut(&mut self) -> Option<&mut ArrayD<f32>> {
        match self {
            Storage::Cpu(arr) => Some(arr),
            #[cfg(feature = "gpu")]
            Storage::Gpu(_) => None,
        }
    }

    pub fn into_cpu(self) -> ArrayD<f32> {
        match self {
            Storage::Cpu(arr) => arr,
            #[cfg(feature = "gpu")]
            Storage::Gpu(t) => t.to_cpu(),
        }
    }
}

impl From<ArrayD<f32>> for Storage {
    fn from(arr: ArrayD<f32>) -> Self {
        Storage::Cpu(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Tensor;
    use crate::autograd::TensorOps;

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
        let arr = ArrayD::from_shape_vec(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
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
