use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

/// GPU tensor backed by CUDA device memory.
#[derive(Clone, Debug)]
pub struct CudaTensor {
    pub shape: Vec<usize>,
    pub buffer: CudaSlice<f32>,
    pub device_id: usize,
}

unsafe impl Send for CudaTensor {}
unsafe impl Sync for CudaTensor {}

impl CudaTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = new_shape.iter().product();
        if expected != self.numel() {
            return Err(format!(
                "CudaTensor reshape: {:?} -> {:?} element count mismatch ({} vs {})",
                self.shape, new_shape, self.numel(), expected
            ));
        }
        Ok(CudaTensor {
            shape: new_shape,
            buffer: self.buffer.clone(),
            device_id: self.device_id,
        })
    }

    pub fn buffer(&self) -> &CudaSlice<f32> {
        &self.buffer
    }

    /// Create a CUDA tensor from host data.
    pub fn from_cpu(stream: &Arc<CudaStream>, shape: Vec<usize>, data: &[f32], device_id: usize) -> Result<Self, String> {
        let numel: usize = shape.iter().product();
        if data.len() < numel {
            return Err(format!(
                "CudaTensor::from_cpu: data len {} < numel {}",
                data.len(),
                numel
            ));
        }
        let buffer = stream
            .clone_htod(&data[..numel])
            .map_err(|e| format!("CudaTensor::from_cpu htod failed: {e}"))?;
        Ok(CudaTensor {
            shape,
            buffer,
            device_id,
        })
    }

    /// Create a zero-filled CUDA tensor directly on GPU — no CPU Vec<f32> intermediate.
    pub fn zeros(stream: &Arc<CudaStream>, shape: Vec<usize>, device_id: usize) -> Result<Self, String> {
        let numel: usize = shape.iter().product();
        let buffer = stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CudaTensor::zeros alloc: {e}"))?;
        Ok(CudaTensor {
            shape,
            buffer,
            device_id,
        })
    }

    /// Read tensor data back to host.
    pub fn to_cpu_vec(&self, stream: &Arc<CudaStream>) -> Result<Vec<f32>, String> {
        stream
            .clone_dtoh(&self.buffer)
            .map_err(|e| format!("CudaTensor::to_cpu_vec dtoh failed: {e}"))
    }
}
