use std::sync::Arc;

use cudarc::driver::CudaSlice;

/// GPU tensor backed by CUDA device memory.
#[derive(Clone, Debug)]
pub struct CudaTensor {
    pub shape: Vec<usize>,
    pub(crate) buffer: CudaSlice<f32>,
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
}
