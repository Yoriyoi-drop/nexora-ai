use cudarc::driver::{CudaDevice, CudaSlice};

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

    /// Create a CUDA tensor from host data.
    pub fn from_cpu(device: &CudaDevice, shape: Vec<usize>, data: &[f32]) -> Result<Self, String> {
        let numel: usize = shape.iter().product();
        if data.len() < numel {
            return Err(format!(
                "CudaTensor::from_cpu: data len {} < numel {}",
                data.len(),
                numel
            ));
        }
        let buffer = device
            .htod_sync_copy(&data[..numel])
            .map_err(|e| format!("CudaTensor::from_cpu htod failed: {e}"))?;
        Ok(CudaTensor {
            shape,
            buffer,
            device_id: device.id(),
        })
    }

    /// Read tensor data back to host.
    pub fn to_cpu_vec(&self, device: &CudaDevice) -> Result<Vec<f32>, String> {
        let numel = self.numel();
        let mut host = vec![0.0f32; numel];
        device
            .dtoh_sync_copy(&self.buffer, &mut host)
            .map_err(|e| format!("CudaTensor::to_cpu_vec dtoh failed: {e}"))?;
        Ok(host)
    }
}
