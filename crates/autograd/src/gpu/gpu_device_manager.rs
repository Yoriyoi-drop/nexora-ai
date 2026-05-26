//! Multi-device GPU manager.
//!
//! Enumerates all physical GPU adapters and creates a [`GpuContext`] per device.
//! Gradient synchronization across devices uses the GPU Gradient AllReduce kernel.

use once_cell::sync::OnceCell;

use super::{GpuContext, GpuDtype, GpuError, GpuTensor};

/// Multi-device GPU manager singleton
static GPU_DEVICES: OnceCell<GpuDeviceManager> = OnceCell::new();

/// Describes a single physical GPU device
#[derive(Debug)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_id: u32,
}

/// Coordinates across multiple physical GPU devices.
pub struct GpuDeviceManager {
    devices: Vec<GpuContext>,
    infos: Vec<GpuDeviceInfo>,
}

impl GpuDeviceManager {
    pub fn init() -> Result<&'static Self, GpuError> {
        GPU_DEVICES.get_or_try_init(|| Self::new_blocking())
    }

    pub fn global() -> Result<&'static Self, GpuError> {
        GPU_DEVICES.get().ok_or(GpuError::NotInitialized)
    }

    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    pub fn device(&self, idx: usize) -> &GpuContext {
        &self.devices[idx]
    }

    pub fn device_info(&self) -> &[GpuDeviceInfo] {
        &self.infos
    }

    /// GPU-native gradient allreduce: flat buffer per param across replicas.
    /// Uses `GRADIENT_ALLREDUCE_WGSL` for on-device averaging.
    pub fn gpu_sync_gradients(
        &self,
        ctx: &GpuContext,
        grads: &[&GpuTensor],
    ) -> Result<(), GpuError> {
        let n = grads.len();
        if n <= 1 {
            return Ok(());
        }

        let numel = grads[0].numel();
        let total_elements = numel * n;

        let flat_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("allreduce_flat"),
            size: (total_elements * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("allreduce_copy"),
            });
        for (i, grad) in grads.iter().enumerate() {
            let offset = (i * numel * 4) as u64;
            enc.copy_buffer_to_buffer(grad.buffer(), 0, &flat_buffer, offset, (numel * 4) as u64);
        }
        ctx.queue.submit(Some(enc.finish()));

        let flat_tensor = GpuTensor {
            shape: vec![total_elements],
            buffer: flat_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("allreduce_out"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_tensor = GpuTensor {
            shape: vec![numel],
            buffer: out_buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        };
        ctx.gradient_allreduce(&flat_tensor, &out_tensor, n as u32)?;

        let mut cb = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("allreduce_copy_back"),
            });
        for grad in grads {
            cb.copy_buffer_to_buffer(
                &out_tensor.buffer(),
                0,
                grad.buffer(),
                0,
                (numel * 4) as u64,
            );
        }
        ctx.queue.submit(Some(cb.finish()));

        Ok(())
    }

    fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

        if adapters.is_empty() {
            return Err(GpuError::NoAdapter);
        }

        let mut devices = Vec::with_capacity(adapters.len());
        let mut infos = Vec::with_capacity(adapters.len());

        for (idx, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            let ctx = GpuContext::from_adapter(adapter).await?;
            devices.push(ctx);
            infos.push(GpuDeviceInfo {
                name: info.name,
                backend: info.backend,
                device_id: idx as u32,
            });
        }

        Ok(Self { devices, infos })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_init() {
        match GpuDeviceManager::init() {
            Ok(mgr) => {
                assert!(mgr.num_devices() >= 1);
                let info = mgr.device_info();
                assert!(!info.is_empty());
                assert!(!info[0].name.is_empty());
            }
            Err(e) => {
                eprintln!("GPU not available: {e}");
            }
        }
    }

    #[test]
    fn test_device_manager_global() {
        if GpuDeviceManager::init().is_ok() {
            assert!(GpuDeviceManager::global().is_ok());
        }
    }
}
