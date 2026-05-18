//! GPU compute backend using wgpu.
//! Cross-platform: NVIDIA, AMD, Intel, Apple Silicon.
//! Only compiles with `feature = "gpu"`.

use std::sync::OnceLock;

use ndarray::ArrayD;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No GPU adapter found")]
    NoAdapter,
    #[error("GPU device error: {0}")]
    Device(String),
    #[error("Buffer creation failed: {0}")]
    Buffer(String),
    #[error("Compute shader error: {0}")]
    Shader(String),
    #[error("GPU not initialized. Call GpuContext::init() first")]
    NotInitialized,
}

/// Global GPU context — lazily initialized once
static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

/// WGPU device + queue singleton
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(GpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Nexora GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;
        Ok(Self { device, queue })
    }

    /// Initialize the global GPU context. Safe to call multiple times.
    pub fn init() -> Result<&'static Self, GpuError> {
        GPU_CTX.get_or_try_init(Self::new_blocking)
    }

    /// Get the global GPU context (must init first)
    pub fn global() -> Result<&'static Self, GpuError> {
        GPU_CTX.get().ok_or(GpuError::NotInitialized)
    }

    /// Check if GPU is available (already initialized)
    pub fn is_available() -> bool {
        GPU_CTX.get().is_some()
    }
}

/// A tensor whose data lives on a GPU device
#[derive(Clone)]
pub struct GpuTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) buffer: wgpu::Buffer,
}

// SAFETY: GpuTensor is Send + Sync because wgpu::Buffer is Send + Sync
unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

impl GpuTensor {
    fn ctx() -> Result<&'static GpuContext, GpuError> {
        GpuContext::global()
    }

    /// Allocate a new GPU tensor from CPU data
    pub fn from_cpu(data: &ArrayD<f32>) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let shape = data.shape().to_vec();
        let flat: &[u8] = bytemuck::cast_slice(data.as_slice().ok_or_else(|| {
            GpuError::Buffer("non-contiguous array".into())
        })?);

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::from_cpu"),
            size: flat.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ctx.queue.write_buffer(&buffer, 0, flat);
        Ok(Self { shape, buffer })
    }

    /// Create a zero-initialized GPU tensor
    pub fn zeros(shape: &[usize]) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let numel: usize = shape.iter().product();
        let byte_size = (numel * 4) as u64; // f32 = 4 bytes

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::zeros"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Zero-fill via staging buffer
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::zeros_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.fill(0);
        }
        staging.unmap();

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &buffer, 0, byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        Ok(Self {
            shape: shape.to_vec(),
            buffer,
        })
    }

    /// Download tensor data back to CPU
    pub fn to_cpu(&self) -> ArrayD<f32> {
        let ctx = Self::ctx().expect("GPU context not initialized");
        let byte_size = (self.numel() * 4) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::to_cpu_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        // Block on readback
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("channel closed").expect("buffer mapping failed");

        let data: &[f32] = bytemuck::cast_slice(&slice.get_mapped_range());
        let result = ArrayD::from_shape_vec(self.shape.clone(), data.to_vec())
            .expect("shape mismatch in GPU→CPU transfer");
        staging.unmap();
        result
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn device_id(&self) -> usize {
        0 // single GPU for now
    }

    /// Buffer reference for compute shader bindings
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
