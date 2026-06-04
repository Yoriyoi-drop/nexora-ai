use ndarray::ArrayD;
use std::sync::mpsc;
use std::time::Duration;

use super::gpu_types::{GpuContext, GpuError};

const GPU_READBACK_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) dtype: GpuDtype,
    /// Which GPU device this tensor lives on. In the current single-GPU
    /// singleton setup this is always 0, but the field is available for
    /// multi-GPU extensions.
    pub(crate) device_id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuDtype {
    F32,
    F16,
    Bf16,
    /// Packed int8: 4 signed 8-bit values per u32 element.
    /// Used for quantized weight storage in GPU memory.
    I8,
}

impl GpuDtype {
    /// Bytes per element for this dtype.
    /// F32=4, F16=2, Bf16=2, I8=1 (packed 4 int8 per u32, but logically 1 byte per element).
    pub fn byte_size(&self) -> usize {
        match self {
            GpuDtype::F32 => 4,
            GpuDtype::F16 => 2,
            GpuDtype::Bf16 => 2,
            GpuDtype::I8 => 1,
        }
    }
}

// SAFETY:
// `GpuTensor` contains:
// - `shape: Vec<usize>` — trivially `Send + Sync`
// - `buffer: wgpu::Buffer` — a wgpu GPU-resource handle. wgpu's `Buffer` is
//   `Send + Sync` because it is reference-counted and thread-safe internally.
// - `dtype: GpuDtype` — a `Copy` enum, trivially `Send + Sync`.
//
// All GPU operations (reads, writes, compute dispatches) are sequenced through
// `GpuContext`'s single queue, so concurrent access to distinct `GpuTensor`
// instances is safe. The Rust compiler cannot verify this automatically because
// `wgpu::Buffer` does not implement `Send + Sync` in wgpu's type system, but
// wgpu's own documentation states that Buffer handles are safe to send across
// threads as long as the device is not dropped. We hold a reference to the
// `GpuContext` via the global `OnceCell`, guaranteeing the device outlives all
// tensors.
unsafe impl Send for GpuTensor {}
// SAFETY: Same reasoning as `Send`. Shared references to `GpuTensor` are safe
// because `buffer` is an atomic ref-counted handle and all mutation goes through
// `GpuContext`'s synchronized queue.
unsafe impl Sync for GpuTensor {}

/// Poll GPU with 30s timeout and wait for map_async result.
/// Prevents threads from hanging forever on GPU readback.
pub(crate) fn readback_with_timeout(
    device: &wgpu::Device,
    rx: &mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
) -> Result<(), GpuError> {
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    });
    rx.recv_timeout(Duration::from_secs(30))
        .map_err(|_| GpuError::Timeout("GPU readback timed out after 30s".into()))?
        .map_err(|e| GpuError::Device(format!("Buffer mapping failed: {e:?}")))?;
    Ok(())
}

impl GpuTensor {
    fn ctx() -> Result<&'static GpuContext, GpuError> {
        GpuContext::global()
    }

    pub fn from_cpu(data: &ArrayD<f32>) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let shape = data.shape().to_vec();
        let data_slice = data.as_slice().ok_or_else(|| {
            GpuError::Buffer("non-contiguous array in from_cpu — copy to contiguous first".into())
        })?;
        let flat: &[u8] = bytemuck::cast_slice(data_slice);

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::from_cpu"),
            size: flat.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let byte_len = flat.len() as u64;
        ctx.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data_slice));
        crate::gpu::gpu_observability::PCIE_WRITE_BYTES
            .fetch_add(byte_len, std::sync::atomic::Ordering::Relaxed);
        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Create an int8 GpuTensor from pre-packed u32 data.
    ///
    /// `data` contains 4 signed int8 values packed per u32 element (little-endian).
    /// `shape` is the logical shape in *int8 elements* (not packed elements).
    /// The underlying buffer will have `(numel() + 3) / 4 * 4` bytes.
    pub fn from_cpu_i8_packed(shape: Vec<usize>, packed_data: &[u32]) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let expected_packed = (shape.iter().product::<usize>() + 3) / 4;
        if packed_data.len() < expected_packed {
            return Err(GpuError::Buffer(format!(
                "from_cpu_i8_packed: expected at least {} packed u32s for shape {:?}, got {}",
                expected_packed,
                shape,
                packed_data.len()
            )));
        }
        let byte_size = expected_packed as u64 * 4;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::from_cpu_i8_packed"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(
            &buffer,
            0,
            bytemuck::cast_slice(&packed_data[..expected_packed]),
        );
        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::I8,
            device_id: 0,
        })
    }

    /// Create a GpuTensor from packed Q4 bytes.
    ///
    /// `packed_bytes` is the output of `quantize_fp32_to_q4`: each byte holds
    /// 2 Q4 values (low nibble = even K, high nibble = odd K).
    /// Shape is the original weight matrix shape `[K, N]` (inner × outer).
    /// Buffer stores 4 bytes per u32, matching the WGSL `array<u32>` binding.
    pub fn from_cpu_q4_packed(shape: Vec<usize>, packed_bytes: &[u8]) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let num_bytes = shape[0] * shape[1] / 2; // K/2 bytes per column
        if packed_bytes.len() < num_bytes {
            return Err(GpuError::Buffer(format!(
                "from_cpu_q4_packed: expected {num_bytes} bytes for shape {shape:?}, got {}",
                packed_bytes.len()
            )));
        }
        let u32_count = (num_bytes + 3) / 4;
        let byte_size = u32_count as u64 * 4;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::from_cpu_q4_packed"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buffer, 0, packed_bytes);
        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::I8,
            device_id: 0,
        })
    }

    /// Create a GpuTensor from a flat f32 slice + shape.
    /// Avoids the ndarray ArrayD allocation required by `from_cpu`.
    pub fn from_slice(shape: Vec<usize>, data: &[f32]) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let numel: usize = shape.iter().product();
        if data.len() < numel {
            return Err(GpuError::Buffer(format!(
                "GpuTensor::from_slice: data len {} < numel {}",
                data.len(),
                numel
            )));
        }
        let byte_len = (numel * 4) as u64;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor::from_slice"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data[..numel]));
        crate::gpu::gpu_observability::PCIE_WRITE_BYTES
            .fetch_add(byte_len, std::sync::atomic::Ordering::Relaxed);
        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
        })
    }

    /// Create a GpuTensor from raw components (buffer + shape).
    /// Useful for wrapping GPU buffers allocated externally.
    pub fn from_raw(shape: Vec<usize>, buffer: wgpu::Buffer, dtype: GpuDtype) -> Self {
        Self {
            shape,
            buffer,
            dtype,
            device_id: 0,
        }
    }

    /// Non-blocking GPU→CPU readback via channel. Hasil siap ketika GPU selesai.
    /// Uses dtype-aware byte sizing for F16/Bf16/I8 support.
    pub fn to_cpu_async(
        &self,
        ctx: &GpuContext,
    ) -> Result<crate::gpu_async::AsyncReadback<Vec<f32>>, GpuError> {
        let byte_size = (self.numel() * self.dtype.byte_size()) as u64;
        ctx.readback_f32_async(&self.buffer, byte_size)
    }

    /// Create a new GpuTensor with a different shape but the same underlying buffer.
    /// Useful for reshaping (e.g., [seq, kv_heads * dim] -> [batch, heads, seq, dim]).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<GpuTensor, GpuError> {
        let expected: usize = new_shape.iter().product();
        if expected != self.shape.iter().product::<usize>() {
            return Err(GpuError::ShapeMismatch(format!(
                "reshape: {:?} -> {:?} element count mismatch",
                self.shape, new_shape
            )));
        }
        Ok(GpuTensor {
            shape: new_shape,
            buffer: self.buffer.clone(),
            dtype: self.dtype,
            device_id: self.device_id,
        })
    }

    /// Create a zero-copy view with a different shape WITHOUT checking element count.
    /// Useful for sub-views of pre-allocated buffers (e.g., KV cache with capacity > seq_len).
    pub fn view_as(&self, new_shape: Vec<usize>) -> GpuTensor {
        GpuTensor {
            shape: new_shape,
            buffer: self.buffer.clone(),
            dtype: self.dtype,
            device_id: self.device_id,
        }
    }

    fn return_staging(ctx: &GpuContext, buffer: wgpu::Buffer, _byte_size: u64, _readable: bool) {
        buffer.destroy();
        let _ = ctx;
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, GpuError> {
        Self::zeros_dtype(shape, GpuDtype::F32)
    }

    /// Create a zero-filled GPU tensor with the specified dtype.
    /// F16 tensors use 2× less VRAM than F32.
    pub fn zeros_dtype(shape: &[usize], dtype: GpuDtype) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let numel: usize = shape.iter().product();
        let esize = dtype.byte_size();
        let byte_size = (numel * esize) as u64;

        // Align to 4 bytes for u32 fill compatibility
        let aligned_size = ((byte_size + 3) / 4) * 4;

        let buffer = ctx
            .alloc_buffer(
                aligned_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            )
            .buffer;

        let tmp = Self {
            shape: shape.to_vec(),
            buffer: buffer.clone(),
            dtype,
            device_id: 0,
        };
        // Zero-fill via u32 (works for any dtype since 0 in any format is 0u32)
        ctx.fill_zero_u32(&tmp)?;

        Ok(Self {
            shape: shape.to_vec(),
            buffer,
            dtype,
            device_id: 0,
        })
    }

    /// Create a GPU tensor filled with 1.0 — avoids CPU round-trip
    pub fn ones(shape: &[usize]) -> Result<Self, GpuError> {
        let t = Self::zeros(shape)?;
        let ctx = Self::ctx()?;
        ctx.fill_constant(&t, 1.0)?;
        Ok(t)
    }

    /// Create a GPU tensor filled with 1.0, with specified dtype.
    pub fn ones_dtype(shape: &[usize], dtype: GpuDtype) -> Result<Self, GpuError> {
        let t = Self::zeros_dtype(shape, dtype)?;
        let ctx = Self::ctx()?;
        ctx.fill_constant(&t, 1.0)?;
        Ok(t)
    }

    fn readback_inner(&self, offset: u64, size: u64) -> Result<(Vec<u8>, wgpu::Buffer), GpuError> {
        let ctx = Self::ctx()?;
        let device = &ctx.device;
        let queue = &ctx.queue;

        // Block if too many concurrent readbacks pending (OOM prevention)
        let acquired = ctx.readback_limiter.acquire(Duration::from_secs(30));
        if !acquired {
            return Err(GpuError::Timeout(
                "GPU readback limiter timed out (30s)".into(),
            ));
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(self.buffer(), offset, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(GPU_READBACK_TIMEOUT_SECS);
        let map_result = loop {
            device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(100)),
            });
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(result) => break result,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if std::time::Instant::now() > deadline {
                        break Err(wgpu::BufferAsyncError);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                Err(_) => break Err(wgpu::BufferAsyncError),
            }
        };
        map_result.map_err(|_| GpuError::Timeout("GPU readback".into()))?;

        let mapped = slice.get_mapped_range();
        let data = mapped.to_vec();
        drop(mapped);
        staging.unmap();

        ctx.readback_limiter.release();

        crate::gpu::gpu_observability::PCIE_READ_BYTES
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);

        Ok((data, staging))
    }

    pub fn to_cpu(&self) -> Result<ArrayD<f32>, GpuError> {
        let ctx = Self::ctx()?;
        ctx.flush();
        let byte_size = (self.numel() * self.dtype.byte_size()) as u64;

        let (data, staging) = self.readback_inner(0, byte_size)?;

        let result = ArrayD::from_shape_vec(self.shape.clone(), {
            let f32_data: &[f32] = bytemuck::cast_slice(&data);
            f32_data.to_vec()
        })
        .map_err(|e| GpuError::ShapeMismatch(format!("to_cpu shape mismatch: {e}")))?;

        Self::return_staging(ctx, staging, byte_size, true);
        Ok(result)
    }

    /// Read back a slice of raw bytes from the buffer at a given offset.
    /// Useful for reading individual token positions from KV cache buffers
    /// without reading the full tensor (avoids large PCIe transfers).
    /// `size` should be calculated with self.dtype.byte_size().
    pub fn to_cpu_raw_bytes_slice(&self, offset: u64, size: u64) -> Result<Vec<u8>, GpuError> {
        let ctx = Self::ctx()?;
        ctx.flush();
        let (data, staging) = self.readback_inner(offset, size)?;
        Self::return_staging(ctx, staging, size, true);
        Ok(data)
    }

    /// Read back raw bytes (for u32 output buffers like sampler tokens)
    pub fn to_cpu_raw_bytes(&self) -> Result<Vec<u8>, GpuError> {
        let ctx = Self::ctx()?;
        ctx.flush();
        let byte_size = (self.numel() * self.dtype.byte_size()) as u64;

        let (data, staging) = self.readback_inner(0, byte_size)?;
        Self::return_staging(ctx, staging, byte_size, true);
        Ok(data)
    }

    /// Read back only the first element for quick validation
    /// Much faster than full tensor readback
    pub fn to_cpu_first_element(&self) -> Result<f32, GpuError> {
        let ctx = Self::ctx()?;
        ctx.flush();

        let (data, staging) = self.readback_inner(0, 4)?;
        let value: f32 = bytemuck::cast_slice::<_, f32>(&data)[0];
        Self::return_staging(ctx, staging, 4, true);
        Ok(value)
    }

    /// Read back a checksum of the tensor for validation
    /// Faster than full readback for large tensors
    pub fn to_cpu_checksum(&self) -> Result<f64, GpuError> {
        let ctx = Self::ctx()?;
        ctx.flush();
        let byte_size = (self.numel() * self.dtype.byte_size()) as u64;

        let (data, staging) = self.readback_inner(0, byte_size)?;
        let f32_data: &[f32] = bytemuck::cast_slice(&data);
        let checksum: f64 = f32_data.iter().map(|&x| x as f64).sum();
        Self::return_staging(ctx, staging, byte_size, true);
        Ok(checksum)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn dtype(&self) -> GpuDtype {
        self.dtype
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// GPU max element reduction
    pub fn max_element_gpu(&self) -> Result<f32, GpuError> {
        let ctx = Self::ctx()?;
        let numel = self.numel();
        if numel == 0 {
            return Ok(f32::NEG_INFINITY);
        }
        if numel == 1 {
            return self.to_cpu_first_element();
        }
        // Use proper GPU reduction via already-compiled WGSL shader
        let result_tensor = ctx.max(self)?;
        result_tensor.to_cpu_first_element()
    }

    /// GPU sum reduction
    pub fn sum_gpu(&self) -> Result<f32, GpuError> {
        let ctx = Self::ctx()?;
        let numel = self.numel();
        if numel == 0 {
            return Ok(0.0);
        }
        if numel == 1 {
            return self.to_cpu_first_element();
        }
        // Use proper GPU reduction via already-compiled WGSL shader
        let result_tensor = ctx.sum(self)?;
        result_tensor.to_cpu_first_element()
    }
}
