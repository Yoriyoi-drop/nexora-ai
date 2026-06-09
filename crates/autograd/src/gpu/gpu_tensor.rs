use ndarray::ArrayD;
use std::sync::mpsc;
use std::time::Duration;

use super::gpu_types::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use super::cuda::CudaTensor;

/// Zero-overhead placeholder — `cuda_tensor: None` compiles regardless of feature.
#[cfg(not(feature = "cuda"))]
type CudaTensorRef = Option<()>;
#[cfg(feature = "cuda")]
type CudaTensorRef = Option<CudaTensor>;

/// Convert a Vec<u8> of f32 bytes into Vec<f32>, safely handling alignment.
fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

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
    /// Optional CUDA tensor backing for zero-copy CUDA op access.
    /// When `Some`, CUDA ops can read/write this tensor without going through
    /// the wgpu→CPU→CUDA bridge. Eliminates the 500μs-2ms PCIe round-trip
    /// for subsequent CUDA operations on the same tensor.
    /// Type is `Option<CudaTensor>` when cuda feature is on, `Option<()>` otherwise.
    /// Both support `cuda_tensor: None` — no cfg needed at construction sites.
    pub(crate) cuda_tensor: CudaTensorRef,
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

// `GpuTensor` is automatically `Send + Sync` because:
// - `wgpu::Buffer` implements `Send + Sync` on native platforms (cfg send_sync is always
//   true on non-wasm targets in wgpu 29+, via atomic-refcounted inner handles).
// - `Vec<usize>` and `GpuDtype` are trivially `Send + Sync`.
// - All GPU operations are sequenced through `GpuContext`'s single queue.

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
        let byte_len = flat.len() as u64;

        // 🟠 MEDIUM #7: Use memory pool instead of direct device.create_buffer
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = ctx.alloc_or_create_buffer(byte_len, usage);
        ctx.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data_slice));
        crate::gpu::gpu_observability::PCIE_WRITE_BYTES
            .fetch_add(byte_len, std::sync::atomic::Ordering::Relaxed);

        #[cfg(feature = "cuda")]
        let cuda_tensor: CudaTensorRef = if let Some(cuda) = ctx.cuda {
            match CudaTensor::from_cpu(
                &cuda.stream, shape.clone(), data_slice, cuda.device_id,
            ) {
                Ok(ct) => {
                    ctx.cache_cuda(&buffer, ct.clone());
                    Some(ct)
                }
                Err(e) => {
                    tracing::warn!("Failed to create CUDA tensor from CPU data: {e}");
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_tensor: CudaTensorRef = None;

        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
            cuda_tensor,
        })
    }

    /// Create an int8 GpuTensor from pre-packed u32 data.
    ///
    /// `data` contains 4 signed int8 values packed per u32 element (little-endian).
    /// `shape` is the logical shape in *int8 elements* (not packed elements).
    /// The underlying buffer will have `(numel() + 3) / 4 * 4` bytes.
    pub fn from_cpu_i8_packed(shape: Vec<usize>, packed_data: &[u32]) -> Result<Self, GpuError> {
        let ctx = Self::ctx()?;
        let expected_packed = shape.iter().product::<usize>().div_ceil(4);
        if packed_data.len() < expected_packed {
            return Err(GpuError::Buffer(format!(
                "from_cpu_i8_packed: expected at least {} packed u32s for shape {:?}, got {}",
                expected_packed,
                shape,
                packed_data.len()
            )));
        }
        let byte_size = expected_packed as u64 * 4;
        // 🟠 MEDIUM #7: Use memory pool instead of direct device.create_buffer
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = ctx.alloc_or_create_buffer(byte_size, usage);
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
            cuda_tensor: None,
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
        let u32_count = num_bytes.div_ceil(4);
        let byte_size = u32_count as u64 * 4;
        // 🟠 MEDIUM #7: Use memory pool instead of direct device.create_buffer
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = ctx.alloc_or_create_buffer(byte_size, usage);
        ctx.queue.write_buffer(&buffer, 0, packed_bytes);
        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::I8,
            device_id: 0,
            cuda_tensor: None,
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

        // 🟠 MEDIUM #7: Use memory pool instead of direct device.create_buffer
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = ctx.alloc_or_create_buffer(byte_len, usage);
        ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data[..numel]));
        crate::gpu::gpu_observability::PCIE_WRITE_BYTES
            .fetch_add(byte_len, std::sync::atomic::Ordering::Relaxed);

        #[cfg(feature = "cuda")]
        let cuda_tensor: CudaTensorRef = if let Some(cuda) = ctx.cuda {
            match CudaTensor::from_cpu(
                &cuda.stream, shape.clone(), &data[..numel], cuda.device_id,
            ) {
                Ok(ct) => {
                    ctx.cache_cuda(&buffer, ct.clone());
                    Some(ct)
                }
                Err(e) => {
                    tracing::warn!("Failed to create CUDA tensor from slice: {e}");
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_tensor: CudaTensorRef = None;

        Ok(Self {
            shape,
            buffer,
            dtype: GpuDtype::F32,
            device_id: 0,
            cuda_tensor,
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
            cuda_tensor: None,
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

    /// Non-blocking GPU→CPU readback using global context (convenience).
    /// Equivalent to `tensor.to_cpu_async(&GpuContext::global()?)`.
    pub fn to_cpu_async_global(
        &self,
    ) -> Result<crate::gpu_async::AsyncReadback<Vec<f32>>, GpuError> {
        let ctx = Self::ctx()?;
        self.to_cpu_async(ctx)
    }

    /// Non-blocking raw bytes readback: starts GPU→staging copy, returns immediately.
    /// Call `recv()` on the result to block until data is ready.
    /// Useful for overlapping batch token readback with next batch compute.
    pub fn to_cpu_raw_bytes_async(
        &self,
        ctx: &GpuContext,
    ) -> Result<crate::gpu_async::AsyncReadback<Vec<u8>>, GpuError> {
        let byte_size = (self.numel() * self.dtype.byte_size()) as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("async_readback_raw_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("async_readback_raw_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready_clone = ready.clone();
        let staging_clone = staging.clone();

        staging.map_async(wgpu::MapMode::Read, 0..byte_size, move |result| {
            if let Ok(()) = result {
                let slice = staging_clone.slice(..byte_size);
                let data = slice.get_mapped_range().to_vec();
                let _ = tx.send(data);
                staging_clone.unmap();
                ready_clone.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        let _ = ctx.device.poll(wgpu::PollType::Poll);

        Ok(crate::gpu_async::AsyncReadback::new(rx, ready))
    }

    /// Non-blocking raw bytes readback using global context (convenience).
    pub fn to_cpu_raw_bytes_async_global(
        &self,
    ) -> Result<crate::gpu_async::AsyncReadback<Vec<u8>>, GpuError> {
        let ctx = Self::ctx()?;
        self.to_cpu_raw_bytes_async(ctx)
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
            cuda_tensor: self.cuda_tensor.clone(),
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
            cuda_tensor: self.cuda_tensor.clone(),
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
        let aligned_size = byte_size.div_ceil(4) * 4;

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
            cuda_tensor: None,
        };
        // Zero-fill via u32 (works for any dtype since 0 in any format is 0u32)
        ctx.fill_zero_u32(&tmp)?;

        Ok(Self {
            shape: shape.to_vec(),
            buffer,
            dtype,
            device_id: 0,
            cuda_tensor: None,
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

        let f32_vec = bytes_to_f32_vec(data);
        let result = ArrayD::from_shape_vec(self.shape.clone(), f32_vec)
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
        let value = f32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
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
        let f32_data = bytes_to_f32_vec(data);
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

    /// Clone the underlying wgpu::Buffer handle (cheap — internally ref-counted).
    pub fn buffer_arc(&self) -> wgpu::Buffer {
        self.buffer.clone()
    }

    /// Returns a reference to the inline CUDA tensor backing, if available.
    /// This is the zero-copy fast path — no wgpu→CPU→CUDA bridge needed.
    #[cfg(feature = "cuda")]
    pub(crate) fn get_cuda(&self) -> Option<&CudaTensor> {
        self.cuda_tensor.as_ref()
    }

    /// Attach a CUDA tensor backing for zero-copy CUDA ops.
    /// Element count must match this tensor's current shape.
    #[cfg(feature = "cuda")]
    pub fn set_cuda_tensor(&mut self, ct: CudaTensor) {
        debug_assert_eq!(self.numel(), ct.numel(),
            "set_cuda_tensor: element count mismatch (tensor {} vs ct {})",
            self.numel(), ct.numel());
        self.cuda_tensor = Some(ct);
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
