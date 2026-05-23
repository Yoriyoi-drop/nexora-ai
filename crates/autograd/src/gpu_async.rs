use std::sync::mpsc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::gpu::{readback_with_timeout, GpuContext, GpuDtype, GpuError, GpuTensor};

// ─── AsyncReadback ─────────────────────────────────────────────────────────────

/// Non-blocking GPU readback — hasil dikirim via channel ketika GPU selesai
pub struct AsyncReadback<T: Send + 'static> {
    pub receiver: mpsc::Receiver<T>,
    ready: Arc<AtomicBool>,
}

impl<T: Send + 'static> AsyncReadback<T> {
    /// Cek apakah hasil sudah siap (non-blocking)
    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// Blocking wait untuk hasil
    /// # Panics
    /// Panics if the sender was dropped before sending a value (GPU context destroyed).
    pub fn recv(&self) -> T {
        // safe: sender stays alive until value is sent; only fails on GPU context destruction
        self.receiver.recv().expect("GPU async readback channel disconnected — GPU context destroyed before readback completed")
    }

    /// Non-blocking check
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

impl GpuContext {
    /// Non-blocking GPU readback: copy buffer ke staging → map_async → hasil via channel
    pub fn readback_f32_async(
        &self,
        buffer: &wgpu::Buffer,
        size_bytes: u64,
    ) -> Result<AsyncReadback<Vec<f32>>, GpuError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("async_readback_staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("async_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size_bytes);
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let ready_clone = ready.clone();
        let staging_clone = staging.clone();

        staging.map_async(wgpu::MapMode::Read, 0..size_bytes, move |result| {
            if let Ok(()) = result {
                let slice = staging_clone.slice(..size_bytes);
                let data = slice.get_mapped_range();
                let floats: Vec<f32> = data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                drop(data);
                staging_clone.unmap();
                let _ = tx.send(floats);
                ready_clone.store(true, Ordering::Release);
            }
        });

        // Non-blocking poll — proses callback jika GPU sudah selesai
        let _ = self.device.poll(wgpu::PollType::Poll);

        Ok(AsyncReadback { receiver: rx, ready })
    }

    /// Proses callback async GPU tanpa blocking
    pub fn poll_async_callbacks(&self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Queue a GPU buffer → staging copy into the reusable encoder (batch-aware).
    /// The staging buffer is NOT yet mapped — call `complete_readback()` after submission.
    /// Use inside `begin_batch_mode()…end_batch_mode()` to pipeline copy with compute.
    pub fn create_readback_staging(
        &self,
        source: &wgpu::Buffer,
        size_bytes: u64,
        label: &str,
    ) -> wgpu::Buffer {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("staging_{}", label)),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.with_encoder(|enc| {
            enc.copy_buffer_to_buffer(source, 0, &staging, 0, size_bytes);
        });
        staging
    }

    /// Complete an async readback: register `map_async` on a staging buffer that was
    /// previously created via `create_readback_staging()` and submitted via `flush()`.
    /// Returns immediately — call `try_recv()` after polling to get the data.
    pub fn complete_readback(&self, staging: &wgpu::Buffer, size_bytes: u64) -> AsyncReadback<Vec<f32>> {
        let (tx, rx) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let ready_clone = ready.clone();
        let staging_clone = staging.clone();

        staging.map_async(wgpu::MapMode::Read, 0..size_bytes, move |result| {
            if let Ok(()) = result {
                let slice = staging_clone.slice(..size_bytes);
                let data = slice.get_mapped_range();
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                drop(data);
                staging_clone.unmap();
                let _ = tx.send(floats);
                ready_clone.store(true, Ordering::Release);
            }
        });

        let _ = self.device.poll(wgpu::PollType::Poll);

        AsyncReadback { receiver: rx, ready }
    }

    /// Poll with a bounded timeout (nanoseconds). Non-blocking if timeout ≤ 0.
    pub fn poll_timeout(&self, timeout_ns: u64) {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_nanos(timeout_ns)),
        });
    }
}

// ─── GpuStagingPool ────────────────────────────────────────────────────────────

pub struct GpuStagingPool {
    write_buffers: Vec<wgpu::Buffer>,
    read_buffers: Vec<wgpu::Buffer>,
    max_size: u64,
}

impl GpuStagingPool {
    pub fn new(ctx: &GpuContext, max_size: u64) -> Self {
        let write_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_write"),
            size: max_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });
        let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_read"),
            size: max_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            write_buffers: vec![write_buf],
            read_buffers: vec![read_buf],
            max_size,
        }
    }

    pub fn acquire_write(&mut self, ctx: &GpuContext, size: u64) -> wgpu::Buffer {
        assert!(size <= self.max_size, "staging pool buffer too small");
        if let Some(buf) = self.write_buffers.pop() {
            buf
        } else {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_write_extra"),
                size: self.max_size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            })
        }
    }

    pub fn release_write(&mut self, buf: wgpu::Buffer) {
        self.write_buffers.push(buf);
    }

    pub fn acquire_read(&mut self, ctx: &GpuContext, size: u64) -> wgpu::Buffer {
        assert!(size <= self.max_size, "staging pool buffer too small");
        if let Some(buf) = self.read_buffers.pop() {
            buf
        } else {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_read_extra"),
                size: self.max_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        }
    }

    pub fn release_read(&mut self, buf: wgpu::Buffer) {
        self.read_buffers.push(buf);
    }
}

// ─── GpuTensor async helpers ──────────────────────────────────────────────────

/// Create a GpuTensor from CPU data using the staging pool.
/// Returns the tensor immediately; the copy may still be in flight.
pub fn tensor_from_cpu_async(
    ctx: &GpuContext,
    pool: &mut GpuStagingPool,
    data: &[f32],
    shape: Vec<usize>,
) -> Result<GpuTensor, GpuError> {
    let byte_size = (data.len() * 4) as u64;
    let storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("async_storage"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = pool.acquire_write(ctx, byte_size);
    ctx.queue
        .write_buffer(&staging, 0, bytemuck::cast_slice(data));

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("async_write_encoder"),
        });
    encoder.copy_buffer_to_buffer(&staging, 0, &storage, 0, byte_size);
    ctx.queue.submit(Some(encoder.finish()));

    pool.release_write(staging);

    Ok(GpuTensor {
        shape,
        buffer: storage,
        dtype: GpuDtype::F32,
    })
}

/// Read GPU tensor back to CPU asynchronously.
/// Submits a copy to staging, returns immediately.
/// Call `poll_tensor_to_cpu` later to get the result.
pub fn tensor_to_cpu_async<'a>(
    ctx: &GpuContext,
    pool: &mut GpuStagingPool,
    tensor: &GpuTensor,
) -> Result<CpuReadback, GpuError> {
    let byte_size = (tensor.numel() * 4) as u64;
    let staging = pool.acquire_read(ctx, byte_size);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("async_read_encoder"),
        });
    encoder.copy_buffer_to_buffer(tensor.buffer(), 0, &staging, 0, byte_size);
    let submission = ctx.queue.submit(Some(encoder.finish()));

    Ok(CpuReadback {
        staging,
        submission,
        byte_size,
        shape: tensor.shape(),
    })
}

/// A pending CPU readback. Call `poll()` to check completion.
pub struct CpuReadback {
    staging: wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
    byte_size: u64,
    shape: Vec<usize>,
}

impl CpuReadback {
    /// Poll until done, then return the CPU data.
    pub fn poll(self, ctx: &GpuContext) -> Vec<f32> {
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // safe: device polling ensures readback completes; only fails on GPU loss
        readback_with_timeout(&ctx.device, &rx)
            .expect("GPU readback failed in CpuReadback::poll");
        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&*mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        result
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// ─── AsyncBatch ────────────────────────────────────────────────────────────────

/// A pre-allocated batch buffer for double-buffered training.
pub struct GpuBatchBuffer {
    pub input: GpuTensor,
    pub target: GpuTensor,
    ready: bool,
}

impl GpuBatchBuffer {
    pub fn new(
        ctx: &GpuContext,
        batch_shape: &[usize],
        target_shape: &[usize],
    ) -> Result<Self, GpuError> {
        Ok(Self {
            input: GpuTensor::zeros(batch_shape)?,
            target: GpuTensor::zeros(target_shape)?,
            ready: false,
        })
    }

    /// Upload data to this buffer. Non-blocking (returns immediately).
    pub fn upload(
        &mut self,
        ctx: &GpuContext,
        pool: &mut GpuStagingPool,
        input_data: &[f32],
        target_data: &[f32],
    ) {
        let input_bytes = (input_data.len() * 4) as u64;
        let staging_in = pool.acquire_write(ctx, input_bytes);
        ctx.queue
            .write_buffer(&staging_in, 0, bytemuck::cast_slice(input_data));

        let target_bytes = (target_data.len() * 4) as u64;
        let staging_tgt = pool.acquire_write(ctx, target_bytes);
        ctx.queue
            .write_buffer(&staging_tgt, 0, bytemuck::cast_slice(target_data));

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_upload"),
            });
        encoder.copy_buffer_to_buffer(&staging_in, 0, self.input.buffer(), 0, input_bytes);
        encoder.copy_buffer_to_buffer(&staging_tgt, 0, self.target.buffer(), 0, target_bytes);
        ctx.queue.submit(Some(encoder.finish()));

        pool.release_write(staging_in);
        pool.release_write(staging_tgt);

        self.ready = true;
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

// ─── AsyncDataPipeline ─────────────────────────────────────────────────────────

/// Manages double-buffered data flow for GPU training.
/// CPU fills one buffer while GPU processes the other, then swap.
pub struct AsyncDataPipeline {
    buffers: [GpuBatchBuffer; 2],
    current: usize,
    staging_pool: GpuStagingPool,
}

impl AsyncDataPipeline {
    pub fn new(
        ctx: &GpuContext,
        batch_shape: &[usize],
        target_shape: &[usize],
        staging_size: u64,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            buffers: [
                GpuBatchBuffer::new(ctx, batch_shape, target_shape)?,
                GpuBatchBuffer::new(ctx, batch_shape, target_shape)?,
            ],
            current: 0,
            staging_pool: GpuStagingPool::new(ctx, staging_size),
        })
    }

    /// Get the buffer currently ready for GPU compute.
    pub fn next_buffer(&mut self) -> &mut GpuBatchBuffer {
        let buf = &mut self.buffers[self.current];
        self.current ^= 1;
        buf
    }

    /// Upload data to the *other* buffer (the one not being computed on).
    pub fn upload_next(&mut self, ctx: &GpuContext, input_data: &[f32], target_data: &[f32]) {
        let next = self.current ^ 1;
        self.buffers[next].upload(ctx, &mut self.staging_pool, input_data, target_data);
    }

    pub fn staging_pool(&mut self) -> &mut GpuStagingPool {
        &mut self.staging_pool
    }

    pub fn current_buffer(&self) -> usize {
        self.current
    }
}
