/// GPU Async Compute Batch — Priority 2
///
/// Accumulates multiple compute dispatches into a single `CommandEncoder`
/// before submitting to the GPU queue. This avoids the CPU↔GPU synchronisation
/// stall that occurs when each op creates and submits its own encoder.
///
/// # Usage
/// ```rust,ignore
/// let mut batch = GpuCommandBatch::new(&ctx);
/// batch.dispatch(&pipeline_a, &bg_a, (wgx, 1, 1));
/// batch.dispatch(&pipeline_b, &bg_b, (wgx, 1, 1));
/// batch.submit();   // single queue submit for both ops
/// ```
use crate::gpu::{GpuContext, GpuError, GpuTensor};

// ─── Command Batch ─────────────────────────────────────────────────────────────

pub struct GpuCommandBatch<'ctx> {
    ctx: &'ctx GpuContext,
    encoder: Option<wgpu::CommandEncoder>,
    op_count: usize,
}

impl<'ctx> GpuCommandBatch<'ctx> {
    pub fn new(ctx: &'ctx GpuContext) -> Self {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nexora_batch_encoder"),
            });
        Self {
            ctx,
            encoder: Some(encoder),
            op_count: 0,
        }
    }

    /// Add a compute dispatch to this batch.
    /// Each call is a separate compute pass inside the same encoder.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        if let Some(enc) = self.encoder.as_mut() {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            self.op_count += 1;
        }
    }

    /// Copy one GPU buffer to another inside the same encoder (zero extra submission).
    pub fn copy_buffer(&mut self, src: &GpuTensor, dst: &GpuTensor) {
        if src.numel() != dst.numel() {
            return;
        }
        let byte_size = (src.numel() * 4) as u64;
        if let Some(enc) = self.encoder.as_mut() {
            enc.copy_buffer_to_buffer(src.buffer(), 0, dst.buffer(), 0, byte_size);
            self.op_count += 1;
        }
    }

    /// Copy a raw `wgpu::Buffer` inside the batch.
    pub fn copy_raw_buffer(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, byte_size: u64) {
        if let Some(enc) = self.encoder.as_mut() {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, byte_size);
            self.op_count += 1;
        }
    }

    /// Submit all accumulated operations in one queue call.
    /// Returns the number of dispatches submitted.
    pub fn submit(mut self) -> usize {
        if let Some(enc) = self.encoder.take() {
            self.ctx.queue.submit(Some(enc.finish()));
        }
        self.op_count
    }

    /// Number of dispatches accumulated so far.
    pub fn op_count(&self) -> usize {
        self.op_count
    }
}

// ─── Async staging helper ──────────────────────────────────────────────────────

/// Uploads CPU data to an existing GPU buffer without blocking the GPU queue.
/// The write is deferred through `wgpu::Queue::write_buffer`, which internally
/// manages a staging ring-buffer — no CPU stall.
pub fn async_upload(ctx: &GpuContext, dst: &wgpu::Buffer, data: &[u8]) {
    ctx.queue.write_buffer(dst, 0, data);
}

/// Uploads a slice of f32 values to an existing GPU tensor.
pub fn async_upload_f32(ctx: &GpuContext, dst: &GpuTensor, data: &[f32]) {
    ctx.queue
        .write_buffer(dst.buffer(), 0, bytemuck::cast_slice(data));
}

// ─── impl GpuContext extensions ────────────────────────────────────────────────

impl GpuContext {
    /// Create a new command batch bound to this context.
    /// Use `GpuCommandBatch` to accumulate multiple dispatches and submit at once.
    pub fn begin_batch(&self) -> GpuCommandBatch<'_> {
        GpuCommandBatch::new(self)
    }

    /// Zero multiple tensors in one queue submission.
    pub fn fill_zeros_batched(&self, tensors: &[&GpuTensor]) -> Result<usize, GpuError> {
        if tensors.is_empty() {
            return Ok(0);
        }
        let mut batch = self.begin_batch();
        for t in tensors {
            self.batch_enqueue_fill_zero(&mut batch, t)?;
        }
        let ops = batch.submit();
        self.sync();
        Ok(ops)
    }

    /// Enqueue a `fill_zero` dispatch into an open batch (single submit with other ops).
    pub fn batch_enqueue_fill_zero(
        &self,
        batch: &mut GpuCommandBatch<'_>,
        t: &GpuTensor,
    ) -> Result<(), GpuError> {
        let pipeline = self
            .pipelines
            .get("fill_zero")
            .ok_or_else(|| GpuError::Pipeline("fill_zero not compiled".into()))?;
        let numel = t.numel() as u32;
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fill_zero_batch_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: t.buffer().as_entire_binding(),
            }],
        });
        batch.dispatch(&pipeline.pipeline, &bg, (numel.div_ceil(256), 1, 1));
        Ok(())
    }

    /// Zero-copy upload: writes CPU bytes into a GPU buffer asynchronously.
    /// Returns immediately; the GPU will see the data on next queue submission.
    pub fn upload_bytes(&self, dst: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(dst, 0, data);
    }

    /// Zero-copy upload for f32 slices.
    pub fn upload_f32(&self, dst: &GpuTensor, data: &[f32]) {
        self.queue
            .write_buffer(dst.buffer(), 0, bytemuck::cast_slice(data));
    }

    /// Allocates a UNIFORM buffer for config/dims data.
    pub(crate) fn alloc_uniform(&self, label: &'static str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
