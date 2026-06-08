use crate::gpu::{GpuContext, GpuError, GpuTensor};

pub struct GpuAdam {
    pub m: Vec<GpuTensor>,
    pub v: Vec<GpuTensor>,
    pub step: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: Option<f32>,
    config_bufs: Vec<wgpu::Buffer>,
}

impl GpuAdam {
    pub fn new(ctx: &GpuContext, params: &[&GpuTensor], lr: f32) -> Result<Self, GpuError> {
        let m = params
            .iter()
            .map(|p| GpuTensor::zeros(&p.shape()))
            .collect::<Result<Vec<_>, _>>()?;
        let v = params
            .iter()
            .map(|p| GpuTensor::zeros(&p.shape()))
            .collect::<Result<Vec<_>, _>>()?;

        let config_bufs = params
            .iter()
            .map(|_p| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("adam_config"),
                    size: 32,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Ok(Self {
            m,
            v,
            step: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            max_grad_norm: None,
            config_bufs,
        })
    }

    /// Zero out gradients on GPU (fill parameter buffers with 0).
    pub fn zero_grad(&self, ctx: &GpuContext, grads: &[&GpuTensor]) -> Result<(), GpuError> {
        for g in grads {
            ctx.fill_zero(g)?;
        }
        Ok(())
    }

    /// Clip gradients on GPU using L2 norm (batched version).
    /// If total_norm > max_grad_norm, scale all gradients by max_norm/total_norm.
    pub fn clip_gradients(
        &self,
        ctx: &GpuContext,
        grads: &[&GpuTensor],
        max_norm: f32,
    ) -> Result<(), GpuError> {
        crate::gpu_grad_clip::clip_gradients_batched(ctx, grads, max_norm)
    }

    /// Perform one AdamW optimizer step.
    ///
    /// Optimizations vs the earlier per-param-encoder approach:
    /// 1. All param dispatches batched in reusable encoder (single submit via ctx.with_encoder)
    /// 2. Pipeline fetched from GpuContext cache (pre-compiled in compile_all_pipelines)
    /// 3. Gradient clipping uses batched clip_gradients_batched
    pub fn step(
        &mut self,
        ctx: &GpuContext,
        params: &[&GpuTensor],
        grads: &[&GpuTensor],
    ) -> Result<(), GpuError> {
        self.step += 1;

        #[cfg(feature = "cuda")]
        if let Some(cuda) = ctx.cuda {
            for i in 0..params.len() {
                if params[i].numel() == 0 { continue; }
                let param_cuda = ctx.cuda_read_tensor(cuda, params[i])?;
                let grad_cuda = ctx.cuda_read_tensor(cuda, grads[i])?;
                let m_cuda = ctx.cuda_read_tensor(cuda, &self.m[i])?;
                let v_cuda = ctx.cuda_read_tensor(cuda, &self.v[i])?;
                cuda.adam_step(&param_cuda, &grad_cuda, &m_cuda, &v_cuda,
                    self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.step)
                    .map_err(|e| GpuError::Compute(format!("CUDA adam_step: {e}")))?;
                let numel = params[i].numel();
                let mut cpu_buf = vec![0.0f32; numel];
                cuda.stream
                    .memcpy_dtoh(&param_cuda.buffer, &mut cpu_buf)
                    .map_err(|e| GpuError::Transfer(format!("CUDA adam_step param dtoh: {e}")))?;
                ctx.queue.write_buffer(params[i].buffer(), 0, bytemuck::cast_slice(&cpu_buf));
                let mut m_cpu = vec![0.0f32; numel];
                cuda.stream
                    .memcpy_dtoh(&m_cuda.buffer, &mut m_cpu)
                    .map_err(|e| GpuError::Transfer(format!("CUDA adam_step m dtoh: {e}")))?;
                ctx.queue.write_buffer(self.m[i].buffer(), 0, bytemuck::cast_slice(&m_cpu));
                let mut v_cpu = vec![0.0f32; numel];
                cuda.stream
                    .memcpy_dtoh(&v_cuda.buffer, &mut v_cpu)
                    .map_err(|e| GpuError::Transfer(format!("CUDA adam_step v dtoh: {e}")))?;
                ctx.queue.write_buffer(self.v[i].buffer(), 0, bytemuck::cast_slice(&v_cpu));
            }
            return Ok(());
        }

        // Gradient clipping on GPU (batched)
        if let Some(max_norm) = self.max_grad_norm {
            self.clip_gradients(ctx, grads, max_norm)?;
        }

        // Pipeline is pre-compiled via compile_all_pipelines in GpuContext::new()
        let pipeline = ctx
            .pipelines
            .get("adam_step")
            .ok_or_else(|| GpuError::Pipeline("adam_step not compiled".into()))?;

        let bias_corr1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_corr2 = 1.0 - self.beta2.powi(self.step as i32);

        // Batch all dispatches using with_encoder (single submit)
        // Only queue writes for config buffers happen before the encoder batch
        for i in 0..params.len() {
            let numel = params[i].numel();
            if numel == 0 {
                continue;
            }

            let cfg: [f32; 8] = [
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bias_corr1,
                bias_corr2,
                self.step as f32,
            ];
            ctx.queue
                .write_buffer(&self.config_bufs[i], 0, bytemuck::cast_slice(&cfg));
        }

        // Now batch all dispatches in the reusable encoder
        for i in 0..params.len() {
            let numel = params[i].numel();
            if numel == 0 {
                continue;
            }

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("adam_step_bg_{}", i)),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params[i].buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grads[i].buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.m[i].buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.v[i].buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.config_bufs[i].as_entire_binding(),
                    },
                ],
            });

            let wg = (numel as u32).div_ceil(256);
            ctx.dispatch(pipeline, &bind_group, (wg, 1, 1));
        }

        Ok(())
    }

    /// Zero out optimizer state (m, v) on GPU.
    pub fn zero_state(&self, ctx: &GpuContext) -> Result<(), GpuError> {
        for mv in self.m.iter().chain(self.v.iter()) {
            ctx.fill_zero(mv)?;
        }
        Ok(())
    }
}
