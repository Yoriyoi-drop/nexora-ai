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
    pipeline: Option<wgpu::ComputePipeline>,
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
                let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("adam_config"),
                    size: 32,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                buf
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
            pipeline: None,
        })
    }

    pub fn ensure_pipeline(&mut self, ctx: &mut GpuContext) {
        if self.pipeline.is_some() {
            return;
        }
        let pipeline = ctx
            .compile_pipeline_cached(
                "adam_step",
                &[
                    crate::gpu::storage_binding(0, true),
                    crate::gpu::storage_binding(1, false),
                    crate::gpu::storage_binding(2, false),
                    crate::gpu::storage_binding(3, false),
                    crate::gpu::uniform_binding(4),
                ],
                std::borrow::Cow::Borrowed(ADAM_WGSL),
                "main",
            )
            .expect("Failed to compile adam pipeline");
        self.pipeline = Some(pipeline);
    }

    /// Zero out gradients on GPU (fill parameter buffers with 0).
    pub fn zero_grad(&self, ctx: &GpuContext, grads: &[&GpuTensor]) -> Result<(), GpuError> {
        for g in grads {
            ctx.fill_zero(g)?;
        }
        Ok(())
    }

    /// Clip gradients on GPU using L2 norm.
    /// If total_norm > max_grad_norm, scale all gradients by max_norm/total_norm.
    pub fn clip_gradients(
        &self,
        ctx: &GpuContext,
        grads: &[&GpuTensor],
        max_norm: f32,
    ) -> Result<(), GpuError> {
        if grads.is_empty() {
            return Ok(());
        }
        // Compute sum of squared L2 norms
        let mut total_sq = 0.0f32;
        for g in grads {
            let norm_t = ctx.l2_norm(g)?;
            let norm_cpu = norm_t.to_cpu();
            let sum_sq = norm_cpu.iter().copied().next().unwrap_or(0.0);
            total_sq += sum_sq;
        }
        let total_norm = total_sq.sqrt();
        if total_norm > max_norm && total_norm > 0.0 {
            let scale = max_norm / total_norm;
            for g in grads {
                ctx.scale_inplace(g, scale)?;
            }
        }
        Ok(())
    }

    pub fn step(
        &mut self,
        ctx: &mut GpuContext,
        params: &[&GpuTensor],
        grads: &[&GpuTensor],
    ) -> Result<(), GpuError> {
        self.ensure_pipeline(ctx);
        let pipeline = self.pipeline.as_ref().unwrap();
        self.step += 1;

        // Gradient clipping on GPU
        if let Some(max_norm) = self.max_grad_norm {
            self.clip_gradients(ctx, grads, max_norm)?;
        }

        let bias_corr1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_corr2 = 1.0 - self.beta2.powi(self.step as i32);

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

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("adam_bg_{}", i)),
                layout: &pipeline.get_bind_group_layout(0),
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

            let wg = (numel as u32 + 255) / 256;
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("adam_encoder_{}", i)),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("adam_step_{}", i)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(wg, 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }

        Ok(())
    }

    pub fn zero_state(&mut self, ctx: &GpuContext) -> Result<(), GpuError> {
        for mv in self.m.iter().chain(self.v.iter()) {
            ctx.fill_zero(mv)?;
        }
        Ok(())
    }
}

const ADAM_WGSL: &str = r"
struct Config {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    step: f32,
};

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&param);
    if (i >= n) { return; }

    let g = grad[i];
    let p = param[i];

    let m_new = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g;
    let v_new = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g * g;

    let m_hat = m_new / cfg.bias_corr1;
    let v_hat = v_new / cfg.bias_corr2;

    let update = cfg.lr * m_hat / (sqrt(v_hat) + cfg.eps);
    let decay = cfg.lr * cfg.weight_decay * p;
    param[i] = p - update - decay;

    m[i] = m_new;
    v[i] = v_new;
}
";
