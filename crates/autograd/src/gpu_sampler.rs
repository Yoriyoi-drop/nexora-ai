use crate::gpu::{GpuContext, GpuTensor, GpuError};

// ─── Pipeline fns ──────────────────────────────────────────────────────────────

pub fn compile_temperature_scale_pipeline(ctx: &GpuContext) -> wgpu::ComputePipeline {
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("temperature_scale_shader"),
        source: wgpu::ShaderSource::Wgsl(TEMPERATURE_SCALE_WGSL.into()),
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("temperature_scale"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

pub fn compile_top_k_mask_pipeline(ctx: &GpuContext) -> wgpu::ComputePipeline {
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("top_k_mask_shader"),
        source: wgpu::ShaderSource::Wgsl(TOP_K_MASK_WGSL.into()),
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("top_k_mask"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

pub fn compile_multinomial_sample_pipeline(ctx: &GpuContext) -> wgpu::ComputePipeline {
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("multinomial_shader"),
        source: wgpu::ShaderSource::Wgsl(MULTINOMIAL_SAMPLE_WGSL.into()),
    });
    ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("multinomial"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

// ─── Sampler ───────────────────────────────────────────────────────────────────

pub struct GpuSamplerPipelines {
    pub temperature_scale: wgpu::ComputePipeline,
    pub top_k_mask: wgpu::ComputePipeline,
    pub multinomial: wgpu::ComputePipeline,
}

pub fn compile_sampler_pipelines(ctx: &GpuContext) -> GpuSamplerPipelines {
    GpuSamplerPipelines {
        temperature_scale: compile_temperature_scale_pipeline(ctx),
        top_k_mask: compile_top_k_mask_pipeline(ctx),
        multinomial: compile_multinomial_sample_pipeline(ctx),
    }
}

/// Sample next token(s) from logits entirely on GPU.
/// `logits`: [batch, vocab] GPU tensor.
/// Returns: [batch] GPU tensor of sampled token IDs.
pub fn gpu_sample(
    ctx: &GpuContext,
    pipelines: &GpuSamplerPipelines,
    logits: &GpuTensor,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
) -> Result<GpuTensor, GpuError> {
    let shape = logits.shape();
    let batch = shape[0];
    let vocab = shape[1] as u32;
    let numel = logits.numel();

    // Working copy of logits
    let mut working_buf = {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample_working"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sample_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(logits.buffer(), 0, &buf, 0, (numel * 4) as u64);
        ctx.queue.submit(Some(encoder.finish()));
        buf
    };

    // Step 1: temperature scale (in-place)
    if (temperature - 1.0).abs() > 1e-6 && temperature > 0.0 {
        let temp_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sample_temp_buf"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&temp_buf, 0, bytemuck::bytes_of(&temperature));

        let scale_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temp_scale_bg"),
            layout: &pipelines.temperature_scale.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: working_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("temp_scale_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temperature_scale"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipelines.temperature_scale);
            cpass.set_bind_group(0, &scale_bg, &[]);
            cpass.dispatch_workgroups((numel as u32 + 255) / 256, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Step 2: softmax (logits → probabilities)
    let logits_gpu = GpuTensor { shape: shape.clone(), buffer: working_buf };
    let probs = ctx.softmax(&logits_gpu)?;
    working_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample_probs_copy"),
        size: (numel * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probs_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(probs.buffer(), 0, &working_buf, 0, (numel * 4) as u64);
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Step 3: top-k mask (in-place on working_buf)
    if top_k > 0 && top_k < vocab {
        let topk_cfg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("topk_cfg_buf"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 2] = [vocab, top_k];
        ctx.queue.write_buffer(&topk_cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let topk_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("topk_bg"),
            layout: &pipelines.top_k_mask.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: working_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: topk_cfg_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("topk_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("topk_mask"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipelines.top_k_mask);
            cpass.set_bind_group(0, &topk_bg, &[]);
            cpass.dispatch_workgroups(batch as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // Step 4: multinomial sample
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample_output"),
        size: (batch as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let seed_cfg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("seed_cfg_buf"),
        size: 12,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let seed_cfg: [u32; 3] = [
        vocab,
        (seed & 0xFFFFFFFF) as u32,
        ((seed >> 32) & 0xFFFFFFFF) as u32,
    ];
    ctx.queue.write_buffer(&seed_cfg_buf, 0, bytemuck::cast_slice(&seed_cfg));

    let sample_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sample_bg"),
        layout: &pipelines.multinomial.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: working_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: seed_cfg_buf.as_entire_binding(),
            },
        ],
    });

    {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sample_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multinomial"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipelines.multinomial);
            cpass.set_bind_group(0, &sample_bg, &[]);
            cpass.dispatch_workgroups(batch as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    Ok(GpuTensor {
        shape: vec![batch],
        buffer: out_buf,
    })
}

// ─── WGSL Shaders ──────────────────────────────────────────────────────────────

const TEMPERATURE_SCALE_WGSL: &str = r"
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> temperature: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&logits);
    if i >= n { return; }
    logits[i] = logits[i] / temperature;
}
";

const TOP_K_MASK_WGSL: &str = r"
@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let vocab = cfg.x;
    let k = cfg.y;
    let row = wg.x;
    let base = row * vocab;

    var shared: array<f32, 256>;

    // Find k-th largest: iterate find-max → zero-out k times
    var threshold: f32 = -3.402823e+38;
    for (var iter: u32 = 0u; iter < k; iter = iter + 1u) {
        // tree-reduce max
        var mx: f32 = -3.402823e+38;
        var i = lid.x;
        while i < vocab {
            let v = probs[base + i];
            if v > mx { mx = v; }
            i += 256u;
        }
        shared[lid.x] = mx;
        workgroupBarrier();

        var stride = 128u;
        while stride > 0u {
            if lid.x < stride {
                if shared[lid.x] < shared[lid.x + stride] {
                    shared[lid.x] = shared[lid.x + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let row_max = shared[0u];
        workgroupBarrier();

        if iter == 0u {
            threshold = row_max;
        }

        // zero out the current max
        i = lid.x;
        while i < vocab {
            if probs[base + i] >= row_max - 1e-6 {
                probs[base + i] = 0.0;
            }
            i += 256u;
        }
        workgroupBarrier();
    }

    // zero out everything below threshold
    var i = lid.x;
    while i < vocab {
        if probs[base + i] > 0.0 && probs[base + i] < threshold - 1e-6 {
            probs[base + i] = 0.0;
        }
        i += 256u;
    }
}
";

const MULTINOMIAL_SAMPLE_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec3<u32>;

fn xorshift64(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return x;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
    return f32(xorshift64(state) & 0x7FFFFFu) / f32(0x7FFFFFu);
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let vocab = cfg.x;
    let seed_lo = cfg.y;
    let seed_hi = cfg.z;
    let row = wg.x;
    let base = row * vocab;

    // Per-row PRNG
    var rng_state = seed_lo ^ (row * 0x9E3779B9u);
    let _ = xorshift64(&rng_state);
    let r = random_f32(&rng_state);

    // Compute total probability mass (re-normalize after top-k zeroing)
    var total: f32 = 0.0;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        total += probs[base + i];
    }

    // Walk CDF until cumulative > r * total
    var cumulative: f32 = 0.0;
    var chosen: u32 = vocab - 1u;
    let target = r * total;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        cumulative += probs[base + i];
        if cumulative >= target {
            chosen = i;
            break;
        }
    }

    output[row] = chosen;
}
";
