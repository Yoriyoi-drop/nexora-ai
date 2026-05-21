use crate::gpu::{GpuContext, GpuDtype, GpuError, GpuTensor};

// ─── Pipeline fns ──────────────────────────────────────────────────────────────

pub fn compile_temperature_scale_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "temperature_scale_sampler",
        &[
            crate::gpu::storage_binding(0, false),
            crate::gpu::uniform_binding(1),
        ],
        std::borrow::Cow::Borrowed(TEMPERATURE_SCALE_WGSL),
        "main",
    )
}

pub fn compile_top_k_mask_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "top_k_mask_sampler",
        &[
            crate::gpu::storage_binding(0, false),
            crate::gpu::uniform_binding(1),
        ],
        std::borrow::Cow::Borrowed(TOP_K_MASK_WGSL),
        "main",
    )
}

pub fn compile_multinomial_sample_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "multinomial_sample_sampler",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
            crate::gpu::uniform_binding(2),
        ],
        std::borrow::Cow::Borrowed(MULTINOMIAL_SAMPLE_WGSL),
        "main",
    )
}

// ─── Sampler ───────────────────────────────────────────────────────────────────

pub struct GpuSamplerPipelines {
    pub temperature_scale: wgpu::ComputePipeline,
    pub top_k_mask: wgpu::ComputePipeline,
    pub multinomial: wgpu::ComputePipeline,
}

pub fn compile_sampler_pipelines(ctx: &mut GpuContext) -> Result<GpuSamplerPipelines, GpuError> {
    Ok(GpuSamplerPipelines {
        temperature_scale: compile_temperature_scale_pipeline(ctx)?,
        top_k_mask: compile_top_k_mask_pipeline(ctx)?,
        multinomial: compile_multinomial_sample_pipeline(ctx)?,
    })
}

// ─── Reusable encoder helper ──────────────────────────────────────────────────

/// Helper: dispatch a compute pass using the reusable encoder (no submit).
fn dispatch_compute(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: (u32, u32, u32),
    label: &str,
) {
    ctx.with_encoder(|enc| {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    });
}

/// Helper: copy buffer-to-buffer using the reusable encoder.
fn copy_buf(ctx: &GpuContext, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
    ctx.with_encoder(|enc| {
        enc.copy_buffer_to_buffer(src, 0, dst, 0, size);
    });
}

// ─── Sampler ───────────────────────────────────────────────────────────────────

/// Sample next token(s) from logits entirely on GPU.
/// `logits`: [batch, vocab] GPU tensor.
/// Returns: [batch] GPU tensor of sampled token IDs.
///
/// All operations are batched into a single command encoder and flushed at the end,
/// reducing CPU↔GPU synchronization overhead.
pub fn gpu_sample(
    ctx: &mut GpuContext,
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
        let pooled_buf = ctx.alloc_buffer(
            (numel * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let buf = pooled_buf.buffer;
        // Copy logits → working buffer (batched into reusable encoder)
        copy_buf(ctx, logits.buffer(), &buf, (numel * 4) as u64);
        buf
    };

    // Step 1: temperature scale (in-place on working_buf)
    if (temperature - 1.0).abs() > 1e-6 && temperature > 0.0 {
        let temp_buf = ctx.alloc_buffer(
            4,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        ).buffer;
        ctx.queue
            .write_buffer(&temp_buf, 0, bytemuck::bytes_of(&temperature));

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

        dispatch_compute(ctx, &pipelines.temperature_scale, &scale_bg, ((numel as u32 + 255) / 256, 1, 1), "temperature_scale");
    }

    // Step 2: softmax (logits → probabilities)
    // NOTE: ctx.softmax() uses self.dispatch() internally, which also uses
    // the reusable encoder — operations are automatically batched.
    let logits_gpu = GpuTensor {
        shape: shape.clone(),
        buffer: working_buf,
        dtype: GpuDtype::F32,
    };
    let probs = ctx.softmax(&logits_gpu)?;
    working_buf = ctx.alloc_buffer(
        (numel * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    ).buffer;
    // Copy softmax output → working buffer (batched into reusable encoder)
    copy_buf(ctx, probs.buffer(), &working_buf, (numel * 4) as u64);

    // Step 3: top-k mask (in-place on working_buf)
    if top_k > 0 && top_k < vocab {
        let topk_cfg_buf = ctx.alloc_buffer(
            8,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        ).buffer;
        let cfg: [u32; 2] = [vocab, top_k];
        ctx.queue
            .write_buffer(&topk_cfg_buf, 0, bytemuck::cast_slice(&cfg));

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

        dispatch_compute(ctx, &pipelines.top_k_mask, &topk_bg, (batch as u32, 1, 1), "topk_mask");
    }

    // Step 4: multinomial sample
    let out_buf = ctx.alloc_buffer(
        (batch as u64) * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    ).buffer;

    let seed_cfg_buf = ctx.alloc_buffer(
        12,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    ).buffer;
    let seed_cfg: [u32; 3] = [
        vocab,
        (seed & 0xFFFFFFFF) as u32,
        ((seed >> 32) & 0xFFFFFFFF) as u32,
    ];
    ctx.queue
        .write_buffer(&seed_cfg_buf, 0, bytemuck::cast_slice(&seed_cfg));

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

    dispatch_compute(ctx, &pipelines.multinomial, &sample_bg, (batch as u32, 1, 1), "multinomial_sample");

    // Flush all accumulated operations — single submit for the entire pipeline
    ctx.flush();

    Ok(GpuTensor {
        shape: vec![batch],
        buffer: out_buf,
        dtype: GpuDtype::F32,
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

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    let vocab = cfg.x;
    let k = cfg.y;
    if k >= vocab { return; }
    let row = wg.x;
    let base = row * vocab;

    // Find k-th largest value without modifying probs
    var prev_max: f32 = 3.402823e+38;
    var threshold: f32 = -3.402823e+38;
    for (var iter: u32 = 0u; iter < k; iter = iter + 1u) {
        var mx: f32 = -3.402823e+38;
        var i = lid.x;
        while i < vocab {
            let v = probs[base + i];
            if v > mx && v < prev_max - 1e-6 { mx = v; }
            i += 256u;
        }
        scratch[lid.x] = mx;
        workgroupBarrier();

        var stride = 128u;
        while stride > 0u {
            if lid.x < stride {
                if scratch[lid.x] < scratch[lid.x + stride] {
                    scratch[lid.x] = scratch[lid.x + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        threshold = scratch[0u];
        workgroupBarrier();
        prev_max = threshold;
    }

    // Zero out everything below the k-th largest
    var i = lid.x;
    while i < vocab {
        if probs[base + i] < threshold - 1e-6 {
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
    var warmup = xorshift64(&rng_state);
    let r = random_f32(&rng_state);

    // Compute total probability mass (re-normalize after top-k zeroing)
    var total: f32 = 0.0;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        total += probs[base + i];
    }

    // Walk CDF until cumulative > r * total
    var cumulative: f32 = 0.0;
    var chosen: u32 = vocab - 1u;
    let threshold = r * total;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        cumulative += probs[base + i];
        if cumulative >= threshold {
            chosen = i;
            break;
        }
    }

    output[row] = chosen;
}
";
