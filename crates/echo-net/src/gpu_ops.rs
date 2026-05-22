use crate::utils::Complex;
use crate::{DLResult, DeepLearningError};
use ndarray::{Array2, ArrayD};
use nexora_autograd::gpu::{GpuContext, GpuTensor};

// ─── WGSL Shaders ──────────────────────────────────────────────────────────────

const IFFT_2D_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       input    : array<f32>;
@group(0) @binding(1) var<storage, read>       kernel   : array<f32>;
@group(0) @binding(2) var<storage, read_write> output   : array<f32>;
@group(0) @binding(3) var<uniform>             cfg      : vec4<u32>;

const PI: f32 = 3.141592653589793;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn ifft_2d_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let N = cfg.x;
    let K = cfg.y;
    let i = id.y;
    let j = id.x;
    if (i >= N || j >= N) { return; }

    var sum = vec2<f32>(0.0, 0.0);
    for (var ki: u32 = 0u; ki < N; ki = ki + 1u) {
        for (var kj: u32 = 0u; kj < N; kj = kj + 1u) {
            let in_idx = (ki * N + kj) * 2u;
            let ctx_v = vec2<f32>(input[in_idx], input[in_idx + 1u]);

            let ki_mod = ki % K;
            let kj_mod = kj % K;
            let k_idx = (ki_mod * K + kj_mod) * 2u;
            let ker = vec2<f32>(kernel[k_idx], kernel[k_idx + 1u]);

            let phase = 2.0 * PI * (f32(i) * f32(ki) / f32(N) + f32(j) * f32(kj) / f32(N));
            let twiddle = vec2<f32>(cos(phase), sin(phase));

            sum = sum + cmul(cmul(ctx_v, ker), twiddle);
        }
    }

    let out_idx = (i * N + j) * 2u;
    output[out_idx]     = sum.x;
    output[out_idx + 1u] = sum.y;
}
"#;

const CONV_2D_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       kernel : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             cfg    : vec4<u32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn conv_2d_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let IR = cfg.x; let IC = cfg.y;
    let KR = cfg.z; let KC = cfg.w;
    let OR = IR + KR - 1u;
    let OC = IC + KC - 1u;

    let oi = id.y; let oj = id.x;
    if (oi >= OR || oj >= OC) { return; }

    var sum = vec2<f32>(0.0, 0.0);
    for (var ki: u32 = 0u; ki < KR; ki = ki + 1u) {
        for (var kj: u32 = 0u; kj < KC; kj = kj + 1u) {
            let ii = oi + ki;
            let ij = oj + kj;
            if (ii < IR && ij < IC) {
                let in_idx = (ii * IC + ij) * 2u;
                let inp = vec2<f32>(input[in_idx], input[in_idx + 1u]);
                let k_idx = (ki * KC + kj) * 2u;
                let ker = vec2<f32>(kernel[k_idx], kernel[k_idx + 1u]);
                sum = sum + cmul(inp, ker);
            }
        }
    }

    let out_idx = (oi * OC + oj) * 2u;
    output[out_idx]     = sum.x;
    output[out_idx + 1u] = sum.y;
}
"#;

// ─── Helpers ───────────────────────────────────────────────────────────────────

fn complex_to_flat(arr: &Array2<Complex>) -> Vec<f32> {
    let mut flat = Vec::with_capacity(arr.len() * 2);
    for c in arr.iter() {
        flat.push(c.real);
        flat.push(c.imag);
    }
    flat
}

fn complex_from_slice(data: &[f32], rows: usize, cols: usize) -> Array2<Complex> {
    let mut result = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let idx = (i * cols + j) * 2;
            result[[i, j]] = Complex::new(data[idx], data[idx + 1]);
        }
    }
    result
}

/// Helper: compile a WGSL compute shader, create bind group, submit, read back.
fn run_compute_shader(
    ctx: &GpuContext,
    wgsl: &str,
    entry_point: &str,
    entries: &[wgpu::BindGroupEntry],
    bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
    workgroups: (u32, u32, u32),
    output_buf: &wgpu::Buffer,
    output_size: u64,
) -> Result<Vec<f32>, DeepLearningError> {
    // Shader module
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("echo_gpu_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
    });

    // Bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("echo_gpu_bgl"),
        entries: bind_group_layout_entries,
    });

    // Pipeline layout
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("echo_gpu_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    // Compute pipeline
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("echo_gpu_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    // Bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("echo_gpu_bg"),
        layout: &bind_group_layout,
        entries,
    });

    // Readback buffer
    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("echo_gpu_readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Encoder
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("echo_gpu_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // Copy output to readback buffer
    encoder.copy_buffer_to_buffer(
        output_buf,
        0,
        &readback_buf,
        0,
        output_size,
    );

    ctx.queue.submit(Some(encoder.finish()));
    ctx.sync();

    // Map readback
    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
    });
    rx.recv_timeout(std::time::Duration::from_secs(30))
        .map_err(|e| DeepLearningError::Computation { reason: format!("GPU readback recv: {e}") })?
        .map_err(|e| DeepLearningError::Computation { reason: format!("GPU map_async: {e:?}") })?;

    let mapped = slice.get_mapped_range();
    let data: &[f32] = bytemuck::cast_slice(&*mapped);
    let result = data.to_vec();
    drop(mapped);
    readback_buf.unmap();

    Ok(result)
}

// ─── GPU 2D IFFT ───────────────────────────────────────────────────────────────

pub fn gpu_ifft_2d(
    context: &Array2<Complex>,
    collapse_kernel: &Array2<Complex>,
    collapse_strength: f32,
) -> DLResult<Array2<Complex>> {
    let ctx = GpuContext::global()
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

    let (rows, cols) = context.dim();
    if rows != cols {
        return Err(DeepLearningError::Computation {
            reason: "IFFT requires square matrix".into(),
        });
    }
    let n = rows as u32;
    let (krows, kcols) = collapse_kernel.dim();
    let k = krows.min(kcols) as u32;

    // Upload input and kernel as flat interleaved f32
    let input_data = complex_to_flat(context);
    let kernel_data = complex_to_flat(collapse_kernel);

    let input_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ifft_input"),
        size: (input_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(&input_data));

    let kernel_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ifft_kernel"),
        size: (kernel_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&kernel_buf, 0, bytemuck::cast_slice(&kernel_data));

    // Output buffer
    let numel = n as usize * n as usize;
    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ifft_output"),
        size: (numel * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Config uniform
    let cfg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ifft_cfg"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cfg: [u32; 4] = [n, k, 0, 0];
    ctx.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

    let result_data = run_compute_shader(
        &ctx,
        IFFT_2D_WGSL,
        "ifft_2d_main",
        &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: kernel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
        ],
        &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
        ((n + 7) / 8, (n + 7) / 8, 1),
        &output_buf,
        (numel * 8) as u64,
    )?;

    let mut result = complex_from_slice(&result_data, rows, cols);
    for val in result.iter_mut() {
        val.real *= collapse_strength;
        val.imag *= collapse_strength;
    }

    Ok(result)
}

// ─── GPU 2D Convolution ────────────────────────────────────────────────────────

pub fn gpu_conv_2d(
    input: &Array2<Complex>,
    kernel: &Array2<Complex>,
) -> DLResult<Array2<Complex>> {
    let ctx = GpuContext::global()
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

    let (ir, ic) = input.dim();
    let (kr, kc) = kernel.dim();
    let or = ir + kr - 1;
    let oc = ic + kc - 1;

    let input_data = complex_to_flat(input);
    let kernel_data = complex_to_flat(kernel);

    let input_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conv_input"),
        size: (input_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(&input_data));

    let kernel_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conv_kernel"),
        size: (kernel_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&kernel_buf, 0, bytemuck::cast_slice(&kernel_data));

    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conv_output"),
        size: (or * oc * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cfg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conv_cfg"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cfg: [u32; 4] = [ir as u32, ic as u32, kr as u32, kc as u32];
    ctx.queue.write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

    let result_data = run_compute_shader(
        &ctx,
        CONV_2D_WGSL,
        "conv_2d_main",
        &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: kernel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cfg_buf.as_entire_binding() },
        ],
        &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
        ((oc as u32 + 7) / 8, (or as u32 + 7) / 8, 1),
        &output_buf,
        (or * oc * 8) as u64,
    )?;

    Ok(complex_from_slice(&result_data, or, oc))
}

// ─── GPU Cosine Similarity Matrix ──────────────────────────────────────────────
//
// Uses CPU for L2 normalization (O(T·D)), GPU matmul for similarity (O(T²·D)).
// Pure CPU approach would be O(T²·D); GPU matmul handles the O(T²·D) part.

pub fn gpu_cosine_similarity_matrix(embeddings: &ArrayD<f32>) -> DLResult<Array2<f32>> {
    let ctx = GpuContext::global()
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

    let shape = embeddings.shape();
    let num_tokens = shape[0];
    let emb_dim = shape[1];

    // CPU: L2 normalize each row (O(T·D))
    let emb_2d = embeddings
        .clone()
        .into_shape(vec![num_tokens, emb_dim])
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
    let mut normed_arr = Array2::zeros((num_tokens, emb_dim));
    for i in 0..num_tokens {
        let mut sq_sum = 0.0;
        for j in 0..emb_dim {
            let v = emb_2d[[i, j]];
            sq_sum += v * v;
        }
        let inv_norm = 1.0 / sq_sum.sqrt().max(1e-8);
        for j in 0..emb_dim {
            normed_arr[[i, j]] = emb_2d[[i, j]] * inv_norm;
        }
    }

    // GPU: upload normalized embeddings as 1D then reshape to [T, D]
    let normed_flat = normed_arr.clone().into_shape(vec![num_tokens * emb_dim])
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
    let normed_gpu = GpuTensor::from_cpu(&normed_flat)
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
    let normed_2d = normed_gpu.reshape(vec![num_tokens, emb_dim])
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

    // GPU: similarity = normed @ normed^T (shape [T, T])
    let normed_t = ctx.transpose(&normed_2d)
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;
    let sim = ctx.matmul(&normed_2d, &normed_t)
        .map_err(|e| DeepLearningError::Computation { reason: e.to_string() })?;

    // Download
    let sim_cpu = sim.to_cpu();
    let sim_slice = sim_cpu.as_slice().ok_or_else(|| {
        DeepLearningError::Computation { reason: "similarity matrix not contiguous".into() }
    })?;

    let mut result = Array2::zeros((num_tokens, num_tokens));
    for i in 0..num_tokens {
        for j in 0..num_tokens {
            result[[i, j]] = sim_slice[i * num_tokens + j];
        }
    }

    Ok(result)
}
