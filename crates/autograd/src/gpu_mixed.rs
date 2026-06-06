//! GPU packed F16 conversion primitives.
//! Weight upload: f32→F16 packed di `preupload_weights_gpu()` (transformer crate).
//! Inference: bulk upconvert F16→f32 temp di awal `forward_gpu_batched()`.
//! ✅ F16 KV cache: `GpuKVCacheEntry.f16_storage` — packed F16 append/read/clear.
//! ✅ F16 Sampler: `GpuContext::gpu_sample_f16()` + `gpu_sample_batched_f16()`.
//! Direncanakan: F16 LM head output — biarkan logits tetap F16 dari matmul, hindari F16→F32 upconvert.
//! Saat ini LM head matmul masih menghasilkan f32. Optimasi throughput via F16 LM head
//! membutuhkan WGSL `enable f16;` atau matmul kustom half-precision di GPU path.

use crate::gpu::{GpuContext, GpuDtype, GpuError, GpuTensor};

#[derive(Clone, Copy, Debug, PartialEq)]
#[derive(Default)]
pub enum GpuDType {
    #[default]
    F32,
    F16,
    BF16,
}

impl GpuDType {
    pub fn bytes_per_element(&self) -> u64 {
        match self {
            GpuDType::F32 => 4,
            GpuDType::F16 | GpuDType::BF16 => 2,
        }
    }

    pub fn buffer_size(&self, numel: usize) -> u64 {
        numel as u64 * self.bytes_per_element()
    }

    pub fn is_half(&self) -> bool {
        matches!(self, GpuDType::F16 | GpuDType::BF16)
    }
}


// ─── GpuLossScaler ─────────────────────────────────────────────────────────────

pub struct GpuLossScaler {
    scale: f32,
    scale_buf: wgpu::Buffer,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_growth: usize,
    max_scale: f32,
    overflow_count: usize,
    total_steps: usize,
}

impl GpuLossScaler {
    pub fn new(ctx: &GpuContext, initial_scale: f32) -> Result<Self, GpuError> {
        let scale_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loss_scaler"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&scale_buf, 0, bytemuck::bytes_of(&initial_scale));

        Ok(Self {
            scale: initial_scale,
            scale_buf,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_growth: 0,
            max_scale: 2_f32.powi(24),
            overflow_count: 0,
            total_steps: 0,
        })
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn scale_buf(&self) -> &wgpu::Buffer {
        &self.scale_buf
    }

    /// Update the GPU uniform buffer with current scale value
    pub fn update_scale_gpu(&self, ctx: &GpuContext) {
        ctx.queue
            .write_buffer(&self.scale_buf, 0, bytemuck::bytes_of(&self.scale));
    }

    /// Record an overflow event and adjust scale down
    pub fn record_overflow(&mut self, ctx: &GpuContext) {
        self.scale *= self.backoff_factor;
        self.steps_since_growth = 0;
        self.overflow_count += 1;
        self.total_steps += 1;
        self.update_scale_gpu(ctx);
    }

    /// Record a clean step and potentially grow scale
    pub fn record_success(&mut self, ctx: &GpuContext) {
        self.total_steps += 1;
        self.steps_since_growth += 1;
        if self.steps_since_growth >= self.growth_interval {
            self.scale = (self.scale * self.growth_factor).min(self.max_scale);
            self.steps_since_growth = 0;
            self.update_scale_gpu(ctx);
        }
    }

    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    pub fn overflow_rate(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.overflow_count as f32 / self.total_steps as f32
        }
    }
}

// ─── Pipeline compilation ──────────────────────────────────────────────────────

pub fn compile_f32_to_f16_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "f32_to_f16",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(F32_TO_F16_WGSL),
        "main",
    )
}

pub fn compile_f16_to_f32_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "f16_to_f32",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(F16_TO_F32_WGSL),
        "main",
    )
}

pub fn compile_f32_to_bf16_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "f32_to_bf16",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(F32_TO_BF16_WGSL),
        "main",
    )
}

pub fn compile_bf16_to_f32_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "bf16_to_f32",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(BF16_TO_F32_WGSL),
        "main",
    )
}

pub fn compile_scale_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "scale",
        &[
            crate::gpu::storage_binding(0, false),
            crate::gpu::uniform_binding(1),
        ],
        std::borrow::Cow::Borrowed(SCALE_WGSL),
        "main",
    )
}

pub fn compile_unscale_pipeline(ctx: &mut GpuContext) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "unscale",
        &[
            crate::gpu::storage_binding(0, false),
            crate::gpu::uniform_binding(1),
        ],
        std::borrow::Cow::Borrowed(UNSCALE_WGSL),
        "main",
    )
}

// ─── Mixed precision pipeline set ──────────────────────────────────────────────

pub struct MixedPrecisionPipelines {
    pub f32_to_f16: wgpu::ComputePipeline,
    pub f16_to_f32: wgpu::ComputePipeline,
    pub f32_to_bf16: wgpu::ComputePipeline,
    pub bf16_to_f32: wgpu::ComputePipeline,
    pub scale: wgpu::ComputePipeline,
    pub unscale: wgpu::ComputePipeline,
}

pub fn compile_mixed_pipelines(ctx: &mut GpuContext) -> Result<MixedPrecisionPipelines, GpuError> {
    Ok(MixedPrecisionPipelines {
        f32_to_f16: compile_f32_to_f16_pipeline(ctx)?,
        f16_to_f32: compile_f16_to_f32_pipeline(ctx)?,
        f32_to_bf16: compile_f32_to_bf16_pipeline(ctx)?,
        bf16_to_f32: compile_bf16_to_f32_pipeline(ctx)?,
        scale: compile_scale_pipeline(ctx)?,
        unscale: compile_unscale_pipeline(ctx)?,
    })
}

// ─── Reusable encoder helpers (internal) ───────────────────────────────────────

/// Helper: runs a compute pass using the reusable encoder from ctx.
/// Takes the reusable encoder, begins a compute pass, dispatches, ends the pass,
/// and puts the encoder back. Does NOT submit — caller must call ctx.flush()
/// before readback.
fn dispatch_with_reusable_encoder(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: (u32, u32, u32),
    pass_label: &str,
) {
    ctx.with_encoder(|enc| {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    });
}

/// Helper: copies buffer content using the reusable encoder.
fn copy_with_reusable_encoder(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    byte_size: u64,
) {
    ctx.with_encoder(|enc| {
        enc.copy_buffer_to_buffer(src, 0, dst, 0, byte_size);
    });
}

// ─── Dispatch helpers ──────────────────────────────────────────────────────────

/// Convert f32 storage buffer to f16 (packed u16) buffer.
/// Input: f32[numel] → Output: u16[numel] (2 bytes each)
pub fn dispatch_f32_to_f16(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    input: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let numel = input.numel();
    let shape = input.shape();
    let out_size = (numel * 2) as u64;

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mixed_out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f32_to_f16_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (numel as u32).div_ceil(256);
    dispatch_with_reusable_encoder(ctx, pipeline, &bind_group, (wg, 1, 1), "f32_to_f16");

    Ok(GpuTensor {
        shape,
        buffer: out_buf,
        dtype: GpuDtype::F32,
        device_id: 0,
    })
}

/// Convert f16 (packed u16) buffer back to f32.
/// Input: u16[numel] → Output: f32[numel]
pub fn dispatch_f16_to_f32(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    input: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let numel = input.numel();
    let shape = input.shape();
    let out_size = (numel * 4) as u64;

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mixed_out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16_to_f32_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (numel as u32).div_ceil(256);
    dispatch_with_reusable_encoder(ctx, pipeline, &bind_group, (wg, 1, 1), "f16_to_f32");

    Ok(GpuTensor {
        shape,
        buffer: out_buf,
        dtype: GpuDtype::F32,
        device_id: 0,
    })
}

/// Scale all elements in-place: output[i] = input[i] * scale
pub fn dispatch_scale_inplace(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    tensor: &GpuTensor,
    scale_buf: &wgpu::Buffer,
) -> Result<(), GpuError> {
    let numel = tensor.numel();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tensor.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scale_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (numel as u32).div_ceil(256);
    dispatch_with_reusable_encoder(ctx, pipeline, &bind_group, (wg, 1, 1), "scale");

    Ok(())
}

/// Unscale in-place: tensor[i] = tensor[i] / scale
pub fn dispatch_unscale_inplace(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    tensor: &GpuTensor,
    scale_buf: &wgpu::Buffer,
) -> Result<(), GpuError> {
    let numel = tensor.numel();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unscale_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tensor.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scale_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (numel as u32).div_ceil(256);
    dispatch_with_reusable_encoder(ctx, pipeline, &bind_group, (wg, 1, 1), "unscale");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PACKED F16: 2 f16 values per u32 (true 2× memory savings)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compile packed F32→F16 pipeline (2 f16 per u32 output element).
pub fn compile_f32_to_f16_packed_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "f32_to_f16_packed",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(F32_TO_F16_PACKED_WGSL),
        "main",
    )
}

/// Compile packed F16→F32 pipeline (reads 2 f16 per u32, writes 2 f32).
pub fn compile_f16_packed_to_f32_pipeline(
    ctx: &mut GpuContext,
) -> Result<wgpu::ComputePipeline, GpuError> {
    ctx.compile_pipeline_cached(
        "f16_packed_to_f32",
        &[
            crate::gpu::storage_binding(0, true),
            crate::gpu::storage_binding(1, false),
        ],
        std::borrow::Cow::Borrowed(F16_PACKED_TO_F32_WGSL),
        "main",
    )
}

/// Convert f32 buffer to packed f16 (2 f16 per u32).
/// Output has `(numel + 1) / 2` u32 elements, dtype=GpuDtype::F16.
pub fn dispatch_f32_to_f16_packed(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    input: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let numel = input.numel();
    let shape = input.shape();
    let packed_size = numel.div_ceil(2);
    let out_bytes = packed_size as u64 * 4;

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f16_packed_out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f32_to_f16_packed_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (packed_size as u32).div_ceil(256);
    dispatch_with_reusable_encoder(ctx, pipeline, &bind_group, (wg, 1, 1), "f32_to_f16_packed");

    Ok(GpuTensor {
        shape,
        buffer: out_buf,
        dtype: GpuDtype::F16,
        device_id: 0,
    })
}

/// Convert packed f16 (2 per u32) back to f32.
/// Output has `input.numel() * 2` f32 elements (but logical shape = input.shape).
pub fn dispatch_f16_packed_to_f32(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    input: &GpuTensor,
) -> Result<GpuTensor, GpuError> {
    let numel = input.numel(); // logical element count
    let shape = input.shape();
    let out_bytes = numel as u64 * 4;

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f16_packed_to_f32_out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("f16_packed_to_f32_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (numel as u32).div_ceil(256);
    dispatch_with_reusable_encoder(ctx, pipeline, &bind_group, (wg, 1, 1), "f16_packed_to_f32");

    Ok(GpuTensor {
        shape,
        buffer: out_buf,
        dtype: GpuDtype::F32,
        device_id: 0,
    })
}

// ─── WGSL Shaders for Packed F16 ────────────────────────────────────────────────

/// F32 → packed F16: converts 2 f32 elements per thread, packs into 1 u32.
/// Each u32 output element contains: lower 16 bits = f16 value 0, upper 16 bits = f16 value 1.
const F32_TO_F16_PACKED_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn f32_to_f16_bits(val: f32) -> u32 {
    let bits = bitcast<u32>(val);
    let sign = (bits >> 16u) & 0x8000u;
    var exp = (bits >> 23u) & 0xFFu;
    var mant = bits & 0x7FFFFFu;

    if (exp == 0u) { return sign; }
    if (exp == 0xFFu) {
        if (mant == 0u) { return sign | 0x7C00u; }
        else { return sign | 0x7C00u | (mant >> 13u); }
    }

    let new_exp = exp - 127u + 15u;
    if (new_exp >= 31u) { return sign | 0x7C00u; }
    if (new_exp <= 0u) { return sign; }

    return sign | (new_exp << 10u) | (mant >> 13u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n_f32 = arrayLength(&input);
    let n_packed = (n_f32 + 1u) / 2u;
    if (i >= n_packed) { return; }

    let lo = f32_to_f16_bits(input[i * 2u]);
    var hi = 0u;
    if (i * 2u + 1u < n_f32) {
        hi = f32_to_f16_bits(input[i * 2u + 1u]);
    }
    output[i] = lo | (hi << 16u);
}
"#;

/// Packed F16 → F32: reads 1 u32, extracts 2 f16 values, writes 2 f32.
/// Each thread handles 1 output f32 element (reads from packed u32 = input[i/2]).
const F16_PACKED_TO_F32_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn f16_bits_to_f32(f16_bits: u32) -> f32 {
    let sign = (f16_bits >> 15u) & 0x1u;
    var exp = (f16_bits >> 10u) & 0x1Fu;
    let mant = f16_bits & 0x3FFu;

    var f32_bits: u32;
    if (exp == 0u) {
        f32_bits = sign << 31u;
    } else if (exp == 31u) {
        f32_bits = (u32(sign) << 31u) | (0xFFu << 23u) | (mant << 13u);
    } else {
        f32_bits = (u32(sign) << 31u) | ((exp - 15u + 127u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f32_bits);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n_f32 = arrayLength(&output);
    if (i >= n_f32) { return; }

    let packed = input[i / 2u];
    let f16_bits = select(packed & 0xFFFFu, (packed >> 16u) & 0xFFFFu, i % 2u == 1u);
    output[i] = f16_bits_to_f32(f16_bits);
}
"#;

/// F32 → F16: pack each f32 into u16 via WGSL bitcast/truncation.
/// WGSL doesn't have native f16 without `enable f16;` feature,
/// so we do manual bit manipulation: f32 bits → sign|exp|mantissa → f16 bits.
const F32_TO_F16_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&input);
    if (i >= n) { return; }

    let bits = bitcast<u32>(input[i]);
    let sign = (bits >> 16u) & 0x8000u;
    var exp = (bits >> 23u) & 0xFFu;
    var mant = bits & 0x7FFFFFu;

    if (exp == 0u) {
        // zero/subnormal — flush to zero
        output[i] = sign;
        return;
    }

    if (exp == 0xFFu) {
        // inf/nan
        if (mant == 0u) {
            output[i] = sign | 0x7C00u; // inf
        } else {
            output[i] = sign | 0x7C00u | (mant >> 13u); // nan
        }
        return;
    }

    // Normal: rebias exponent from 127 to 15
    let new_exp = exp - 127u + 15u;
    if (new_exp >= 31u) {
        // overflow — inf
        output[i] = sign | 0x7C00u;
        return;
    }
    if (new_exp <= 0u) {
        // underflow — zero
        output[i] = sign;
        return;
    }

    let f16_bits = sign | (new_exp << 10u) | (mant >> 13u);
    output[i] = f16_bits;
}
"#;

/// F16 → F32: expand each u16 back to f32
const F16_TO_F32_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&output);
    if (i >= n) { return; }

    let f16_bits = input[i] & 0xFFFFu;
    let sign = (f16_bits >> 15u) & 0x1u;
    var exp = (f16_bits >> 10u) & 0x1Fu;
    let mant = f16_bits & 0x3FFu;

    var f32_bits: u32;

    if (exp == 0u) {
        if (mant == 0u) {
            // zero
            f32_bits = sign << 31u;
        } else {
            // subnormal — flush to zero
            f32_bits = sign << 31u;
        }
    } else if (exp == 31u) {
        // inf/nan
        let f32_exp = 0xFFu;
        let f32_mant = mant << 13u;
        f32_bits = (u32(sign) << 31u) | (f32_exp << 23u) | f32_mant;
    } else {
        // normal: rebias exponent from 15 to 127
        let f32_exp = exp - 15u + 127u;
        let f32_mant = mant << 13u;
        f32_bits = (u32(sign) << 31u) | (f32_exp << 23u) | f32_mant;
    }

    output[i] = bitcast<f32>(f32_bits);
}
"#;

/// F32 → BF16: truncate lower 16 bits of f32 mantissa
const F32_TO_BF16_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&input);
    if (i >= n) { return; }

    let bits = bitcast<u32>(input[i]);
    // Round to nearest even (add 1 to the bit that will be shifted out, check tie)
    let rounding_bias = ((bits >> 16u) & 1u) + 0x7FFFu;
    let bf16_bits = (bits + rounding_bias) >> 16u;
    output[i] = bf16_bits;
}
"#;

/// BF16 → F32: shift left by 16 bits
const BF16_TO_F32_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&output);
    if (i >= n) { return; }
    output[i] = bitcast<f32>(input[i] << 16u);
}
"#;

/// Scale in-place: tensor[i] = tensor[i] * scale
const SCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;
@group(0) @binding(1) var<uniform> scale: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&tensor);
    if (i >= n) { return; }
    tensor[i] = tensor[i] * scale;
}
"#;

/// Unscale in-place: tensor[i] = tensor[i] / scale
const UNSCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;
@group(0) @binding(1) var<uniform> scale: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&tensor);
    if (i >= n) { return; }
    tensor[i] = tensor[i] / scale;
}
"#;
