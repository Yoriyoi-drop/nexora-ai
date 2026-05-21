/// GPU Fused Kernels — Priority 1
///
/// Combines matmul + bias + activation into a single compute shader dispatch,
/// eliminating intermediate VRAM roundtrips and reducing synchronisation overhead.
use crate::gpu::{storage_binding, uniform_binding, GpuContext, GpuDtype, GpuError, GpuTensor};

// ─── Activation codes ─────────────────────────────────────────────────────────
pub const ACT_IDENTITY: u32 = 0;
pub const ACT_GELU: u32 = 1;
pub const ACT_SILU: u32 = 2;
pub const ACT_RELU: u32 = 3;

// ─── WGSL Shader ──────────────────────────────────────────────────────────────
// A[M,K]  B[K,N]  bias[N]  →  C[M,N] = activation(A@B + bias)
// act: 0=identity 1=GELU 2=SiLU 3=ReLU
// Tiled version with small TILE_SIZE=8 to reduce register pressure for fused kernels
pub(crate) const FUSED_MATMUL_BIAS_WGSL: &str = r#"
struct FusedDims { M: u32, K: u32, N: u32, act: u32 };

@group(0) @binding(0) var<storage, read>       mat_a : array<f32>;
@group(0) @binding(1) var<storage, read>       mat_b : array<f32>;
@group(0) @binding(2) var<storage, read>       bias_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_c : array<f32>;
@group(0) @binding(4) var<uniform>             dims  : FusedDims;

const TILE_SIZE: u32 = 8;
var<workgroup> tA: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tB: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn gelu_act(x: f32) -> f32 {
    let t = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(t));
}
fn silu_act(x: f32) -> f32 { return x * (1.0 / (1.0 + exp(-x))); }

fn apply_act(x: f32, act: u32) -> f32 {
    if act == 1u { return gelu_act(x); }
    if act == 2u { return silu_act(x); }
    if act == 3u { return max(0.0, x); }
    return x;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn fused_matmul_bias_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let M = dims.M; let K = dims.K; let N = dims.N;
    var acc = 0.0f;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + lid.x;
        let b_row = t * TILE_SIZE + lid.y;

        if (row < M && a_col < K) { tA[lid.y][lid.x] = mat_a[row * K + a_col]; }
        else                       { tA[lid.y][lid.x] = 0.0; }

        if (b_row < K && col < N) { tB[lid.y][lid.x] = mat_b[b_row * N + col]; }
        else                       { tB[lid.y][lid.x] = 0.0; }

        workgroupBarrier();
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + tA[lid.y][k] * tB[k][lid.x];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        out_c[row * N + col] = apply_act(acc + bias_v[col], dims.act);
    }
}
"#;

// ─── Online (single-pass) Softmax ─────────────────────────────────────────────
// Uses Welford-style online max+sum for numerically stable softmax in one pass
// per row, avoiding a second global read.
pub(crate) const FUSED_ONLINE_SOFTMAX_WGSL: &str = r#"
struct SoftmaxCfg { rows: u32, cols: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: SoftmaxCfg;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn fused_online_softmax_main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(workgroup_id)         wid : vec3<u32>,
) {
    let row   = wid.x;
    let cols  = cfg.cols;
    let base  = row * cols;
    if (row >= cfg.rows) { return; }

    // Pass 1: find row max (parallel reduction in workgroup)
    var local_max = -1e38f;
    var c = lid.x;
    loop {
        if (c >= cols) { break; }
        local_max = max(local_max, inp[base + c]);
        c = c + 256u;
    }
    wg_data[lid.x] = local_max;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { wg_data[lid.x] = max(wg_data[lid.x], wg_data[lid.x + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_max = wg_data[0];
    workgroupBarrier();

    // Pass 2: compute shifted exp sum
    var local_sum = 0.0f;
    c = lid.x;
    loop {
        if (c >= cols) { break; }
        local_sum = local_sum + exp(inp[base + c] - row_max);
        c = c + 256u;
    }
    wg_data[lid.x] = local_sum;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { wg_data[lid.x] = wg_data[lid.x] + wg_data[lid.x + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_sum = wg_data[0];
    workgroupBarrier();

    // Pass 3: normalise
    c = lid.x;
    loop {
        if (c >= cols) { break; }
        out[base + c] = exp(inp[base + c] - row_max) / row_sum;
        c = c + 256u;
    }
}
"#;

// ─── impl GpuContext — Fused Op compilation ────────────────────────────────────
impl GpuContext {
    pub(crate) fn compile_fused_ops(&mut self) -> Result<(), GpuError> {
        // Fused matmul + bias + activation
        let fused_tile = self.caps.optimal_fused_tile_size();
        self.compile_fused_matmul_bias(fused_tile)?;

        // Online softmax
        self.compile_pipeline(
            "fused_online_softmax",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                uniform_binding(2),
            ],
            std::borrow::Cow::Borrowed(FUSED_ONLINE_SOFTMAX_WGSL),
            "fused_online_softmax_main",
        )?;

        Ok(())
    }

    fn compile_fused_matmul_bias(&mut self, _tile: u32) -> Result<(), GpuError> {
        // Shader uses fixed TILE_SIZE=8 to reduce register pressure for fused kernels
        self.compile_pipeline(
            "fused_matmul_bias",
            &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
            ],
            std::borrow::Cow::Borrowed(FUSED_MATMUL_BIAS_WGSL),
            "fused_matmul_bias_main",
        )
    }

    // ─── Internal dispatch ─────────────────────────────────────────────────────

    fn fused_matmul_bias_dispatch(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
        act: u32,
    ) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape.clone(), b_shape.clone()));
        }
        let (m, k, n) = (a_shape[0] as u32, a_shape[1] as u32, b_shape[1] as u32);

        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_mm_out"),
            size: (m as u64) * (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_mm_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims: [u32; 4] = [m, k, n, act];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&dims));

        let pipeline = self
            .pipelines
            .get("fused_matmul_bias")
            .ok_or_else(|| GpuError::Pipeline("fused_matmul_bias not compiled".into()))?;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_mm_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        // New shader uses workgroup_size(256, 1, 1) - dispatch based on total elements
        let total_elements = m * n;
        let wg_x = (total_elements + 255) / 256;
        self.dispatch(pipeline, &bg, (wg_x, 1, 1));

        Ok(GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: out,
            dtype: GpuDtype::F32,
        })
    }

    fn fused_matmul_bias_dispatch_profiled(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
        act: u32,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
            return Err(GpuError::MatMulShape(a_shape.clone(), b_shape.clone()));
        }
        let (m, k, n) = (a_shape[0] as u32, a_shape[1] as u32, b_shape[1] as u32);

        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_mm_out"),
            size: (m as u64) * (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_mm_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dims: [u32; 4] = [m, k, n, act];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&dims));

        let pipeline = self
            .pipelines
            .get("fused_matmul_bias")
            .ok_or_else(|| GpuError::Pipeline("fused_matmul_bias not compiled".into()))?;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_mm_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        // Use adaptive tile size based on matrix dimensions
        let tile_size = self.caps.adaptive_fused_tile_size(m as usize, n as usize, k as usize);
        let wgx = (n + tile_size - 1) / tile_size; // cols → x
        let wgy = (m + tile_size - 1) / tile_size; // rows → y
        let elapsed = self.dispatch_profiled(pipeline, &bg, (wgx, wgy, 1));

        Ok((GpuTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: out,
            dtype: GpuDtype::F32,
        }, elapsed))
    }

    // ─── Public API ────────────────────────────────────────────────────────────

    /// `C = A @ B + bias`
    pub fn fused_matmul_bias(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_matmul_bias_dispatch(a, b, bias, ACT_IDENTITY)
    }

    /// `C = GELU(A @ B + bias)` — GPT-2 / BERT style FFN
    pub fn fused_matmul_bias_gelu(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let (m, k, n) = (a_shape[0] as u32, a_shape[1] as u32, b_shape[1] as u32);

        self.fused_matmul_bias_gelu_routed(a, b, bias, false)
    }

    fn fused_matmul_bias_gelu_routed(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
        force_fused: bool,
    ) -> Result<GpuTensor, GpuError> {
        let (m, k, n) = (
            a.shape()[0] as u32,
            a.shape()[1] as u32,
            b.shape()[1] as u32,
        );
        let max_dim = m.max(k).max(n);

        // Small: matmul+bias then activation (better occupancy than fused tile-8).
        // Huge: split to avoid fused-kernel register pressure.
        // Mid (512–1024): single fused dispatch.
        if !force_fused && (max_dim < 512 || max_dim > 1024) {
            let mut mat = self.fused_matmul_bias(a, b, bias)?;
            self.gelu_inplace(&mut mat)?;
            return Ok(mat);
        }

        self.fused_matmul_bias_dispatch(a, b, bias, ACT_GELU)
    }

    pub fn fused_matmul_bias_gelu_profiled(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let (m, k, n) = (a_shape[0] as u32, a_shape[1] as u32, b_shape[1] as u32);

        let start = std::time::Instant::now();
        let out = self.fused_matmul_bias_gelu_routed(a, b, bias, false)?;
        Ok((out, start.elapsed()))
    }

    /// Fully fused `C = GELU(A @ B + bias)` without size-based routing.
    pub fn fused_matmul_bias_gelu_forced(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_matmul_bias_gelu_routed(a, b, bias, true)
    }

    /// Fully fused profiled `C = GELU(A @ B + bias)` without the large-kernel split.
    pub fn fused_matmul_bias_gelu_profiled_forced(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<(GpuTensor, std::time::Duration), GpuError> {
        self.fused_matmul_bias_dispatch_profiled(a, b, bias, ACT_GELU)
    }

    /// `C = SiLU(A @ B + bias)` — LLaMA / Mistral gate projection
    pub fn fused_matmul_bias_silu(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_matmul_bias_dispatch(a, b, bias, ACT_SILU)
    }

    /// `C = ReLU(A @ B + bias)`
    pub fn fused_matmul_bias_relu(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        self.fused_matmul_bias_dispatch(a, b, bias, ACT_RELU)
    }

    /// Online stable softmax — single dispatch, avoids 2-pass bandwidth overhead.
    /// `out[r, c] = softmax(inp[r, :])[c]`
    pub fn fused_softmax_online(
        &self,
        inp: &GpuTensor,
    ) -> Result<GpuTensor, GpuError> {
        let shape = inp.shape();
        if shape.len() < 2 {
            return Err(GpuError::ShapeMismatch("softmax needs ≥2D tensor".into()));
        }
        let rows = shape[..shape.len() - 1].iter().product::<usize>() as u32;
        let cols = *shape.last().unwrap() as u32;
        let numel = inp.numel();

        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_softmax_out"),
            size: (numel * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_softmax_cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cfg: [u32; 4] = [rows, cols, 0, 0];
        self.queue
            .write_buffer(&cfg_buf, 0, bytemuck::cast_slice(&cfg));

        let pipeline = self
            .pipelines
            .get("fused_online_softmax")
            .ok_or_else(|| GpuError::Pipeline("fused_online_softmax not compiled".into()))?;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_softmax_bg"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cfg_buf.as_entire_binding(),
                },
            ],
        });

        // one workgroup per row
        self.dispatch(pipeline, &bg, (rows, 1, 1));

        Ok(GpuTensor {
            shape: shape.to_vec(),
            buffer: out,
            dtype: GpuDtype::F32,
        })
    }
}
