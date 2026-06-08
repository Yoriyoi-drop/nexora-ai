pub(crate) const MATMUL_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_tiled_main(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (dims.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile of A
        let a_row = row;
        let a_col = t * TILE_SIZE + lid.y;
        if (a_row < dims.M && a_col < dims.K) {
            tile_a[lid.x][lid.y] = a[a_row * dims.K + a_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // Load tile of B
        let b_row = t * TILE_SIZE + lid.x;
        let b_col = col;
        if (b_row < dims.K && b_col < dims.N) {
            tile_b[lid.x][lid.y] = b[b_row * dims.N + b_col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // Accumulate
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.1b: VEC4<F32> VECTORIZED MATMUL
//  Uses vec4<f32> loads from global memory to achieve 4× bandwidth utilization
//  vs scalar f32 loads. Requires M, N, K to be multiples of 4 for full benefit.
//  Falls back to scalar tiled matmul when alignment doesn't match.
// ═══════════════════════════════════════════════════════════════════════════════
pub(crate) const MATMUL_TILED_VEC4_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};
const VEC4: u32 = 4u;

struct Vec4Array { data: array<vec4<f32>>; }

@group(0) @binding(0) var<storage, read> a_vec4: Vec4Array;
@group(0) @binding(1) var<storage, read> b_vec4: Vec4Array;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn load_vec4_a(linear_idx: u32) -> f32 {
    let vi = linear_idx / VEC4;
    let ei = linear_idx % VEC4;
    return a_vec4.data[vi][ei];
}

fn load_vec4_b(linear_idx: u32) -> f32 {
    let vi = linear_idx / VEC4;
    let ei = linear_idx % VEC4;
    return b_vec4.data[vi][ei];
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_vec4_main(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (dims.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_linear = row * dims.K + t * TILE_SIZE + lid.y;
        tile_a[lid.x][lid.y] = select(0.0, load_vec4_a(a_linear), a_linear < dims.M * dims.K);

        let b_linear = (t * TILE_SIZE + lid.x) * dims.N + col;
        tile_b[lid.x][lid.y] = select(0.0, load_vec4_b(b_linear), b_linear < dims.K * dims.N);

        workgroupBarrier();

        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.2: INT8 QUANTIZED MATMUL
// ═══════════════════════════════════════════════════════════════════════════════
//
// A is packed int8 weights (4 values per u32), B is f32 activations.
// Dequantization happens on-the-fly: int8 → f32 × scale.
// Output C is f32.
//
// The shader uses the same tiled matmul structure as MATMUL_TILED_WGSL but
// adds int8 unpacking for the A matrix. Only the weight matrix (A) is
// quantized — B (activations) stays f32 for precision during training.

pub(crate) const MATMUL_INT8_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A is packed as u32 — each u32 holds 4 int8 values (little-endian byte order).
// B and C remain f32.
@group(0) @binding(0) var<storage, read> a_packed: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
    scale: f32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn unpack_int8(packed: u32, byte_idx: u32) -> f32 {
    let shift = byte_idx * 8u;
    let byte = (packed >> shift) & 0xFFu;
    var val = f32(byte);
    if (byte > 127u) {
        val = val - 256.0;
    }
    return val * uniforms.scale;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_int8_main(@builtin(local_invocation_id) lid: vec3<u32>,
                    @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (packed int8 → f32) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            let a_flat_idx = a_global_row * uniforms.K + a_global_col;
            let packed_idx = a_flat_idx / 4u;
            let byte_off = a_flat_idx % 4u;
            tile_a[lid.x][lid.y] = unpack_int8(a_packed[packed_idx], byte_off);
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (f32) ──
        let b_global_row = t * TILE_SIZE + lid.x;
        let b_global_col = col;
        if (b_global_row < uniforms.K && b_global_col < uniforms.N) {
            tile_b[lid.x][lid.y] = b[b_global_row * uniforms.N + b_global_col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // ── Accumulate ──
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.2a: F16 PACKED WEIGHT MATMUL (activation f32 × packed f16 weight)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Activation (a) is f32, weight (b) is packed F16 (2 values per u32).
// Weight buffer is half the size of f32 equivalent — saves VRAM bandwidth.
// On read: unpack 2 f16 per u32 → convert to f32 → accumulate in f32.
// Output C is f32 (same precision as regular matmul).
//
// The shader follows the same tiled structure as MATMUL_TILED_WGSL but with
// packed F16 reads for the B (weight) matrix. Only the weight matrix is F16 —
// activation and output remain f32 for precision during accumulation.

pub(crate) const MATMUL_F16_TILED_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A is f32 activations [M, K]; B is packed f16 weights [K, N] (2 f16 per u32);
// C is f32 output [M, N].
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

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

fn unpack_f16(packed: u32, idx: u32) -> f32 {
    let shift = idx * 16u;
    let f16_bits = (packed >> shift) & 0xFFFFu;
    return f16_bits_to_f32(f16_bits);
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_f16_main(@builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (f32 activation) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            tile_a[lid.x][lid.y] = a[a_global_row * uniforms.K + a_global_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (packed f16 weight → f32) ──
        let b_global_row = t * TILE_SIZE + lid.x;
        let b_global_col = col;
        if (b_global_row < uniforms.K && b_global_col < uniforms.N) {
            let b_flat_idx = b_global_row * uniforms.N + b_global_col;
            let packed_idx = b_flat_idx / 2u;
            let sub_idx = b_flat_idx % 2u;
            tile_b[lid.x][lid.y] = unpack_f16(b_packed[packed_idx], sub_idx);
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // ── Accumulate in f32 ──
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
//  PHASE 1.2b: INT8 WEIGHT MATMUL (activation f32 × packed int8 weight RHS)
// ═══════════════════════════════════════════════════════════════════════════════
//
// A is f32 activations [M, K], B is packed int8 weights [N, K] (ORIGINAL
// orientation — NOT transposed!). The shader computes C = A × dequant(B)^T.
//
// Weight matrix W of shape [output_dim, hidden] is stored in its original
// row-major layout. The shader reads B[j][k] = W[col][t*TILE + lid.x] and
// accumulates C[i][j] = sum_k A[i][k] * dequant(W[j][k]).
//
// This avoids transposing the int8 data and matches the inference pattern:
// activation(f32) @ weight^T.

pub(crate) const MATMUL_INT8_WEIGHT_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A = f32 activations [M, K]; B = packed int8 weights [N, K] (NOT transposed); C = f32 output [M, N]
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> zero_points: array<f32>;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn unpack_int8_weight(packed: u32, byte_idx: u32, col: u32) -> f32 {
    let shift = byte_idx * 8u;
    let byte = (packed >> shift) & 0xFFu;
    var val = f32(byte);
    if (byte > 127u) {
        val = val - 256.0;
    }
    return val * scales[col] + zero_points[col];
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_int8_weight_main(@builtin(local_invocation_id) lid: vec3<u32>,
                           @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (f32 activation) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            tile_a[lid.x][lid.y] = a[a_global_row * uniforms.K + a_global_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (packed int8 weight [N, K] — NOT transposed) ──
        // B[j][k] = W[col][t*TILE + lid.x] — weight matrix in original orientation
        let b_j = col;                                    // N dimension
        let b_k = t * TILE_SIZE + lid.x;                  // K dimension
        if (b_j < uniforms.N && b_k < uniforms.K) {
            let b_flat_idx = b_j * uniforms.K + b_k;
            let packed_idx = b_flat_idx / 4u;
            let byte_off = b_flat_idx % 4u;
            tile_b[lid.x][lid.y] = unpack_int8_weight(b_packed[packed_idx], byte_off, b_j);
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // ── Accumulate C[row][col] += sum_k A[row][k] * dequant(W[col][k]) ──
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

pub(crate) const MATMUL_INT4_WEIGHT_WGSL: &str = r#"
const TILE_SIZE: u32 = {{TILE_SIZE}};

// A = f32 activations [M, K]; B = packed Q4 weights [(K/2), N] as u32 (4 packed bytes per u32);
// C = f32 output [M, N]; scales [groups, N] — per-group per-column
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    GroupSize: u32,
    Tile: u32,
};

@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> scales: array<f32>;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

fn extract_q4_from_byte(packed: u32, byte_idx: u32, high_nibble: u32) -> f32 {
    let shift = byte_idx * 8u;
    let byte = (packed >> shift) & 0xFFu;
    let nibble = select(byte & 0x0Fu, (byte >> 4u) & 0x0Fu, high_nibble != 0u);
    return f32(nibble) - 8.0;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn matmul_int4_weight_main(@builtin(local_invocation_id) lid: vec3<u32>,
                           @builtin(workgroup_id) wg_id: vec3<u32>) {
    let row = wg_id.x * TILE_SIZE + lid.x;
    let col = wg_id.y * TILE_SIZE + lid.y;

    var sum = 0.0;
    let num_tiles = (uniforms.K + TILE_SIZE - 1) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // ── Load tile of A (f32 activation) ──
        let a_global_row = row;
        let a_global_col = t * TILE_SIZE + lid.y;
        if (a_global_row < uniforms.M && a_global_col < uniforms.K) {
            tile_a[lid.x][lid.y] = a[a_global_row * uniforms.K + a_global_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        // ── Load tile of B (packed Q4 weight) ──
        // Flat layout: b_packed[pair_idx * N + col] where pair_idx = k/2
        // 4 packed bytes per u32, each byte has 2 Q4 values (low/high nibble)
        let b_j = col;
        let b_k = t * TILE_SIZE + lid.x;
        if (b_j < uniforms.N && b_k < uniforms.K) {
            let pair_idx = b_k / 2u;
            let q4_in_pair = b_k % 2u;
            let byte_idx = pair_idx * uniforms.N + b_j;
            let u32_idx = byte_idx / 4u;
            let byte_in_u32 = byte_idx % 4u;

            let packed_val = b_packed[u32_idx];
            let q4_val = extract_q4_from_byte(packed_val, byte_in_u32, q4_in_pair);

            let group = b_k / uniforms.GroupSize;
            let scale = scales[group * uniforms.N + b_j];
            tile_b[lid.x][lid.y] = q4_val * scale;
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[lid.x][i] * tile_b[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        c[row * uniforms.N + col] = sum;
    }
}
"#;

pub(crate) const ELEMENTWISE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Cfg {
    numel: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(3) var<uniform> cfg: Cfg;

fn gelu_f32(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn elementwise_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) {
        return;
    }

    let a_val = a[i];
    // For binary ops, b[i] is used; for unary, b[i] is same as a[i] or ignored
    let b_val = b[i];

    switch cfg.op {
        case 0: { // Add
            out[i] = a_val + b_val;
        }
        case 1: { // Sub
            out[i] = a_val - b_val;
        }
        case 2: { // Mul
            out[i] = a_val * b_val;
        }
        case 3: { // Div
            out[i] = a_val / b_val;
        }
        case 4: { // Neg
            out[i] = -a_val;
        }
        case 5: { // Exp
            out[i] = exp(a_val);
        }
        case 6: { // Ln
            out[i] = log(max(a_val, 1.0e-38));
        }
        case 7: { // Powf
            out[i] = pow(a_val, b_val);
        }
        case 8: { // Sqrt
            out[i] = sqrt(a_val);
        }
        case 9: { // Relu
            out[i] = max(a_val, 0.0);
        }
        case 10: { // Gelu
            out[i] = gelu_f32(a_val);
        }
        case 11: { // Sigmoid
            out[i] = sigmoid_f32(a_val);
        }
        case 12: { // Tanh
            out[i] = tanh(a_val);
        }
        case 13: { // Silu
            out[i] = a_val * sigmoid_f32(a_val);
        }
        case 14: { // LeakyRelu — b_val acts as negative_slope
            out[i] = select(a_val * b_val, a_val, a_val > 0.0);
        }
        case 15: { // BinaryCrossEntropy — a=prediction, b=target
            let p = clamp(a_val, 1.0e-7, 1.0 - 1.0e-7);
            out[i] = -(b_val * log(p) + (1.0 - b_val) * log(1.0 - p));
        }
        case 17: { // Swiglu — gate(a) * x(b)
            out[i] = (a_val * sigmoid_f32(a_val)) * b_val;
        }
        case 18: { // Step — 1 if a > 0 else 0
            out[i] = select(0.0, 1.0, a_val > 0.0);
        }
        default: {
            out[i] = a_val;
        }
    }
}
"#;

pub(crate) const ELEMENTWISE_INPLACE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

struct Cfg {
    numel: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(1) var<uniform> cfg: Cfg;

fn gelu_f32(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

fn sigmoid_f32(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn elementwise_inplace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= cfg.numel) {
        return;
    }

    let x = data[i];
    switch cfg.op {
        case 4: { data[i] = -x; }
        case 5: { data[i] = exp(x); }
        case 6: { data[i] = log(x); }
        case 7: { data[i] = pow(x, 0.0); }
        case 8: { data[i] = sqrt(x); }
        case 9: { data[i] = max(x, 0.0); }
        case 10: { data[i] = gelu_f32(x); }
        case 11: { data[i] = sigmoid_f32(x); }
        case 12: { data[i] = tanh(x); }
        case 13: { data[i] = x * sigmoid_f32(x); }
        case 14: { // LeakyRelu — use _pad0 reinterpreted as f32 for negative_slope
            let slope = bitcast<f32>(cfg._pad0);
            data[i] = select(x * slope, x, x > 0.0);
        }
        case 18: { // Step — 1 if x > 0 else 0
            data[i] = select(0.0, 1.0, x > 0.0);
        }
        default: { data[i] = x; }
    }
}
"#;

// ── Phase 1.1.5: Gradient Clip ────────────────────────────────────────────────

pub(crate) const GRADIENT_CLIP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(1) var<storage, read> norm_sq: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> wg_scale: f32;
var<workgroup> wg_norm: f32;

@compute @workgroup_size(BLOCK_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let max_norm = bitcast<f32>(cfg.y);

    if (lid.x == 0u) {
        let n = norm_sq[0];
        wg_norm = sqrt(n);
        if (wg_norm > max_norm && wg_norm > 0.0) {
            wg_scale = max_norm / wg_norm;
            output[3] = 1.0;
        } else {
            wg_scale = 1.0;
            output[3] = 0.0;
        }
        output[0] = wg_norm;
        output[1] = max_norm;
        output[2] = wg_scale;
    }
    workgroupBarrier();

    let numel = arrayLength(&gradient);
    if (idx < numel) {
        gradient[idx] = gradient[idx] * wg_scale;
    }
}
"#;

// ── Phase 1.2: Reduce ─────────────────────────────────────────────────────────

pub(crate) const REDUCE_WGSL_TEMPLATE: &str = r#"
const BLOCK_SIZE: u32 = 256;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Cfg {
    numel: u32,
    num_groups: u32,
};

@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

fn reduce_op(a: f32, b: f32) -> f32 {
    let op = {{OP}}u;
    // Sum=0, Max=1, Min=2
    if (op == 0u) { return a + b; }
    if (op == 1u) { if (a > b) { return a; } else { return b; } }
    return min(a, b);
}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce_main(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wg_id: vec3<u32>) {
    let group_idx = wg_id.x;
    let items_per_group = (cfg.numel + cfg.num_groups - 1) / cfg.num_groups;

    // Load
    var val: f32;
    let base = group_idx * items_per_group;
    let idx = base + lid.x;
    if (idx < cfg.numel) {
        val = input[idx];
    } else {
        val = 0.0;
    }
    scratch[lid.x] = val;
    workgroupBarrier();

    // Tree-reduce
    var stride = BLOCK_SIZE / 2;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = reduce_op(scratch[lid.x], scratch[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2;
    }

    // Write result
    if (lid.x == 0u) {
        output[group_idx] = scratch[0];
    }
}
"#;

// ── Phase 2.1: Softmax (stable, per-row) ──────────────────────────────────────

pub(crate) const SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn softmax_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let dim = cfg.y;
    let row = wg.x;
    if (row >= batch) { return; }

    let base = row * dim;

    // === Pass 1: find row max ===
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < dim) {
        let v = input[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = mx;
    workgroupBarrier();

    // tree-reduce max
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (scratch[lid.x] < scratch[lid.x + stride]) {
                scratch[lid.x] = scratch[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0u];
    workgroupBarrier();

    // === Pass 2: exp(x - max) and sum ===
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        sum += e;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    // tree-reduce sum
    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = scratch[0u];
    workgroupBarrier();

    // === Pass 3: normalize ===
    var norm_sum = row_sum;
    if (norm_sum == 0.0) { norm_sum = 1.0; }
    i = lid.x;
    while (i < dim) {
        output[base + i] = output[base + i] / norm_sum;
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.2: RMSNorm ───────────────────────────────────────────────────────

pub(crate) const RMSNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn rms_norm_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;

    let base = row * dim;

    // === Pass 1: sum(x²) ===
    var ss: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        let v = input[base + i];
        ss += v * v;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = ss;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let rms = sqrt(scratch[0u] / f32(dim) + eps);
    workgroupBarrier();

    // === Pass 2: normalize ===
    i = lid.x;
    while (i < dim) {
        output[base + i] = (input[base + i] / rms) * weight[i];
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.3: RMSNorm Backward ──────────────────────────────────────────────

pub(crate) const RMSNORM_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn rms_norm_bwd_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;
    let base = row * dim;

    // Pass 1: sum(x²) → rms → inv_rms
    var ssq: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        let xv = input[base + i];
        ssq += xv * xv;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = ssq;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let rms = sqrt(scratch[0u] / f32(dim) + eps);
    let inv_rms = 1.0 / rms;

    // Pass 2: sum(grad * x)
    var sum_x_g: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        sum_x_g += input[base + i] * grad[base + i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum_x_g;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let row_sum_x_g = scratch[0u];

    let rms_grad_factor = -inv_rms * inv_rms * inv_rms / f32(dim);

    // Pass 3: compute dx (per element) + accumulate dw (atomic across rows)
    i = lid.x;
    while (i < dim) {
        let xv = input[base + i];
        let gv = grad[base + i];
        let wv = weight[i];
        dx[base + i] = gv * wv * inv_rms + wv * xv * rms_grad_factor * row_sum_x_g;
        // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
        loop {
            let prev_bits = atomicLoad(&dw[i]);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + gv * xv * inv_rms);
            let res = atomicCompareExchangeWeak(&dw[i], prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.3: Cross-Entropy ─────────────────────────────────────────────────

pub(crate) const CROSS_ENTROPY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn cross_entropy_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let num_classes = cfg.y;
    let row = wg.x;
    if (row >= batch) { return; }

    let base = row * num_classes;
    let label = u32(targets[row]);

    // === Pass 1: find row max (stable) ===
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < num_classes) {
        let v = logits[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = mx;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (scratch[lid.x] < scratch[lid.x + stride]) {
                scratch[lid.x] = scratch[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0u];
    workgroupBarrier();

    // === Pass 2: sum(exp(x - max)) ===
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < num_classes) {
        sum += exp(logits[base + i] - row_max);
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let log_sum_exp = log(scratch[0u]) + row_max;
    workgroupBarrier();

    // === Pass 3: loss = log_sum_exp - logits[label] ===
    if (lid.x == 0u) {
        let target_logit = logits[base + label];
        losses[row] = log_sum_exp - target_logit;
    }
}
"#;

pub(crate) const CROSS_ENTROPY_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> softmax: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256u)
fn cross_entropy_bwd_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let batch = cfg.x;
    let classes = cfg.y;
    let total = batch * classes;
    let idx = gid.x;
    if (idx >= total) { return; }
    let b = idx / classes;
    let c = idx % classes;
    let t = u32(targets[b]);
    let p = softmax[idx];
    let g = grad[b];
    d_logits[idx] = g * (p - select(0.0, 1.0, c == t));
}
"#;

// ── Phase 2.4: Embedding (gather) ────────────────────────────────────────────

pub(crate) const EMBEDDING_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ids: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn embedding_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vocab_size = cfg.x;
    let dim = cfg.y;
    let seq_len = cfg.z;

    let idx = gid.x;
    if (idx >= seq_len * dim) { return; }

    let d = idx % dim;
    let s = idx / dim;
    let token_id = u32(ids[s]);
    output[idx] = weight[token_id * dim + d];
}
"#;

pub(crate) const EMBEDDING_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ids: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_weight: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256u)
fn embedding_backward_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vocab_size = cfg.x;
    let dim = cfg.y;
    let num_ids = cfg.z;
    let total = num_ids * dim;

    let idx = gid.x;
    if (idx >= total) { return; }

    let d = idx % dim;
    let s = idx / dim;
    let token_id = u32(ids[s]);
    if (token_id < vocab_size) {
        let g = grad[idx];
        // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
        let atom_ptr = &d_weight[token_id * dim + d];
        loop {
            let prev_bits = atomicLoad(atom_ptr);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + g);
            let res = atomicCompareExchangeWeak(atom_ptr, prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
    }
}
"#;

// ── Phase 2.5: LayerNorm ────────────────────────────────────────────────────

pub(crate) const LAYERNORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn layer_norm_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;

    let base = row * dim;

    // === Pass 1: mean ===
    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        sum += input[base + i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = scratch[0u] / f32(dim);
    workgroupBarrier();

    // === Pass 2: variance ===
    var var_sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let diff = input[base + i] - mean;
        var_sum += diff * diff;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = var_sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let variance = scratch[0u] / f32(dim);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // === Pass 3: normalize ===
    i = lid.x;
    while (i < dim) {
        let normalized = (input[base + i] - mean) * inv_std;
        output[base + i] = normalized * weight[i] + bias[i];
        i += BLOCK_SIZE;
    }
}
"#;

// ── Phase 2.6: Transpose ────────────────────────────────────────────────────

pub(crate) const TRANSPOSE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = cfg.x;
    let cols = cfg.y;
    let idx = gid.x;
    if (idx >= rows * cols) { return; }
    let r = idx / cols;
    let c = idx % cols;
    output[c * rows + r] = input[r * cols + c];
}
"#;

// ── Phase 3: Fused Attention (Flash Attention-style) ─────────────────────────

pub(crate) const FUSED_ATTENTION_WGSL: &str = r#"
const BLOCK_SIZE = 256u;
const TILE_SIZE = 32u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg1: vec4<u32>;
@group(0) @binding(5) var<uniform> cfg2: vec4<u32>;

var<workgroup> q_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> kv_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> score_scratch: array<f32, TILE_SIZE>;
var<workgroup> exp_scratch: array<f32, TILE_SIZE>;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn fused_attention_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch = cfg1.x;
    let heads = cfg1.y;
    let seq_len = cfg1.z;
    let dim = cfg1.w;
    let scale = bitcast<f32>(cfg2.x);
    let causal = cfg2.y;

    let tid = lid.x;

    let wg_base = cfg2.z;
    let wg_flat = wg_base + wg_id.x;
    let q_pos = wg_flat % seq_len;
    let head = (wg_flat / seq_len) % heads;
    let batch_idx = wg_flat / (seq_len * heads);

    if (batch_idx >= batch) { return; }
    if (tid >= dim) { return; }

    let head_stride = seq_len * dim;
    let batch_stride = heads * head_stride;
    let q_off = batch_idx * batch_stride + head * head_stride + q_pos * dim;
    let kv_base = batch_idx * batch_stride + head * head_stride;

    // Load Q row
    q_scratch[tid] = q[q_off + tid];
    workgroupBarrier();

    // Online softmax state
    var m: f32 = -3.402823e+38;
    var d: f32 = 0.0;
    var o: f32 = 0.0;

    // Loop over KV tiles
    var tile_start: u32 = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // === Step 1: compute scores for this tile ===
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;

            kv_scratch[tid] = k[kv_base + k_pos * dim + tid];
            workgroupBarrier();

            var partial = q_scratch[tid] * kv_scratch[tid];
            scratch[tid] = partial;
            workgroupBarrier();

            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) {
                    scratch[tid] = scratch[tid] + scratch[tid + stride];
                }
                workgroupBarrier();
                stride = stride / 2u;
            }

            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) {
                    s = -3.402823e+38;
                }
                score_scratch[ki] = s;
            }
            workgroupBarrier();
        }

        // === Step 2: max of tile scores ===
        if (tid < tile_size) {
            scratch[tid] = score_scratch[tid];
        } else {
            scratch[tid] = -3.402823e+38;
        }
        workgroupBarrier();

        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) {
                    scratch[tid] = scratch[tid + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let m_tile = scratch[0u];
        workgroupBarrier();

        // === Step 3: exp(score - m_tile) and sum ===
        if (tid < tile_size) {
            let e = exp(score_scratch[tid] - m_tile);
            exp_scratch[tid] = e;
            scratch[tid] = e;
        } else {
            scratch[tid] = 0.0;
        }
        workgroupBarrier();

        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                scratch[tid] = scratch[tid] + scratch[tid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let sum_exp_tile = scratch[0u];
        workgroupBarrier();

        // === Step 4: Online softmax update ===
        let m_new = max(m, m_tile);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(m_tile - m_new);

        d = old_scale * d + tile_scale * sum_exp_tile;
        o = o * old_scale;

        // === Step 5: Accumulate V ===
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = v[kv_base + k_pos * dim + tid];
            workgroupBarrier();

            o += tile_scale * exp_scratch[ki] * kv_scratch[tid];
            workgroupBarrier();
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // === Final: output = o / d ===
    output[q_off + tid] = o / d;
}
"#;

pub(crate) const FUSED_ATTENTION_BACKWARD_WGSL: &str = r#"
const BLOCK_SIZE = 256u;
const TILE_SIZE = 32u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> dO: array<f32>;
@group(0) @binding(4) var<storage, read_write> dq: array<f32>;
@group(0) @binding(5) var<storage, read_write> dk: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> dv: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> cfg1: vec4<u32>;
@group(0) @binding(8) var<uniform> cfg2: vec4<u32>;

var<workgroup> q_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> kv_scratch: array<f32, BLOCK_SIZE>;
var<workgroup> score_scratch: array<f32, TILE_SIZE>;
var<workgroup> exp_scratch: array<f32, TILE_SIZE>;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn fused_attn_backward_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch_heads = cfg1.x;
    let seq_len = cfg1.y;
    let dim = cfg1.z;
    let scale = bitcast<f32>(cfg1.w);
    let causal = cfg2.x;

    let tid = lid.x;
    let wg_base = cfg2.y;
    let wg_flat = wg_base + wg_id.x;
    let q_pos = wg_flat % seq_len;
    let head = wg_flat / seq_len;
    if (head >= batch_heads) { return; }
    if (tid >= dim) { return; }

    let head_stride = seq_len * dim;
    let q_off = head * head_stride + q_pos * dim;
    let base = head * head_stride;

    // Load Q row into workgroup memory
    q_scratch[tid] = q[q_off + tid];
    workgroupBarrier();

    // ── Pass 1: find softmax normalization constants + sum(P*dP) ──
    var m: f32 = -3.402823e+38;
    var d_norm: f32 = 0.0;
    var sum_P_dP: f32 = 0.0;

    var tile_start: u32 = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // Compute scores for this tile
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = k[base + k_pos * dim + tid];
            workgroupBarrier();
            var partial = q_scratch[tid] * kv_scratch[tid];
            scratch[tid] = partial;
            workgroupBarrier();
            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
                workgroupBarrier();
                stride = stride / 2u;
            }
            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) { s = -3.402823e+38; }
                score_scratch[ki] = s;
            }
            workgroupBarrier();
        }

        // Tile max
        if (tid < tile_size) { scratch[tid] = score_scratch[tid]; }
        else { scratch[tid] = -3.402823e+38; }
        workgroupBarrier();
        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) { scratch[tid] = scratch[tid + stride]; }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_max = scratch[0u];
        workgroupBarrier();

        // Tile exp(score - tile_max) and sum
        if (tid < tile_size) {
            let e = exp(score_scratch[tid] - tile_max);
            exp_scratch[tid] = e;
            scratch[tid] = e;
        } else { scratch[tid] = 0.0; }
        workgroupBarrier();
        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_sum = scratch[0u];
        workgroupBarrier();

        // Online softmax: d = old_scale * d + tile_scale * tile_sum
        let m_new = max(m, tile_max);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(tile_max - m_new);
        let d_prev = d_norm;
        d_norm = old_scale * d_norm + tile_scale * tile_sum;

        // Compute dP and accumulate sum(P*dP)
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = v[base + k_pos * dim + tid];
            workgroupBarrier();
            let dp_partial = dO[q_off + tid] * kv_scratch[tid];
            scratch[tid] = dp_partial;
            workgroupBarrier();
            var stride2 = BLOCK_SIZE / 2u;
            while (stride2 > 0u) {
                if (tid < stride2) { scratch[tid] = scratch[tid] + scratch[tid + stride2]; }
                workgroupBarrier();
                stride2 = stride2 / 2u;
            }
            if (tid == 0u) {
                let P_k = tile_scale * exp_scratch[ki] / d_norm;
                let dP_k = scratch[0u];
                // Rescale old contributions: old_scale * d_prev / d_norm * sum_P_dP
                // Add new: P_k * dP_k
                sum_P_dP = old_scale * d_prev / d_norm * sum_P_dP + P_k * dP_k;
            }
            workgroupBarrier();
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // ── Pass 2: recompute, compute dS, accumulate dQ + dK + dV ──
    m = -3.402823e+38;
    d_norm = 0.0;
    var dq_acc: f32 = 0.0;

    tile_start = 0u;
    while (tile_start < seq_len) {
        let tile_end = min(tile_start + TILE_SIZE, seq_len);
        let tile_size = tile_end - tile_start;

        // Recompute scores
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            kv_scratch[tid] = k[base + k_pos * dim + tid];
            workgroupBarrier();
            var partial = q_scratch[tid] * kv_scratch[tid];
            scratch[tid] = partial;
            workgroupBarrier();
            var stride = BLOCK_SIZE / 2u;
            while (stride > 0u) {
                if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
                workgroupBarrier();
                stride = stride / 2u;
            }
            if (tid == 0u) {
                var s = scratch[0u] / scale;
                if (causal == 1u && k_pos > q_pos) { s = -3.402823e+38; }
                score_scratch[ki] = s;
            }
            workgroupBarrier();
        }

        // Tile max
        if (tid < tile_size) { scratch[tid] = score_scratch[tid]; }
        else { scratch[tid] = -3.402823e+38; }
        workgroupBarrier();
        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) {
                if (scratch[tid] < scratch[tid + stride]) { scratch[tid] = scratch[tid + stride]; }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_max = scratch[0u];
        workgroupBarrier();

        if (tid < tile_size) {
            let e = exp(score_scratch[tid] - tile_max);
            exp_scratch[tid] = e;
            scratch[tid] = e;
        } else { scratch[tid] = 0.0; }
        workgroupBarrier();
        stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let tile_sum = scratch[0u];
        workgroupBarrier();

        let m_new = max(m, tile_max);
        let old_scale = exp(m - m_new);
        let tile_scale = exp(tile_max - m_new);
        d_norm = old_scale * d_norm + tile_scale * tile_sum;

        // For each key in this tile: dQ, dK, dV
        for (var ki: u32 = 0u; ki < tile_size; ki++) {
            let k_pos = tile_start + ki;
            let P_k = tile_scale * exp_scratch[ki] / d_norm;

            // dP_k = dot(dO[q_pos], V[k_pos]) — recompute
            kv_scratch[tid] = v[base + k_pos * dim + tid];
            workgroupBarrier();
            let dp_partial = dO[q_off + tid] * kv_scratch[tid];
            scratch[tid] = dp_partial;
            workgroupBarrier();
            var stride3 = BLOCK_SIZE / 2u;
            while (stride3 > 0u) {
                if (tid < stride3) { scratch[tid] = scratch[tid] + scratch[tid + stride3]; }
                workgroupBarrier();
                stride3 = stride3 / 2u;
            }
            let dP_k = scratch[0u];
            workgroupBarrier();

            // dS = P_k * (dP_k - sum_P_dP)
            let dS = P_k * (dP_k - sum_P_dP);

            // Load K for dQ + dK
            kv_scratch[tid] = k[base + k_pos * dim + tid];
            workgroupBarrier();

            // dQ[d] += dS * K[k, d] / scale
            dq_acc += dS * kv_scratch[tid] / scale;

            // dV[k, d] += P_k * dO[q, d]  (atomic)
            let dv_off = base + k_pos * dim + tid;
            let dv_val = P_k * dO[q_off + tid];
            // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
            loop {
                let prev_bits = atomicLoad(&dv[dv_off]);
                let prev_val = bitcast<f32>(prev_bits);
                let new_bits = bitcast<u32>(prev_val + dv_val);
                let res = atomicCompareExchangeWeak(&dv[dv_off], prev_bits, new_bits);
                if (res.exchanged) { break; }
            }

            // dK[k, d] += dS * Q[q, d] / scale  (atomic)
            let dk_off = base + k_pos * dim + tid;
            let dk_val = dS * q_scratch[tid] / scale;
            // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
            loop {
                let prev_bits = atomicLoad(&dk[dk_off]);
                let prev_val = bitcast<f32>(prev_bits);
                let new_bits = bitcast<u32>(prev_val + dk_val);
                let res = atomicCompareExchangeWeak(&dk[dk_off], prev_bits, new_bits);
                if (res.exchanged) { break; }
            }
        }

        m = m_new;
        tile_start += TILE_SIZE;
    }

    // Write dQ for this (head, q_pos, d)
    dq[q_off + tid] = dq_acc;
}
"#;

pub(crate) const FILL_ZERO_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256)
fn fill_zero_main(@builtin(global_invocation_id) id: vec3<u32>) {
    buf[id.x] = 0.0;
}
"#;

pub(crate) const FILL_CONSTANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec2<u32>;

@compute @workgroup_size(256)
fn fill_constant_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let numel = cfg.x;
    let value = bitcast<f32>(cfg.y);
    if (id.x < numel) {
        buf[id.x] = value;
    }
}
"#;

pub(crate) const FILL_ZERO_U32_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;

@compute @workgroup_size(256)
fn fill_zero_u32_main(@builtin(global_invocation_id) id: vec3<u32>) {
    buf[id.x] = 0u;
}
"#;

pub(crate) const MOE_SCATTER_ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> expert_out: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: vec2<u32>;

// cfg.x = hidden_size
// cfg.y = n_tokens_in_expert

@compute @workgroup_size(256)
fn moe_scatter_add_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let hidden_size = cfg.x;
    let n_tokens = cfg.y;
    let total = n_tokens * hidden_size;
    if (id.x >= total) { return; }
    let token_in_expert = id.x / hidden_size;
    let dim = id.x % hidden_size;
    let output_row = indices[token_in_expert];
    let weight = weights[token_in_expert];
    let out_idx = output_row * hidden_size + dim;
    output[out_idx] = output[out_idx] + expert_out[id.x] * weight;
}
"#;

pub(crate) const SCALE_INPLACE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec2<u32>;

@compute @workgroup_size(256)
fn scale_inplace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let numel = cfg.x;
    let scale = bitcast<f32>(cfg.y);
    if (id.x < numel) {
        buf[id.x] = buf[id.x] * scale;
    }
}
"#;

pub(crate) const GRADIENT_ALLREDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> grad_buffers: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_grads: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;
// cfg.x = numel per replica (total float count per gradient set)
// cfg.y = num_replicas

@compute @workgroup_size(256)
fn gradient_allreduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numel = cfg.x;
    let num_replicas = cfg.y;
    let idx = gid.x;
    if (idx >= numel) { return; }

    var sum = 0.0;
    for (var r = 0u; r < num_replicas; r++) {
        sum += grad_buffers[r * numel + idx];
    }
    out_grads[idx] = sum / f32(num_replicas);
}
"#;

pub(crate) const L2_NORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn l2_norm_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = cfg.x;
    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < numel) {
        sum += input[i] * input[i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        output[0] = scratch[0u];
    }
}
"#;

pub(crate) const CAUSAL_SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn causal_softmax_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let batch = cfg.x;
    let dim = cfg.y;
    let row = wg.x / dim;
    let col = wg.x % dim;
    if (row >= batch) { return; }

    let base = row * dim;
    let causal_end = col + 1u;

    // Pass 1: find max over causal prefix
    var mx: f32 = -3.402823e+38;
    var i = lid.x;
    while (i < causal_end) {
        let v = input[base + i];
        if (v > mx) { mx = v; }
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = mx;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (scratch[lid.x] < scratch[lid.x + stride]) {
                scratch[lid.x] = scratch[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0u];
    workgroupBarrier();

    // Pass 2: exp and sum over causal prefix
    var sum: f32 = 0.0;
    i = lid.x;
    while (i < causal_end) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        sum += e;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_sum = scratch[0u];
    workgroupBarrier();

    // Pass 3: normalize (only for positions <= col)
    var norm_sum = row_sum;
    if (norm_sum == 0.0) { norm_sum = 1.0; }
    i = lid.x;
    while (i < causal_end) {
        output[base + i] = output[base + i] / norm_sum;
        i += BLOCK_SIZE;
    }
    // Zero out positions beyond causal_end
    i = lid.x + causal_end;
    while (i < dim) {
        output[base + i] = 0.0;
        i += BLOCK_SIZE;
    }
}
"#;

pub(crate) const ROTARY_EMBEDDING_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> x : array<f32>;
@group(0) @binding(1) var<storage, read> cos : array<f32>;
@group(0) @binding(2) var<storage, read> sin : array<f32>;
struct RotaryConfig {
    total_rows: u32,
    dim: u32,
    half: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> cfg : RotaryConfig;

@compute @workgroup_size(256)
fn rotary_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= cfg.total_rows * cfg.half {
        return;
    }
    let row = idx / cfg.half;
    let pair = idx % cfg.half;

    let i1 = row * cfg.dim + pair;
    let i2 = row * cfg.dim + pair + cfg.half;

    let v1 = x[i1];
    let v2 = x[i2];
    let c = cos[pair];
    let s = sin[pair];

    x[i1] = v1 * c - v2 * s;
    x[i2] = v1 * s + v2 * c;
}
"#;

pub(crate) const REPEAT_HEADS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
struct RepeatConfig {
    seq: u32,
    kv_heads: u32,
    q_heads: u32,
    dim: u32,
};
@group(0) @binding(2) var<uniform> cfg : RepeatConfig;

@compute @workgroup_size(256)
fn repeat_heads_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = cfg.seq * cfg.q_heads * cfg.dim;
    if idx >= total {
        return;
    }
    // dst layout: [q_heads, seq, dim] (head-major, compatible with fused_attention)
    let qh = idx / (cfg.seq * cfg.dim);
    let rem1 = idx % (cfg.seq * cfg.dim);
    let s = rem1 / cfg.dim;
    let d = rem1 % cfg.dim;

    // Map q_heads -> kv_heads (grouped)
    let groups = cfg.q_heads / cfg.kv_heads;
    let kvh = qh / groups;
    let src_idx = s * (cfg.kv_heads * cfg.dim) + kvh * cfg.dim + d;
    dst[idx] = src[src_idx];
}
"#;
pub(crate) const TEMPERATURE_SCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

@compute @workgroup_size(256)
fn temperature_scale_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let temp = bitcast<f32>(cfg.x);
    let numel = cfg.y;
    if (id.x < numel) {
        logits[id.x] = logits[id.x] / temp;
    }
}
"#;

pub(crate) const TOP_K_MASK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> wg_buf: array<f32, BLOCK_SIZE>;
var<workgroup> wg_idx: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn top_k_mask_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let vocab = cfg.x;
    let k = cfg.y;
    let row = wg.x;
    let base = row * vocab;

    // Find top-k threshold using local maxima
    var local_max: f32 = -1.0;
    var i = lid.x;
    while (i < vocab) {
        let p = probs[base + i];
        if (p > local_max) { local_max = p; }
        i += BLOCK_SIZE;
    }
    wg_buf[lid.x] = local_max;
    workgroupBarrier();

    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            if (wg_buf[lid.x] < wg_buf[lid.x + stride]) {
                wg_buf[lid.x] = wg_buf[lid.x + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = wg_buf[0u];
    workgroupBarrier();

    // Compute threshold as a fraction of row_max
    let threshold = row_max * 0.01;

    // Count how many exceed threshold
    var count: u32 = 0u;
    i = lid.x;
    while (i < vocab) {
        if (probs[base + i] >= threshold) {
            count = count + 1u;
        }
        i += BLOCK_SIZE;
    }
    wg_idx[lid.x] = count;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            wg_idx[lid.x] = wg_idx[lid.x] + wg_idx[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total_above = wg_idx[0u];
    workgroupBarrier();

    // Only keep top-k
    let effective_k = min(k, total_above);
    var kept: u32 = 0u;
    i = lid.x;
    while (i < vocab) {
        if (probs[base + i] >= threshold && kept < effective_k) {
            kept = kept + 1u;
        } else if (probs[base + i] < threshold && lid.x < vocab) {
            probs[base + i] = 0.0;
        }
        i += BLOCK_SIZE;
    }
}
"#;

pub(crate) const TOP_P_MASK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> wg_buf: array<f32, BLOCK_SIZE>;
var<workgroup> wg_idx: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn top_p_mask_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let vocab = cfg.x;
    let top_p = bitcast<f32>(cfg.y);
    if (top_p <= 0.0 || top_p >= 1.0) {
        return;
    }
    let row = wg.x;
    let base = row * vocab;

    // Iteratively accumulate largest probabilities until cum >= top_p
    var threshold: f32 = 3.402823e+38;
    var cum: f32 = 0.0;

    for (var iter: u32 = 0u; iter < vocab; iter = iter + 1u) {
        // Find max among elements below current threshold
        var mx: f32 = -3.402823e+38;
        var mx_idx: u32 = 0u;
        var i = lid.x;
        while (i < vocab) {
            let v = probs[base + i];
            if (v > mx && v < threshold - 1e-6) {
                mx = v;
                mx_idx = i;
            }
            i += BLOCK_SIZE;
        }
        wg_buf[lid.x] = mx;
        wg_idx[lid.x] = mx_idx;
        workgroupBarrier();

        // Workgroup reduction: find max and its index
        var stride = BLOCK_SIZE / 2u;
        while (stride > 0u) {
            if (lid.x < stride) {
                if (wg_buf[lid.x] < wg_buf[lid.x + stride]) {
                    wg_buf[lid.x] = wg_buf[lid.x + stride];
                    wg_idx[lid.x] = wg_idx[lid.x + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        let max_val = wg_buf[0u];
        if (max_val <= 0.0) { break; }

        cum += max_val;
        threshold = max_val;

        if (cum >= top_p) { break; }
    }

    // Zero out everything below the threshold
    var i = lid.x;
    while (i < vocab) {
        if (probs[base + i] < threshold - 1e-6) {
            probs[base + i] = 0.0;
        }
        i += BLOCK_SIZE;
    }
}
"#;

pub(crate) const DROPOUT_MASK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> mask: array<f32>;
@group(0) @binding(1) var<uniform> cfg: vec4<u32>;

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

@compute @workgroup_size(256)
fn dropout_mask_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= cfg.x) { return; }
    let rate = bitcast<f32>(cfg.y);
    let scale = bitcast<f32>(cfg.z);
    var rng_state = cfg.w ^ (idx * 0x9E3779B9u);
    _ = xorshift64(&rng_state);
    let r = random_f32(&rng_state);
    mask[idx] = select(0.0, scale, r >= rate);
}
"#;

pub(crate) const MULTINOMIAL_SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;

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
fn multinomial_main(@builtin(workgroup_id) wg: vec3<u32>) {
    let vocab = cfg.x;
    let seed_lo = cfg.y;
    let seed_hi = cfg.z;
    let row = wg.x;
    let base = row * vocab;

    var rng_state = seed_lo ^ (row * 0x9E3779B9u);
    var warmup = xorshift64(&rng_state);
    let r = random_f32(&rng_state);

    var total: f32 = 0.0;
    for (var i: u32 = 0u; i < vocab; i = i + 1u) {
        total += probs[base + i];
    }

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
"#;

// ─── ADAM OPTIMIZER WGSL ───────────────────────────────────────────────────────

pub(crate) const ADAM_WGSL: &str = r#"
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
"#;

pub(crate) const LAYERNORM_BACKWARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> dw: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> db: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> cfg: vec4<u32>;

const BLOCK_SIZE = 256u;
var<workgroup> scratch: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn layer_norm_bwd_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let dim = cfg.y;
    let eps = bitcast<f32>(cfg.z);
    let row = wg.x;
    let base = row * dim;

    var sum: f32 = 0.0;
    var i = lid.x;
    while (i < dim) {
        sum += input[base + i];
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = scratch[0u] / f32(dim);

    var var_sum: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let diff = input[base + i] - mean;
        var_sum += diff * diff;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = var_sum;
    workgroupBarrier();
    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let variance = scratch[0u] / f32(dim);
    let sigma = sqrt(variance + eps);
    let inv_sigma = 1.0 / sigma;

    var sum_dy: f32 = 0.0;
    var sum_dy_xhat: f32 = 0.0;
    i = lid.x;
    while (i < dim) {
        let gv = grad[base + i];
        let x_hat = (input[base + i] - mean) * inv_sigma;
        sum_dy += gv;
        sum_dy_xhat += gv * x_hat;
        i += BLOCK_SIZE;
    }
    scratch[lid.x] = sum_dy;
    scratch[lid.x + BLOCK_SIZE / 2u] = sum_dy_xhat;
    workgroupBarrier();

    stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let row_sum_dy = scratch[0u];

    i = lid.x;
    while (i < BLOCK_SIZE / 2u) {
        scratch[i] = scratch[i + BLOCK_SIZE / 2u];
        i += BLOCK_SIZE / 2u;
    }
    workgroupBarrier();

    stride = BLOCK_SIZE / 4u;
    while (stride > 0u) {
        if (lid.x < stride) { scratch[lid.x] += scratch[lid.x + stride]; }
        workgroupBarrier();
        stride >>= 1u;
    }
    let row_sum_dy_xhat = scratch[0u];

    let inv_n = 1.0 / f32(dim);

    i = lid.x;
    while (i < dim) {
        let x_hat = (input[base + i] - mean) * inv_sigma;
        let gv = grad[base + i];
        let wv = weight[i];
        dx[base + i] = inv_sigma * (gv - row_sum_dy * inv_n - x_hat * row_sum_dy_xhat * inv_n);
        // inline atomic_add_f32 (wgpu 29.x forbids ptr<storage> as function arg)
        loop {
            let prev_bits = atomicLoad(&dw[i]);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + gv * x_hat);
            let res = atomicCompareExchangeWeak(&dw[i], prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
        loop {
            let prev_bits = atomicLoad(&db[i]);
            let prev_val = bitcast<f32>(prev_bits);
            let new_bits = bitcast<u32>(prev_val + gv);
            let res = atomicCompareExchangeWeak(&db[i], prev_bits, new_bits);
            if (res.old_value == prev_bits) { break; }
        }
        i += BLOCK_SIZE;
    }
}
"#;
