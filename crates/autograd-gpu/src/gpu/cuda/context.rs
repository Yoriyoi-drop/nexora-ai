use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaDevice, CudaSlice, CudaStream, CudaFunction, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use once_cell::sync::OnceCell;

use super::tensor::CudaTensor;

static CUDA_CTX: OnceCell<Arc<CudaRuntime>> = OnceCell::new();

/// Global CUDA runtime singleton.
pub struct CudaRuntime {
    pub device: CudaDevice,
    pub context: CudaContext,
    pub stream: CudaStream,
    pub blas: CudaBlas,
    pub blas_lt: CudaBlasLT,
    pub device_id: usize,
    kernels: Mutex<HashMap<String, CudaFunction>>,
    scratch: Mutex<Option<(usize, CudaSlice<f32>)>>,
}

impl CudaRuntime {
    pub fn init(device_id: usize) -> Result<&'static Self, String> {
        CUDA_CTX
            .get_or_try_init(|| Self::new(device_id).map(Arc::new))
            .map(|arc| arc.as_ref())
    }

    pub fn global() -> Result<&'static Self, String> {
        CUDA_CTX
            .get()
            .ok_or_else(|| "CUDA runtime not initialized. Call CudaRuntime::init() first".into())
    }

    /// Try to initialize CUDA runtime on device 0.
    /// Returns `None` if no CUDA device is available (graceful fallback).
    pub fn try_init() -> Option<&'static Self> {
        Self::init(0).ok()
    }

    fn new(device_id: usize) -> Result<Self, String> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| format!("Failed to create CudaDevice({device_id}): {e}"))?;
        let context = device.context().clone();
        let stream = device
            .default_stream()
            .map_err(|e| format!("Failed to get default stream: {e}"))?;
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("Failed to create CudaBlas: {e}"))?;
        let blas_lt = CudaBlasLT::new(stream.clone())
            .map_err(|e| format!("Failed to create CudaBlasLT: {e}"))?;
        Ok(Self {
            device,
            context,
            stream,
            blas,
            blas_lt,
            device_id,
            kernels: Mutex::new(HashMap::new()),
            scratch: Mutex::new(None),
        })
    }

    /// Get or allocate a scratch buffer of at least `numel` f32 elements.
    /// Contents are undefined (may contain stale data) — only use for ops where
    /// every element is fully written by the kernel (elementwise, unary, etc.).
    /// Avoids per-op `alloc_zeros` overhead.
    fn scratch_f32(&self, numel: usize) -> Result<CudaSlice<f32>, String> {
        let mut lock = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((cap, ref buf)) = *lock {
            if cap >= numel {
                return Ok(buf.clone());
            }
        }
        let buf = self
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA scratch alloc({numel}): {e}"))?;
        *lock = Some((numel, buf.clone()));
        Ok(buf)
    }

    // ── CUDA kernel compilation ──────────────────────────────────────

    fn get_or_compile_kernel(&self, name: &str, source: &str) -> Result<CudaFunction, String> {
        {
            let lock = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(f) = lock.get(name) {
                return Ok(f.clone());
            }
        }
        let ptx = compile_ptx(source).map_err(|e| format!("NVRTC compile '{name}': {e}"))?;
        let module = self
            .context
            .load_module(ptx)
            .map_err(|e| format!("CUDA load module '{name}': {e}"))?;
        let func = module
            .load_function(name)
            .map_err(|e| format!("CUDA load function '{name}': {e}"))?;
        self.kernels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), func.clone());
        Ok(func)
    }

    // ── Matmul via cuBLAS ────────────────────────────────────────────

    pub fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
        let a_shape = &a.shape;
        let b_shape = &b.shape;

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!(
                "cuBLAS matmul: both inputs must be 2D, got {}D and {}D",
                a_shape.len(),
                b_shape.len()
            ));
        }
        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "cuBLAS matmul: inner dims must match, got {} and {}",
                a_shape[1], b_shape[0]
            ));
        }

        let m = b_shape[1] as i32;
        let n = a_shape[0] as i32;
        let k = a_shape[1] as i32;

        let result_len = (m as usize) * (n as usize);
        let mut result = self
            .scratch_f32(result_len)
            .map_err(|e| format!("cuBLAS matmul scratch: {e}"))?;

        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            m,
            n,
            k,
            alpha: 1.0f32,
            lda: m,
            ldb: k,
            beta: 0.0f32,
            ldc: m,
        };

        unsafe {
            self.blas
                .gemm(cfg, &b.buffer, &a.buffer, &mut result)
                .map_err(|e| format!("cuBLAS gemm failed: {e}"))?;
        }

        Ok(CudaTensor {
            shape: vec![a_shape[0], b_shape[1]],
            buffer: result,
            device_id: self.device_id,
        })
    }

    // ── Elementwise binary ops ───────────────────────────────────────

    fn elementwise_binary(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        op: &str,
        expr: &str,
    ) -> Result<CudaTensor, String> {
        let numel = a.numel();
        if b.numel() != numel {
            if b.numel() == 1 {
                return self.elementwise_unary_scalar(a, b, op, expr);
            }
            return Err(format!(
                "CUDA elementwise '{}': shape mismatch {:?} vs {:?}",
                op, a.shape, b.shape
            ));
        }

        let kernel_name = format!("elem_bin_{}", op);
        let source = format!(
            r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, const float* __restrict__ b, size_t numel) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{ out[i] = {expr}; }}
}}
"#
        );
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;

        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA elem {} scratch: {e}", op))?;

        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(b.buffer());
            builder.arg(&numel);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA {} launch: {e}", op))?;
        }

        Ok(CudaTensor {
            shape: a.shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    fn elementwise_unary_scalar(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        op: &str,
        expr: &str,
    ) -> Result<CudaTensor, String> {
        let numel = a.numel();
        let kernel_name = format!("elem_scalar_{}", op);
        let source = format!(
            r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, float scalar, size_t numel) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{ out[i] = {expr}; }}
}}
"#
        );
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;

        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA scalar {} scratch: {e}", op))?;

        let scalar = self
            .stream
            .clone_dtoh_sync(b.buffer())
            .map_err(|e| format!("CUDA scalar {} readback: {e}", op))?
            .first()
            .copied()
            .unwrap_or(0.0);

        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(&scalar);
            builder.arg(&numel);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA {} launch: {e}", op))?;
        }

        Ok(CudaTensor {
            shape: a.shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    pub fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
        // Try same-shape or scalar first
        if a.numel() == b.numel() || b.numel() == 1 {
            return self.elementwise_binary(a, b, "add", "a[i] + b[i]");
        }
        // Broadcasting: bias [1, dim] + hidden [batch, dim]
        // Tile the smaller tensor across the larger one
        self.add_broadcast(a, b)
    }

    /// Broadcast add: supports [1, D] + [N, D] and [D] + [N, D] patterns.
    fn add_broadcast(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
        let (tensor, bias) = if a.numel() > b.numel() {
            (a, b)
        } else {
            (b, a)
        };
        let t_shape = tensor.shape();
        let b_numel = bias.numel();
        let b_flat = bias.numel() == 1;
        if t_shape.len() != 2 {
            return Err(format!(
                "CUDA broadcast add: tensor must be 2D, got {:?}",
                t_shape
            ));
        }
        let cols = t_shape[1];
        let rows = t_shape[0];
        if !b_flat && b_numel != cols {
            return Err(format!(
                "CUDA broadcast add: bias len {} != tensor cols {}",
                b_numel, cols
            ));
        }
        let numel = tensor.numel();
        let kernel_name = "add_broadcast_row";
        let source = r#"
extern "C" __global__ void add_broadcast_row(float* __restrict__ out,
    const float* __restrict__ tensor, const float* __restrict__ bias,
    size_t rows, size_t cols, int bias_is_scalar) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int row = i / cols;
    unsigned int col = i % cols;
    float b_val = bias_is_scalar ? bias[0] : bias[col];
    out[i] = tensor[i] + b_val;
}
"#;
        let func = self.get_or_compile_kernel(kernel_name, source)?;
        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA add_broadcast scratch: {e}"))?;
        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        let bias_is_scalar: i32 = if b_flat { 1 } else { 0 };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(tensor.buffer());
            builder.arg(bias.buffer());
            builder.arg(&rows);
            builder.arg(&cols);
            builder.arg(&bias_is_scalar);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA add_broadcast launch: {e}"))?;
        }
        Ok(CudaTensor {
            shape: t_shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    pub fn sub(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_binary(a, b, "sub", "a[i] - b[i]")
    }

    pub fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_binary(a, b, "mul", "a[i] * b[i]")
    }

    pub fn div(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_binary(a, b, "div", "a[i] / b[i]")
    }

    // ── Elementwise unary ops ────────────────────────────────────────

    pub fn neg(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "neg", "-a[i]")
    }

    pub fn exp(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "exp", "expf(a[i])")
    }

    pub fn sqrt(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "sqrt", "sqrtf(a[i])")
    }

    pub fn relu(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "relu", "fmaxf(0.0f, a[i])")
    }

    pub fn gelu(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(
            a,
            "gelu",
            "0.5f * a[i] * (1.0f + tanhf(0.79788456f * (a[i] + 0.044715f * a[i] * a[i] * a[i])))",
        )
    }

    pub fn gelu_inplace(&self, a: &mut CudaTensor) -> Result<(), String> {
        let result = self.gelu(a)?;
        a.buffer = result.buffer;
        Ok(())
    }

    pub fn silu(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "silu", "a[i] / (1.0f + expf(-a[i]))")
    }

    pub fn sigmoid(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "sigmoid", "1.0f / (1.0f + expf(-a[i]))")
    }

    pub fn ln(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "ln", "logf(a[i])")
    }

    pub fn tanh(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(a, "tanh", "tanhf(a[i])")
    }

    pub fn powf(&self, a: &CudaTensor, exponent: f32) -> Result<CudaTensor, String> {
        let numel = a.numel();
        let kernel_name = format!("elem_powf_{}", exponent.abs());
        let source = format!(
            r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, size_t numel) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{ out[i] = powf(a[i], {e}f); }}
}}
"#,
            e = exponent
        );
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;

        let mut out = self
            .stream
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA powf scratch: {e}"))?;

        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(&numel);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA powf launch: {e}"))?;
        }

        Ok(CudaTensor {
            shape: a.shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    fn elementwise_unary(
        &self,
        a: &CudaTensor,
        op: &str,
        expr: &str,
    ) -> Result<CudaTensor, String> {
        let numel = a.numel();
        let kernel_name = format!("elem_unary_{}", op);
        let source = format!(
            r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, size_t numel) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{ out[i] = {expr}; }}
}}
"#
        );
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;

        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA {} scratch: {e}", op))?;

        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(&numel);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA {} launch: {e}", op))?;
        }

        Ok(CudaTensor {
            shape: a.shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    // ── Transpose ────────────────────────────────────────────────────

    pub fn transpose(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        let shape = &a.shape;
        if shape.len() != 2 {
            return Err(format!(
                "CUDA transpose: expected 2D, got {}D",
                shape.len()
            ));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = a.numel();

        let func = self.get_or_compile_kernel(
            "transpose_2d",
            r#"
extern "C" __global__ void transpose_2d(float* __restrict__ out,
    const float* __restrict__ a, size_t rows, size_t cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols;
    unsigned int c = i % cols;
    out[c * rows + r] = a[i];
}
"#,
        )?;
        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA transpose scratch: {e}"))?;
        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(&rows);
            builder.arg(&cols);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA transpose launch: {e}"))?;
        }
        Ok(CudaTensor {
            shape: vec![cols, rows],
            buffer: out,
            device_id: self.device_id,
        })
    }

    // ── Fused Attention (FlashAttention-style) ──────────────────────

    pub fn fused_attention(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: f32,
        causal: bool,
    ) -> Result<CudaTensor, String> {
        let q_shape = &q.shape;
        if q_shape.len() != 4 {
            return Err(format!(
                "CUDA fused_attention: Q must be 4D [B,H,S,D], got {:?}",
                q_shape
            ));
        }
        let batch = q_shape[0];
        let heads = q_shape[1];
        let seq_len = q_shape[2];
        let dim = q_shape[3];

        let numel = q.numel();
        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA fused_attention scratch: {e}"))?;

        let kernel_name = "flash_attn";
        let block_size: u32 = 256;
        let tile_size: u32 = 32;
        let causal_u32: u32 = if causal { 1 } else { 0 };

        let source = format!(r#"
extern "C" __global__ void flash_attn(float* __restrict__ output,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    size_t batch, size_t heads, size_t seq_len, size_t dim, float scale, unsigned int causal) {{

    unsigned int wg_flat = blockIdx.x;
    unsigned int q_pos = wg_flat % seq_len;
    unsigned int head = (wg_flat / seq_len) % heads;
    unsigned int batch_idx = wg_flat / (seq_len * heads);

    if (batch_idx >= batch) return;

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* score_tile = shared + blockDim.x;
    float* exp_tile = shared + blockDim.x + {tile_size};
    float* scratch = shared + blockDim.x + 2 * {tile_size};

    unsigned int tid = threadIdx.x;

    size_t head_stride = seq_len * dim;
    size_t batch_stride = heads * head_stride;
    size_t q_off = batch_idx * batch_stride + head * head_stride + q_pos * dim;
    size_t kv_base = batch_idx * batch_stride + head * head_stride;

    // Load Q row
    if (tid < dim) {{
        q_shared[tid] = q[q_off + tid];
    }}
    __syncthreads();

    float m = -INFINITY;
    float d = 0.0f;
    float o_buf = 0.0f;

    for (unsigned int tile_start = 0; tile_start < seq_len; tile_start += {tile_size}) {{
        unsigned int tile_end = min(tile_start + {tile_size}, seq_len);
        unsigned int tile_sz = tile_end - tile_start;

        // Step 1: compute scores for this tile
        for (unsigned int ki = 0; ki < tile_sz; ki++) {{
            unsigned int k_pos = tile_start + ki;

            float partial = 0.0f;
            for (unsigned int d_i = tid; d_i < dim; d_i += blockDim.x) {{
                float kv_val = k[kv_base + k_pos * dim + d_i];
                partial += q_shared[d_i] * kv_val;
            }}

            // Warp-level reduction for partial sum
            #pragma unroll
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {{
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
            }}

            if (tid == 0) {{
                float s = partial * scale;
                if (causal && k_pos > q_pos) {{
                    s = -INFINITY;
                }}
                score_tile[ki] = s;
            }}
            __syncthreads();
        }}

        // Step 2: max of tile scores
        float m_tile = -INFINITY;
        for (unsigned int ki = tid; ki < tile_sz; ki += blockDim.x) {{
            if (score_tile[ki] > m_tile) m_tile = score_tile[ki];
        }}
        scratch[tid] = m_tile;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
            if (tid < s) {{
                if (scratch[tid + s] > scratch[tid]) scratch[tid] = scratch[tid + s];
            }}
            __syncthreads();
        }}
        m_tile = scratch[0];
        __syncthreads();

        // Step 3: exp(score - m_tile) and tile sum
        float sum_exp_tile = 0.0f;
        for (unsigned int ki = tid; ki < tile_sz; ki += blockDim.x) {{
            float e = expf(score_tile[ki] - m_tile);
            exp_tile[ki] = e;
            sum_exp_tile += e;
        }}
        scratch[tid] = sum_exp_tile;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
            if (tid < s) {{
                scratch[tid] += scratch[tid + s];
            }}
            __syncthreads();
        }}
        sum_exp_tile = scratch[0];
        __syncthreads();

        // Step 4: Online softmax update
        float m_new = fmaxf(m, m_tile);
        float old_scale = expf(m - m_new);
        float tile_scale = expf(m_tile - m_new);
        d = old_scale * d + tile_scale * sum_exp_tile;
        o_buf = o_buf * old_scale;

        // Step 5: Accumulate V contribution
        for (unsigned int ki = 0; ki < tile_sz; ki++) {{
            unsigned int k_pos = tile_start + ki;
            float v_val = (tid < dim) ? v[kv_base + k_pos * dim + tid] : 0.0f;
            o_buf += tile_scale * exp_tile[ki] * v_val;
        }}

        m = m_new;
    }}

    // Step 6: Normalize
    if (tid < dim) {{
        output[q_off + tid] = o_buf / d;
    }}
}}
"#);

        let kfunc = self.get_or_compile_kernel(kernel_name, &source)?;

        let total_wgs = (batch * heads * seq_len) as u32;
        let shared_mem = (block_size + 2 * tile_size + block_size) as u32 * 4; // 4 bytes per float

        let cfg = LaunchConfig {
            grid: total_wgs,
            block: block_size,
            shared_mem_bytes: shared_mem,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&kfunc);
            builder.arg(&mut out);
            builder.arg(q.buffer());
            builder.arg(k.buffer());
            builder.arg(v.buffer());
            builder.arg(&batch);
            builder.arg(&heads);
            builder.arg(&seq_len);
            builder.arg(&dim);
            builder.arg(&scale);
            builder.arg(&causal_u32);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA flash_attn launch: {e}"))?;
        }

        Ok(CudaTensor {
            shape: q_shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    // ── Softmax ──────────────────────────────────────────────────────

    pub fn softmax(&self, a: &CudaTensor) -> Result<CudaTensor, String> {
        let shape = &a.shape;
        if shape.len() != 2 {
            return Err(format!("CUDA softmax: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = a.numel();
        let block_size = cols.next_power_of_two().min(1024).max(32);

        let func = self.get_or_compile_kernel(
            "softmax_2d",
            r#"
extern "C" __global__ void softmax_2d(float* __restrict__ out,
    const float* __restrict__ a, size_t rows, size_t cols) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    // Pass 1: find per-thread max, then reduce
    float max_val = -INFINITY;
    for (unsigned int i = tid; i < cols; i += n) {
        float v = a[row * cols + i];
        if (v > max_val) max_val = v;
    }
    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = n / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }

    float global_max = shared[0];
    __syncthreads();

    // Pass 2: compute exp(x - max) and sum
    float sum = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {
        float e = expf(a[row * cols + i] - global_max);
        out[row * cols + i] = e;
        sum += e;
    }
    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = n / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float global_sum = shared[0];
    __syncthreads();

    // Pass 3: normalize
    float inv_sum = global_sum > 0.0f ? 1.0f / global_sum : 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {
        out[row * cols + i] *= inv_sum;
    }
}
"#,
        )?;

        let mut out = self
            .stream
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA softmax scratch: {e}"))?;

        let cfg = LaunchConfig {
            grid: rows as u32,
            block: block_size as u32,
            shared_mem_bytes: (block_size * 4) as u32,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(&rows);
            builder.arg(&cols);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA softmax launch: {e}"))?;
        }

        Ok(CudaTensor {
            shape: shape.clone(),
            buffer: out,
            device_id: self.device_id,
        })
    }

    // ── Scatter-add (fused MoE) ──────────────────────────────────────

    /// Scatter expert outputs into an output buffer with per-token weights.
    /// `output`: [batch_size, hidden_size] — accumulates in-place
    /// `expert_out`: [n_tokens, hidden_size] — expert forward result
    /// `token_indices`: [n_tokens] — global token positions (i32)
    /// `weights`: [n_tokens] — routing weights
    pub fn scatter_add_weighted(
        &self,
        output: &mut CudaTensor,
        expert_out: &CudaTensor,
        token_indices: &CudaSlice<i32>,
        weights: &CudaSlice<f32>,
    ) -> Result<(), String> {
        let shape = expert_out.shape();
        if shape.len() != 2 {
            return Err("scatter_add: expert_out must be 2D".into());
        }
        let n_tokens = shape[0] as i32;
        let hidden_size = shape[1] as i32;
        let numel = (n_tokens * hidden_size) as usize;

        let func = self.get_or_compile_kernel(
            "scatter_add_weighted",
            r#"
extern "C" __global__ void scatter_add_weighted(float* __restrict__ output,
    const float* __restrict__ expert_out,
    const int* __restrict__ token_indices,
    const float* __restrict__ weights,
    int n_tokens, int hidden_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (unsigned int)n_tokens * (unsigned int)hidden_size;
    if (i < total) {
        unsigned int token_local = i / (unsigned int)hidden_size;
        unsigned int feature = i % (unsigned int)hidden_size;
        int token_global = token_indices[token_local];
        float weight = weights[token_local];
        atomicAdd(&output[token_global * (unsigned int)hidden_size + feature],
                  expert_out[i] * weight);
    }
}
"#,
        )?;

        let cfg = LaunchConfig {
            grid: ((numel + 255) / 256) as u32,
            block: 256,
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut output.buffer);
            builder.arg(expert_out.buffer());
            builder.arg(token_indices);
            builder.arg(weights);
            builder.arg(&n_tokens);
            builder.arg(&hidden_size);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA scatter_add launch: {e}"))?;
        }
        Ok(())
    }

    // ── INT4 Matmul ──────────────────────────────────────────────────

    /// INT4 matmul: `C[M, N] = A[M, K] @ B_int4[K/2, N]` with per-group scales.
    ///
    /// Q4 packing: 2 values per byte (low nibble = k even, high nibble = k odd).
    /// B layout: `[(K/2), N]` row-major, indices `(k/2) * N + n`.
    /// Scales: `[groups, N]` — one scale per group_size elements per column.
    /// Dequant: `val = (q4 - 8) * scale` (symmetric, center at 0).
    pub fn matmul_int4(
        &self,
        a: &CudaTensor,
        b_packed: &CudaTensor,
        scales: &CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<CudaTensor, String> {
        let a_shape = a.shape();
        let b_shape = b_packed.shape();
        let s_shape = scales.shape();

        if a_shape != &vec![m, k] {
            return Err(format!(
                "CUDA matmul_int4: A shape mismatch, expected [M={m}, K={k}], got {:?}",
                a_shape
            ));
        }
        if b_shape.len() != 2 || b_shape[0] != k / 2 || b_shape[1] != n {
            return Err(format!(
                "CUDA matmul_int4: B shape mismatch, expected [{}, {}], got {:?}",
                k / 2,
                n,
                b_shape
            ));
        }
        let groups = k.div_ceil(group_size);
        if s_shape != &vec![groups, n] {
            return Err(format!(
                "CUDA matmul_int4: scales shape mismatch, expected [{groups}, {n}], got {:?}",
                s_shape
            ));
        }

        let num_groups = groups as i32;
        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let gs_i32 = group_size as i32;

        let numel = m * n;
        let mut out = self
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA matmul_int4 scratch: {e}"))?;

        // 1 thread per output element: grid(M, ceil(N/256)), block(256)
        let block_size: u32 = 256;
        let grid_x = m_i32 as u32;
        let grid_y = ((n_i32 as u32) + block_size - 1) / block_size;

        let kernel_name = "matmul_int4";
        let source = format!(
            r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, const unsigned char* __restrict__ b_packed,
    const float* __restrict__ scales, int M, int N, int K, int groups, int group_size) {{

    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int cols_per_thread = (N + blockDim.x - 1) / blockDim.x;

    for (int tj = 0; tj < cols_per_thread; tj++) {{
        int col = tid + tj * blockDim.x;
        if (col >= N) break;

        float sum = 0.0f;
        for (int g = 0; g < groups; g++) {{
            int g_start = g * group_size;
            int g_end = min(g_start + group_size, K);
            float scale = scales[g * N + col];

            for (int kk = g_start; kk < g_end; kk += 2) {{
                float a0 = a[row * K + kk];
                float a1 = a[row * K + kk + 1];

                int packed_idx = (kk / 2) * N + col;
                unsigned char packed = b_packed[packed_idx];

                int b0 = (packed >> 0) & 0x0F;
                int b1 = (packed >> 4) & 0x0F;
                float b0f = (b0 - 8.0f) * scale;
                float b1f = (b1 - 8.0f) * scale;

                sum += a0 * b0f + a1 * b1f;
            }}
        }}
        out[row * N + col] = sum;
    }}
}}
"#
        );

        let func = self.get_or_compile_kernel(kernel_name, &source)?;

        let cfg = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: block_size,
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out);
            builder.arg(a.buffer());
            builder.arg(b_packed.buffer());
            builder.arg(scales.buffer());
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&num_groups);
            builder.arg(&gs_i32);
            builder
                .launch(cfg)
                .map_err(|e| format!("CUDA matmul_int4 launch: {e}"))?;
        }

        Ok(CudaTensor {
            shape: vec![m, n],
            buffer: out,
            device_id: self.device_id,
        })
    }
}
