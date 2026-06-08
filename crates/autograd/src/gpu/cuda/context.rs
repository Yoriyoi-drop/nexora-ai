use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use once_cell::sync::OnceCell;

use super::tensor::CudaTensor;

static CUDA_CTX: OnceCell<Arc<CudaRuntime>> = OnceCell::new();

// ── PTX Disk Cache ─────────────────────────────────────────────────

/// Persists compiled PTX between runs to avoid NVRTC overhead on warm starts.
/// Cache key = hash(kernel_source + kernel_name), stored in ~/.cache/nexora/ptx/.
struct PtxCache {
    cache_dir: PathBuf,
}

impl PtxCache {
    fn new() -> Self {
        let cache_dir = std::env::var("HOME")
            .map(|h| PathBuf::from(h).join(".cache/nexora/ptx"))
            .unwrap_or_else(|_| PathBuf::from("/tmp/nexora/ptx"));
        let _ = std::fs::create_dir_all(&cache_dir);
        PtxCache { cache_dir }
    }

    fn key(name: &str, source: &str) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        source.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn load(&self, name: &str, source: &str) -> Option<String> {
        let path = self.cache_dir.join(Self::key(name, source));
        match std::fs::read_to_string(&path) {
            Ok(ptx) => {
                tracing::debug!("PTX cache hit: {}", path.display());
                Some(ptx)
            }
            Err(_) => None,
        }
    }

    fn save(&self, name: &str, source: &str, ptx: &str) {
        let path = self.cache_dir.join(Self::key(name, source));
        if let Err(e) = std::fs::write(&path, ptx) {
            tracing::warn!("PTX cache write failed: {e}");
        }
    }
}

macro_rules! _launch_1d {
    ($self:ident, $func:ident, $numel:expr, $out:ident $(, $arg:expr)*) => {{
        let cfg = LaunchConfig {
            grid_dim: ((($numel as u32 + 255) / 256).max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = $self.stream.launch_builder(&$func);
            builder.arg(&mut $out);
            $(builder.arg($arg);)*
            builder.launch(cfg).map_err(|e| format!("kernel launch: {e}"))?;
        }
    }};
}

macro_rules! compile_simple {
    ($self:ident, $name:expr, $source:expr) => {{
        $self.get_or_compile_kernel($name, $source)
    }};
}

/// Global CUDA runtime singleton.
pub struct CudaRuntime {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub blas_lt: CudaBlasLT,
    pub device_id: usize,
    kernels: Mutex<HashMap<String, CudaFunction>>,
    scratch: Mutex<Option<(usize, CudaSlice<f32>)>>,
    ptx_cache: PtxCache,
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
            .map(|arc| arc.as_ref())
            .ok_or_else(|| "CUDA runtime not initialized. Call CudaRuntime::init() first".into())
    }

    /// Try to initialize CUDA runtime on device 0.
    /// Returns `None` if no CUDA device is available (graceful fallback).
    pub fn try_init() -> Option<&'static Self> {
        Self::init(0).ok()
    }

    fn new(device_id: usize) -> Result<Self, String> {
        let context = CudaContext::new(device_id)
            .map_err(|e| format!("Failed to create CudaContext({device_id}): {e}"))?;
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("Failed to create CudaBlas: {e}"))?;
        let blas_lt = CudaBlasLT::new(stream.clone())
            .map_err(|e| format!("Failed to create CudaBlasLT: {e}"))?;
        Ok(Self {
            context,
            stream,
            blas,
            blas_lt,
            device_id,
            kernels: Mutex::new(HashMap::new()),
            scratch: Mutex::new(None),
            ptx_cache: PtxCache::new(),
        })
    }

    /// Get or allocate a scratch buffer of at least `numel` f32 elements.
    /// Contents are undefined (may contain stale data) — only use for ops where
    /// every element is fully written by the kernel (elementwise, unary, etc.).
    /// Avoids per-op `alloc_zeros` overhead.
    pub fn scratch_f32(&self, numel: usize) -> Result<CudaSlice<f32>, String> {
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
        let ptx: String = match self.ptx_cache.load(name, source) {
            Some(cached) => cached,
            None => {
                let compiled =
                    compile_ptx(source).map_err(|e| format!("NVRTC compile '{name}': {e}"))?;
                let ptx_str = compiled.to_src();
                self.ptx_cache.save(name, source, &ptx_str);
                ptx_str
            }
        };
        let module = self
            .context
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
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
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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
            .clone_dtoh(b.buffer())
            .map_err(|e| format!("CUDA scalar {} readback: {e}", op))?
            .first()
            .copied()
            .unwrap_or(0.0);

        let cfg = LaunchConfig {
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA powf scratch: {e}"))?;

        let cfg = LaunchConfig {
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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
            grid_dim: (total_wgs, 1, 1),
            block_dim: (block_size, 1, 1),
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

    // ── FlashDecoding ─────────────────────────────────────────────────

    /// Two-pass FlashDecoding: parallelizes over KV chunks for efficient decode (S_q=1).
    ///
    /// Pass 1: 1 block per (B, H, kv_chunk) — each block computes partial online softmax
    ///         output for its chunk. Writes chunked (m, d, o) to staging buffers.
    ///
    /// Pass 2: 1 block per (B, H) — reduces across chunks using online softmax combine,
    ///         writing the final attention output [B, H, 1, D].
    ///
    /// Chunk size = 128 by default. Number of chunks = ceil(kv_len / chunk_size).
    pub fn flash_decoding(
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
                "CUDA flash_decoding: Q must be 4D [B,H,S,D], got {:?}",
                q_shape
            ));
        }
        let batch = q_shape[0];
        let heads = q_shape[1];
        let dim = q_shape[3];

        let k_shape = &k.shape;
        let kv_len = k_shape[2];

        let tile_size: u32 = 32;
        let block_size: u32 = 256;
        let chunk_size: usize = 128;
        let num_chunks = ((kv_len + chunk_size - 1) / chunk_size).max(1);
        let causal_u32: u32 = if causal { 1 } else { 0 };

        // ── Pass 1: partial attention per KV chunk ───────────────────
        let partial_o_size = batch * heads * num_chunks * dim;
        let partial_md_size = batch * heads * num_chunks;

        let mut partial_o = self
            .scratch_f32(partial_o_size)
            .map_err(|e| format!("flash_decoding partial_o scratch: {e}"))?;
        let mut partial_m = self
            .scratch_f32(partial_md_size)
            .map_err(|e| format!("flash_decoding partial_m scratch: {e}"))?;
        let mut partial_d = self
            .scratch_f32(partial_md_size)
            .map_err(|e| format!("flash_decoding partial_d scratch: {e}"))?;

        let src_pass1 = format!(
            r#"
extern "C" __global__ void flash_decoding(
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_d,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    size_t batch, size_t heads, size_t kv_len, size_t dim,
    size_t num_chunks, unsigned int chunk_size,
    float scale, unsigned int causal) {{

    unsigned int wg_flat = blockIdx.x;
    unsigned int chunk = wg_flat % num_chunks;
    unsigned int head = (wg_flat / num_chunks) % heads;
    unsigned int batch_idx = wg_flat / (num_chunks * heads);
    if (batch_idx >= batch) return;

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* score_tile = shared + blockDim.x;
    float* exp_tile = shared + blockDim.x + {tile_size};
    float* scratch = shared + blockDim.x + 2 * {tile_size};

    unsigned int tid = threadIdx.x;

    size_t head_stride = kv_len * dim;
    size_t batch_stride = heads * head_stride;
    size_t q_off = batch_idx * batch_stride + head * head_stride;
    size_t kv_base = batch_idx * batch_stride + head * head_stride;

    if (tid < dim) q_shared[tid] = q[q_off + tid];
    __syncthreads();

    size_t kv_start = (size_t)chunk * chunk_size;
    size_t kv_end = kv_start + chunk_size;
    if (kv_end > kv_len) kv_end = kv_len;

    float m = -INFINITY;
    float d = 0.0f;
    float o_buf = 0.0f;

    for (size_t tile_start = kv_start; tile_start < kv_end; tile_start += {tile_size}) {{
        size_t tile_end = tile_start + {tile_size};
        if (tile_end > kv_end) tile_end = kv_end;
        unsigned int tile_sz = (unsigned int)(tile_end - tile_start);

        for (unsigned int ki = 0; ki < tile_sz; ki++) {{
            size_t k_pos = tile_start + ki;
            float partial = 0.0f;
            for (size_t d_i = tid; d_i < dim; d_i += blockDim.x) {{
                partial += k[kv_base + k_pos * dim + d_i] * q_shared[d_i];
            }}
            #pragma unroll
            for (unsigned int offset = 16; offset > 0; offset >>= 1) {{
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
            }}
            if (tid == 0) {{
                float s = partial * scale;
                if (causal && k_pos > 0) s = -INFINITY;
                score_tile[ki] = s;
            }}
            __syncthreads();
        }}

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

        float sum_exp_tile = 0.0f;
        for (unsigned int ki = tid; ki < tile_sz; ki += blockDim.x) {{
            float e = expf(score_tile[ki] - m_tile);
            exp_tile[ki] = e;
            sum_exp_tile += e;
        }}
        scratch[tid] = sum_exp_tile;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }}
        sum_exp_tile = scratch[0];
        __syncthreads();

        float m_new = fmaxf(m, m_tile);
        float old_scale = expf(m - m_new);
        float tile_scale = expf(m_tile - m_new);
        d = old_scale * d + tile_scale * sum_exp_tile;
        o_buf = o_buf * old_scale;

        for (unsigned int ki = 0; ki < tile_sz; ki++) {{
            size_t k_pos = tile_start + ki;
            float v_val = (tid < dim) ? v[kv_base + k_pos * dim + tid] : 0.0f;
            o_buf += tile_scale * exp_tile[ki] * v_val;
        }}

        m = m_new;
    }}

    size_t po_base = batch_idx * (heads * num_chunks * dim) + head * (num_chunks * dim);
    if (tid < dim) {{
        partial_o[po_base + chunk * dim + tid] = o_buf;
    }}
    if (tid == 0) {{
        size_t pm_base = batch_idx * (heads * num_chunks) + head * num_chunks;
        partial_m[pm_base + chunk] = m;
        partial_d[pm_base + chunk] = d;
    }}
}}
"#
        );

        let kfunc1 = self.get_or_compile_kernel("flash_decoding", &src_pass1)?;
        let total_wgs1 = (batch * heads * num_chunks) as u32;
        let shared_mem1 = (block_size + 2 * tile_size + block_size) as u32 * 4;

        let cfg1 = LaunchConfig {
            grid_dim: (total_wgs1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem1,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&kfunc1);
            builder.arg(&mut partial_o);
            builder.arg(&mut partial_m);
            builder.arg(&mut partial_d);
            builder.arg(q.buffer());
            builder.arg(k.buffer());
            builder.arg(v.buffer());
            builder.arg(&batch);
            builder.arg(&heads);
            builder.arg(&kv_len);
            builder.arg(&dim);
            builder.arg(&num_chunks);
            builder.arg(&chunk_size);
            builder.arg(&scale);
            builder.arg(&causal_u32);
            builder
                .launch(cfg1)
                .map_err(|e| format!("flash_decoding pass1 launch: {e}"))?;
        }

        // ── Pass 2: reduce across chunks ─────────────────────────────
        let numel_out = batch * heads * 1 * dim;
        let mut out = self
            .scratch_f32(numel_out)
            .map_err(|e| format!("flash_decoding output scratch: {e}"))?;

        let src_reduce = format!(
            r#"
extern "C" __global__ void flash_decoding_reduce(
    float* __restrict__ output,
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_d,
    size_t batch, size_t heads, size_t num_chunks, size_t dim) {{

    unsigned int head = blockIdx.x % heads;
    unsigned int batch_idx = blockIdx.x / heads;
    if (batch_idx >= batch) return;

    unsigned int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* m_sh = shared;
    float* d_sh = shared + num_chunks;

    size_t base_o = (batch_idx * heads + head) * num_chunks * dim;
    size_t base_md = (batch_idx * heads + head) * num_chunks;

    if (tid < num_chunks) {{
        m_sh[tid] = partial_m[base_md + tid];
        d_sh[tid] = partial_d[base_md + tid];
    }}
    __syncthreads();

    if (tid == 0) {{
        float m_global = -INFINITY;
        for (unsigned int c = 0; c < num_chunks; c++) {{
            if (m_sh[c] > m_global) m_global = m_sh[c];
        }}
        shared[num_chunks + num_chunks] = m_global;
    }}
    __syncthreads();
    float m_global = shared[num_chunks + num_chunks];

    if (tid < dim) {{
        float o_local = 0.0f;
        float d_global = 0.0f;
        for (unsigned int c = 0; c < num_chunks; c++) {{
            float chunk_scale = expf(m_sh[c] - m_global);
            d_global += chunk_scale * d_sh[c];
            o_local += chunk_scale * partial_o[base_o + c * dim + tid];
        }}
        output[(batch_idx * heads + head) * dim + tid] = o_local / d_global;
    }}
}}
"#
        );

        let kfunc2 = self.get_or_compile_kernel("flash_decoding_reduce", &src_reduce)?;
        let total_wgs2 = (batch * heads) as u32;
        let shared_mem2 = (num_chunks * 2 + 1) as u32 * 4;

        let cfg2 = LaunchConfig {
            grid_dim: (total_wgs2, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem2,
        };
        unsafe {
            let mut builder = self.stream.launch_builder(&kfunc2);
            builder.arg(&mut out);
            builder.arg(&partial_o);
            builder.arg(&partial_m);
            builder.arg(&partial_d);
            builder.arg(&batch);
            builder.arg(&heads);
            builder.arg(&num_chunks);
            builder.arg(&dim);
            builder
                .launch(cfg2)
                .map_err(|e| format!("flash_decoding reduce launch: {e}"))?;
        }

        Ok(CudaTensor {
            shape: vec![batch, heads, 1, dim],
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
            .scratch_f32(numel)
            .map_err(|e| format!("CUDA softmax scratch: {e}"))?;

        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
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
            grid_dim: (((numel + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
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

        if *a_shape != vec![m, k] {
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
        if *s_shape != vec![groups, n] {
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
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, 1, 1),
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

    // ── Buffer ops ──────────────────────────────────────────────────

    pub fn fill_zero(&self, t: &CudaTensor) -> Result<(), String> {
        let numel = t.numel();
        let func = compile_simple!(self, "fill_zero_f32", r#"
extern "C" __global__ void fill_zero_f32(float* buf, size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) buf[i] = 0.0f;
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&t.buffer); b.arg(&numel); b.launch(cfg).map_err(|e| format!("fill_zero: {e}"))?;
        }
        Ok(())
    }

    pub fn fill_constant(&self, t: &CudaTensor, value: f32) -> Result<(), String> {
        let numel = t.numel();
        let func = compile_simple!(self, "fill_constant_f32", r#"
extern "C" __global__ void fill_constant_f32(float* buf, float val, size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) buf[i] = val;
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&t.buffer); b.arg(&value); b.arg(&numel); b.launch(cfg).map_err(|e| format!("fill_constant: {e}"))?;
        }
        Ok(())
    }

    pub fn scale_inplace(&self, t: &CudaTensor, scale: f32) -> Result<(), String> {
        let numel = t.numel();
        let func = compile_simple!(self, "scale_inplace_f32", r#"
extern "C" __global__ void scale_inplace_f32(float* buf, float scale, size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) buf[i] *= scale;
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&t.buffer); b.arg(&scale); b.arg(&numel); b.launch(cfg).map_err(|e| format!("scale_inplace: {e}"))?;
        }
        Ok(())
    }

    // ── Reduce ops ──────────────────────────────────────────────────

    fn reduce_1d(&self, input: &CudaTensor, op: &str, expr: &str, init: &str) -> Result<CudaTensor, String> {
        let shape = input.shape();
        let numel = input.numel();
        if shape.len() != 1 && !(shape.len() == 2 && shape[0] == 1) {
            // Flatten for 1D reduction
            let flat = input.reshape(vec![1, numel])?;
            return self.reduce_1d(&flat, op, expr, init);
        }
        let kernel_name = format!("reduce_{}_1d", op);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, size_t numel) {{
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    float val = {init};
    for (unsigned int i = tid; i < numel; i += blockDim.x) {{
        val = {expr};
    }}
    shared[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] = {expr};
        }}
        __syncthreads();
    }}
    if (tid == 0) out[0] = shared[0];
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let mut out_buf = self.stream.alloc_zeros::<f32>(1).map_err(|e| format!("reduce scratch: {e}"))?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out_buf); b.arg(input.buffer()); b.arg(&numel);
            b.launch(cfg).map_err(|e| format!("reduce_{} launch: {e}", op))?;
        }
        Ok(CudaTensor { shape: vec![1], buffer: out_buf, device_id: self.device_id })
    }

    pub fn sum(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        self.reduce_1d(input, "sum", "val + a[i]", "0.0f")
    }

    pub fn max_val(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        self.reduce_1d(input, "max", "fmaxf(val, a[i])", "-INFINITY")
    }

    pub fn min_val(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        self.reduce_1d(input, "min", "fminf(val, a[i])", "INFINITY")
    }

    pub fn mean(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let sum = self.sum(input)?;
        let numel = input.numel() as f32;
        self.elementwise_unary_scalar_impl(&sum, "div", "a[i] / scalar", numel)
    }

    pub fn l2_norm(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let numel = input.numel();
        let kernel_name = "l2_norm_f32";
        let source = r#"
extern "C" __global__ void l2_norm_f32(float* __restrict__ out,
    const float* __restrict__ a, size_t numel) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < numel; i += blockDim.x) {
        sum_sq += a[i] * a[i];
    }
    shared[tid] = sum_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { shared[tid] += shared[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[0] = sqrtf(shared[0]);
}"#;
        let func = self.get_or_compile_kernel(kernel_name, source)?;
        let mut out = self.stream.alloc_zeros::<f32>(1).map_err(|e| format!("l2_norm scratch: {e}"))?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(input.buffer()); b.arg(&numel);
            b.launch(cfg).map_err(|e| format!("l2_norm launch: {e}"))?;
        }
        Ok(CudaTensor { shape: vec![1], buffer: out, device_id: self.device_id })
    }

    // ── RMS Norm ────────────────────────────────────────────────────

    pub fn rms_norm(&self, x: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor, String> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(format!("rms_norm: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = x.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("rms_norm scratch: {e}"))?;

        let kernel_name = format!("rms_norm_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ x, const float* __restrict__ weight,
    size_t rows, size_t cols, float eps) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        float v = x[row * cols + i];
        sum_sq += v * v;
    }}
    shared[tid] = sum_sq;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }}
    float rms = sqrtf(shared[0] / (float)cols + eps);
    float inv_rms = 1.0f / rms;
    for (unsigned int i = tid; i < cols; i += n) {{
        out[row * cols + i] = x[row * cols + i] * inv_rms * weight[i];
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let block = cols.next_power_of_two().min(256).max(32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: (block * 4) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(x.buffer()); b.arg(weight.buffer());
            b.arg(&rows); b.arg(&cols); b.arg(&eps);
            b.launch(cfg).map_err(|e| format!("rms_norm launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    pub fn rms_norm_backward(&self, input: &CudaTensor, weight: &CudaTensor, grad: &CudaTensor, eps: f32) -> Result<(CudaTensor, CudaTensor), String> {
        let shape = input.shape();
        let rows = shape[0];
        let cols = shape[1];
        let numel = input.numel();
        let mut dx = self.scratch_f32(numel).map_err(|e| format!("rms_norm_bwd scratch: {e}"))?;
        let mut dw = self.scratch_f32(cols).map_err(|e| format!("rms_norm_bwd dw scratch: {e}"))?;
        self.fill_zero(&CudaTensor { shape: vec![cols], buffer: dw.clone(), device_id: self.device_id })?;

        let kernel_name = format!("rms_norm_bwd_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(
    float* __restrict__ dx, float* __restrict__ dw,
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ grad,
    size_t rows, size_t cols, float eps) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        float v = x[row * cols + i];
        sum_sq += v * v;
    }}
    shared[tid] = sum_sq;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }}
    float rms = sqrtf(shared[0] / (float)cols + eps);
    float inv_rms = 1.0f / rms;
    float inv_norm = inv_rms / (float)cols;

    float sum_dot = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        float xi = x[row * cols + i];
        float gi = grad[row * cols + i];
        float wi = weight[i];
        float normed = xi * inv_rms;
        dx[row * cols + i] = gi * wi * inv_rms - normed * (gi * wi * xi * inv_norm);
        sum_dot += gi * normed;
    }}
    shared[tid] = sum_dot;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }}
    float row_dot = shared[0];

    for (unsigned int i = tid; i < cols; i += n) {{
        float xi = x[row * cols + i];
        float gi = grad[row * cols + i];
        float normed = xi * inv_rms;
        dx[row * cols + i] = (gi - row_dot * normed) * weight[i] * inv_rms;
    }}

    for (unsigned int i = tid; i < cols; i += n) {{
        float xi = x[row * cols + i];
        float gi = grad[row * cols + i];
        atomicAdd(&dw[i], gi * normed);
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let block = cols.next_power_of_two().min(256).max(32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: (block * 4) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut dx); b.arg(&mut dw);
            b.arg(input.buffer()); b.arg(weight.buffer()); b.arg(grad.buffer());
            b.arg(&rows); b.arg(&cols); b.arg(&eps);
            b.launch(cfg).map_err(|e| format!("rms_norm_bwd launch: {e}"))?;
        }
        Ok((CudaTensor { shape: shape.clone(), buffer: dx, device_id: self.device_id },
            CudaTensor { shape: vec![cols], buffer: dw, device_id: self.device_id }))
    }

    // ── Layer Norm ──────────────────────────────────────────────────

    pub fn layer_norm(&self, x: &CudaTensor, weight: &CudaTensor, bias: &CudaTensor, eps: f32) -> Result<CudaTensor, String> {
        let shape = x.shape();
        if shape.len() != 2 {
            return Err(format!("layer_norm: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = x.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("layer_norm scratch: {e}"))?;

        let kernel_name = format!("layer_norm_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, size_t rows, size_t cols, float eps) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    float sum = 0.0f, sum_sq = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        float v = x[row * cols + i];
        sum += v;
        sum_sq += v * v;
    }}
    shared[tid] = sum;
    shared[tid + n] = sum_sq;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] += shared[tid + s];
            shared[tid + n] += shared[tid + s + n];
        }}
        __syncthreads();
    }}
    float mean = shared[0] / (float)cols;
    float var = shared[n] / (float)cols - mean * mean;
    float inv_std = rsqrtf(var + eps);
    for (unsigned int i = tid; i < cols; i += n) {{
        out[row * cols + i] = (x[row * cols + i] - mean) * inv_std * weight[i] + bias[i];
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let block = cols.next_power_of_two().min(256).max(32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: (block * 8) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(x.buffer()); b.arg(weight.buffer()); b.arg(bias.buffer());
            b.arg(&rows); b.arg(&cols); b.arg(&eps);
            b.launch(cfg).map_err(|e| format!("layer_norm launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    pub fn layer_norm_backward(&self, input: &CudaTensor, weight: &CudaTensor, _bias: &CudaTensor, grad: &CudaTensor, eps: f32) -> Result<(CudaTensor, CudaTensor, CudaTensor), String> {
        let shape = input.shape();
        let rows = shape[0];
        let cols = shape[1];
        let numel = input.numel();
        let mut dx = self.scratch_f32(numel).map_err(|e| format!("ln_bwd dx scratch: {e}"))?;
        let mut dw = self.scratch_f32(cols).map_err(|e| format!("ln_bwd dw scratch: {e}"))?;
        let mut db = self.scratch_f32(cols).map_err(|e| format!("ln_bwd db scratch: {e}"))?;
        self.fill_zero(&CudaTensor { shape: vec![cols], buffer: dw.clone(), device_id: self.device_id })?;
        self.fill_zero(&CudaTensor { shape: vec![cols], buffer: db.clone(), device_id: self.device_id })?;

        let kernel_name = format!("ln_bwd_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(
    float* __restrict__ dx, float* __restrict__ dw, float* __restrict__ db,
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ grad, size_t rows, size_t cols, float eps) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    float sum = 0.0f, sum_sq = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        float v = x[row * cols + i];
        sum += v;
        sum_sq += v * v;
    }}
    shared[tid] = sum; shared[tid + n] = sum_sq;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ shared[tid] += shared[tid + s]; shared[tid + n] += shared[tid + s + n]; }}
        __syncthreads();
    }}
    float mean = shared[0] / (float)cols;
    float var = shared[n] / (float)cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    float d_mean = 0.0f, d_var = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        float xi = x[row * cols + i];
        float gi = grad[row * cols + i];
        float wi = weight[i];
        float normed = (xi - mean) * inv_std;
        d_mean += gi * wi * (-inv_std);
        d_var += gi * wi * (xi - mean) * (-0.5f * inv_std * inv_std * inv_std);
    }}
    shared[tid] = d_mean; shared[tid + n] = d_var;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ shared[tid] += shared[tid + s]; shared[tid + n] += shared[tid + s + n]; }}
        __syncthreads();
    }}
    float g_mean = shared[0], g_var = shared[n];

    for (unsigned int i = tid; i < cols; i += n) {{
        float xi = x[row * cols + i];
        float gi = grad[row * cols + i];
        float wi = weight[i];
        float normed = (xi - mean) * inv_std;
        float dxi = gi * wi * inv_std + g_mean / (float)cols
            + g_var * 2.0f * (xi - mean) / (float)cols;
        dx[row * cols + i] = dxi;
        atomicAdd(&dw[i], gi * normed);
        atomicAdd(&db[i], gi);
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let block = cols.next_power_of_two().min(256).max(32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: (block * 8) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut dx); b.arg(&mut dw); b.arg(&mut db);
            b.arg(input.buffer()); b.arg(weight.buffer()); b.arg(grad.buffer());
            b.arg(&rows); b.arg(&cols); b.arg(&eps);
            b.launch(cfg).map_err(|e| format!("ln_bwd launch: {e}"))?;
        }
        Ok((CudaTensor { shape: shape.clone(), buffer: dx, device_id: self.device_id },
            CudaTensor { shape: vec![cols], buffer: dw, device_id: self.device_id },
            CudaTensor { shape: vec![cols], buffer: db, device_id: self.device_id }))
    }

    // ── Cross Entropy ───────────────────────────────────────────────

    pub fn cross_entropy(&self, logits: &CudaTensor, targets: &CudaTensor) -> Result<CudaTensor, String> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(format!("cross_entropy: expected 2D logits, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = self.stream.alloc_zeros::<f32>(1).map_err(|e| format!("ce scratch: {e}"))?;

        let kernel_name = format!("cross_entropy_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ loss,
    const float* __restrict__ logits, const float* __restrict__ targets,
    size_t rows, size_t cols) {{
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    float total_loss = 0.0f;
    for (unsigned int row = 0; row < rows; row++) {{
        if (tid == 0) {{
            float max_val = -INFINITY;
            for (unsigned int j = 0; j < cols; j++) {{
                max_val = fmaxf(max_val, logits[row * cols + j]);
            }}
            float sum_exp = 0.0f;
            for (unsigned int j = 0; j < cols; j++) {{
                sum_exp += expf(logits[row * cols + j] - max_val);
            }}
            int target = (int)targets[row];
            float log_softmax = logits[row * cols + target] - max_val - logf(sum_exp);
            total_loss += -log_softmax;
        }}
    }}
    if (tid == 0) loss[0] = total_loss / (float)rows;
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(logits.buffer()); b.arg(targets.buffer());
            b.arg(&rows); b.arg(&cols);
            b.launch(cfg).map_err(|e| format!("cross_entropy launch: {e}"))?;
        }
        Ok(CudaTensor { shape: vec![1], buffer: out, device_id: self.device_id })
    }

    pub fn cross_entropy_backward(&self, softmax: &CudaTensor, grad: &CudaTensor, targets: &CudaTensor) -> Result<CudaTensor, String> {
        let shape = softmax.shape();
        let rows = shape[0];
        let cols = shape[1];
        let numel = softmax.numel();
        let mut dlogits = self.scratch_f32(numel).map_err(|e| format!("ce_bwd scratch: {e}"))?;

        let kernel_name = format!("ce_bwd_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ dlogits,
    const float* __restrict__ softmax, const float* __restrict__ grad,
    const float* __restrict__ targets, size_t rows, size_t cols) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * cols;
    if (i >= total) return;
    unsigned int row = i / cols;
    unsigned int col = i % cols;
    float g = grad[row];  // scalar gradient from upstream
    int target = (int)targets[row];
    float soft = softmax[i];
    float diag = (col == (unsigned int)target) ? 1.0f : 0.0f;
    dlogits[i] = g * (soft - diag) / (float)rows;
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut dlogits); b.arg(softmax.buffer()); b.arg(grad.buffer()); b.arg(targets.buffer());
            b.arg(&rows); b.arg(&cols);
            b.launch(cfg).map_err(|e| format!("ce_bwd launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: dlogits, device_id: self.device_id })
    }

    // ── Embedding ───────────────────────────────────────────────────

    pub fn embedding(&self, ids: &CudaTensor, weight: &CudaTensor) -> Result<CudaTensor, String> {
        let ids_shape = ids.shape();
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(format!("embedding: weight must be 2D, got {}D", w_shape.len()));
        }
        let vocab = w_shape[0];
        let dim = w_shape[1];
        let num_ids = ids.numel();
        let out_shape: Vec<usize> = ids_shape.iter().chain(std::iter::once(&dim)).copied().collect();
        let mut out = self.scratch_f32(num_ids * dim).map_err(|e| format!("embedding scratch: {e}"))?;

        let kernel_name = format!("embedding_f32_{}", dim);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ ids_f, const float* __restrict__ weight,
    size_t num_ids, size_t vocab, size_t dim) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_ids * dim;
    if (i >= total) return;
    unsigned int id_pos = i / dim;
    unsigned int d = i % dim;
    int token_id = (int)ids_f[id_pos];
    if (token_id >= 0 && token_id < (int)vocab) {{
        out[i] = weight[token_id * dim + d];
    }} else {{
        out[i] = 0.0f;
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let total = num_ids * dim;
        let cfg = LaunchConfig { grid_dim: (((total as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(ids.buffer()); b.arg(weight.buffer());
            b.arg(&num_ids); b.arg(&vocab); b.arg(&dim);
            b.launch(cfg).map_err(|e| format!("embedding launch: {e}"))?;
        }
        Ok(CudaTensor { shape: out_shape, buffer: out, device_id: self.device_id })
    }

    pub fn embedding_backward(&self, ids: &CudaTensor, grad: &CudaTensor, vocab_size: usize) -> Result<CudaTensor, String> {
        let dim = if grad.shape().len() >= 2 { grad.shape()[grad.ndim() - 1] } else { 0 };
        let num_ids = ids.numel();
        let mut dw = self.scratch_f32(vocab_size * dim).map_err(|e| format!("embedding_bwd scratch: {e}"))?;
        self.fill_zero(&CudaTensor { shape: vec![vocab_size * dim], buffer: dw.clone(), device_id: self.device_id })?;

        let grad_shape = grad.shape();
        let grad_flat = if grad_shape.len() > 2 {
            let flat: Vec<usize> = vec![num_ids, dim];
            grad.reshape(flat)?
        } else {
            grad.clone()
        };

        let func = compile_simple!(self, "embedding_bwd_f32", r#"
extern "C" __global__ void embedding_bwd_f32(float* __restrict__ dw,
    const float* __restrict__ ids_f, const float* __restrict__ grad,
    size_t num_ids, size_t dim) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_ids * dim;
    if (i >= total) return;
    unsigned int id_pos = i / dim;
    unsigned int d = i % dim;
    int token_id = (int)ids_f[id_pos];
    atomicAdd(&dw[token_id * dim + d], grad[i]);
}"#)?;
        let total = num_ids * dim;
        let cfg = LaunchConfig { grid_dim: (((total as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut dw); b.arg(ids.buffer()); b.arg(grad_flat.buffer());
            b.arg(&num_ids); b.arg(&dim);
            b.launch(cfg).map_err(|e| format!("embedding_bwd launch: {e}"))?;
        }
        Ok(CudaTensor { shape: vec![vocab_size, dim], buffer: dw, device_id: self.device_id })
    }

    // ── Causal Softmax ──────────────────────────────────────────────

    pub fn causal_softmax(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(format!("causal_softmax: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = input.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("causal_softmax scratch: {e}"))?;

        let kernel_name = format!("causal_softmax_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, size_t rows, size_t cols) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    float max_val = -INFINITY;
    for (unsigned int j = tid; j <= row && j < cols; j += n) {{
        max_val = fmaxf(max_val, a[row * cols + j]);
    }}
    extern __shared__ float shared[];
    shared[tid] = max_val;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }}
    float global_max = shared[0];
    __syncthreads();

    float sum_exp = 0.0f;
    for (unsigned int j = tid; j <= row && j < cols; j += n) {{
        float e = expf(a[row * cols + j] - global_max);
        out[row * cols + j] = e;
        sum_exp += e;
    }}
    // Zero out positions beyond row
    for (unsigned int j = tid + row + 1; j < cols; j += n) {{
        out[row * cols + j] = 0.0f;
    }}
    shared[tid] = sum_exp;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }}
    float global_sum = shared[0];
    __syncthreads();

    float inv_sum = global_sum > 0.0f ? 1.0f / global_sum : 0.0f;
    for (unsigned int j = tid; j <= row && j < cols; j += n) {{
        out[row * cols + j] *= inv_sum;
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let block = cols.next_power_of_two().min(256).max(32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: (block * 4) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(input.buffer()); b.arg(&rows); b.arg(&cols);
            b.launch(cfg).map_err(|e| format!("causal_softmax launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    // ── Rotary Embedding ────────────────────────────────────────────

    pub fn rotary_embedding(&self, x: &CudaTensor, cos: &CudaTensor, sin: &CudaTensor, head_dim: u32) -> Result<CudaTensor, String> {
        let shape = x.shape();
        let numel = x.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("rotary scratch: {e}"))?;

        let func = compile_simple!(self, "rotary_embedding_f32", r#"
extern "C" __global__ void rotary_embedding_f32(float* __restrict__ out,
    const float* __restrict__ x, const float* __restrict__ cos_t,
    const float* __restrict__ sin_t, size_t numel, unsigned int head_dim) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    unsigned int pos = (i / head_dim);
    unsigned int d = i % head_dim;
    unsigned int half = head_dim / 2;
    float cos_val = cos_t[pos * half + (d % half)];
    float sin_val = sin_t[pos * half + (d % half)];
    if (d < half) {
        float x_pair = x[i + half];
        out[i] = x[i] * cos_val - x_pair * sin_val;
    } else {
        float x_pair = x[i - half];
        out[i] = x[i] * cos_val + x_pair * sin_val;
    }
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(x.buffer()); b.arg(cos.buffer()); b.arg(sin.buffer());
            b.arg(&numel); b.arg(&head_dim);
            b.launch(cfg).map_err(|e| format!("rotary launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    // ── Repeat Heads (GQA) ──────────────────────────────────────────

    pub fn repeat_heads(&self, src: &CudaTensor, kv_heads: u32, q_heads: u32, dim: u32) -> Result<CudaTensor, String> {
        let shape = src.shape();
        if shape.len() != 4 {
            return Err(format!("repeat_heads: expected 4D, got {}D", shape.len()));
        }
        let batch = shape[0];
        let seq = shape[2];
        let numel = batch as u32 * q_heads * seq as u32 * dim;
        let out_shape = vec![batch, q_heads as usize, seq, dim as usize];
        let mut out = self.scratch_f32(numel as usize).map_err(|e| format!("repeat_heads scratch: {e}"))?;

        let func = compile_simple!(self, "repeat_heads_f32", r#"
extern "C" __global__ void repeat_heads_f32(float* __restrict__ out,
    const float* __restrict__ src, size_t batch, unsigned int kv_heads,
    unsigned int q_heads, unsigned int seq, unsigned int dim) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * q_heads * seq * dim;
    if (i >= total) return;
    unsigned int d = i % dim;
    unsigned int s = (i / dim) % seq;
    unsigned int h = (i / (seq * dim)) % q_heads;
    unsigned int b = i / (q_heads * seq * dim);
    unsigned int src_h = h % kv_heads;
    size_t src_idx = ((b * kv_heads + src_h) * seq + s) * dim + d;
    out[i] = src[src_idx];
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(src.buffer()); b.arg(&batch); b.arg(&kv_heads);
            b.arg(&q_heads); b.arg(&seq); b.arg(&dim);
            b.launch(cfg).map_err(|e| format!("repeat_heads launch: {e}"))?;
        }
        Ok(CudaTensor { shape: out_shape, buffer: out, device_id: self.device_id })
    }

    // ── Sampling ops ────────────────────────────────────────────────

    pub fn temperature_scale(&self, logits: &CudaTensor, temperature: f32) -> Result<CudaTensor, String> {
        if temperature <= 0.0 || temperature.abs() < 1e-8 {
            return Ok(logits.clone());
        }
        self.elementwise_unary_scalar_impl(logits, "div", "a[i] / scalar", temperature)
    }

    pub fn top_k_mask(&self, logits: &CudaTensor, k: u32) -> Result<CudaTensor, String> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(format!("top_k: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = logits.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("top_k scratch: {e}"))?;

        let kernel_name = format!("top_k_mask_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ logits, size_t rows, size_t cols, unsigned int k) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    // Copy to local
    extern __shared__ float shared[];
    for (unsigned int i = tid; i < cols; i += n) {{
        shared[i] = logits[row * cols + i];
    }}
    __syncthreads();

    // Find k-th largest value (simple iterative selection)
    float kth_val = -INFINITY;
    for (unsigned int ki = 0; ki < k && ki < cols; ki++) {{
        float max_v = -INFINITY;
        unsigned int max_idx = 0;
        for (unsigned int i = 0; i < cols; i++) {{
            if (shared[i] > max_v) {{ max_v = shared[i]; max_idx = i; }}
        }}
        if (ki == k - 1) kth_val = max_v;
        shared[max_idx] = -INFINITY;
    }}

    for (unsigned int i = tid; i < cols; i += n) {{
        out[row * cols + i] = (logits[row * cols + i] >= kth_val) ? logits[row * cols + i] : -INFINITY;
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (cols.min(256).max(32) as u32, 1, 1), shared_mem_bytes: (cols * 4) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(logits.buffer()); b.arg(&rows); b.arg(&cols); b.arg(&k);
            b.launch(cfg).map_err(|e| format!("top_k launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    pub fn top_p_mask(&self, logits: &CudaTensor, p: f32) -> Result<CudaTensor, String> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(format!("top_p: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = logits.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("top_p scratch: {e}"))?;

        let kernel_name = format!("top_p_mask_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ logits, size_t rows, size_t cols, float p) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    extern __shared__ float shared[];
    for (unsigned int i = tid; i < cols; i += n) {{
        shared[i] = logits[row * cols + i];
    }}
    __syncthreads();

    // Sort descending (simple bubble sort for shared memory)
    for (unsigned int i = 0; i < cols - 1; i++) {{
        for (unsigned int j = 0; j < cols - i - 1; j++) {{
            if (shared[j] < shared[j + 1]) {{
                float tmp = shared[j];
                shared[j] = shared[j + 1];
                shared[j + 1] = tmp;
            }}
        }}
    }}

    // Softmax the sorted values
    float max_v = shared[0];
    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < cols; i++) {{
        sum_exp += expf(shared[i] - max_v);
    }}
    float cum = 0.0f;
    float threshold = -INFINITY;
    for (unsigned int i = 0; i < cols; i++) {{
        float prob = expf(shared[i] - max_v) / sum_exp;
        cum += prob;
        if (cum > p) {{ threshold = shared[i]; break; }}
    }}

    for (unsigned int i = tid; i < cols; i += n) {{
        out[row * cols + i] = (logits[row * cols + i] >= threshold) ? logits[row * cols + i] : -INFINITY;
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (cols.min(256).max(32) as u32, 1, 1), shared_mem_bytes: (cols * 4) as u32 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(logits.buffer()); b.arg(&rows); b.arg(&cols); b.arg(&p);
            b.launch(cfg).map_err(|e| format!("top_p launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    pub fn multinomial_sample(&self, logits: &CudaTensor, seed: u64) -> Result<CudaTensor, String> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(format!("multinomial: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = self.stream.alloc_zeros::<f32>(rows).map_err(|e| format!("multinomial scratch: {e}"))?;

        let kernel_name = format!("multinomial_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ logits, size_t rows, size_t cols, unsigned long long seed) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int n = blockDim.x;

    extern __shared__ float shared[];

    // Find max
    float max_v = -INFINITY;
    for (unsigned int i = tid; i < cols; i += n) {{
        max_v = fmaxf(max_v, logits[row * cols + i]);
    }}
    shared[tid] = max_v;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }}
    float gmax = shared[0];
    __syncthreads();

    // Compute softmax denominator
    float sum_exp = 0.0f;
    for (unsigned int i = tid; i < cols; i += n) {{
        sum_exp += expf(logits[row * cols + i] - gmax);
    }}
    shared[tid] = sum_exp;
    __syncthreads();
    for (unsigned int s = n / 2; s > 0; s >>= 1) {{
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }}
    float total = shared[0];
    __syncthreads();

    // CDF walk with PRNG
    if (tid == 0) {{
        unsigned long long rng = seed + row * 6364136223846793005ULL;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float r = (float)(rng >> 33) / (float)(1UL << 31);
        float cum = 0.0f;
        for (unsigned int i = 0; i < cols; i++) {{
            cum += expf(logits[row * cols + i] - gmax) / total;
            if (r <= cum || i == cols - 1) {{ out[row] = (float)i; break; }}
        }}
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (cols.min(256).max(32) as u32, 1, 1), shared_mem_bytes: (cols.min(256).max(32) as u32 * 4) };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(logits.buffer()); b.arg(&rows); b.arg(&cols); b.arg(&seed);
            b.launch(cfg).map_err(|e| format!("multinomial launch: {e}"))?;
        }
        Ok(CudaTensor { shape: vec![rows], buffer: out, device_id: self.device_id })
    }

    pub fn dropout_mask(&self, mask: &CudaTensor, rate: f32, scale: f32, seed: u32) -> Result<(), String> {
        let numel = mask.numel();
        let func = compile_simple!(self, "dropout_mask_f32", r#"
extern "C" __global__ void dropout_mask_f32(float* __restrict__ mask,
    float rate, float scale, unsigned int seed, size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    unsigned int rng = seed + i * 1103515245u;
    rng = rng * 1103515245u + 12345u;
    float r = (float)(rng & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    mask[i] = (r >= rate) ? scale : 0.0f;
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mask.buffer); b.arg(&rate); b.arg(&scale); b.arg(&seed); b.arg(&numel);
            b.launch(cfg).map_err(|e| format!("dropout_mask launch: {e}"))?;
        }
        Ok(())
    }

    // ── Gradient ops ────────────────────────────────────────────────

    pub fn gradient_clip(&self, grads: &CudaTensor, max_norm: f32) -> Result<(), String> {
        let numel = grads.numel();
        // Compute L2 norm squared
        let mut norm_sq = self.stream.alloc_zeros::<f32>(1).map_err(|e| format!("clip norm scratch: {e}"))?;
        let func1 = compile_simple!(self, "grad_norm_sq_f32", r#"
extern "C" __global__ void grad_norm_sq_f32(float* __restrict__ out,
    const float* __restrict__ g, size_t numel) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < numel; i += blockDim.x) sum_sq += g[i] * g[i];
    shared[tid] = sum_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = shared[0];
}"#)?;
        let cfg1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
        unsafe {
            let mut b = self.stream.launch_builder(&func1);
            b.arg(&mut norm_sq); b.arg(grads.buffer()); b.arg(&numel);
            b.launch(cfg1).map_err(|e| format!("norm_sq launch: {e}"))?;
        }

        let norm_val: f32 = self.stream.clone_dtoh(&norm_sq).map_err(|e| format!("clip norm readback: {e}"))?.first().copied().unwrap_or(0.0);
        let norm_val = norm_val.sqrt();
        if norm_val <= max_norm || norm_val.abs() < 1e-8 { return Ok(()); }
        let scale = max_norm / norm_val;
        self.scale_inplace(grads, scale)
    }

    pub fn gradient_allreduce(&self, grad_buffers: &CudaTensor, out: &CudaTensor, num_replicas: u32) -> Result<(), String> {
        let numel = out.numel();
        let inv = 1.0f32 / num_replicas as f32;
        let func = compile_simple!(self, "grad_allreduce_f32", r#"
extern "C" __global__ void grad_allreduce_f32(float* __restrict__ out,
    const float* __restrict__ in, float inv_n, size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] += in[i] * inv_n;
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&out.buffer); b.arg(grad_buffers.buffer()); b.arg(&inv); b.arg(&numel);
            b.launch(cfg).map_err(|e| format!("grad_allreduce launch: {e}"))?;
        }
        Ok(())
    }

    pub fn adam_step(&self, param: &CudaTensor, grad: &CudaTensor, m: &CudaTensor, v: &CudaTensor,
        lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, step: u32) -> Result<(), String> {
        let numel = param.numel();
        let func = compile_simple!(self, "adam_step_f32", r#"
extern "C" __global__ void adam_step_f32(
    float* __restrict__ param, const float* __restrict__ grad,
    float* __restrict__ m, float* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float wd, unsigned int step,
    size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float g = grad[i];
    float m_i = beta1 * m[i] + (1.0f - beta1) * g;
    float v_i = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_i;
    v[i] = v_i;
    float m_hat = m_i / (1.0f - powf(beta1, (float)step));
    float v_hat = v_i / (1.0f - powf(beta2, (float)step));
    float update = lr * m_hat / (sqrtf(v_hat) + eps);
    param[i] = param[i] - update + wd * param[i];
}"#)?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&param.buffer); b.arg(grad.buffer());
            b.arg(&m.buffer); b.arg(&v.buffer);
            b.arg(&lr); b.arg(&beta1); b.arg(&beta2); b.arg(&eps); b.arg(&weight_decay); b.arg(&step);
            b.arg(&numel);
            b.launch(cfg).map_err(|e| format!("adam_step launch: {e}"))?;
        }
        Ok(())
    }

    // ── Fused ops ───────────────────────────────────────────────────

    pub fn fused_matmul_bias(&self, a: &CudaTensor, b: &CudaTensor, bias: &CudaTensor, activation: &str) -> Result<CudaTensor, String> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let mut out = self.scratch_f32(m * n).map_err(|e| format!("fused_mm_bias scratch: {e}"))?;

        let activation_expr = match activation {
            "gelu" => "0.5f * val * (1.0f + tanhf(0.79788456f * (val + 0.044715f * val * val * val)))",
            "relu" => "fmaxf(0.0f, val)",
            "silu" => "val / (1.0f + expf(-val))",
            _ => "val",
        };

        let kernel_name = format!("fused_mm_bias_{}", activation);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ bias, size_t M, size_t N, size_t K) {{
    unsigned int row = blockIdx.x;
    unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (size_t i = 0; i < K; i++) {{
        sum += a[row * K + i] * b[i * N + col];
    }}
    float val = sum + bias[col];
    out[row * N + col] = {activation_expr};
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (m as u32, ((n as u32) + 15) / 16, 1), block_dim: (16, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(&mut out); builder.arg(a.buffer()); builder.arg(b.buffer()); builder.arg(bias.buffer());
            builder.arg(&m); builder.arg(&n); builder.arg(&k);
            builder.launch(cfg).map_err(|e| format!("fused_mm_bias launch: {e}"))?;
        }
        Ok(CudaTensor { shape: vec![m, n], buffer: out, device_id: self.device_id })
    }

    pub fn fused_online_softmax(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        // Single-pass online softmax (Welford-style max+sum)
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(format!("online_softmax: expected 2D, got {}D", shape.len()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let numel = input.numel();
        let mut out = self.scratch_f32(numel).map_err(|e| format!("online_softmax scratch: {e}"))?;

        let kernel_name = format!("online_softmax_f32_{}", cols);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, size_t rows, size_t cols) {{
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float m = -INFINITY, d = 0.0f;
    for (unsigned int j = 0; j < cols; j++) {{
        float x = a[row * cols + j];
        float m_new = fmaxf(m, x);
        float old_scale = expf(m - m_new);
        d = old_scale * d + expf(x - m_new);
        m = m_new;
    }}
    float inv_d = 1.0f / d;
    unsigned int tid = threadIdx.x;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {{
        out[row * cols + j] = expf(a[row * cols + j] - m) * inv_d;
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (cols.min(256).max(32) as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(input.buffer()); b.arg(&rows); b.arg(&cols);
            b.launch(cfg).map_err(|e| format!("online_softmax launch: {e}"))?;
        }
        Ok(CudaTensor { shape: shape.clone(), buffer: out, device_id: self.device_id })
    }

    // ── Fused Attention Backward ────────────────────────────────────

    pub fn fused_attention_backward(&self, q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, grad: &CudaTensor,
        scale: f32, causal: bool) -> Result<(CudaTensor, CudaTensor, CudaTensor), String> {
        let q_shape = q.shape();
        let batch = q_shape[0];
        let heads = q_shape[1];
        let seq_len = q_shape[2];
        let dim = q_shape[3];
        let numel = q.numel();
        let mut dq = self.scratch_f32(numel).map_err(|e| format!("attn_bwd dq scratch: {e}"))?;
        let mut dk = self.scratch_f32(numel).map_err(|e| format!("attn_bwd dk scratch: {e}"))?;
        let mut dv = self.scratch_f32(numel).map_err(|e| format!("attn_bwd dv scratch: {e}"))?;
        self.fill_zero(&CudaTensor { shape: q_shape.clone(), buffer: dq.clone(), device_id: self.device_id })?;
        self.fill_zero(&CudaTensor { shape: q_shape.clone(), buffer: dk.clone(), device_id: self.device_id })?;
        self.fill_zero(&CudaTensor { shape: q_shape.clone(), buffer: dv.clone(), device_id: self.device_id })?;

        let causal_u32: u32 = if causal { 1 } else { 0 };
        let block_size: u32 = 256;

        let kernel_name = format!("attn_bwd_f32_{}", dim);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ grad,
    size_t batch, size_t heads, size_t seq_len, size_t dim, float scale,
    unsigned int causal) {{

    unsigned int wg_flat = blockIdx.x;
    unsigned int q_pos = wg_flat % seq_len;
    unsigned int head = (wg_flat / seq_len) % heads;
    unsigned int b = wg_flat / (seq_len * heads);
    if (b >= batch) return;

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* score_tile = shared + dim;

    unsigned int tid = threadIdx.x;
    size_t head_stride = seq_len * dim;
    size_t batch_stride = heads * head_stride;
    size_t q_off = b * batch_stride + head * head_stride + q_pos * dim;
    size_t kv_base = b * batch_stride + head * head_stride;
    float scale_inv = 1.0f / scale;

    // Load Q row
    if (tid < dim) q_shared[tid] = q[q_off + tid];
    __syncthreads();

    // Precompute scores and softmax for all positions
    float* scores = shared + 2 * dim;
    for (unsigned int kv_pos = 0; kv_pos < seq_len; kv_pos++) {{
        if (causal && kv_pos > q_pos) {{ scores[kv_pos] = -INFINITY; continue; }}
        float s = 0.0f;
        for (unsigned int d_i = tid; d_i < dim; d_i += blockDim.x) {{
            s += q_shared[d_i] * k[kv_base + kv_pos * dim + d_i];
        }}
        // Warp reduction
        for (unsigned int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFF, s, off);
        if (tid == 0) scores[kv_pos] = s * scale;
        __syncthreads();
    }}

    // Online softmax
    float m = -INFINITY, d = 0.0f;
    for (unsigned int kv_pos = 0; kv_pos < seq_len; kv_pos++) {{
        float s = scores[kv_pos];
        if (s <= -INFINITY / 2) continue;
        float m_new = fmaxf(m, s);
        float old_scale = expf(m - m_new);
        d = old_scale * d + expf(s - m_new);
        m = m_new;
    }}

    // dV: grad * softmax_weight
    for (unsigned int kv_pos = tid; kv_pos < seq_len; kv_pos += blockDim.x) {{
        float attn = expf(scores[kv_pos] - m) / d;
        for (unsigned int d_i = 0; d_i < dim; d_i++) {{
            float g_val = grad[(b * heads + head) * head_stride + q_pos * dim + d_i];
            atomicAdd(&dv[kv_base + kv_pos * dim + d_i], attn * g_val * scale_inv);
        }}
    }}

    // dK: grad * softmax * q_weighted
    for (unsigned int kv_pos = tid; kv_pos < seq_len; kv_pos += blockDim.x) {{
        float attn = expf(scores[kv_pos] - m) / d;
        float g_dot_q = 0.0f;
        for (unsigned int d_i = 0; d_i < dim; d_i++) {{
            unsigned int g_off = (b * heads + head) * head_stride + q_pos * dim + d_i;
            g_dot_q += grad[g_off] * q_shared[d_i];
        }}
        // Warp reduction for g_dot_q
        for (unsigned int off = 16; off > 0; off >>= 1)
            g_dot_q += __shfl_xor_sync(0xFFFFFFFF, g_dot_q, off);
        if (tid == 0) {{
            for (unsigned int d_i = 0; d_i < dim; d_i++) {{
                float g_val = grad[(b * heads + head) * head_stride + q_pos * dim + d_i];
                atomicAdd(&dk[kv_base + kv_pos * dim + d_i],
                    (g_val - attn * g_dot_q * scale_inv * q_shared[d_i]) * scale_inv);
            }}
        }}
        __syncthreads();
    }}

    // dQ
    for (unsigned int d_i = tid; d_i < dim; d_i += blockDim.x) {{
        float sum_gk = 0.0f;
        for (unsigned int kv_pos = 0; kv_pos < seq_len; kv_pos++) {{
            if (causal && kv_pos > q_pos) continue;
            float attn = expf(scores[kv_pos] - m) / d;
            float gv = grad[(b * heads + head) * head_stride + q_pos * dim + d_i];
            float k_val = k[kv_base + kv_pos * dim + d_i];
            float v_val = v[kv_base + kv_pos * dim + d_i];
            sum_gk += attn * k_val * (gv - attn * v_val * scale_inv);
        }}
        dq[q_off + d_i] = sum_gk * scale;
    }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let total_wgs = (batch * heads * seq_len) as u32;
        let shared_size = (2 * dim + seq_len) as u32;
        let cfg = LaunchConfig { grid_dim: (total_wgs, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: shared_size * 4 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut dq); b.arg(&mut dk); b.arg(&mut dv);
            b.arg(q.buffer()); b.arg(k.buffer()); b.arg(v.buffer()); b.arg(grad.buffer());
            b.arg(&batch); b.arg(&heads); b.arg(&seq_len); b.arg(&dim); b.arg(&scale); b.arg(&causal_u32);
            b.launch(cfg).map_err(|e| format!("attn_bwd launch: {e}"))?;
        }
        Ok((CudaTensor { shape: q_shape.clone(), buffer: dq, device_id: self.device_id },
            CudaTensor { shape: q_shape.clone(), buffer: dk, device_id: self.device_id },
            CudaTensor { shape: q_shape.clone(), buffer: dv, device_id: self.device_id }))
    }

    // ── Helper: scalar unary (a[i] OP scalar) ───────────────────────

    fn elementwise_unary_scalar_impl(&self, a: &CudaTensor, op: &str, expr: &str, scalar: f32) -> Result<CudaTensor, String> {
        let numel = a.numel();
        let kernel_name = format!("elem_scalar_{}", op);
        let source = format!(r#"
extern "C" __global__ void {kernel_name}(float* __restrict__ out,
    const float* __restrict__ a, float scalar, size_t numel) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {{ out[i] = {expr}; }}
}}
"#);
        let func = self.get_or_compile_kernel(&kernel_name, &source)?;
        let mut out = self.scratch_f32(numel).map_err(|e| format!("scalar {} scratch: {e}", op))?;
        let cfg = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut out); b.arg(a.buffer()); b.arg(&scalar); b.arg(&numel);
            b.launch(cfg).map_err(|e| format!("scalar {} launch: {e}", op))?;
        }
        Ok(CudaTensor { shape: a.shape.clone(), buffer: out, device_id: self.device_id })
    }

    // ── Misc ────────────────────────────────────────────────────────

    pub fn leaky_relu(&self, input: &CudaTensor, negative_slope: f32) -> Result<CudaTensor, String> {
        self.elementwise_unary_scalar_impl(input, "leaky_relu",
            "a[i] >= 0.0f ? a[i] : a[i] * scalar", negative_slope)
    }

    pub fn swiglu(&self, input: &CudaTensor) -> Result<CudaTensor, String> {
        self.elementwise_unary(input, "swiglu", "a[i] / (1.0f + expf(-a[i])) * a[i]")
    }

    pub fn binary_cross_entropy(&self, input: &CudaTensor, target: &CudaTensor) -> Result<CudaTensor, String> {
        let numel = input.numel();
        let mut scratch = self.scratch_f32(numel).map_err(|e| format!("bce scratch: {e}"))?;
        let func = compile_simple!(self, "binary_ce_f32", r#"
extern "C" __global__ void binary_ce_f32(float* __restrict__ out,
    const float* __restrict__ input, const float* __restrict__ target, size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float p = fminf(fmaxf(input[i], 1e-7f), 1.0f - 1e-7f);
    float t = target[i];
    out[i] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
}"#)?;
        let mut out = self.stream.alloc_zeros::<f32>(1).map_err(|e| format!("bce out scratch: {e}"))?;
        // Compute per-element BCE first
        let cfg1 = LaunchConfig { grid_dim: (((numel as u32 + 255) / 256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut scratch); b.arg(input.buffer()); b.arg(target.buffer()); b.arg(&numel);
            b.launch(cfg1).map_err(|e| format!("bce elem launch: {e}"))?;
        }
        // Then sum
        let sum_func = compile_simple!(self, "bce_sum_f32", r#"
extern "C" __global__ void bce_sum_f32(float* __restrict__ out,
    const float* __restrict__ a, size_t numel) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    float val = 0.0f;
    for (unsigned int i = tid; i < numel; i += blockDim.x) val += a[i];
    shared[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = shared[0];
}"#)?;
        let cfg2 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
        unsafe {
            let mut b = self.stream.launch_builder(&sum_func);
            b.arg(&mut out); b.arg(&scratch); b.arg(&numel);
            b.launch(cfg2).map_err(|e| format!("bce sum launch: {e}"))?;
        }
        Ok(CudaTensor { shape: vec![1], buffer: out, device_id: self.device_id })
    }

    // ── Causal attention (thin wrapper calling fused_attention) ─────

    pub fn causal_attention(&self, q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: f32) -> Result<CudaTensor, String> {
        self.fused_attention(q, k, v, scale, true)
    }
}
