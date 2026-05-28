use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaDevice, CudaStream, CudaFunction, LaunchConfig};
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
        })
    }

    // ── CUDA kernel compilation ──────────────────────────────────────

    fn get_or_compile_kernel(&self, name: &str, source: &str) -> Result<CudaFunction, String> {
        if let Some(f) = self.kernels.lock().unwrap().get(name) {
            return Ok(f.clone());
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
            .unwrap()
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
            .stream
            .alloc_zeros::<f32>(result_len)
            .map_err(|e| format!("cuBLAS matmul alloc: {e}"))?;

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
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA elem {} alloc: {e}", op))?;

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
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA scalar {} alloc: {e}", op))?;

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
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA add_broadcast alloc: {e}"))?;
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
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA powf alloc: {e}"))?;

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
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA {} alloc: {e}", op))?;

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
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA transpose alloc: {e}"))?;
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
    for (unsigned int i = tid; i < cols; i += n) {
        out[row * cols + i] /= global_sum;
    }
}
"#,
        )?;

        let mut out = self
            .stream
            .alloc_zeros::<f32>(numel)
            .map_err(|e| format!("CUDA softmax alloc: {e}"))?;

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
}
