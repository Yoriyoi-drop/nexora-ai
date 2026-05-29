use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;

use crate::gpu_caps::GpuCapabilities;

// ─── Detailed Timing ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct DetailedTiming {
    pub encode: std::time::Duration,
    pub submit: std::time::Duration,
    pub gpu_execute: std::time::Duration,
    pub map_async: std::time::Duration,
    pub readback: std::time::Duration,
    pub end_to_end: std::time::Duration,
}

impl DetailedTiming {
    pub fn display(&self) -> String {
        format!(
            "encode={:?} submit={:?} gpu={:?} map_async={:?} readback={:?} total={:?}",
            self.encode,
            self.submit,
            self.gpu_execute,
            self.map_async,
            self.readback,
            self.end_to_end
        )
    }
}

// ─── Errors ────────────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No GPU adapter found")]
    NoAdapter,
    #[error("GPU device error: {0}")]
    Device(String),
    #[error("Buffer creation failed: {0}")]
    Buffer(String),
    #[error("Compute shader error: {0}")]
    Shader(String),
    #[error("GPU not initialized. Call GpuContext::init() first")]
    NotInitialized,
    #[error("MatMul shape mismatch: a={0:?} b={1:?}")]
    MatMulShape(Vec<usize>, Vec<usize>),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    #[error("Pipeline compilation failed: {0}")]
    Pipeline(String),
    #[error("Element-wise shape mismatch: a={0:?} b={1:?}")]
    ElemShape(Vec<usize>, Vec<usize>),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("GPU operation timed out: {0}")]
    Timeout(String),
    #[error("Dtype mismatch: {0}")]
    Dtype(String),
    #[error("Conversion error: {0}")]
    Conversion(String),
    #[error("Lock error: {0}")]
    LockError(String),
    /// GPU device was lost or removed (e.g. TDR, driver crash, physical disconnect).
    #[error("Device lost: {0}")]
    DeviceLost(String),
    /// GPU out of memory (VRAM exhaustion).
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    /// Shader compilation failure.
    #[error("Shader compilation failed: {0}")]
    ShaderCompile(String),
    /// Watchdog timeout — GPU operation exceeded deadline.
    #[error("Watchdog timeout: {0}")]
    WatchdogTimeout(String),
}

impl GpuError {
    /// Categorize error for recovery dispatching.
    pub fn kind(&self) -> GpuErrorKind {
        match self {
            GpuError::DeviceLost(_) => GpuErrorKind::DeviceLost,
            GpuError::OutOfMemory(_) => GpuErrorKind::OutOfMemory,
            GpuError::ShaderCompile(_) | GpuError::Shader(_) | GpuError::Pipeline(_) => {
                GpuErrorKind::ShaderCompile
            }
            GpuError::WatchdogTimeout(_) | GpuError::Timeout(_) => GpuErrorKind::Timeout,
            GpuError::NoAdapter | GpuError::NotInitialized => GpuErrorKind::Fatal,
            GpuError::Buffer(_) => {
                // Buffer allocation failure could be OOM or device lost
                GpuErrorKind::Transient
            }
            _ => GpuErrorKind::Transient,
        }
    }

    /// Returns true if the error is recoverable (retry + fallback makes sense).
    pub fn is_recoverable(&self) -> bool {
        !matches!(self.kind(), GpuErrorKind::Fatal)
    }
}

/// High-level error category for recovery dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuErrorKind {
    /// Device lost — requires full context rebuild.
    DeviceLost,
    /// Out of memory — purge caches, retry with smaller allocations.
    OutOfMemory,
    /// Shader compile failure — fall back to CPU or skip pipeline.
    ShaderCompile,
    /// Operation timed out — retry with fresh encoder.
    Timeout,
    /// Transient error — safe to retry with backoff.
    Transient,
    /// Fatal error — no recovery possible, abort GPU path.
    Fatal,
}

// ─── GPU Backend Selection ─────────────────────────────────────────────────

/// The active GPU compute backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    /// WebGPU (wgpu) — default on all platforms.
    Wgpu,
    /// Native CUDA — requires `cuda` feature + NVIDIA GPU + CUDA toolkit.
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::Wgpu
    }
}

/// Counting semaphore for limiting concurrent GPU readback operations.
/// Prevents OOM when CPU submits readbacks faster than GPU processes them.
/// Default max permits: 16 concurrent readbacks.
pub(crate) struct ReadbackLimiter {
    max_permits: usize,
    available: Mutex<usize>,
    condvar: std::sync::Condvar,
}

impl ReadbackLimiter {
    pub fn new(max_permits: usize) -> Self {
        Self {
            max_permits,
            available: Mutex::new(max_permits),
            condvar: std::sync::Condvar::new(),
        }
    }

    /// Acquire a permit, blocking until one is available.
    /// Times out after `timeout` and returns false if no permit acquired.
    pub fn acquire(&self, timeout: Duration) -> bool {
        let mut guard = self.available.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = std::time::Instant::now() + timeout;
        while *guard == 0 {
            let remaining = deadline
                .checked_duration_since(std::time::Instant::now())
                .unwrap_or(Duration::ZERO);
            if remaining.is_zero() {
                return false;
            }
            let (new_guard, timeout_result) = self
                .condvar
                .wait_timeout(guard, remaining)
                .unwrap_or_else(|e| {
                    let (g, t) = e.into_inner();
                    (g, t)
                });
            guard = new_guard;
            if timeout_result.timed_out() {
                return false;
            }
        }
        *guard -= 1;
        true
    }

    /// Non-blocking try-acquire. Returns true if a permit was acquired.
    pub fn try_acquire(&self) -> bool {
        let mut avail = self.available.lock().unwrap_or_else(|e| e.into_inner());
        if *avail == 0 {
            return false;
        }
        *avail -= 1;
        true
    }

    /// Release a permit back to the pool, waking one waiter.
    pub fn release(&self) {
        let mut avail = self.available.lock().unwrap_or_else(|e| e.into_inner());
        *avail = (*avail + 1).min(self.max_permits);
        self.condvar.notify_one();
    }
}

/// Maximum concurrent GPU readback operations before backpressure kicks in.
pub(crate) const MAX_PENDING_READBACKS: usize = 16;

// ─── Singleton Context ─────────────────────────────────────────────────────────

pub(crate) static GPU_CTX: OnceCell<GpuContext> = OnceCell::new();

// ─── GpuContext ────────────────────────────────────────────────────────────────

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: GpuCapabilities,
    pub(crate) pipelines: HashMap<String, CompiledPipeline>,
    pub(crate) shader_cache: HashMap<u64, wgpu::ShaderModule>,
    pub(crate) bind_group_layout_cache: HashMap<u64, wgpu::BindGroupLayout>,
    /// Thread-safe bind group cache for &self access (used by dispatch methods)
    /// Uses `std::sync::Mutex` — all GPU dispatch methods are sync. Never used in async context.
    pub(crate) bind_group_cache_mutex: Mutex<HashMap<u64, wgpu::BindGroup>>,
    /// Thread-safe memory pool for &self access (used by dispatch methods)
    /// Uses `std::sync::Mutex` — all GPU dispatch methods are sync. Never used in async context.
    pub(crate) memory_pool: Mutex<crate::gpu_memory::GpuMemoryPool>,
    pub(crate) profiling_query_set: Option<wgpu::QuerySet>,
    pub(crate) query_pool_size: usize,
    pub(crate) query_index: std::sync::atomic::AtomicUsize,
    /// Reusable query resolve buffer to avoid alloc per profiled dispatch
    pub(crate) query_resolve_buf: Option<wgpu::Buffer>,
    /// Reusable readback buffer for timestamp queries
    pub(crate) query_readback_buf: Option<Mutex<wgpu::Buffer>>,
    // ── Reusable command encoder (Mutex-protected for &self access) ───
    // WARNING: This is a global encoder mutex which can cause contention under
    // heavy multi-threaded GPU dispatch. For high-throughput GPU workloads (>100 ops/sec),
    // use GpuCommandBatch (see gpu_batch.rs) which creates per-batch encoders,
    // avoiding this mutex entirely. The global encoder is kept for simple
    // single-threaded use and fallback paths only.
    pub(crate) current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
    pub(crate) ops_since_flush: std::sync::atomic::AtomicUsize,
    pub(crate) auto_flush_ops: usize,
    // ── Batch mode ───────────────────────────────────────────────────
    pub(crate) batch_depth: std::sync::atomic::AtomicUsize,
    // ── Persistent pipeline cache (disk-backed) ──────────────────────
    pub(crate) wgpu_cache: Option<wgpu::PipelineCache>,
    pub(crate) disk_cache: Option<crate::persistent_cache::PipelineDiskCache>,
    pub(crate) cache_key: u64,
    // ── Active backend ───────────────────────────────────────────────
    pub(crate) backend: GpuBackend,
    /// Limits concurrent GPU readback operations to prevent OOM.
    /// CPU-side backpressure: blocks readback_inner when too many pending.
    pub(crate) readback_limiter: ReadbackLimiter,
    /// Optional CUDA runtime — available only when `cuda` feature is enabled
    /// and an NVIDIA GPU with CUDA toolkit is detected at runtime.
    /// Uses `'static` lifetime because it borrows from the global singleton (`OnceCell<Arc<CudaRuntime>>`).
    #[cfg(feature = "cuda")]
    pub(crate) cuda: Option<&'static crate::gpu::cuda::CudaRuntime>,
}

impl GpuContext {
    /// Returns the active GPU compute backend.
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Returns a reference to the CUDA runtime if available.
    #[cfg(feature = "cuda")]
    pub fn cuda_runtime(&self) -> Option<&'static crate::gpu::cuda::CudaRuntime> {
        self.cuda
    }
}

pub(crate) struct CompiledPipeline {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU adapter info for display
#[derive(Debug)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub compute_backend: GpuBackend,
}

// ─── Element-wise op codes ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElemOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Neg = 4,
    Exp = 5,
    Ln = 6,
    Powf = 7,
    Sqrt = 8,
    Relu = 9,
    Gelu = 10,
    Sigmoid = 11,
    Tanh = 12,
    Silu = 13,
    LeakyRelu = 14,
    BinaryCrossEntropy = 15,
    LogSoftmax = 16,
    Swiglu = 17,
    /// Heaviside step: 1 if x > 0 else 0. Used in relu_backward.
    Step = 18,
}

// ─── Reduce op codes ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReduceOp {
    Sum = 0,
    Max = 1,
    Min = 2,
}

// ─── Binding helpers ────────────────────────────────────────────────────────────

pub(crate) fn storage_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_binding(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
