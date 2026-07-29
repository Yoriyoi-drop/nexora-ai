//! Hybrid Deferred Execution Engine
//!
//! GNAC menggunakan sistem eksekusi hybrid:
//! - Eager Execution saat editing visual (responsif)
//! - Compiled Graph Execution saat training final (optimal)
//!
//! Pipeline: User-built graph → IR → Graph Optimizer → Backend Runtime

pub mod backend;
pub mod compiled;
pub mod eager;
pub mod ir;
pub mod optimizer;

pub use compiled::*;
pub use eager::*;
pub use ir::*;
pub use optimizer::*;

/// Backend target untuk eksekusi.
///
/// # Realita
/// - **CPU** — implementasi penuh, 16 op (MatMul, Add, Mul, Relu, Gelu, Softmax, dll).
/// - **WGPU** — GPU compute via wgpu-based `GpuContext`.
///   Support: MatMul, Add, Mul, Relu, Gelu, Softmax, Transpose, LayerNorm.
///   Op lain fallback ke CPU (download→compute→upload).
/// - **CUDA** — Native CUDA via `nexora-autograd`. Requires `cuda` feature + CUDA toolkit.
///   Support: semua op WGPU + FlashAttention. Auto-fallback ke WGPU jika CUDA unavailable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionBackend {
    /// GPU execution via wgpu-based GpuContext.
    WGPU,
    /// Native CUDA execution via cudarc + cuBLAS + NVRTC.
    /// Auto-detected at runtime; falls back to WGPU if unavailable.
    CUDA,
    /// Implementasi CPU penuh.
    CPU,
}

impl ExecutionBackend {
    pub fn name(&self) -> &str {
        match self {
            ExecutionBackend::WGPU => "WGPU (wgpu-based GPU)",
            ExecutionBackend::CUDA => "CUDA (native NVIDIA GPU)",
            ExecutionBackend::CPU => "CPU",
        }
    }

    /// Auto-detect best available backend at runtime.
    /// Preference: CUDA > WGPU > CPU
    pub fn auto_detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            if let Ok(ctx) = crate::autograd::gpu::GpuContext::global() {
                if ctx.backend() == crate::autograd::gpu::GpuBackend::Cuda {
                    return ExecutionBackend::CUDA;
                }
            }
        }
        #[cfg(feature = "gpu")]
        {
            if crate::autograd::gpu::GpuContext::global().is_ok() {
                return ExecutionBackend::WGPU;
            }
        }
        ExecutionBackend::CPU
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_backend_name() {
        assert_eq!(ExecutionBackend::WGPU.name(), "WGPU (wgpu-based GPU)");
        assert_eq!(ExecutionBackend::CUDA.name(), "CUDA (native NVIDIA GPU)");
        assert_eq!(ExecutionBackend::CPU.name(), "CPU");
    }

    #[test]
    fn test_execution_backend_auto_detect() {
        let backend = ExecutionBackend::auto_detect();
        // Should not panic; will return CPU if no GPU available
        assert!(matches!(backend, ExecutionBackend::CPU | ExecutionBackend::WGPU | ExecutionBackend::CUDA));
    }
}
