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
/// - **WGPU** — GPU compute via wgpu-based `GpuContext`. Bukan CUDA runtime.
///   Support: MatMul, Add, Mul, Relu, Gelu, Softmax, Transpose, LayerNorm.
///   Op lain fallback ke CPU (download→compute→upload).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionBackend {
    /// GPU execution via wgpu-based GpuContext.
    /// BUKAN CUDA — tidak pakai cuda_runtime/cublas/cudnn.
    /// Support GPU compute via WGSL shaders.
    WGPU,
    /// Implementasi CPU penuh.
    CPU,
}

impl ExecutionBackend {
    pub fn name(&self) -> &str {
        match self {
            ExecutionBackend::WGPU => "WGPU (wgpu-based GPU)",
            ExecutionBackend::CPU => "CPU",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_backend_name() {
        assert_eq!(ExecutionBackend::WGPU.name(), "WGPU (wgpu-based GPU)");
        assert_eq!(ExecutionBackend::CPU.name(), "CPU");
    }
}
