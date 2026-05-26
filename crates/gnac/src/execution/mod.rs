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
/// - **CPU** — satu-satunya backend yang berfungsi penuh.
/// - **CUDA** — menggunakan `GpuContext` (wgpu), bukan CUDA runtime sungguhan.
///   Tidak ada `cuda_runtime`/`cublas`/`cudnn` dependency.
/// - **Vulkan, TPU, WebGPU** — dideklarasikan untuk pengembangan masa depan,
///   tapi belum ada implementasi. Memilih backend ini akan return error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionBackend {
    /// GPU execution via wgpu-based GpuContext.
    /// BUKAN CUDA sungguhan — tidak pakai cuda_runtime/cublas/cudnn.
    /// Berfungsi untuk GPU compute, tapi mungkin lebih lambat dari CPU untuk beberapa op.
    CUDA,
    /// ❌ Belum diimplementasikan. Akan return error.
    #[deprecated(note = "Vulkan backend is not yet implemented. Use CPU or CUDA instead.")]
    Vulkan,
    /// ❌ Belum diimplementasikan. Akan return error.
    #[deprecated(note = "TPU backend is not yet implemented. Use CPU or CUDA instead.")]
    TPU,
    /// ❌ Belum diimplementasikan. Akan return error.
    #[deprecated(note = "WebGPU backend is not yet implemented. Use CPU or CUDA instead.")]
    WebGPU,
    /// Satu-satunya backend dengan implementasi penuh.
    CPU,
}

impl ExecutionBackend {
    pub fn name(&self) -> &str {
        match self {
            ExecutionBackend::CUDA => "CUDA (wgpu-based)",
            ExecutionBackend::Vulkan => "Vulkan (unimplemented)",
            ExecutionBackend::TPU => "TPU (unimplemented)",
            ExecutionBackend::WebGPU => "WebGPU (unimplemented)",
            ExecutionBackend::CPU => "CPU",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_backend_name() {
        assert_eq!(ExecutionBackend::CUDA.name(), "CUDA (wgpu-based)");
        assert_eq!(ExecutionBackend::CPU.name(), "CPU");
    }

    #[test]
    #[allow(deprecated)]
    fn test_execution_backend_debug() {
        let b = ExecutionBackend::Vulkan;
        assert!(!format!("{:?}", b).is_empty());
    }
}
