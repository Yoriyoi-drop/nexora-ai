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

/// Backend target untuk eksekusi
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionBackend {
    CUDA,
    Vulkan,
    TPU,
    WebGPU,
    CPU,
}

impl ExecutionBackend {
    pub fn name(&self) -> &str {
        match self {
            ExecutionBackend::CUDA => "CUDA",
            ExecutionBackend::Vulkan => "Vulkan",
            ExecutionBackend::TPU => "TPU",
            ExecutionBackend::WebGPU => "WebGPU",
            ExecutionBackend::CPU => "CPU",
        }
    }
}

/// Mode eksekusi
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    Eager,
    Compiled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_backend_name() {
        assert_eq!(ExecutionBackend::CUDA.name(), "CUDA");
        assert_eq!(ExecutionBackend::CPU.name(), "CPU");
    }

    #[test]
    fn test_execution_backend_debug() {
        let b = ExecutionBackend::Vulkan;
        assert!(!format!("{:?}", b).is_empty());
    }
}
