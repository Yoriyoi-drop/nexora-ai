//! Hybrid Deferred Execution Engine
//!
//! GNAC menggunakan sistem eksekusi hybrid:
//! - Eager Execution saat editing visual (responsif)
//! - Compiled Graph Execution saat training final (optimal)
//!
//! Pipeline: User-built graph → IR → Graph Optimizer → Backend Runtime

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
