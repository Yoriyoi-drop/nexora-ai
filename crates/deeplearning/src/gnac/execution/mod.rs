//! Hybrid Deferred Execution Engine
//!
//! GNAC menggunakan sistem eksekusi hybrid:
//! - Eager Execution saat editing visual (responsif)
//! - Compiled Graph Execution saat training final (optimal)
//!
//! Pipeline: User-built graph → IR → Graph Optimizer → Backend Runtime

pub mod ir;
pub mod eager;
pub mod compiled;
pub mod optimizer;

pub use ir::*;
pub use eager::*;
pub use compiled::*;
pub use optimizer::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;

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
