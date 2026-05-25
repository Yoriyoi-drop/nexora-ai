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

/// Hybrid execution (eager/compiled) — planned but not yet implemented.
///
/// Current status:
///   The `ExecutionMode` enum was removed because the eager/compiled dispatch
///   path was never wired to a backend. All execution currently uses the
///   `ExecutionBackend` directly without a mode layer.
///
/// What needs to be done:
///   1. Reintroduce `ExecutionMode` (eager | compiled) in this module.
///   2. Add a `mode` field to the execution pipeline config.
///   3. Wire eager mode to run ops immediately through the backend.
///   4. Wire compiled mode to build a fusion graph first (see `gnac::ir`),
///      then launch the compiled kernel via the backend.
///
/// Files to modify:
///   - `crates/gnac/src/execution/mod.rs`       — add enum + dispatch
///   - `crates/gnac/src/execution/backend.rs`   — add compile() method
///   - `crates/gnac/src/ir/mod.rs`              — fusion graph builder
///   - `crates/gnac/src/pipeline.rs`            — config plumbing

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
