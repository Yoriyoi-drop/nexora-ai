//! Re-export crate for the autograd ecosystem.
//! All core types come from `nexora-autograd-core`.
//! GPU types come from `nexora-autograd-gpu` (cfg-gated).

// Re-export everything from autograd-core
pub use nexora_autograd_core::*;

// GPU modules (from autograd-gpu, cfg-gated)
#[cfg(feature = "gpu")]
pub use nexora_autograd_gpu::*;
#[cfg(feature = "gpu")]
pub use nexora_autograd_gpu::gpu;
