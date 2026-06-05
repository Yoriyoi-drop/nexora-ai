//! GPU-accelerated computation for Nexora Autograd
//! Supports wgpu (Vulkan/Metal/DX12) and CUDA backends.

pub mod gpu_adam;
pub mod gpu_async;
pub mod gpu_backward;
pub mod gpu_batch;
pub mod gpu_bench;
pub mod gpu_caps;
pub mod gpu_check;
pub mod gpu_core_layout;
pub mod gpu_fused;
pub mod gpu_grad_clip;
pub mod gpu_kv_cache;
pub mod gpu_memory;
pub mod gpu_mixed;
pub mod gpu_profiler;
pub mod gpu_sampler;
pub mod gpu_sedc;
pub mod persistent_cache;

pub mod gpu;

// Re-exports
pub use gpu_batch::GpuCommandBatch;
pub use gpu_bench::{gflops, matmul_flops, MatmulBatchTiming, MatmulGflopsFloor};
pub use gpu_caps::GpuCapabilities;
pub use gpu_grad_clip::GpuGradClipResult;
