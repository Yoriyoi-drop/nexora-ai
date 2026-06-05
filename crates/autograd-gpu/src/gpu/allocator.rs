

use super::gpu_types::*;
#[cfg(feature = "cuda")]
use super::cuda::CudaTensor;
/// Maximum workgroups per dimension for wgpu/WebGPU (65535).
/// Any dispatch exceeding this must be chunked across multiple dispatches
/// using buffer-slice bindings with per-chunk offsets.
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Default workgroup size used by most Nexora WGSL shaders.
const DEFAULT_WORKGROUP_SIZE: u32 = 256;

impl GpuContext {
    // ── Memory pool helpers ─────────────────────────────────────────────────────

    /// Allocate a buffer from the memory pool with automatic bucket sizing
    pub fn alloc_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> crate::gpu_memory::PooledBuffer {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .alloc(size, usage)
    }

    /// Allocate a buffer through the memory pool, falling back to
    /// `device.create_buffer` if the pool is unavailable.
    pub fn alloc_or_create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.alloc_buffer(size, usage).buffer
    }

    /// Return a buffer to the memory pool for reuse
    pub fn dealloc_buffer(&self, buf: crate::gpu_memory::PooledBuffer) {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .dealloc(buf);
    }

    /// Get memory pool statistics
    pub fn memory_stats(&self) -> crate::gpu_memory::PoolStats {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .stats()
            .clone()
    }

    /// Clear the memory pool (free all cached buffers)
    pub fn clear_memory_pool(&self) {
        self.memory_pool
            .lock()
            .unwrap_or_else(|e| {
                tracing::warn!("Lock poisoned: {}", e);
                e.into_inner()
            })
            .clear();
    }
}
