//! GPU-level observability atomics.
//!
//! These are defined here (in `nexora-autograd`) so both GPU code
//! (`gpu_context.rs`, `gpu_memory.rs`, etc.) and the inference crate
//! can read/write them without circular dependencies.

use std::sync::atomic::AtomicU64;

/// Cumulative GPU busy time in nanoseconds (time spent in `sync` / `device.poll(Wait, …)`).
pub static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);

/// Cumulative bytes read from GPU (PCIe readback from buffer mappings).
pub static PCIE_READ_BYTES: AtomicU64 = AtomicU64::new(0);

/// Cumulative bytes written to GPU (PCIe upload via `write_buffer`).
pub static PCIE_WRITE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Memory pool allocation count.
pub static POOL_ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Memory pool deallocation count.
pub static POOL_DEALLOCS: AtomicU64 = AtomicU64::new(0);
/// Memory pool cache hits (reused a pooled buffer).
pub static POOL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
/// Memory pool cache misses (had to create new buffer).
pub static POOL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
/// Total bytes allocated by the memory pool.
pub static POOL_BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
/// Total bytes reused from the memory pool cache.
pub static POOL_BYTES_REUSED: AtomicU64 = AtomicU64::new(0);
