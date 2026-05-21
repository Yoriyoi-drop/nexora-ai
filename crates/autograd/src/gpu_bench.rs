//! Shared GPU benchmark helpers — used by tests and optional runtime profiling.

use std::time::Duration;

/// FLOPs for `C = A @ B` with shapes `[m,k] @ [k,n]`.
#[inline]
pub fn matmul_flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64
}

/// GFLOPS from flop count and duration.
#[inline]
pub fn gflops(flops: f64, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    flops / secs / 1e9
}

/// Per-iteration GFLOPS for square `n×n` matmul.
#[inline]
pub fn square_matmul_gflops(n: usize, per_iter: Duration) -> f64 {
    gflops(matmul_flops(n, n, n), per_iter)
}

/// Minimum expected GFLOPS for smoke tests (debug builds, wgpu/Vulkan).
#[derive(Debug, Clone, Copy)]
pub struct MatmulGflopsFloor {
    pub n128: f64,
    pub n256: f64,
    pub n512: f64,
    pub n1024: f64,
}

impl Default for MatmulGflopsFloor {
    fn default() -> Self {
        Self {
            n128: 0.5,
            n256: 1.0,
            n512: 5.0,
            n1024: 10.0,
        }
    }
}

impl MatmulGflopsFloor {
    pub fn min_for(&self, n: usize) -> f64 {
        match n {
            128 => self.n128,
            256 => self.n256,
            512 => self.n512,
            1024 => self.n1024,
            _ => 0.5,
        }
    }
}

/// Result of a batched matmul timing call.
#[derive(Debug, Clone, Copy)]
pub struct MatmulBatchTiming {
    pub batch_size: usize,
    pub gpu_total: Duration,
    pub gpu_per_iter: Duration,
    pub gflops_per_iter: f64,
}

impl MatmulBatchTiming {
    pub fn from_square(n: usize, batch_size: usize, gpu_total: Duration) -> Self {
        let gpu_per_iter = gpu_total
            .checked_div(batch_size as u32)
            .unwrap_or(gpu_total);
        let gflops_per_iter = square_matmul_gflops(n, gpu_per_iter);
        Self {
            batch_size,
            gpu_total,
            gpu_per_iter,
            gflops_per_iter,
        }
    }
}
