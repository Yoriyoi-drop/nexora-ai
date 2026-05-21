// ─── Phase 8: CPU Parallelism & Stabilization ─────────────────────────────────
//
// 8.1  CoreLayout — CPU topology detection + thread affinity helpers
// 8.2  OpenBLAS / env tuning recommendations (inline doc)
// 8.3  GpuDebugConfig — debug flags for sync, NaN check, deterministic mode
//
// Usage:
//   let layout = CoreLayout::detect();
//   layout.set_recommended_env();     // RAYON_NUM_THREADS, OMP_NUM_THREADS, etc.
//   layout.pin_current_thread();      // pin calling thread to a compute core
//
//   let cfg = GpuDebugConfig::from_env();  // read environment variables
//   let ctx = GpuContext::init_with_debug(cfg)?;

use std::fmt;

// ─── CPU Topology ──────────────────────────────────────────────────────────────

/// Describes the logical CPU core layout of the host machine.
///
/// **Convention:** The first 1–2 cores are reserved for data-loading / I/O
/// (rayon thread pool, async tokenizer), while the remaining cores host the
/// wgpu compute backend and any CPU‑fallback work.
#[derive(Debug, Clone)]
pub struct CoreLayout {
    /// Total logical cores on the machine (`num_cpus::get()`).
    pub total_cores: usize,
    /// Core indices reserved for data‑loading / ray on (e.g. `[0, 1]`).
    pub data_cores: Vec<usize>,
    /// Core indices left for wgpu compute / model forward (e.g. `[2, 3, …]`).
    pub compute_cores: Vec<usize>,
}

impl CoreLayout {
    /// Detect CPU topology from the running system.
    ///
    /// **Strategy:**
    /// - If ≥4 cores → reserve 2 cores for data, rest for compute.
    /// - If 2–3 cores → reserve 1 core for data.
    /// - Single‑core → everything runs on the same core (no separation).
    pub fn detect() -> Self {
        let total = num_cpus::get().max(1);

        let (data, compute) = if total >= 4 {
            // Reserve first 2 cores for data loading / I/O
            (vec![0, 1], (2..total).collect())
        } else if total >= 2 {
            // Reserve first core
            (vec![0], (1..total).collect())
        } else {
            (vec![0], vec![])
        };

        Self {
            total_cores: total,
            data_cores: data,
            compute_cores: compute,
        }
    }

    /// Recommended thread count for the rayon thread‑pool (data loading).
    pub fn rayon_threads(&self) -> usize {
        self.data_cores.len().max(1)
    }

    /// Recommended thread count for OpenBLAS (fallback CPU path).
    pub fn openblas_threads(&self) -> usize {
        self.data_cores.len().max(1)
    }

    /// Set environment variables to recommended values for hybrid GPU/CPU runs.
    ///
    /// If the process has already started its thread pools, this has no effect
    /// on those pools — call **before** spawning any threads.
    ///
    /// ```sh
    /// export OPENBLAS_NUM_THREADS=2
    /// export RAYON_NUM_THREADS=2
    /// export OMP_NUM_THREADS=1
    /// ```
    pub fn set_recommended_env(&self) {
        let rt = self.rayon_threads();
        let ot = self.openblas_threads();

        if std::env::var("RAYON_NUM_THREADS").is_err() {
            std::env::set_var("RAYON_NUM_THREADS", rt.to_string());
        }
        if std::env::var("OPENBLAS_NUM_THREADS").is_err() {
            std::env::set_var("OPENBLAS_NUM_THREADS", ot.to_string());
        }
        if std::env::var("OMP_NUM_THREADS").is_err() {
            std::env::set_var("OMP_NUM_THREADS", "1");
        }
        // wgpu backend threads should be free (no pinning, no cap)
    }

    /// Pin the calling thread to the first available compute core.
    ///
    /// Returns `true` if the pin succeeded, `false` if pinning is unsupported
    /// on this platform or there are no compute cores.
    ///
    /// **Linux only** — uses `libc::sched_setaffinity`.  No‑op on other
    /// platforms.
    #[cfg(target_os = "linux")]
    pub fn pin_to_compute(&self) -> bool {
        if self.compute_cores.is_empty() {
            return false;
        }
        self.pin_to_core(self.compute_cores[0])
    }

    /// Pin the calling thread to a specific logical core.
    ///
    /// **Linux only** — uses `libc::sched_setaffinity`.  Returns `false` on
    /// other platforms or if core index is out of range.
    #[cfg(target_os = "linux")]
    pub fn pin_to_core(&self, core: usize) -> bool {
        if core >= self.total_cores {
            return false;
        }
        // Build CPU set bitmask
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        unsafe { libc::CPU_SET(core, &mut set) };
        let ret = unsafe {
            libc::sched_setaffinity(
                0,                                 // calling thread
                std::mem::size_of::<libc::cpu_set_t>(),
                &set,
            )
        };
        ret == 0
    }

    /// Pin the calling thread to the lowest‑numbered data core.
    #[cfg(target_os = "linux")]
    pub fn pin_to_data(&self) -> bool {
        self.data_cores
            .first()
            .map(|&c| self.pin_to_core(c))
            .unwrap_or(false)
    }

    /// No‑op on non‑Linux platforms (always returns `false`).
    #[cfg(not(target_os = "linux"))]
    pub fn pin_to_compute(&self) -> bool {
        false
    }

    /// No‑op on non‑Linux platforms.
    #[cfg(not(target_os = "linux"))]
    pub fn pin_to_data(&self) -> bool {
        false
    }

    /// Human‑readable summary.
    pub fn display(&self) -> String {
        format!(
            "CoreLayout: {} total | data={:?} ({} threads) | compute={:?}",
            self.total_cores,
            self.data_cores,
            self.rayon_threads(),
            self.compute_cores,
        )
    }
}

// ─── OpenBLAS / Environment Tuning (8.2) ──────────────────────────────────────
//
// For best CPU‑fallback performance, ensure the following packages are
// installed and the recommended env vars are set (CoreLayout::set_recommended_env
// does this automatically at runtime):
//
// ```sh
// # Install OpenBLAS (Ubuntu/Debian)
// sudo apt install libopenblas-dev
//
// # Or with conda
// conda install openblas
//
// # Recommended env vars (set before starting the process)
// export OPENBLAS_NUM_THREADS=2     # match data_cores count
// export RAYON_NUM_THREADS=2        # match data_cores count
// export OMP_NUM_THREADS=1          # keep OpenMP serial to avoid oversubscription
// ```
//
// To enable BLAS acceleration in `ndarray`, add `"blas"` to the ndarray
// feature list in `crates/deeplearning/Cargo.toml` and pick a vendor crate:
//
// ```toml
// ndarray = { workspace = true, features = ["blas"] }
// openblas-src = { workspace = true }
// ```

// ─── GPU Debug Configuration (8.3) ────────────────────────────────────────────

/// Debug / stabilisation flags injected into `GpuContext` at initialisation.
///
/// ## Flags
///
/// | Flag | Env var | Description |
/// |---|---|---|
/// | `sync_execution` | `NEXORA_GPU_SYNC` | Call `device.poll(wait)` after every `flush()` — catches race conditions early. |
/// | `verbose_tensor_check` | `NEXORA_GPU_VERBOSE` | Log tensor shapes and op names on every dispatch (performance impact). |
/// | `deterministic` | `NEXORA_GPU_DETERMINISTIC` | Use fixed seed for all GPU‑side randomness (sampling, dropout mask). |
/// | `kernel_validation` | `NEXORA_GPU_VALIDATE` | (Future) compare GPU kernel output against CPU reference. |
///
/// # Example
///
/// ```ignore
/// let cfg = GpuDebugConfig { sync_execution: true, ..Default::default() };
/// let ctx = GpuContext::init_with_debug(cfg);
/// ```
#[derive(Debug, Clone)]
pub struct GpuDebugConfig {
    /// After every `flush()`, call `device.poll(wait)` to synchronise the
    /// CPU with the GPU submission.  This catches race conditions early at
    /// the cost of throughput.
    pub sync_execution: bool,

    /// Log every dispatch (pipeline name, workgroup count, tensor shapes).
    /// Useful for debugging which ops are being run.
    pub verbose_tensor_check: bool,

    /// Use a fixed seed for all GPU‑side randomness, making sampling and
    /// dropout mask generation reproducible across runs.
    pub deterministic: bool,

    /// (Future) Run a CPU reference kernel alongside each GPU dispatch and
    /// assert the outputs match within a tolerance.
    pub kernel_validation: bool,
}

impl GpuDebugConfig {
    /// Read flags from environment variables:
    ///
    /// - `NEXORA_GPU_SYNC=1` → `sync_execution: true`
    /// - `NEXORA_GPU_VERBOSE=1` → `verbose_tensor_check: true`
    /// - `NEXORA_GPU_DETERMINISTIC=1` → `deterministic: true`
    /// - `NEXORA_GPU_VALIDATE=1` → `kernel_validation: true`
    pub fn from_env() -> Self {
        Self {
            sync_execution: std::env::var("NEXORA_GPU_SYNC")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
            verbose_tensor_check: std::env::var("NEXORA_GPU_VERBOSE")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
            deterministic: std::env::var("NEXORA_GPU_DETERMINISTIC")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
            kernel_validation: std::env::var("NEXORA_GPU_VALIDATE")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
        }
    }
}

impl Default for GpuDebugConfig {
    fn default() -> Self {
        Self {
            sync_execution: false,
            verbose_tensor_check: false,
            deterministic: false,
            kernel_validation: false,
        }
    }
}

impl fmt::Display for GpuDebugConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpuDebugConfig(sync={} verbose={} deterministic={} validate={})",
            if self.sync_execution { "on" } else { "off" },
            if self.verbose_tensor_check { "on" } else { "off" },
            if self.deterministic { "on" } else { "off" },
            if self.kernel_validation { "on" } else { "off" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_layout_detect() {
        let layout = CoreLayout::detect();
        assert!(layout.total_cores >= 1, "at least 1 core");
        assert!(!layout.data_cores.is_empty(), "data_cores non-empty");
        if layout.total_cores >= 4 {
            assert_eq!(layout.data_cores.len(), 2, "reserve 2 cores for data when ≥4 total");
        }
        println!("{}", layout.display());
    }

    #[test]
    fn test_core_layout_env_vars() {
        let layout = CoreLayout::detect();
        layout.set_recommended_env();

        let rt = std::env::var("RAYON_NUM_THREADS").unwrap_or_default();
        let ot = std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_default();
        let omp = std::env::var("OMP_NUM_THREADS").unwrap_or_default();

        assert_eq!(rt, layout.rayon_threads().to_string());
        assert_eq!(ot, layout.openblas_threads().to_string());
        assert_eq!(omp, "1");
    }

    #[test]
    fn test_debug_config_default() {
        let cfg = GpuDebugConfig::default();
        assert!(!cfg.sync_execution);
        assert!(!cfg.verbose_tensor_check);
        assert!(!cfg.deterministic);
        assert!(!cfg.kernel_validation);
    }

    #[test]
    fn test_debug_config_from_env() {
        // Set env vars
        std::env::set_var("NEXORA_GPU_SYNC", "1");
        std::env::set_var("NEXORA_GPU_VERBOSE", "1");
        std::env::set_var("NEXORA_GPU_DETERMINISTIC", "1");
        std::env::set_var("NEXORA_GPU_VALIDATE", "1");

        let cfg = GpuDebugConfig::from_env();
        assert!(cfg.sync_execution);
        assert!(cfg.verbose_tensor_check);
        assert!(cfg.deterministic);
        assert!(cfg.kernel_validation);

        // Clean up
        std::env::remove_var("NEXORA_GPU_SYNC");
        std::env::remove_var("NEXORA_GPU_VERBOSE");
        std::env::remove_var("NEXORA_GPU_DETERMINISTIC");
        std::env::remove_var("NEXORA_GPU_VALIDATE");
    }
}
