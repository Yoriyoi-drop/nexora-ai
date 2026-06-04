//! VRAM Budget Tracker — monitor, predict, and prevent OOM.
//!
//! Tracks per-component VRAM usage and provides OOM prevention:
//! - Model weights (dense + expert)
//! - KV cache
//! - Activations (per batch)
//! - Temporary buffers
//!
//! Usage:
//! ```ignore
//! let budget = VramBudget::new(24_000_000_000); // 24GB GPU
//! budget.reserve_model(200_000_000_000)?; // 200B model at Q4 ≈ 25GB
//! let permit = budget.try_allocate(512_000_000)?; // 512MB for batch
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Pressure level for VRAM usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramPressure {
    /// All good — usage below 70%.
    Ok,
    /// Warning — usage at 70–85%. Trigger proactive eviction.
    Warning,
    /// Critical — usage at 85–95%. Reject new requests, force eviction.
    Critical,
    /// OOM — usage above 95%. Immediate action required.
    Oom,
}

impl VramPressure {
    pub fn is_ok(&self) -> bool {
        matches!(self, VramPressure::Ok)
    }
}

/// Reservation handle — VRAM is held until this is dropped.
#[must_use]
pub struct VramReservation {
    bytes: u64,
    budget: Option<Arc<Mutex<VramBudget>>>,
}

impl VramReservation {
    pub fn new(bytes: u64, budget: Arc<Mutex<VramBudget>>) -> Self {
        Self { bytes, budget: Some(budget) }
    }
}

impl Drop for VramReservation {
    fn drop(&mut self) {
        if let Some(budget) = self.budget.take() {
            if let Ok(mut b) = budget.lock() {
                b.release_internal(self.bytes);
            }
        }
    }
}

/// Component-specific VRAM usage breakdown.
#[derive(Debug, Clone, Default)]
pub struct VramBreakdown {
    pub model_weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub expert_weights_bytes: u64,
    pub activations_bytes: u64,
    pub temp_buffers_bytes: u64,
}

/// VRAM budget tracker for a single GPU device.
///
/// Thread-safe via `Arc<Mutex<>>` for cross-component access.
pub struct VramBudget {
    /// Total VRAM on device (bytes).
    total_bytes: u64,
    /// Reserved for system/OS/framework overhead.
    reserved_bytes: u64,
    /// Per-component breakdown.
    breakdown: VramBreakdown,
    /// Peak usage tracking.
    peak_used_bytes: AtomicU64,
    /// Soft limit — trigger eviction above this.
    eviction_threshold: u64,
    /// Hard limit — reject requests above this.
    critical_threshold: u64,
}

impl VramBudget {
    /// Create budget for a device with `total_vram` bytes.
    /// Reserves 5% for system overhead by default.
    pub fn new(total_bytes: u64) -> Self {
        let reserved = (total_bytes as f64 * 0.05) as u64;
        let eviction = (total_bytes as f64 * 0.70) as u64;
        let critical = (total_bytes as f64 * 0.90) as u64;
        Self {
            total_bytes,
            reserved_bytes: reserved,
            breakdown: VramBreakdown::default(),
            peak_used_bytes: AtomicU64::new(0),
            eviction_threshold: eviction,
            critical_threshold: critical,
        }
    }

    /// Auto-configure limits based on model size and expert count.
    pub fn auto_configure(&mut self, model_params: usize, _num_experts: usize, bits_per_weight: usize) {
        let bytes_per_param = bits_per_weight as f64 / 8.0;
        let model_weight_bytes = (model_params as f64 * bytes_per_param) as u64;
        // Dense weights always resident
        self.breakdown.model_weights_bytes = model_weight_bytes;
        // Expert budget: try to keep ~20% of experts resident
        let expert_bytes = model_weight_bytes / 5; // rough: experts are part of model
        self.breakdown.expert_weights_bytes = expert_bytes.min(self.total_bytes / 3);
        // Set eviction threshold: model + expert + headroom
        let total_fixed = model_weight_bytes + expert_bytes;
        let min_headroom = 512_000_000u64; // 512MB minimum
        self.eviction_threshold = (total_fixed + min_headroom).min(self.total_bytes - self.reserved_bytes);
        self.critical_threshold = (self.total_bytes as f64 * 0.90) as u64;
    }

    /// Current total used VRAM (bytes).
    pub fn used_bytes(&self) -> u64 {
        self.breakdown.model_weights_bytes
            + self.breakdown.kv_cache_bytes
            + self.breakdown.expert_weights_bytes
            + self.breakdown.activations_bytes
            + self.breakdown.temp_buffers_bytes
    }

    /// Available VRAM for new allocations (bytes).
    pub fn available(&self) -> u64 {
        self.total_bytes
            .saturating_sub(self.reserved_bytes)
            .saturating_sub(self.used_bytes())
    }

    /// Usage ratio (0.0–1.0).
    pub fn usage_ratio(&self) -> f64 {
        self.used_bytes() as f64 / self.total_bytes as f64
    }

    /// Current pressure level.
    pub fn pressure(&self) -> VramPressure {
        let ratio = self.usage_ratio();
        if ratio >= 0.95 {
            VramPressure::Oom
        } else if ratio >= 0.85 {
            VramPressure::Critical
        } else if ratio >= 0.70 {
            VramPressure::Warning
        } else {
            VramPressure::Ok
        }
    }

    /// Can we allocate `bytes` without going over critical?
    pub fn can_allocate(&self, bytes: u64) -> bool {
        self.used_bytes() + bytes + self.reserved_bytes < self.critical_threshold
    }

    /// Reserve `bytes` of VRAM. Returns `Err` if would exceed critical.
    /// Uses `self_arc` so the returned VramReservation can release on drop.
    pub fn reserve(&mut self, bytes: u64, self_arc: Arc<Mutex<VramBudget>>) -> Result<VramReservation, String> {
        if !self.can_allocate(bytes) {
            return Err(format!(
                "VRAM allocation of {} bytes would exceed critical threshold ({} used / {} total)",
                bytes,
                self.used_bytes(),
                self.total_bytes
            ));
        }
        self.breakdown.temp_buffers_bytes += bytes;
        self.peak_used_bytes.fetch_max(self.used_bytes(), Ordering::Relaxed);
        Ok(VramReservation { bytes, budget: Some(self_arc) })
    }

    fn release_internal(&mut self, bytes: u64) {
        self.breakdown.temp_buffers_bytes = self.breakdown.temp_buffers_bytes.saturating_sub(bytes);
    }

    /// Update model weight usage (call when model config changes).
    pub fn set_model_weights(&mut self, bytes: u64) {
        self.breakdown.model_weights_bytes = bytes;
    }

    /// Update expert weight GPU usage (call after offloader swap).
    pub fn set_expert_weights(&mut self, bytes: u64) {
        self.breakdown.expert_weights_bytes = bytes;
        self.peak_used_bytes.fetch_max(self.used_bytes(), Ordering::Relaxed);
    }

    /// Update KV cache usage.
    pub fn set_kv_cache(&mut self, bytes: u64) {
        self.breakdown.kv_cache_bytes = bytes;
    }

    /// Update activation memory (per-batch).
    pub fn set_activations(&mut self, bytes: u64) {
        self.breakdown.activations_bytes = bytes;
    }

    /// Peak VRAM usage observed (bytes).
    pub fn peak_used(&self) -> u64 {
        self.peak_used_bytes.load(Ordering::Relaxed)
    }

    /// Total VRAM on device.
    pub fn total(&self) -> u64 {
        self.total_bytes
    }

    /// Get current breakdown.
    pub fn breakdown(&self) -> &VramBreakdown {
        &self.breakdown
    }

    /// Component breakdown summary strings.
    pub fn summary(&self) -> Vec<(String, u64)> {
        vec![
            ("model".into(), self.breakdown.model_weights_bytes),
            ("experts".into(), self.breakdown.expert_weights_bytes),
            ("kv_cache".into(), self.breakdown.kv_cache_bytes),
            ("activations".into(), self.breakdown.activations_bytes),
            ("temp".into(), self.breakdown.temp_buffers_bytes),
        ]
    }

    /// Print human-readable VRAM status.
    pub fn print_status(&self) {
        let total_gb = self.total_bytes as f64 / 1e9;
        let used_gb = self.used_bytes() as f64 / 1e9;
        let avail_gb = self.available() as f64 / 1e9;
        tracing::info!(
            "VRAM: {:.2}GB used / {:.2}GB total ({:.2}GB free, pressure={:?})",
            used_gb, total_gb, avail_gb, self.pressure()
        );
        for (name, bytes) in self.summary() {
            if bytes > 0 {
                tracing::info!("  {}: {:.2}GB", name, bytes as f64 / 1e9);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_budget_create() {
        let budget = VramBudget::new(24_000_000_000); // 24GB
        assert_eq!(budget.total(), 24_000_000_000);
        assert!(budget.available() > 0);
        assert_eq!(budget.pressure(), VramPressure::Ok);
    }

    #[test]
    fn test_auto_configure() {
        let mut budget = VramBudget::new(24_000_000_000);
        // 200B model at Q4 ≈ 25GB storage, expert count = 256
        budget.auto_configure(200_000_000_000, 256, 4);
        assert!(budget.breakdown.model_weights_bytes > 0);
        assert!(budget.eviction_threshold > 0);
    }

    #[test]
    fn test_reserve_and_release() {
        let budget = Arc::new(Mutex::new(VramBudget::new(24_000_000_000)));
        let reservation = {
            let mut b = budget.lock().unwrap();
            let reservation = b.reserve(1_000_000_000, budget.clone()).unwrap();
            assert!(!b.can_allocate(24_000_000_000));
            reservation
        };
        drop(reservation);
        let b = budget.lock().unwrap();
        assert!(b.can_allocate(1_000_000_000));
    }

    #[test]
    fn test_pressure_levels() {
        let mut budget = VramBudget::new(1000);
        assert_eq!(budget.pressure(), VramPressure::Ok);
        budget.set_model_weights(750);
        assert_eq!(budget.pressure(), VramPressure::Warning);
        budget.set_model_weights(900);
        assert_eq!(budget.pressure(), VramPressure::Critical);
        budget.set_model_weights(960);
        assert_eq!(budget.pressure(), VramPressure::Oom);
    }
}
