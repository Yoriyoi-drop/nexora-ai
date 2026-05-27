use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;

// ─── Health status ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHealth {
    /// GPU is healthy and all operations are on-device.
    Healthy,
    /// GPU is degraded — some operations have fallen back to CPU.
    Degraded,
    /// GPU is unavailable — all operations are on CPU fallback.
    Unavailable,
}

// ─── Telemetry ───────────────────────────────────────────────────────────────

pub struct GpuTelemetry {
    // VRAM tracking (bytes)
    vram_total: AtomicU64,
    vram_used: AtomicU64,
    vram_free: AtomicU64,
    vram_peak_used: AtomicU64,

    // Error tracking
    error_count: AtomicU64,
    fallback_count: AtomicU64,
    oom_events: AtomicU64,
    device_lost_events: AtomicU64,

    // Timing
    last_healthy_time: Mutex<Option<Instant>>,
    last_error_time: Mutex<Option<Instant>>,
    first_error_time: Mutex<Option<Instant>>,

    // Time window for rate calculation
    window_error_count: AtomicU64,
    window_total_ops: AtomicU64,
    window_start: Mutex<Instant>,
}

impl GpuTelemetry {
    pub fn new() -> Self {
        Self {
            vram_total: AtomicU64::new(0),
            vram_used: AtomicU64::new(0),
            vram_free: AtomicU64::new(0),
            vram_peak_used: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
            oom_events: AtomicU64::new(0),
            device_lost_events: AtomicU64::new(0),
            last_healthy_time: Mutex::new(Some(Instant::now())),
            last_error_time: Mutex::new(None),
            first_error_time: Mutex::new(None),
            window_error_count: AtomicU64::new(0),
            window_total_ops: AtomicU64::new(0),
            window_start: Mutex::new(Instant::now()),
        }
    }

    /// Update VRAM usage statistics (called periodically or on allocation).
    pub fn update_vram(&self, used: u64, free: u64, total: u64) {
        self.vram_total.store(total, Ordering::Relaxed);
        self.vram_used.store(used, Ordering::Relaxed);
        self.vram_free.store(free, Ordering::Relaxed);

        // Track peak usage
        let mut prev = self.vram_peak_used.load(Ordering::Relaxed);
        while used > prev {
            match self.vram_peak_used.compare_exchange(prev, used, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(current) => prev = current,
            }
        }
    }

    /// Record a GPU error event.
    pub fn record_error(&self, is_oom: bool, is_device_lost: bool) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        self.window_error_count.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut t) = self.last_error_time.lock() {
            if t.is_none() {
                *t = Some(Instant::now());
            }
        }
        if let Ok(mut t) = self.first_error_time.lock() {
            t.get_or_insert_with(Instant::now);
        }
        if is_oom {
            self.oom_events.fetch_add(1, Ordering::Relaxed);
        }
        if is_device_lost {
            self.device_lost_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a GPU→CPU fallback event.
    pub fn record_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a GPU operation (for rate calculation).
    pub fn record_operation(&self) {
        self.window_total_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark GPU as healthy (error-free period).
    pub fn mark_healthy(&self) {
        if let Ok(mut t) = self.last_healthy_time.lock() {
            *t = Some(Instant::now());
        }
    }

    /// Error rate in the current time window (0.0 – 1.0).
    pub fn error_rate(&self) -> f64 {
        let errors = self.window_error_count.load(Ordering::Relaxed);
        let total = self.window_total_ops.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        errors as f64 / total as f64
    }

    /// Fallback ratio (0.0 – 1.0).
    pub fn fallback_ratio(&self) -> f64 {
        let errors = self.error_count.load(Ordering::Relaxed);
        let fallbacks = self.fallback_count.load(Ordering::Relaxed);
        let total = errors + fallbacks;
        if total == 0 {
            return 0.0;
        }
        fallbacks as f64 / total as f64
    }

    /// Overall health status combining circuit breaker and error rate.
    pub fn health_status(&self) -> GpuHealth {
        // If circuit breaker is open, GPU is unavailable
        if super::gpu_recovery::RECOVERY_MANAGER.is_circuit_open() {
            return GpuHealth::Unavailable;
        }

        // Check error rate in recent window
        let rate = self.error_rate();
        if rate > 0.5 {
            GpuHealth::Unavailable
        } else if rate > 0.1 {
            GpuHealth::Degraded
        } else {
            GpuHealth::Healthy
        }
    }

    /// VRAM fragmentation ratio (estimated: 1 - free/total).
    /// 0.0 = no fragmentation, 1.0 = completely fragmented.
    pub fn vram_fragmentation_ratio(&self) -> f64 {
        let total = self.vram_total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let used = self.vram_used.load(Ordering::Relaxed);
        let free = self.vram_free.load(Ordering::Relaxed);
        // Fragmentation is estimated as the ratio of used-to-total beyond free
        // A simple heuristic: if used + free > total, we have fragmentation
        let accounted = used.saturating_add(free);
        if accounted > total {
            (accounted.saturating_sub(total)) as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Reset the time window (called periodically to keep rates current).
    pub fn reset_window(&self) {
        self.window_error_count.store(0, Ordering::Relaxed);
        self.window_total_ops.store(0, Ordering::Relaxed);
        if let Ok(mut t) = self.window_start.lock() {
            *t = Instant::now();
        }
    }

    /// Returns a JSON snapshot of all telemetry metrics.
    pub fn metrics(&self) -> serde_json::Value {
        let vram_total = self.vram_total.load(Ordering::Relaxed);
        let vram_used = self.vram_used.load(Ordering::Relaxed);
        let vram_free = self.vram_free.load(Ordering::Relaxed);
        let vram_pct = if vram_total > 0 {
            (vram_used as f64 / vram_total as f64) * 100.0
        } else {
            0.0
        };

        serde_json::json!({
            "health": format!("{:?}", self.health_status()),
            "vram": {
                "total_bytes": vram_total,
                "used_bytes": vram_used,
                "free_bytes": vram_free,
                "peak_used_bytes": self.vram_peak_used.load(Ordering::Relaxed),
                "usage_percent": format!("{:.1}", vram_pct),
                "fragmentation_ratio": format!("{:.4}", self.vram_fragmentation_ratio()),
            },
            "errors": {
                "total": self.error_count.load(Ordering::Relaxed),
                "fallbacks": self.fallback_count.load(Ordering::Relaxed),
                "oom_events": self.oom_events.load(Ordering::Relaxed),
                "device_lost_events": self.device_lost_events.load(Ordering::Relaxed),
                "error_rate": format!("{:.4}", self.error_rate()),
                "fallback_ratio": format!("{:.4}", self.fallback_ratio()),
            },
        })
    }
}

// ─── Singleton ───────────────────────────────────────────────────────────────

pub static TELEMETRY: Lazy<GpuTelemetry> = Lazy::new(GpuTelemetry::new);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_starts_healthy() {
        let t = GpuTelemetry::new();
        assert!(t.health_status() == GpuHealth::Healthy);
        assert_eq!(t.error_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_telemetry_tracks_vram() {
        let t = GpuTelemetry::new();
        t.update_vram(2_000_000_000, 6_000_000_000, 8_000_000_000);

        assert_eq!(t.vram_used.load(Ordering::Relaxed), 2_000_000_000);
        assert_eq!(t.vram_free.load(Ordering::Relaxed), 6_000_000_000);
        assert_eq!(t.vram_total.load(Ordering::Relaxed), 8_000_000_000);
        assert_eq!(t.vram_peak_used.load(Ordering::Relaxed), 2_000_000_000);
    }

    #[test]
    fn test_telemetry_tracks_peak_vram() {
        let t = GpuTelemetry::new();
        t.update_vram(1_000, 9_000, 10_000);
        t.update_vram(5_000, 5_000, 10_000);
        t.update_vram(3_000, 7_000, 10_000);

        assert_eq!(t.vram_peak_used.load(Ordering::Relaxed), 5_000);
    }

    #[test]
    fn test_telemetry_error_rate() {
        let t = GpuTelemetry::new();
        assert_eq!(t.error_rate(), 0.0);

        t.record_operation();
        t.record_operation();
        t.record_operation();
        t.record_error(false, false); // 1 error in 4 ops = 0.25
        t.record_operation();

        assert!((t.error_rate() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_telemetry_fallback_ratio() {
        let t = GpuTelemetry::new();
        t.record_error(false, false);
        t.record_error(false, false);
        t.record_fallback(); // 1 fallback in 3 = 0.333

        assert!((t.fallback_ratio() - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_telemetry_resets_window() {
        let t = GpuTelemetry::new();
        t.record_error(false, false);
        t.record_error(false, false);
        t.record_operation();
        assert!((t.error_rate() - 2.0 / 3.0).abs() < 0.01);

        t.reset_window();
        assert_eq!(t.error_rate(), 0.0);
    }

    #[test]
    fn test_vram_fragmentation() {
        let t = GpuTelemetry::new();
        t.update_vram(6_000, 6_000, 10_000); // used+free exceeds total by 2GB
        let frag = t.vram_fragmentation_ratio();
        assert!(frag > 0.0);
    }
}
