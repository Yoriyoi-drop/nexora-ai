use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::gpu::GpuContext;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// Max allowed time (ms) per GPU operation before considered hung.
    pub timeout_ms: u64,
    /// How often (ms) the watchdog checks for hang detection.
    pub check_interval_ms: u64,
    /// Max consecutive hang detections before triggering full reset.
    pub max_hangs_before_reset: u32,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,  // 30s per operation
            check_interval_ms: 5_000, // check every 5s
            max_hangs_before_reset: 3,
        }
    }
}

// ─── Watchdog ────────────────────────────────────────────────────────────────

pub struct GpuWatchdog {
    /// Timestamp of last successful GPU operation.
    last_activity: Mutex<Instant>,
    /// Number of consecutive hang detections.
    hang_count: AtomicU32,
    /// Total number of watchdog-triggered resets.
    reset_count: AtomicU64,
    /// Total number of timeouts detected.
    timeout_count: AtomicU64,
    /// Whether the watchdog is actively monitoring.
    enabled: AtomicU32, // 0 = disabled, 1 = enabled
    config: Mutex<WatchdogConfig>,
}

impl GpuWatchdog {
    pub fn new(config: WatchdogConfig) -> Self {
        Self {
            last_activity: Mutex::new(Instant::now()),
            hang_count: AtomicU32::new(0),
            reset_count: AtomicU64::new(0),
            timeout_count: AtomicU64::new(0),
            enabled: AtomicU32::new(1),
            config: Mutex::new(config),
        }
    }

    /// Mark GPU activity — resets the hang timer.
    pub fn ping(&self) {
        if let Ok(mut last) = self.last_activity.lock() {
            *last = Instant::now();
        }
        self.hang_count.store(0, Ordering::Relaxed);
    }

    /// Enable or disable watchdog monitoring.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled as u32, Ordering::Release);
        if enabled {
            self.ping();
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire) != 0
    }

    /// Check if the GPU appears hung (no activity within timeout).
    /// Returns `true` if a hang is detected.
    pub fn check(&self) -> bool {
        if !self.is_enabled() {
            return false;
        }

        let elapsed = match self.last_activity.lock() {
            Ok(last) => last.elapsed(),
            Err(_) => return false,
        };

        let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        if elapsed > Duration::from_millis(cfg.timeout_ms) {
            let hangs = self.hang_count.fetch_add(1, Ordering::Relaxed) + 1;
            self.timeout_count.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "GPU watchdog: no activity for {:?} (hang {}/{} before reset)",
                elapsed,
                hangs,
                cfg.max_hangs_before_reset
            );

            if hangs >= cfg.max_hangs_before_reset {
                tracing::error!(
                    "GPU watchdog: {} consecutive hangs detected — triggering reset!",
                    hangs
                );
                self.trigger_reset();
            }
            return true;
        }
        false
    }

    /// Trigger a GPU reset (called on hang threshold or externally).
    /// Attempts to rebuild the GPU context.
    pub fn trigger_reset(&self) {
        self.reset_count.fetch_add(1, Ordering::Relaxed);
        self.hang_count.store(0, Ordering::Relaxed);

        // Attempt to rebuild GPU context
        if let Ok(ctx) = GpuContext::global() {
            match ctx.try_rebuild() {
                Ok(_) => {
                    tracing::info!("GPU watchdog: context rebuilt successfully after reset");
                    self.ping();
                }
                Err(e) => {
                    tracing::error!("GPU watchdog: context rebuild failed after reset: {:?}", e);
                }
            }
        } else {
            tracing::warn!("GPU watchdog: cannot reset — GPU context not initialized");
        }
    }

    /// Returns a JSON snapshot of watchdog metrics.
    pub fn metrics(&self) -> serde_json::Value {
        let hang_count = self.hang_count.load(Ordering::Relaxed);
        let reset_count = self.reset_count.load(Ordering::Relaxed);
        let timeout_count = self.timeout_count.load(Ordering::Relaxed);
        serde_json::json!({
            "enabled": self.is_enabled(),
            "hang_count": hang_count,
            "reset_count": reset_count,
            "timeout_count": timeout_count,
            "max_hangs_before_reset": {
                let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
                cfg.max_hangs_before_reset
            },
            "timeout_ms": {
                let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
                cfg.timeout_ms
            },
        })
    }
}

// ─── Singleton ───────────────────────────────────────────────────────────────

pub static WATCHDOG: Lazy<GpuWatchdog> = Lazy::new(|| GpuWatchdog::new(WatchdogConfig::default()));

/// Call this after any successful GPU operation to keep the watchdog happy.
pub fn watchdog_ping() {
    WATCHDOG.ping();
}

/// Call periodically from background thread to check for GPU hangs.
pub fn watchdog_check() {
    WATCHDOG.check();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_starts_enabled() {
        let wd = GpuWatchdog::new(WatchdogConfig::default());
        assert!(wd.is_enabled());
        assert_eq!(wd.hang_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_watchdog_ping_resets_hang_count() {
        let wd = GpuWatchdog::new(WatchdogConfig::default());
        wd.hang_count.store(3, Ordering::Relaxed);
        wd.ping();
        assert_eq!(wd.hang_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_watchdog_disabled_does_not_check() {
        let wd = GpuWatchdog::new(WatchdogConfig {
            timeout_ms: 0, // would trigger immediately
            ..Default::default()
        });
        wd.set_enabled(false);
        assert!(!wd.check()); // disabled = never hung
    }

    #[test]
    fn test_watchdog_metrics_contain_counters() {
        let wd = GpuWatchdog::new(WatchdogConfig::default());
        wd.timeout_count.fetch_add(5, Ordering::Relaxed);
        let m = wd.metrics();
        assert_eq!(m["timeout_count"], 5);
        assert!(m["enabled"].as_bool().unwrap_or(false));
    }
}
