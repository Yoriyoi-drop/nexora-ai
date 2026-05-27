use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::gpu::gpu_types::{GpuError, GpuErrorKind};

use once_cell::sync::Lazy;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Max retries per operation before circuit break.
    pub max_retries: u32,
    /// Base backoff in ms (doubles each retry).
    pub base_backoff_ms: u64,
    /// Max backoff cap in ms.
    pub max_backoff_ms: u64,
    /// Jitter factor (0.0–1.0) to stagger retries.
    pub jitter_factor: f64,
    /// Consecutive failures before circuit opens.
    pub circuit_break_threshold: u32,
    /// How long the circuit stays open (ms) before transitioning to HalfOpen.
    pub circuit_break_duration_ms: u64,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_backoff_ms: 50,
            max_backoff_ms: 10_000,
            jitter_factor: 0.2,
            circuit_break_threshold: 10,
            circuit_break_duration_ms: 30_000,
        }
    }
}

/// Per-error-type retry limits. DeviceLost and OutOfMemory are more severe
/// and get fewer retries before full circuit break.
impl RecoveryConfig {
    pub fn max_retries_for(&self, kind: GpuErrorKind) -> u32 {
        match kind {
            GpuErrorKind::DeviceLost => self.max_retries.min(2),
            GpuErrorKind::OutOfMemory => self.max_retries.min(1),
            GpuErrorKind::ShaderCompile => self.max_retries.min(1),
            GpuErrorKind::Timeout => self.max_retries,
            GpuErrorKind::Transient => self.max_retries,
            GpuErrorKind::Fatal => 0,
        }
    }
}

// ─── Circuit Breaker States ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    HalfOpen,
    Open,
}

impl CircuitState {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => CircuitState::Closed,
            1 => CircuitState::HalfOpen,
            _ => CircuitState::Open,
        }
    }

    fn to_u32(self) -> u32 {
        match self {
            CircuitState::Closed => 0,
            CircuitState::HalfOpen => 1,
            CircuitState::Open => 2,
        }
    }
}

// ─── Recovery Manager ────────────────────────────────────────────────────────

pub struct GpuRecoveryManager {
    circuit_state: AtomicU32,
    consecutive_failures: AtomicU32,
    total_failures: AtomicU64,
    recovered_count: AtomicU64,
    circuit_break_events: AtomicU64,
    device_lost_count: AtomicU64,
    oom_count: AtomicU64,
    shader_error_count: AtomicU64,
    timeout_count: AtomicU64,
    last_failure_type: Mutex<Option<GpuErrorKind>>,
    last_failure_time: Mutex<Option<Instant>>,
    last_recovery_time: Mutex<Option<Instant>>,
    config: Mutex<RecoveryConfig>,
}

impl GpuRecoveryManager {
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            circuit_state: AtomicU32::new(0),
            consecutive_failures: AtomicU32::new(0),
            total_failures: AtomicU64::new(0),
            recovered_count: AtomicU64::new(0),
            circuit_break_events: AtomicU64::new(0),
            device_lost_count: AtomicU64::new(0),
            oom_count: AtomicU64::new(0),
            shader_error_count: AtomicU64::new(0),
            timeout_count: AtomicU64::new(0),
            last_failure_type: Mutex::new(None),
            last_failure_time: Mutex::new(None),
            last_recovery_time: Mutex::new(None),
            config: Mutex::new(config),
        }
    }

    /// Record a GPU error and update circuit breaker state.
    /// Returns true if the circuit just tripped open.
    pub fn record_failure(&self, error: &GpuError) -> bool {
        let kind = error.kind();
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);

        // Track per-type counters
        match kind {
            GpuErrorKind::DeviceLost => {
                self.device_lost_count.fetch_add(1, Ordering::Relaxed);
            }
            GpuErrorKind::OutOfMemory => {
                self.oom_count.fetch_add(1, Ordering::Relaxed);
            }
            GpuErrorKind::ShaderCompile => {
                self.shader_error_count.fetch_add(1, Ordering::Relaxed);
            }
            GpuErrorKind::Timeout => {
                self.timeout_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        if let Ok(mut last_type) = self.last_failure_type.lock() {
            *last_type = Some(kind);
        }
        if let Ok(mut last_time) = self.last_failure_time.lock() {
            *last_time = Some(Instant::now());
        }

        let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        let threshold = cfg.circuit_break_threshold;
        drop(cfg);

        let consecutive = self.consecutive_failures.load(Ordering::Relaxed);
        if consecutive >= threshold {
            let prev = self.circuit_state.swap(CircuitState::Open.to_u32(), Ordering::Release);
            if prev != CircuitState::Open.to_u32() {
                self.circuit_break_events.fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    "GPU circuit breaker OPEN after {} consecutive failures (threshold={})",
                    consecutive,
                    threshold
                );
                return true;
            }
        }
        false
    }

    /// Record a successful GPU operation — resets circuit to Closed.
    pub fn record_recovery(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.recovered_count.fetch_add(1, Ordering::Relaxed);
        let prev = self.circuit_state.swap(CircuitState::Closed.to_u32(), Ordering::Release);
        if let Ok(mut t) = self.last_recovery_time.lock() {
            *t = Some(Instant::now());
        }
        if prev != CircuitState::Closed.to_u32() {
            tracing::info!("GPU circuit breaker CLOSED after successful operation");
        }
    }

    /// Returns current circuit breaker state, accounting for cooldown.
    pub fn circuit_state(&self) -> CircuitState {
        let state = CircuitState::from_u32(self.circuit_state.load(Ordering::Acquire));
        if state == CircuitState::Open {
            // Check if cooldown has elapsed → transition to HalfOpen
            if let Ok(t) = self.last_failure_time.lock() {
                if let Some(last_fail) = *t {
                    let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
                    let elapsed = last_fail.elapsed().as_millis() as u64;
                    if elapsed >= cfg.circuit_break_duration_ms {
                        self.circuit_state
                            .store(CircuitState::HalfOpen.to_u32(), Ordering::Release);
                        tracing::warn!(
                            "GPU circuit breaker HALF-OPEN after {}ms cooldown",
                            elapsed
                        );
                        return CircuitState::HalfOpen;
                    }
                }
            }
        }
        CircuitState::from_u32(self.circuit_state.load(Ordering::Acquire))
    }

    /// Returns true if the circuit is Open (all GPU ops should fallback to CPU).
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_state() == CircuitState::Open
    }

    /// Execute a GPU operation with retry + backoff + circuit breaker.
    /// Returns `Ok(result)` on success or the last `Err` after exhausting retries.
    pub fn try_execute<F, T>(&self, op: F) -> Result<T, GpuError>
    where
        F: Fn() -> Result<T, GpuError>,
    {
        // Check circuit breaker
        if self.is_circuit_open() {
            return Err(GpuError::DeviceLost(
                "Circuit breaker open — GPU operations blocked".into(),
            ));
        }

        let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        let max_retries = cfg.max_retries;
        let base_backoff = cfg.base_backoff_ms;
        let max_backoff = cfg.max_backoff_ms;
        let jitter_factor = cfg.jitter_factor;
        drop(cfg);

        let mut last_err = None;
        for attempt in 0..=max_retries {
            if attempt > 0 {
                let backoff_ms = self.backoff_duration(attempt, base_backoff, max_backoff, jitter_factor);
                std::thread::sleep(backoff_ms);
            }

            match op() {
                Ok(result) => {
                    self.record_recovery();
                    return Ok(result);
                }
                Err(e) => {
                    let tripped = self.record_failure(&e);
                    let kind = e.kind();
                    let retries_left = max_retries.saturating_sub(attempt);

                    // Don't retry fatal errors
                    if kind == GpuErrorKind::Fatal || retries_left == 0 {
                        return Err(e);
                    }

                    // DeviceLost gets fewer retries
                    if attempt >= self.config.lock().map(|c| c.max_retries_for(kind)).unwrap_or(1) {
                        return Err(e);
                    }

                    tracing::warn!(
                        "GPU op failed (attempt {}/{}): {:?}. {}",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        if tripped { "Circuit breaker OPENED!" } else { "Retrying..." }
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| GpuError::Device("Max retries exhausted".into())))
    }

    /// Compute exponential backoff with jitter.
    fn backoff_duration(
        &self,
        attempt: u32,
        base_ms: u64,
        max_ms: u64,
        jitter: f64,
    ) -> Duration {
        let exp = base_ms.saturating_mul(1u64 << attempt.saturating_sub(1).min(10));
        let capped = exp.min(max_ms);
        let jitter_amount = (capped as f64 * jitter) as u64;
        let jittered = if jitter_amount > 0 {
            let j = rand::random::<u64>() % jitter_amount;
            capped.saturating_add(j)
        } else {
            capped
        };
        Duration::from_millis(jittered)
    }

    /// Returns a JSON snapshot of recovery metrics.
    pub fn metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "circuit_state": format!("{:?}", self.circuit_state()),
            "consecutive_failures": self.consecutive_failures.load(Ordering::Relaxed),
            "total_failures": self.total_failures.load(Ordering::Relaxed),
            "recovered_count": self.recovered_count.load(Ordering::Relaxed),
            "circuit_break_events": self.circuit_break_events.load(Ordering::Relaxed),
            "device_lost_count": self.device_lost_count.load(Ordering::Relaxed),
            "oom_count": self.oom_count.load(Ordering::Relaxed),
            "shader_error_count": self.shader_error_count.load(Ordering::Relaxed),
            "timeout_count": self.timeout_count.load(Ordering::Relaxed),
        })
    }
}

// ─── Singleton ───────────────────────────────────────────────────────────────

pub static RECOVERY_MANAGER: Lazy<GpuRecoveryManager> =
    Lazy::new(|| GpuRecoveryManager::new(RecoveryConfig::default()));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let mgr = GpuRecoveryManager::new(RecoveryConfig::default());
        assert_eq!(mgr.circuit_state(), CircuitState::Closed);
        assert!(!mgr.is_circuit_open());
    }

    #[test]
    fn test_circuit_breaker_opens_on_threshold() {
        let config = RecoveryConfig {
            circuit_break_threshold: 3,
            circuit_break_duration_ms: 60_000,
            ..Default::default()
        };
        let mgr = GpuRecoveryManager::new(config);

        // 3 consecutive failures should open the circuit
        for i in 0..3 {
            let err = GpuError::Device(format!("failure {}", i));
            mgr.record_failure(&err);
        }
        assert_eq!(mgr.circuit_state(), CircuitState::Open);
        assert!(mgr.is_circuit_open());
    }

    #[test]
    fn test_circuit_breaker_closes_on_recovery() {
        let config = RecoveryConfig {
            circuit_break_threshold: 2,
            ..Default::default()
        };
        let mgr = GpuRecoveryManager::new(config);

        mgr.record_failure(&GpuError::Device("failure 1".into()));
        mgr.record_failure(&GpuError::Device("failure 2".into()));
        assert!(mgr.is_circuit_open());

        mgr.record_recovery();
        assert_eq!(mgr.circuit_state(), CircuitState::Closed);
        assert!(!mgr.is_circuit_open());
    }

    #[test]
    fn test_retry_success_after_failures() {
        let config = RecoveryConfig {
            max_retries: 3,
            base_backoff_ms: 1,
            ..Default::default()
        };
        let mgr = GpuRecoveryManager::new(config);

        let mut call_count = 0;
        let result = mgr.try_execute(|| {
            call_count += 1;
            if call_count < 3 {
                Err(GpuError::Timeout("transient".into()))
            } else {
                Ok(42)
            }
        });

        assert_eq!(result, Ok(42));
        assert_eq!(call_count, 3);
        assert_eq!(mgr.recovered_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_retry_exhaustion_returns_last_error() {
        let config = RecoveryConfig {
            max_retries: 2,
            base_backoff_ms: 1,
            ..Default::default()
        };
        let mgr = GpuRecoveryManager::new(config);

        let mut call_count = 0;
        let result = mgr.try_execute(|| {
            call_count += 1;
            Err(GpuError::Buffer("OOM".into()))
        });

        assert!(result.is_err());
        assert_eq!(call_count, 3); // initial + 2 retries = 3 calls
    }

    #[test]
    fn test_backoff_duration_increases() {
        let config = RecoveryConfig::default();
        let mgr = GpuRecoveryManager::new(config);

        let d1 = mgr.backoff_duration(1, 100, 10_000, 0.0);
        let d2 = mgr.backoff_duration(2, 100, 10_000, 0.0);
        let d3 = mgr.backoff_duration(3, 100, 10_000, 0.0);

        assert!(d1 <= d2);
        assert!(d2 <= d3);
    }

    #[test]
    fn test_device_lost_gets_fewer_retries() {
        let config = RecoveryConfig {
            max_retries: 5,
            ..Default::default()
        };
        assert_eq!(config.max_retries_for(GpuErrorKind::DeviceLost), 2);
        assert_eq!(config.max_retries_for(GpuErrorKind::OutOfMemory), 1);
        assert_eq!(config.max_retries_for(GpuErrorKind::Transient), 5);
        assert_eq!(config.max_retries_for(GpuErrorKind::Fatal), 0);
    }

    #[test]
    fn test_device_lost_counted_separately() {
        let mgr = GpuRecoveryManager::new(RecoveryConfig::default());
        mgr.record_failure(&GpuError::DeviceLost("TDR".into()));
        mgr.record_failure(&GpuError::OutOfMemory("VRAM full".into()));
        mgr.record_failure(&GpuError::ShaderCompile("invalid WGSL".into()));
        mgr.record_failure(&GpuError::WatchdogTimeout("5s deadline".into()));

        let m = mgr.metrics();
        assert_eq!(m["device_lost_count"], 1);
        assert_eq!(m["oom_count"], 1);
        assert_eq!(m["shader_error_count"], 1);
        assert_eq!(m["timeout_count"], 1);
        assert_eq!(m["total_failures"], 4);
    }
}
