use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Metrics collector untuk agent scaling decisions
pub struct ScalingMetrics {
    /// Rata-rata CPU utilization (0.0 - 1.0)
    pub avg_cpu_util: std::sync::RwLock<f64>,
    /// Rata-rata memory utilization (0.0 - 1.0)
    pub avg_mem_util: std::sync::RwLock<f64>,
    /// Jumlah request dalam queue
    pub queue_depth: AtomicU64,
    /// Jumlah agent aktif
    pub active_agents: AtomicU64,
    /// Jumlah request diproses per detik
    pub requests_per_sec: AtomicU64,
    /// Total request dalam window
    total_requests: AtomicU64,
    window_start: std::sync::RwLock<Instant>,
}

impl ScalingMetrics {
    pub fn new() -> Self {
        Self {
            avg_cpu_util: std::sync::RwLock::new(0.0),
            avg_mem_util: std::sync::RwLock::new(0.0),
            queue_depth: AtomicU64::new(0),
            active_agents: AtomicU64::new(3),
            requests_per_sec: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            window_start: std::sync::RwLock::new(Instant::now()),
        }
    }

    pub fn record_request(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_active_agents(&self, count: usize) {
        self.active_agents
            .store(count as u64, Ordering::Relaxed);
    }

    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth as u64, Ordering::Relaxed);
    }

    pub fn set_utilization(&self, cpu: f64, mem: f64) {
        *self.avg_cpu_util.write().unwrap() = cpu;
        *self.avg_mem_util.write().unwrap() = mem;
    }

    /// Hitung request rate dan reset counter
    pub fn calculate_rps(&self) -> f64 {
        let total = self.total_requests.swap(0, Ordering::Relaxed);
        let elapsed = self.window_start.read().unwrap().elapsed().as_secs_f64();
        *self.window_start.write().unwrap() = Instant::now();
        let rps = if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        };
        self.requests_per_sec
            .store(rps as u64, Ordering::Relaxed);
        rps
    }
}

impl Default for ScalingMetrics {
    fn default() -> Self {
        Self::new()
    }
}
