use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::debug;
use uuid::Uuid;

use super::batching::GpuBatchManager;
use super::task_queue::GpuTaskQueue;

/// GPU Kernel task
#[derive(Debug, Clone)]
pub struct GpuKernel {
    pub id: Uuid,
    pub name: String,
    pub priority: u32,
    pub estimated_duration_us: u64,
    pub required_memory_mb: u64,
    pub stream_id: Option<Uuid>,
}

/// GPU Scheduler — jadwalkan kernel GPU agar tidak idle
pub struct GpuScheduler {
    task_queue: Arc<GpuTaskQueue>,
    #[allow(dead_code)]
    batch_manager: Arc<GpuBatchManager>,
    gpu_count: usize,
    gpu_util: Vec<AtomicU64>,
    gpu_active: Vec<AtomicBool>,
    total_kernels: AtomicU64,
    total_time_us: AtomicU64,
}

impl GpuScheduler {
    pub fn new(gpu_count: usize) -> Self {
        Self {
            task_queue: Arc::new(GpuTaskQueue::new()),
            batch_manager: Arc::new(GpuBatchManager::new(gpu_count)),
            gpu_count,
            gpu_util: (0..gpu_count).map(|_| AtomicU64::new(0)).collect(),
            gpu_active: (0..gpu_count).map(|_| AtomicBool::new(false)).collect(),
            total_kernels: AtomicU64::new(0),
            total_time_us: AtomicU64::new(0),
        }
    }

    pub fn submit_kernel(&self, kernel: GpuKernel) {
        let estimated = kernel.estimated_duration_us;
        self.task_queue.push(kernel);
        self.total_kernels.fetch_add(1, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(estimated, Ordering::Relaxed);
        debug!("GPU kernel queued");
    }

    pub fn submit_batch(&self, kernels: Vec<GpuKernel>) {
        for k in kernels {
            self.submit_kernel(k);
        }
    }

    /// Dapatkan GPU paling idle
    pub fn least_loaded_gpu(&self) -> usize {
        let mut best = 0;
        let mut best_load = u64::MAX;
        for i in 0..self.gpu_count {
            let load = self.gpu_util[i].load(Ordering::Relaxed);
            if load < best_load {
                best_load = load;
                best = i;
            }
        }
        best
    }

    pub fn gpu_utilization(&self) -> Vec<f64> {
        (0..self.gpu_count)
            .map(|i| {
                if self.gpu_active[i].load(Ordering::Relaxed) {
                    let util = self.gpu_util[i].load(Ordering::Relaxed) as f64;
                    (util / 1000.0).min(1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    pub fn queue_depth(&self) -> usize {
        self.task_queue.len()
    }

    pub fn stats(&self) -> GpuSchedulerStats {
        GpuSchedulerStats {
            total_kernels: self.total_kernels.load(Ordering::Relaxed),
            total_estimated_us: self.total_time_us.load(Ordering::Relaxed),
            queue_depth: self.task_queue.len(),
            gpu_utilization: self.gpu_utilization(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuSchedulerStats {
    pub total_kernels: u64,
    pub total_estimated_us: u64,
    pub queue_depth: usize,
    pub gpu_utilization: Vec<f64>,
}
