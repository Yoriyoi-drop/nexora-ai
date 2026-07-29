use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// GPU-aware scheduler — track GPU memory dan task assignment
pub struct GpuAwareScheduler {
    gpu_memory_total: Vec<u64>,
    gpu_memory_used: Vec<AtomicUsize>,
    gpu_task_count: Vec<AtomicUsize>,
    task_to_gpu: DashMap<uuid::Uuid, usize>,
    task_gpu_memory: DashMap<uuid::Uuid, u64>,
}

impl GpuAwareScheduler {
    pub fn new(gpu_count: usize, memory_per_gpu: Vec<u64>) -> Self {
        let total = if memory_per_gpu.len() == gpu_count {
            memory_per_gpu
        } else {
            vec![16 * 1024 * 1024 * 1024; gpu_count]
        };
        Self {
            gpu_memory_total: total,
            gpu_memory_used: (0..gpu_count).map(|_| AtomicUsize::new(0)).collect(),
            gpu_task_count: (0..gpu_count).map(|_| AtomicUsize::new(0)).collect(),
            task_to_gpu: DashMap::new(),
            task_gpu_memory: DashMap::new(),
        }
    }

    /// Assign task ke GPU dengan memory paling tersedia
    pub fn assign_gpu(&self, task_id: uuid::Uuid, required_memory: u64) -> Option<usize> {
        let (best_gpu, _) = (0..self.gpu_memory_total.len())
            .map(|i| {
                let used = self.gpu_memory_used[i].load(Ordering::Relaxed) as u64;
                let free = self.gpu_memory_total[i].saturating_sub(used);
                (i, free)
            })
            .max_by_key(|&(_, free)| free)?;

        if self.gpu_memory_total[best_gpu]
            .saturating_sub(self.gpu_memory_used[best_gpu].load(Ordering::Relaxed) as u64)
            >= required_memory
        {
            self.gpu_memory_used[best_gpu]
                .fetch_add(required_memory as usize, Ordering::Relaxed);
            self.gpu_task_count[best_gpu].fetch_add(1, Ordering::Relaxed);
            self.task_to_gpu.insert(task_id, best_gpu);
            self.task_gpu_memory.insert(task_id, required_memory);
            return Some(best_gpu);
        }
        None
    }

    pub fn release_gpu(&self, task_id: &uuid::Uuid) {
        if let Some((_, gpu)) = self.task_to_gpu.remove(task_id) {
            if let Some((_, mem)) = self.task_gpu_memory.remove(task_id) {
                self.gpu_memory_used[gpu].fetch_sub(mem as usize, Ordering::Relaxed);
                self.gpu_task_count[gpu].fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    pub fn gpu_utilization(&self) -> Vec<f64> {
        (0..self.gpu_memory_total.len())
            .map(|i| {
                let used = self.gpu_memory_used[i].load(Ordering::Relaxed) as f64;
                let total = self.gpu_memory_total[i] as f64;
                if total > 0.0 { used / total } else { 0.0 }
            })
            .collect()
    }

    pub fn gpu_count(&self) -> usize {
        self.gpu_memory_total.len()
    }
}
