use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use super::scheduler::GpuKernel;

/// GPU Batch Manager — kumpulkan kernel untuk batch processing
pub struct GpuBatchManager {
    batches: Vec<Mutex<VecDeque<Vec<GpuKernel>>>>,
    max_batch_size: usize,
    total_batches: AtomicU64,
}

impl GpuBatchManager {
    pub fn new(gpu_count: usize) -> Self {
        Self {
            batches: (0..gpu_count)
                .map(|_| Mutex::new(VecDeque::new()))
                .collect(),
            max_batch_size: 32,
            total_batches: AtomicU64::new(0),
        }
    }

    pub fn add_to_batch(&self, gpu_id: usize, kernel: GpuKernel) {
        if gpu_id >= self.batches.len() {
            return;
        }
        let mut batch_queue = self.batches[gpu_id].lock();
        if let Some(last) = batch_queue.back_mut() {
            if last.len() < self.max_batch_size {
                last.push(kernel);
                return;
            }
        }
        batch_queue.push_back(vec![kernel]);
    }

    pub fn pop_batch(&self, gpu_id: usize) -> Option<Vec<GpuKernel>> {
        if gpu_id >= self.batches.len() {
            return None;
        }
        let batch = self.batches[gpu_id].lock().pop_front()?;
        self.total_batches.fetch_add(1, Ordering::Relaxed);
        Some(batch)
    }

    pub fn batch_count(&self, gpu_id: usize) -> usize {
        if gpu_id >= self.batches.len() {
            return 0;
        }
        self.batches[gpu_id].lock().len()
    }

    pub fn total_batches(&self) -> u64 {
        self.total_batches.load(Ordering::Relaxed)
    }
}
