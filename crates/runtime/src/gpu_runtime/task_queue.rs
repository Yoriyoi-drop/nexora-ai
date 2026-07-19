use parking_lot::Mutex;
use std::collections::VecDeque;
use uuid::Uuid;

use super::scheduler::GpuKernel;

/// Multi-priority GPU task queue
pub struct GpuTaskQueue {
    high: Mutex<VecDeque<GpuKernel>>,
    normal: Mutex<VecDeque<GpuKernel>>,
    low: Mutex<VecDeque<GpuKernel>>,
}

impl GpuTaskQueue {
    pub fn new() -> Self {
        Self {
            high: Mutex::new(VecDeque::new()),
            normal: Mutex::new(VecDeque::new()),
            low: Mutex::new(VecDeque::new()),
        }
    }

    pub fn push(&self, kernel: GpuKernel) {
        let queue = if kernel.priority >= 80 {
            &self.high
        } else if kernel.priority >= 40 {
            &self.normal
        } else {
            &self.low
        };
        queue.lock().push_back(kernel);
    }

    pub fn pop(&self) -> Option<GpuKernel> {
        self.high
            .lock()
            .pop_front()
            .or_else(|| self.normal.lock().pop_front().or_else(|| self.low.lock().pop_front()))
    }

    pub fn peek(&self) -> Option<Uuid> {
        self.high
            .lock()
            .front()
            .map(|k| k.id)
            .or_else(|| {
                self.normal
                    .lock()
                    .front()
                    .map(|k| k.id)
                    .or_else(|| self.low.lock().front().map(|k| k.id))
            })
    }

    pub fn len(&self) -> usize {
        self.high.lock().len() + self.normal.lock().len() + self.low.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.high.lock().is_empty()
            && self.normal.lock().is_empty()
            && self.low.lock().is_empty()
    }

    pub fn clear(&self) {
        self.high.lock().clear();
        self.normal.lock().clear();
        self.low.lock().clear();
    }
}

impl Default for GpuTaskQueue {
    fn default() -> Self {
        Self::new()
    }
}
