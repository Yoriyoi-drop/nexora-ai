use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

/// GPU Prefetch Manager — prefetch data ke GPU sebelum dibutuhkan
pub struct GpuPrefetch {
    queue: Mutex<VecDeque<PrefetchRequest>>,
    active: AtomicBool,
}

pub struct PrefetchRequest {
    pub key: String,
    pub data: Vec<u8>,
    pub target_gpu: usize,
    pub priority: u32,
}

impl GpuPrefetch {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            active: AtomicBool::new(false),
        }
    }

    pub fn enqueue(&self, request: PrefetchRequest) {
        self.queue.lock().push_back(request);
    }

    pub fn dequeue(&self) -> Option<PrefetchRequest> {
        self.queue.lock().pop_front()
    }

    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

impl Default for GpuPrefetch {
    fn default() -> Self {
        Self::new()
    }
}
