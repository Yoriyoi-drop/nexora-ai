use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Async GPU Upload Manager — upload data ke GPU tanpa blocking host
pub struct AsyncGpuUpload {
    upload_queue: Mutex<VecDeque<UploadRequest>>,
    total_uploaded: AtomicU64,
    total_bytes: AtomicU64,
}

pub struct UploadRequest {
    pub key: String,
    pub data: Vec<u8>,
    pub target_gpu: usize,
    pub priority: u32,
}

impl AsyncGpuUpload {
    pub fn new() -> Self {
        Self {
            upload_queue: Mutex::new(VecDeque::new()),
            total_uploaded: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
        }
    }

    pub fn enqueue(&self, request: UploadRequest) {
        let size = request.data.len();
        self.upload_queue.lock().push_back(request);
        self.total_bytes
            .fetch_add(size as u64, Ordering::Relaxed);
    }

    pub fn dequeue(&self) -> Option<UploadRequest> {
        let req = self.upload_queue.lock().pop_front()?;
        self.total_uploaded.fetch_add(1, Ordering::Relaxed);
        Some(req)
    }

    pub fn queue_depth(&self) -> usize {
        self.upload_queue.lock().len()
    }

    pub fn total_uploaded(&self) -> u64 {
        self.total_uploaded.load(Ordering::Relaxed)
    }

    pub fn pending_bytes(&self) -> u64 {
        self.total_bytes
            .fetch_sub(0, Ordering::Relaxed)
    }
}

impl Default for AsyncGpuUpload {
    fn default() -> Self {
        Self::new()
    }
}
