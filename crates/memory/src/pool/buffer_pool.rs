use parking_lot::Mutex;
use std::collections::HashMap;

/// Multi-size buffer pool — alokasikan buffer sesuai ukuran yang diminta
pub struct BufferPool {
    pools: Mutex<HashMap<usize, Vec<Vec<u8>>>>,
    max_per_size: usize,
}

impl BufferPool {
    pub fn new(max_per_size: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_per_size,
        }
    }

    pub fn acquire(&self, size: usize) -> Vec<u8> {
        let mut pools = self.pools.lock();
        if let Some(bufs) = pools.get_mut(&size) {
            if let Some(buf) = bufs.pop() {
                return buf;
            }
        }
        Vec::with_capacity(size)
    }

    pub fn release(&self, mut buf: Vec<u8>) {
        let size = buf.capacity();
        buf.clear();
        let mut pools = self.pools.lock();
        if let Some(bufs) = pools.get_mut(&size) {
            if bufs.len() < self.max_per_size {
                bufs.push(buf);
            }
        } else {
            pools.insert(size, vec![buf]);
        }
    }

    pub fn stats(&self) -> BufferPoolStats {
        let pools = self.pools.lock();
        let mut total = 0usize;
        let mut count = 0usize;
        for (size, bufs) in pools.iter() {
            total += size * bufs.len();
            count += bufs.len();
        }
        BufferPoolStats { total_bytes: total, total_buffers: count }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BufferPoolStats {
    pub total_bytes: usize,
    pub total_buffers: usize,
}
