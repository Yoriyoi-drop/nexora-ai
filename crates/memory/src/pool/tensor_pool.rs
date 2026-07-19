use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::Arc;

/// Reusable tensor buffer — hindari alloc/free berulang
#[derive(Debug)]
pub struct TensorBuffer {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub in_use: bool,
}

impl TensorBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            shape: Vec::new(),
            in_use: false,
        }
    }

    pub fn reset(&mut self, shape: Vec<usize>) {
        let total: usize = shape.iter().product();
        self.data.clear();
        self.data.reserve(total);
        self.data.resize(total, 0.0);
        self.shape = shape;
        self.in_use = true;
    }

    pub fn clear(&mut self) {
        self.in_use = false;
        self.shape.clear();
    }
}

pub struct TensorPool {
    name: String,
    pool: Mutex<Vec<TensorBuffer>>,
    max_size: usize,
    created: std::sync::atomic::AtomicUsize,
    hits: std::sync::atomic::AtomicUsize,
    misses: std::sync::atomic::AtomicUsize,
}

impl TensorPool {
    pub fn new(name: impl Into<String>, max_size: usize) -> Self {
        Self {
            name: name.into(),
            pool: Mutex::new(Vec::with_capacity(max_size)),
            max_size,
            created: std::sync::atomic::AtomicUsize::new(0),
            hits: std::sync::atomic::AtomicUsize::new(0),
            misses: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn acquire(&self, min_capacity: usize) -> TensorBuffer {
        let mut pool = self.pool.lock();
        if let Some(pos) = pool.iter().position(|b| !b.in_use && b.data.capacity() >= min_capacity)
        {
            let mut buf = pool.swap_remove(pos);
            buf.in_use = true;
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return buf;
        }
        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let buf = TensorBuffer::new(min_capacity);
        self.created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        buf
    }

    pub fn release(&self, mut buf: TensorBuffer) {
        buf.clear();
        let mut pool = self.pool.lock();
        if pool.len() < self.max_size {
            pool.push(buf);
        }
    }

    pub fn stats(&self) -> TensorPoolStats {
        TensorPoolStats {
            name: self.name.clone(),
            pool_size: self.pool.lock().len(),
            max_size: self.max_size,
            created: self.created.load(std::sync::atomic::Ordering::Relaxed),
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorPoolStats {
    pub name: String,
    pub pool_size: usize,
    pub max_size: usize,
    pub created: usize,
    pub hits: usize,
    pub misses: usize,
}

/// Registry of named tensor pools
pub struct GlobalTensorPools {
    pools: DashMap<String, Arc<TensorPool>>,
}

impl GlobalTensorPools {
    pub fn new() -> Self {
        Self {
            pools: DashMap::new(),
        }
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        max_size: usize,
    ) -> Arc<TensorPool> {
        let name: String = name.into();
        let pool = Arc::new(TensorPool::new(name.clone(), max_size));
        self.pools.insert(name, pool.clone());
        pool
    }

    pub fn get(&self, name: &str) -> Option<Arc<TensorPool>> {
        self.pools.get(name).map(|p| p.clone())
    }
}

impl Default for GlobalTensorPools {
    fn default() -> Self {
        Self::new()
    }
}
