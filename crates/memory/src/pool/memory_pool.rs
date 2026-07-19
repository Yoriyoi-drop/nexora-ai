use dashmap::DashMap;
use parking_lot::Mutex;
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, warn};

/// Aligned memory block dari pool
#[derive(Debug)]
pub struct MemoryBlock {
    pub ptr: *mut u8,
    pub size: usize,
    pub align: usize,
    pub pool_id: usize,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let layout = Layout::from_size_align(self.size, self.align).unwrap();
                dealloc(self.ptr, layout);
            }
        }
    }
}

/// Config per pool
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub name: String,
    pub block_size: usize,
    pub max_blocks: usize,
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            name: String::from("default"),
            block_size: 65536,
            max_blocks: 1024,
            alignment: 64,
        }
    }
}

/// Pool of pre-allocated fixed-size memory blocks
pub struct MemoryPool {
    config: MemoryPoolConfig,
    free: Mutex<Vec<MemoryBlock>>,
    allocated: Arc<AtomicUsize>,
    total_bytes: Arc<AtomicUsize>,
    max_bytes: Arc<AtomicUsize>,
}

impl MemoryPool {
    pub fn new(config: MemoryPoolConfig) -> Self {
        let total = config.block_size * config.max_blocks;
        Self {
            config,
            free: Mutex::new(Vec::with_capacity(1024)),
            allocated: Arc::new(AtomicUsize::new(0)),
            total_bytes: Arc::new(AtomicUsize::new(total)),
            max_bytes: Arc::new(AtomicUsize::new(total)),
        }
    }

    pub fn preallocate(&self, count: usize) {
        let mut free = self.free.lock();
        for _ in 0..count {
            let layout = Layout::from_size_align(self.config.block_size, self.config.alignment)
                .expect("invalid layout");
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                warn!("MemoryPool::preallocate OOM: {} blocks", count);
                return;
            }
            free.push(MemoryBlock {
                ptr,
                size: self.config.block_size,
                align: self.config.alignment,
                pool_id: 0,
            });
        }
        debug!(
            "MemoryPool[{}] preallocated {} blocks x {} bytes = {} total",
            self.config.name,
            count,
            self.config.block_size,
            count * self.config.block_size
        );
    }

    pub fn acquire(&self) -> Option<MemoryBlock> {
        let mut free = self.free.lock();
        let block = free.pop()?;
        self.allocated.fetch_add(1, Ordering::Relaxed);
        Some(block)
    }

    pub fn release(&self, block: MemoryBlock) {
        self.allocated.fetch_sub(1, Ordering::Relaxed);
        let mut free = self.free.lock();
        if free.len() < self.config.max_blocks {
            free.push(block);
        }
    }

    pub fn allocated_count(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    pub fn free_count(&self) -> usize {
        self.free.lock().len()
    }

    pub fn usage_ratio(&self) -> f64 {
        let total = self.config.max_blocks;
        if total == 0 {
            return 0.0;
        }
        self.allocated.load(Ordering::Relaxed) as f64 / total as f64
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }
}

/// Global memory pool registry
pub struct GlobalMemoryPools {
    pools: DashMap<String, Arc<MemoryPool>>,
}

impl GlobalMemoryPools {
    pub fn new() -> Self {
        Self {
            pools: DashMap::new(),
        }
    }

    pub fn register(&self, name: impl Into<String>, pool: MemoryPool) -> Arc<MemoryPool> {
        let name: String = name.into();
        let pool = Arc::new(pool);
        self.pools.insert(name, pool.clone());
        pool
    }

    pub fn get(&self, name: &str) -> Option<Arc<MemoryPool>> {
        self.pools.get(name).map(|p| p.clone())
    }

    pub fn unregister(&self, name: &str) {
        self.pools.remove(name);
    }
}

impl Default for GlobalMemoryPools {
    fn default() -> Self {
        Self::new()
    }
}
