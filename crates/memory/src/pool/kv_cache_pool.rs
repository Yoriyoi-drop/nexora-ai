use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct KVCacheBlock {
    pub key_block: Vec<f32>,
    pub value_block: Vec<f32>,
    pub sequence_id: u64,
    pub layer: usize,
    pub position: usize,
    pub ref_count: AtomicUsize,
}

impl KVCacheBlock {
    pub fn new(key_block: Vec<f32>, value_block: Vec<f32>, layer: usize, position: usize) -> Self {
        Self {
            key_block,
            value_block,
            sequence_id: 0,
            layer,
            position,
            ref_count: AtomicUsize::new(1),
        }
    }

    pub fn increment_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Relaxed).saturating_sub(1)
    }
}

pub struct KVCachePool {
    blocks: RwLock<Vec<Arc<KVCacheBlock>>>,
    max_blocks: usize,
    total_allocated: AtomicUsize,
    active_blocks: AtomicUsize,
}

impl KVCachePool {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            blocks: RwLock::new(Vec::with_capacity(max_blocks)),
            max_blocks,
            total_allocated: AtomicUsize::new(0),
            active_blocks: AtomicUsize::new(0),
        }
    }

    pub fn allocate(
        &self,
        key_block: Vec<f32>,
        value_block: Vec<f32>,
        layer: usize,
        position: usize,
    ) -> Option<Arc<KVCacheBlock>> {
        if self.active_blocks.load(Ordering::Relaxed) >= self.max_blocks {
            return None;
        }
        let block = Arc::new(KVCacheBlock::new(key_block, value_block, layer, position));
        self.blocks.write().push(block.clone());
        self.total_allocated.fetch_add(1, Ordering::Relaxed);
        self.active_blocks.fetch_add(1, Ordering::Relaxed);
        Some(block)
    }

    pub fn release(&self, block: &KVCacheBlock) {
        if block.decrement_ref() == 0 {
            self.active_blocks.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_blocks.load(Ordering::Relaxed)
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    pub fn utilization(&self) -> f64 {
        self.active_blocks.load(Ordering::Relaxed) as f64 / self.max_blocks as f64
    }
}

pub struct GlobalKVCachePool {
    pool: Arc<KVCachePool>,
}

impl GlobalKVCachePool {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            pool: Arc::new(KVCachePool::new(max_blocks)),
        }
    }

    pub fn pool(&self) -> &Arc<KVCachePool> {
        &self.pool
    }
}

impl Default for GlobalKVCachePool {
    fn default() -> Self {
        Self::new(65536)
    }
}
