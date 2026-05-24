use std::collections::{HashMap, VecDeque};
use std::time::Instant;

const SIZE_BUCKETS: &[u64] = &[
    1 << 10,  // 1KB
    1 << 12,  // 4KB
    1 << 14,  // 16KB
    1 << 16,  // 64KB
    1 << 18,  // 256KB
    1 << 20,  // 1MB
    1 << 22,  // 4MB
    1 << 24,  // 16MB
    1 << 26,  // 64MB
    1 << 28,  // 256MB
    1 << 30,  // 1GB
    1 << 31,  // 2GB
    (1 << 30) + (1 << 30), // 4GB
];

pub fn bucket_for(size: u64) -> usize {
    for (i, &b) in SIZE_BUCKETS.iter().enumerate() {
        if size <= b {
            return i;
        }
    }
    SIZE_BUCKETS.len() - 1
}

fn bucket_size(idx: usize) -> u64 {
    SIZE_BUCKETS[idx]
}

#[derive(Debug)]
pub struct PooledBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pool_key: Option<(usize, wgpu::BufferUsages)>,
}

impl PooledBuffer {
    pub fn as_entire_binding(&self) -> wgpu::BindingResource {
        self.buffer.as_entire_binding()
    }
}

const MAX_POOL_CAPACITY: usize = 512;
const LRU_EVICT_THRESHOLD: usize = 384;

#[derive(Debug)]
pub struct GpuMemoryPool {
    device: wgpu::Device,
    free_buffers: HashMap<(usize, wgpu::BufferUsages), VecDeque<(wgpu::Buffer, Instant)>>,
    stats: PoolStats,
    max_capacity: usize,
}

#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub allocs: u64,
    pub deallocs: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub bytes_allocated: u64,
    pub bytes_reused: u64,
}

impl GpuMemoryPool {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            free_buffers: HashMap::new(),
            stats: PoolStats::default(),
            max_capacity: MAX_POOL_CAPACITY,
        }
    }

    pub fn with_max_capacity(mut self, max: usize) -> Self {
        self.max_capacity = max;
        self
    }

    pub fn alloc(&mut self, size: u64, usage: wgpu::BufferUsages) -> PooledBuffer {
        let bucket = bucket_for(size);
        let b_size = bucket_size(bucket);
        let key = (bucket, usage);

        if let Some(mut list) = self.free_buffers.remove(&key) {
            if let Some((buf, _)) = list.pop_back() {
                if !list.is_empty() {
                    self.free_buffers.insert(key, list);
                }
                self.stats.cache_hits += 1;
                self.stats.bytes_reused += b_size;
                return PooledBuffer {
                    buffer: buf,
                    size: b_size,
                    usage,
                    pool_key: Some((bucket, usage)),
                };
            }
        }

        self.stats.cache_misses += 1;
        self.stats.allocs += 1;
        self.stats.bytes_allocated += b_size;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: b_size,
            usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        PooledBuffer {
            buffer,
            size: b_size,
            usage,
            pool_key: Some((bucket, usage)),
        }
    }

    pub fn dealloc(&mut self, buf: PooledBuffer) {
        self.stats.deallocs += 1;
        if let Some(key) = buf.pool_key {
            let list = self.free_buffers.entry(key).or_default();
            if list.len() < self.max_capacity {
                list.push_back((buf.buffer, Instant::now()));
            }
            // LRU eviction if over threshold
            while self.total_free_count() > LRU_EVICT_THRESHOLD {
                self.evict_one_lru();
            }
        }
    }

    pub fn return_buffer_raw(&mut self, buffer: wgpu::Buffer, key: (usize, wgpu::BufferUsages)) {
        self.dealloc(PooledBuffer {
            buffer,
            size: bucket_size(key.0),
            usage: key.1,
            pool_key: Some(key),
        });
    }

    fn total_free_count(&self) -> usize {
        self.free_buffers.values().map(|v| v.len()).sum()
    }

    fn evict_one_lru(&mut self) {
        let mut oldest: Option<(usize, wgpu::BufferUsages)> = None;
        let mut oldest_time: Option<Instant> = None;

        for (key, list) in &self.free_buffers {
            if let Some((_, time)) = list.front() {
                match oldest_time {
                    None => {
                        oldest_time = Some(*time);
                        oldest = Some(*key);
                    }
                    Some(old) if *time < old => {
                        oldest_time = Some(*time);
                        oldest = Some(*key);
                    }
                    _ => {}
                }
            }
        }

        if let Some(key) = oldest {
            if let Some(list) = self.free_buffers.get_mut(&key) {
                list.pop_front();
            }
        }
    }

    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.stats = PoolStats::default();
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    pub fn set_max_capacity(&mut self, max: usize) {
        self.max_capacity = max;
    }
}
