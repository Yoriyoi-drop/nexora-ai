use std::collections::{HashMap, VecDeque};
use std::time::Instant;

const SIZE_BUCKETS: &[u64] = &[
    1 << 10,                // 1KB
    1 << 12,                // 4KB
    1 << 14,                // 16KB
    1 << 16,                // 64KB
    1 << 18,                // 256KB
    1 << 20,                // 1MB
    (1 << 20) + (1 << 19), // 1.5MB
    1 << 21,                // 2MB
    (1 << 21) + (1 << 20), // 3MB
    1 << 22,                // 4MB
    1 << 24,                // 16MB
    1 << 26,                // 64MB
    1 << 28,                // 256MB
    1 << 30,                // 1GB
    1 << 31,                // 2GB
    (1 << 30) + (1 << 30),  // 4GB
    1 << 33,                // 8GB
    1 << 34,                // 16GB
    1 << 35,                // 32GB
    1 << 36,                // 64GB
    (1 << 36) + (1 << 34),  // 80GB
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
const EVICTION_TTL: std::time::Duration = std::time::Duration::from_secs(30);

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
    pub dealloc_dropped: u64,
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
                crate::gpu::gpu_observability::POOL_CACHE_HITS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::gpu::gpu_observability::POOL_BYTES_REUSED
                    .fetch_add(b_size, std::sync::atomic::Ordering::Relaxed);
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
        crate::gpu::gpu_observability::POOL_CACHE_MISSES
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::gpu::gpu_observability::POOL_ALLOCS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::gpu::gpu_observability::POOL_BYTES_ALLOCATED
            .fetch_add(b_size, std::sync::atomic::Ordering::Relaxed);

        let extra = if usage.contains(wgpu::BufferUsages::MAP_READ)
            || usage.contains(wgpu::BufferUsages::MAP_WRITE)
        {
            wgpu::BufferUsages::empty()
        } else {
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: b_size,
            usage: usage | extra,
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
        crate::gpu::gpu_observability::POOL_DEALLOCS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(key) = buf.pool_key {
            let list = self.free_buffers.entry(key).or_default();
            evict_expired_in_bucket(list);
            if list.len() >= self.max_capacity {
                self.stats.dealloc_dropped += 1;
            } else {
                list.push_back((buf.buffer, Instant::now()));
            }
            while self.total_free_count() > LRU_EVICT_THRESHOLD {
                self.evict_one_lru();
            }
        }
    }

    /// Garbage collect all expired buffers across all buckets
    pub fn gc(&mut self) {
        for list in self.free_buffers.values_mut() {
            evict_expired_in_bucket(list);
        }
        // Force wgpu to process buffer destruction: GPU memory reclaimed
        self.device.poll(wgpu::PollType::Poll);
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
            // Skip empty buckets
            if list.is_empty() {
                continue;
            }
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
                // Remove bucket if now empty to avoid iterating over dead keys
                if list.is_empty() {
                    self.free_buffers.remove(&key);
                }
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
        // Trim excess buffers to fit new capacity per bucket
        for list in self.free_buffers.values_mut() {
            while list.len() > self.max_capacity {
                list.pop_front();
            }
        }
    }
}

/// Prune buffers in a bucket that have been idle longer than EVICTION_TTL
fn evict_expired_in_bucket(list: &mut VecDeque<(wgpu::Buffer, Instant)>) {
    let cutoff = Instant::now() - EVICTION_TTL;
    while let Some(front) = list.front() {
        if front.1 < cutoff {
            list.pop_front();
        } else {
            break;
        }
    }
}
