use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
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
    1 << 34,                // 16GB — max, enough for 24–48 GB GPUs
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
    pub fn as_entire_binding(&self) -> wgpu::BindingResource<'_> {
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
    /// Optional coordinator for unified VRAM tracking
    coordinator: Option<std::sync::Weak<Mutex<MemoryCoordinator>>>,
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
            coordinator: None,
        }
    }

    pub fn with_max_capacity(mut self, max: usize) -> Self {
        self.max_capacity = max;
        self
    }

    /// Link to a MemoryCoordinator for unified VRAM tracking.
    pub fn with_coordinator(mut self, coord: &Arc<Mutex<MemoryCoordinator>>) -> Self {
        self.coordinator = Some(Arc::downgrade(coord));
        self
    }

    /// Update the coordinator with current pool usage.
    fn update_coordinator(&mut self) {
        if let Some(weak) = &self.coordinator {
            if let Some(coord) = weak.upgrade() {
                if let Ok(mut c) = coord.lock() {
                    c.set_pool_used(self.stats.bytes_allocated);
                }
            }
        }
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
                crate::autograd::gpu::gpu_observability::POOL_CACHE_HITS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::autograd::gpu::gpu_observability::POOL_BYTES_REUSED
                    .fetch_add(b_size, std::sync::atomic::Ordering::Relaxed);
                let result = PooledBuffer {
                    buffer: buf,
                    size: b_size,
                    usage,
                    pool_key: Some((bucket, usage)),
                };
                self.update_coordinator();
                return result;
            }
        }

        self.stats.cache_misses += 1;
        self.stats.allocs += 1;
        self.stats.bytes_allocated += b_size;
        crate::autograd::gpu::gpu_observability::POOL_CACHE_MISSES
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::autograd::gpu::gpu_observability::POOL_ALLOCS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::autograd::gpu::gpu_observability::POOL_BYTES_ALLOCATED
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

        let result = PooledBuffer {
            buffer,
            size: b_size,
            usage,
            pool_key: Some((bucket, usage)),
        };
        self.update_coordinator();
        result
    }

    pub fn dealloc(&mut self, buf: PooledBuffer) {
        self.stats.deallocs += 1;
        crate::autograd::gpu::gpu_observability::POOL_DEALLOCS
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
        self.update_coordinator();
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

    /// Compact the memory pool — evict oldest expired buffers and
    /// drop excess buffers from over-full buckets to reduce GPU memory pressure.
    /// Returns number of bytes freed.
    pub fn compact(&mut self) -> u64 {
        let before_total = self.stats.bytes_allocated;
        // Step 1: GC expired buffers
        self.gc();
        // Step 2: Trim buckets down to 75% capacity, evicting oldest first
        let target_per_bucket = (self.max_capacity as f64 * 0.75) as usize;
        for list in self.free_buffers.values_mut() {
            while list.len() > target_per_bucket {
                list.pop_front(); // oldest first
                self.stats.dealloc_dropped += 1;
            }
        }
        // Step 3: Recalculate total remaining bytes from bucket sizes
        let remaining: u64 = self.free_buffers.iter()
            .flat_map(|((bucket_idx, _usage), list)| {
                let b_size = bucket_size(*bucket_idx);
                list.iter().map(move |_| b_size)
            })
            .sum();
        let bytes_freed = before_total.saturating_sub(remaining);
        if bytes_freed > 0 {
            tracing::debug!("GpuMemoryPool::compact: freed {:.2} MB", bytes_freed as f64 / 1_048_576.0);
        }
        // Force GPU to release memory
        self.device.poll(wgpu::PollType::Poll);
        bytes_freed
    }

    /// Compute external fragmentation ratio — fraction of total allocated size
    /// that is wasted due to partial buckets.
    pub fn fragmentation_ratio(&self) -> f64 {
        let total_allocated = self.stats.bytes_allocated;
        if total_allocated == 0 {
            return 0.0;
        }
        let total_wasted: u64 = self.free_buffers.values()
            .flat_map(|v| v.iter())
            .map(|(buf, _)| {
                let buf_size = buf.size();
                let bucket = bucket_for(buf_size);
                let b_size = bucket_size(bucket);
                b_size.saturating_sub(buf_size)
            })
            .sum();
        total_wasted as f64 / total_allocated as f64
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

// ─── Unified Memory Coordinator ──────────────────────────────────────────────

/// Tracks a single external memory consumer (e.g., KV cache) alongside the GPU pool.
/// Memberikan pandangan terpadu tentang total VRAM yang digunakan.
///
/// Cara kerja:
/// - GPU pool allocations dilaporkan oleh `GpuMemoryPool` langsung
/// - External allocations (KV cache blocks, activation buffers) dilaporkan
///   oleh consumer melalui `register_external` / `update_external`
/// - `total_used_bytes()` return gabungan keduanya
#[derive(Debug)]
pub struct MemoryCoordinator {
    /// Total VRAM pada device (bytes).
    total_vram: u64,
    /// Bytes yang teralokasi oleh GpuMemoryPool.
    pool_used_bytes: u64,
    /// Bytes yang teralokasi oleh external consumer.
    external_used_bytes: u64,
    /// Soft limit — total allocation di atas ini trigger eviction.
    _eviction_threshold: u64,
    /// Hard limit — reject allocation di atas ini.
    critical_threshold: u64,
}

impl MemoryCoordinator {
    pub fn new(total_vram: u64) -> Self {
        let eviction = (total_vram as f64 * 0.70) as u64;
        let critical = (total_vram as f64 * 0.90) as u64;
        Self {
            total_vram,
            pool_used_bytes: 0,
            external_used_bytes: 0,
            _eviction_threshold: eviction,
            critical_threshold: critical,
        }
    }

    pub fn total_vram(&self) -> u64 { self.total_vram }
    pub fn pool_used(&self) -> u64 { self.pool_used_bytes }
    pub fn external_used(&self) -> u64 { self.external_used_bytes }
    pub fn total_used(&self) -> u64 { self.pool_used_bytes + self.external_used_bytes }
    pub fn available(&self) -> u64 { self.total_vram.saturating_sub(self.total_used()) }
    pub fn usage_ratio(&self) -> f64 { self.total_used() as f64 / self.total_vram as f64 }

    pub fn set_pool_used(&mut self, bytes: u64) { self.pool_used_bytes = bytes; }
    pub fn add_pool_used(&mut self, bytes: u64) { self.pool_used_bytes += bytes; }
    pub fn release_pool(&mut self, bytes: u64) {
        self.pool_used_bytes = self.pool_used_bytes.saturating_sub(bytes);
    }

    /// Register external allocation (KV cache, activations, etc).
    pub fn set_external(&mut self, bytes: u64) { self.external_used_bytes = bytes; }
    pub fn add_external(&mut self, bytes: u64) { self.external_used_bytes += bytes; }
    pub fn release_external(&mut self, bytes: u64) {
        self.external_used_bytes = self.external_used_bytes.saturating_sub(bytes);
    }

    /// Cek apakah total allocation (existing + new) akan exceed critical threshold.
    pub fn can_allocate(&self, additional_bytes: u64) -> bool {
        self.total_used() + additional_bytes < self.critical_threshold
    }

    /// Berapa banyak VRAM yang bisa dialokasikan sebelum critical.
    pub fn headroom(&self) -> u64 {
        self.critical_threshold.saturating_sub(self.total_used())
    }
}

#[cfg(test)]
mod memory_coordinator_tests {
    use super::*;

    #[test]
    fn test_memory_coordinator_basic() {
        let mut coord = MemoryCoordinator::new(24_000_000_000);
        assert_eq!(coord.total_vram(), 24_000_000_000);
        assert!(coord.can_allocate(1_000_000_000));

        coord.set_pool_used(10_000_000_000);
        coord.set_external(8_000_000_000);
        assert_eq!(coord.total_used(), 18_000_000_000);
        assert!(coord.can_allocate(1_000_000_000));
        assert!(!coord.can_allocate(10_000_000_000));
    }

    #[test]
    fn test_memory_coordinator_release() {
        let mut coord = MemoryCoordinator::new(1000);
        coord.add_pool_used(300);
        coord.add_external(200);
        assert_eq!(coord.total_used(), 500);

        coord.release_pool(100);
        coord.release_external(100);
        assert_eq!(coord.total_used(), 300);
    }

    #[test]
    fn test_memory_coordinator_headroom() {
        let mut coord = MemoryCoordinator::new(1000);
        // critical at 90% = 900
        assert_eq!(coord.headroom(), 900);

        coord.set_pool_used(500);
        coord.set_external(300);
        assert_eq!(coord.total_used(), 800);
        assert_eq!(coord.headroom(), 100);
    }
}
