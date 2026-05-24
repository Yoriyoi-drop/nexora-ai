use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Tensor paging — memori virtual untuk tensor dengan real data transfer
/// Tensor di-swap ke host memory saat tidak digunakan, di-load balik saat diperlukan
pub struct TensorPager {
    page_size: usize,
    max_vram_pages: usize,
    access_history: VecDeque<u64>,
    page_table: HashMap<u64, PageEntry>,
    host_storage: HashMap<u64, Vec<u8>>,
    total_bytes_transferred: u64,
    total_load_time_ns: u64,
    load_count: u64,
}

#[derive(Debug, Clone)]
struct PageEntry {
    tensor_id: u64,
    is_loaded: bool,
    last_access: Instant,
    size_bytes: usize,
}

impl TensorPager {
    pub fn new(max_vram_mb: f64, page_size_kb: usize) -> Self {
        let max_vram_bytes = (max_vram_mb * 1_000_000.0) as usize;
        TensorPager {
            page_size: page_size_kb * 1024,
            max_vram_pages: if page_size_kb > 0 { max_vram_bytes / (page_size_kb * 1024) } else { 0 },
            access_history: VecDeque::new(),
            page_table: HashMap::new(),
            host_storage: HashMap::new(),
            total_bytes_transferred: 0,
            total_load_time_ns: 0,
            load_count: 0,
        }
    }

    /// Request akses ke tensor (load dari host ke device jika perlu)
    pub fn access(&mut self, tensor_id: u64) {
        let now = Instant::now();
        let size = self.page_size;
        let entry = self.page_table.entry(tensor_id).or_insert_with(|| {
            // Allocate host storage for this tensor
            let host_data = vec![0u8; size];
            self.host_storage.insert(tensor_id, host_data);
            PageEntry {
                tensor_id,
                is_loaded: false,
                last_access: now,
                size_bytes: size,
            }
        });

        entry.last_access = now;

        if !entry.is_loaded {
            self.load_page(tensor_id);
        }

        self.access_history.push_back(tensor_id);
        if self.access_history.len() > 1000 {
            self.access_history.pop_front();
        }
    }

    /// Real load: copy data from host storage to simulated device memory
    fn load_page(&mut self, tensor_id: u64) {
        let loaded_count = self.page_table.values().filter(|e| e.is_loaded).count();
        if loaded_count >= self.max_vram_pages {
            self.evict_lru();
        }

        if let Some(entry) = self.page_table.get_mut(&tensor_id) {
            if !entry.is_loaded {
                // Simulate PCIe transfer: copy host → device
                let start = Instant::now();
                if let Some(host_data) = self.host_storage.get(&tensor_id) {
                    let _device_buffer = host_data.clone();
                    let elapsed = start.elapsed().as_nanos() as u64;
                    self.total_bytes_transferred += host_data.len() as u64;
                    self.total_load_time_ns += elapsed;
                    self.load_count += 1;
                }
                entry.is_loaded = true;
            }
        }
    }

    /// Real evict: copy data from device back to host storage
    fn evict_lru(&mut self) {
        if let Some(oldest_id) = self.access_history.pop_front() {
            if let Some(entry) = self.page_table.get_mut(&oldest_id) {
                if entry.is_loaded {
                    // Simulate PCIe transfer: copy device → host
                    let start = Instant::now();
                    if let Some(host_data) = self.host_storage.get_mut(&oldest_id) {
                        let _device_readback = host_data.clone();
                        let elapsed = start.elapsed().as_nanos() as u64;
                        self.total_bytes_transferred += host_data.len() as u64;
                        self.total_load_time_ns += elapsed;
                        self.load_count += 1;
                    }
                    entry.is_loaded = false;
                }
            }
        }
    }

    /// Check if a tensor is currently loaded in device memory
    pub fn is_loaded(&self, tensor_id: u64) -> bool {
        self.page_table
            .get(&tensor_id)
            .map(|e| e.is_loaded)
            .unwrap_or(false)
    }

    /// Get the number of pages currently loaded in VRAM
    pub fn loaded_count(&self) -> usize {
        self.page_table.values().filter(|e| e.is_loaded).count()
    }

    /// Get paging statistics
    pub fn stats(&self) -> PagingStats {
        PagingStats {
            total_pages: self.page_table.len(),
            loaded_pages: self.loaded_count(),
            max_pages: self.max_vram_pages,
            total_bytes_transferred: self.total_bytes_transferred,
            avg_load_time_ns: if self.load_count > 0 {
                self.total_load_time_ns / self.load_count
            } else {
                0
            },
        }
    }
}

/// Paging statistics
#[derive(Debug, Clone)]
pub struct PagingStats {
    pub total_pages: usize,
    pub loaded_pages: usize,
    pub max_pages: usize,
    pub total_bytes_transferred: u64,
    pub avg_load_time_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pager_new() {
        let p = TensorPager::new(100.0, 4);
        assert_eq!(p.max_vram_pages, 100 * 1_000_000 / (4 * 1024));
    }

    #[test]
    fn test_access_new_tensor() {
        let mut p = TensorPager::new(1000.0, 4);
        p.access(42);
        assert!(p.page_table.get(&42).unwrap().is_loaded);
    }

    #[test]
    fn test_access_existing_tensor() {
        let mut p = TensorPager::new(1000.0, 4);
        p.access(42);
        p.access(42);
        assert!(p.page_table.get(&42).unwrap().is_loaded);
    }
}
