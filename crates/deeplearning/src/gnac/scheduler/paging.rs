use std::collections::{HashMap, VecDeque};

/// Tensor paging — memori virtual untuk tensor
/// Tensor di-swap ke host memory saat tidak digunakan, di-load balik saat diperlukan
pub struct TensorPager {
    page_size: usize,
    max_vram_pages: usize,
    access_history: VecDeque<u64>,
    page_table: HashMap<u64, PageEntry>,
}

#[derive(Debug, Clone)]
struct PageEntry {
    tensor_id: u64,
    is_loaded: bool,
    last_access: std::time::Instant,
    size_bytes: usize,
}

impl TensorPager {
    pub fn new(max_vram_mb: f64, page_size_kb: usize) -> Self {
        let max_vram_bytes = (max_vram_mb * 1_000_000.0) as usize;
        TensorPager {
            page_size: page_size_kb * 1024,
            max_vram_pages: max_vram_bytes / (page_size_kb * 1024),
            access_history: VecDeque::new(),
            page_table: HashMap::new(),
        }
    }

    /// Request akses ke tensor (load dari host jika perlu)
    pub fn access(&mut self, tensor_id: u64) {
        let now = std::time::Instant::now();
        let entry = self.page_table.entry(tensor_id).or_insert(PageEntry {
            tensor_id,
            is_loaded: false,
            last_access: now,
            size_bytes: self.page_size,
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

    fn load_page(&mut self, tensor_id: u64) {
        // Evict pages if VRAM penuh
        let loaded_count = self.page_table.values().filter(|e| e.is_loaded).count();
        if loaded_count >= self.max_vram_pages {
            self.evict_lru();
        }

        if let Some(entry) = self.page_table.get_mut(&tensor_id) {
            entry.is_loaded = true;
        }
    }

    fn evict_lru(&mut self) {
        // Evict page yang paling lama tidak diakses
        if let Some(oldest_id) = self.access_history.pop_front() {
            if let Some(entry) = self.page_table.get_mut(&oldest_id) {
                entry.is_loaded = false;
            }
        }
    }
}
