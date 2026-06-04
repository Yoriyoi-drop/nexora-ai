use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use memmap2::Mmap;
use ndarray::Array2;
use serde::{Deserialize, Serialize};



/// Metadata weight Oracle yang di-mmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapWeightInfo {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub rows: usize,
    pub cols: usize,
}

/// Konfigurasi Memory-Mapped Oracle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapOracleConfig {
    /// Path ke file weight (.bin atau .safetensors)
    pub weights_path: PathBuf,
    /// Load semua weight ke RAM (false = lazy page loading via mmap)
    pub preload: bool,
    /// Cache decoded weights (reduce deserialization)
    pub enable_cache: bool,
    /// Max cache entries
    pub max_cache_entries: usize,
    /// Ukuran header dalam bytes
    pub header_size: u64,
}

impl Default for MmapOracleConfig {
    fn default() -> Self {
        Self {
            weights_path: PathBuf::from("oracle.bin"),
            preload: false,
            enable_cache: true,
            max_cache_entries: 100,
            header_size: 4096,
        }
    }
}

/// Memory-Mapped Oracle — load weight dari disk via mmap, bukan load seluruh file ke RAM
pub struct MemoryMappedOracle {
    config: MmapOracleConfig,
    mmap: Mmap,
    weight_index: HashMap<String, MmapWeightInfo>,
    cache: Mutex<HashMap<String, Array2<f32>>>,
    is_loaded: AtomicBool,
    total_mapped_bytes: u64,
    total_decoded: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl MemoryMappedOracle {
    /// Buat instance baru dari file weight
    pub fn new(config: MmapOracleConfig) -> Result<Self, MmapError> {
        let path = &config.weights_path;
        if !path.exists() {
            return Err(MmapError::FileNotFound(path.to_string_lossy().to_string()));
        }

        let file = File::open(path).map_err(|e| {
            MmapError::IoError(format!("Failed to open {}: {}", path.display(), e))
        })?;

        let _file_size = file.metadata().map_err(|e| {
            MmapError::IoError(format!("Failed to get metadata: {}", e))
        })?.len();

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                MmapError::IoError(format!("Failed to mmap {}: {}", path.display(), e))
            })?
        };

        let total_mapped_bytes = mmap.len() as u64;

        Ok(Self {
            config,
            mmap,
            weight_index: HashMap::new(),
            cache: Mutex::new(HashMap::new()),
            is_loaded: AtomicBool::new(false),
            total_mapped_bytes,
            total_decoded: 0,
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Bangun index weight dari header file
    pub fn build_index(&mut self, weights: Vec<MmapWeightInfo>) {
        for w in weights {
            self.weight_index.insert(w.name.clone(), w);
        }
        self.is_loaded.store(true, Ordering::SeqCst);
    }

    /// Dapatkan weight sebagai Array2<f32> — lazy load dari mmap
    pub fn get_weight(&mut self, name: &str) -> Result<Array2<f32>, MmapError> {
        if !self.is_loaded.load(Ordering::SeqCst) {
            return Err(MmapError::NotLoaded);
        }

        // Check cache
        if self.config.enable_cache {
            let cache = self.cache.lock().map_err(|e| {
                MmapError::CacheError(e.to_string())
            })?;
            if let Some(cached) = cache.get(name) {
                self.cache_hits += 1;
                return Ok(cached.clone());
            }
        }
        self.cache_misses += 1;

        // Get weight info
        let info = self.weight_index.get(name).ok_or_else(|| {
            MmapError::WeightNotFound(name.to_string())
        })?;

        // Read from mmap
        let start = info.offset as usize;
        let end = start + info.length as usize;
        if end > self.mmap.len() {
            return Err(MmapError::OutOfBounds {
                name: name.to_string(),
                requested: end,
                available: self.mmap.len(),
            });
        }

        let raw_bytes = &self.mmap[start..end];
        let weight = self.bytes_to_array(raw_bytes, info.rows, info.cols)?;
        self.total_decoded += 1;

        // Cache if enabled
        if self.config.enable_cache {
            let mut cache = self.cache.lock().map_err(|e| {
                MmapError::CacheError(e.to_string())
            })?;
            if cache.len() >= self.config.max_cache_entries {
                cache.clear();
            }
            cache.insert(name.to_string(), weight.clone());
        }

        Ok(weight)
    }

    /// Decode bytes → Array2<f32> (little-endian f32)
    fn bytes_to_array(&self, bytes: &[u8], rows: usize, cols: usize) -> Result<Array2<f32>, MmapError> {
        let expected = rows * cols * 4;
        if bytes.len() < expected {
            return Err(MmapError::FormatError(format!(
                "Expected {} bytes for {}x{} weight, got {}",
                expected, rows, cols, bytes.len()
            )));
        }

        let mut data = Vec::with_capacity(rows * cols);
        for chunk in bytes.chunks_exact(4).take(rows * cols) {
            let val = f32::from_le_bytes(
                chunk.try_into().map_err(|_| {
                    MmapError::FormatError("Failed to parse f32 from bytes".into())
                })?,
            );
            data.push(val);
        }

        Array2::from_shape_vec((rows, cols), data).map_err(|e| {
            MmapError::FormatError(format!("Shape error: {}", e))
        })
    }

    /// Pre-decode semua weight ke cache
    pub fn preload_all(&mut self) -> Result<usize, MmapError> {
        let names: Vec<String> = self.weight_index.keys().cloned().collect();
        let mut loaded = 0;
        for name in &names {
            self.get_weight(name)?;
            loaded += 1;
        }
        Ok(loaded)
    }

    /// Bersihkan cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Stats
    pub fn stats(&self) -> MmapStats {
        MmapStats {
            total_mapped_bytes: self.total_mapped_bytes,
            total_weights: self.weight_index.len() as u64,
            total_decoded: self.total_decoded,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            is_loaded: self.is_loaded.load(Ordering::SeqCst),
            cache_size: self.cache.lock().map(|c| c.len()).unwrap_or(0),
            cache_capacity: self.config.max_cache_entries,
        }
    }

    /// Hitung berapa RAM yang dihemat vs load semua
    pub fn memory_savings(&self) -> MemorySavings {
        let resident_pages = self.estimate_resident_pages();
        let decoded_total = self.total_decoded * 4; // approximate
        MemorySavings {
            file_size: self.total_mapped_bytes,
            resident_bytes: resident_pages,
            decoded_bytes: decoded_total,
            savings_bytes: self.total_mapped_bytes.saturating_sub(resident_pages),
        }
    }

    fn estimate_resident_pages(&self) -> u64 {
        let page_size = 4096u64;
        let pages = self.mmap.len() as u64 / page_size;
        pages * page_size / 8
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapStats {
    pub total_mapped_bytes: u64,
    pub total_weights: u64,
    pub total_decoded: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub is_loaded: bool,
    pub cache_size: usize,
    pub cache_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySavings {
    pub file_size: u64,
    pub resident_bytes: u64,
    pub decoded_bytes: u64,
    pub savings_bytes: u64,
}

#[derive(Debug)]
pub enum MmapError {
    FileNotFound(String),
    IoError(String),
    NotLoaded,
    WeightNotFound(String),
    OutOfBounds { name: String, requested: usize, available: usize },
    FormatError(String),
    CacheError(String),
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapError::FileNotFound(p) => write!(f, "Weight file not found: {}", p),
            MmapError::IoError(e) => write!(f, "IO error: {}", e),
            MmapError::NotLoaded => write!(f, "Weight index not loaded"),
            MmapError::WeightNotFound(n) => write!(f, "Weight not found: {}", n),
            MmapError::OutOfBounds { name, requested, available } => {
                write!(f, "Weight {} out of bounds: requested {} bytes, available {}", name, requested, available)
            }
            MmapError::FormatError(e) => write!(f, "Format error: {}", e),
            MmapError::CacheError(e) => write!(f, "Cache error: {}", e),
        }
    }
}

impl std::error::Error for MmapError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    fn create_test_weight_file(path: &Path, rows: usize, cols: usize) -> MmapWeightInfo {
        let mut file = File::create(path).unwrap();
        let mut data = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                data.extend_from_slice(&((i * cols + j) as f32).to_le_bytes());
            }
        }
        file.write_all(&data).unwrap();

        MmapWeightInfo {
            name: "test_weight".into(),
            offset: 0,
            length: data.len() as u64,
            rows,
            cols,
        }
    }

    #[test]
    fn test_mmap_create() {
        let path = std::env::temp_dir().join("oracle_test_weights.bin");
        let info = create_test_weight_file(&path, 4, 8);

        let mut mmap_oracle = MemoryMappedOracle::new(MmapOracleConfig {
            weights_path: path.clone(),
            enable_cache: false,
            ..Default::default()
        })
        .unwrap();

        mmap_oracle.build_index(vec![info]);
        let weight = mmap_oracle.get_weight("test_weight").unwrap();
        assert_eq!(weight.dim(), (4, 8));
        assert!((weight[[0, 0]] - 0.0).abs() < 0.001);
        assert!((weight[[3, 7]] - 31.0).abs() < 0.001);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_file_not_found() {
        let result = MemoryMappedOracle::new(MmapOracleConfig {
            weights_path: PathBuf::from("/nonexistent/path.bin"),
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_not_loaded_error() {
        let path = std::env::temp_dir().join("oracle_unloaded_test.bin");
        let _info = create_test_weight_file(&path, 2, 2);

        let mut mmap_oracle = MemoryMappedOracle::new(MmapOracleConfig {
            weights_path: path.clone(),
            ..Default::default()
        })
        .unwrap();

        let result = mmap_oracle.get_weight("any");
        assert!(result.is_err());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_mmap_cache() {
        let path = std::env::temp_dir().join("oracle_cache_test.bin");
        let info = create_test_weight_file(&path, 2, 4);

        let mut mmap_oracle = MemoryMappedOracle::new(MmapOracleConfig {
            weights_path: path.clone(),
            enable_cache: true,
            max_cache_entries: 10,
            ..Default::default()
        })
        .unwrap();

        mmap_oracle.build_index(vec![info]);
        let _w1 = mmap_oracle.get_weight("test_weight").unwrap();
        let _w2 = mmap_oracle.get_weight("test_weight").unwrap();
        let stats = mmap_oracle.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_memory_savings() {
        let path = std::env::temp_dir().join("oracle_savings_test.bin");
        let info = create_test_weight_file(&path, 64, 128);

        let mut mmap_oracle = MemoryMappedOracle::new(MmapOracleConfig {
            weights_path: path.clone(),
            enable_cache: false,
            ..Default::default()
        })
        .unwrap();

        mmap_oracle.build_index(vec![info]);
        let savings = mmap_oracle.memory_savings();
        assert!(savings.file_size > 0);
        assert!(savings.savings_bytes > 0);

        std::fs::remove_file(path).ok();
    }
}
