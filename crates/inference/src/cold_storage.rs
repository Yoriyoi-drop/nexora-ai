use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};

use nexora_transformer::KVCacheEntry;

#[derive(Debug, Clone)]
pub struct ColdStorageConfig {
    pub storage_dir: PathBuf,
    pub max_disk_bytes: u64,
    pub ttl_secs: u64,
    pub max_files: usize,
}

impl Default for ColdStorageConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("/tmp/nexora-cold-cache"),
            max_disk_bytes: 10_737_418_240,
            ttl_secs: 3600,
            max_files: 100000,
        }
    }
}

pub struct ColdStorage {
    config: ColdStorageConfig,
    file_sizes: HashMap<PathBuf, u64>,
    total_bytes: u64,
    num_files: usize,
}

impl ColdStorage {
    pub fn new(config: ColdStorageConfig) -> Self {
        let _ = fs::create_dir_all(&config.storage_dir);
        let mut storage = Self { file_sizes: HashMap::new(), total_bytes: 0, num_files: 0, config };
        storage.scan_existing();
        storage
    }

    fn scan_existing(&mut self) {
        let dir = match fs::read_dir(&self.config.storage_dir) {
            Ok(d) => d,
            Err(e) => { warn!("ColdStorage: cannot read dir: {e}"); return; }
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "kcache") {
                if let Ok(meta) = fs::metadata(&path) {
                    let size = meta.len();
                    self.file_sizes.insert(path, size);
                    self.total_bytes += size;
                    self.num_files += 1;
                }
            }
        }
        info!("ColdStorage: scanned {} files, {:.2} MB", self.num_files, self.total_bytes as f64 / 1_048_576.0);
    }

    fn hash_key(tokens: &[u32]) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    fn file_path(&self, key: &str) -> PathBuf {
        self.config.storage_dir.join(format!("{key}.kcache"))
    }

    pub fn save(&mut self, tokens: &[u32], entries: &[KVCacheEntry]) -> Result<(), String> {
        let key = Self::hash_key(tokens);
        let path = self.file_path(&key);
        if path.exists() { return Ok(()); }
        self.evict_if_needed(entries);
        let bytes = serialize_compressed_entries(entries)?;
        if let Some(parent) = path.parent() { let _ = fs::create_dir_all(parent); }
        let mut file = fs::File::create(&path).map_err(|e| format!("create cold file: {e}"))?;
        file.write_all(&bytes).map_err(|e| format!("write cold file: {e}"))?;
        let size = bytes.len() as u64;
        self.file_sizes.insert(path, size);
        self.total_bytes += size;
        self.num_files += 1;
        debug!("ColdStorage: saved {} entries ({:.2} KB)", entries.len(), size as f64 / 1024.0);
        Ok(())
    }

    pub fn load(&self, tokens: &[u32]) -> Option<Vec<KVCacheEntry>> {
        let key = Self::hash_key(tokens);
        let path = self.file_path(&key);
        if !path.exists() { return None; }
        let bytes = fs::read(&path).ok()?;
        let mut entries = deserialize_compressed_entries(&bytes)?;
        for entry in &mut entries { entry.ensure_dequantized(); }
        let _ = touch_file(&path);
        debug!("ColdStorage: loaded {} entries", entries.len());
        Some(entries)
    }

    pub fn evict_stale(&mut self) -> usize {
        let now = SystemTime::now();
        let ttl = Duration::from_secs(self.config.ttl_secs);
        let stale: Vec<PathBuf> = self.file_sizes.keys()
            .filter(|path| fs::metadata(path).and_then(|m| m.modified())
                .map(|t| now.duration_since(t).ok().map_or(false, |d| d > ttl)).unwrap_or(false))
            .cloned().collect();
        let mut evicted = 0;
        for path in stale {
            if let Some(size) = self.file_sizes.remove(&path) {
                self.total_bytes = self.total_bytes.saturating_sub(size);
                self.num_files -= 1;
                let _ = fs::remove_file(&path);
                evicted += 1;
            }
        }
        if evicted > 0 { debug!("ColdStorage: evicted {evicted} stale files"); }
        evicted
    }

    fn evict_lru_internal(&mut self, target_bytes: u64, target_files: usize) -> usize {
        if self.total_bytes <= target_bytes && self.num_files <= target_files { return 0; }
        let mut candidates: Vec<(PathBuf, u64, SystemTime)> = self.file_sizes.keys()
            .filter_map(|path| {
                let meta = fs::metadata(path).ok()?;
                let mtime = meta.modified().ok()?;
                let size = self.file_sizes.get(path).copied().unwrap_or(0);
                Some((path.clone(), size, mtime))
            }).collect();
        candidates.sort_by(|a, b| a.2.cmp(&b.2));
        let mut freed = 0u64;
        let mut evicted = 0;
        for (path, size, _) in candidates {
            if self.total_bytes <= target_bytes && self.num_files <= target_files { break; }
            if let Some(s) = self.file_sizes.remove(&path) { self.total_bytes = self.total_bytes.saturating_sub(s); freed += s; }
            self.num_files -= 1;
            let _ = fs::remove_file(&path);
            evicted += 1;
        }
        if evicted > 0 { debug!("ColdStorage: LRU evicted {evicted} files ({:.2} MB)", freed as f64 / 1_048_576.0); }
        evicted
    }

    fn evict_if_needed(&mut self, entries: &[KVCacheEntry]) {
        let approx = estimate_compressed_size(entries) as u64;
        let target_bytes = self.config.max_disk_bytes.saturating_sub(approx);
        let target_files = self.config.max_files.saturating_sub(1);
        self.evict_stale();
        self.evict_lru_internal(target_bytes, target_files);
    }

    pub fn clear(&mut self) {
        for path in self.file_sizes.keys() { let _ = fs::remove_file(path); }
        self.file_sizes.clear();
        self.total_bytes = 0;
        self.num_files = 0;
    }

    pub fn stats(&self) -> ColdStorageStats {
        ColdStorageStats { num_files: self.num_files, total_bytes: self.total_bytes, storage_dir: self.config.storage_dir.clone() }
    }
}

#[derive(Debug, Clone)]
pub struct ColdStorageStats {
    pub num_files: usize,
    pub total_bytes: u64,
    pub storage_dir: PathBuf,
}

fn serialize_compressed_entries(entries: &[KVCacheEntry]) -> Result<Vec<u8>, String> {
    let mut buf = Vec::with_capacity(entries.len() * 1024);
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        buf.extend_from_slice(&(entry.kv_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(entry.num_kv_heads as u32).to_le_bytes());
        buf.extend_from_slice(&(entry.head_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(entry.compressed_seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(entry.k_compressed.len() as u32).to_le_bytes());
        buf.extend_from_slice(&entry.k_compressed);
        buf.extend_from_slice(&(entry.v_compressed.len() as u32).to_le_bytes());
        buf.extend_from_slice(&entry.v_compressed);
        buf.extend_from_slice(&(entry.k_scales.len() as u32).to_le_bytes());
        for &s in &entry.k_scales { buf.extend_from_slice(&s.to_le_bytes()); }
        buf.extend_from_slice(&(entry.v_scales.len() as u32).to_le_bytes());
        for &s in &entry.v_scales { buf.extend_from_slice(&s.to_le_bytes()); }
    }
    Ok(buf)
}

fn deserialize_compressed_entries(bytes: &[u8]) -> Option<Vec<KVCacheEntry>> {
    let mut pos = 0usize;
    let num_entries = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
    let mut entries = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let kv_dim = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let num_kv_heads = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let head_dim = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let compressed_seq_len = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let k_len = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let k_compressed = bytes.get(pos..pos+k_len)?.to_vec(); pos += k_len;
        let v_len = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let v_compressed = bytes.get(pos..pos+v_len)?.to_vec(); pos += v_len;
        let ks_len = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let k_scales: Vec<f32> = (0..ks_len).map(|i| {
            let start = pos + i * 4;
            f32::from_le_bytes(bytes[start..start+4].try_into().unwrap())
        }).collect(); pos += ks_len * 4;
        let vs_len = u32::from_le_bytes(bytes.get(pos..pos+4)?.try_into().ok()?) as usize; pos += 4;
        let v_scales: Vec<f32> = (0..vs_len).map(|i| {
            let start = pos + i * 4;
            f32::from_le_bytes(bytes[start..start+4].try_into().unwrap())
        }).collect(); pos += vs_len * 4;
        entries.push(KVCacheEntry {
            k: Vec::new(), v: Vec::new(), kv_dim, num_kv_heads, head_dim,
            k_compressed, v_compressed, k_scales, v_scales, compressed_seq_len,
        });
    }
    Some(entries)
}

fn estimate_compressed_size(entries: &[KVCacheEntry]) -> usize {
    4 + entries.iter().map(|e| {
        16 + e.k_compressed.len() + 4 + e.v_compressed.len() + 4 + e.k_scales.len() * 4 + 4 + e.v_scales.len() * 4 + 4
    }).sum::<usize>()
}

fn touch_file(path: &Path) -> Result<(), std::io::Error> {
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let ts = filetime::FileTime::from_unix_time(now, 0);
    filetime::set_file_mtime(path, ts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_entry(kv_dim: usize, seq_len: usize) -> KVCacheEntry {
        let total = kv_dim * seq_len;
        KVCacheEntry {
            k: (0..total).map(|i| i as f32).collect(),
            v: (0..total).map(|i| (i * 2) as f32).collect(),
            kv_dim, num_kv_heads: 2, head_dim: kv_dim / 2,
            k_compressed: Vec::new(), v_compressed: Vec::new(),
            k_scales: Vec::new(), v_scales: Vec::new(), compressed_seq_len: 0,
        }
    }

    fn test_entries() -> Vec<KVCacheEntry> {
        let mut entries = vec![test_entry(8, 4), test_entry(8, 4)];
        for entry in &mut entries { entry.compress(); }
        entries
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let entries = test_entries();
        let bytes = serialize_compressed_entries(&entries).unwrap();
        let loaded = deserialize_compressed_entries(&bytes).unwrap();
        assert_eq!(loaded.len(), entries.len());
        for (i, (orig, e)) in entries.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(orig.kv_dim, e.kv_dim, "entry {i}");
            assert_eq!(orig.k_compressed, e.k_compressed);
            assert_eq!(orig.v_compressed, e.v_compressed);
        }
    }

    #[test]
    fn test_cold_storage_save_load() {
        let dir = TempDir::new().unwrap();
        let cfg = ColdStorageConfig { storage_dir: dir.path().to_path_buf(), max_disk_bytes: 1_000_000_000, ttl_secs: 3600, max_files: 100 };
        let mut storage = ColdStorage::new(cfg);
        let entries = test_entries();
        storage.save(&[1, 2, 3, 4], &entries).unwrap();
        let loaded = storage.load(&[1, 2, 3, 4]);
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.len(), entries.len());
        for e in &loaded { assert!(!e.k.is_empty(), "dequantized"); }
    }

    #[test]
    fn test_cold_storage_miss() {
        let dir = TempDir::new().unwrap();
        let cfg = ColdStorageConfig { storage_dir: dir.path().to_path_buf(), ..Default::default() };
        let storage = ColdStorage::new(cfg);
        assert!(storage.load(&[99, 98]).is_none());
    }

    #[test]
    fn test_cold_storage_lru_eviction() {
        let dir = TempDir::new().unwrap();
        let cfg = ColdStorageConfig { storage_dir: dir.path().to_path_buf(), max_disk_bytes: 999_999_999, ttl_secs: 3600, max_files: 2 };
        let mut storage = ColdStorage::new(cfg);
        let entries = test_entries();
        storage.save(&[1, 2], &entries).unwrap();
        storage.save(&[3, 4], &entries).unwrap();
        storage.save(&[5, 6], &entries).unwrap();
        assert!(storage.load(&[1, 2]).is_none(), "oldest evicted");
        assert!(storage.load(&[3, 4]).is_some(), "exists");
        assert!(storage.load(&[5, 6]).is_some(), "exists");
    }
}
