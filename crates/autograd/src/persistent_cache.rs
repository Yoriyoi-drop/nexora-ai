use std::path::PathBuf;

const CACHE_VERSION: u64 = 1;

/// Manages disk-backed persistent wgpu pipeline cache.
#[derive(Debug)]
pub struct PipelineDiskCache {
    cache_dir: PathBuf,
}

impl PipelineDiskCache {
    /// Create disk cache in XDG-compatible cache directory.
    pub fn new() -> Self {
        let cache_dir = default_cache_dir();
        let _ = std::fs::create_dir_all(&cache_dir);
        Self { cache_dir }
    }

    /// Create disk cache from explicit directory path.
    pub fn with_dir(cache_dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&cache_dir);
        Self { cache_dir }
    }

    /// Full path for a given cache key.
    fn cache_path(&self, key: u64) -> PathBuf {
        self.cache_dir.join(format!("pipeline_v{CACHE_VERSION}_{:016x}.bin", key))
    }

    /// Load cached pipeline data from disk.
    /// Returns `None` if no cache exists or data is empty.
    pub fn load(&self, key: u64) -> Option<Vec<u8>> {
        let path = self.cache_path(key);
        if path.exists() {
            match std::fs::read(&path) {
                Ok(data) if !data.is_empty() => Some(data),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Save pipeline cache data to disk.
    /// Does nothing if data is empty or write fails.
    pub fn save(&self, key: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        // Write to temp file first, then rename for atomicity
        let path = self.cache_path(key);
        let tmp = path.with_extension("bin.tmp");
        if let Err(e) = std::fs::write(&tmp, data) {
            tracing::warn!("Failed to write pipeline cache: {}", e);
            return;
        }
        let _ = std::fs::rename(&tmp, &path);
    }
}

impl Default for PipelineDiskCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve XDG/pipeline-cache dir.
fn default_cache_dir() -> PathBuf {
    let base = if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(dir)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        PathBuf::from(".")
    };
    base.join("nexora").join("pipeline-cache")
}
