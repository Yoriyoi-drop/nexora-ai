use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::scanner::ShardPath;
use crate::types::DataSample;

pub struct DatasetCache {
    token_cache: Arc<RwLock<TokenizerCache>>,
    mmap_registry: Arc<RwLock<HashMap<String, MmapEntry>>>,
    mmap_order: Arc<Mutex<VecDeque<String>>>,
    cache_dir: PathBuf,
    max_entries: usize,
}

struct MmapEntry {
    path: PathBuf,
    size: u64,
}

impl DatasetCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            warn!(
                "Failed to create cache directory {}: {}",
                cache_dir.display(),
                e
            );
        }
        Self {
            token_cache: Arc::new(RwLock::new(TokenizerCache::new(cache_dir.join("tokens")))),
            mmap_registry: Arc::new(RwLock::new(HashMap::new())),
            mmap_order: Arc::new(Mutex::new(VecDeque::new())),
            cache_dir,
            max_entries: 10_000,
        }
    }

    pub fn token_cache(&self) -> Arc<RwLock<TokenizerCache>> {
        self.token_cache.clone()
    }

    pub async fn register_mmap(&self, shard: &ShardPath) {
        let key = shard.path.to_string_lossy().to_string();
        let mut reg = self.mmap_registry.write();
        let mut order = self.mmap_order.lock().await;

        if !reg.contains_key(&key) {
            order.push_back(key.clone());
            while reg.len() >= self.max_entries {
                if let Some(oldest) = order.pop_front() {
                    reg.remove(&oldest);
                }
            }
        }

        reg.insert(
            key,
            MmapEntry {
                path: shard.path.clone(),
                size: shard.size_bytes,
            },
        );
    }

    pub fn is_mmap_registered(&self, path: &str) -> bool {
        self.mmap_registry.read().contains_key(path)
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

#[cfg(feature = "mmap")]
pub fn read_arrow_mmap(
    path: &Path,
    source: crate::types::SourceInfo,
) -> anyhow::Result<Vec<DataSample>> {
    use arrow::array::Array;
    use arrow::ipc::reader::FileReader;
    use memmap2::Mmap;

    let file = std::fs::File::open(path)?;
    // SAFETY: `Mmap::map` requires the underlying file to remain valid for the
    // lifetime of the mapping. `file` is kept alive by the `Mmap` retaining a
    // reference to the `Arc<File>` internally. The mapping is read-only and the
    // file is not modified during the lifetime of the mmap. The file handle is
    // not dropped until the `Mmap` is dropped, ensuring the mapping stays valid.
    let mmap = unsafe { Mmap::map(&file)? };
    let cursor = std::io::Cursor::new(&mmap[..]);
    let reader = FileReader::try_new(cursor, None)?;

    let schema = reader.schema();
    let text_idx = schema
        .index_of("text")
        .or_else(|_| schema.index_of("Text"))?;

    let mut samples = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        let col = batch.column(text_idx);

        // Try StringArray (i32 offsets) first, then LargeStringArray (i64 offsets)
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
            for i in 0..arr.len() {
                let text = if arr.is_null(i) {
                    String::new()
                } else {
                    arr.value(i).to_string()
                };
                samples.push(DataSample {
                    id: uuid::Uuid::new_v4(),
                    text,
                    token_ids: None,
                    metadata: HashMap::new(),
                    source: source.clone(),
                    stats: crate::types::SampleStats::default(),
                    domains: vec![],
                    score: None,
                    curriculum_level: None,
                });
            }
        } else if let Some(arr) = col
            .as_any()
            .downcast_ref::<arrow::array::LargeStringArray>()
        {
            for i in 0..arr.len() {
                let text = if arr.is_null(i) {
                    String::new()
                } else {
                    arr.value(i).to_string()
                };
                samples.push(DataSample {
                    id: uuid::Uuid::new_v4(),
                    text,
                    token_ids: None,
                    metadata: HashMap::new(),
                    source: source.clone(),
                    stats: crate::types::SampleStats::default(),
                    domains: vec![],
                    score: None,
                    curriculum_level: None,
                });
            }
        } else {
            anyhow::bail!("Text column must be String or LargeString type");
        }
    }

    info!(
        "mmap read {} samples from {}",
        samples.len(),
        path.display()
    );
    Ok(samples)
}

#[cfg(not(feature = "mmap"))]
pub fn read_arrow_mmap(
    path: &Path,
    _source: crate::types::SourceInfo,
) -> anyhow::Result<Vec<DataSample>> {
    Err(anyhow::anyhow!(
        "mmap feature not enabled (need feature 'mmap'): {}",
        path.display()
    ))
}

pub struct TokenizerCache {
    cache_dir: PathBuf,
    memory_cache: HashMap<String, Vec<u32>>,
    max_memory_entries: usize,
    access_order: RefCell<VecDeque<String>>,
}

impl TokenizerCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            warn!(
                "Failed to create tokenizer cache dir {}: {}",
                cache_dir.display(),
                e
            );
        }
        Self {
            cache_dir,
            memory_cache: HashMap::new(),
            max_memory_entries: 100_000,
            access_order: RefCell::new(VecDeque::new()),
        }
    }

    pub fn get(&self, text: &str) -> Option<&Vec<u32>> {
        if self.memory_cache.contains_key(text) {
            let mut order = self.access_order.borrow_mut();
            order.retain(|k| k != text);
            order.push_back(text.to_string());
        }
        self.memory_cache.get(text)
    }

    pub fn insert(&mut self, text: String, tokens: Vec<u32>) {
        if self.memory_cache.contains_key(&text) {
            self.access_order.borrow_mut().retain(|k| k != &text);
        } else if self.memory_cache.len() >= self.max_memory_entries {
            if let Some(lru_key) = self.access_order.borrow_mut().pop_front() {
                self.memory_cache.remove(&lru_key);
            }
        }
        self.access_order.borrow_mut().push_back(text.clone());
        self.memory_cache.insert(text, tokens);
    }

    pub fn save_to_disk(&self, shard_name: &str, samples: &[DataSample]) -> anyhow::Result<()> {
        use std::io::Write;

        let path = self.cache_dir.join(format!("{}.tokens.cache", shard_name));
        let mut file = std::fs::File::create(&path)?;

        for sample in samples {
            if let Some(ref tokens) = sample.token_ids {
                let line = tokens
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(file, "{}", line)?;
            }
        }

        debug!("Tokenizer cache saved: {}", path.display());
        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        self.memory_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: uuid::Uuid::new_v4(),
            text: text.into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_dataset_cache_new() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DatasetCache::new(dir.path().join("cache"));
        assert!(cache.cache_dir().exists());
        assert!(!cache.is_mmap_registered("test"));
    }

    #[test]
    fn test_tokenizer_cache_new() {
        let dir = tempfile::tempdir().unwrap();
        let cache = TokenizerCache::new(dir.path().join("tokens"));
        assert_eq!(cache.memory_usage(), 0);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_tokenizer_cache_insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = TokenizerCache::new(dir.path().join("tokens"));
        cache.insert("hello".into(), vec![1, 2, 3]);
        assert_eq!(cache.memory_usage(), 1);
        let result = cache.get("hello");
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_tokenizer_cache_lru_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = TokenizerCache::new(dir.path().join("tokens"));
        cache.max_memory_entries = 2;
        cache.insert("a".into(), vec![1]);
        cache.insert("b".into(), vec![2]);
        assert_eq!(cache.memory_usage(), 2);
        cache.insert("c".into(), vec![3]);
        assert_eq!(cache.memory_usage(), 2);
        // "a" should have been evicted
        assert!(cache.get("a").is_none());
    }

    #[test]
    fn test_tokenizer_cache_save_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = TokenizerCache::new(dir.path().join("tokens"));
        cache.insert("text".into(), vec![10, 20, 30]);

        let samples = vec![sample("text")];
        // The sample doesn't have token_ids set, so save should produce an empty file
        let result = cache.save_to_disk("test_shard", &samples);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dataset_cache_token_cache_method() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DatasetCache::new(dir.path().join("cache"));
        let tc = cache.token_cache();
        assert!(tc.read().memory_usage() == 0);
    }

    #[tokio::test]
    async fn test_register_mmap() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DatasetCache::new(dir.path().join("cache"));
        let shard = ShardPath {
            path: dir.path().join("test.arrow"),
            compression: crate::dataset::compression::Compression::None,
            size_bytes: 100,
            split: "train".into(),
        };
        let path_str = shard.path.to_string_lossy().to_string();
        assert!(!cache.is_mmap_registered(&path_str));
        cache.register_mmap(&shard).await;
        assert!(cache.is_mmap_registered(&path_str));
    }
}
