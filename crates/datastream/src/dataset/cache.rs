use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug, warn};

use crate::types::DataSample;
use super::scanner::ShardPath;

pub struct DatasetCache {
    token_cache: Arc<RwLock<TokenizerCache>>,
    mmap_registry: Arc<RwLock<HashMap<String, MmapEntry>>>,
    cache_dir: PathBuf,
}

struct MmapEntry {
    path: PathBuf,
    size: u64,
}

impl DatasetCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            token_cache: Arc::new(RwLock::new(TokenizerCache::new(cache_dir.join("tokens")))),
            mmap_registry: Arc::new(RwLock::new(HashMap::new())),
            cache_dir,
        }
    }

    pub fn token_cache(&self) -> Arc<RwLock<TokenizerCache>> {
        self.token_cache.clone()
    }

    pub fn register_mmap(&self, shard: &ShardPath) {
        let mut reg = self.mmap_registry.write();
        reg.insert(shard.path.to_string_lossy().to_string(), MmapEntry {
            path: shard.path.clone(),
            size: shard.size_bytes,
        });
    }

    pub fn is_mmap_registered(&self, path: &str) -> bool {
        self.mmap_registry.read().contains_key(path)
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

#[cfg(feature = "mmap")]
pub fn read_arrow_mmap(path: &Path, source: crate::types::SourceInfo) -> anyhow::Result<Vec<DataSample>> {
    use arrow::array::Array;
    use memmap2::Mmap;
    use arrow::ipc::reader::FileReader;
    use arrow::array::AsArray;

    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let cursor = std::io::Cursor::new(&mmap[..]);
    let reader = FileReader::try_new(cursor, None)?;

    let schema = reader.schema();
    let text_idx = schema.index_of("text")
        .or_else(|_| schema.index_of("Text"))?;

    let mut samples = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        let col = batch.column(text_idx);

        // Try StringArray (i32 offsets) first, then LargeStringArray (i64 offsets)
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
            for i in 0..arr.len() {
                let text = if arr.is_null(i) { String::new() } else { arr.value(i).to_string() };
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
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
            for i in 0..arr.len() {
                let text = if arr.is_null(i) { String::new() } else { arr.value(i).to_string() };
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

    info!("mmap read {} samples from {}", samples.len(), path.display());
    Ok(samples)
}

#[cfg(not(feature = "mmap"))]
pub fn read_arrow_mmap(path: &Path, _source: crate::types::SourceInfo) -> anyhow::Result<Vec<DataSample>> {
    Err(anyhow::anyhow!("mmap feature not enabled (need feature 'mmap'): {}", path.display()))
}

pub struct TokenizerCache {
    cache_dir: PathBuf,
    memory_cache: HashMap<String, Vec<u32>>,
    max_memory_entries: usize,
}

impl TokenizerCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        if let Err(e) = std::fs::create_dir_all(&cache_dir) {
            warn!("Failed to create tokenizer cache dir {}: {}", cache_dir.display(), e);
        }
        Self {
            cache_dir,
            memory_cache: HashMap::new(),
            max_memory_entries: 100_000,
        }
    }

    pub fn get(&self, text: &str) -> Option<&Vec<u32>> {
        self.memory_cache.get(text)
    }

    pub fn insert(&mut self, text: String, tokens: Vec<u32>) {
        if self.memory_cache.len() >= self.max_memory_entries {
            self.memory_cache.clear();
        }
        self.memory_cache.insert(text, tokens);
    }

    pub fn save_to_disk(&self, shard_name: &str, samples: &[DataSample]) -> anyhow::Result<()> {
        use std::io::Write;

        let path = self.cache_dir.join(format!("{}.tokens.cache", shard_name));
        let mut file = std::fs::File::create(&path)?;

        for sample in samples {
            if let Some(ref tokens) = sample.token_ids {
                let line = tokens.iter()
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
