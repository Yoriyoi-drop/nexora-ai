use std::path::{Path, PathBuf};
use tracing::warn;

use super::compression::Compression;

#[derive(Debug, Clone)]
pub struct ShardPath {
    pub path: PathBuf,
    pub compression: Compression,
    pub size_bytes: u64,
    pub split: String,
}

pub struct ShardScanner {
    pub recursive: bool,
    pub allowed_extensions: Vec<String>,
}

impl Default for ShardScanner {
    fn default() -> Self {
        Self {
            recursive: true,
            allowed_extensions: vec![
                "arrow".into(),
                "arrow.zst".into(),
                "arrow.zstd".into(),
                "arrow.lz4".into(),
            ],
        }
    }
}

impl ShardScanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn scan(&self, path: &Path) -> Vec<ShardPath> {
        if path.is_dir() {
            self.scan_dir(path)
        } else if path.is_file() {
            self.scan_file(path)
        } else {
            Vec::new()
        }
    }

    fn scan_dir(&self, dir: &Path) -> Vec<ShardPath> {
        let mut shards = Vec::new();

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                warn!("Cannot read directory {}: {}", dir.display(), e);
                return shards;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() && self.recursive {
                shards.extend(self.scan_dir(&path));
            } else if path.is_file() {
                if let Some(s) = self.classify_shard(&path) {
                    shards.push(s);
                }
            }
        }

        shards.sort_by(|a, b| a.path.cmp(&b.path));
        shards
    }

    fn scan_file(&self, path: &Path) -> Vec<ShardPath> {
        self.classify_shard(path).into_iter().collect()
    }

    fn classify_shard(&self, path: &Path) -> Option<ShardPath> {
        let fname = path.file_name()?.to_string_lossy().to_lowercase();
        let ext = path.extension()?.to_string_lossy().to_lowercase();

        if ext == "arrow" {
            let compression = Compression::None;
            let size = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
            let split = detect_split(&fname);
            return Some(ShardPath { path: path.to_path_buf(), compression, size_bytes: size, split });
        }

        let full_name = path.file_name()?.to_string_lossy().to_lowercase();
        if full_name.ends_with(".arrow.zst") || full_name.ends_with(".arrow.zstd") {
            let size = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
            let split = detect_split(&fname);
            return Some(ShardPath { path: path.to_path_buf(), compression: Compression::Zstd, size_bytes: size, split });
        }
        if full_name.ends_with(".arrow.lz4") {
            let size = std::fs::metadata(path).ok().map(|m| m.len()).unwrap_or(0);
            let split = detect_split(&fname);
            return Some(ShardPath { path: path.to_path_buf(), compression: Compression::Lz4, size_bytes: size, split });
        }

        None
    }
}

fn detect_split(fname: &str) -> String {
    let lower = fname.to_lowercase();
    if lower.contains("train") { "train".into() }
    else if lower.contains("val") || lower.contains("validation") { "val".into() }
    else if lower.contains("test") { "test".into() }
    else if lower.contains("reinforcement") || lower.contains("rl") { "reinforcement".into() }
    else if lower.contains("synthetic") || lower.contains("synth") { "synthetic".into() }
    else if lower.contains("instruct") { "instruction".into() }
    else { "train".into() }
}
