use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub name: String,
    pub version: String,
    pub format: String,
    pub compression: Option<String>,
    pub total_samples: u64,
    pub total_shards: usize,
    pub shards: Vec<ShardInfo>,
    pub features: Vec<String>,
    pub schema: HashMap<String, String>,
    pub created_at: String,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub path: String,
    pub split: String,
    pub samples: u64,
    pub size_bytes: u64,
    pub compression: Option<String>,
    pub checksum: Option<String>,
}

impl DatasetManifest {
    pub fn from_path(path: &Path) -> Result<Self, ManifestError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ManifestError::Io(e.to_string()))?;
        serde_json::from_str(&content)
            .map_err(|e| ManifestError::Parse(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), ManifestError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ManifestError::Serialize(e.to_string()))?;
        std::fs::write(path, content)
            .map_err(|e| ManifestError::Io(e.to_string()))
    }

    pub fn shards_for_split(&self, split: &str) -> Vec<&ShardInfo> {
        self.shards.iter().filter(|s| s.split == split).collect()
    }

    pub fn total_for_split(&self, split: &str) -> u64 {
        self.shards.iter()
            .filter(|s| s.split == split)
            .map(|s| s.samples)
            .sum()
    }
}

#[derive(Debug)]
pub enum ManifestError {
    Io(String),
    Parse(String),
    Serialize(String),
    NotFound(String),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(msg) => write!(f, "IO error: {}", msg),
            ManifestError::Parse(msg) => write!(f, "Parse error: {}", msg),
            ManifestError::Serialize(msg) => write!(f, "Serialize error: {}", msg),
            ManifestError::NotFound(msg) => write!(f, "Not found: {}", msg),
        }
    }
}

impl std::error::Error for ManifestError {}
