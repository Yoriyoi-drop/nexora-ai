use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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
        let content =
            std::fs::read_to_string(path).map_err(|e| ManifestError::Io(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| ManifestError::Parse(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), ManifestError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ManifestError::Serialize(e.to_string()))?;
        std::fs::write(path, content).map_err(|e| ManifestError::Io(e.to_string()))
    }

    pub fn shards_for_split(&self, split: &str) -> Vec<&ShardInfo> {
        self.shards.iter().filter(|s| s.split == split).collect()
    }

    pub fn total_for_split(&self, split: &str) -> u64 {
        self.shards
            .iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_from_nonexistent_path() {
        let result = DatasetManifest::from_path(Path::new("/nonexistent/manifest.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_shard_info_default() {
        let shard = ShardInfo {
            path: "data.arrow".into(),
            split: "train".into(),
            samples: 1000,
            size_bytes: 50000,
            compression: None,
            checksum: None,
        };
        assert_eq!(shard.split, "train");
        assert_eq!(shard.samples, 1000);
    }

    #[test]
    fn test_shards_for_split() {
        let manifest = DatasetManifest {
            name: "test".into(),
            version: "1.0".into(),
            format: "arrow".into(),
            compression: None,
            total_samples: 3000,
            total_shards: 3,
            shards: vec![
                ShardInfo {
                    path: "train1.arrow".into(),
                    split: "train".into(),
                    samples: 1000,
                    size_bytes: 50000,
                    compression: None,
                    checksum: None,
                },
                ShardInfo {
                    path: "val1.arrow".into(),
                    split: "val".into(),
                    samples: 500,
                    size_bytes: 25000,
                    compression: None,
                    checksum: None,
                },
                ShardInfo {
                    path: "train2.arrow".into(),
                    split: "train".into(),
                    samples: 2000,
                    size_bytes: 100000,
                    compression: None,
                    checksum: None,
                },
            ],
            features: vec![],
            schema: std::collections::HashMap::new(),
            created_at: "now".into(),
            checksum: None,
        };

        let train_shards = manifest.shards_for_split("train");
        assert_eq!(train_shards.len(), 2);
        assert_eq!(manifest.total_for_split("train"), 3000);

        let val_shards = manifest.shards_for_split("val");
        assert_eq!(val_shards.len(), 1);
        assert_eq!(manifest.total_for_split("val"), 500);
    }

    #[test]
    fn test_manifest_display_errors() {
        let io_err = ManifestError::Io("disk full".into());
        assert_eq!(format!("{}", io_err), "IO error: disk full");

        let parse_err = ManifestError::Parse("invalid json".into());
        assert_eq!(format!("{}", parse_err), "Parse error: invalid json");
    }
}
