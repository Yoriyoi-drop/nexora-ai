use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::manifest::{DatasetManifest, ManifestError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub total_samples: u64,
    pub format: String,
    pub compression: Option<String>,
    pub created_at: String,
}

pub struct DatasetRegistry {
    datasets: HashMap<String, Vec<DatasetInfo>>,
    registry_path: Option<PathBuf>,
}

impl DatasetRegistry {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            registry_path: None,
        }
    }

    pub fn with_registry_path(mut self, path: PathBuf) -> Self {
        self.registry_path = Some(path);
        self
    }

    pub fn register(&mut self, path: &std::path::Path) -> Result<(), RegistryError> {
        let manifest_path = path.join("manifest.json");
        if !manifest_path.exists() {
            return Err(RegistryError::NoManifest(manifest_path));
        }

        let manifest = DatasetManifest::from_path(&manifest_path)
            .map_err(|e| RegistryError::Manifest(e.to_string()))?;

        let info = DatasetInfo {
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            path: path.to_path_buf(),
            total_samples: manifest.total_samples,
            format: manifest.format.clone(),
            compression: manifest.compression.clone(),
            created_at: manifest.created_at.clone(),
        };

        self.datasets
            .entry(manifest.name.clone())
            .or_default()
            .push(info);

        info!("Dataset registered: {} v{} ({} samples)", manifest.name, manifest.version, manifest.total_samples);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Vec<DatasetInfo>> {
        self.datasets.get(name)
    }

    pub fn get_version(&self, name: &str, version: &str) -> Option<&DatasetInfo> {
        self.datasets.get(name)?.iter().find(|d| d.version == version)
    }

    pub fn latest(&self, name: &str) -> Option<&DatasetInfo> {
        self.datasets.get(name)?.last()
    }

    pub fn list_datasets(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }

    pub fn save_registry(&self) -> Result<(), RegistryError> {
        let path = match &self.registry_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };

        let content = serde_json::to_string_pretty(&self.datasets)
            .map_err(|e| RegistryError::Serialize(e.to_string()))?;
        std::fs::write(&path, content)
            .map_err(|e| RegistryError::Io(e.to_string()))?;
        info!("Registry saved: {}", path.display());
        Ok(())
    }

    pub fn load_registry(path: &std::path::Path) -> Result<Self, RegistryError> {
        if !path.exists() {
            return Ok(Self::new().with_registry_path(path.to_path_buf()));
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| RegistryError::Io(e.to_string()))?;
        let datasets: HashMap<String, Vec<DatasetInfo>> = serde_json::from_str(&content)
            .map_err(|e| RegistryError::Parse(e.to_string()))?;

        info!("Registry loaded: {} datasets", datasets.len());
        Ok(Self {
            datasets,
            registry_path: Some(path.to_path_buf()),
        })
    }
}

#[derive(Debug)]
pub enum RegistryError {
    NoManifest(PathBuf),
    Manifest(String),
    Io(String),
    Parse(String),
    Serialize(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::NoManifest(p) => write!(f, "No manifest.json at {}", p.display()),
            RegistryError::Manifest(msg) => write!(f, "Manifest error: {}", msg),
            RegistryError::Io(msg) => write!(f, "Registry IO: {}", msg),
            RegistryError::Parse(msg) => write!(f, "Registry parse: {}", msg),
            RegistryError::Serialize(msg) => write!(f, "Registry serialize: {}", msg),
        }
    }
}

impl std::error::Error for RegistryError {}
