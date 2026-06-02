//! Model Registry
//!
//! Centralized model registration and discovery system

pub mod specialists;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Model metadata for registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub framework: ModelFramework,
    pub version: String,
    pub capabilities: Vec<String>,
    pub max_input_length: usize,
    pub max_output_length: usize,
    pub supported_formats: Vec<String>,
    pub resource_requirements: ResourceRequirements,
    pub registration_time: std::time::SystemTime,
}

/// Model framework types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelFramework {
    ATQS,     // Compression
    CAFFEINE, // Multimodal
    SACA,     // Reasoning
    SPARO,    // Alignment
    Custom(String),
}

/// Resource requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_memory_mb: usize,
    pub recommended_memory_mb: usize,
    pub min_cpu_cores: usize,
    pub recommended_cpu_cores: usize,
    pub requires_gpu: bool,
    pub gpu_memory_mb: Option<usize>,
}

/// Model registry for centralized model management
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, ModelMetadata>>>,
    framework_index: Arc<RwLock<HashMap<ModelFramework, Vec<String>>>>,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            framework_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new model — atomic dual-index update eliminates TOCTOU race.
    pub async fn register_model(&self, metadata: ModelMetadata) -> Result<()> {
        let mut models = self.models.write().await;
        if models.contains_key(&metadata.id) {
            warn!("Model {} already registered, updating", metadata.id);
        }
        models.insert(metadata.id.clone(), metadata.clone());

        let mut framework_index = self.framework_index.write().await;
        let model_ids = framework_index
            .entry(metadata.framework.clone())
            .or_insert_with(Vec::new);
        if !model_ids.contains(&metadata.id) {
            model_ids.push(metadata.id.clone());
        }
        drop(framework_index);
        drop(models);

        info!("Registered model: {} ({})", metadata.name, metadata.id);
        debug!("Model metadata: {:?}", metadata);

        Ok(())
    }

    /// Unregister a model — atomic dual-index update eliminates TOCTOU race.
    pub async fn unregister_model(&self, model_id: &str) -> Result<bool> {
        let mut models = self.models.write().await;
        let metadata = models.remove(model_id);

        if let Some(metadata) = metadata {
            let mut framework_index = self.framework_index.write().await;
            if let Some(model_ids) = framework_index.get_mut(&metadata.framework) {
                model_ids.retain(|id| id != model_id);
                if model_ids.is_empty() {
                    framework_index.remove(&metadata.framework);
                }
            }
            drop(framework_index);
            drop(models);

            info!("Unregistered model: {}", model_id);
            Ok(true)
        } else {
            drop(models);
            warn!("Model {} not found for unregistration", model_id);
            Ok(false)
        }
    }

    /// Get model metadata by ID
    pub async fn get_model(&self, model_id: &str) -> Option<ModelMetadata> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// List all models
    pub async fn list_models(&self) -> Vec<ModelMetadata> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// List models by framework
    pub async fn list_models_by_framework(&self, framework: &ModelFramework) -> Vec<ModelMetadata> {
        let models = self.models.read().await;
        let framework_index = self.framework_index.read().await;

        if let Some(model_ids) = framework_index.get(framework) {
            model_ids
                .iter()
                .filter_map(|id| models.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Search models by capability
    pub async fn search_by_capability(&self, capability: &str) -> Vec<ModelMetadata> {
        let models = self.models.read().await;
        models
            .values()
            .filter(|model| model.capabilities.contains(&capability.to_string()))
            .cloned()
            .collect()
    }

    /// Get models that can handle specific input format
    pub async fn get_models_for_format(&self, format: &str) -> Vec<ModelMetadata> {
        let models = self.models.read().await;
        models
            .values()
            .filter(|model| model.supported_formats.contains(&format.to_string()))
            .cloned()
            .collect()
    }

    /// Get registry statistics
    pub async fn get_statistics(&self) -> RegistryStatistics {
        let models = self.models.read().await;
        let framework_index = self.framework_index.read().await;

        let mut framework_counts = HashMap::new();
        for (framework, model_ids) in framework_index.iter() {
            framework_counts.insert(framework.clone(), model_ids.len());
        }

        RegistryStatistics {
            total_models: models.len(),
            framework_counts,
            last_registration: models.values().map(|m| m.registration_time).max(),
        }
    }

    /// Validate model metadata
    pub async fn validate_model(&self, metadata: &ModelMetadata) -> Result<()> {
        if metadata.id.is_empty() {
            return Err(anyhow::anyhow!("Model ID cannot be empty"));
        }

        if metadata.name.is_empty() {
            return Err(anyhow::anyhow!("Model name cannot be empty"));
        }

        if metadata.version.is_empty() {
            return Err(anyhow::anyhow!("Model version cannot be empty"));
        }

        if metadata.max_input_length == 0 {
            return Err(anyhow::anyhow!("Max input length must be > 0"));
        }

        if metadata.max_output_length == 0 {
            return Err(anyhow::anyhow!("Max output length must be > 0"));
        }

        if metadata.resource_requirements.min_memory_mb == 0 {
            return Err(anyhow::anyhow!("Min memory must be > 0"));
        }

        Ok(())
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    pub total_models: usize,
    pub framework_counts: HashMap<ModelFramework, usize>,
    pub last_registration: Option<std::time::SystemTime>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            min_memory_mb: 512,
            recommended_memory_mb: 1024,
            min_cpu_cores: 1,
            recommended_cpu_cores: 2,
            requires_gpu: false,
            gpu_memory_mb: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata(id: &str) -> ModelMetadata {
        ModelMetadata {
            id: id.to_string(),
            name: format!("model-{}", id),
            framework: ModelFramework::SACA,
            version: "1.0.0".to_string(),
            capabilities: vec!["reasoning".to_string(), "code".to_string()],
            max_input_length: 4096,
            max_output_length: 1024,
            supported_formats: vec!["text".to_string()],
            resource_requirements: ResourceRequirements::default(),
            registration_time: std::time::SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_register_model() {
        let reg = ModelRegistry::new();
        let meta = sample_metadata("test-1");
        let result = reg.register_model(meta).await;
        assert!(result.is_ok());
        assert_eq!(reg.list_models().await.len(), 1);
    }

    #[tokio::test]
    async fn test_register_duplicate_updates() {
        let reg = ModelRegistry::new();
        let meta = sample_metadata("dup-1");
        reg.register_model(meta.clone()).await.unwrap();
        let mut updated = meta;
        updated.name = "model-dup-1-v2".to_string();
        updated.version = "2.0.0".to_string();
        let result = reg.register_model(updated.clone()).await;
        assert!(result.is_ok());
        let stored = reg.get_model("dup-1").await.unwrap();
        assert_eq!(stored.version, "2.0.0");
        assert_eq!(reg.list_models().await.len(), 1);
    }

    #[tokio::test]
    async fn test_unregister_model() {
        let reg = ModelRegistry::new();
        let meta = sample_metadata("to-unreg");
        reg.register_model(meta).await.unwrap();
        let result = reg.unregister_model("to-unreg").await.unwrap();
        assert!(result);
        assert!(reg.get_model("to-unreg").await.is_none());
        assert_eq!(reg.list_models().await.len(), 0);
    }

    #[tokio::test]
    async fn test_unregister_nonexistent() {
        let reg = ModelRegistry::new();
        let result = reg.unregister_model("ghost").await.unwrap();
        assert!(!result);
    }

    #[tokio::test]
    async fn test_get_model_not_found() {
        let reg = ModelRegistry::new();
        assert!(reg.get_model("nope").await.is_none());
    }

    #[tokio::test]
    async fn test_list_models_empty() {
        let reg = ModelRegistry::new();
        assert!(reg.list_models().await.is_empty());
    }

    #[tokio::test]
    async fn test_list_models_by_framework() {
        let reg = ModelRegistry::new();
        let saca = sample_metadata("saca-1");
        reg.register_model(saca).await.unwrap();
        let atqs_meta = ModelMetadata {
            framework: ModelFramework::ATQS,
            ..sample_metadata("atqs-1")
        };
        reg.register_model(atqs_meta).await.unwrap();
        let saca_models = reg.list_models_by_framework(&ModelFramework::SACA).await;
        assert_eq!(saca_models.len(), 1);
        let atqs_models = reg.list_models_by_framework(&ModelFramework::ATQS).await;
        assert_eq!(atqs_models.len(), 1);
        let caffeine_models = reg.list_models_by_framework(&ModelFramework::CAFFEINE).await;
        assert!(caffeine_models.is_empty());
    }

    #[tokio::test]
    async fn test_search_by_capability() {
        let reg = ModelRegistry::new();
        let meta = sample_metadata("cap-1");
        reg.register_model(meta).await.unwrap();
        let results = reg.search_by_capability("code").await;
        assert_eq!(results.len(), 1);
        let results = reg.search_by_capability("vision").await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_models_for_format() {
        let reg = ModelRegistry::new();
        let meta = sample_metadata("fmt-1");
        reg.register_model(meta).await.unwrap();
        let results = reg.get_models_for_format("text").await;
        assert_eq!(results.len(), 1);
        let results = reg.get_models_for_format("image").await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_statistics_empty() {
        let reg = ModelRegistry::new();
        let stats = reg.get_statistics().await;
        assert_eq!(stats.total_models, 0);
        assert!(stats.framework_counts.is_empty());
        assert!(stats.last_registration.is_none());
    }

    #[tokio::test]
    async fn test_get_statistics_after_register() {
        let reg = ModelRegistry::new();
        reg.register_model(sample_metadata("stat-1")).await.unwrap();
        let stats = reg.get_statistics().await;
        assert_eq!(stats.total_models, 1);
        assert_eq!(*stats.framework_counts.get(&ModelFramework::SACA).unwrap(), 1);
        assert!(stats.last_registration.is_some());
    }

    #[tokio::test]
    async fn test_validate_model_valid() {
        let reg = ModelRegistry::new();
        let meta = sample_metadata("valid");
        assert!(reg.validate_model(&meta).await.is_ok());
    }

    #[tokio::test]
    async fn test_validate_model_empty_id() {
        let reg = ModelRegistry::new();
        let meta = ModelMetadata {
            id: "".to_string(),
            ..sample_metadata("eid")
        };
        assert!(reg.validate_model(&meta).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_model_empty_name() {
        let reg = ModelRegistry::new();
        let meta = ModelMetadata {
            name: "".to_string(),
            ..sample_metadata("enm")
        };
        assert!(reg.validate_model(&meta).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_model_zero_input() {
        let reg = ModelRegistry::new();
        let meta = ModelMetadata {
            max_input_length: 0,
            ..sample_metadata("zin")
        };
        assert!(reg.validate_model(&meta).await.is_err());
    }

    #[tokio::test]
    async fn test_framework_index_consistency_after_unregister() {
        let reg = ModelRegistry::new();
        reg.register_model(sample_metadata("fc-1")).await.unwrap();
        reg.unregister_model("fc-1").await.unwrap();
        let saca_models = reg.list_models_by_framework(&ModelFramework::SACA).await;
        assert!(saca_models.is_empty());
        let stats = reg.get_statistics().await;
        assert_eq!(stats.total_models, 0);
    }

    #[tokio::test]
    async fn test_multiple_frameworks() {
        let reg = ModelRegistry::new();
        let frameworks = [
            (ModelFramework::SACA, "m-saca"),
            (ModelFramework::ATQS, "m-atqs"),
            (ModelFramework::CAFFEINE, "m-caff"),
            (ModelFramework::SPARO, "m-sparo"),
            (ModelFramework::Custom("mine".into()), "m-cust"),
        ];
        for (fw, id) in &frameworks {
            let meta = ModelMetadata {
                framework: fw.clone(),
                ..sample_metadata(id)
            };
            reg.register_model(meta).await.unwrap();
        }
        assert_eq!(reg.list_models().await.len(), frameworks.len());
        for (fw, _) in &frameworks {
            let models = reg.list_models_by_framework(fw).await;
            assert_eq!(models.len(), 1, "framework {:?} should have 1 model", fw);
        }
    }

    #[tokio::test]
    async fn test_concurrent_register_and_unregister() {
        let reg = std::sync::Arc::new(ModelRegistry::new());
        let mut handles = Vec::new();
        for i in 0..10 {
            let r = reg.clone();
            handles.push(tokio::spawn(async move {
                let meta = sample_metadata(&format!("conc-{}", i));
                r.register_model(meta).await.unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(reg.list_models().await.len(), 10);
    }

    #[tokio::test]
    async fn test_register_and_unregister_same_model_twice() {
        let reg = ModelRegistry::new();
        reg.register_model(sample_metadata("twice")).await.unwrap();
        assert!(reg.unregister_model("twice").await.unwrap());
        assert!(reg.register_model(sample_metadata("twice")).await.is_ok());
        assert_eq!(reg.list_models().await.len(), 1);
    }

    #[tokio::test]
    async fn test_framework_index_removed_when_empty() {
        let reg = ModelRegistry::new();
        reg.register_model(sample_metadata("frem")).await.unwrap();
        reg.unregister_model("frem").await.unwrap();
        let stats = reg.get_statistics().await;
        assert!(stats.framework_counts.is_empty());
    }

    #[tokio::test]
    async fn test_custom_framework() {
        let reg = ModelRegistry::new();
        let meta = ModelMetadata {
            framework: ModelFramework::Custom("my-fw".into()),
            ..sample_metadata("custom-1")
        };
        reg.register_model(meta).await.unwrap();
        let models = reg
            .list_models_by_framework(&ModelFramework::Custom("my-fw".into()))
            .await;
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_resource_requirements_defaults() {
        let req = ResourceRequirements::default();
        assert_eq!(req.min_memory_mb, 512);
        assert_eq!(req.recommended_memory_mb, 1024);
        assert!(!req.requires_gpu);
    }

    #[test]
    fn test_model_registry_default() {
        let reg = ModelRegistry::default();
        assert_eq!(reg.models.try_read().unwrap().len(), 0);
    }

    #[test]
    fn test_model_framework_debug_clone() {
        let fw = ModelFramework::SACA;
        let fw2 = fw.clone();
        assert_eq!(format!("{:?}", fw), format!("{:?}", fw2));
    }
}
