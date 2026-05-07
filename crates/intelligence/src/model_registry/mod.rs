//! Model Registry
//! 
//! Centralized model registration and discovery system

pub mod specialists;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelFramework {
    ATQS,      // Compression
    CAFFEINE,  // Multimodal
    SACA,       // Reasoning
    SPARO,      // Alignment
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
    
    /// Register a new model
    pub async fn register_model(&self, metadata: ModelMetadata) -> Result<()> {
        // Check if model already exists
        {
            let models = self.models.read().await;
            if models.contains_key(&metadata.id) {
                warn!("Model {} already registered, updating", metadata.id);
            }
        }
        
        // Register model
        {
            let mut models = self.models.write().await;
            models.insert(metadata.id.clone(), metadata.clone());
        }
        
        // Update framework index
        {
            let mut framework_index = self.framework_index.write().await;
            let model_ids = framework_index.entry(metadata.framework.clone()).or_insert_with(Vec::new);
            if !model_ids.contains(&metadata.id) {
                model_ids.push(metadata.id.clone());
            }
        }
        
        info!("Registered model: {} ({})", metadata.name, metadata.id);
        debug!("Model metadata: {:?}", metadata);
        
        Ok(())
    }
    
    /// Unregister a model
    pub async fn unregister_model(&self, model_id: &str) -> Result<bool> {
        let metadata = {
            let mut models = self.models.write().await;
            models.remove(model_id)
        };
        
        if let Some(metadata) = metadata {
            // Update framework index
            {
                let mut framework_index = self.framework_index.write().await;
                if let Some(model_ids) = framework_index.get_mut(&metadata.framework) {
                    model_ids.retain(|id| id != model_id);
                    if model_ids.is_empty() {
                        framework_index.remove(&metadata.framework);
                    }
                }
            }
            
            info!("Unregistered model: {}", model_id);
            Ok(true)
        } else {
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
            last_registration: models
                .values()
                .map(|m| m.registration_time)
                .max(),
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
