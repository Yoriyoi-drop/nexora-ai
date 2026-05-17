//! NXR Model Registry
//! 
//! Central registry for all NXR models with discovery and management

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::{
    model_identity::{NxrModelId, ModelMeta, ModelTier},
    capability_spec::CapabilityVector,
    base_model::NxrModel,
    model_config::NxrModelConfig,
    safety_gate::CapabilityLock,
};

/// A single atomic registry entry holding all components for one model.
#[derive(Clone)]
struct RegistryEntry {
    model: Option<Arc<dyn NxrModel<Config = serde_json::Value, Metrics = serde_json::Value, State = serde_json::Value>>>,
    metadata: ModelMeta,
    capabilities: CapabilityVector,
    config: NxrModelConfig,
}

/// Model registry for managing all NXR models
pub struct NxrModelRegistry {
    entries: Arc<RwLock<HashMap<NxrModelId, RegistryEntry>>>,
}

impl fmt::Debug for NxrModelRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NxrModelRegistry")
            .field("registered_count", &self.entries.try_read().map(|m| m.len()).unwrap_or(0))
            .finish()
    }
}

impl NxrModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a model (with safety gate enforcement)
    pub async fn register_model(
        &self,
        model_id: NxrModelId,
        model: Arc<dyn NxrModel<Config = serde_json::Value, Metrics = serde_json::Value, State = serde_json::Value>>,
        meta: ModelMeta,
        model_capabilities: CapabilityVector,
        config: NxrModelConfig,
    ) -> Result<(), RegistryError> {
        // Validate model identity matches
        if model.identity().id != meta.id {
            return Err(RegistryError::IdentityMismatch);
        }

        // Enforce safety capability locks at registration
        let cap_lock = CapabilityLock::new();
        let mut caps = model_capabilities;
        cap_lock.enforce(&mut caps).await.map_err(|e| RegistryError::Validation(e.to_string()))?;

        // Atomic insert: all four components under a single write lock
        let mut entries = self.entries.write().await;
        if entries.contains_key(&model_id) {
            return Err(RegistryError::AlreadyRegistered(model_id));
        }
        entries.insert(model_id, RegistryEntry {
            model: Some(model),
            metadata: meta,
            capabilities: caps,
            config,
        });

        Ok(())
    }

    /// Register model metadata and configuration without a model instance.
    /// Prefer `register_model` when a model instance is available.
    pub async fn register_components(
        &self,
        model_id: NxrModelId,
        meta: ModelMeta,
        model_capabilities: CapabilityVector,
        config: NxrModelConfig,
    ) -> Result<(), RegistryError> {
        // Enforce safety capability locks at registration
        let cap_lock = CapabilityLock::new();
        let mut caps = model_capabilities;
        cap_lock.enforce(&mut caps).await.map_err(|e| RegistryError::Validation(e.to_string()))?;

        let mut entries = self.entries.write().await;
        if entries.contains_key(&model_id) {
            return Err(RegistryError::AlreadyRegistered(model_id));
        }
        entries.insert(model_id, RegistryEntry {
            model: None,
            metadata: meta,
            capabilities: caps,
            config,
        });

        Ok(())
    }

    /// Unregister a model
    pub async fn unregister_model(&self, model_id: &NxrModelId) -> Result<(), RegistryError> {
        let mut entries = self.entries.write().await;
        if entries.remove(model_id).is_none() {
            return Err(RegistryError::NotFound(*model_id));
        }
        Ok(())
    }

    /// Get model by ID
    pub async fn get_model(
        &self,
        model_id: &NxrModelId,
    ) -> Result<Arc<dyn NxrModel<Config = serde_json::Value, Metrics = serde_json::Value, State = serde_json::Value>>, RegistryError> {
        let entries = self.entries.read().await;
        entries
            .get(model_id)
            .and_then(|e| e.model.clone())
            .ok_or(RegistryError::NotFound(*model_id))
    }

    /// Get model metadata
    pub async fn get_metadata(&self, model_id: &NxrModelId) -> Result<ModelMeta, RegistryError> {
        let entries = self.entries.read().await;
        entries
            .get(model_id)
            .map(|e| e.metadata.clone())
            .ok_or(RegistryError::NotFound(*model_id))
    }

    /// Get model capabilities
    pub async fn get_capabilities(&self, model_id: &NxrModelId) -> Result<CapabilityVector, RegistryError> {
        let entries = self.entries.read().await;
        entries
            .get(model_id)
            .map(|e| e.capabilities.clone())
            .ok_or(RegistryError::NotFound(*model_id))
    }

    /// Get model configuration
    pub async fn get_configuration(&self, model_id: &NxrModelId) -> Result<NxrModelConfig, RegistryError> {
        let entries = self.entries.read().await;
        entries
            .get(model_id)
            .map(|e| e.config.clone())
            .ok_or(RegistryError::NotFound(*model_id))
    }

    /// List all registered models
    pub async fn list_models(&self) -> Vec<NxrModelId> {
        let entries = self.entries.read().await;
        entries.keys().cloned().collect()
    }

    /// List models by tier
    pub async fn list_models_by_tier(&self, tier: ModelTier) -> Vec<NxrModelId> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|(_, e)| e.metadata.tier == tier)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find models by capability
    pub async fn find_models_by_capability(
        &self,
        domain: &super::capability_spec::CapabilityDomain,
        min_level: super::capability_spec::CapabilityLevel,
    ) -> Vec<NxrModelId> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|(_, e)| e.capabilities.has_capability(domain, min_level))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get best model for task
    pub async fn get_best_model_for_task(&self, task: &Task) -> Result<NxrModelId, RegistryError> {
        let entries = self.entries.read().await;
        if entries.is_empty() {
            return Err(RegistryError::NoModelsAvailable);
        }

        // Score each model for this task
        let mut scored_models: Vec<(NxrModelId, f32)> = Vec::new();
        for (model_id, entry) in entries.iter() {
            let score = self.score_model_for_task(&entry.capabilities, task);
            scored_models.push((*model_id, score));
        }

        // Sort by score (descending)
        scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return the best scoring model
        scored_models
            .into_iter()
            .next()
            .map(|(id, _)| id)
            .ok_or(RegistryError::NoSuitableModel)
    }

    /// Score model for specific task
    fn score_model_for_task(&self, capabilities: &CapabilityVector, task: &Task) -> f32 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        for (domain, required_level) in &task.required_capabilities {
            let weight = task.capability_weights.get(domain).unwrap_or(&1.0);
            total_weight += weight;

            if let Some(capability) = capabilities.get_capability(domain) {
                if capability.level >= *required_level {
                    // Full score if meets requirement, partial if close
                    let level_diff = capability.level as u8 as i16 - *required_level as u8 as i16;
                    let capability_score = if level_diff >= 0 {
                        1.0
                    } else {
                        (capability.level as u8 as f32 / *required_level as u8 as f32).min(0.8)
                    };
                    score += capability_score * weight;
                }
            }
        }

        if total_weight == 0.0 {
            0.0
        } else {
            score / total_weight
        }
    }

    /// Update model metadata
    pub async fn update_metadata(&self, model_id: &NxrModelId, meta: ModelMeta) -> Result<(), RegistryError> {
        let mut entries = self.entries.write().await;
        let entry = entries.get_mut(model_id)
            .ok_or(RegistryError::NotFound(*model_id))?;
        entry.metadata = meta;
        Ok(())
    }

    /// Update model capabilities (enforces safety capability locks)
    pub async fn update_capabilities(&self, model_id: &NxrModelId, mut capabilities: CapabilityVector) -> Result<(), RegistryError> {
        let cap_lock = CapabilityLock::new();
        cap_lock.enforce(&mut capabilities).await.map_err(|e| RegistryError::Validation(e.to_string()))?;

        let mut entries = self.entries.write().await;
        let entry = entries.get_mut(model_id)
            .ok_or(RegistryError::NotFound(*model_id))?;
        entry.capabilities = capabilities;
        Ok(())
    }

    /// Get registry statistics
    pub async fn get_statistics(&self) -> RegistryStatistics {
        let entries = self.entries.read().await;

        let mut tier_counts: HashMap<ModelTier, usize> = HashMap::new();
        let mut total_parameters = 0u64;
        let mut experimental_count = 0;

        for entry in entries.values() {
            let meta = &entry.metadata;
            *tier_counts.entry(meta.tier).or_insert(0) += 1;
            if let Some(params) = meta.parameter_count {
                total_parameters += params;
            }
            if meta.experimental {
                experimental_count += 1;
            }
        }

        RegistryStatistics {
            total_models: entries.len(),
            tier_counts,
            total_parameters,
            experimental_count,
            last_updated: chrono::Utc::now(),
        }
    }

    /// Validate all registered models
    pub async fn validate_all(&self) -> Vec<ValidationResult> {
        let entries = self.entries.read().await;
        let mut results = Vec::new();

        for (model_id, entry) in entries.iter() {
            match &entry.model {
                Some(model) => {
                    match model.validate().await {
                        Ok(validation) => {
                            results.push(ValidationResult {
                                model_id: *model_id,
                                success: validation.is_valid,
                                errors: validation.errors,
                                warnings: validation.warnings,
                            });
                        }
                        Err(e) => {
                            results.push(ValidationResult {
                                model_id: *model_id,
                                success: false,
                                errors: vec![e.to_string()],
                                warnings: Vec::new(),
                            });
                        }
                    }
                }
                None => {
                    results.push(ValidationResult {
                        model_id: *model_id,
                        success: false,
                        errors: vec!["No model instance registered".to_string()],
                        warnings: Vec::new(),
                    });
                }
            }
        }

        results
    }
}

impl Default for NxrModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry error
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    NotFound(NxrModelId),
    
    #[error("Model already registered: {0}")]
    AlreadyRegistered(NxrModelId),
    
    #[error("Identity mismatch between model and metadata")]
    IdentityMismatch,
    
    #[error("No models available")]
    NoModelsAvailable,
    
    #[error("No suitable model for task")]
    NoSuitableModel,
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Task definition for model selection
#[derive(Debug, Clone)]
pub struct Task {
    /// Task name
    pub name: String,
    /// Task description
    pub description: String,
    /// Required capabilities
    pub required_capabilities: HashMap<super::capability_spec::CapabilityDomain, super::capability_spec::CapabilityLevel>,
    /// Capability weights
    pub capability_weights: HashMap<super::capability_spec::CapabilityDomain, f32>,
    /// Task priority
    pub priority: TaskPriority,
    /// Resource constraints
    pub resource_constraints: Option<ResourceConstraints>,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Resource constraints for task
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum memory in GB
    pub max_memory_gb: f32,
    /// Maximum latency in ms
    pub max_latency_ms: u32,
    /// Require GPU
    pub require_gpu: bool,
    /// Maximum cost
    pub max_cost: Option<f32>,
}

/// Registry validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Model ID
    pub model_id: NxrModelId,
    /// Validation success
    pub success: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Total number of models
    pub total_models: usize,
    /// Models by tier
    pub tier_counts: HashMap<ModelTier, usize>,
    /// Total parameters across all models
    pub total_parameters: u64,
    /// Number of experimental models
    pub experimental_count: usize,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Global model registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<Arc<NxrModelRegistry>> = std::sync::OnceLock::new();

/// Get global model registry
pub fn global_registry() -> Arc<NxrModelRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| Arc::new(NxrModelRegistry::new())).clone()
}

/// Initialize global registry with default models
pub async fn initialize_global_registry() -> Result<(), RegistryError> {
    let registry = global_registry();

    for model_id in crate::shared::model_identity::NxrModelId::all() {
        let config = crate::shared::model_config::NxrModelConfig::for_model(model_id);
        let caps = crate::shared::capability_spec::predefined::get_capabilities(model_id);
        let tier = model_id.tier();
        let meta = crate::shared::model_identity::ModelMeta::new(
            model_id,
            tier,
            "0.1.0".to_string(),
            model_id.fullname().to_string(),
        );

        // Use register_components instead of direct field access.
        // Note: model instances are not created here; they are loaded
        // on-demand by the model server. For eager instantiation,
        // replace with register_model() once all NXR models implement
        // a uniform no-arg constructor.
        registry.register_components(model_id, meta, caps, config).await?;
    }

    Ok(())
}
