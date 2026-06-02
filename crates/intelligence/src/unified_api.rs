//! Consolidated Unified API for Nexora AI Models
//!
//! Re-exports non-conflicting types from serving layer and provides
//! trait-based wrappers for polymorphism. Single entry point for
//! IntegrationMode — no duplicates.

pub use crate::serving::unified_api::{
    ExpertRouter, HasMoeFfn, HasMoeFfnConfig, RouterConfig, RoutingDecision,
    UnifiedConfig, UnifiedModel, UnifiedSolution, UnifiedStats,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use nexora_foundation::reasoning::{
    CodingTask as SacaCodingTask, SACAConfig, SACAIntegration,
    TaskContext as SacaTaskContext,
};

/// Trait-based unified model interface (wraps UnifiedModel for polymorphism).
#[async_trait]
pub trait UnifiedModelTrait: Send + Sync {
    async fn generate_code(
        &self,
        task: &CodingTask,
    ) -> Result<CodeSolution, ModelError>;
    fn get_statistics(&self) -> ModelStatistics;
}

/// Factory creating trait-object models from UnifiedModelFactory.
pub struct UnifiedModelFactoryTrait;

impl UnifiedModelFactoryTrait {
    pub async fn create_basic_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        let inner = UnifiedModelFactory::create_basic_coder()
            .await
            .map_err(|e| ModelError::InitializationFailed(e.to_string()))?;
        Ok(Box::new(TraitWrapper { inner }))
    }

    pub async fn create_compressed_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        let inner = UnifiedModelFactory::create_compressed_coder()
            .await
            .map_err(|e| ModelError::InitializationFailed(e.to_string()))?;
        Ok(Box::new(TraitWrapper { inner }))
    }

    pub async fn create_multimodal_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        let inner = UnifiedModelFactory::create_multimodal_coder()
            .await
            .map_err(|e| ModelError::InitializationFailed(e.to_string()))?;
        Ok(Box::new(TraitWrapper { inner }))
    }

    pub async fn create_expert_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        let inner = UnifiedModelFactory::create_expert_coder()
            .await
            .map_err(|e| ModelError::InitializationFailed(e.to_string()))?;
        Ok(Box::new(TraitWrapper { inner }))
    }

    pub async fn create_full_integration() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        let inner = UnifiedModelFactory::create_full_integration()
            .await
            .map_err(|e| ModelError::InitializationFailed(e.to_string()))?;
        Ok(Box::new(TraitWrapper { inner }))
    }
}

use crate::serving::unified_api::IntegrationMode as ServingIntegrationMode;

struct TraitWrapper {
    inner: UnifiedModel,
}

#[async_trait]
impl UnifiedModelTrait for TraitWrapper {
    async fn generate_code(
        &self,
        task: &CodingTask,
    ) -> Result<CodeSolution, ModelError> {
        let saca_task = SacaCodingTask {
            description: task.description.clone(),
            requirements: task.requirements.clone(),
            constraints: task.constraints.clone(),
            context: task.context.as_ref().map(|c| SacaTaskContext {
                repository_path: c.repository_path.clone(),
                existing_files: c.existing_files.clone(),
                dependencies: c.dependencies.clone(),
                coding_standards: c.coding_standards.clone(),
            }),
        };
        let inner_solution = self
            .inner
            .generate_code(&saca_task)
            .await
            .map_err(|e| ModelError::GenerationFailed(e.to_string()))?;

        let integration_mode = match inner_solution.integration_mode {
            ServingIntegrationMode::SACAOnly => IntegrationMode::Basic,
            ServingIntegrationMode::SACAWithATQS => IntegrationMode::Compressed,
            ServingIntegrationMode::SACAWithCaffeine => IntegrationMode::Multimodal,
            ServingIntegrationMode::SACAWithHasMoe => IntegrationMode::Expert,
            ServingIntegrationMode::FullIntegration => IntegrationMode::Full,
        };

        Ok(CodeSolution {
            quality_score: inner_solution.quality_score as f64,
            execution_time: inner_solution.execution_time,
            integration_mode,
            atqs_compression_applied: inner_solution.atqs_compression_applied,
            compression_ratio: inner_solution.compression_ratio,
            caffeine_multimodal_applied: inner_solution.caffeine_multimodal_enhanced,
            has_moe_routing_applied: inner_solution.has_moe_routing_applied,
            routing_efficiency: inner_solution.routing_efficiency as f64,
            generated_code: inner_solution.base_solution.final_code.clone(),
        })
    }

    fn get_statistics(&self) -> ModelStatistics {
        ModelStatistics {
            models_enabled: 0,
            atqs_enabled: false,
            caffeine_enabled: false,
            has_moe_enabled: false,
        }
    }
}

// ─── Backward-compatible data types ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTask {
    pub description: String,
    pub requirements: Vec<String>,
    pub constraints: Vec<String>,
    pub context: Option<TaskContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub repository_path: Option<String>,
    pub existing_files: Vec<String>,
    pub dependencies: Vec<String>,
    pub coding_standards: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSolution {
    pub quality_score: f64,
    pub execution_time: std::time::Duration,
    pub integration_mode: IntegrationMode,
    pub atqs_compression_applied: bool,
    pub compression_ratio: f64,
    pub caffeine_multimodal_applied: bool,
    pub has_moe_routing_applied: bool,
    pub routing_efficiency: f64,
    pub generated_code: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IntegrationMode {
    Basic,
    Compressed,
    Multimodal,
    Expert,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatistics {
    pub models_enabled: u32,
    pub atqs_enabled: bool,
    pub caffeine_enabled: bool,
    pub has_moe_enabled: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Code generation failed: {0}")]
    GenerationFailed(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

// Re-export the original UnifiedModelFactory alias for discoverability.
pub use crate::serving::unified_api::UnifiedModelFactory as ServingModelFactory;
