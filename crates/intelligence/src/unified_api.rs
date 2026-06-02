//! Consolidated Unified API for Nexora AI Models
//!
//! Re-exports non-conflicting types from serving layer and provides
//! trait-based wrappers for polymorphism. Single entry point for
//! IntegrationMode — no duplicates.

pub use crate::serving::unified_api::{
    ExpertRouter, HasMoeFfn, HasMoeFfnConfig, RouterConfig, RoutingDecision,
    UnifiedConfig, UnifiedModel, UnifiedModelFactory, UnifiedSolution, UnifiedStats,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_mode_debug_clone() {
        let modes = [
            IntegrationMode::Basic,
            IntegrationMode::Compressed,
            IntegrationMode::Multimodal,
            IntegrationMode::Expert,
            IntegrationMode::Full,
        ];
        for mode in &modes {
            let cloned = mode.clone();
            assert_eq!(format!("{:?}", mode), format!("{:?}", cloned));
        }
    }

    #[test]
    fn test_integration_mode_partial_eq() {
        assert_eq!(IntegrationMode::Basic, IntegrationMode::Basic);
        assert_ne!(IntegrationMode::Basic, IntegrationMode::Full);
    }

    #[test]
    fn test_coding_task_default_fields() {
        let task = CodingTask {
            description: "test".into(),
            requirements: vec![],
            constraints: vec![],
            context: None,
        };
        assert_eq!(task.description, "test");
        assert!(task.requirements.is_empty());
        assert!(task.context.is_none());
    }

    #[test]
    fn test_coding_task_with_context() {
        let ctx = TaskContext {
            repository_path: Some("/repo".into()),
            existing_files: vec!["main.rs".into()],
            dependencies: vec!["tokio".into()],
            coding_standards: [("rust".into(), "clippy".into())].into(),
        };
        let task = CodingTask {
            description: "code".into(),
            requirements: vec!["fast".into()],
            constraints: vec!["no-std".into()],
            context: Some(ctx),
        };
        assert!(task.context.is_some());
        assert_eq!(
            task.context.as_ref().unwrap().repository_path,
            Some("/repo".into())
        );
    }

    #[test]
    fn test_code_solution_defaults() {
        let sol = CodeSolution {
            quality_score: 0.85,
            execution_time: std::time::Duration::from_secs(2),
            integration_mode: IntegrationMode::Expert,
            atqs_compression_applied: false,
            compression_ratio: 0.0,
            caffeine_multimodal_applied: true,
            has_moe_routing_applied: true,
            routing_efficiency: 0.92,
            generated_code: "fn main() {}".into(),
        };
        assert_eq!(sol.quality_score, 0.85);
        assert!(sol.caffeine_multimodal_applied);
        assert_eq!(sol.routing_efficiency, 0.92);
    }

    #[test]
    fn test_model_error_initialization() {
        let err = ModelError::InitializationFailed("oom".into());
        assert_eq!(err.to_string(), "Model initialization failed: oom");
    }

    #[test]
    fn test_model_error_generation() {
        let err = ModelError::GenerationFailed("timeout".into());
        assert_eq!(err.to_string(), "Code generation failed: timeout");
    }

    #[test]
    fn test_model_error_invalid_config() {
        let err = ModelError::InvalidConfiguration("bad param".into());
        assert_eq!(err.to_string(), "Invalid configuration: bad param");
    }

    #[test]
    fn test_model_statistics_defaults() {
        let stats = ModelStatistics {
            models_enabled: 0,
            atqs_enabled: false,
            caffeine_enabled: false,
            has_moe_enabled: false,
        };
        assert_eq!(stats.models_enabled, 0);
        assert!(!stats.atqs_enabled);
    }

    #[test]
    fn test_serving_model_factory_alias() {
        // Verify the alias points to the same type as serving
        let _ = ServingModelFactory;
    }

    #[test]
    fn test_code_solution_serde_roundtrip() {
        let sol = CodeSolution {
            quality_score: 0.75,
            execution_time: std::time::Duration::from_millis(500),
            integration_mode: IntegrationMode::Full,
            atqs_compression_applied: true,
            compression_ratio: 0.5,
            caffeine_multimodal_applied: false,
            has_moe_routing_applied: true,
            routing_efficiency: 0.88,
            generated_code: "let x = 1;".into(),
        };
        let json = serde_json::to_string(&sol).unwrap();
        let deserialized: CodeSolution = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.quality_score, 0.75);
        assert_eq!(deserialized.integration_mode, IntegrationMode::Full);
        assert!(deserialized.atqs_compression_applied);
    }

    #[test]
    fn test_coding_task_serde_roundtrip() {
        let ctx = TaskContext {
            repository_path: None,
            existing_files: vec![],
            dependencies: vec![],
            coding_standards: std::collections::HashMap::new(),
        };
        let task = CodingTask {
            description: "write tests".into(),
            requirements: vec!["cover all branches".into()],
            constraints: vec!["no external deps".into()],
            context: Some(ctx),
        };
        let json = serde_json::to_string(&task).unwrap();
        let deserialized: CodingTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.description, "write tests");
    }

    #[test]
    fn test_model_error_impl_error() {
        let err = ModelError::GenerationFailed("e".into());
        let deref_err: &dyn std::error::Error = &err;
        assert!(deref_err.source().is_none());
    }
}
