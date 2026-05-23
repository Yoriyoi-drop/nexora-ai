//! Unified API for Nexora AI Models
//!
//! Provides a unified interface for accessing all AI models and frameworks
//! in the Nexora ecosystem through a single factory interface.

use async_trait::async_trait;
use tracing::warn;
use nexora_foundation::atqs::{compression::CompressionEngine, ATQSConfig};
use nexora_foundation::multimodal::caffeine::{
    types::MultiModalInputs, types::TextInput, Caffeine, CaffeineConfig,
};
use nexora_foundation::reasoning::{
    CodingTask as SacaCodingTask, EnhancedSACASolution, SACAConfig, SACAIntegration,
    TaskContext as SacaTaskContext,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Unified model interface that combines all AI frameworks
#[async_trait]
pub trait UnifiedModelTrait {
    /// Generate code based on the given task
    async fn generate_code(&self, task: &CodingTask) -> Result<CodeSolution, ModelError>;

    /// Get model statistics and configuration
    fn get_statistics(&self) -> ModelStatistics;
}

/// Factory for creating unified model instances
pub struct UnifiedModelFactory;

impl UnifiedModelFactory {
    /// Create a basic SACA-only model
    pub async fn create_basic_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        Ok(Box::new(BasicSacaModel::new().await))
    }

    /// Create a SACA + ATQS compressed model
    pub async fn create_compressed_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        Ok(Box::new(CompressedSacaModel::new().await))
    }

    /// Create a SACA + CAFFEINE multimodal model
    pub async fn create_multimodal_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        Ok(Box::new(MultimodalSacaModel::new().await))
    }

    /// Create a SACA + HAS-MoE expert model
    pub async fn create_expert_coder() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        Ok(Box::new(ExpertSacaModel::new().await))
    }

    /// Create a full integration model with all frameworks
    pub async fn create_full_integration() -> Result<Box<dyn UnifiedModelTrait>, ModelError> {
        Ok(Box::new(FullIntegrationModel::new().await))
    }
}

/// Coding task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTask {
    pub description: String,
    pub requirements: Vec<String>,
    pub constraints: Vec<String>,
    pub context: Option<TaskContext>,
}

/// Task context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub repository_path: Option<String>,
    pub existing_files: Vec<String>,
    pub dependencies: Vec<String>,
    pub coding_standards: HashMap<String, String>,
}

/// Generated code solution
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

/// Model integration modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationMode {
    BasicSaca,
    SacaAtqs,
    SacaCaffeine,
    SacaHasMoe,
    FullIntegration,
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatistics {
    pub integration_mode: IntegrationMode,
    pub models_enabled: u32,
    pub atqs_enabled: bool,
    pub caffeine_enabled: bool,
    pub has_moe_enabled: bool,
}

/// Model error types
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Code generation failed: {0}")]
    GenerationFailed(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

// Helper: convert unified CodingTask to SACA CodingTask
fn to_saca_coding_task(task: &CodingTask) -> SacaCodingTask {
    SacaCodingTask {
        description: task.description.clone(),
        requirements: task.requirements.clone(),
        constraints: task.constraints.clone(),
        context: task.context.as_ref().map(|c| SacaTaskContext {
            repository_path: c.repository_path.clone(),
            existing_files: c.existing_files.clone(),
            dependencies: c.dependencies.clone(),
            coding_standards: c.coding_standards.clone(),
        }),
    }
}

// Helper: convert EnhancedSACASolution to CodeSolution with given integration mode
fn enhanced_to_code_solution(
    enhanced: EnhancedSACASolution,
    mode: IntegrationMode,
) -> CodeSolution {
    CodeSolution {
        quality_score: enhanced.base_solution.quality_score as f64,
        execution_time: std::time::Duration::from_millis(
            enhanced.base_solution.execution_time.num_milliseconds() as u64,
        ),
        integration_mode: mode,
        atqs_compression_applied: enhanced.atqs_compression_applied,
        compression_ratio: enhanced.compression_ratio,
        caffeine_multimodal_applied: enhanced.caffeine_multimodal_enhanced,
        has_moe_routing_applied: enhanced.has_moe_routing_applied,
        routing_efficiency: enhanced.routing_efficiency as f64,
        generated_code: enhanced.base_solution.final_code,
    }
}

// Mock model implementations

/// Basic SACA model implementation
struct BasicSacaModel {
    saca: Option<Arc<SACAIntegration>>,
}

impl BasicSacaModel {
    async fn new() -> Self {
        let saca = match SACAIntegration::new(SACAConfig::default()).await {
            Ok(s) => Some(Arc::new(s)),
            Err(e) => {
                warn!("SACAIntegration initialization failed: {}", e);
                None
            }
        };
        BasicSacaModel { saca }
    }

    fn with_saca(saca: SACAIntegration) -> Self {
        BasicSacaModel {
            saca: Some(Arc::new(saca)),
        }
    }
}

#[async_trait]
impl UnifiedModelTrait for BasicSacaModel {
    async fn generate_code(&self, task: &CodingTask) -> Result<CodeSolution, ModelError> {
        if let Some(ref saca) = self.saca {
            let saca_task = to_saca_coding_task(task);
            let enhanced = saca
                .solve_with_models(saca_task)
                .await
                .map_err(|e| ModelError::GenerationFailed(e.to_string()))?;
            Ok(enhanced_to_code_solution(
                enhanced,
                IntegrationMode::BasicSaca,
            ))
        } else {
            return Err(ModelError::GenerationFailed(
                "SACA engine not available (failed to initialize)".to_string(),
            ));
        }
    }

    fn get_statistics(&self) -> ModelStatistics {
        ModelStatistics {
            integration_mode: IntegrationMode::BasicSaca,
            models_enabled: 1,
            atqs_enabled: false,
            caffeine_enabled: false,
            has_moe_enabled: false,
        }
    }
}

/// Helper: initialize SACAIntegration with optional extensions
async fn init_saca_with_extensions(
    atqs_config: Option<ATQSConfig>,
    caffeine_config: Option<CaffeineConfig>,
    has_moe_config: Option<super::serving::unified_api::HasMoeFfnConfig>,
) -> Option<Arc<SACAIntegration>> {
    let mut saca = match SACAIntegration::new(SACAConfig::default()).await {
        Ok(s) => s,
        Err(e) => {
            warn!("SACAIntegration initialization failed in extensions: {}", e);
            return None;
        }
    };

    if let Some(cfg) = atqs_config {
        let engine = match CompressionEngine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                warn!("CompressionEngine initialization failed: {}", e);
                return None;
            }
        };
        saca = saca.with_atqs_compression(Arc::new(engine));
    }

    if let Some(cfg) = caffeine_config {
        let caffeine = match Caffeine::new(cfg) {
            Ok(c) => c,
            Err(e) => {
                warn!("Caffeine initialization failed: {}", e);
                return None;
            }
        };
        saca = saca.with_caffeine(Arc::new(tokio::sync::Mutex::new(caffeine)));
    }

    if let Some(_cfg) = has_moe_config {
        let router = nexora_foundation::has_moe_ffn::routing::Router::new(768, 8, 2);
        saca = saca.with_has_moe_routing(Arc::new(router));
    }

    Some(Arc::new(saca))
}

/// Compressed SACA + ATQS model implementation
struct CompressedSacaModel {
    saca: Option<Arc<SACAIntegration>>,
}

impl CompressedSacaModel {
    async fn new() -> Self {
        let saca = init_saca_with_extensions(Some(ATQSConfig::default()), None, None).await;
        CompressedSacaModel { saca }
    }

    fn with_saca(saca: SACAIntegration) -> Self {
        CompressedSacaModel {
            saca: Some(Arc::new(saca)),
        }
    }
}

#[async_trait]
impl UnifiedModelTrait for CompressedSacaModel {
    async fn generate_code(&self, task: &CodingTask) -> Result<CodeSolution, ModelError> {
        if let Some(ref saca) = self.saca {
            let saca_task = to_saca_coding_task(task);
            let enhanced = saca
                .solve_with_models(saca_task)
                .await
                .map_err(|e| ModelError::GenerationFailed(e.to_string()))?;
            Ok(enhanced_to_code_solution(
                enhanced,
                IntegrationMode::SacaAtqs,
            ))
        } else {
            return Err(ModelError::GenerationFailed(
                "SACA+ATQS engine not available (failed to initialize)".to_string(),
            ));
        }
    }

    fn get_statistics(&self) -> ModelStatistics {
        ModelStatistics {
            integration_mode: IntegrationMode::SacaAtqs,
            models_enabled: 2,
            atqs_enabled: true,
            caffeine_enabled: false,
            has_moe_enabled: false,
        }
    }
}

/// Multimodal SACA + CAFFEINE model implementation
struct MultimodalSacaModel {
    saca: Option<Arc<SACAIntegration>>,
}

impl MultimodalSacaModel {
    async fn new() -> Self {
        let saca = init_saca_with_extensions(None, Some(CaffeineConfig::medium_model()), None).await;
        MultimodalSacaModel { saca }
    }

    fn with_saca(saca: SACAIntegration) -> Self {
        MultimodalSacaModel {
            saca: Some(Arc::new(saca)),
        }
    }
}

#[async_trait]
impl UnifiedModelTrait for MultimodalSacaModel {
    async fn generate_code(&self, task: &CodingTask) -> Result<CodeSolution, ModelError> {
        if let Some(ref saca) = self.saca {
            let saca_task = to_saca_coding_task(task);
            let enhanced = saca
                .solve_with_models(saca_task)
                .await
                .map_err(|e| ModelError::GenerationFailed(e.to_string()))?;
            Ok(enhanced_to_code_solution(
                enhanced,
                IntegrationMode::SacaCaffeine,
            ))
        } else {
            return Err(ModelError::GenerationFailed(
                "SACA+CAFFEINE engine not available (failed to initialize)".to_string(),
            ));
        }
    }

    fn get_statistics(&self) -> ModelStatistics {
        ModelStatistics {
            integration_mode: IntegrationMode::SacaCaffeine,
            models_enabled: 2,
            atqs_enabled: false,
            caffeine_enabled: true,
            has_moe_enabled: false,
        }
    }
}

/// Expert SACA + HAS-MoE model implementation
struct ExpertSacaModel {
    saca: Option<Arc<SACAIntegration>>,
}

impl ExpertSacaModel {
    async fn new() -> Self {
        let saca = init_saca_with_extensions(
            None,
            None,
            Some(super::serving::unified_api::HasMoeFfnConfig::medium_model()),
        ).await;
        ExpertSacaModel { saca }
    }

    fn with_saca(saca: SACAIntegration) -> Self {
        ExpertSacaModel {
            saca: Some(Arc::new(saca)),
        }
    }
}

#[async_trait]
impl UnifiedModelTrait for ExpertSacaModel {
    async fn generate_code(&self, task: &CodingTask) -> Result<CodeSolution, ModelError> {
        if let Some(ref saca) = self.saca {
            let saca_task = to_saca_coding_task(task);
            let enhanced = saca
                .solve_with_models(saca_task)
                .await
                .map_err(|e| ModelError::GenerationFailed(e.to_string()))?;
            Ok(enhanced_to_code_solution(
                enhanced,
                IntegrationMode::SacaHasMoe,
            ))
        } else {
            return Err(ModelError::GenerationFailed(
                "SACA+HAS-MoE engine not available (failed to initialize)".to_string(),
            ));
        }
    }

    fn get_statistics(&self) -> ModelStatistics {
        ModelStatistics {
            integration_mode: IntegrationMode::SacaHasMoe,
            models_enabled: 2,
            atqs_enabled: false,
            caffeine_enabled: false,
            has_moe_enabled: true,
        }
    }
}

/// Full integration model with all frameworks
struct FullIntegrationModel {
    saca: Option<Arc<SACAIntegration>>,
}

impl FullIntegrationModel {
    async fn new() -> Self {
        let saca = init_saca_with_extensions(
            Some(ATQSConfig::default()),
            Some(CaffeineConfig::medium_model()),
            Some(super::serving::unified_api::HasMoeFfnConfig::medium_model()),
        ).await;
        FullIntegrationModel { saca }
    }

    fn with_saca(saca: SACAIntegration) -> Self {
        FullIntegrationModel {
            saca: Some(Arc::new(saca)),
        }
    }
}

#[async_trait]
impl UnifiedModelTrait for FullIntegrationModel {
    async fn generate_code(&self, task: &CodingTask) -> Result<CodeSolution, ModelError> {
        if let Some(ref saca) = self.saca {
            let saca_task = to_saca_coding_task(task);
            let enhanced = saca
                .solve_with_models(saca_task)
                .await
                .map_err(|e| ModelError::GenerationFailed(e.to_string()))?;
            Ok(enhanced_to_code_solution(
                enhanced,
                IntegrationMode::FullIntegration,
            ))
        } else {
            return Err(ModelError::GenerationFailed(
                "Full integration engine not available (failed to initialize)".to_string(),
            ));
        }
    }

    fn get_statistics(&self) -> ModelStatistics {
        ModelStatistics {
            integration_mode: IntegrationMode::FullIntegration,
            models_enabled: 4,
            atqs_enabled: true,
            caffeine_enabled: true,
            has_moe_enabled: true,
        }
    }
}
