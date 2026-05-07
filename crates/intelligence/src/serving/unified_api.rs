//! Unified API for Integrated Nexora Models
//! 
//! Provides a single interface for using all integrated models:
//! - SACA (Systematic Adaptive Code Architecture)
//! - ATQS (Adaptive Tensor Quantization & Sparsification)
//! - CAFFEINE (Contrastive-Aware Fusion Framework)
//! - HAS-MoE-FFN (Hybrid Adaptive Structured MoE-FFN)

use crate::saca::{SACA, SACAConfig, CodingTask, SACASolution, SACAIntegration};
use crate::atqs::{ATQSConfig, compression::CompressionEngine};
use crate::caffeine::{Caffeine, CaffeineConfig, types::{MultiModalInputs, MultiModalOutputs}};
use crate::has_moe_ffn::{HasMoeFfn, HasMoeFfnConfig, router::ExpertRouter};
use std::sync::{Arc, Mutex};
use tracing::{info, debug, warn};

// Explicit Result type to avoid ambiguity
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Unified configuration for all models
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    pub saca_config: SACAConfig,
    pub atqs_config: Option<ATQSConfig>,
    pub caffeine_config: Option<CaffeineConfig>,
    pub has_moe_config: Option<HasMoeFfnConfig>,
    pub integration_mode: IntegrationMode,
}

/// Integration modes for different use cases
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationMode {
    /// SACA only (basic code generation)
    SACAOnly,
    /// SACA + ATQS (code generation with compression)
    SACAWithATQS,
    /// SACA + CAFFEINE (code generation with multimodal processing)
    SACAWithCaffeine,
    /// SACA + HAS-MoE-FFN (code generation with expert routing)
    SACAWithHasMoe,
    /// Full integration (all models)
    FullIntegration,
}

/// Unified model interface
pub struct UnifiedModel {
    config: UnifiedConfig,
    saca_integration: SACAIntegration,
    caffeine_model: Option<Arc<Mutex<Caffeine>>>,
    has_moe_model: Option<Arc<HasMoeFfn>>,
}

impl UnifiedModel {
    /// Create new unified model instance
    pub async fn new(config: UnifiedConfig) -> Result<Self> {
        info!("Initializing Unified Model with integration mode: {:?}", config.integration_mode);
        
        // Initialize SACA integration
        let mut saca_integration = SACAIntegration::new(config.saca_config.clone()).await?;
        
        // Add ATQS compression if enabled
        if let Some(atqs_config) = &config.atqs_config {
            match config.integration_mode {
                IntegrationMode::SACAWithATQS | IntegrationMode::FullIntegration => {
                    let compression_engine = Arc::new(CompressionEngine::new(atqs_config.clone())
                        .map_err(|e| format!("Compression engine creation failed: {}", e))?);
                    saca_integration = saca_integration.with_atqs_compression(compression_engine);
                    info!("ATQS compression enabled");
                }
                _ => {}
            }
        }
        
        // Add Caffeine model if enabled
        let caffeine_model = if let Some(caffeine_config) = &config.caffeine_config {
            match config.integration_mode {
                IntegrationMode::SACAWithCaffeine | IntegrationMode::FullIntegration => {
                    let caffeine = Arc::new(Mutex::new(Caffeine::new(caffeine_config.clone())?));
                    saca_integration = saca_integration.with_caffeine(caffeine.clone());
                    info!("CAFFEINE multimodal processing enabled");
                    Some(caffeine)
                }
                _ => None,
            }
        } else {
            None
        };
        
        // Add HAS-MoE-FFN routing if enabled
        let has_moe_model = if let Some(has_moe_config) = &config.has_moe_config {
            match config.integration_mode {
                IntegrationMode::SACAWithHasMoe | IntegrationMode::FullIntegration => {
                    let has_moe = Arc::new(HasMoeFfn::new(has_moe_config.clone())?);
                    // Note: In practice, we'd need to extract the router from HasMoeFfn
                    // For now, we'll create a separate router
                    let router = Arc::new(ExpertRouter::new(has_moe_config.router_config.clone())?);
                    saca_integration = saca_integration.with_has_moe_routing(router);
                    info!("HAS-MoE-FFN expert routing enabled");
                    Some(has_moe)
                }
                _ => None,
            }
        } else {
            None
        };
        
        Ok(Self {
            config,
            saca_integration,
            caffeine_model,
            has_moe_model,
        })
    }
    
    /// Solve coding task using integrated models
    pub async fn generate_code(&self, task: &CodingTask) -> Result<UnifiedSolution> {
        info!("Starting unified coding task solution");
        
        let start_time = std::time::Instant::now();
        
        // Use SACA integration for enhanced solution
        let enhanced_solution = self.saca_integration.solve_with_models(task.clone()).await?;
        
        // Additional processing based on integration mode
        let mut solution = UnifiedSolution {
            base_solution: enhanced_solution.base_solution,
            atqs_compression_applied: enhanced_solution.atqs_compression_applied,
            caffeine_multimodal_enhanced: enhanced_solution.caffeine_multimodal_enhanced,
            has_moe_routing_applied: enhanced_solution.has_moe_routing_applied,
            compression_ratio: enhanced_solution.compression_ratio,
            routing_efficiency: enhanced_solution.routing_efficiency,
            multimodal_features: enhanced_solution.multimodal_features,
            execution_time: start_time.elapsed(),
            integration_mode: self.config.integration_mode.clone(),
            quality_score: 0.0,
        };
        
        // Apply additional processing if needed
        match self.config.integration_mode {
            IntegrationMode::FullIntegration => {
                self.apply_full_integration_processing(&mut solution, &task).await?;
            }
            IntegrationMode::SACAWithCaffeine => {
                let multimodal_input = MultiModalInputs {
                    text: None,
                    image: None,
                    audio: None,
                    video: None,
                    context: None,
                };
                let output = self.process_multimodal(&multimodal_input).await?;
            }
            IntegrationMode::SACAWithHasMoe => {
                self.apply_has_moe_routing(&mut solution, &task).await?;
            }
            _ => {}
        }
        
        info!("Unified solution completed in {:?}", solution.execution_time);
        Ok(solution)
    }
    
    /// Process multimodal inputs
    pub async fn process_multimodal(&self, inputs: &MultiModalInputs) -> Result<MultiModalOutputs> {
        if let Some(caffeine) = &self.caffeine_model {
            Ok(caffeine.lock().map_err(|e| format!("Failed to lock caffeine model: {}", e))?.forward(&inputs)?)
        } else {
            Err("CAFFEINE model not enabled".into())
        }
    }
    
    /// Apply full integration processing
    async fn apply_full_integration_processing(&self, solution: &mut UnifiedSolution, task: &CodingTask) -> Result<()> {
        debug!("Applying full integration processing");
        
        // Apply cross-model optimizations
        if solution.atqs_compression_applied && solution.caffeine_multimodal_enhanced {
            // Optimize compressed code based on multimodal insights
            solution.quality_score += 0.02;
        }
        
        if solution.has_moe_routing_applied && solution.routing_efficiency > 0.9 {
            // Boost quality for excellent routing
            solution.quality_score += 0.01;
        }
        
        // Ensure quality score stays within bounds
        solution.quality_score = solution.quality_score.min(1.0);
        
        Ok(())
    }
    
    /// Apply CAFFEINE post-processing
    async fn apply_caffeine_multimodal(&self, solution: &mut UnifiedSolution, task: &CodingTask) -> Result<()> {
        debug!("Applying CAFFEINE post-processing");
        
        // Additional multimodal enhancements could be applied here
        if solution.caffeine_multimodal_enhanced {
            solution.quality_score += 0.01;
            solution.quality_score = solution.quality_score.min(1.0);
        }
        
        Ok(())
    }
    
    /// Apply HAS-MoE-FFN post-processing
    async fn apply_has_moe_routing(&self, solution: &mut UnifiedSolution, task: &CodingTask) -> Result<()> {
        debug!("Applying HAS-MoE-FFN post-processing");
        
        // Additional routing optimizations could be applied here
        if solution.has_moe_routing_applied && solution.routing_efficiency > 0.8 {
            solution.quality_score += 0.01;
            solution.quality_score = solution.quality_score.min(1.0);
        }
        
        Ok(())
    }
    
    /// Get model statistics
    pub fn get_statistics(&self) -> UnifiedStats {
        let integration_stats = self.saca_integration.get_integration_stats();
        
        UnifiedStats {
            integration_mode: self.config.integration_mode.clone(),
            models_enabled: integration_stats.total_models_enabled,
            atqs_enabled: integration_stats.atqs_enabled,
            caffeine_enabled: integration_stats.caffeine_enabled,
            has_moe_enabled: integration_stats.has_moe_enabled,
        }
    }
}

/// Unified solution result
#[derive(Debug, Clone)]
pub struct UnifiedSolution {
    pub base_solution: SACASolution,
    pub atqs_compression_applied: bool,
    pub caffeine_multimodal_enhanced: bool,
    pub has_moe_routing_applied: bool,
    pub compression_ratio: f64,
    pub routing_efficiency: f32,
    pub multimodal_features: Vec<f32>,
    pub execution_time: std::time::Duration,
    pub integration_mode: IntegrationMode,
    pub quality_score: f32,
}

/// Unified statistics
#[derive(Debug, Clone)]
pub struct UnifiedStats {
    pub integration_mode: IntegrationMode,
    pub models_enabled: u32,
    pub atqs_enabled: bool,
    pub caffeine_enabled: bool,
    pub has_moe_enabled: bool,
}

/// Unified model factory
pub struct UnifiedModelFactory;

impl UnifiedModelFactory {
    /// Create model for basic code generation
    pub async fn create_basic_coder() -> Result<UnifiedModel> {
        let config = UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: None,
            caffeine_config: None,
            has_moe_config: None,
            integration_mode: IntegrationMode::SACAOnly,
        };
        UnifiedModel::new(config).await
    }
    
    /// Create model for code generation with compression
    pub async fn create_compressed_coder() -> Result<UnifiedModel> {
        let config = UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: Some(ATQSConfig::default()),
            caffeine_config: None,
            has_moe_config: None,
            integration_mode: IntegrationMode::SACAWithATQS,
        };
        UnifiedModel::new(config).await
    }
    
    /// Create model for multimodal code generation
    pub async fn create_multimodal_coder() -> Result<UnifiedModel> {
        let config = UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: None,
            caffeine_config: Some(CaffeineConfig::medium_model()),
            has_moe_config: None,
            integration_mode: IntegrationMode::SACAWithCaffeine,
        };
        UnifiedModel::new(config).await
    }
    
    /// Create model for expert-routed code generation
    pub async fn create_expert_coder() -> Result<UnifiedModel> {
        let config = UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: None,
            caffeine_config: None,
            has_moe_config: Some(HasMoeFfnConfig::medium_model()),
            integration_mode: IntegrationMode::SACAWithHasMoe,
        };
        UnifiedModel::new(config).await
    }
    
    /// Create full integration model
    pub async fn create_full_integration() -> Result<UnifiedModel> {
        let config = UnifiedConfig {
            saca_config: SACAConfig::default(),
            atqs_config: Some(ATQSConfig::default()),
            caffeine_config: Some(CaffeineConfig::medium_model()),
            has_moe_config: Some(HasMoeFfnConfig::medium_model()),
            integration_mode: IntegrationMode::FullIntegration,
        };
        UnifiedModel::new(config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unified_model_creation() {
        let model = UnifiedModelFactory::create_basic_coder().await;
        assert!(model.is_ok());
    }
    
    #[tokio::test]
    async fn test_integration_modes() {
        let basic = UnifiedModelFactory::create_basic_coder().await.unwrap();
        let stats = basic.get_statistics();
        assert_eq!(stats.models_enabled, 0);
        
        let compressed = UnifiedModelFactory::create_compressed_coder().await.unwrap();
        let stats = compressed.get_statistics();
        assert!(stats.atqs_enabled);
    }
}
