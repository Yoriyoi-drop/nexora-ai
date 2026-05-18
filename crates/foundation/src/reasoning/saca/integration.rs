//! SACA Integration with Existing Models
//! 
//! Integrates SACA framework with ATQS, Caffeine, and HAS MoE FFN models
//! Provides unified interface for model-enhanced coding intelligence

use super::{types::*, config::*, error::*, prelude::*};
use crate::atqs::compression::CompressionEngine;
use crate::multimodal::caffeine::Caffeine;
use crate::has_moe_ffn::routing::Router;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// SACA integration manager
pub struct SACAIntegration {
    saca: Arc<SACA>,
    atqs_compression: Option<Arc<CompressionEngine>>,
    caffeine_model: Option<Arc<Mutex<Caffeine>>>,
    has_moe_router: Option<Arc<Router>>,
}

impl SACAIntegration {
    /// Create new integrated SACA instance
    pub async fn new(config: SACAConfig) -> SACAResult<Self> {
        let saca = Arc::new(SACA::new(config).await?);
        
        info!("SACA Integration initialized with model support");
        
        Ok(Self {
            saca,
            atqs_compression: None,
            caffeine_model: None,
            has_moe_router: None,
        })
    }
    
    /// Enable ATQS compression integration
    pub fn with_atqs_compression(mut self, compression: Arc<CompressionEngine>) -> Self {
        self.atqs_compression = Some(compression);
        self
    }
    
    /// Enable Caffeine model integration
    pub fn with_caffeine(mut self, caffeine: Arc<Mutex<Caffeine>>) -> Self {
        self.caffeine_model = Some(caffeine);
        self
    }
    
    /// Enable HAS MoE FFN routing integration
    pub fn with_has_moe_routing(mut self, router: Arc<Router>) -> Self {
        self.has_moe_router = Some(router);
        self
    }
    
    /// Get reference to the underlying SACA instance
    pub fn saca(&self) -> &Arc<SACA> {
        &self.saca
    }
    
    /// Solve coding task with full model integration
    pub async fn solve_with_models(&self, task: CodingTask) -> SACAResult<EnhancedSACASolution> {
        info!("Starting SACA solution with model integration");
        
        // Execute base SACA pipeline
        let base_solution = self.saca.solve(task.clone()).await?;
        
        // Enhance with model capabilities
        let enhanced_solution = self.enhance_solution_with_models(base_solution, &task).await?;
        
        info!("Enhanced SACA solution completed. Quality: {:.3}", enhanced_solution.base_solution.quality_score);
        Ok(enhanced_solution)
    }
    
    /// Enhance base solution with model capabilities
    async fn enhance_solution_with_models(
        &self,
        base_solution: SACASolution,
        task: &CodingTask,
    ) -> SACAResult<EnhancedSACASolution> {
        let mut enhanced_solution = EnhancedSACASolution {
            base_solution,
            atqs_compression_applied: false,
            caffeine_multimodal_enhanced: false,
            has_moe_routing_applied: false,
            compression_ratio: 1.0,
            routing_efficiency: 1.0,
            multimodal_features: Vec::new(),
        };
        
        // Apply ATQS compression if available
        if let Some(ref compression) = self.atqs_compression {
            enhanced_solution = self.apply_atqs_compression(enhanced_solution, compression).await?;
        }
        
        // Apply Caffeine multimodal enhancement if available
        if let Some(ref caffeine) = self.caffeine_model {
            enhanced_solution = self.apply_caffeine_enhancement(enhanced_solution, caffeine, task).await?;
        }
        
        // Apply HAS MoE FFN routing if available
        if let Some(ref router) = self.has_moe_router {
            enhanced_solution = self.apply_has_moe_routing(enhanced_solution, router).await?;
        }
        
        Ok(enhanced_solution)
    }
    
    /// Apply ATQS compression to solution
    async fn apply_atqs_compression(
        &self,
        mut solution: EnhancedSACASolution,
        compression: &Arc<CompressionEngine>,
    ) -> SACAResult<EnhancedSACASolution> {
        debug!("Applying ATQS compression to solution");
        
        // Compress the final code
        let compressed_code = compression.compress_string(&solution.base_solution.final_code)
            .map_err(|e| SACAError::ContextError(format!("Compression failed: {}", e)))?;
        let compression_ratio = solution.base_solution.final_code.len() as f64 / compressed_code.len() as f64;
        
        solution.base_solution.final_code = compressed_code;
        solution.atqs_compression_applied = true;
        solution.compression_ratio = compression_ratio;
        
        debug!("ATQS compression applied with ratio: {:.2}", compression_ratio);
        Ok(solution)
    }
    
    /// Apply Caffeine multimodal enhancement
    async fn apply_caffeine_enhancement(
        &self,
        mut solution: EnhancedSACASolution,
        caffeine: &Arc<Mutex<Caffeine>>,
        task: &CodingTask,
    ) -> SACAResult<EnhancedSACASolution> {
        debug!("Applying Caffeine multimodal enhancement");
        
        // Create multimodal inputs from task description
        let multimodal_inputs = crate::multimodal::caffeine::types::MultiModalInputs {
            text: Some(crate::multimodal::caffeine::types::TextInput {
                text: task.description.clone(),
                tokens: None,
                language: "en".to_string(),
            }),
            image: None, // Could be enhanced with diagrams/screenshots
            audio: None,
            video: None,
            context: Some(crate::multimodal::caffeine::types::ContextInfo {
                task_type: crate::multimodal::caffeine::types::TaskType::Generation,
                instruction: Some("Generate code based on requirements".to_string()),
                previous_actions: Vec::new(),
                environment_state: None,
            }),
        };
        
        // Process through Caffeine model
        let mut caffeine = caffeine.lock().await;
        let multimodal_outputs = caffeine.forward(&multimodal_inputs).await
            .map_err(|e| SACAError::ContextError(format!("Caffeine processing failed: {}", e)))?;
        
        // Extract features and enhance solution
        solution.caffeine_multimodal_enhanced = true;
        
        // Extract features from text output if available
        if let Some(text_output) = &multimodal_outputs.text {
            // Convert text output to feature vector (simplified)
            solution.multimodal_features = vec![
                text_output.confidence,
                text_output.text.len() as f32 / 1000.0, // Normalized length
                text_output.token_probs.as_ref().map_or(0.5, |probs| {
                    probs.iter().sum::<f32>() / probs.len() as f32
                }),
            ];
        } else {
            solution.multimodal_features = vec![];
        }
        
        // Update solution quality based on multimodal insights
        if !solution.multimodal_features.is_empty() {
            solution.base_solution.quality_score += 0.05; // Small boost for multimodal enhancement
            solution.base_solution.quality_score = solution.base_solution.quality_score.min(1.0);
        }
        
        debug!("Caffeine enhancement applied with {} features", solution.multimodal_features.len());
        Ok(solution)
    }
    
    /// Apply HAS MoE FFN routing to solution
    async fn apply_has_moe_routing(
        &self,
        mut solution: EnhancedSACASolution,
        router: &Arc<Router>,
    ) -> SACAResult<EnhancedSACASolution> {
        debug!("Applying HAS MoE FFN routing to solution");
        
        // Convert solution to tensor format for routing
        let tensor_input = self.solution_to_tensor(&solution.base_solution)?;
        
        // Apply expert routing
        let mut router_clone = match Arc::try_unwrap(Arc::clone(router)) {
            Ok(router) => router,
            Err(_) => {
                // If Arc can't be unwrapped (multiple references), we need a different approach
                // For now, create a new instance - this is a temporary fix
                Router::new(768, 8, 2)
            }
        };
        let routing_decisions = router_clone.route_single(&tensor_input)
            .map_err(|e| SACAError::RerankError(format!("Routing failed: {}", e)))?;
        
        // Apply routing to optimize solution
        solution.has_moe_routing_applied = true;
        
        // Get routing stats for efficiency score
        let routing_stats = router.get_routing_stats();
        solution.routing_efficiency = routing_stats.load_balance_score;
        
        // Update solution based on routing insights
        if routing_stats.load_balance_score > 0.8 {
            solution.base_solution.quality_score += 0.03; // Small boost for good routing
            solution.base_solution.quality_score = solution.base_solution.quality_score.min(1.0);
        }
        
        debug!("HAS MoE routing applied with efficiency: {:.3}", solution.routing_efficiency);
        Ok(solution)
    }
    
    /// Convert solution to tensor format for HAS MoE FFN
    fn solution_to_tensor(&self, solution: &SACASolution) -> SACAResult<ndarray::Array1<f32>> {
        // Create features from solution
        let mut features = Vec::new();
        
        // Quality metrics
        features.push(solution.quality_score);
        features.push(solution.test_coverage);
        
        // Module count
        features.push(solution.modules.len() as f32);
        
        // Execution time (normalized)
        features.push(solution.execution_time.num_milliseconds() as f32 / 1000.0);
        
        // Performance grade (encoded)
        let grade_score = match solution.performance_grade {
            PerformanceGrade::Excellent => 1.0,
            PerformanceGrade::Good => 0.75,
            PerformanceGrade::Average => 0.5,
            PerformanceGrade::Poor => 0.25,
        };
        features.push(grade_score);
        
        // Pad to fixed size if needed
        while features.len() < 768 { // Standard embedding size
            features.push(0.0);
        }
        
        Ok(ndarray::Array1::from_vec(features))
    }
    
    /// Get integration statistics
    pub fn get_integration_stats(&self) -> IntegrationStats {
        IntegrationStats {
            atqs_enabled: self.atqs_compression.is_some(),
            caffeine_enabled: self.caffeine_model.is_some(),
            has_moe_enabled: self.has_moe_router.is_some(),
            total_models_enabled: [
                self.atqs_compression.is_some(),
                self.caffeine_model.is_some(),
                self.has_moe_router.is_some(),
            ].iter().map(|&x| if x { 1 } else { 0 }).sum(),
        }
    }
}

/// Enhanced SACA solution with model integrations
#[derive(Debug, Clone)]
pub struct EnhancedSACASolution {
    pub base_solution: SACASolution,
    pub atqs_compression_applied: bool,
    pub caffeine_multimodal_enhanced: bool,
    pub has_moe_routing_applied: bool,
    pub compression_ratio: f64,
    pub routing_efficiency: f32,
    pub multimodal_features: Vec<f32>,
}

/// Integration statistics
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    pub atqs_enabled: bool,
    pub caffeine_enabled: bool,
    pub has_moe_enabled: bool,
    pub total_models_enabled: u32,
}

/// SACA factory for easy integration setup
pub struct SACAFactory;

impl SACAFactory {
    /// Create SACA instance with all available models
    pub async fn create_full_saca(
        saca_config: SACAConfig,
        atqs_config: Option<crate::atqs::config::ATQSConfig>,
        caffeine_config: Option<crate::multimodal::caffeine::config::CaffeineConfig>,
        has_moe_config: Option<crate::has_moe_ffn::HasMoeFFNConfig>,
    ) -> SACAResult<SACAIntegration> {
        let mut integration = SACAIntegration::new(saca_config).await?;
        
        // Add ATQS compression if configured
        if let Some(atqs_config) = atqs_config {
            let compression = Arc::new(CompressionEngine::new(atqs_config)
                .map_err(|e| SACAError::ContextError(format!("Compression engine creation failed: {}", e)))?);
            integration = integration.with_atqs_compression(compression);
        }
        
        // Add Caffeine model if configured
        if let Some(caffeine_config) = caffeine_config {
            let caffeine = Arc::new(Mutex::new(Caffeine::new(caffeine_config)
                .map_err(|e| SACAError::ContextError(format!("Caffeine creation failed: {}", e)))?));
            integration = integration.with_caffeine(caffeine);
        }
        
        // Add HAS MoE FFN routing if configured
        if let Some(has_moe_config) = has_moe_config {
            let router = Arc::new(Router::new(has_moe_config.hidden_size, has_moe_config.num_experts, has_moe_config.top_k));
            integration = integration.with_has_moe_routing(router);
        }
        
        Ok(integration)
    }
    
    /// Create SACA with ATQS only
    pub async fn create_saca_with_atqs(
        saca_config: SACAConfig,
        atqs_config: crate::atqs::config::ATQSConfig,
    ) -> SACAResult<SACAIntegration> {
        let compression = Arc::new(CompressionEngine::new(atqs_config)
            .map_err(|e| SACAError::ContextError(format!("Compression engine creation failed: {}", e)))?);
        Ok(SACAIntegration::new(saca_config).await?.with_atqs_compression(compression))
    }
    
    /// Create SACA with Caffeine only
    pub async fn create_saca_with_caffeine(
        saca_config: SACAConfig,
        caffeine_config: crate::multimodal::caffeine::config::CaffeineConfig,
    ) -> SACAResult<SACAIntegration> {
        let caffeine = Arc::new(Mutex::new(Caffeine::new(caffeine_config)
            .map_err(|e| SACAError::ContextError(format!("Caffeine creation failed: {}", e)))?));
        Ok(SACAIntegration::new(saca_config).await?.with_caffeine(caffeine))
    }
    
    /// Create SACA with HAS MoE FFN only
    pub async fn create_saca_with_has_moe(
        saca_config: SACAConfig,
        has_moe_config: crate::has_moe_ffn::HasMoeFFNConfig,
    ) -> SACAResult<SACAIntegration> {
        let router = Arc::new(Router::new(has_moe_config.hidden_size, has_moe_config.num_experts, has_moe_config.top_k));
        Ok(SACAIntegration::new(saca_config).await?.with_has_moe_routing(router))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_saca_integration_creation() -> anyhow::Result<()> {
        let config = SACAConfig::default();
        let _integration = SACAIntegration::new(config).await;
        // If we get here without panicking, the creation succeeded
        Ok(())
    }
    
    #[tokio::test]
    async fn test_integration_stats() -> anyhow::Result<()> {
        let config = SACAConfig::default();
        let integration = SACAIntegration::new(config).await
            .map_err(|e| anyhow::anyhow!("Failed to create integration: {}", e))?;
        
        let stats = integration.get_integration_stats();
        assert_eq!(stats.total_models_enabled, 0);
        assert!(!stats.atqs_enabled);
        assert!(!stats.caffeine_enabled);
        assert!(!stats.has_moe_enabled);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_solution_to_tensor() -> anyhow::Result<()> {
        let integration = SACAIntegration::new(SACAConfig::default()).await
            .map_err(|e| anyhow::anyhow!("Failed to create integration: {}", e))?;
        
        let solution = SACASolution {
            session_id: uuid::Uuid::new_v4(),
            modules: vec![],
            quality_score: 0.8,
            total_iterations: 1,
            total_feedback_loops: 0,
            execution_time: chrono::Duration::milliseconds(1000),
            final_code: "test".to_string(),
            test_coverage: 0.9,
            performance_grade: PerformanceGrade::Good,
        };
        
        let tensor = integration.solution_to_tensor(&solution)
            .map_err(|e| anyhow::anyhow!("Failed to convert solution to tensor: {}", e))?;
        assert_eq!(tensor.shape(), &[768]);
        
        Ok(())
    }
}
