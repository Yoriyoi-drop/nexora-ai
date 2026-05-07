//! CAFFEINE: Contrastive-Aware Fusion Framework with Efficient Instruction-following and Narrative Embodiment
//! 
//! Foundation model multimodal holistik yang menggabungkan:
//! - Regional Contrastive Visual Encoder (dari CLIP)
//! - Hierarchical Tri-Query Former (dari BLIP-2)
//! - Unified Discrete Multimodal Token Space (dari MIO)
//! - Instruction-Aware Components (dari LLaVA)
//! - Agentic Action Head (dari Magma)
//! 
//! CAFFEINE terintegrasi dengan ATQS untuk compression dan HAS-MoE-FFN untuk expert routing.

pub mod encoders;
pub mod qformer;
pub mod tokenizer;
pub mod action_head;
pub mod config;
pub mod types;
pub mod error;
pub mod utils;
pub mod prelude;

// Re-export main components
pub use config::*;
pub use types::*;
pub use error::*;

/// Main CAFFEINE implementation
pub struct Caffeine {
    config: CaffeineConfig,
    encoders: crate::caffeine::encoders::MultiModalEncoders,
    qformer: crate::caffeine::qformer::TriQueryFormer,
    tokenizer: crate::caffeine::tokenizer::UnifiedTokenizer,
    action_head: crate::caffeine::action_head::AgenticActionHead,
    
    // Integration with existing modules
    atqs_compression: Option<crate::atqs::compression::CompressionEngine>,
    has_moe_router: Option<crate::has_moe_ffn::router::ExpertRouter>,
}

impl Caffeine {
    /// Create new CAFFEINE instance
    pub fn new(config: CaffeineConfig) -> crate::caffeine::error::Result<Self> {
        let encoders = crate::caffeine::encoders::MultiModalEncoders::new(config.encoders_config.clone())?;
        let qformer = crate::caffeine::qformer::TriQueryFormer::new(config.qformer_config.clone())?;
        let tokenizer = crate::caffeine::tokenizer::UnifiedTokenizer::new(config.tokenizer_config.clone())?;
        let action_head = crate::caffeine::action_head::AgenticActionHead::new(config.action_config.clone())?;
        
        // Initialize ATQS compression if enabled
        let atqs_compression = if config.enable_atqs_compression {
            Some(crate::atqs::compression::CompressionEngine::new(config.atqs_config.clone().unwrap_or_default())?)
        } else {
            None
        };
        
        // Initialize HAS-MoE-FFN router if enabled
        let has_moe_router = if config.enable_has_moe_routing {
            Some(crate::has_moe_ffn::router::ExpertRouter::new(config.has_moe_config.clone().unwrap_or_default())?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            encoders,
            qformer,
            tokenizer,
            action_head,
            atqs_compression,
            has_moe_router,
        })
    }
    
    /// Forward pass through CAFFEINE pipeline
    pub fn forward(&mut self, inputs: &crate::caffeine::types::MultiModalInputs) -> crate::caffeine::error::Result<crate::caffeine::types::MultiModalOutputs> {
        // Stage 1: Multi-modal encoding
        let encoded_features = self.encoders.encode(inputs)?;
        
        // Stage 2: Tri-query transformation
        let query_features = self.qformer.transform(&encoded_features)?;
        
        // Stage 3: Tokenization
        let tokens = self.tokenizer.tokenize(&query_features)?;
        
        // Stage 4: Apply ATQS compression if enabled
        let compressed_tokens = if let Some(ref mut compression) = self.atqs_compression {
            compression.compress(tokens)?
        } else {
            tokens
        };
        
        // Stage 5: Apply HAS-MoE-FFN routing if enabled
        let routed_tokens = if self.has_moe_router.is_some() {
            // Convert to format expected by HAS-MoE-FFN
            let tensor_input = self.tokens_to_tensor(&compressed_tokens)?;
            if let Some(ref mut router) = self.has_moe_router {
                let routing_decisions = router.route(&tensor_input)?;
                self.apply_routing(compressed_tokens, routing_decisions)?
            } else {
                compressed_tokens
            }
        } else {
            compressed_tokens
        };
        
        // Stage 6: Action head processing
        let outputs = self.action_head.process(routed_tokens, inputs)?;
        
        Ok(outputs)
    }
    
    /// Convert tokens to tensor format for HAS-MoE-FFN
    fn tokens_to_tensor(&self, tokens: &[crate::caffeine::types::UnifiedToken]) -> crate::caffeine::error::Result<ndarray::ArrayD<f32>> {
        // Convert tokens to tensor representation
        let mut data = Vec::with_capacity(tokens.len() * 768); // Assuming 768-dim embeddings
        for token in tokens {
            data.extend(token.embedding.iter());
        }
        
        let shape = vec![tokens.len(), 768];
        Ok(ndarray::ArrayD::from_shape_vec(shape, data)?)
    }
    
    /// Apply routing decisions to tokens
    fn apply_routing(&self, tokens: Vec<crate::caffeine::types::UnifiedToken>, _routing_decisions: Vec<crate::has_moe_ffn::types::RoutingDecision>) -> crate::caffeine::error::Result<Vec<crate::caffeine::types::UnifiedToken>> {
        // For now, return tokens unchanged
        // TODO: Implement actual routing logic
        Ok(tokens)
    }
    
    /// Get configuration
    pub fn config(&self) -> &CaffeineConfig {
        &self.config
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> crate::caffeine::types::PerformanceStats {
        crate::caffeine::types::PerformanceStats {
            total_tokens_processed: 0,
            compression_ratio: self.atqs_compression.as_ref().map(|c| c.get_compression_ratio()).unwrap_or(1.0),
            routing_efficiency: self.has_moe_router.as_ref().map(|r| r.get_routing_stats().load_balance_score).unwrap_or(1.0),
            average_latency_ms: 0.0,
            memory_usage_mb: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_caffeine_creation() {
        let config = CaffeineConfig::default();
        let caffeine = Caffeine::new(config);
        assert!(caffeine.is_ok());
    }
}
