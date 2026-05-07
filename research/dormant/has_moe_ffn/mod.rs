//! Hybrid Adaptive Structured MoE-FFN (HAS-MoE-FFN)
//! 
//! Implementasi lengkap dari arsitektur MoE-FFN yang menggabungkan:
//! - SwiGLU Activation dengan structured matrices
//! - Dynamic expert routing yang context-aware
//! - Load balancing mechanisms
//! - Efficient aggregation layer

pub mod config;
pub mod router;
pub mod experts;
pub mod aggregation;
pub mod load_balancer;
pub mod types;
pub mod error;
pub mod prelude;


// Re-export main components
pub use config::*;
pub use router::*;
pub use experts::*;
pub use aggregation::*;
pub use load_balancer::*;
pub use types::*;
pub use error::*;

/// Main HAS-MoE-FFN implementation
pub struct HasMoeFfn {
    config: HasMoeFfnConfig,
    router: ExpertRouter,
    experts: Vec<crate::has_moe_ffn::experts::StructuredSwiGLUExpert>,
    aggregator: AggregationLayer,
    load_balancer: LoadBalancer,
}

impl HasMoeFfn {
    /// Create new HAS-MoE-FFN instance
    pub fn new(config: HasMoeFfnConfig) -> crate::has_moe_ffn::error::Result<Self> {
        let router = ExpertRouter::new(config.router_config.clone())?;
        let mut experts = Vec::with_capacity(config.num_experts);
        
        for i in 0..config.num_experts {
            let expert = crate::has_moe_ffn::experts::StructuredSwiGLUExpert::new(
                config.expert_config.clone(),
                format!("expert_{}", i),
            )?;
            experts.push(expert);
        }
        
        let aggregator = AggregationLayer::new(config.aggregation_config.clone())?;
        let load_balancer = LoadBalancer::new(config.load_balancer_config.clone())?;
        
        Ok(Self {
            config,
            router,
            experts,
            aggregator,
            load_balancer,
        })
    }
    
    /// Forward pass through HAS-MoE-FFN
    pub fn forward(&mut self, input: &ndarray::ArrayD<f32>) -> crate::has_moe_ffn::error::Result<ndarray::ArrayD<f32>> {
        // Step 1: Route input to selected experts
        let routing_decisions = self.router.route(input)?;
        
        // Step 2: Load balancing
        let balanced_routing = self.load_balancer.balance(routing_decisions, &self.experts)?;
        
        // Step 3: Process through selected experts
        let mut expert_outputs = Vec::new();
        for (expert_idx, expert_input) in balanced_routing {
            if expert_idx < self.experts.len() {
                let output = self.experts[expert_idx].forward(&expert_input)?;
                let expert_output = ExpertOutput {
                    expert_id: expert_idx,
                    output,
                    confidence: 0.8, // Default confidence
                    computation_cost: 1.0, // Default cost
                    specialization: self.experts[expert_idx].specialization().clone(),
                };
                expert_outputs.push(expert_output);
            }
        }
        
        // Step 4: Aggregate expert outputs
        let aggregated = self.aggregator.aggregate(expert_outputs, input)?;
        
        Ok(aggregated)
    }
    
    /// Get configuration
    pub fn config(&self) -> &HasMoeFfnConfig {
        &self.config
    }
    
    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }
    
    /// Get expert by index
    pub fn expert(&self, index: usize) -> Option<&crate::has_moe_ffn::experts::StructuredSwiGLUExpert> {
        self.experts.get(index)
    }
    
    /// Get routing statistics
    pub fn get_routing_stats(&self) -> crate::has_moe_ffn::router::RoutingStats {
        use std::collections::HashMap;
        
        // Placeholder implementation - would return actual routing statistics
        let mut expert_utilization = HashMap::new();
        for i in 0..self.experts.len() {
            expert_utilization.insert(i, 0.0);
        }
        
        crate::has_moe_ffn::router::RoutingStats {
            total_routings: 0,
            expert_utilization,
            average_confidence: 0.0,
            load_balance_score: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_has_moe_ffn_creation() {
        let config = HasMoeFfnConfig::default();
        let has_moe = HasMoeFfn::new(config);
        assert!(has_moe.is_ok());
    }
}
