//! Configuration for HAS-MoE-FFN

use crate::has_moe_ffn::types::*;
use serde::{Deserialize, Serialize};

/// Main configuration for HAS-MoE-FFN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HasMoeFfnConfig {
    pub model_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub router_config: RouterConfig,
    pub expert_config: SwiGLUExpertConfig,
    pub aggregation_config: AggregationConfig,
    pub load_balancer_config: LoadBalancerConfig,
    pub training_config: TrainingConfig,
    pub performance_config: PerformanceConfig,
}

impl Default for HasMoeFfnConfig {
    fn default() -> Self {
        Self {
            model_dim: 4096,
            num_experts: 8,
            top_k: 2,
            router_config: RouterConfig::default(),
            expert_config: SwiGLUExpertConfig::default(),
            aggregation_config: AggregationConfig::default(),
            load_balancer_config: LoadBalancerConfig::default(),
            training_config: TrainingConfig::default(),
            performance_config: PerformanceConfig::default(),
        }
    }
}

impl HasMoeFfnConfig {
    /// Create new configuration with custom parameters
    pub fn new(
        model_dim: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        let mut config = Self::default();
        config.model_dim = model_dim;
        config.num_experts = num_experts;
        config.top_k = top_k;
        
        // Update dependent configurations
        config.router_config.num_experts = num_experts;
        config.router_config.top_k = top_k;
        config.expert_config.input_dim = model_dim;
        config.expert_config.output_dim = model_dim;
        
        config
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), crate::has_moe_ffn::error::HasMoeFfnError> {
        if self.model_dim == 0 {
            return Err(crate::has_moe_ffn::error::HasMoeFfnError::config(
                "model_dim must be > 0"
            ));
        }
        
        if self.num_experts == 0 {
            return Err(crate::has_moe_ffn::error::HasMoeFfnError::config(
                "num_experts must be > 0"
            ));
        }
        
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(crate::has_moe_ffn::error::HasMoeFfnError::config(
                "top_k must be > 0 and <= num_experts"
            ));
        }
        
        if self.expert_config.input_dim != self.model_dim {
            return Err(crate::has_moe_ffn::error::HasMoeFfnError::config(
                "expert input_dim must match model_dim"
            ));
        }
        
        if self.expert_config.output_dim != self.model_dim {
            return Err(crate::has_moe_ffn::error::HasMoeFfnError::config(
                "expert output_dim must match model_dim"
            ));
        }
        
        Ok(())
    }
    
    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        let expert_params = self.expert_parameters();
        let router_params = self.router_parameters();
        let aggregation_params = self.aggregation_parameters();
        
        expert_params + router_params + aggregation_params
    }
    
    /// Get expert parameters count
    pub fn expert_parameters(&self) -> usize {
        let gate_params = self.expert_config.input_dim * self.expert_config.hidden_dim;
        let up_params = self.expert_config.input_dim * self.expert_config.hidden_dim;
        let down_params = self.expert_config.hidden_dim * self.expert_config.output_dim;
        
        // Apply structured matrix reduction
        let reduction_factor = if self.expert_config.matrix_config.use_low_rank {
            0.3 // 70% reduction
        } else {
            1.0
        };
        
        ((gate_params + up_params + down_params) as f32 * reduction_factor) as usize
    }
    
    /// Get router parameters count
    pub fn router_parameters(&self) -> usize {
        // Router network parameters
        self.model_dim * self.num_experts * 2 // gate and up projections
    }
    
    /// Get aggregation parameters count
    pub fn aggregation_parameters(&self) -> usize {
        match self.aggregation_config.method {
            AggregationMethod::WeightedSum => self.num_experts,
            AggregationMethod::Attention => self.model_dim * self.num_experts,
            AggregationMethod::Gating => self.model_dim * self.num_experts,
            AggregationMethod::LearnedMixing => self.model_dim * self.num_experts * 2,
        }
    }
    
    /// Get memory usage estimate in MB
    pub fn memory_usage_mb(&self) -> f32 {
        let params = self.total_parameters();
        let param_size_mb = (params * 4) as f32 / (1024.0 * 1024.0); // f32 parameters
        let activation_memory_mb = (self.model_dim * self.top_k * 4) as f32 / (1024.0 * 1024.0);
        
        param_size_mb + activation_memory_mb + 50.0 // overhead buffer
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub gradient_clip_norm: f32,
    pub use_mixed_precision: bool,
    pub checkpoint_interval: usize,
    pub expert_dropout: f32,
    pub router_dropout: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            warmup_steps: 1000,
            max_steps: 100000,
            gradient_clip_norm: 1.0,
            use_mixed_precision: true,
            checkpoint_interval: 1000,
            expert_dropout: 0.1,
            router_dropout: 0.1,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub use_cuda: bool,
    pub use_flash_attention: bool,
    pub memory_efficient: bool,
    pub parallel_experts: bool,
    pub expert_parallel_degree: usize,
    pub cache_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 2048,
            use_cuda: true,
            use_flash_attention: true,
            memory_efficient: true,
            parallel_experts: true,
            expert_parallel_degree: 4,
            cache_size: 1000,
        }
    }
}

/// Specialized configurations for different use cases
impl HasMoeFfnConfig {
    /// Configuration for small models
    pub fn small_model() -> Self {
        Self::new(2048, 4, 2)
    }
    
    /// Configuration for medium models
    pub fn medium_model() -> Self {
        Self::new(4096, 8, 2)
    }
    
    /// Configuration for large models
    pub fn large_model() -> Self {
        Self::new(8192, 16, 4)
    }
    
    /// Configuration for inference-optimized models
    pub fn inference_optimized() -> Self {
        let mut config = Self::medium_model();
        config.performance_config.memory_efficient = true;
        config.performance_config.parallel_experts = true;
        config.expert_config.activation_dropout = 0.0;
        config.training_config.expert_dropout = 0.0;
        config.training_config.router_dropout = 0.0;
        config
    }
    
    /// Configuration for training-optimized models
    pub fn training_optimized() -> Self {
        let mut config = Self::medium_model();
        config.training_config.use_mixed_precision = true;
        config.training_config.expert_dropout = 0.1;
        config.training_config.router_dropout = 0.1;
        config.performance_config.use_cuda = true;
        config
    }
}
