//! Model configuration mapping — NXR model ID → TransformerConfig + ModelTier.
//!
//! Single responsibility: define model architecture parameters for each model variant.

use nexora_deeplearning::quantization::QFormat;
use nexora_shared::model_identity::{ModelTier, NxrModelId};
use nexora_transformer::TransformerConfig;

/// Resolve a `TransformerConfig` for the given model ID.
pub fn transformer_config_for(model_id: NxrModelId) -> TransformerConfig {
    let vocab_size = 50257;
    let shared_q4 = QFormat::Q4 { group_size: 128 };
    match model_id {
        NxrModelId::Omnis => TransformerConfig {
            vocab_size,
            hidden_size: 512,
            num_heads: 8,
            num_kv_heads: 1,
            num_layers: 16,
            max_seq_len: 2048,
            intermediate_size: 2048,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_cache: true,
            num_experts: 8,
            top_k_experts: 2,
            expert_intermediate_size: 512,
            use_domain_experts: true,
            shared_expert: 4,
            quantization: shared_q4,
            use_half_precision: true,
            shard: Default::default(),
        },
        NxrModelId::Axiom => TransformerConfig {
            vocab_size,
            hidden_size: 384,
            num_heads: 8,
            num_kv_heads: 1,
            num_layers: 10,
            max_seq_len: 1024,
            intermediate_size: 1536,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_cache: true,
            num_experts: 8,
            top_k_experts: 2,
            expert_intermediate_size: 384,
            use_domain_experts: true,
            shared_expert: 4,
            quantization: shared_q4,
            use_half_precision: true,
            shard: Default::default(),
        },
        NxrModelId::Genesis | NxrModelId::Nexum => TransformerConfig {
            vocab_size,
            hidden_size: 256,
            num_heads: 8,
            num_kv_heads: 1,
            num_layers: 6,
            max_seq_len: 1024,
            intermediate_size: 1024,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_cache: true,
            num_experts: 4,
            top_k_experts: 1,
            expert_intermediate_size: 256,
            use_domain_experts: false,
            shared_expert: 2,
            quantization: shared_q4,
            use_half_precision: true,
            shard: Default::default(),
        },
        NxrModelId::Cipher
        | NxrModelId::Vortex
        | NxrModelId::Aether
        | NxrModelId::Spectra
        | NxrModelId::Swift
        | NxrModelId::Kronos => TransformerConfig {
            vocab_size,
            hidden_size: 128,
            num_heads: 4,
            num_kv_heads: 1,
            num_layers: 3,
            max_seq_len: 512,
            intermediate_size: 512,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
            use_cache: true,
            num_experts: 0,
            top_k_experts: 0,
            expert_intermediate_size: 0,
            use_domain_experts: false,
            shared_expert: 0,
            quantization: shared_q4,
            use_half_precision: true,
            shard: Default::default(),
        },
    }
}

/// Resolve the `ModelTier` for the given model ID.
pub fn model_tier_for(id: NxrModelId) -> ModelTier {
    match id {
        NxrModelId::Omnis | NxrModelId::Axiom | NxrModelId::Genesis => ModelTier::Ultra,
        NxrModelId::Vortex | NxrModelId::Aether | NxrModelId::Nexum => ModelTier::Apex,
        NxrModelId::Spectra | NxrModelId::Cipher => ModelTier::Pro,
        NxrModelId::Swift => ModelTier::Edge,
        NxrModelId::Kronos => ModelTier::Core,
    }
}
