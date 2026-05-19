use std::sync::Arc;
use tracing::info;

use nexora_transformer::TransformerConfig;

use crate::shared::{
    model_registry::{global_registry, RegistryError},
    model_identity::NxrModelId,
    model_config::NxrModelConfig,
    capability_spec::predefined as cap_predefined,
    ModelMeta, ModelTier,
};
use crate::causal_lm_model::{CausalLmModel, MiniTokenizer};
use crate::shared::NxrModel;

fn tier_config(model_id: NxrModelId, vocab_size: usize) -> TransformerConfig {
    let cfg = NxrModelConfig::for_model(model_id);
    let low = || TransformerConfig {
        vocab_size,
        hidden_size: 128,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 3,
        max_seq_len: 512,
        intermediate_size: 512,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        use_cache: true,
    };
    let mid = || TransformerConfig {
        vocab_size,
        hidden_size: 256,
        num_heads: 8,
        num_kv_heads: 4,
        num_layers: 6,
        max_seq_len: 1024,
        intermediate_size: 1024,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        use_cache: true,
    };
    let high = || TransformerConfig {
        vocab_size,
        hidden_size: 384,
        num_heads: 8,
        num_kv_heads: 4,
        num_layers: 10,
        max_seq_len: 1024,
        intermediate_size: 1536,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        use_cache: true,
    };
    let flagship = || TransformerConfig {
        vocab_size,
        hidden_size: 512,
        num_heads: 8,
        num_kv_heads: 4,
        num_layers: 16,
        max_seq_len: 2048,
        intermediate_size: 2048,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
        use_cache: true,
    };

    match model_id {
        NxrModelId::Omnis => flagship(),
        NxrModelId::Axiom => high(),
        NxrModelId::Genesis => mid(),
        NxrModelId::Nexum => mid(),
        NxrModelId::Cipher => low(),
        NxrModelId::Vortex => low(),
        NxrModelId::Aether => low(),
        NxrModelId::Spectra => low(),
        NxrModelId::Swift => low(),
        NxrModelId::Kronos => low(),
    }
}

/// Create and register a single causal LM model instance.
async fn register_causal_lm(
    model_id: NxrModelId,
    vocab_size: usize,
    transformer_config: TransformerConfig,
) -> Result<(), RegistryError> {
    let registry = global_registry();
    let cfg = NxrModelConfig::for_model(model_id);
    let caps = cap_predefined::get_capabilities(model_id);
    let meta = ModelMeta::new(
        model_id,
        model_id.tier(),
        "0.1.0".to_string(),
        model_id.fullname().to_string(),
    );

    let pcount = transformer_config.parameter_count();
    let mut model = CausalLmModel::new(model_id, transformer_config.clone());
    let mini_tok = MiniTokenizer::new(vocab_size);
    model.load_tokenizer(mini_tok).await;

    let params = serde_json::json!({
        "transformer_config": {
            "vocab_size": transformer_config.vocab_size,
            "hidden_size": transformer_config.hidden_size,
            "num_heads": transformer_config.num_heads,
            "num_kv_heads": transformer_config.num_kv_heads,
            "num_layers": transformer_config.num_layers,
            "max_seq_len": transformer_config.max_seq_len,
            "intermediate_size": transformer_config.intermediate_size,
        }
    });
    model.initialize(params).await
        .map_err(|e| RegistryError::Validation(e.to_string()))?;

    let model_arc = Arc::new(model);
    let model_trait: Arc<dyn crate::shared::NxrModel<Config = serde_json::Value, Metrics = serde_json::Value, State = serde_json::Value>> = model_arc.clone();
    let model_raw: Arc<dyn std::any::Any + Send + Sync> = model_arc;
    registry.register_model_raw(
        model_id,
        model_trait,
        Some(model_raw),
        meta,
        caps,
        cfg,
    ).await?;

    info!("Registered {} | {} params ({:.1}M) ✓", model_id, pcount, pcount as f64 / 1_000_000.0);
    Ok(())
}

pub async fn initialize_foundation_models() -> Result<(), RegistryError> {
    let vocab_size = 512;

    // Register ALL 10 NXR models with per-tier CausalLM instances
    for model_id in NxrModelId::all() {
        let tc = tier_config(model_id, vocab_size);
        register_causal_lm(model_id, vocab_size, tc).await?;
    }

    info!("All 10 NXR foundation models are ACTIVE ✓");
    Ok(())
}
