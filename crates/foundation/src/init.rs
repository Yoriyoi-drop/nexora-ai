use std::sync::Arc;
use tracing::info;

use nexora_transformer::TransformerConfig;

use crate::causal_lm_model::{CausalLmModel, MiniTokenizer};
use crate::shared::NxrModel;
use crate::shared::{
    capability_spec::predefined as cap_predefined,
    model_config::NxrModelConfig,
    model_identity::NxrModelId,
    model_registry::{global_registry, RegistryError},
    ModelMeta,
};

fn tier_config(model_id: NxrModelId, vocab_size: usize) -> TransformerConfig {
    let _cfg = NxrModelConfig::for_model(model_id);
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

    // Echo-Net APSS injection enabled by default for all models
    info!(
        "EchoNet APSS injection enabled after layer 2 for {}",
        model_id
    );

    // SEDC weight compression is opt-in — must be explicitly enabled
    info!(
        "SEDC weight compression — must be explicitly enabled for {}",
        model_id
    );

    let mut model = CausalLmModel::new(model_id, transformer_config.clone());
    model = model.with_echo_net(crate::causal_lm_model::EchoNetInjectionConfig::default());
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
    model
        .initialize(params)
        .await
        .map_err(|e| RegistryError::Validation(e.to_string()))?;

    let model_arc = Arc::new(model);
    let model_trait: Arc<
        dyn crate::shared::NxrModel<
            Config = serde_json::Value,
            Metrics = serde_json::Value,
            State = serde_json::Value,
        >,
    > = model_arc.clone();
    let model_raw: Arc<dyn std::any::Any + Send + Sync> = model_arc;
    registry
        .register_model_raw(model_id, model_trait, Some(model_raw), meta, caps, cfg)
        .await?;

    info!(
        "Registered {} | {} params ({:.1}M) ✓",
        model_id,
        pcount,
        pcount as f64 / 1_000_000.0
    );
    Ok(())
}

pub async fn initialize_foundation_models() -> Result<(), RegistryError> {
    let vocab_size = 50257;

    // Register only actively used models (Omnis by default).
    // Set NEXORA_ALL_MODELS=1 or enable feature "all-models" to load all 10.
    let model_ids: Vec<NxrModelId> = if std::env::var("NEXORA_ALL_MODELS").is_ok() {
        info!("NEXORA_ALL_MODELS set — registering all 10 NXR models");
        NxrModelId::all()
    } else {
        vec![NxrModelId::Omnis]
    };

    for model_id in model_ids {
        let tc = tier_config(model_id, vocab_size);
        register_causal_lm(model_id, vocab_size, tc).await?;
    }

    info!("Foundation model(s) registered ✓");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_shared::model_identity::NxrModelId;

    #[test]
    fn test_tier_config_omnis() {
        let config = tier_config(NxrModelId::Omnis, 50257);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 16);
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.vocab_size, 50257);
    }

    #[test]
    fn test_tier_config_axiom() {
        let config = tier_config(NxrModelId::Axiom, 50257);
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_layers, 10);
    }

    #[test]
    fn test_tier_config_genesis() {
        let config = tier_config(NxrModelId::Genesis, 50257);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 6);
    }

    #[test]
    fn test_tier_config_nexum() {
        let config = tier_config(NxrModelId::Nexum, 50257);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 6);
    }

    #[test]
    fn test_tier_config_low_tier() {
        for id in &[
            NxrModelId::Cipher,
            NxrModelId::Vortex,
            NxrModelId::Aether,
            NxrModelId::Spectra,
            NxrModelId::Swift,
            NxrModelId::Kronos,
        ] {
            let config = tier_config(*id, 100);
            assert_eq!(config.hidden_size, 128, "failed for {:?}", id);
            assert_eq!(config.num_layers, 3);
            assert_eq!(config.vocab_size, 100);
        }
    }

    #[test]
    fn test_tier_config_different_vocab_sizes() {
        let c1 = tier_config(NxrModelId::Swift, 100);
        let c2 = tier_config(NxrModelId::Swift, 50000);
        assert_eq!(c1.vocab_size, 100);
        assert_eq!(c2.vocab_size, 50000);
        assert_eq!(c1.hidden_size, c2.hidden_size); // same model config
    }

    #[test]
    fn test_tier_config_all_have_sane_values() {
        for id in NxrModelId::all() {
            let config = tier_config(id, 50257);
            assert!(config.hidden_size > 0, "hidden_size zero for {:?}", id);
            assert!(config.num_heads > 0);
            assert!(config.num_layers > 0);
            assert!(config.max_seq_len > 0);
            assert!(config.intermediate_size > 0);
        }
    }
}
