use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

use nexora_models::foundation::transformer_config_for;
use nexora_models::wire_model;
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

/// Single source of truth for model config — delegates to `nexora_models::foundation`.
fn tier_config(model_id: NxrModelId, vocab_size: usize) -> TransformerConfig {
    let mut cfg = transformer_config_for(model_id);
    cfg.vocab_size = vocab_size;
    cfg
}

/// Model IDs that are initialized with full random weights (active at startup).
const ACTIVE_MODEL_IDS: [NxrModelId; 2] = [NxrModelId::Omnis, NxrModelId::Axiom];

/// Create and register a single causal LM model instance.
/// When `active` is true, the model gets full random weights (ready for inference).
/// When `active` is false, the model uses `new_empty()` — no block weights loaded,
/// suitable for lazy on-demand loading from checkpoint.
/// If a checkpoint path is provided (via `checkpoints`), standby models can
/// load pre-trained weights at startup.
async fn register_causal_lm(
    model_id: NxrModelId,
    vocab_size: usize,
    transformer_config: TransformerConfig,
    active: bool,
    checkpoints: &HashMap<NxrModelId, String>,
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

    if active {
        model
            .initialize(params)
            .await
            .map_err(|e| RegistryError::Validation(e.to_string()))?;
    } else {
        // Standby — register with empty model, load from checkpoint on demand
        model
            .initialize_empty()
            .await
            .map_err(|e| RegistryError::Validation(e.to_string()))?;

        // If a checkpoint path is configured, load weights now
        if let Some(ckpt_path) = checkpoints.get(&model_id) {
            if std::path::Path::new(ckpt_path).exists() {
                info!("Loading checkpoint for standby model {} from {}", model_id, ckpt_path);
                if let Err(e) = model.load_checkpoint(ckpt_path).await {
                    warn!("Failed to load checkpoint for {}: {} — remaining standby", model_id, e);
                }
            } else {
                info!("Checkpoint path for {} not found: {} — remaining standby", model_id, ckpt_path);
            }
        }
    }

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
        "Registered {} | {} params ({:.1}M) {} ✓",
        model_id,
        pcount,
        pcount as f64 / 1_000_000.0,
        if active { "ACTIVE" } else { "STANDBY" }
    );
    Ok(())
}

pub async fn initialize_foundation_models_with_checkpoints(
    checkpoints: HashMap<NxrModelId, String>,
) -> Result<(), RegistryError> {
    let vocab_size = 50257;

    let model_ids = NxrModelId::all();

    for model_id in &model_ids {
        let active = ACTIVE_MODEL_IDS.contains(model_id);
        let tc = tier_config(*model_id, vocab_size);
        register_causal_lm(*model_id, vocab_size, tc, active, &checkpoints).await?;
    }

    wire_delegation_agents(&model_ids).await?;

    info!("All 10 NXR foundation models registered (2 active, 8 standby) ✓");
    Ok(())
}

/// Backward-compatible: no checkpoint paths, all standby models remain empty.
pub async fn initialize_foundation_models() -> Result<(), RegistryError> {
    initialize_foundation_models_with_checkpoints(HashMap::new()).await
}

/// Wire delegation agents for models that have weights loaded.
async fn wire_delegation_agents(model_ids: &[NxrModelId]) -> Result<(), RegistryError> {
    let registry = global_registry();
    for model_id in model_ids {
        if let Ok(model_raw) = registry.get_model_raw(model_id).await {
            if let Some(causal_lm_model) = model_raw.downcast_ref::<CausalLmModel>() {
                if let Some(model_arc) = causal_lm_model.get_model_arc().await {
                    wire_model(*model_id, model_arc);
                    info!("Delegation agent wired for {}", model_id);
                } else {
                    info!("Delegation agent for {} — weights not loaded (standby)", model_id);
                }
            }
        }
    }
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
