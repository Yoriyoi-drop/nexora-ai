use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tracing::{info, warn};

use nexora_models::foundation::transformer_config_for;
use nexora_transformer::TransformerConfig;

use crate::causal_lm_model::{CausalLmModel, MiniTokenizer};
use crate::shared::{
    capability_spec::predefined as cap_predefined,
    model_config::NxrModelConfig,
    model_identity::{ModelTier, NxrModelId},
    model_registry::{global_registry, RegistryError},
    ModelMeta,
};

/// Single source of truth for model config — delegates to `nexora_models::foundation`.
fn tier_config(model_id: NxrModelId, vocab_size: usize) -> TransformerConfig {
    let mut cfg = transformer_config_for(model_id);
    cfg.vocab_size = vocab_size;
    cfg
}

/// Semua model standby di startup — satu shared backbone di-load lazy
/// pas pertama kali ada request. Tidak ada tier, tidak ada eviction.

/// Create and register a single causal LM model instance (standby/lazy).
/// Semua model share satu backbone CausalLM dari SingleBackboneRegistry.
async fn register_causal_lm(
    model_id: NxrModelId,
    vocab_size: usize,
    transformer_config: TransformerConfig,
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

    info!(
        "EchoNet APSS injection enabled after layer 2 for {}",
        model_id
    );

    info!(
        "SEDC weight compression — must be explicitly enabled for {}",
        model_id
    );

    let mut model = CausalLmModel::new(model_id, transformer_config.clone());
    model = model.with_echo_net(crate::causal_lm_model::EchoNetInjectionConfig::default());
    let mini_tok = MiniTokenizer::new(vocab_size);
    model.load_tokenizer(mini_tok).await;

    // Lazy: semua model standby — backbone di-load on-demand via SingleBackboneRegistry
    let ckpt_path = checkpoints.get(&model_id);
    if let Some(path) = ckpt_path {
        model
            .initialize_empty()
            .await
            .map_err(|e| RegistryError::Validation(e.to_string()))?;

        if std::path::Path::new(path).exists() {
            info!("Loading checkpoint for model {} from {}", model_id, path);
            if let Err(e) = model.load_checkpoint(path).await {
                warn!("Failed to load checkpoint for {}: {} — using shared backbone", model_id, e);
            }
        } else {
            info!("Checkpoint path for {} not found: {} — using shared backbone", model_id, path);
        }
    } else {
        info!("Model {} — lazy/standby (shared backbone via SingleBackboneRegistry)", model_id);
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
        "Registered {} | {} params ({:.1}M) STANDBY (lazy) ✓",
        model_id,
        pcount,
        pcount as f64 / 1_000_000.0,
    );
    Ok(())
}

pub async fn initialize_foundation_models_with_checkpoints(
    checkpoints: HashMap<NxrModelId, String>,
) -> Result<(), RegistryError> {
    let vocab_size = 50257;

    // Initialize foundation subsystems (monitoring, memory, utils, erp)
    static INIT_CTX: OnceLock<crate::InitContext> = OnceLock::new();
    INIT_CTX.get_or_init(|| crate::init_subsystems());

    let model_ids = NxrModelId::all();

    for model_id in &model_ids {
        let tc = tier_config(*model_id, vocab_size);
        register_causal_lm(*model_id, vocab_size, tc, &checkpoints).await?;
    }

    wire_delegation_agents(&model_ids).await?;

    info!("All 10 NXR foundation models registered (lazy/standby — single backbone loads on-demand) ✓");
    Ok(())
}

/// Backward-compatible: no checkpoint paths, all standby models remain empty.
pub async fn initialize_foundation_models() -> Result<(), RegistryError> {
    initialize_foundation_models_with_checkpoints(HashMap::new()).await
}

/// Wire ALL 10 delegation agents WITHOUT loading backbone.
/// Setiap delegation agent panggil `resolve_single_backbone()` pas pertama kali infer.
async fn wire_delegation_agents(model_ids: &[NxrModelId]) -> Result<(), RegistryError> {
    for model_id in model_ids {
        info!(
            "Delegation agent for {} wired (lazy — single backbone loads on demand)",
            model_id
        );
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
        assert_eq!(c1.hidden_size, c2.hidden_size);
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
