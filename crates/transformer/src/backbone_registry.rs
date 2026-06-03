use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use super::config::{ModelTier, TransformerConfig};
use super::model::CausalLM;
use crate::TransformerResult;

/// TierBackboneRegistry — shared backbone per tier.
///
/// Memastikan model dalam tier yang sama (misal Ultra: Omnis, Axiom, Genesis)
/// menggunakan satu instance `Arc<CausalLM>` yang sama. Weight di-load sekali,
/// semua model di tier itu pakai bersama.
///
/// Core tier: dimensi identik dengan Pro (hidden=3200, layers=32, heads=32),
/// tapi dense (no MoE). Disimpan sebagai entri terpisah di registry.
/// Pro dan Core tetap bisa share KV cache runtime karena dimensi sama.
static REGISTRY: OnceLock<RwLock<HashMap<ModelTier, Arc<CausalLM>>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<ModelTier, Arc<CausalLM>>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Clear all cached backbones — frees memory.
pub fn clear_all_backbones() {
    if let Ok(mut reg) = registry().write() {
        reg.clear();
    }
}

/// Get the backbone for a specific tier, creating it if necessary.
///
/// - Ultra/Apex/Pro/Edge: dibuat dari preset, di-cache per tier
/// - Core: dibuat dari preset Pro tapi dengan MoE dimatikan (num_experts=0),
///   disimpan sebagai entri `ModelTier::Core` sendiri
pub fn resolve_tier_backbone(tier: ModelTier) -> TransformerResult<Arc<CausalLM>> {
    // Fast path: coba read lock dulu
    if let Ok(reg) = registry().read() {
        if let Some(backbone) = reg.get(&tier) {
            return Ok(Arc::clone(backbone));
        }
    }

    // Slow path: create backbone baru
    let config = match tier {
        ModelTier::Core => {
            // Core: pakai dimensi Pro (hidden=3200, layers=32, heads=32)
            // tapi dense (num_experts=0), dan context window 500K
            let mut c = TransformerConfig::preset(ModelTier::Pro);
            c.num_experts = 0;
            c.top_k_experts = 0;
            c.expert_intermediate_size = 0;
            c.max_seq_len = 500_000;
            c
        }
        _ => TransformerConfig::preset(tier),
    };

    let model = Arc::new(CausalLM::new(config));

    {
        let mut reg = registry().write().map_err(|e| {
            crate::TransformerError::Implementation(format!(
                "TierBackboneRegistry lock poisoned: {}",
                e
            ))
        })?;
        // Double-check: mungkin sudah di-create oleh thread lain
        reg.entry(tier).or_insert_with(|| Arc::clone(&model));
    }

    Ok(model)
}

/// Get the backbone with custom config overrides.
/// Config dimulai dari preset tier, lalu dimodifikasi oleh closure.
pub fn resolve_tier_backbone_with_config<F>(
    tier: ModelTier,
    modifier: F,
) -> TransformerResult<Arc<CausalLM>>
where
    F: FnOnce(&mut TransformerConfig),
{
    // Cek cache dulu (fast path)
    if let Ok(reg) = registry().read() {
        if let Some(backbone) = reg.get(&tier) {
            return Ok(Arc::clone(backbone));
        }
    }

    let mut config = TransformerConfig::preset(tier);
    modifier(&mut config);

    let model = Arc::new(CausalLM::new(config));

    {
        let mut reg = registry().write().map_err(|e| {
            crate::TransformerError::Implementation(format!(
                "TierBackboneRegistry lock poisoned: {}",
                e
            ))
        })?;
        reg.entry(tier).or_insert_with(|| Arc::clone(&model));
    }

    Ok(model)
}

/// Get the number of registered backbones.
pub fn registered_tier_count() -> usize {
    registry().read().map(|r| r.len()).unwrap_or(0)
}

/// Check if a tier's backbone is already registered.
pub fn has_tier_backbone(tier: ModelTier) -> bool {
    registry()
        .read()
        .map(|r| r.contains_key(&tier))
        .unwrap_or(false)
}

/// Parameter count for a tier's backbone.
pub fn tier_parameter_count(tier: ModelTier) -> usize {
    match tier {
        ModelTier::Core => {
            let mut c = TransformerConfig::preset(ModelTier::Pro);
            c.num_experts = 0;
            c.top_k_experts = 0;
            c.expert_intermediate_size = 0;
            c.parameter_count()
        }
        _ => TransformerConfig::preset(tier).parameter_count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_tier_shares_backbone() {
        clear_all_backbones();
        let ultra = resolve_tier_backbone(ModelTier::Ultra).unwrap();
        let ultra2 = resolve_tier_backbone(ModelTier::Ultra).unwrap();
        assert!(Arc::ptr_eq(&ultra, &ultra2));
    }

    #[test]
    fn test_core_has_pro_dimensions_but_no_experts() {
        clear_all_backbones();
        let pro = resolve_tier_backbone(ModelTier::Pro).unwrap();
        let core = resolve_tier_backbone(ModelTier::Core).unwrap();

        // Dimensi sama
        assert_eq!(pro.config.hidden_size, core.config.hidden_size);
        assert_eq!(pro.config.num_layers, core.config.num_layers);
        assert_eq!(pro.config.num_heads, core.config.num_heads);

        // Tapi experts berbeda
        assert!(pro.config.is_moe());
        assert!(!core.config.is_moe());
        assert_eq!(core.config.num_experts, 0);

        // Context window Core lebih kecil
        assert!(core.config.max_seq_len < pro.config.max_seq_len);

        // Backbone terpisah (bukan Arc yang sama)
        assert!(!Arc::ptr_eq(&pro, &core));
    }

    #[test]
    fn test_tiers_independent() {
        clear_all_backbones();
        let ultra = resolve_tier_backbone(ModelTier::Ultra).unwrap();
        let apex = resolve_tier_backbone(ModelTier::Apex).unwrap();
        let edge = resolve_tier_backbone(ModelTier::Edge).unwrap();
        assert!(!Arc::ptr_eq(&ultra, &apex));
        assert!(!Arc::ptr_eq(&ultra, &edge));
        assert!(!Arc::ptr_eq(&apex, &edge));
    }

    #[test]
    fn test_core_parameter_count_less_than_pro() {
        let pro_params = tier_parameter_count(ModelTier::Pro);
        let core_params = tier_parameter_count(ModelTier::Core);
        assert!(core_params < pro_params);
        // Core ~4B vs Pro ~9B
        assert!(core_params > 0);
    }

    #[test]
    fn test_registry_deduplicates() {
        clear_all_backbones();
        assert_eq!(registered_tier_count(), 0);

        resolve_tier_backbone(ModelTier::Ultra).unwrap();
        resolve_tier_backbone(ModelTier::Pro).unwrap();
        resolve_tier_backbone(ModelTier::Pro).unwrap(); // duplicate
        resolve_tier_backbone(ModelTier::Core).unwrap();

        // Ultra, Pro, Core — 3 entries (Apex dan Edge belum di-resolve)
        assert_eq!(registered_tier_count(), 3);
    }
}
