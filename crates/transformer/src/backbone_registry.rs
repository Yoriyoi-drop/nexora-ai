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
/// Core tier: dimensi sendiri (hidden=2048, layers=20, heads=16, dense/no MoE).
/// Disimpan sebagai entri terpisah di registry.
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
            // Core: dedicated preset — hidden=2048, layers=20, heads=16, dense (no MoE)
            TransformerConfig::preset(ModelTier::Core)
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

/// Unload a specific tier's backbone from the registry — frees VRAM.
pub fn unload_tier_backbone(tier: ModelTier) -> TransformerResult<()> {
    let mut reg = registry().write().map_err(|e| {
        crate::TransformerError::Implementation(format!(
            "TierBackboneRegistry lock poisoned: {}",
            e
        ))
    })?;
    if reg.remove(&tier).is_some() {
        tracing::info!("Unloaded tier {:?} backbone", tier);
    }
    Ok(())
}

/// List all currently loaded tier backbones.
pub fn get_loaded_tiers() -> Vec<ModelTier> {
    registry()
        .read()
        .map(|r| r.keys().copied().collect())
        .unwrap_or_default()
}

/// Estimate VRAM usage (in MB) for a loaded tier backbone at its configured quantization.
pub fn tier_vram_estimate_mb(tier: ModelTier) -> u64 {
    let params = tier_parameter_count(tier);
    let cfg = TransformerConfig::preset(tier);
    ((params as f64 * cfg.bytes_per_param() as f64) / (1024.0 * 1024.0)).ceil() as u64
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
    TransformerConfig::preset(tier).parameter_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ShardConfig;
    use nexora_quantization::QFormat;

    fn tiny_config(is_core: bool) -> TransformerConfig {
        TransformerConfig {
            vocab_size: 16,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 1,
            max_seq_len: if is_core { 4 } else { 8 },
            intermediate_size: 8,
            rope_theta: 10000.0,
            use_cache: true,
            norm_eps: 1e-6,
            num_experts: if is_core { 0 } else { 2 },
            top_k_experts: if is_core { 0 } else { 1 },
            expert_intermediate_size: if is_core { 0 } else { 4 },
            quantization: QFormat::Q8 { group_size: 128 },
            use_half_precision: false,
            shard: ShardConfig::default(),
            shared_expert: 0,
            use_domain_experts: false,
        }
    }

    fn resolve_tiny_backbone(is_core: bool) -> TransformerResult<Arc<CausalLM>> {
        let config = tiny_config(is_core);
        let tier = if is_core { ModelTier::Core } else { ModelTier::Pro };
        if let Ok(reg) = registry().read() {
            if let Some(backbone) = reg.get(&tier) {
                return Ok(Arc::clone(backbone));
            }
        }
        let model = Arc::new(CausalLM::new(config));
        let mut reg = registry().write().map_err(|e| {
            crate::TransformerError::Implementation(format!(
                "TierBackboneRegistry lock poisoned: {}",
                e
            ))
        })?;
        reg.entry(tier).or_insert_with(|| Arc::clone(&model));
        Ok(model)
    }

    #[test]
    fn test_same_tier_shares_backbone() {
        clear_all_backbones();
        let m1 = resolve_tiny_backbone(false).unwrap();
        let m2 = resolve_tiny_backbone(false).unwrap();
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_core_has_pro_dimensions_but_no_experts() {
        clear_all_backbones();
        let pro = resolve_tiny_backbone(false).unwrap();
        let core = resolve_tiny_backbone(true).unwrap();

        assert_eq!(pro.config.hidden_size, core.config.hidden_size);
        assert_eq!(pro.config.num_layers, core.config.num_layers);
        assert_eq!(pro.config.num_heads, core.config.num_heads);

        assert!(pro.config.is_moe());
        assert!(!core.config.is_moe());
        assert_eq!(core.config.num_experts, 0);

        assert!(core.config.max_seq_len < pro.config.max_seq_len);

        assert!(!Arc::ptr_eq(&pro, &core));
    }

    #[test]
    fn test_tiers_independent() {
        clear_all_backbones();

        let ultra = {
            let cfg = TransformerConfig {
                vocab_size: 16,
                hidden_size: 8,
                num_heads: 4,
                num_kv_heads: 2,
                num_layers: 2,
                max_seq_len: 16,
                intermediate_size: 16,
                num_experts: 4,
                top_k_experts: 2,
                expert_intermediate_size: 8,
                ..tiny_config(false)
            };
            let tier = ModelTier::Ultra;
            if let Ok(reg) = registry().read() {
                if let Some(b) = reg.get(&tier) {
                    Arc::clone(b)
                } else {
                    drop(reg);
                    let m = Arc::new(CausalLM::new(cfg));
                    let mut reg = registry().write().unwrap();
                    reg.entry(tier).or_insert_with(|| Arc::clone(&m));
                    m
                }
            } else {
                let m = Arc::new(CausalLM::new(cfg));
                let mut reg = registry().write().unwrap();
                reg.entry(tier).or_insert_with(|| Arc::clone(&m));
                m
            }
        };

        let apex = {
            let cfg = TransformerConfig {
                hidden_size: 6,
                num_heads: 2,
                num_kv_heads: 1,
                intermediate_size: 12,
                ..tiny_config(false)
            };
            let tier = ModelTier::Apex;
            if let Ok(reg) = registry().read() {
                if let Some(b) = reg.get(&tier) {
                    Arc::clone(b)
                } else {
                    drop(reg);
                    let m = Arc::new(CausalLM::new(cfg));
                    let mut reg = registry().write().unwrap();
                    reg.entry(tier).or_insert_with(|| Arc::clone(&m));
                    m
                }
            } else {
                let m = Arc::new(CausalLM::new(cfg));
                let mut reg = registry().write().unwrap();
                reg.entry(tier).or_insert_with(|| Arc::clone(&m));
                m
            }
        };

        let edge = {
            let cfg = TransformerConfig {
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                num_layers: 1,
                intermediate_size: 8,
                max_seq_len: 4,
                ..tiny_config(false)
            };
            let tier = ModelTier::Edge;
            if let Ok(reg) = registry().read() {
                if let Some(b) = reg.get(&tier) {
                    Arc::clone(b)
                } else {
                    drop(reg);
                    let m = Arc::new(CausalLM::new(cfg));
                    let mut reg = registry().write().unwrap();
                    reg.entry(tier).or_insert_with(|| Arc::clone(&m));
                    m
                }
            } else {
                let m = Arc::new(CausalLM::new(cfg));
                let mut reg = registry().write().unwrap();
                reg.entry(tier).or_insert_with(|| Arc::clone(&m));
                m
            }
        };

        assert!(!Arc::ptr_eq(&ultra, &apex));
        assert!(!Arc::ptr_eq(&ultra, &edge));
        assert!(!Arc::ptr_eq(&apex, &edge));
    }

    #[test]
    fn test_core_parameter_count_less_than_pro() {
        let pro_params = tier_parameter_count(ModelTier::Pro);
        let core_params = tier_parameter_count(ModelTier::Core);
        assert!(core_params < pro_params);
        assert!(core_params > 0);
    }

    #[test]
    fn test_registry_deduplicates() {
        clear_all_backbones();
        assert_eq!(registered_tier_count(), 0);

        resolve_tiny_backbone(false).unwrap();
        {
            let cfg = TransformerConfig {
                hidden_size: 8,
                num_heads: 4,
                num_kv_heads: 2,
                intermediate_size: 16,
                num_experts: 4,
                top_k_experts: 2,
                expert_intermediate_size: 8,
                ..tiny_config(false)
            };
            let tier = ModelTier::Ultra;
            let m = Arc::new(CausalLM::new(cfg));
            let mut reg = registry().write().unwrap();
            reg.entry(tier).or_insert_with(|| Arc::clone(&m));
        }
        resolve_tiny_backbone(false).unwrap(); // duplicate Pro
        resolve_tiny_backbone(true).unwrap();

        assert_eq!(registered_tier_count(), 3);
    }
}
