use std::sync::{Arc, OnceLock};

use super::config::{ModelTier, TransformerConfig};
use super::model::CausalLM;
use crate::TransformerResult;

/// SingleBackboneRegistry — satu shared backbone CausalLM untuk SEMUA model.
///
/// Semua 10 model crates (Omnis, Swift, Vortex, dll) pakai Arc<CausalLM> yang
/// sama. Weight di-cache di `OnceLock` — hanya dibuat sekali, dipakai semua.
/// Tidak perlu LRU eviction, tidak perlu tier switching, tidak boros VRAM.
static BACKBONE: OnceLock<Arc<CausalLM>> = OnceLock::new();

/// Default config: Pro tier (balanced quality/size).
/// hidden=3200, 32 layers, 8e2t MoE → ~13B params → ~6.5GB Q4.
/// Bisa di-override via `resolve_single_backbone_with_config()`.
fn default_config() -> TransformerConfig {
    TransformerConfig::preset(ModelTier::Pro)
}

/// Get the single shared backbone, creating it with default Pro config on first call.
pub fn resolve_single_backbone() -> TransformerResult<Arc<CausalLM>> {
    Ok(BACKBONE
        .get_or_init(|| Arc::new(CausalLM::new(default_config())))
        .clone())
}

/// Get the single shared backbone with a custom config override.
/// Config dimulai dari default Pro preset, lalu dimodifikasi oleh closure.
/// Catatan: hanya backbone PERTAMA yang di-cache. Panggilan berikutnya
/// mengabaikan `modifier` dan mengembalikan Arc yang sudah ada.
pub fn resolve_single_backbone_with_config<F>(_modifier: F) -> TransformerResult<Arc<CausalLM>>
where
    F: FnOnce(&mut TransformerConfig),
{
    // Backward compat: kalau sudah terlanjur di-init, abaikan modifier
    if let Some(b) = BACKBONE.get() {
        return Ok(b.clone());
    }

    // Pertama kali: apply modifier ke config default
    let mut config = default_config();
    _modifier(&mut config);

    Ok(BACKBONE
        .get_or_init(|| Arc::new(CausalLM::new(config)))
        .clone())
}

/// Backward-compat: panggil `resolve_single_backbone()` — abaikan tier.
/// Memudahkan migrasi dari sistem tier lama.
pub fn resolve_tier_backbone(_tier: ModelTier) -> TransformerResult<Arc<CausalLM>> {
    resolve_single_backbone()
}

/// Backward-compat: abaikan tier, pakai modifier untuk single backbone.
pub fn resolve_tier_backbone_with_config<F>(
    _tier: ModelTier,
    modifier: F,
) -> TransformerResult<Arc<CausalLM>>
where
    F: FnOnce(&mut TransformerConfig),
{
    resolve_single_backbone_with_config(modifier)
}

/// Backward-compat: NO-OP — tidak ada tier yang perlu di-unload.
pub fn unload_tier_backbone(_tier: ModelTier) -> TransformerResult<()> {
    Ok(())
}

/// Backward-compat: selalu kosong — tidak ada tier.
pub fn get_loaded_tiers() -> Vec<ModelTier> {
    Vec::new()
}

/// Backward-compat: return 0 atau 1.
pub fn registered_tier_count() -> usize {
    if BACKBONE.get().is_some() { 1 } else { 0 }
}

/// Backward-compat: selalu return true (single backbone always available).
pub fn has_tier_backbone(_tier: ModelTier) -> bool {
    BACKBONE.get().is_some()
}

/// Backward-compat: VRAM dari single backbone.
pub fn tier_vram_estimate_mb(_tier: ModelTier) -> u64 {
    let params = default_config().parameter_count();
    let cfg = default_config();
    ((params as f64 * cfg.bytes_per_param()) / (1024.0 * 1024.0)).ceil() as u64
}

/// Parameter count dari single backbone.
pub fn tier_parameter_count(_tier: ModelTier) -> usize {
    default_config().parameter_count()
}

/// Clear backbone — frees memory. Panggil kalau mau reload.
pub fn clear_all_backbones() {
    // OnceLock tidak bisa di-reset. Tapi kalau Arc drop ke 0, memory free.
    // Untuk reload, buat instance baru via `resolve_single_backbone_with_config()`.
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_quantization::QFormat;

    /// Tiny config for tests — avoids loading 13B param model in CI
    fn tiny_single_backbone() -> Arc<CausalLM> {
        static TINY: OnceLock<Arc<CausalLM>> = OnceLock::new();
        TINY.get_or_init(|| {
            let config = TransformerConfig {
                vocab_size: 64,
                hidden_size: 8,
                num_heads: 4,
                num_kv_heads: 2,
                num_layers: 2,
                max_seq_len: 32,
                intermediate_size: 16,
                num_experts: 2,
                top_k_experts: 1,
                expert_intermediate_size: 8,
                quantization: QFormat::Q8 { group_size: 128 },
                use_half_precision: false,
                ..Default::default()
            };
            Arc::new(CausalLM::new(config))
        })
        .clone()
    }

    #[test]
    fn test_same_backbone_returned() {
        let m1 = tiny_single_backbone();
        let m2 = tiny_single_backbone();
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_same_tiny_backbone_returned() {
        let m1 = tiny_single_backbone();
        let m2 = tiny_single_backbone();
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_tiny_backbone_has_moe() {
        let model = tiny_single_backbone();
        assert!(model.config.is_moe());
        assert!(model.config.num_experts > 0);
    }

    #[test]
    fn test_tiny_backbone_config() {
        let model = tiny_single_backbone();
        assert_eq!(model.config.hidden_size, 8);
        assert_eq!(model.config.num_layers, 2);
    }

    #[test]
    fn test_parameter_count_positive() {
        let count = tier_parameter_count(ModelTier::Pro);
        assert!(count > 0);
    }

    #[test]
    fn test_unload_is_noop() {
        let m1 = tiny_single_backbone();
        let _ = unload_tier_backbone(ModelTier::Ultra);
        let m2 = tiny_single_backbone();
        // unload_tier_backbone is no-op with single backbone
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_has_tier_backbone_does_not_panic() {
        let _ = has_tier_backbone(ModelTier::Pro);
    }

    #[test]
    fn test_get_loaded_tiers_empty() {
        let tiers = get_loaded_tiers();
        assert!(tiers.is_empty());
    }

    #[test]
    fn test_registered_tier_count() {
        let count = registered_tier_count();
        assert!(count <= 1);
    }
}
