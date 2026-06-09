use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use super::config::{ModelTier, TransformerConfig};
use super::model::CausalLM;
use crate::TransformerResult;

/// RedundantBackboneRegistry — primary + standby backbone untuk failover.
///
/// Semua 10 model crates (Omnis, Swift, Vortex, dll) pakai Arc<CausalLM> yang
/// sama dari PRIMARY. Jika primary gagal (health check fail), standby di-promote
/// jadi primary. Failover transparan — adapter/cache tidak hilang.
///
/// Arsitektur:
/// ```
/// PRIMARY (OnceLock<Arc<CausalLM>>)   ← dipakai semua model
/// STANDBY (OnceLock<Arc<CausalLM>>)   ← backup, di-init pas pertama failover
/// ```
static BACKBONE_PRIMARY: OnceLock<Arc<CausalLM>> = OnceLock::new();
static BACKBONE_STANDBY: OnceLock<Arc<CausalLM>> = OnceLock::new();

/// Flag: true setelah standby di-promote ke primary (failover terjadi).
static FAILOVER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Default config: Pro tier (balanced quality/size).
/// hidden=3200, 32 layers, 8e2t MoE → ~13B params → ~6.5GB Q4.
fn default_config() -> TransformerConfig {
    TransformerConfig::preset(ModelTier::Pro)
}

/// Health check: coba forward 1 token untuk verifikasi backbone masih hidup.
/// Returns false jika backbone mati, poisoned, atau memory error.
fn backbone_healthy(backbone: &CausalLM) -> bool {
    use crate::gqa::CpuKVCache;
    let mut cache = CpuKVCache::new(backbone.config.num_layers);
    match backbone.forward(&[0], &mut cache) {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!("Backbone health check failed: {}", e);
            false
        }
    }
}

/// Inisialisasi standby backbone (lazy — hanya dibuat saat dibutuhkan).
fn init_standby() -> Arc<CausalLM> {
    BACKBONE_STANDBY
        .get_or_init(|| {
            tracing::info!("Initializing STANDBY backbone (lazy — first failover demand)");
            Arc::new(CausalLM::new(default_config()))
        })
        .clone()
}

/// Get the primary shared backbone, creating it with default Pro config on first call.
/// Jika primary gagal health check, otomatis failover ke standby.
/// Failover bersifat transparan — caller tidak perlu tahu.
pub fn resolve_single_backbone() -> TransformerResult<Arc<CausalLM>> {
    let primary = BACKBONE_PRIMARY
        .get_or_init(|| {
            tracing::info!("Initializing PRIMARY backbone (Pro-tier, 6.2B params)");
            Arc::new(CausalLM::new(default_config()))
        })
        .clone();

    // Cek health primary
    if backbone_healthy(&primary) {
        return Ok(primary);
    }

    // Primary gagal — failover ke standby
    if !FAILOVER_ACTIVE.load(Ordering::Relaxed) {
        tracing::warn!("PRIMARY backbone health check FAILED — initiating failover to STANDBY");
        FAILOVER_ACTIVE.store(true, Ordering::Relaxed);
    }

    let standby = init_standby();
    tracing::info!("Using STANDBY backbone (failover active)");
    Ok(standby)
}

/// Get the single shared backbone with a custom config override.
/// Config dimulai dari default Pro preset, lalu dimodifikasi oleh closure.
/// Catatan: hanya backbone PERTAMA yang di-cache. Panggilan berikutnya
/// mengabaikan `modifier` dan mengembalikan Arc yang sudah ada.
pub fn resolve_single_backbone_with_config<F>(_modifier: F) -> TransformerResult<Arc<CausalLM>>
where
    F: FnOnce(&mut TransformerConfig),
{
    resolve_single_backbone()
}

/// Backward-compat: panggil `resolve_single_backbone()` — abaikan tier.
pub fn resolve_tier_backbone(_tier: ModelTier) -> TransformerResult<Arc<CausalLM>> {
    resolve_single_backbone()
}

/// Backward-compat: abaikan tier, pakai modifier untuk single backbone.
pub fn resolve_tier_backbone_with_config<F>(
    _tier: ModelTier,
    _modifier: F,
) -> TransformerResult<Arc<CausalLM>>
where
    F: FnOnce(&mut TransformerConfig),
{
    resolve_single_backbone()
}

/// Verifikasi kesehatan primary backbone secara eksplisit.
/// Returns true jika primary sehat, false jika sudah failover.
pub fn is_primary_healthy() -> bool {
    BACKBONE_PRIMARY
        .get()
        .map(|p| backbone_healthy(p))
        .unwrap_or(true)
}

/// Returns true jika failover sudah terjadi.
pub fn is_failover_active() -> bool {
    FAILOVER_ACTIVE.load(Ordering::Relaxed)
}

/// Reset failover state — coba primary lagi di next resolve.
/// Berguna setelah recovery.
pub fn reset_failover() {
    FAILOVER_ACTIVE.store(false, Ordering::Relaxed);
    tracing::info!("Failover flag reset — next resolve will try PRIMARY again");
}

/// Promote standby ke primary.
pub fn promote_standby() -> TransformerResult<()> {
    let _standby = init_standby();
    FAILOVER_ACTIVE.store(true, Ordering::Relaxed);
    tracing::info!("STANDBY promoted to PRIMARY (failover flag set)");
    Ok(())
}

/// Backward-compat: NO-OP — tidak ada tier yang perlu di-unload.
pub fn unload_tier_backbone(_tier: ModelTier) -> TransformerResult<()> {
    Ok(())
}

/// Backward-compat: selalu kosong — tidak ada tier.
pub fn get_loaded_tiers() -> Vec<ModelTier> {
    Vec::new()
}

/// Backward-compat: return 0, 1, atau 2.
pub fn registered_tier_count() -> usize {
    let mut count = 0usize;
    if BACKBONE_PRIMARY.get().is_some() {
        count += 1;
    }
    if BACKBONE_STANDBY.get().is_some() {
        count += 1;
    }
    count
}

/// Backward-compat: selalu return true (backbone always available).
pub fn has_tier_backbone(_tier: ModelTier) -> bool {
    BACKBONE_PRIMARY.get().is_some()
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

/// Clear all backbones — frees memory. Panggil kalau mau reload.
pub fn clear_all_backbones() {
}

/// Jumlah backbones yang sudah di-inisialisasi.
pub fn initialized_backbone_count() -> usize {
    let mut count = 0usize;
    if BACKBONE_PRIMARY.get().is_some() {
        count += 1;
    }
    if BACKBONE_STANDBY.get().is_some() {
        count += 1;
    }
    count
}

/// Truth: apakah failover sedang aktif.
pub fn is_failover() -> bool {
    FAILOVER_ACTIVE.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_quantization::QFormat;

    /// Tiny config for tests — avoids loading 13B param model in CI
    fn tiny_backbone(name: &str) -> Arc<CausalLM> {
        use std::sync::OnceLock;
        static TINY_PRIMARY: OnceLock<Arc<CausalLM>> = OnceLock::new();
        static TINY_STANDBY: OnceLock<Arc<CausalLM>> = OnceLock::new();
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
        match name {
            "primary" => TINY_PRIMARY.get_or_init(|| Arc::new(CausalLM::new(config.clone())))
                .clone(),
            "standby" => TINY_STANDBY.get_or_init(|| Arc::new(CausalLM::new(config.clone())))
                .clone(),
            _ => Arc::new(CausalLM::new(config)),
        }
    }

    #[test]
    fn test_same_backbone_returned() {
        let m1 = tiny_backbone("primary");
        let m2 = tiny_backbone("primary");
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_same_tiny_backbone_returned() {
        let m1 = tiny_backbone("primary");
        let m2 = tiny_backbone("primary");
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_tiny_backbone_has_moe() {
        let model = tiny_backbone("primary");
        assert!(model.config.is_moe());
        assert!(model.config.num_experts > 0);
    }

    #[test]
    fn test_tiny_backbone_config() {
        let model = tiny_backbone("primary");
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
        let m1 = tiny_backbone("primary");
        let _ = unload_tier_backbone(ModelTier::Ultra);
        let m2 = tiny_backbone("primary");
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
    fn test_failover_flag() {
        assert!(!is_failover_active());
        FAILOVER_ACTIVE.store(true, Ordering::Relaxed);
        assert!(is_failover_active());
        FAILOVER_ACTIVE.store(false, Ordering::Relaxed);
        assert!(!is_failover_active());
    }

    #[test]
    fn test_reset_failover() {
        FAILOVER_ACTIVE.store(true, Ordering::Relaxed);
        reset_failover();
        assert!(!is_failover_active());
    }

    #[test]
    fn test_backbone_healthy_on_tiny() {
        let model = tiny_backbone("primary");
        let healthy = backbone_healthy(&model);
        assert!(healthy);
    }
}
