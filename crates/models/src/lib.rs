//! NXR Model Series — model architecture definitions and agent-based implementations.
//!
//! All model types (Omnis, Vortex, Aether, …) are backed by the same CausalLM with
//! a different `TransformerConfig` size via `foundation` crate.
//! Per-model delegation modules inject classifiers (real MLP) and route to subsystems
//! (SACA reasoning, MoE gating, Caffeine multimodal, Oracle verifiers).
//!
//! Legacy `simulated-models` feature removed — all inference uses real neural networks.

// Re-export shared components from nexora-shared
pub use nexora_shared::*;

pub mod classifier_util;
pub mod foundation;
pub mod aether;
pub mod axiom;
pub mod cipher;
pub mod genesis;
pub mod kronos;
pub mod nexum;
pub mod omnis;
pub mod spectra;
pub mod swift;
pub mod vortex;
pub mod specialist;

use std::sync::Arc;
use nexora_shared::model_identity::NxrModelId;
use nexora_transformer::CausalLM;

/// Inject a CausalLM model instance into the corresponding delegation agent singleton.
/// Called during system initialization so the agent shares the registry's model.
pub fn wire_model(model_id: NxrModelId, model_arc: Arc<CausalLM>) {
    match model_id {
        NxrModelId::Omnis => omnis::delegation::inject_model(model_arc),
        NxrModelId::Vortex => vortex::delegation::inject_model(model_arc),
        NxrModelId::Aether => aether::delegation::inject_model(model_arc),
        NxrModelId::Spectra => spectra::delegation::inject_model(model_arc),
        NxrModelId::Nexum => nexum::delegation::inject_model(model_arc),
        NxrModelId::Axiom => axiom::delegation::inject_model(model_arc),
        NxrModelId::Cipher => cipher::delegation::inject_model(model_arc),
        NxrModelId::Swift => swift::delegation::inject_model(model_arc),
        NxrModelId::Kronos => kronos::delegation::inject_model(model_arc),
        NxrModelId::Genesis => genesis::delegation::inject_model(model_arc),
    }
}

// ─── Cross-layer integration (Phase 5 wiring) ───────────────────────
// Nyata: quantized weight storage untuk model weights
fn _models_quantize(weights: &ndarray::Array2<f32>, dtype: nexora_quantization::QuantizedDtype) -> nexora_quantization::QuantizedTensor {
    nexora_quantization::quantize_linear(weights, dtype)
}

// Nyata: database untuk model registry persistence
fn _models_db() -> nexora_database::DatabaseManager {
    nexora_database::DatabaseManager::new()
}

// Nyata: monitoring untuk model serving metrics
fn _models_monitoring() -> nexora_monitoring::MonitoringSystem {
    nexora_monitoring::MonitoringSystem::new(nexora_monitoring::MonitoringConfig::default())
}

// Nyata: memory for model context management
fn _models_memory() -> nexora_memory::MemoryManager {
    nexora_memory::MemoryManager::new()
}

// Nyata: utils for text processing in models
fn _models_utils() -> nexora_utils::UtilsManager {
    nexora_utils::UtilsManager::default()
}
