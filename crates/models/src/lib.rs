//! NXR Model Series — model architecture definitions and agent-based implementations.
//!
//! # Real vs Simulated
//!
//! **Real neural network inference** happens in:
//! - `foundation` — wraps `CausalLM` (transformer) behind the `NxrModel` trait.
//!   Every model ID (Omnis, Vortex, Aether, …) is backed by the same CausalLM with
//!   a different `TransformerConfig` size. This is the PRIMARY inference path.
//!
//! **Simulated / agent-based architectures** (gated behind `simulated-models` feature):
//! - Each model's `architecture` submodule (e.g. `omnis::architecture::OmnisArchitecture`)
//!   contains keyword-matching and template-returning "architectures". They are NOT
//!   neural networks. They exist for legacy agent orchestration flows.
//! - The per-model `NxrOmnisModel` etc. delegate actual `infer()` calls to the real
//!   `foundation` CausalLM-backed model, bypassing the fake architecture.
//!
//! Enable `simulated-models` feature to compile architecture submodules (default: off).

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

// Nyata: isolation untuk model-level security check
fn _models_isolation_check(agent_id: uuid::Uuid) -> std::result::Result<(), nexora_isolation::IsolationCheckError> {
    let config = nexora_isolation::config::IsolationConfig::default();
    let orch = nexora_isolation::IsolationOrchestrator::new(config);
    orch.pre_inference_check(agent_id)
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
