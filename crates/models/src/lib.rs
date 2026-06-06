//! NXR Model Series — re-export crate.
//!
//! Re-exports all 10 model crates under their canonical names
//! (omnis, vortex, aether, …) and provides the `wire_model` function
//! for injecting a CausalLM into all delegation singletons.

pub use nexora_model_core::*;
pub use nexora_model_omnis as omnis;
pub use nexora_model_vortex as vortex;
pub use nexora_model_aether as aether;
pub use nexora_model_axiom as axiom;
pub use nexora_model_cipher as cipher;
pub use nexora_model_genesis as genesis;
pub use nexora_model_kronos as kronos;
pub use nexora_model_nexum as nexum;
pub use nexora_model_spectra as spectra;
pub use nexora_model_swift as swift;

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
