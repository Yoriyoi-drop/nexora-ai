//! NXR Model Implementations
//! 
//! Foundation implementations for all 10 NXR model series

#[path = "stub.rs"]
pub mod omnis;
#[path = "stub.rs"]
pub mod vortex;
#[path = "stub.rs"]
pub mod aether;
#[path = "stub.rs"]
pub mod spectra;
#[path = "stub.rs"]
pub mod nexum;
#[path = "stub.rs"]
pub mod axiom;
#[path = "stub.rs"]
pub mod cipher;
#[path = "stub.rs"]
pub mod swift;
#[path = "stub.rs"]
pub mod kronos;
#[path = "stub.rs"]
pub mod genesis;

// Re-export all models
pub use omnis::*;
pub use vortex::*;
pub use aether::*;
pub use spectra::*;
pub use nexum::*;
pub use axiom::*;
pub use cipher::*;
pub use swift::*;
pub use kronos::*;
pub use genesis::*;

use crate::shared::model_identity::NxrModelId;

/// Get model implementation by ID
pub fn get_model_implementation(model_id: NxrModelId) -> Box<dyn std::any::Any> {
    match model_id {
        NxrModelId::Omnis => Box::new(omnis::NxrOmnisModel::new()),
        NxrModelId::Vortex => Box::new(vortex::NxrVortexModel::new()),
        NxrModelId::Aether => Box::new(aether::NxrAetherModel::new()),
        NxrModelId::Spectra => Box::new(spectra::NxrSpectraModel::new()),
        NxrModelId::Nexum => Box::new(nexum::NxrNexumModel::new()),
        NxrModelId::Axiom => Box::new(axiom::NxrAxiomModel::new()),
        NxrModelId::Cipher => Box::new(cipher::NxrCipherModel::new()),
        NxrModelId::Swift => Box::new(swift::NxrSwiftModel::new()),
        NxrModelId::Kronos => Box::new(kronos::NxrKronosModel::new()),
        NxrModelId::Genesis => Box::new(genesis::NxrGenesisModel::new()),
    }
}
