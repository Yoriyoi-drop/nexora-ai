//! NXR Model Implementations
//! 
//! Foundation implementations for all 10 NXR model series

pub mod omnis;
pub mod vortex;
pub mod aether;
pub mod spectra;
pub mod nexum;
pub mod axiom;
pub mod cipher;
pub mod swift;
pub mod kronos;
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
        NxrModelId::Omnis => Box::new(omnis::NxrOmnisModel),
        NxrModelId::Vortex => Box::new(vortex::NxrVortexModel),
        NxrModelId::Aether => Box::new(aether::NxrAetherModel),
        NxrModelId::Spectra => Box::new(spectra::NxrSpectraModel),
        NxrModelId::Nexum => Box::new(nexum::NxrNexumModel),
        NxrModelId::Axiom => Box::new(axiom::NxrAxiomModel),
        NxrModelId::Cipher => Box::new(cipher::NxrCipherModel),
        NxrModelId::Swift => Box::new(swift::NxrSwiftModel),
        NxrModelId::Kronos => Box::new(kronos::NxrKronosModel),
        NxrModelId::Genesis => Box::new(genesis::NxrGenesisModel),
    }
}
