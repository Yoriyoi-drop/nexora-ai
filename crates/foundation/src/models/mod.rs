//! NXR Model Implementations
//! 
//! Foundation implementations for all 10 NXR model series

#[path = "foundation.rs"]
pub mod omnis;
#[path = "foundation.rs"]
pub mod vortex;
#[path = "foundation.rs"]
pub mod aether;
#[path = "foundation.rs"]
pub mod spectra;
#[path = "foundation.rs"]
pub mod nexum;
#[path = "foundation.rs"]
pub mod axiom;
#[path = "foundation.rs"]
pub mod cipher;
#[path = "foundation.rs"]
pub mod swift;
#[path = "foundation.rs"]
pub mod kronos;
#[path = "foundation.rs"]
pub mod genesis;

pub mod transformer;

// Re-export all models
pub use omnis::*;
#[allow(unused_imports)]
pub use vortex::*;
#[allow(unused_imports)]
pub use aether::*;
#[allow(unused_imports)]
pub use spectra::*;
#[allow(unused_imports)]
pub use nexum::*;
#[allow(unused_imports)]
pub use axiom::*;
#[allow(unused_imports)]
pub use cipher::*;
#[allow(unused_imports)]
pub use swift::*;
#[allow(unused_imports)]
pub use kronos::*;
#[allow(unused_imports)]
pub use genesis::*;
pub use transformer::{
    TransformerConfig, CausalLM, KVCacheEntry, RMSNorm, RoPE,
};

use crate::shared::model_identity::NxrModelId;
use serde::{Serialize, Deserialize};

/// Evaluation result for a single sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub input: String,
    pub target: String,
    pub predicted: String,
    pub loss: f32,
    pub correct: bool,
}

/// Complete evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub model_path: String,
    pub test_data_path: String,
    pub total_samples: usize,
    pub average_loss: f32,
    pub accuracy: f32,
    pub correct_predictions: usize,
    pub timestamp: String,
    pub detailed_results: Vec<EvaluationResult>,
}

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
