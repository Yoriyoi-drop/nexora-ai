//! NXR Model Implementations
//! 
//! Foundation implementations for all 10 NXR model series

// Individual model implementations (each model has its own directory)
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

pub mod transformer;

// Foundation model types — each wraps OnceLock<CausalLM> as the core transformer backbone.
// These are the primary model implementations used by the training pipeline.
pub mod foundation;

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

/// Get model implementation by ID — returns the foundation (CausalLM-backed) model.
/// Per-directory agent-based models are still available via their module paths.
pub fn get_model_implementation(model_id: NxrModelId) -> Box<dyn std::any::Any> {
    match model_id {
        NxrModelId::Omnis => Box::new(foundation::NxrOmnisModel::new()),
        NxrModelId::Vortex => Box::new(foundation::NxrVortexModel::new()),
        NxrModelId::Aether => Box::new(foundation::NxrAetherModel::new()),
        NxrModelId::Spectra => Box::new(foundation::NxrSpectraModel::new()),
        NxrModelId::Nexum => Box::new(foundation::NxrNexumModel::new()),
        NxrModelId::Axiom => Box::new(foundation::NxrAxiomModel::new()),
        NxrModelId::Cipher => Box::new(foundation::NxrCipherModel::new()),
        NxrModelId::Swift => Box::new(foundation::NxrSwiftModel::new()),
        NxrModelId::Kronos => Box::new(foundation::NxrKronosModel::new()),
        NxrModelId::Genesis => Box::new(foundation::NxrGenesisModel::new()),
    }
}
