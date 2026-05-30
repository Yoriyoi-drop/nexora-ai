pub mod alignment;
/// ORACLE - Optimized Retrieval-Augmented Code Learning Engine
///
/// Arsitektur pelatihan LLM generasi berikutnya yang menyatukan 6 metode
/// menjadi satu pipeline terpadu untuk pemahaman dan generasi kode skala besar.
pub mod backbone;
pub mod code_utils;
pub mod linters;
pub mod pretraining;
pub mod rope;
pub mod trainer;
#[deprecated(note = "renamed to linters — accurate naming for regex-based pattern detectors")]
pub mod verifiers {
    pub use super::linters::*;
}

// Re-export main components for easier access
pub use alignment::*;
pub use backbone::*;
pub use code_utils::*;
pub use linters::*;
pub use pretraining::*;
pub use rope::*;
pub use trainer::*;

/// Prelude module untuk import umum
pub mod prelude {
    pub use super::alignment::*;
    pub use super::backbone::*;
    pub use super::code_utils::*;
    pub use super::linters::*;
    pub use super::pretraining::*;
    pub use super::rope::*;
    pub use super::trainer::*;
}
