pub mod alignment;
/// ORACLE - Optimized Retrieval-Augmented Code Learning Engine
///
/// Arsitektur pelatihan LLM generasi berikutnya yang menyatukan 6 metode
/// menjadi satu pipeline terpadu untuk pemahaman dan generasi kode skala besar.
pub mod backbone;
pub mod code_utils;
/// Context compression — perpendek input sebelum masuk Oracle (Fix 4)
pub mod context_compression;
pub mod linters;
/// Memory-mapped Oracle — load weight via mmap, bukan load seluruh file (Fix 7)
pub mod mmap;
/// Oracle Pool — maksimal 2 instance, NXR lain antre (Fix 5)
pub mod pool;
pub mod pretraining;
/// Quantization — FP8/Q8 untuk OracleBackbone weights (Fix 3)
pub mod quantization;
pub mod rope;
/// Shared Oracle Memory — global KV store, semua NXR baca sumber sama (Fix 2)
pub mod shared_memory;
pub mod trainer;
/// Oracle TTL — time-to-live eviction untuk segment Oracle (Fix 6)
pub mod ttl;
// Re-export main components for easier access
pub use alignment::*;
pub use backbone::*;
pub use code_utils::*;
pub use context_compression::*;
pub use linters::*;
pub use mmap::*;
pub use pool::*;
pub use pretraining::*;
pub use quantization::*;
pub use rope::*;
pub use shared_memory::*;
pub use trainer::*;
pub use ttl::*;

/// Prelude module untuk import umum
pub mod prelude {
    pub use super::alignment::*;
    pub use super::backbone::*;
    pub use super::code_utils::*;
    pub use super::context_compression::*;
    pub use super::linters::*;
    pub use super::mmap::*;
    pub use super::pool::*;
    pub use super::pretraining::*;
    pub use super::quantization::*;
    pub use super::rope::*;
    pub use super::shared_memory::*;
    pub use super::trainer::*;
    pub use super::ttl::*;
}
