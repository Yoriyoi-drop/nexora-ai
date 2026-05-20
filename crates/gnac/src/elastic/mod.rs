//! Elastic Inference Graph — runtime adaptive architecture
//!
//! Model dapat menonaktifkan cabang mahal, mengurangi precision,
//! mengganti jalur inferensi, atau menyesuaikan depth berdasarkan
//! hardware dan kompleksitas input.

pub mod depth;
pub mod precision;
pub mod router;

pub use depth::*;
pub use precision::*;
pub use router::*;

/// Strategi adaptasi elastis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElasticStrategy {
    Lightweight,   // Performa maksimal
    Balanced,      // Trade-off
    HighPrecision, // Akurasi maksimal
}
