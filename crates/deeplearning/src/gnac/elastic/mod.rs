//! Elastic Inference Graph — runtime adaptive architecture
//!
//! Model dapat menonaktifkan cabang mahal, mengurangi precision,
//! mengganti jalur inferensi, atau menyesuaikan depth berdasarkan
//! hardware dan kompleksitas input.

pub mod router;
pub mod precision;
pub mod depth;

pub use router::*;
pub use precision::*;
pub use depth::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;

/// Strategi adaptasi elastis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElasticStrategy {
    Lightweight,   // Performa maksimal
    Balanced,      // Trade-off
    HighPrecision, // Akurasi maksimal
}
