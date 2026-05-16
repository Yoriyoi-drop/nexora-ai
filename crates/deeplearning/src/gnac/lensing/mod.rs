//! Neural Graph Lensing — mode observasi kontekstual
//!
//! Solusi skalabilitas visual: alih-alih seluruh graf selalu terlihat,
//! GNAC menyediakan lensa khusus yang hanya menyorot node relevan
//! terhadap objective aktif. Mendukung hierarchical graph compression
//! dengan MetaNode yang tetap mempertahankan metrik agregat.

pub mod gradient_lens;
pub mod attention_lens;
pub mod latency_lens;
pub mod memory_lens;
pub mod entropy_lens;

pub use gradient_lens::*;
pub use attention_lens::*;
pub use latency_lens::*;
pub use memory_lens::*;
pub use entropy_lens::*;

use crate::gnac::canvas::NeuralGraph;
use uuid::Uuid;

/// Tipe lensa yang tersedia
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LensType {
    GradientFailure,
    AttentionFlow,
    Latency,
    Memory,
    ActivationEntropy,
}

/// Hasil observasi dari lensa
#[derive(Debug, Clone)]
pub struct LensObservation {
    pub lens_type: LensType,
    pub highlighted_nodes: Vec<Uuid>,
    pub highlighted_edges: Vec<Uuid>,
    pub summary: String,
    pub severity: ObservationSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObservationSeverity {
    Info,
    Warning,
    Critical,
}

/// Trait umum untuk semua lensa
pub trait NeuralLens {
    fn lens_type(&self) -> LensType;
    fn observe(&self, graph: &NeuralGraph) -> LensObservation;
    fn name(&self) -> &str;
}
