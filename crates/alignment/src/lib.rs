// Foundation Alignment Framework (SPARO)
//
// Advanced AI alignment and safety system.
// Safety and Performance Alignment through Reinforcement Optimization (SPARO) framework
// untuk advanced AI alignment, safety, dan performance optimization.
// Now integrated with NXR-NEXUM for enhanced multi-agent alignment coordination.

pub mod sparo;
pub mod isolation;
pub mod hallucination;

// Re-export main components
pub use sparo::*;

// Hapus temporary — SparoNexumIntegration pindah ke crate terpisah untuk hindari circular dependency
