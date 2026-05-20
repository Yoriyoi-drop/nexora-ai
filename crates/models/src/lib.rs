//! NXR Model Series — 10 model types with agent-based architectures

// Re-export shared components from nexora-shared
pub use nexora_shared::*;

// Model series modules
pub mod aether;
pub mod axiom;
pub mod cipher;
pub mod genesis;
pub mod kronos;
pub mod nexum;
pub mod omnis;
pub mod spectra;
pub mod swift;
pub mod vortex;

// Foundation module (CausalLM-based primary models)
pub mod foundation;
