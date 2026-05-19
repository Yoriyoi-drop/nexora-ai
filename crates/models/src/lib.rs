//! NXR Model Series — 10 model types with agent-based architectures

// Re-export shared components from nexora-shared
pub use nexora_shared::*;

// Model series modules
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

// Foundation module (CausalLM-based primary models)
pub mod foundation;
