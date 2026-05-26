//! NXR Model Series — model architecture definitions and agent-based implementations.
//!
//! # Real vs Simulated
//!
//! **Real neural network inference** happens in:
//! - `foundation` — wraps `CausalLM` (transformer) behind the `NxrModel` trait.
//!   Every model ID (Omnis, Vortex, Aether, …) is backed by the same CausalLM with
//!   a different `TransformerConfig` size. This is the PRIMARY inference path.
//!
//! **Simulated / agent-based architectures** (gated behind `simulated-models` feature):
//! - Each model's `architecture` submodule (e.g. `omnis::architecture::OmnisArchitecture`)
//!   contains keyword-matching and template-returning "architectures". They are NOT
//!   neural networks. They exist for legacy agent orchestration flows.
//! - The per-model `NxrOmnisModel` etc. delegate actual `infer()` calls to the real
//!   `foundation` CausalLM-backed model, bypassing the fake architecture.
//!
//! Enable `simulated-models` feature to compile architecture submodules (default: off).

// Re-export shared components from nexora-shared
pub use nexora_shared::*;

// Foundation module — CausalLM-backed real inference (always available)
pub mod foundation;

// Per-model agent-based modules.
// The `architecture` submodule within each is gated behind `simulated-models`.
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