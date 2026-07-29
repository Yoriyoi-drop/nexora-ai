//! Deep Learning Library for Nexora AI — merged crate
//!
//! Modules:
//! - `autograd`  — tape-based autograd engine
//! - `star_x`    — STAR-X tensor ops
//! - `gnac`      — GNAC graph composer
//! - `echo_net`  — ECHO-Net Ω
//! - `quantization` — weight quantization helpers

pub mod autograd;
pub mod star_x;
pub mod gnac;
pub mod echo_net;
pub mod quantization;

/// Re-export all autograd symbols at crate root for backward compatibility
pub use autograd::*;

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::autograd::gpu::*;
}
