//! Deep Learning Library for Nexora AI — hub crate
//!
//! Re-exports from sub-crates:
//! - `autograd`  — tape-based autograd engine (nexora-autograd)
//! - `star_x`    — STAR-X tensor ops (nexora-star-x)
//! - `gnac`      — GNAC graph composer (nexora-gnac)
//! - `echo_net`  — ECHO-Net Ω (nexora-echo-net)

use thiserror::Error;

/// Unified error type supporting conversions from all sub-crate error types.
#[derive(Error, Debug)]
pub enum DeepLearningError {
    #[error("{0}")]
    Autograd(#[from] nexora_autograd::DeepLearningError),
    #[error("{0}")]
    StarX(#[from] nexora_star_x::DeepLearningError),
    #[error("{0}")]
    Gnac(#[from] nexora_gnac::DeepLearningError),
    #[error("{0}")]
    EchoNet(#[from] nexora_echo_net::DeepLearningError),
    #[error("Configuration error: {reason}")]
    Configuration { reason: String },
}

impl From<ndarray::ShapeError> for DeepLearningError {
    fn from(err: ndarray::ShapeError) -> Self {
        DeepLearningError::Autograd(nexora_autograd::DeepLearningError::from(err))
    }
}

pub type DLResult<T> = Result<T, DeepLearningError>;

pub mod autograd {
    pub use nexora_autograd::*;
}
pub mod star_x {
    pub use nexora_star_x::*;
}
pub mod gnac {
    pub use nexora_gnac::*;
}
pub mod echo_net {
    pub use nexora_echo_net::*;
}

#[cfg(feature = "gpu")]
pub use nexora_autograd::gpu;
