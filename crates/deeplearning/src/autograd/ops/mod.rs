pub mod macros;
pub mod activation;
pub mod math;
pub mod matmul;
pub mod nn;
pub mod reduce;
pub mod shape;
pub mod views;

use std::sync::atomic::AtomicU64;

/// Counter for GPU→CPU fallback events across math ops.
pub(crate) static GPU_MATH_FALLBACKS: AtomicU64 = AtomicU64::new(0);

pub fn gpu_math_fallback_count() -> u64 {
    GPU_MATH_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

pub use activation::{gelu, leaky_relu, relu, sigmoid, silu, swiglu, tanh};
pub use math::{add, div, exp, ln, mul, neg, powf, sqrt, sub};
pub use matmul::matmul;
pub use nn::{
    binary_cross_entropy, causal_softmax, cross_entropy_loss, dropout, embedding,
    layer_norm_2d, log_softmax, mse_loss, rms_norm_2d, softmax,
};
pub use nn::layer_norm_2d as layer_norm;
pub use reduce::{mean, sum};
pub use shape::{reshape, transpose};
pub use views::{cat, stack};
