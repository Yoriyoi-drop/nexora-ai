//! Deep Learning Library for Nexora AI
//!
//! Implementasi modern deep learning architectures termasuk:
//! - ECHO-Net Ω (Entropic Contextual Holographic Oscillation Network)
//! - STAR-X (Selective Temporal Adaptive Resonance Network)
//! - RNN, LSTM, GRU layers
//! - Various deep learning components


pub mod star_x;
pub mod echo_net;

// Re-export main components
pub use echo_net::*;
pub use star_x::*;


use anyhow::Result;

/// Type alias untuk Result dengan DeepLearningError
pub type DLResult<T> = Result<T, DeepLearningError>;

/// Common error types for deep learning components
#[derive(Debug, thiserror::Error)]
pub enum DeepLearningError {
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Invalid input dimension: {dim}")]
    InvalidDimension { dim: usize },
    
    #[error("Memory allocation failed: {reason}")]
    MemoryAllocation { reason: String },
    
    #[error("Computation error: {reason}")]
    Computation { reason: String },
    
    #[error("Configuration error: {reason}")]
    Configuration { reason: String },
}

impl From<ndarray::ShapeError> for DeepLearningError {
    fn from(_err: ndarray::ShapeError) -> Self {
        DeepLearningError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
        }
    }
}


/// Common traits for deep learning components
pub mod traits {
    use super::*;
    
    /// Trait untuk forward pass
    pub trait Forward {
        type Input;
        type Output;
        
        fn forward(&self, input: &Self::Input) -> DLResult<Self::Output>;
    }
    
    /// Trait untuk backward pass (training)
    pub trait Backward {
        type Gradient;
        
        fn backward(&self, grad: &Self::Gradient) -> DLResult<Self::Gradient>;
    }
    
    /// Trait untuk stateful components (seperti RNN)
    pub trait Stateful {
        type State;
        
        fn reset_state(&mut self);
        fn get_state(&self) -> &Self::State;
        fn set_state(&mut self, state: Self::State);
    }
    
    /// Trait untuk components yang bisa di-train
    pub trait Trainable {
        fn parameters(&self) -> Vec<&[f32]>;
        fn parameters_mut(&mut self) -> Vec<&mut [f32]>;
        fn gradients(&self) -> Vec<&[f32]>;
        fn gradients_mut(&mut self) -> Vec<&mut [f32]>;
    }
}
