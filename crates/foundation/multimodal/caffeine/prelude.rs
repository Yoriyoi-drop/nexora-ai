//! Prelude module for CAFFEINE

pub use crate::caffeine::config::*;
pub use crate::caffeine::types::*;
pub use crate::caffeine::error::*;

// Re-export encoders
pub use crate::caffeine::encoders::*;
pub use crate::caffeine::qformer::*;
pub use crate::caffeine::tokenizer::*;
pub use crate::caffeine::action_head::*;
pub use crate::caffeine::utils::*;

// Re-export common types
pub use ndarray::{Array, ArrayD, ArrayView};
pub use std::error::Error;
pub use std::fmt::{Debug, Display};

// Re-export common types for easier access
pub use crate::caffeine::types::{
    ModalityType, TaskType, ActionType, ExecutionResult,
    ImageFormat, BBoxFormat, VisionModelType, AudioModelType,
    VideoModelType, TextModelType,
};
