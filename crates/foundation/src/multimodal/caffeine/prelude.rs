//! Prelude module for CAFFEINE

pub use crate::multimodal::caffeine::config::*;
pub use crate::multimodal::caffeine::types::*;
pub use crate::multimodal::caffeine::error::*;

// Re-export encoders
pub use crate::multimodal::caffeine::encoders::*;
pub use crate::multimodal::caffeine::qformer::*;
pub use crate::multimodal::caffeine::tokenizer::*;
pub use crate::multimodal::caffeine::action_head::*;
pub use crate::multimodal::caffeine::utils::*;

// Re-export common types
pub use ndarray::{Array, ArrayD, ArrayView};
pub use std::error::Error;
pub use std::fmt::{Debug, Display};

// Re-export common types for easier access
pub use crate::multimodal::caffeine::types::{
    ModalityType, TaskType, ActionType, ExecutionResult,
    ImageFormat, BBoxFormat, VisionModelType, AudioModelType,
    VideoModelType, TextModelType,
};
