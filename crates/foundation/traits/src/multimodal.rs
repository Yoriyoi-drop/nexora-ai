//! Multimodal Backend Trait
//!
//! Defines the interface for multimodal implementations (CAFFEINE, etc.)

use async_trait::async_trait;
use ndarray::ArrayD;
use std::collections::HashMap;
use crate::FoundationResult;

/// Multimodal input
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub text: Option<String>,
    pub image: Option<ArrayD<f32>>,
    pub audio: Option<ArrayD<f32>>,
    pub video: Option<ArrayD<f32>>,
    pub metadata: MultimodalMetadata,
}

#[derive(Debug, Clone)]
pub struct MultimodalMetadata {
    pub input_type: InputType,
    pub dimensions: Vec<usize>,
    pub format: String,
}

#[derive(Debug, Clone)]
pub enum InputType {
    TextOnly,
    ImageOnly,
    AudioOnly,
    VideoOnly,
    TextImage,
    TextAudio,
    TextVideo,
    MultiModal,
}

/// Multimodal output
#[derive(Debug, Clone)]
pub struct MultimodalOutput {
    pub embeddings: ArrayD<f32>,
    pub features: HashMap<String, ArrayD<f32>>,
    pub attention_maps: Option<Vec<ArrayD<f32>>>,
    pub metadata: OutputMetadata,
}

#[derive(Debug, Clone)]
pub struct OutputMetadata {
    pub output_dim: Vec<usize>,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

/// Core multimodal backend trait
#[async_trait]
pub trait MultimodalBackend: Send + Sync {
    /// Process multimodal input
    async fn process(&self, input: MultimodalInput) -> FoundationResult<MultimodalOutput>;
    
    /// Get supported input types
    fn supported_types(&self) -> Vec<InputType>;
    
    /// Fuse multiple modalities
    async fn fuse(&self, inputs: Vec<MultimodalInput>) -> FoundationResult<MultimodalOutput>;
    
    /// Extract features from specific modality
    async fn extract_features(&self, input: MultimodalInput, modality: &str) -> FoundationResult<ArrayD<f32>>;
}
