//! Type definitions for CAFFEINE

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Unified token representation for all modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedToken {
    /// Token ID in unified vocabulary
    pub token_id: usize,
    /// Modality type
    pub modality: ModalityType,
    /// Embedding representation
    pub embedding: Vec<f32>,
    /// Position in sequence
    pub position: usize,
    /// Timestamp for temporal data
    pub timestamp: Option<f32>,
    /// Spatial coordinates for visual data
    pub spatial_coords: Option<(f32, f32, f32, f32)>, // (x, y, w, h)
}

/// Modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    Text,
    Image,
    Audio,
    Video,
    Action,
}

/// Multi-modal input representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalInputs {
    /// Text input
    pub text: Option<TextInput>,
    /// Image input
    pub image: Option<ImageInput>,
    /// Audio input
    pub audio: Option<AudioInput>,
    /// Video input
    pub video: Option<VideoInput>,
    /// Context information
    pub context: Option<ContextInfo>,
}

/// Image input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    /// Image data (raw pixels or encoded file bytes)
    pub data: Vec<u8>,
    /// Image format
    pub format: ImageFormat,
    /// Width (0 = auto-detect from encoded data if possible)
    pub width: usize,
    /// Height (0 = auto-detect from encoded data if possible)
    pub height: usize,
    /// Channels (3 = RGB, 4 = RGBA, 1 = Grayscale)
    pub channels: usize,
}

impl ImageInput {
    /// Create an ImageInput from raw pixel data with known dimensions.
    pub fn from_raw(data: Vec<u8>, width: usize, height: usize, channels: usize) -> Self {
        Self { data, format: ImageFormat::Raw, width, height, channels }
    }

    /// Create an ImageInput from encoded file bytes (PNG/JPEG).
    /// Parses the header to determine dimensions and format.
    /// Falls back to default 224x224 RGB if header parsing fails.
    pub fn from_encoded(data: Vec<u8>) -> Self {
        let mut input = Self { data, format: ImageFormat::PNG, width: 0, height: 0, channels: 3 };
        input.detect_format();
        if input.width == 0 { input.width = 224; }
        if input.height == 0 { input.height = 224; }
        input
    }

    /// Detect image format and parse dimensions from header.
    fn detect_format(&mut self) {
        if self.data.len() < 24 { return; }

        // PNG: 8-byte signature + IHDR chunk
        if self.data[0] == 0x89 && self.data[1] == b'P' && self.data[2] == b'N' && self.data[3] == b'G' {
            self.format = ImageFormat::PNG;
            // IHDR chunk starts at byte 16 (8 sig + 4 len + 4 "IHDR")
            if self.data.len() >= 24 {
                self.width = u32::from_be_bytes([self.data[16], self.data[17], self.data[18], self.data[19]]) as usize;
                self.height = u32::from_be_bytes([self.data[20], self.data[21], self.data[22], self.data[23]]) as usize;
                self.channels = if self.data.len() > 24 { match self.data[24] { 6 => 4, 2 => 3, 0 => 1, _ => 3 } } else { 3 };
            }
            return;
        }

        // JPEG: starts with FF D8
        if self.data[0] == 0xFF && self.data[1] == 0xD8 {
            self.format = ImageFormat::JPEG;
            // Parse SOF markers for dimensions
            let mut i = 2;
            while i + 8 < self.data.len() {
                if self.data[i] == 0xFF {
                    match self.data[i + 1] {
                        0xC0 | 0xC1 | 0xC2 => {
                            if i + 9 < self.data.len() {
                                self.height = u16::from_be_bytes([self.data[i + 5], self.data[i + 6]]) as usize;
                                self.width = u16::from_be_bytes([self.data[i + 7], self.data[i + 8]]) as usize;
                                self.channels = self.data[i + 9] as usize;
                            }
                            return;
                        }
                        0xD9 => return, // EOI
                        _ => {
                            let seg_len = u16::from_be_bytes([self.data[i + 2], self.data[i + 3]]) as usize;
                            i += seg_len + 2;
                            continue;
                        }
                    }
                }
                i += 1;
            }
        }
    }
}

/// Text input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInput {
    /// Raw text
    pub text: String,
    /// Tokenized representation
    pub tokens: Option<Vec<usize>>,
    /// Language
    pub language: String,
}

/// Audio input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInput {
    /// Audio data (float PCM samples)
    pub data: Vec<f32>,
    /// Sample rate
    pub sample_rate: usize,
    /// Duration in seconds
    pub duration: f32,
    /// Number of channels
    pub channels: usize,
}

impl AudioInput {
    /// Create an AudioInput from float PCM samples.
    pub fn from_pcm(data: Vec<f32>, sample_rate: usize, channels: usize) -> Self {
        let duration = if sample_rate > 0 {
            (data.len() as f32) / (sample_rate as f32 * channels as f32)
        } else {
            0.0
        };
        Self { data, sample_rate, duration, channels }
    }
}

/// Video input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInput {
    /// Frame data
    pub frames: Vec<ImageInput>,
    /// Frame rate
    pub frame_rate: usize,
    /// Duration in seconds
    pub duration: f32,
    /// Audio track (optional)
    pub audio: Option<AudioInput>,
}

/// Context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo {
    /// Task type
    pub task_type: TaskType,
    /// Instruction
    pub instruction: Option<String>,
    /// Previous actions
    pub previous_actions: Vec<Action>,
    /// Environment state
    pub environment_state: Option<EnvironmentState>,
}

/// Task types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Generation,
    Retrieval,
    Reasoning,
    Planning,
    Grounding,
    Summarization,
    Translation,
}

/// Action representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Action type
    pub action_type: ActionType,
    /// Parameters
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    Click,
    Type,
    Scroll,
    Drag,
    Wait,
    Navigate,
    Extract,
    Analyze,
}

/// Environment state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentState {
    /// Screen size
    pub screen_size: (usize, usize),
    /// Current application
    pub current_application: String,
    /// Available elements
    pub available_elements: Vec<UIElement>,
}

/// UI element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIElement {
    /// Element type
    pub element_type: String,
    /// Position
    pub position: (f32, f32),
    /// Size
    pub size: (f32, f32),
    /// Text content
    pub text: Option<String>,
    /// Attributes
    pub attributes: std::collections::HashMap<String, String>,
}

/// Multi-modal output representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalOutputs {
    /// Text output
    pub text: Option<TextOutput>,
    /// Image output
    pub image: Option<ImageOutput>,
    /// Audio output
    pub audio: Option<AudioOutput>,
    /// Video output
    pub video: Option<VideoOutput>,
    /// Action outputs
    pub actions: Vec<ActionOutput>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Text output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextOutput {
    /// Generated text
    pub text: String,
    /// Token probabilities
    pub token_probs: Option<Vec<f32>>,
    /// Confidence score
    pub confidence: f32,
}

/// Image output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOutput {
    /// Image data
    pub data: Vec<u8>,
    /// Format
    pub format: ImageFormat,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Description
    pub description: Option<String>,
}

/// Audio output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutput {
    /// Audio data
    pub data: Vec<f32>,
    /// Sample rate
    pub sample_rate: usize,
    /// Duration
    pub duration: f32,
    /// Transcription
    pub transcription: Option<String>,
}

/// Video output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOutput {
    /// Frame data
    pub frames: Vec<ImageOutput>,
    /// Frame rate
    pub frame_rate: usize,
    /// Duration
    pub duration: f32,
    /// Audio track
    pub audio: Option<AudioOutput>,
}

/// Action output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutput {
    /// Action
    pub action: Action,
    /// Execution result
    pub result: ExecutionResult,
    /// Execution time
    pub execution_time_ms: f32,
}

/// Execution result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionResult {
    Success,
    Failure,
    Partial,
    Timeout,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total processing time
    pub total_time_ms: f32,
    /// Encoding time
    pub encoding_time_ms: f32,
    /// Query transformation time
    pub query_time_ms: f32,
    /// Tokenization time
    pub tokenization_time_ms: f32,
    /// Action processing time
    pub action_time_ms: f32,
    /// Memory usage
    pub memory_usage_mb: f32,
    /// GPU utilization
    pub gpu_utilization_percent: f32,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total tokens processed
    pub total_tokens_processed: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Routing efficiency
    pub routing_efficiency: f32,
    /// Average latency
    pub average_latency_ms: f32,
    /// Memory usage
    pub memory_usage_mb: usize,
}

/// Encoded features from multi-modal encoders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedFeatures {
    /// Image features
    pub image_features: Option<ArrayD<f32>>,
    /// Audio features
    pub audio_features: Option<ArrayD<f32>>,
    /// Video features
    pub video_features: Option<ArrayD<f32>>,
    /// Text features
    pub text_features: Option<ArrayD<f32>>,
    /// Regional features (for spatial alignment)
    pub regional_features: Option<Vec<ArrayD<f32>>>,
}

/// Query features from Tri-Query Former
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Semantic query features
    pub semantic_features: ArrayD<f32>,
    /// Spatial query features
    pub spatial_features: ArrayD<f32>,
    /// Temporal query features
    pub temporal_features: ArrayD<f32>,
    /// Attention weights
    pub attention_weights: Option<ArrayD<f32>>,
}

/// Spatial grounding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialGrounding {
    /// Bounding boxes
    pub bounding_boxes: Vec<BoundingBox>,
    /// Segmentation masks
    pub segmentation_masks: Option<Vec<SegmentationMask>>,
    /// Confidence scores
    pub confidence_scores: Vec<f32>,
    /// Class labels
    pub class_labels: Vec<String>,
}

/// Bounding box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Coordinates (x1, y1, x2, y2)
    pub coords: (f32, f32, f32, f32),
    /// Format
    pub format: BBoxFormat,
    /// Label
    pub label: Option<String>,
    /// Confidence
    pub confidence: f32,
}

/// Segmentation mask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationMask {
    /// Mask data
    pub mask: Vec<u8>,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Label
    pub label: Option<String>,
    /// Confidence
    pub confidence: f32,
}

/// Action plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    /// Planned actions
    pub actions: Vec<Action>,
    /// Plan description
    pub description: String,
    /// Estimated duration
    pub estimated_duration_ms: f32,
    /// Success probability
    pub success_probability: f32,
}

/// Image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WEBP,
    BMP,
    TIFF,
    Raw,
}

/// Bounding box formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BBoxFormat {
    XYWH,
    XYXY,
    CXCYWH,
}

/// Vision model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionModelType {
    ViT,
    ResNet,
    CLIPViT,
    EfficientNet,
}

/// Audio model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioModelType {
    Whisper,
    Wav2Vec2,
    HuBERT,
    Data2Vec,
}

/// Video model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoModelType {
    ViT3D,
    VideoTransformer,
    TimeSformer,
}

/// Text model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextModelType {
    BERT,
    RoBERTa,
    GPT2,
    T5,
}
