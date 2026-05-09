//! Configuration types for CAFFEINE

use serde::{Deserialize, Serialize};
use crate::atqs::compression::adaptive_rank::CompressionEngine;
use crate::atqs::config::ATQSConfig;
use crate::has_moe_ffn::HasMoeFFNConfig;

/// Main CAFFEINE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaffeineConfig {
    /// Multi-modal encoder configuration
    pub encoders_config: EncodersConfig,
    /// Tri-Query Former configuration
    pub qformer_config: QFormerConfig,
    /// Unified tokenizer configuration
    pub tokenizer_config: TokenizerConfig,
    /// Action head configuration
    pub action_config: ActionConfig,
    /// ATQS compression configuration (optional)
    pub atqs_config: Option<ATQSConfig>,
    /// HAS-MoE-FFN routing configuration (optional)
    pub has_moe_config: Option<HasMoeFFNConfig>,
    /// Enable ATQS compression
    pub enable_atqs_compression: bool,
    /// Enable HAS-MoE-FFN routing
    pub enable_has_moe_routing: bool,
    /// Model dimension
    pub model_dim: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Dropout rate
    pub dropout_rate: f32,
}

impl Default for CaffeineConfig {
    fn default() -> Self {
        Self {
            encoders_config: EncodersConfig::default(),
            qformer_config: QFormerConfig::default(),
            tokenizer_config: TokenizerConfig::default(),
            action_config: ActionConfig::default(),
            atqs_config: None,
            has_moe_config: None,
            enable_atqs_compression: false,
            enable_has_moe_routing: false,
            model_dim: 768,
            max_sequence_length: 2048,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            hidden_dim: 3072,
            dropout_rate: 0.1,
        }
    }
}

impl CaffeineConfig {
    /// Create small model configuration
    pub fn small_model() -> Self {
        let mut config = Self::default();
        config.model_dim = 512;
        config.hidden_dim = 2048;
        config.num_attention_heads = 8;
        config.num_hidden_layers = 6;
        config
    }
    
    /// Create medium model configuration
    pub fn medium_model() -> Self {
        let mut config = Self::default();
        config.model_dim = 768;
        config.hidden_dim = 3072;
        config.num_attention_heads = 12;
        config.num_hidden_layers = 12;
        config
    }
    
    /// Create large model configuration
    pub fn large_model() -> Self {
        let mut config = Self::default();
        config.model_dim = 1024;
        config.hidden_dim = 4096;
        config.num_attention_heads = 16;
        config.num_hidden_layers = 24;
        config
    }
    
    /// Enable ATQS compression with custom config
    pub fn with_atqs_compression(mut self, atqs_config: ATQSConfig) -> Self {
        self.atqs_config = Some(atqs_config);
        self.enable_atqs_compression = true;
        self
    }
    
    /// Enable HAS-MoE-FFN routing with custom config
    pub fn with_has_moe_routing(mut self, has_moe_config: HasMoeFFNConfig) -> Self {
        self.has_moe_config = Some(has_moe_config);
        self.enable_has_moe_routing = true;
        self
    }
}

/// Multi-modal encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodersConfig {
    /// Image encoder configuration
    pub image_encoder: ImageEncoderConfig,
    /// Audio encoder configuration
    pub audio_encoder: AudioEncoderConfig,
    /// Video encoder configuration
    pub video_encoder: VideoEncoderConfig,
    /// Text encoder configuration
    pub text_encoder: TextEncoderConfig,
    /// Enable regional contrastive alignment
    pub enable_regional_alignment: bool,
    /// Number of regional patches
    pub num_regional_patches: usize,
}

impl Default for EncodersConfig {
    fn default() -> Self {
        Self {
            image_encoder: ImageEncoderConfig::default(),
            audio_encoder: AudioEncoderConfig::default(),
            video_encoder: VideoEncoderConfig::default(),
            text_encoder: TextEncoderConfig::default(),
            enable_regional_alignment: true,
            num_regional_patches: 49, // 7x7 grid
        }
    }
}

/// Image encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEncoderConfig {
    /// Vision model type
    pub model_type: VisionModelType,
    /// Pretrained model name
    pub pretrained_model: String,
    /// Output dimension
    pub output_dim: usize,
    /// Patch size
    pub patch_size: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl Default for ImageEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: VisionModelType::ViT,
            pretrained_model: "openai/clip-vit-large-patch14".to_string(),
            output_dim: 768,
            patch_size: 14,
            num_layers: 12,
        }
    }
}

/// Audio encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEncoderConfig {
    /// Audio model type
    pub model_type: AudioModelType,
    /// Sample rate
    pub sample_rate: usize,
    /// Window size
    pub window_size: usize,
    /// Hop length
    pub hop_length: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: AudioModelType::Whisper,
            sample_rate: 16000,
            window_size: 400,
            hop_length: 160,
            output_dim: 768,
        }
    }
}

/// Video encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEncoderConfig {
    /// Video model type
    pub model_type: VideoModelType,
    /// Frame sampling rate
    pub frame_rate: usize,
    /// Number of frames to sample
    pub num_frames: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl Default for VideoEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: VideoModelType::ViT3D,
            frame_rate: 1,
            num_frames: 8,
            output_dim: 768,
        }
    }
}

/// Text encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoderConfig {
    /// Text model type
    pub model_type: TextModelType,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: TextModelType::BERT,
            vocab_size: 30522,
            max_length: 512,
            output_dim: 768,
        }
    }
}

/// Tri-Query Former configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QFormerConfig {
    /// Number of semantic query tokens
    pub num_semantic_queries: usize,
    /// Number of spatial query tokens
    pub num_spatial_queries: usize,
    /// Number of temporal query tokens
    pub num_temporal_queries: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout_rate: f32,
}

impl Default for QFormerConfig {
    fn default() -> Self {
        Self {
            num_semantic_queries: 32,
            num_spatial_queries: 16,
            num_temporal_queries: 16,
            hidden_dim: 768,
            num_attention_heads: 12,
            num_layers: 6,
            dropout_rate: 0.1,
        }
    }
}

/// Unified tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Token dimension
    pub token_dim: usize,
    /// Codebook size for VQ-VAE
    pub codebook_size: usize,
    /// Number of codebooks
    pub num_codebooks: usize,
    /// Commitment loss weight
    pub commitment_weight: f32,
    /// Maximum sequence length
    pub max_sequence_length: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            token_dim: 768,
            codebook_size: 1024,
            num_codebooks: 8,
            commitment_weight: 0.25,
            max_sequence_length: 2048,
        }
    }
}

/// Action head configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConfig {
    /// Enable semantic output
    pub enable_semantic_output: bool,
    /// Enable spatial grounding
    pub enable_spatial_grounding: bool,
    /// Enable action planning
    pub enable_action_planning: bool,
    /// Number of action steps
    pub max_action_steps: usize,
    /// Bounding box format
    pub bbox_format: BBoxFormat,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            enable_semantic_output: true,
            enable_spatial_grounding: true,
            enable_action_planning: true,
            max_action_steps: 10,
            bbox_format: BBoxFormat::XYWH,
        }
    }
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

/// Bounding box formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BBoxFormat {
    XYWH,
    XYXY,
    CXCYWH,
}
