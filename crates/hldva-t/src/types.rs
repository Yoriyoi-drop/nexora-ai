//! Tipe data dan struktur untuk HLDVA-T
//!
//! Mendefinisikan tipe-tipe data yang digunakan dalam pipeline HLDVA-T

use nexora_atqs::Tensor;
use std::collections::HashMap;

/// Representasi timestep dalam DDPM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestep(pub usize);

/// Representasi resolusi
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

impl Resolution {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub fn area(&self) -> usize {
        self.width * self.height
    }
}

/// Representasi latent space
#[derive(Debug, Clone)]
pub struct LatentSpace {
    pub data: Tensor,
    pub resolution: Resolution,
    pub channels: usize,
}

impl LatentSpace {
    pub fn new(data: Tensor, resolution: Resolution, channels: usize) -> Self {
        Self {
            data,
            resolution,
            channels,
        }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.resolution.height, self.resolution.width, self.channels)
    }
}

/// CLIP embedding untuk conditioning
#[derive(Debug, Clone)]
pub struct ClipEmbedding {
    pub text_features: Tensor,
    pub image_features: Option<Tensor>,
    pub attention_mask: Option<Tensor>,
}

impl ClipEmbedding {
    pub fn new(text_features: Tensor) -> Self {
        Self {
            text_features,
            image_features: None,
            attention_mask: None,
        }
    }

    pub fn with_image_features(mut self, image_features: Tensor) -> Self {
        self.image_features = Some(image_features);
        self
    }

    pub fn with_attention_mask(mut self, attention_mask: Tensor) -> Self {
        self.attention_mask = Some(attention_mask);
        self
    }
}

/// Noise prediction dari DiT
#[derive(Debug, Clone)]
pub struct NoisePrediction {
    pub predicted_noise: Tensor,
    pub timestep: Timestep,
    pub confidence: Option<f32>,
}

impl NoisePrediction {
    pub fn new(predicted_noise: Tensor, timestep: Timestep) -> Self {
        Self {
            predicted_noise,
            timestep,
            confidence: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// Hasil dari cascaded upsampling
#[derive(Debug, Clone)]
pub struct UpsamplingResult {
    pub latent: LatentSpace,
    pub stage: usize,
    pub quality_metrics: HashMap<String, f32>,
}

impl UpsamplingResult {
    pub fn new(latent: LatentSpace, stage: usize) -> Self {
        Self {
            latent,
            stage,
            quality_metrics: HashMap::new(),
        }
    }
}

/// Input untuk HLDVA-T pipeline
#[derive(Debug, Clone)]
pub struct HLDVAInput {
    /// Text prompt
    pub prompt: String,

    /// Optional negative prompt
    pub negative_prompt: Option<String>,

    /// Target resolution
    pub target_resolution: Resolution,

    /// Seed untuk reproducibility
    pub seed: Option<u64>,

    /// Guidance scale untuk classifier-free guidance
    pub guidance_scale: f32,

    /// Jumlah inference steps
    pub num_inference_steps: usize,
}

impl Default for HLDVAInput {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            target_resolution: Resolution::new(1024, 1024),
            seed: None,
            guidance_scale: 7.5,
            num_inference_steps: 50,
        }
    }
}

/// Output dari HLDVA-T pipeline
#[derive(Debug, Clone)]
pub struct HLDVAOutput {
    /// Generated image
    pub image: Tensor,

    /// Final latent representation
    pub final_latent: LatentSpace,

    /// Intermediate results dari setiap stage
    pub intermediate_latents: Vec<LatentSpace>,

    /// Generation metrics
    pub metrics: GenerationMetrics,

    /// Waktu eksekusi
    pub execution_time_ms: u64,
}

/// Metrik untuk generation quality
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    /// FID score (jika ada ground truth)
    pub fid: Option<f32>,

    /// CLIP score
    pub clip_score: Option<f32>,

    /// Inception Score
    pub inception_score: Option<f32>,

    /// Precision dan recall
    pub precision: Option<f32>,
    pub recall: Option<f32>,

    /// Divergence metrics
    pub kl_divergence: Option<f32>,

    /// Perceptual quality
    pub lpips: Option<f32>,
}

/// Training batch data
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Images
    pub images: Tensor,

    /// Text prompts
    pub prompts: Vec<String>,

    /// Timesteps untuk batch ini
    pub timesteps: Vec<Timestep>,

    /// Noise yang ditambahkan
    pub noise: Tensor,

    /// Latent representations
    pub latents: Tensor,
}

impl TrainingBatch {
    pub fn new(
        images: Tensor,
        prompts: Vec<String>,
        timesteps: Vec<Timestep>,
        noise: Tensor,
        latents: Tensor,
    ) -> Self {
        Self {
            images,
            prompts,
            timesteps,
            noise,
            latents,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.prompts.len()
    }
}

/// Training state untuk curriculum learning
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current stage (0-3)
    pub current_stage: usize,

    /// Current epoch
    pub current_epoch: usize,

    /// Current step
    pub current_step: usize,

    /// Total steps completed
    pub total_steps: usize,

    /// Learning rate saat ini
    pub current_lr: f32,

    /// Loss history
    pub loss_history: Vec<f32>,

    /// Metrics history
    pub metrics_history: HashMap<String, Vec<f32>>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            current_stage: 0,
            current_epoch: 0,
            current_step: 0,
            total_steps: 0,
            current_lr: 1e-4,
            loss_history: Vec::new(),
            metrics_history: HashMap::new(),
        }
    }
}

/// Error types untuk HLDVA-T
#[derive(Debug, thiserror::Error)]
pub enum HLDVAError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Tensor error: {0}")]
    Tensor(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Evaluation error: {0}")]
    Evaluation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<nexora_atqs::TensorError> for HLDVAError {
    fn from(err: nexora_atqs::TensorError) -> Self {
        HLDVAError::Tensor(err.to_string())
    }
}

/// Result type untuk HLDVA-T operations
pub type HLDVAResult<T> = Result<T, HLDVAError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep() {
        let t = Timestep(42);
        assert_eq!(t.0, 42);
    }

    #[test]
    fn test_resolution() {
        let r = Resolution::new(1024, 768);
        assert_eq!(r.width, 1024);
        assert_eq!(r.height, 768);
        assert_eq!(r.area(), 1024 * 768);
    }

    #[test]
    fn test_latent_space() {
        let data = Tensor::new(vec![0.5; 64], vec![8, 8, 1]);
        let res = Resolution::new(64, 64);
        let ls = LatentSpace::new(data, res, 4);
        assert_eq!(ls.channels, 4);
        assert_eq!(ls.shape(), (64, 64, 4));
    }

    #[test]
    fn test_clip_embedding() {
        let data = Tensor::new(vec![0.1; 768], vec![768]);
        let emb = ClipEmbedding::new(data);
        assert!(emb.image_features.is_none());
        let img = Tensor::new(vec![0.2; 768], vec![768]);
        let emb = emb.with_image_features(img);
        assert!(emb.image_features.is_some());
    }

    #[test]
    fn test_noise_prediction() {
        let data = Tensor::new(vec![0.0; 100], vec![10, 10]);
        let np = NoisePrediction::new(data, Timestep(5)).with_confidence(0.95);
        assert_eq!(np.timestep.0, 5);
        assert_eq!(np.confidence, Some(0.95));
    }

    #[test]
    fn test_upsampling_result() {
        let data = Tensor::new(vec![0.0; 64], vec![8, 8, 1]);
        let ls = LatentSpace::new(data, Resolution::new(64, 64), 4);
        let ur = UpsamplingResult::new(ls, 1);
        assert_eq!(ur.stage, 1);
    }

    #[test]
    fn test_hldva_input_default() {
        let input = HLDVAInput::default();
        assert_eq!(input.target_resolution.width, 1024);
        assert_eq!(input.guidance_scale, 7.5);
    }

    #[test]
    fn test_generation_metrics_default() {
        let m = GenerationMetrics::default();
        assert!(m.fid.is_none());
    }

    #[test]
    fn test_training_batch() {
        let images = Tensor::new(vec![0.0; 256], vec![4, 64]);
        let prompts = vec!["cat".to_string(), "dog".to_string()];
        let timesteps = vec![Timestep(10), Timestep(50)];
        let noise = Tensor::new(vec![0.1; 256], vec![4, 64]);
        let latents = Tensor::new(vec![0.2; 256], vec![4, 64]);
        let batch = TrainingBatch::new(images, prompts, timesteps, noise, latents);
        assert_eq!(batch.batch_size(), 2);
    }

    #[test]
    fn test_training_state_default() {
        let s = TrainingState::default();
        assert_eq!(s.current_stage, 0);
        assert_eq!(s.current_lr, 1e-4);
    }

    #[test]
    fn test_hldva_error() {
        let err = HLDVAError::Config("missing field".to_string());
        assert_eq!(format!("{}", err), "Configuration error: missing field");
    }

    #[test]
    fn test_hldva_result() {
        let r: HLDVAResult<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }
}

/// Trait untuk komponen HLDVA-T yang bisa di-train
pub trait Trainable {
    type TrainingConfig;

    fn train(&mut self, config: Self::TrainingConfig) -> HLDVAResult<()>;

    fn evaluate(&self) -> HLDVAResult<GenerationMetrics>;

    fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> HLDVAResult<()>;

    fn load_checkpoint<P: AsRef<std::path::Path>>(&mut self, path: P) -> HLDVAResult<()>;
}

/// Trait untuk komponen HLDVA-T yang bisa di-inference
pub trait Inference {
    type Input;
    type Output;

    fn infer(&self, input: Self::Input) -> HLDVAResult<Self::Output>;

    fn batch_infer(&self, inputs: Vec<Self::Input>) -> HLDVAResult<Vec<Self::Output>>;
}


