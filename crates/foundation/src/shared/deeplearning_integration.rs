//! Deep Learning Integration Layer untuk NXR Models
//!
//! Layer ini menyediakan integrasi antara deeplearning crate dan semua NXR models
//! Mendukung STAR-X dan ECHO-Net Ω architectures

use super::foundation_components::FoundationComponents;

use nexora_deeplearning::{
    star_x::{
        StarXConfig, StarXState, StarXMetics,
        TemporalGatingHierarchy,
        SparseCausalAttention,
        HarmonicTemporalEncoding,
        SelectiveStateUpdate,
        EpisodicMemoryRetention,
    },
    echo_net::{EchoNetConfig, EchoNetState, EchoNetMetrics},
    traits::{Forward, Backward, Stateful, Trainable},
    DLResult, DeepLearningError,
};
use nexora_deeplearning::star_x::core::{
    HierarchicalGating, SparseAttention, SelectiveUpdate, EpisodicMemory,
};
use ndarray::ArrayD;
use parking_lot::RwLock as PRwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tipe arsitektur deep learning yang digunakan
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DLArchitecture {
    /// STAR-X - Selective Temporal Adaptive Resonance Network
    StarX,
    /// ECHO-Net Ω - Entropic Contextual Holographic Oscillation Network
    EchoNet,
    /// Hybrid - Kombinasi STAR-X dan ECHO-Net
    Hybrid,
}

/// Konfigurasi deep learning untuk NXR models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLearningConfig {
    /// Tipe arsitektur yang digunakan
    pub architecture: DLArchitecture,
    /// Konfigurasi STAR-X (jika digunakan)
    pub star_x_config: Option<StarXConfig>,
    /// Konfigurasi ECHO-Net (jika digunakan)
    pub echo_net_config: Option<EchoNetConfig>,
    /// Enable training mode
    pub training_enabled: bool,
    /// Enable inference optimization
    pub inference_optimized: bool,
    /// Batch size untuk processing
    pub batch_size: usize,
    /// Enable gradient checkpointing untuk memory efficiency
    pub gradient_checkpointing: bool,
}

impl Default for DeepLearningConfig {
    fn default() -> Self {
        Self {
            architecture: DLArchitecture::StarX,
            star_x_config: Some(StarXConfig::default()),
            echo_net_config: None,
            training_enabled: false,
            inference_optimized: true,
            batch_size: 32,
            gradient_checkpointing: true,
        }
    }
}

impl DeepLearningConfig {
    pub fn star_x() -> Self {
        Self {
            architecture: DLArchitecture::StarX,
            star_x_config: Some(StarXConfig::default()),
            echo_net_config: None,
            ..Default::default()
        }
    }

    pub fn echo_net() -> Self {
        Self {
            architecture: DLArchitecture::EchoNet,
            star_x_config: None,
            echo_net_config: Some(EchoNetConfig::default()),
            ..Default::default()
        }
    }

    pub fn hybrid() -> Self {
        Self {
            architecture: DLArchitecture::Hybrid,
            star_x_config: Some(StarXConfig::default()),
            echo_net_config: Some(EchoNetConfig::default()),
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self.architecture {
            DLArchitecture::StarX => {
                if self.star_x_config.is_none() {
                    return Err("STAR-X config required for StarX architecture".to_string());
                }
            }
            DLArchitecture::EchoNet => {
                if self.echo_net_config.is_none() {
                    return Err("ECHO-Net config required for EchoNet architecture".to_string());
                }
            }
            DLArchitecture::Hybrid => {
                if self.star_x_config.is_none() || self.echo_net_config.is_none() {
                    return Err("Both STAR-X and ECHO-Net config required for Hybrid architecture".to_string());
                }
            }
        }
        Ok(())
    }
}

/// State deep learning untuk NXR models
#[derive(Debug, Clone)]
pub enum DeepLearningState {
    StarX(StarXState),
    EchoNet(EchoNetState),
    Hybrid {
        star_x: StarXState,
        echo_net: EchoNetState,
    },
}

impl DeepLearningState {
    pub fn reset(&mut self) {
        match self {
            DeepLearningState::StarX(state) => state.reset(),
            DeepLearningState::EchoNet(state) => state.reset(),
            DeepLearningState::Hybrid { star_x, echo_net } => {
                star_x.reset();
                echo_net.reset();
            }
        }
    }
}

/// Metrics deep learning untuk NXR models
#[derive(Debug, Clone)]
pub enum DeepLearningMetrics {
    StarX(StarXMetics),
    EchoNet(EchoNetMetrics),
    Hybrid {
        star_x: StarXMetics,
        echo_net: EchoNetMetrics,
    },
}

/// STAR-X component pipeline
struct StarXPipeline {
    tgh: TemporalGatingHierarchy,
    sca: SparseCausalAttention,
    hte: HarmonicTemporalEncoding,
    ssu: SelectiveStateUpdate,
    emr: PRwLock<EpisodicMemoryRetention>,
}

// EchoNetModel is not thread-safe (uses Rc internally),
// so EchoNet uses state-based forward/backward.

/// Deep Learning Engine - wrapper untuk semua arsitektur
pub struct DeepLearningEngine {
    config: DeepLearningConfig,
    state: Arc<RwLock<DeepLearningState>>,
    metrics: Arc<RwLock<DeepLearningMetrics>>,
    starx_pipeline: Option<StarXPipeline>,
}

impl DeepLearningEngine {
    pub fn new(config: DeepLearningConfig) -> DLResult<Self> {
        config.validate()
            .map_err(|e| DeepLearningError::Configuration { reason: e })?;

        let state = match config.architecture {
            DLArchitecture::StarX => {
                let star_x_config = config.star_x_config.as_ref()
                    .ok_or_else(|| DeepLearningError::Configuration { reason: "star_x_config required for StarX architecture".into() })?;
                DeepLearningState::StarX(StarXState::new(star_x_config)?)
            }
            DLArchitecture::EchoNet => {
                let echo_net_config = config.echo_net_config.as_ref()
                    .ok_or_else(|| DeepLearningError::Configuration { reason: "echo_net_config required for EchoNet architecture".into() })?;
                DeepLearningState::EchoNet(EchoNetState::new(echo_net_config)?)
            }
            DLArchitecture::Hybrid => {
                let star_x_config = config.star_x_config.as_ref()
                    .ok_or_else(|| DeepLearningError::Configuration { reason: "star_x_config required for Hybrid architecture".into() })?;
                let echo_net_config = config.echo_net_config.as_ref()
                    .ok_or_else(|| DeepLearningError::Configuration { reason: "echo_net_config required for Hybrid architecture".into() })?;
                DeepLearningState::Hybrid {
                    star_x: StarXState::new(star_x_config)?,
                    echo_net: EchoNetState::new(echo_net_config)?,
                }
            }
        };

        let metrics = match config.architecture {
            DLArchitecture::StarX => DeepLearningMetrics::StarX(StarXMetics::default()),
            DLArchitecture::EchoNet => DeepLearningMetrics::EchoNet(EchoNetMetrics::default()),
            DLArchitecture::Hybrid => DeepLearningMetrics::Hybrid {
                star_x: StarXMetics::default(),
                echo_net: EchoNetMetrics::default(),
            }
        };

        let starx_pipeline = match config.architecture {
            DLArchitecture::StarX | DLArchitecture::Hybrid => {
                let sc = config.star_x_config.as_ref().unwrap();
                Some(StarXPipeline {
                    tgh: TemporalGatingHierarchy::new(
                        sc.input_size, sc.hidden_size,
                        sc.micro_gate_size, sc.meso_gate_size, sc.macro_gate_size,
                        sc.chunk_size,
                    )?,
                    sca: SparseCausalAttention::new(
                        sc.hidden_size, sc.attention_heads,
                        sc.max_sparse_connections, sc.entropy_regularization,
                    )?,
                    hte: HarmonicTemporalEncoding::new(
                        sc.temporal_embedding_dim, sc.harmonic_frequencies, 1024,
                    )?,
                    ssu: SelectiveStateUpdate::new(
                        sc.hidden_size, sc.hidden_size,
                        sc.update_threshold, sc.relevance_alpha,
                    )?,
                    emr: PRwLock::new(EpisodicMemoryRetention::new(
                        sc.memory_size, sc.hidden_size, sc.memory_write_threshold,
                    )?),
                })
            }
            DLArchitecture::EchoNet => None,
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
            metrics: Arc::new(RwLock::new(metrics)),
            starx_pipeline,
        })
    }

    pub fn config(&self) -> &DeepLearningConfig {
        &self.config
    }

    pub async fn state(&self) -> DeepLearningState {
        self.state.read().await.clone()
    }

    pub async fn update_state(&self, new_state: DeepLearningState) {
        *self.state.write().await = new_state;
    }

    pub async fn metrics(&self) -> DeepLearningMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn update_metrics(&self, updater: impl FnOnce(&mut DeepLearningMetrics)) {
        updater(&mut *self.metrics.write().await);
    }

    pub async fn reset(&self) {
        self.state.write().await.reset();
    }

    pub async fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let state = self.state.read().await;

        match &*state {
            DeepLearningState::StarX(_) => {
                let pipeline = self.starx_pipeline.as_ref()
                    .ok_or_else(|| DeepLearningError::Configuration {
                        reason: "StarX pipeline not initialized".into()
                    })?;

                let starx_state = match &*state {
                    DeepLearningState::StarX(s) => s.clone(),
                    _ => return Err(DeepLearningError::Configuration {
                        reason: "Expected StarX state".into()
                    }),
                };

                let hte_output = pipeline.hte.forward(input)?;

                let chunk_context = ArrayD::zeros(vec![starx_state.hidden_state.len()]);

                let (micro, meso, macro_out_v) = pipeline.tgh.process_hierarchical(
                    &hte_output,
                    &starx_state.hidden_state,
                    &chunk_context,
                    &starx_state.episodic_memory,
                )?;

                let fused = pipeline.tgh.fuse_hierarchical(
                    &micro, &meso, &macro_out_v,
                    (1.0/3.0, 1.0/3.0, 1.0/3.0),
                )?;

                let attended = pipeline.sca.forward(&fused)?;

                let relevance = pipeline.ssu.compute_relevance(&fused, &attended)?;
                let output = pipeline.ssu.selective_update(
                    &fused, &attended, &relevance,
                    starx_state.resonance_factor,
                )?;

                drop(state);
                let mut state_w = self.state.write().await;
                if let DeepLearningState::StarX(ref mut s) = *state_w {
                    s.hidden_state = output.clone();
                    s.temporal_position += 1;
                }

                Ok(output)
            }
            DeepLearningState::EchoNet(echo_net_state) => {
                let flat_input = input.clone().into_shape(input.len())
                    .map_err(DeepLearningError::from)?;
                let amp_dim = echo_net_state.amplitude_spectrum.len().max(1);
                let phase_dim = echo_net_state.semantic_phase.len().max(1);

                let result_data: Vec<f32> = flat_input.iter().enumerate().map(|(i, x)| {
                    let amp = echo_net_state.amplitude_spectrum.as_slice()
                        .and_then(|s| s.get(i % amp_dim))
                        .copied()
                        .unwrap_or(0.0);
                    let phase = echo_net_state.semantic_phase.as_slice()
                        .and_then(|s| s.get(i % phase_dim))
                        .copied()
                        .unwrap_or(0.0);
                    (x * amp + phase).tanh()
                }).collect();

                ArrayD::from_shape_vec(input.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::Hybrid { star_x, echo_net } => {
                let starx_output = {
                    let pipeline = self.starx_pipeline.as_ref()
                        .ok_or_else(|| DeepLearningError::Configuration {
                            reason: "StarX pipeline not initialized for Hybrid".into()
                        })?;

                    let hte_output = pipeline.hte.forward(input)?;

                    let (micro, meso, macro_out_v) = pipeline.tgh.process_hierarchical(
                        &hte_output,
                        &star_x.hidden_state,
                        &ArrayD::zeros(vec![star_x.hidden_state.len()]),
                        &star_x.episodic_memory,
                    )?;

                    let fused = pipeline.tgh.fuse_hierarchical(
                        &micro, &meso, &macro_out_v,
                        (1.0/3.0, 1.0/3.0, 1.0/3.0),
                    )?;

                    let attended = pipeline.sca.forward(&fused)?;
                    let relevance = pipeline.ssu.compute_relevance(&fused, &attended)?;
                    pipeline.ssu.selective_update(
                        &fused, &attended, &relevance,
                        star_x.resonance_factor,
                    )?
                };

                let echo_output = {
                    let flat_input = input.clone().into_shape(input.len())
                        .map_err(DeepLearningError::from)?;
                    let amp_dim = echo_net.amplitude_spectrum.len().max(1);
                    let phase_dim = echo_net.semantic_phase.len().max(1);

                    let result_data: Vec<f32> = flat_input.iter().enumerate().map(|(i, x)| {
                        let amp = echo_net.amplitude_spectrum.as_slice()
                            .and_then(|s| s.get(i % amp_dim))
                            .copied()
                            .unwrap_or(0.0);
                        let phase = echo_net.semantic_phase.as_slice()
                            .and_then(|s| s.get(i % phase_dim))
                            .copied()
                            .unwrap_or(0.0);
                        (x * amp + phase).tanh()
                    }).collect();

                    ArrayD::from_shape_vec(input.shape().to_vec(), result_data)
                        .map_err(DeepLearningError::from)?
                };

                let out = &starx_output + &echo_output;

                drop(state);
                let mut state_w = self.state.write().await;
                if let DeepLearningState::Hybrid { ref mut star_x, .. } = *state_w {
                    star_x.hidden_state = starx_output;
                    star_x.temporal_position += 1;
                }

                Ok(out)
            }
        }
    }

    pub async fn backward(&self, gradient: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        if !self.config.training_enabled {
            return Err(DeepLearningError::Configuration {
                reason: "Training not enabled".to_string(),
            });
        }

        let state = self.state.read().await;

        match &*state {
            DeepLearningState::StarX(_) => {
                let pipeline = self.starx_pipeline.as_ref()
                    .ok_or_else(|| DeepLearningError::Configuration {
                        reason: "StarX pipeline not initialized".into()
                    })?;

                let starx_state = match &*state {
                    DeepLearningState::StarX(s) => s.clone(),
                    _ => return Err(DeepLearningError::Configuration {
                        reason: "Expected StarX state".into()
                    }),
                };

                let hidden_size = starx_state.hidden_state.len();
                let flat_grad = gradient.clone().into_shape(gradient.len())
                    .map_err(DeepLearningError::from)?;

                let result_data: Vec<f32> = flat_grad.iter().enumerate().map(|(i, g)| {
                    let w = starx_state.hidden_state[i % hidden_size];
                    let relevance = if starx_state.memory_priorities.is_empty() {
                        1.0
                    } else {
                        starx_state.memory_priorities[i % starx_state.memory_priorities.len()]
                    };
                    g * (w.abs() + 1.0) * (relevance.abs() + 0.1).recip()
                }).collect();

                ArrayD::from_shape_vec(gradient.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::EchoNet(_) => {
                let flat_grad = gradient.clone().into_shape(gradient.len())
                    .map_err(DeepLearningError::from)?;

                let result_data: Vec<f32> = flat_grad.iter().map(|g| {
                    g * 0.1
                }).collect();

                ArrayD::from_shape_vec(gradient.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::Hybrid { star_x, echo_net } => {
                let flat_grad = gradient.clone().into_shape(gradient.len())
                    .map_err(DeepLearningError::from)?;
                let hidden_size = star_x.hidden_state.len();

                let result_data: Vec<f32> = flat_grad.iter().enumerate().map(|(i, g)| {
                    let w = star_x.hidden_state[i % hidden_size];
                    g * w.abs()
                }).collect();

                ArrayD::from_shape_vec(gradient.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
        }
    }

    pub async fn process_text(&self, text: &str) -> DLResult<String> {
        let input = self.text_to_tensor(text)?;
        let output = self.forward(&input).await?;
        self.tensor_to_text(&output)
    }

    fn text_to_tensor(&self, text: &str) -> DLResult<ArrayD<f32>> {
        let config = &self.config;
        let target_dim = config.star_x_config.as_ref()
            .map(|c| c.input_size)
            .or_else(|| config.echo_net_config.as_ref().map(|c| c.embedding_dim))
            .unwrap_or(512);

        let mut features = vec![0.0f32; target_dim];
        let bytes = text.as_bytes();
        let step = (bytes.len().max(1) as f32 / target_dim as f32).ceil() as usize;

        for (i, &byte) in bytes.iter().enumerate() {
            let idx = (i / step.max(1)) % target_dim;
            features[idx] += byte as f32 / 255.0;
        }

        let norm: f32 = features.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in features.iter_mut() {
                *v /= norm;
            }
        }

        ArrayD::from_shape_vec(vec![1, target_dim], features)
            .map_err(DeepLearningError::from)
    }

    fn tensor_to_text(&self, tensor: &ArrayD<f32>) -> DLResult<String> {
        let flat: Vec<f32> = tensor.iter().copied().collect();
        if flat.is_empty() {
            return Ok(String::new());
        }

        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        let variance = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
        let std = variance.sqrt();
        let entropy = -flat.iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| v * v.ln())
            .sum::<f32>() / flat.len() as f32;
        let sparsity = flat.iter().filter(|&&v| v.abs() < 0.01).count();

        Ok(format!(
            "DL Output — mean:{:.4} std:{:.4} entropy:{:.4} sparsity:{}/{} dim:{}",
            mean, std, entropy, sparsity, flat.len(), flat.len()
        ))
    }
}

/// Trait untuk models yang memiliki akses ke FoundationComponents
pub trait HasComponents {
    fn components(&self) -> &FoundationComponents;
}

/// Trait untuk models yang menggunakan deep learning
#[async_trait::async_trait]
pub trait DeepLearningModel: HasComponents {
    fn dl_engine(&self) -> &DeepLearningEngine {
        &self.components().dl_engine
    }

    async fn dl_process(&self, input: &str) -> DLResult<String> {
        self.dl_engine().process_text(input).await
    }

    async fn enable_training(&mut self) -> DLResult<()> {
        Ok(())
    }

    async fn disable_training(&mut self) -> DLResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_learning_config_validation() {
        let config = DeepLearningConfig::star_x();
        assert!(config.validate().is_ok());

        let mut invalid_config = DeepLearningConfig::default();
        invalid_config.star_x_config = None;
        assert!(invalid_config.validate().is_err());
    }

    #[tokio::test]
    async fn test_deep_learning_engine_creation() {
        let config = DeepLearningConfig::star_x();
        let engine = DeepLearningEngine::new(config).unwrap();
        assert_eq!(engine.config().architecture, DLArchitecture::StarX);
    }

    #[tokio::test]
    async fn test_text_processing() {
        let config = DeepLearningConfig::star_x();
        let engine = DeepLearningEngine::new(config).unwrap();
        let result = engine.process_text("test input").await;
        assert!(result.is_ok());
    }
}
