//! Deep Learning Integration Layer untuk NXR Models
//!
//! Layer ini menyediakan integrasi antara deeplearning crate dan semua NXR models
//! Mendukung STAR-X dan ECHO-Net Ω architectures

use nexora_deeplearning::{
    star_x::{StarXConfig, StarXState, StarXMetics},
    echo_net::{EchoNetConfig, EchoNetState, EchoNetMetrics},
    traits::{Forward, Backward, Stateful, Trainable},
    DLResult, DeepLearningError,
};
use ndarray::ArrayD;
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
    #[serde(skip)]
    pub star_x_config: Option<StarXConfig>,
    
    /// Konfigurasi ECHO-Net (jika digunakan)
    #[serde(skip)]
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
    /// Buat konfigurasi untuk STAR-X
    pub fn star_x() -> Self {
        Self {
            architecture: DLArchitecture::StarX,
            star_x_config: Some(StarXConfig::default()),
            echo_net_config: None,
            ..Default::default()
        }
    }
    
    /// Buat konfigurasi untuk ECHO-Net
    pub fn echo_net() -> Self {
        Self {
            architecture: DLArchitecture::EchoNet,
            star_x_config: None,
            echo_net_config: Some(EchoNetConfig::default()),
            ..Default::default()
        }
    }
    
    /// Buat konfigurasi hybrid
    pub fn hybrid() -> Self {
        Self {
            architecture: DLArchitecture::Hybrid,
            star_x_config: Some(StarXConfig::default()),
            echo_net_config: Some(EchoNetConfig::default()),
            ..Default::default()
        }
    }
    
    /// Validasi konfigurasi
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
    /// State STAR-X
    StarX(StarXState),
    /// State ECHO-Net
    EchoNet(EchoNetState),
    /// State Hybrid (keduanya)
    Hybrid {
        star_x: StarXState,
        echo_net: EchoNetState,
    },
}

impl DeepLearningState {
    /// Reset state
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
    /// Metrics STAR-X
    StarX(StarXMetics),
    /// Metrics ECHO-Net
    EchoNet(EchoNetMetrics),
    /// Metrics Hybrid (keduanya)
    Hybrid {
        star_x: StarXMetics,
        echo_net: EchoNetMetrics,
    },
}

/// Deep Learning Engine - wrapper untuk semua arsitektur
pub struct DeepLearningEngine {
    /// Konfigurasi
    config: DeepLearningConfig,
    /// State
    state: Arc<RwLock<DeepLearningState>>,
    /// Metrics
    metrics: Arc<RwLock<DeepLearningMetrics>>,
}

impl DeepLearningEngine {
    /// Buat engine baru
    pub fn new(config: DeepLearningConfig) -> DLResult<Self> {
        config.validate()
            .map_err(|e| DeepLearningError::Configuration { reason: e })?;
        
        let state = match config.architecture {
            DLArchitecture::StarX => {
                let star_x_config = config.star_x_config.as_ref().unwrap();
                DeepLearningState::StarX(StarXState::new(star_x_config)?)
            }
            DLArchitecture::EchoNet => {
                let echo_net_config = config.echo_net_config.as_ref().unwrap();
                DeepLearningState::EchoNet(EchoNetState::new(echo_net_config)?)
            }
            DLArchitecture::Hybrid => {
                let star_x_config = config.star_x_config.as_ref().unwrap();
                let echo_net_config = config.echo_net_config.as_ref().unwrap();
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
        
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
            metrics: Arc::new(RwLock::new(metrics)),
        })
    }
    
    /// Get config
    pub fn config(&self) -> &DeepLearningConfig {
        &self.config
    }
    
    /// Get state
    pub async fn state(&self) -> DeepLearningState {
        let state = self.state.read().await;
        state.clone()
    }
    
    /// Update state
    pub async fn update_state(&self, new_state: DeepLearningState) {
        let mut state = self.state.write().await;
        *state = new_state;
    }
    
    /// Get metrics
    pub async fn metrics(&self) -> DeepLearningMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
    
    /// Update metrics
    pub async fn update_metrics(&self, updater: impl FnOnce(&mut DeepLearningMetrics)) {
        let mut metrics = self.metrics.write().await;
        updater(&mut metrics);
    }
    
    /// Reset state
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
        state.reset();
    }
    
    /// Forward pass
    pub async fn forward(&self, input: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        let state = self.state.read().await;

        match &*state {
            DeepLearningState::StarX(star_x_state) => {
                let flat_input = input.clone().into_shape(input.len())?;
                let model_dim = star_x_state.hidden_state.len();
                let bias = star_x_state.episodic_memory.iter().sum::<f32>() / star_x_state.episodic_memory.len() as f32;

                let result_data: Vec<f32> = flat_input.iter().enumerate().map(|(i, x)| {
                    let w = star_x_state.hidden_state[i % model_dim];
                    (x * w + bias).tanh()
                }).collect();

                ArrayD::from_shape_vec(input.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::EchoNet(echo_net_state) => {
                let flat_input = input.clone().into_shape(input.len())?;
                let amp_dim = echo_net_state.amplitude_spectrum.len();
                let phase_dim = echo_net_state.semantic_phase.len();

                let result_data: Vec<f32> = flat_input.iter().enumerate().map(|(i, x)| {
                    let amp = echo_net_state.amplitude_spectrum[i % amp_dim];
                    let phase = echo_net_state.semantic_phase[i % phase_dim];
                    (x * amp + phase).tanh()
                }).collect();

                ArrayD::from_shape_vec(input.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::Hybrid { star_x, echo_net } => {
                let flat_input = input.clone().into_shape(input.len())?;
                let model_dim = star_x.hidden_state.len();
                let amp_dim = echo_net.amplitude_spectrum.len();
                let phase_dim = echo_net.semantic_phase.len();

                let result_data: Vec<f32> = flat_input.iter().enumerate().map(|(i, x)| {
                    let w = star_x.hidden_state[i % model_dim];
                    let amp = echo_net.amplitude_spectrum[i % amp_dim];
                    let phase = echo_net.semantic_phase[i % phase_dim];
                    let star_x_contrib = x * w;
                    let echo_net_contrib = (x * amp + phase).tanh();
                    (star_x_contrib + echo_net_contrib).tanh()
                }).collect();

                ArrayD::from_shape_vec(input.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
        }
    }
    
    /// Backward pass (training)
    pub async fn backward(&self, gradient: &ArrayD<f32>) -> DLResult<ArrayD<f32>> {
        if !self.config.training_enabled {
            return Err(DeepLearningError::Configuration {
                reason: "Training not enabled".to_string(),
            });
        }

        let state = self.state.read().await;

        match &*state {
            DeepLearningState::StarX(star_x_state) => {
                let flat_grad = gradient.clone().into_shape(gradient.len())?;
                let model_dim = star_x_state.hidden_state.len();

                let result_data: Vec<f32> = flat_grad.iter().enumerate().map(|(i, g)| {
                    let w = star_x_state.hidden_state[i % model_dim];
                    g * w
                }).collect();

                ArrayD::from_shape_vec(gradient.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::EchoNet(echo_net_state) => {
                let flat_grad = gradient.clone().into_shape(gradient.len())?;
                let amp_dim = echo_net_state.amplitude_spectrum.len();

                let result_data: Vec<f32> = flat_grad.iter().enumerate().map(|(i, g)| {
                    let amp = echo_net_state.amplitude_spectrum[i % amp_dim];
                    g * amp
                }).collect();

                ArrayD::from_shape_vec(gradient.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
            DeepLearningState::Hybrid { star_x, echo_net } => {
                let flat_grad = gradient.clone().into_shape(gradient.len())?;
                let model_dim = star_x.hidden_state.len();
                let amp_dim = echo_net.amplitude_spectrum.len();

                let result_data: Vec<f32> = flat_grad.iter().enumerate().map(|(i, g)| {
                    let w = star_x.hidden_state[i % model_dim];
                    let amp = echo_net.amplitude_spectrum[i % amp_dim];
                    g * (w + amp)
                }).collect();

                ArrayD::from_shape_vec(gradient.shape(), result_data)
                    .map_err(DeepLearningError::from)
            }
        }
    }
    
    /// Process text input untuk inference
    pub async fn process_text(&self, text: &str) -> DLResult<String> {
        // Convert text ke tensor
        let input = self.text_to_tensor(text)?;
        
        // Forward pass
        let output = self.forward(&input).await?;
        
        // Convert tensor ke text
        self.tensor_to_text(&output)
    }
    
    /// Convert text ke tensor representation
    fn text_to_tensor(&self, text: &str) -> DLResult<ArrayD<f32>> {
        // Simple embedding: character frequencies
        let mut freq = [0f32; 256];
        for byte in text.bytes() {
            freq[byte as usize] += 1.0;
        }
        
        // Normalize
        let sum: f32 = freq.iter().sum();
        if sum > 0.0 {
            for val in freq.iter_mut() {
                *val /= sum;
            }
        }
        
        ArrayD::from_shape_vec(ndarray::IxDyn(&[freq.len()]), freq.to_vec())
            .map_err(DeepLearningError::from)
    }
    
    /// Convert tensor ke text representation
    fn tensor_to_text(&self, tensor: &ArrayD<f32>) -> DLResult<String> {
        // Simple conversion: tensor stats
        let mean = tensor.mean().unwrap_or(0.0);
        let std = tensor.std(0.0);
        
        Ok(format!("DL Output - Mean: {:.4}, Std: {:.4}", mean, std))
    }
}

/// Trait untuk models yang menggunakan deep learning
#[async_trait::async_trait]
pub trait DeepLearningModel {
    /// Get deep learning engine
    fn dl_engine(&self) -> &DeepLearningEngine;
    
    /// Get deep learning engine mutable
    fn dl_engine_mut(&mut self) -> &mut DeepLearningEngine;
    
    /// Process input dengan deep learning
    async fn dl_process(&self, input: &str) -> DLResult<String> {
        self.dl_engine().process_text(input).await
    }
    
    /// Enable training mode
    async fn enable_training(&mut self) -> DLResult<()> {
        let mut state = self.dl_engine_mut().state.write().await;
        // Update config untuk enable training
        Ok(())
    }
    
    /// Disable training mode (inference only)
    async fn disable_training(&mut self) -> DLResult<()> {
        let mut state = self.dl_engine_mut().state.write().await;
        // Update config untuk disable training
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
