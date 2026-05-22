pub mod apss;
pub mod derr;
#[cfg(feature = "gpu")]
pub mod gpu_ops;
pub mod irr;
pub mod isc;
pub mod mbhw;
pub mod model;
pub mod prm;
pub mod rhc;
pub mod sse;
pub mod tkrr;
pub mod training;
pub mod utils;

pub use apss::*;
pub use derr::*;
pub use irr::*;
pub use isc::*;
pub use mbhw::*;
pub use prm::*;
pub use rhc::*;
pub use sse::*;
pub use tkrr::*;
pub use training::*;
pub use utils::*;

pub type DLResult<T> = std::result::Result<T, DeepLearningError>;

#[derive(Debug, thiserror::Error)]
pub enum DeepLearningError {
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
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

use ndarray::ArrayD;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EchoNetConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_frequency_bands: usize,
    pub amplitude_dim: usize,
    pub phase_dim: usize,
    pub resonance_dim: usize,
    pub num_bands: usize,
    pub band_frequencies: Vec<f32>,
    pub kernel_size: usize,
    pub compression_levels: usize,
    pub compression_ratio: f32,
    pub memory_size: usize,
    pub decay_alpha: f32,
    pub write_threshold: f32,
    pub reasoning_steps: usize,
    pub reasoning_alpha: f32,
    pub energy_weight: f32,
    pub entropy_weight: f32,
    pub coherence_weight: f32,
    pub top_k: usize,
    pub routing_threshold: f32,
    pub phase_lr: f32,
    pub energy_clip: f32,
    pub streaming_window: usize,
    pub update_frequency: usize,
    pub update_threshold: f32,
    pub memory_write_threshold: f32,
    pub resonance_factor: f32,
    pub output_size: usize,
}

impl Default for EchoNetConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50000,
            embedding_dim: 1024,
            max_frequency_bands: 16,
            amplitude_dim: 512,
            phase_dim: 256,
            resonance_dim: 256,
            num_bands: 8,
            band_frequencies: vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            kernel_size: 64,
            compression_levels: 5,
            compression_ratio: 0.5,
            memory_size: 100000,
            decay_alpha: 0.01,
            write_threshold: 0.1,
            reasoning_steps: 3,
            reasoning_alpha: 0.1,
            energy_weight: 1.0,
            entropy_weight: 0.5,
            coherence_weight: 0.3,
            top_k: 64,
            routing_threshold: 0.01,
            phase_lr: 0.01,
            energy_clip: 10.0,
            streaming_window: 1024,
            update_frequency: 100,
            update_threshold: 0.1,
            memory_write_threshold: 0.1,
            resonance_factor: 0.1,
            output_size: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EchoNetState {
    pub amplitude_spectrum: ArrayD<f32>,
    pub semantic_phase: ArrayD<f32>,
    pub resonance_energy: ArrayD<f32>,
    pub holographic_bands: Vec<ArrayD<f32>>,
    pub compressed_memory: Vec<ArrayD<f32>>,
    pub persistent_memory: ArrayD<f32>,
    pub memory_priorities: Vec<f32>,
    pub memory_timestamps: Vec<usize>,
    pub current_query: ArrayD<f32>,
    pub reasoning_history: Vec<ArrayD<f32>>,
    pub streaming_buffer: ArrayD<f32>,
    pub temporal_position: usize,
    pub phase_gradients: ArrayD<f32>,
    pub energy_history: Vec<f32>,
}

impl EchoNetState {
    pub fn new(config: &EchoNetConfig) -> DLResult<Self> {
        Ok(Self {
            amplitude_spectrum: ArrayD::zeros(vec![config.amplitude_dim]),
            semantic_phase: ArrayD::zeros(vec![config.phase_dim]),
            resonance_energy: ArrayD::zeros(vec![config.resonance_dim]),
            holographic_bands: vec![ArrayD::zeros(vec![config.embedding_dim]); config.num_bands],
            compressed_memory: vec![
                ArrayD::zeros(vec![config.embedding_dim]);
                config.compression_levels
            ],
            persistent_memory: ArrayD::zeros(vec![config.memory_size, config.embedding_dim]),
            memory_priorities: vec![0.0; config.memory_size],
            memory_timestamps: vec![0; config.memory_size],
            current_query: ArrayD::zeros(vec![config.embedding_dim]),
            reasoning_history: vec![],
            streaming_buffer: ArrayD::zeros(vec![config.streaming_window, config.embedding_dim]),
            temporal_position: 0,
            phase_gradients: ArrayD::zeros(vec![config.phase_dim]),
            energy_history: vec![],
        })
    }

    pub fn reset(&mut self) {
        self.amplitude_spectrum.fill(0.0);
        self.semantic_phase.fill(0.0);
        self.resonance_energy.fill(0.0);
        for band in &mut self.holographic_bands {
            band.fill(0.0);
        }
        for memory in &mut self.compressed_memory {
            memory.fill(0.0);
        }
        self.persistent_memory.fill(0.0);
        self.memory_priorities.fill(0.0);
        self.memory_timestamps.fill(0);
        self.current_query.fill(0.0);
        self.reasoning_history.clear();
        self.streaming_buffer.fill(0.0);
        self.temporal_position = 0;
        self.phase_gradients.fill(0.0);
        self.energy_history.clear();
    }
}

#[derive(Debug, Clone)]
pub struct EchoNetMetrics {
    pub memory_utilization: f32,
    pub resonance_efficiency: f32,
    pub compression_ratio: f32,
    pub retrieval_accuracy: f32,
    pub reasoning_depth: f32,
    pub streaming_latency: f32,
    pub semantic_coherence: f32,
    pub energy_stability: f32,
    pub throughput_tokens_per_second: f32,
    pub sparsity_ratio: f32,
    pub update_frequency: f32,
}

impl Default for EchoNetMetrics {
    fn default() -> Self {
        Self {
            memory_utilization: 0.0,
            resonance_efficiency: 0.0,
            compression_ratio: 0.0,
            retrieval_accuracy: 0.0,
            reasoning_depth: 0.0,
            streaming_latency: 0.0,
            semantic_coherence: 0.0,
            energy_stability: 0.0,
            throughput_tokens_per_second: 0.0,
            sparsity_ratio: 0.0,
            update_frequency: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComplexTensor {
    pub real: ArrayD<f32>,
    pub imag: ArrayD<f32>,
}

impl ComplexTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            real: ArrayD::zeros(shape.clone()),
            imag: ArrayD::zeros(shape),
        }
    }

    pub fn from_polar(amplitude: &ArrayD<f32>, phase: &ArrayD<f32>) -> DLResult<Self> {
        if amplitude.shape() != phase.shape() {
            return Err(DeepLearningError::ShapeMismatch {
                expected: amplitude.shape().to_vec(),
                actual: phase.shape().to_vec(),
            });
        }
        let mut real = ArrayD::zeros(amplitude.shape());
        let mut imag = ArrayD::zeros(amplitude.shape());
        for ((idx, &amp), &ph) in amplitude.indexed_iter().zip(phase.iter()) {
            real[idx.clone()] = amp * ph.cos();
            imag[idx.clone()] = amp * ph.sin();
        }
        Ok(Self { real, imag })
    }

    pub fn amplitude(&self) -> ArrayD<f32> {
        let mut result = ArrayD::zeros(self.real.shape());
        for ((idx, &real_val), &imag_val) in self.real.indexed_iter().zip(self.imag.iter()) {
            result[idx.clone()] = (real_val * real_val + imag_val * imag_val).sqrt();
        }
        result
    }

    pub fn phase(&self) -> ArrayD<f32> {
        let mut result = ArrayD::zeros(self.real.shape());
        for ((idx, &real_val), &imag_val) in self.real.indexed_iter().zip(self.imag.iter()) {
            result[idx.clone()] = imag_val.atan2(real_val);
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct HolographicWave {
    pub amplitude: ArrayD<f32>,
    pub phase: ArrayD<f32>,
    pub frequency: ArrayD<f32>,
}

impl HolographicWave {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            amplitude: ArrayD::zeros(shape.clone()),
            phase: ArrayD::zeros(shape.clone()),
            frequency: ArrayD::zeros(shape),
        }
    }

    pub fn to_complex(&self) -> DLResult<ComplexTensor> {
        ComplexTensor::from_polar(&self.amplitude, &self.phase)
    }

    pub fn evaluate(&self, position: f32) -> ArrayD<f32> {
        let mut result = ArrayD::zeros(self.amplitude.shape());
        for ((idx, &amp), (&phase, &freq)) in self
            .amplitude
            .indexed_iter()
            .zip(self.phase.iter().zip(self.frequency.iter()))
        {
            let value = amp * (2.0 * std::f32::consts::PI * freq * position + phase).cos();
            result[idx.clone()] = value;
        }
        result
    }
}

#[cfg(test)]
pub mod model_tests {
    use super::*;

    #[test]
    fn test_echo_net_parameters() {
        let config = EchoNetConfig {
            vocab_size: 100,
            embedding_dim: 16,
            amplitude_dim: 8,
            phase_dim: 4,
            resonance_dim: 4,
            num_bands: 1,
            band_frequencies: vec![1.0],
            kernel_size: 2,
            memory_size: 10,
            compression_levels: 1,
            compression_ratio: 0.5,
            output_size: 4,
            ..Default::default()
        };
        let model = model::EchoNetModel::new(config).unwrap();
        let params = model.parameters();
        assert!(
            !params.is_empty(),
            "EchoNet should have trainable parameters"
        );
        assert_eq!(params.len(), 9, "Expected 9 trainable parameter tensors");
    }
}
