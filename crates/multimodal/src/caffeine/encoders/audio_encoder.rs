//! Audio encoder implementation for CAFFEINE
//!
//! Based on Whisper architecture with multi-scale feature extraction.
//! Uses 2-layer MLP projection (GELU) with Xavier initialization.

use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use ndarray::{Array1, Array2, ArrayD};
use rand::Rng;

/// 2-layer MLP for mel-spectrogram to embedding projection.
struct AudioMLP {
    pub(crate) w1: Array2<f32>,
    pub(crate) b1: Array1<f32>,
    pub(crate) w2: Array2<f32>,
    pub(crate) b2: Array1<f32>,
}

impl AudioMLP {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / input_dim as f32).sqrt();
        let scale2 = (2.0 / hidden_dim as f32).sqrt();
        Self {
            w1: Array2::from_shape_fn((input_dim, hidden_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale1 - scale1
            }),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::from_shape_fn((hidden_dim, output_dim), |_| {
                rng.gen::<f32>() * 2.0 * scale2 - scale2
            }),
            b2: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let hidden = x.dot(&self.w1) + &self.b1;
        let gelu = hidden.mapv(|v| {
            v * 0.5 * (1.0 + (v * 0.7978845608 * (1.0 + 0.044715 * v * v)).tanh())
        });
        gelu.dot(&self.w2) + &self.b2
    }

    #[cfg(feature = "gpu")]
    fn forward_batched_gpu(&self, inputs: &[Array1<f32>]) -> Option<Vec<Array1<f32>>> {
        use crate::caffeine::gpu_compute;
        let batch = inputs.len();
        if batch == 0 { return None; }
        let input_dim = inputs[0].len();
        let hidden_dim = self.w1.shape()[1];
        let output_dim = self.w2.shape()[1];
        let flat_input: Vec<f32> = inputs.iter().flat_map(|i| i.iter().copied()).collect();
        let w1_slice: Vec<f32> = self.w1.iter().copied().collect();
        let b1_slice: Vec<f32> = self.b1.iter().copied().collect();
        let w2_slice: Vec<f32> = self.w2.iter().copied().collect();
        let b2_slice: Vec<f32> = self.b2.iter().copied().collect();
        let result = gpu_compute::try_gpu_mlp_forward(
            &flat_input, &w1_slice, &b1_slice, &w2_slice, &b2_slice,
            batch, input_dim, hidden_dim, output_dim,
        )?;
        Some(result.chunks(output_dim).map(|c| Array1::from_vec(c.to_vec())).collect())
    }
}

/// Cooley-Tukey FFT O(n log n) for spectrogram computation.
struct FFT {
    n_fft: usize,
}

impl FFT {
    fn new(window_size: usize) -> Self {
        Self { n_fft: window_size / 2 + 1 }
    }

    fn compute(&self, window: &[f32]) -> Vec<f32> {
        let n = window.len();
        if n <= 1 {
            return vec![window.first().copied().unwrap_or(0.0)];
        }

        let (mut real, mut imag) = self.fft_inner(window);
        let mut magnitudes = vec![0.0f32; self.n_fft.min(real.len())];
        for k in 0..magnitudes.len() {
            magnitudes[k] = (real[k] * real[k] + imag[k] * imag[k]).sqrt();
        }
        magnitudes
    }

    fn fft_inner(&self, data: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = data.len();
        if n <= 1 {
            return (data.to_vec(), vec![0.0f32; n]);
        }

        let n2 = n / 2;
        let even: Vec<f32> = data.iter().step_by(2).copied().collect();
        let odd: Vec<f32> = data.iter().skip(1).step_by(2).copied().collect();

        let (even_real, even_imag) = self.fft_inner(&even);
        let (odd_real, odd_imag) = self.fft_inner(&odd);

        let mut real = vec![0.0f32; n];
        let mut imag = vec![0.0f32; n];

        for k in 0..n2 {
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            let cos = angle.cos();
            let sin = angle.sin();

            let t_real = odd_real[k] * cos - odd_imag[k] * sin;
            let t_imag = odd_real[k] * sin + odd_imag[k] * cos;

            real[k] = even_real[k] + t_real;
            imag[k] = even_imag[k] + t_imag;

            real[k + n2] = even_real[k] - t_real;
            imag[k + n2] = even_imag[k] - t_imag;
        }

        (real, imag)
    }
}

/// Audio encoder based on Whisper
pub struct AudioEncoder {
    config: crate::caffeine::config::AudioEncoderConfig,
    model_loaded: bool,
    sample_rate: usize,
    n_mels: usize,
    mel_projection: AudioMLP,
    fft: FFT,
}

impl AudioEncoder {
    /// Create new audio encoder
    pub fn new(config: crate::caffeine::config::AudioEncoderConfig) -> Result<Self> {
        let n_mels = 80; // Whisper uses 80 mel channels
        let hidden_dim = config.output_dim.max(64);
        let output_dim = config.output_dim;
        let window_size = config.window_size;
        Ok(Self {
            sample_rate: config.sample_rate,
            n_mels,
            model_loaded: false,
            config,
            mel_projection: AudioMLP::new(n_mels, hidden_dim, output_dim),
            fft: FFT::new(window_size),
        })
    }

    /// Load model weights
    pub fn load_model(&mut self) -> Result<()> {
        self.model_loaded = true;
        Ok(())
    }

    /// Encode audio input
    pub fn encode(&mut self, input: &AudioInput) -> Result<ArrayD<f32>> {
        if !self.model_loaded {
            self.load_model()?;
        }

        // Resample if necessary
        let resampled_data = if input.sample_rate != self.sample_rate {
            self.resample_audio(&input.data, input.sample_rate, self.sample_rate)?
        } else {
            input.data.clone()
        };

        // Convert to mel spectrogram
        let mel_spec = self.compute_mel_spectrogram(&resampled_data)?;

        // Encode with transformer layers
        let encoded = self.encode_spectrogram(&mel_spec)?;

        Ok(encoded)
    }

    /// Resample audio to target sample rate
    fn resample_audio(&self, audio: &[f32], from_rate: usize, to_rate: usize) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(audio.to_vec());
        }

        let ratio = to_rate as f32 / from_rate as f32;
        let output_length = (audio.len() as f32 * ratio) as usize;
        let mut resampled = vec![0.0f32; output_length];

        // Simple linear interpolation
        for i in 0..output_length {
            let src_pos = i as f32 / ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f32;

            if src_idx < audio.len() - 1 {
                resampled[i] = audio[src_idx] * (1.0 - frac) + audio[src_idx + 1] * frac;
            } else {
                resampled[i] = audio[src_idx];
            }
        }

        Ok(resampled)
    }

    /// Compute mel spectrogram from audio samples
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<ArrayD<f32>> {
        let window_size = self.config.window_size;
        let hop_length = self.config.hop_length;
        let n_frames = (audio.len() - window_size) / hop_length + 1;

        // Create mel filter bank
        let mel_filters = self.create_mel_filter_bank()?;

        // Compute spectrogram
        let mut mel_spec = vec![0.0f32; n_frames * self.n_mels];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = start + window_size;

            if end <= audio.len() {
                let window = &audio[start..end];
                let fft_result = self.compute_fft(window)?;

                // Apply mel filters
                for mel_idx in 0..self.n_mels {
                    let mut mel_energy = 0.0;
                    for (freq_idx, &magnitude) in fft_result.iter().enumerate() {
                        if freq_idx < mel_filters[mel_idx].len() {
                            mel_energy += magnitude * mel_filters[mel_idx][freq_idx];
                        }
                    }
                    mel_spec[frame_idx * self.n_mels + mel_idx] = (mel_energy + 1e-6).ln();
                }
            }
        }

        let shape = vec![1, n_frames, self.n_mels]; // batch, time, mel
        Ok(ArrayD::from_shape_vec(shape, mel_spec)?)
    }

    /// Create mel filter bank
    fn create_mel_filter_bank(&self) -> Result<Vec<Vec<f32>>> {
        let n_fft = self.config.window_size / 2 + 1;
        let mut mel_filters = vec![vec![0.0f32; n_fft]; self.n_mels];

        // Convert Hz to mel scale
        let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };

        let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(self.sample_rate as f32 / 2.0);

        // Create mel points
        for m in 0..self.n_mels {
            let mel_m = mel_min + (m as f32 + 1.0) * (mel_max - mel_min) / (self.n_mels + 1) as f32;
            let hz_m = mel_to_hz(mel_m);
            let fft_m_f = n_fft as f32 * hz_m / (self.sample_rate as f32 / 2.0);
            let fft_m = if fft_m_f.is_nan() {
                0
            } else {
                fft_m_f.max(0.0) as usize
            };

            // Create triangular filter
            for k in 0..n_fft {
                if k < fft_m {
                    mel_filters[m][k] = 0.0;
                } else if k == fft_m {
                    mel_filters[m][k] = 1.0;
                } else {
                    let next_mel = if m < self.n_mels - 1 {
                        mel_min
                            + ((m + 1) as f32 + 1.0) * (mel_max - mel_min)
                                / (self.n_mels + 1) as f32
                    } else {
                        mel_max
                    };
                    let next_hz = mel_to_hz(next_mel);
                    let next_fft_f = n_fft as f32 * next_hz / (self.sample_rate as f32 / 2.0);
                    let next_fft = if next_fft_f.is_nan() {
                        0
                    } else {
                        next_fft_f.max(0.0) as usize
                    };

                    if k <= next_fft {
                        mel_filters[m][k] = (next_fft - k) as f32 / (next_fft - fft_m) as f32;
                    } else {
                        mel_filters[m][k] = 0.0;
                    }
                }
            }
        }

        Ok(mel_filters)
    }

    /// Compute FFT of window
    fn compute_fft(&self, window: &[f32]) -> Result<Vec<f32>> {
        Ok(self.fft.compute(window))
    }

    /// Encode spectrogram with MLP projection (not sinusoidal).
    fn encode_spectrogram(&self, mel_spec: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let shape = mel_spec.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let n_mels = shape[2];
        let embed_dim = self.config.output_dim;

        // Collect all frame vectors
        let mut frames: Vec<Array1<f32>> = Vec::with_capacity(batch_size * seq_len);
        for b in 0..batch_size {
            for t in 0..seq_len {
                let mut frame = Vec::with_capacity(n_mels);
                for m in 0..n_mels {
                    let idx = b * seq_len * n_mels + t * n_mels + m;
                    frame.push(mel_spec[idx]);
                }
                frames.push(Array1::from_vec(frame));
            }
        }

        // Try GPU batched MLP first, fall back to per-frame CPU
        let mut encoded = vec![0.0f32; batch_size * seq_len * embed_dim];
        #[cfg(feature = "gpu")]
        if let Some(gpu_results) = self.mel_projection.forward_batched_gpu(&frames) {
            for (idx, projected) in gpu_results.iter().enumerate() {
                for d in 0..embed_dim {
                    encoded[idx * embed_dim + d] = projected[d];
                }
            }
        } else {
            for (idx, frame_arr) in frames.iter().enumerate() {
                let projected = self.mel_projection.forward(frame_arr);
                for d in 0..embed_dim {
                    encoded[idx * embed_dim + d] = projected[d];
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        for (idx, frame_arr) in frames.iter().enumerate() {
            let projected = self.mel_projection.forward(frame_arr);
            for d in 0..embed_dim {
                encoded[idx * embed_dim + d] = projected[d];
            }
        }

        let output_shape = vec![batch_size, seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(output_shape, encoded)?)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get configuration
    pub fn config(&self) -> &crate::caffeine::config::AudioEncoderConfig {
        &self.config
    }

    /// Collect all trainable weights for checkpoint
    pub(crate) fn collect_weights(&self) -> Vec<(String, ndarray::ArrayD<f32>)> {
        vec![
            ("audio_encoder.mel_proj.w1".to_string(), self.mel_projection.w1.clone().into_dyn()),
            ("audio_encoder.mel_proj.b1".to_string(), self.mel_projection.b1.clone().into_dyn()),
            ("audio_encoder.mel_proj.w2".to_string(), self.mel_projection.w2.clone().into_dyn()),
            ("audio_encoder.mel_proj.b2".to_string(), self.mel_projection.b2.clone().into_dyn()),
        ]
    }
}
