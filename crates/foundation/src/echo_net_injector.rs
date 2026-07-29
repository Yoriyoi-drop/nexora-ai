use ndarray::{Array1, Array2, ArrayD};
use nexora_deeplearning::echo_net::{AdaptivePhaseSeparationStabilizer, HolographicWave};
use nexora_transformer::{LayerInjector, TransformerError, TransformerResult};

/// Ring buffer entry: hidden state + phase vector for one token position.
#[derive(Clone)]
struct TokenPhase {
    hidden: Array1<f32>,
    phase: Array1<f32>,
}

/// Injects EchoNet APSS processing between transformer layers.
///
/// Maintains a sliding window of hidden states and their holographic phase
/// representations. After each transformer layer at the configured index,
/// runs APSS phase stabilization and applies a learned-free amplitude
/// modulation to the current hidden state based on the phase adjustment.
pub struct EchoNetInjector {
    apss: AdaptivePhaseSeparationStabilizer,
    buffer: Vec<TokenPhase>,
    max_window: usize,
    alpha: f32,
    hidden_size: usize,
}

impl std::fmt::Debug for EchoNetInjector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EchoNetInjector")
            .field("max_window", &self.max_window)
            .field("alpha", &self.alpha)
            .field("hidden_size", &self.hidden_size)
            .field("buffer_len", &self.buffer.len())
            .finish()
    }
}

impl EchoNetInjector {
    pub fn new(
        hidden_size: usize,
        phase_separation_strength: f32,
        max_window: usize,
        alpha: f32,
    ) -> TransformerResult<Self> {
        if hidden_size == 0 {
            return Err(TransformerError::Implementation(
                "hidden_size must be greater than 0".to_string(),
            ));
        }
        let apss = AdaptivePhaseSeparationStabilizer::new(
            hidden_size,
            phase_separation_strength,
            0.5, // similarity_threshold
            0.5, // max_phase_adjustment
        )
        .map_err(|e| TransformerError::Implementation(format!("APSS init: {}", e)))?;

        Ok(Self {
            apss,
            buffer: Vec::with_capacity(max_window),
            max_window,
            alpha,
            hidden_size,
        })
    }
}

impl LayerInjector for EchoNetInjector {
    fn reset(&mut self) {
        self.buffer.clear();
    }

    fn after_layer(
        &mut self,
        _layer_idx: usize,
        h: &mut Array2<f32>,
        _pos: usize,
    ) -> TransformerResult<()> {
        let current = h.row(0).to_owned();

        // Initialize phase for this position if not yet tracked
        let phase = if self.buffer.len() <= _pos {
            Array1::zeros(self.hidden_size)
        } else {
            self.buffer[_pos].phase.clone()
        };

        // Add to buffer (or update existing)
        let entry = TokenPhase {
            hidden: current.clone(),
            phase,
        };

        if _pos < self.buffer.len() {
            self.buffer[_pos] = entry;
        } else {
            self.buffer.push(entry);
        }

        // Prune if over max window
        while self.buffer.len() > self.max_window {
            self.buffer.remove(0);
        }

        // Need at least 2 tokens for APSS to do anything meaningful
        if self.buffer.len() < 2 {
            return Ok(());
        }

        // Build HolographicWave from buffer
        let seq_len = self.buffer.len();
        let mut amplitude_data = Vec::with_capacity(seq_len * self.hidden_size);
        let mut phase_data = Vec::with_capacity(seq_len * self.hidden_size);
        let frequency_data = vec![0.0_f32; seq_len * self.hidden_size];

        for tp in &self.buffer {
            amplitude_data.extend_from_slice(tp.hidden.as_slice().ok_or_else(|| {
                TransformerError::Implementation("hidden state not contiguous".into())
            })?);
            phase_data.extend_from_slice(tp.phase.as_slice().ok_or_else(|| {
                TransformerError::Implementation("phase vector not contiguous".into())
            })?);
        }

        let amplitude = ArrayD::from_shape_vec(vec![seq_len, self.hidden_size], amplitude_data)
            .map_err(|e| TransformerError::Implementation(format!("amplitude shape: {}", e)))?;
        let phase = ArrayD::from_shape_vec(vec![seq_len, self.hidden_size], phase_data)
            .map_err(|e| TransformerError::Implementation(format!("phase shape: {}", e)))?;
        let frequency = ArrayD::from_shape_vec(vec![seq_len, self.hidden_size], frequency_data)
            .map_err(|e| TransformerError::Implementation(format!("freq shape: {}", e)))?;

        let embeddings = amplitude.clone();
        let mut wave = HolographicWave {
            amplitude,
            phase,
            frequency,
        };

        // Save pre-APSS phases to compute delta
        let pre_phases = wave.phase.clone();

        // Run APSS
        self.apss
            .forward(&mut wave, &embeddings)
            .map_err(|e| TransformerError::Implementation(format!("APSS forward: {}", e)))?;

        // Compute phase delta for current (last) position
        let last_idx = wave.phase.shape()[0] - 1;
        let hidden_dim = wave.phase.shape()[1];

        let mut phase_delta_sum = 0.0_f32;
        for i in 0..hidden_dim {
            let before = pre_phases[[last_idx, i]];
            let after = wave.phase[[last_idx, i]];
            let diff = after - before;
            // Wrap to [-π, π]
            let wrapped = if diff > std::f32::consts::PI {
                diff - 2.0 * std::f32::consts::PI
            } else if diff < -std::f32::consts::PI {
                diff + 2.0 * std::f32::consts::PI
            } else {
                diff
            };
            phase_delta_sum += wrapped.abs();
        }
        let avg_phase_delta = phase_delta_sum / hidden_dim as f32;

        // Modulate current hidden state amplitude by phase change
        // h' = h * (1 + alpha * tanh(avg_phase_delta))
        let modulation = 1.0 + self.alpha * avg_phase_delta.tanh();
        h.mapv_inplace(|x| x * modulation);

        // Store updated phase back
        if let Some(last_entry) = self.buffer.last_mut() {
            let new_phase_slice = wave.phase.slice(ndarray::s![last_idx, ..]);
            last_entry.phase.assign(&new_phase_slice.to_owned());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_transformer::LayerInjector;

    #[test]
    fn test_echo_net_injector_new() {
        let injector = EchoNetInjector::new(64, 0.1, 10, 0.5).unwrap();
        assert_eq!(injector.max_window, 10);
        assert_eq!(injector.alpha, 0.5);
        assert_eq!(injector.hidden_size, 64);
    }

    #[test]
    fn test_echo_net_injector_new_zero_hidden_size() {
        let result = EchoNetInjector::new(0, 0.1, 10, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_echo_net_injector_debug() {
        let injector = EchoNetInjector::new(64, 0.1, 5, 0.3).unwrap();
        let debug = format!("{:?}", injector);
        assert!(debug.contains("max_window"));
        assert!(debug.contains("alpha"));
        assert!(debug.contains("hidden_size"));
    }

    #[test]
    fn test_echo_net_injector_buffer_empty_on_start() {
        let injector = EchoNetInjector::new(64, 0.1, 10, 0.5).unwrap();
        assert!(injector.buffer.is_empty());
    }

    #[test]
    fn test_echo_net_injector_after_layer_buffer_grows() {
        let mut injector = EchoNetInjector::new(64, 0.1, 10, 0.5).unwrap();
        let mut h = Array2::zeros((1, 64));
        // No panic on first call (buffer < 2, early return)
        let result = injector.after_layer(2, &mut h, 0);
        assert!(result.is_ok());
        assert_eq!(injector.buffer.len(), 1);
    }

    #[test]
    fn test_echo_net_injector_after_layer_two_tokens() {
        let mut injector = EchoNetInjector::new(64, 0.1, 10, 0.5).unwrap();
        let mut h1 = Array2::zeros((1, 64));
        let mut h2 = Array2::ones((1, 64));
        assert!(injector.after_layer(2, &mut h1, 0).is_ok());
        assert!(injector.after_layer(2, &mut h2, 1).is_ok());
        // With 2+ tokens, APSS runs
        assert!(injector.buffer.len() >= 2);
    }

    #[test]
    fn test_echo_net_injector_prune_buffer() {
        let mut injector = EchoNetInjector::new(64, 0.1, 3, 0.5).unwrap();
        for pos in 0..5 {
            let mut h = Array2::from_elem((1, 64), pos as f32);
            injector.after_layer(2, &mut h, pos).unwrap();
        }
        assert!(injector.buffer.len() <= 3);
    }
}
