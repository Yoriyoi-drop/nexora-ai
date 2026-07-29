//! Evaluation metrics and harness for Nexora models.
//!
//! Provides `EvalMetrics` and evaluation helpers that work with
//! any `CausalLM` via `TrainableCausalLM`.

use nexora_deeplearning::autograd::ops::cross_entropy_loss;
use nexora_deeplearning::autograd::{clear_tape, Tensor, TensorOps};
#[cfg(feature = "gpu")]
use nexora_deeplearning::autograd::set_gpu_auto_create;
use nexora_transformer::{CausalLM, TrainableCausalLM, TransformerConfig};
use tracing::info;

/// Try to enable GPU acceleration for evaluation.
/// Returns `true` if GPU is active and auto-create was enabled.
#[cfg(feature = "gpu")]
pub fn try_init_gpu() -> bool {
    match nexora_deeplearning::autograd::gpu::GpuContext::init() {
        Ok(_ctx) => {
            set_gpu_auto_create(true);
            info!("Evaluation GPU acceleration enabled");
            true
        }
        Err(e) => {
            info!("GPU not available for evaluation: {e}");
            false
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub fn try_init_gpu() -> bool {
    false
}

/// Evaluation metrics for language model evaluation.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalMetrics {
    pub avg_loss: f64,
    pub perplexity: f64,
    pub total_tokens: usize,
}

impl EvalMetrics {
    pub fn new(avg_loss: f64, total_tokens: usize) -> Self {
        Self {
            avg_loss,
            perplexity: if total_tokens > 0 { avg_loss.exp() } else { 1.0 },
            total_tokens,
        }
    }

    /// Compute evaluation metrics by running the model on pre-tokenized sequences.
    ///
    /// Each sequence in `sequences` should be token IDs (including the final token
    /// that serves as the target for the second-to-last position).
    /// `seq_length` controls the sliding window size for loss computation.
    pub fn evaluate_loss(
        model: &CausalLM,
        _config: &TransformerConfig,
        sequences: &[Vec<u32>],
        seq_length: usize,
    ) -> Self {
        let mut total_loss = 0.0_f64;
        let mut total_tokens = 0;

        // Enable GPU acceleration if available.
        // This causes all Tensor::from_slice() calls below to create GPU tensors,
        // and the forward pass routes through GPU via autograd automatically.
        let _gpu_active = try_init_gpu();

        let trainable = TrainableCausalLM::from_inference(model);

        for tokens in sequences {
            if tokens.len() < 2 {
                continue;
            }
            for chunk in tokens.chunks(seq_length + 1) {
                if chunk.len() < 2 {
                    continue;
                }
                let input_t = Tensor::from_slice(
                    &chunk[..chunk.len() - 1]
                        .iter()
                        .map(|&x| x as f32)
                        .collect::<Vec<_>>(),
                    &[chunk.len() - 1],
                );
                let target_t = Tensor::from_slice(
                    &chunk[1..].iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    &[chunk.len() - 1],
                );
                let logits = trainable.forward(&input_t);
                let loss = cross_entropy_loss(&logits, &target_t).mean();
                let step_loss = loss.data()[0] as f64;
                if !step_loss.is_finite() {
                    info!("Non-finite loss encountered during evaluation — skipping chunk");
                    continue;
                }
                total_loss += step_loss * (chunk.len() - 1) as f64;
                total_tokens += chunk.len() - 1;
            }
        }

        clear_tape();

        info!(
            "Evaluation complete: avg_loss={:.4}, perplexity={:.2}, tokens={}",
            if total_tokens > 0 {
                total_loss / total_tokens as f64
            } else {
                0.0
            },
            if total_tokens > 0 {
                (total_loss / total_tokens as f64).exp()
            } else {
                0.0
            },
            total_tokens
        );

        Self::new(
            if total_tokens > 0 {
                total_loss / total_tokens as f64
            } else {
                0.0
            },
            total_tokens,
        )
    }
}

/// Trait for models that can be evaluated.
pub trait Evaluate {
    /// Run evaluation on tokenized input sequences, returning metrics.
    fn evaluate(&self, config: &TransformerConfig, sequences: &[Vec<u32>], seq_length: usize) -> EvalMetrics;
}

impl Evaluate for CausalLM {
    fn evaluate(&self, _config: &TransformerConfig, sequences: &[Vec<u32>], seq_length: usize) -> EvalMetrics {
        EvalMetrics::evaluate_loss(self, _config, sequences, seq_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_metrics_new() {
        let metrics = EvalMetrics::new(2.3026, 100);
        assert!((metrics.perplexity - 10.0).abs() < 0.1);
        assert_eq!(metrics.total_tokens, 100);
    }

    #[test]
    fn test_eval_metrics_zero_tokens() {
        let metrics = EvalMetrics::new(0.0, 0);
        assert_eq!(metrics.perplexity, 1.0);
    }
}
