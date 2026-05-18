use ndarray::Array1;
use rand::Rng;
use crate::models::transformer::CausalLM;
use super::config::TransformerConfig;
use super::gqa::KVCacheEntry;

#[derive(Debug, Clone)]
pub struct MTPConfig {
    pub num_predictions: usize,
    pub head_hidden_dim: usize,
    pub temperature: f32,
    pub top_k: usize,
}

impl Default for MTPConfig {
    fn default() -> Self {
        Self {
            num_predictions: 2,
            head_hidden_dim: 256,
            temperature: 0.0,
            top_k: 0,
        }
    }
}

impl MTPConfig {
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

pub struct MTPHeads {
    pub config: MTPConfig,
    pub hidden_norms: Vec<ndarray::Array1<f32>>,
    pub head_weights: Vec<ndarray::Array2<f32>>,
    pub head_biases: Vec<ndarray::Array1<f32>>,
}

impl MTPHeads {
    pub fn new(config: MTPConfig, vocab_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (hidden_size as f32).sqrt().recip();
        let head_hidden = config.head_hidden_dim;

        let mut hidden_norms = Vec::with_capacity(config.num_predictions);
        let mut head_weights = Vec::with_capacity(config.num_predictions);
        let mut head_biases = Vec::with_capacity(config.num_predictions);

        for _ in 0..config.num_predictions {
            let norm = ndarray::Array1::from_shape_fn(head_hidden, |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            });
            hidden_norms.push(norm);

            let w = ndarray::Array2::from_shape_fn((vocab_size, head_hidden), |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            });
            head_weights.push(w);

            let b = ndarray::Array1::from_shape_fn(vocab_size, |_| {
                rng.gen::<f32>() * 2.0 * scale - scale
            });
            head_biases.push(b);
        }

        Self { config, hidden_norms, head_weights, head_biases }
    }

    pub fn forward(&self, last_hidden: &[f32], depth: usize) -> Array1<f32> {
        let hidden_size = last_hidden.len();
        let head_hidden = self.config.head_hidden_dim;

        let idx = depth.min(self.config.num_predictions - 1);

        let mut projected = Array1::zeros(head_hidden);
        let head_hidden_actual = self.hidden_norms[idx].len();
        let proj_len = head_hidden.min(hidden_size);
        for j in 0..proj_len {
            projected[j] = last_hidden[j % hidden_size] * self.hidden_norms[idx][j % head_hidden_actual];
        }

        let mut logits = Array1::zeros(self.head_weights[idx].shape()[0]);
        let vocab_size = logits.len();
        for i in 0..vocab_size {
            let mut dot = self.head_biases[idx][i];
            for j in 0..head_hidden_actual.min(head_hidden) {
                let hv = if j < projected.len() { projected[j] } else { 0.0 };
                dot += hv * self.head_weights[idx][[i, j]];
            }
            logits[i] = dot;
        }

        logits
    }

    fn softmax(logits: &Array1<f32>) -> Array1<f32> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        Array1::from_shape_fn(logits.len(), |i| {
            if sum > 0.0 { exps[i] / sum } else { 1.0 / logits.len() as f32 }
        })
    }

    pub fn sample(&self, logits: &Array1<f32>) -> u32 {
        let t = self.config.temperature;
        if t <= 0.0 {
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] { best = i; }
            }
            return best as u32;
        }
        let scaled: Array1<f32> = logits.mapv(|v| v / t);
        let probs = Self::softmax(&scaled);

        let k = self.config.top_k;
        if k > 0 && k < probs.len() {
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            let k = k.min(probs.len());
            let sum_k: f32 = indices[..k].iter().map(|&i| probs[i]).sum();
            if sum_k <= 0.0 { return indices[0] as u32; }
            let r: f32 = rand::random::<f32>() * sum_k;
            let mut cum = 0.0;
            for &i in &indices[..k] {
                cum += probs[i];
                if cum >= r { return i as u32; }
            }
            return indices[k - 1] as u32;
        }

        let r: f32 = rand::random();
        let mut cum = 0.0;
        for i in 0..probs.len() {
            cum += probs[i];
            if cum >= r { return i as u32; }
        }
        (probs.len() - 1) as u32
    }
}

pub struct MTPInference {
    pub model: CausalLM,
    pub mtp_heads: MTPHeads,
    pub config: MTPConfig,
}

impl MTPInference {
    pub fn new(model: CausalLM, mtp_heads: MTPHeads) -> Self {
        let config = mtp_heads.config.clone();
        Self { model, mtp_heads, config }
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        max_tokens: usize,
    ) -> (Vec<u32>, Vec<KVCacheEntry>) {
        let mut cache = self.model.reset_cache();

        for &token_id in prompt_ids {
            self.model.forward(&[token_id], &mut cache);
        }

        let mut output = Vec::new();
        let mut last_id = *prompt_ids.last().unwrap_or(&0);

        let mut i = 0;
        while i < max_tokens {
            let logits = self.model.forward(&[last_id], &mut cache);
            let main_id = crate::models::transformer::model::sample_token(
                &logits,
                0.0,
                0,
            );
            output.push(main_id);

            for d in 0..self.config.num_predictions {
                let mut ph = Array1::zeros(logits.len());
                let len = ph.len().min(logits.len());
                for j in 0..len {
                    ph[j] = match d {
                        0 if logits.len() > j => logits[j],
                        _ if d > 0 && !output.is_empty() => {
                            let prev = output[output.len().saturating_sub(d + 1)] as usize;
                            if prev < logits.len() { logits[prev] } else { 0.0 }
                        }
                        _ => 0.0,
                    };
                }
                let draft_logits = self.mtp_heads.forward(ph.as_slice().unwrap_or(&[]), d);
                let draft_id = self.mtp_heads.sample(&draft_logits);

                if draft_id == main_id || self.config.temperature > 0.5 {
                    if i + 1 < max_tokens && d < self.config.num_predictions.saturating_sub(1) {
                        let verified_logits = self.model.forward(&[draft_id], &mut cache);
                        let verified_id = crate::models::transformer::model::sample_token(
                            &verified_logits,
                            0.0,
                            0,
                        );
                        if verified_id == draft_id {
                            output.push(draft_id);
                            i += 1;
                            last_id = draft_id;
                        }
                    }
                }
            }

            i += 1;
            last_id = main_id;
            if main_id == 0 {
                break;
            }
        }

        (output, cache)
    }

    pub fn mtp_loss(&self, last_hidden: &[f32], target_tokens: &[u32]) -> f64 {
        if target_tokens.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0f64;
        for d in 0..self.config.num_predictions.min(target_tokens.len()) {
            let logits = self.mtp_heads.forward(last_hidden, d);
            let target = target_tokens[d] as usize;
            if target < logits.len() {
                let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let log_sum: f32 = logits.iter().map(|&x| (x - max_val).exp().ln()).sum();
                let log_prob = logits[target] - max_val - log_sum;
                total_loss += -log_prob as f64;
            }
        }
        total_loss / self.config.num_predictions as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transformer::TransformerConfig;

    fn test_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 1000,
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 128,
            intermediate_size: 256,
            ..Default::default()
        }
    }

    #[test]
    fn test_mtp_heads_creation() {
        let config = MTPConfig::default();
        let heads = MTPHeads::new(config, 1000, 64);
        assert_eq!(heads.hidden_norms.len(), 2);
        assert_eq!(heads.head_weights.len(), 2);
        assert_eq!(heads.head_biases.len(), 2);
    }

    #[test]
    fn test_mtp_forward_produces_logits() {
        let config = MTPConfig::default();
        let heads = MTPHeads::new(config, 1000, 64);
        let last_hidden = vec![0.1f32; 64];
        let logits = heads.forward(&last_hidden, 0);
        assert_eq!(logits.len(), 1000);
    }

    #[test]
    fn test_mtp_loss_finite() {
        let config = MTPConfig::default();
        let heads = MTPHeads::new(config, 1000, 64);
        let cfg = MTPConfig::default();
        let model = CausalLM::new(test_config());
        let mtp = MTPInference::new(model, heads);
        let loss = mtp.mtp_loss(&[0.1f32; 64], &[5, 10]);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_mtp_generate_does_not_panic() {
        let config = MTPConfig { temperature: 1.0, ..Default::default() };
        let heads = MTPHeads::new(config.clone(), 1000, 64);
        let model = CausalLM::new(test_config());
        let mtp = MTPInference::new(model, heads);
        let (tokens, _) = mtp.generate(&[1, 2, 3], 10);
        assert!(!tokens.is_empty());
    }
}
