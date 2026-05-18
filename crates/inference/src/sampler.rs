use rand::Rng;
use rand::SeedableRng;
use tracing::debug;

use crate::{InferenceError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    Temperature,
    TopK,
    TopP,
    TemperatureTopKTopP,
}

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub method: SamplingMethod,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_prob: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: SamplingMethod::TemperatureTopKTopP,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            min_prob: 1e-6,
            seed: None,
        }
    }
}

pub struct Sampler {
    config: SamplingConfig,
    rng: Option<rand::rngs::StdRng>,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = config.seed.map(rand::rngs::StdRng::seed_from_u64);
        Self { config, rng }
    }

    pub fn with_seed(config: SamplingConfig, seed: u64) -> Self {
        let mut cfg = config;
        cfg.seed = Some(seed);
        Self {
            rng: Some(rand::rngs::StdRng::seed_from_u64(seed)),
            config: cfg,
        }
    }

    pub fn reset_seed(&mut self, seed: u64) {
        self.rng = Some(rand::rngs::StdRng::seed_from_u64(seed));
        self.config.seed = Some(seed);
    }

    /// Sample a token index from logits.
    /// Pipeline: logits → temperature scaling → softmax → top-k filter → top-p filter → sample
    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        tracing::trace!(len = logits.len(), method = ?self.config.method, "sampling from logits");

        if logits.is_empty() {
            return Err(InferenceError::DecodingError("Empty logits".to_string()));
        }

        self.validate_logits(logits)?;

        let probs = match self.config.method {
            SamplingMethod::Greedy => return Ok(self.argmax(logits)),
            SamplingMethod::Temperature => self.sample_temperature(logits),
            SamplingMethod::TopK => self.sample_topk(logits),
            SamplingMethod::TopP => self.sample_topp(logits),
            SamplingMethod::TemperatureTopKTopP => self.sample_full(logits),
        }?;

        let idx = self.sample_from_probs(&probs)?;
        Ok(idx)
    }

    pub fn sample_multiple(&mut self, logits: &[f32], count: usize) -> Result<Vec<usize>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count > logits.len() {
            return Err(InferenceError::DecodingError(format!(
                "Cannot sample {} tokens from {} logits",
                count,
                logits.len()
            )));
        }

        let mut results = Vec::with_capacity(count);
        let mut remaining = logits.to_vec();

        for _ in 0..count {
            let idx = self.sample(&remaining)?;
            results.push(idx);
            remaining[idx] = f32::NEG_INFINITY;
        }

        Ok(results)
    }

    pub fn score_tokens(&self, logits: &[f32]) -> Result<Vec<(usize, f32)>> {
        let probs = softmax(logits);
        let mut scored: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    pub fn get_stats(&self) -> SamplingStats {
        SamplingStats {
            method: self.config.method.clone(),
            temperature: self.config.temperature,
            top_k: self.config.top_k,
            top_p: self.config.top_p,
            min_prob: self.config.min_prob,
        }
    }

    // --- private ---

    fn sample_temperature(&self, logits: &[f32]) -> Result<Vec<f32>> {
        Ok(scaled_softmax(logits, self.config.temperature))
    }

    fn sample_topk(&self, logits: &[f32]) -> Result<Vec<f32>> {
        Ok(top_k_filter(&softmax(logits), self.config.top_k))
    }

    fn sample_topp(&self, logits: &[f32]) -> Result<Vec<f32>> {
        Ok(top_p_filter(&softmax(logits), self.config.top_p))
    }

    fn sample_full(&self, logits: &[f32]) -> Result<Vec<f32>> {
        let mut probs = scaled_softmax(logits, self.config.temperature);
        if self.config.top_k > 0 && self.config.top_k < probs.len() {
            probs = top_k_filter(&probs, self.config.top_k);
        }
        if self.config.top_p > 0.0 && self.config.top_p < 1.0 {
            probs = top_p_filter(&probs, self.config.top_p);
        }
        Ok(probs)
    }

    fn validate_logits(&self, logits: &[f32]) -> Result<()> {
        if logits.iter().any(|&v| v.is_nan() || v.is_infinite()) {
            return Err(InferenceError::DecodingError(
                "Logits contain NaN or infinity".to_string(),
            ));
        }
        Ok(())
    }

    fn sample_from_probs(&mut self, probs: &[f32]) -> Result<usize> {
        let r: f32 = match &mut self.rng {
            Some(rng) => rng.gen(),
            None => rand::thread_rng().gen(),
        };
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                return Ok(i);
            }
        }
        Ok(probs.len() - 1)
    }

    fn argmax(&self, logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(SamplingConfig::default())
    }
}

#[derive(Debug, Clone)]
pub struct SamplingStats {
    pub method: SamplingMethod,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_prob: f32,
}

// -- Advanced sampler with repetition penalty --
pub struct AdvancedSampler {
    base: Sampler,
    history: Vec<usize>,
    max_history: usize,
    repetition_penalty: f32,
}

impl AdvancedSampler {
    pub fn new(sampler: Sampler, max_history: usize, repetition_penalty: f32) -> Self {
        Self {
            base: sampler,
            history: Vec::with_capacity(max_history),
            max_history,
            repetition_penalty,
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        let mut adjusted = logits.to_vec();
        for &past in &self.history {
            if let Some(v) = adjusted.get_mut(past) {
                *v /= self.repetition_penalty.max(0.01);
            }
        }
        let idx = self.base.sample(&adjusted)?;
        self.history.push(idx);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        Ok(idx)
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

#[derive(Debug, Clone)]
pub struct RepetitionStats {
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub max_frequency: usize,
    pub repetition_rate: f32,
}

// --- Utility functions ---

pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return logits.to_vec();
    }
    logits.iter().map(|&l| l / temperature).collect()
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut result = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &l in logits {
        let e = (l - max_val).exp();
        sum += e;
        result.push(e);
    }
    if sum <= 0.0 || !sum.is_finite() {
        let n = logits.len() as f32;
        result.iter_mut().for_each(|v| *v = 1.0 / n);
    } else {
        for v in result.iter_mut() {
            *v /= sum;
        }
    }
    result
}

/// Combines temperature scaling + softmax into a single pass (one Vec allocation)
fn scaled_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return softmax(logits);
    }
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut result = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &l in logits {
        let e = ((l - max_val) / temperature).exp();
        sum += e;
        result.push(e);
    }
    if sum <= 0.0 || !sum.is_finite() {
        let n = logits.len() as f32;
        result.iter_mut().for_each(|v| *v = 1.0 / n);
    } else {
        for v in result.iter_mut() {
            *v /= sum;
        }
    }
    result
}

pub fn top_k_filter(probs: &[f32], k: usize) -> Vec<f32> {
    if k == 0 || k >= probs.len() {
        return probs.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = indexed[k.saturating_sub(1)].1;
    let mut filtered: Vec<f32> = probs
        .iter()
        .map(|&p| if p >= threshold { p } else { 0.0 })
        .collect();

    let sum: f32 = filtered.iter().sum();
    if sum > 0.0 {
        for v in filtered.iter_mut() {
            *v /= sum;
        }
    }
    filtered
}

pub fn top_p_filter(probs: &[f32], p: f32) -> Vec<f32> {
    if p <= 0.0 || p >= 1.0 - f32::EPSILON {
        return probs.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cum += prob;
        if cum >= p {
            cutoff = i + 1;
            break;
        }
    }

    let mut filtered = vec![0.0; probs.len()];
    for &(idx, prob) in indexed.iter().take(cutoff) {
        filtered[idx] = prob;
    }

    let sum: f32 = filtered.iter().sum();
    if sum > 0.0 {
        for v in filtered.iter_mut() {
            *v /= sum;
        }
    }
    filtered
}

// --- Presets ---
pub mod configs {
    use super::*;

    pub fn greedy() -> SamplingConfig {
        SamplingConfig {
            method: SamplingMethod::Greedy,
            ..Default::default()
        }
    }

    pub fn conservative() -> SamplingConfig {
        SamplingConfig {
            method: SamplingMethod::TemperatureTopKTopP,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.85,
            ..Default::default()
        }
    }

    pub fn creative() -> SamplingConfig {
        SamplingConfig {
            method: SamplingMethod::TemperatureTopKTopP,
            temperature: 1.2,
            top_k: 100,
            top_p: 0.95,
            ..Default::default()
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty_input() {
        let logits: Vec<f32> = vec![];
        let probs = softmax(&logits);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_all_identical() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);
        for p in &probs {
            assert!((p - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = vec![0.1, 5.0, 0.5, 2.0];
        let scaled = apply_temperature(&logits, 0.0);
        assert_eq!(scaled, logits); // no-op, should use greedy separately
    }

    #[test]
    fn test_temperature_scales_logits() {
        let logits = vec![1.0, 2.0, 3.0];
        let scaled = apply_temperature(&logits, 2.0);
        assert!((scaled[0] - 0.5).abs() < 1e-5);
        assert!((scaled[1] - 1.0).abs() < 1e-5);
        assert!((scaled[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_filter() {
        let probs = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let filtered = top_k_filter(&probs, 2);
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(filtered[1], 0.5 / 0.8); // 0.5/(0.5+0.3)
        assert_eq!(filtered[2], 0.3 / 0.8);
        assert_eq!(filtered[0], 0.0);
    }

    #[test]
    fn test_top_p_filter() {
        let probs = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let filtered = top_p_filter(&probs, 0.8);
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sampler_greedy_returns_max() {
        let config = SamplingConfig {
            method: SamplingMethod::Greedy,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let logits = vec![0.1, 9.0, 1.0, 2.0];
        assert_eq!(sampler.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn test_sampler_deterministic_seed() {
        let config = SamplingConfig {
            method: SamplingMethod::TemperatureTopKTopP,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            seed: Some(42),
            ..Default::default()
        };
        let mut s1 = Sampler::new(config.clone());
        let mut s2 = Sampler::new(config);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(s1.sample(&logits).unwrap(), s2.sample(&logits).unwrap());
    }

    #[test]
    fn test_sampler_handles_nan() {
        let config = SamplingConfig::default();
        let mut sampler = Sampler::new(config);
        let logits = vec![1.0, f32::NAN, 3.0];
        assert!(sampler.sample(&logits).is_err());
    }

    #[test]
    fn test_advanced_sampler_repetition_penalty() {
        let sampler = Sampler::new(SamplingConfig {
            method: SamplingMethod::Greedy,
            ..Default::default()
        });
        let mut adv = AdvancedSampler::new(sampler, 100, 1.2);
        let logits = vec![10.0, 0.1, 0.1]; // token 0 is max
        let first = adv.sample(&logits).unwrap();
        assert_eq!(first, 0);
    }

    #[test]
    fn test_score_tokens_ordered() {
        let sampler = Sampler::new(SamplingConfig::default());
        let logits = vec![1.0, 5.0, 2.0];
        let scored = sampler.score_tokens(&logits).unwrap();
        assert_eq!(scored[0].0, 1); // highest prob
    }
}
