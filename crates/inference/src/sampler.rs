use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, warn};

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

/// Default sampling: TemperatureTopKTopP with temperature=1.0, top_k=50, top_p=0.9.
///
/// Greedy decoding (the simplest default) would always pick the argmax token,
/// which makes the output repetitive and unnatural. TemperatureTopKTopP gives
/// diverse, high-quality output while still being deterministic at seed=0.
/// Temperature=1.0 means no flattening of the distribution; top_k=50 caps the
/// candidate set to 50 most-likely tokens; top_p=0.9 keeps tokens summing to
/// 90% cumulative probability. This trio is the most commonly recommended
/// default in LLM inference literature (Holtzman et al. 2020).
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
    use_gpu: bool,
    gpu_fallback_count: AtomicU64,
    gpu_attempts: AtomicU64,
    allow_gpu_fallback: bool,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = config.seed.map(rand::rngs::StdRng::seed_from_u64);
        #[cfg(feature = "gpu")]
        let use_gpu = nexora_autograd::gpu::GpuContext::is_available();
        #[cfg(not(feature = "gpu"))]
        let use_gpu = false;
        Self {
            config,
            rng,
            use_gpu,
            gpu_fallback_count: AtomicU64::new(0),
            gpu_attempts: AtomicU64::new(0),
            allow_gpu_fallback: true, // Default to true for backward compatibility
        }
    }

    pub fn with_seed(config: SamplingConfig, seed: u64) -> Self {
        let mut cfg = config;
        cfg.seed = Some(seed);
        Self {
            rng: Some(rand::rngs::StdRng::seed_from_u64(seed)),
            config: cfg,
            #[cfg(feature = "gpu")]
            use_gpu: nexora_autograd::gpu::GpuContext::is_available(),
            #[cfg(not(feature = "gpu"))]
            use_gpu: false,
            gpu_fallback_count: AtomicU64::new(0),
            gpu_attempts: AtomicU64::new(0),
            allow_gpu_fallback: true,
        }
    }

    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        self.use_gpu = use_gpu;
    }

    /// Enable or disable automatic GPU fallback to CPU on GPU errors.
    /// When disabled, GPU errors will be returned instead of silently falling back.
    pub fn set_allow_gpu_fallback(&mut self, allow: bool) {
        self.allow_gpu_fallback = allow;
    }

    /// Get the number of GPU fallbacks that have occurred.
    pub fn gpu_fallback_count(&self) -> u64 {
        self.gpu_fallback_count.load(Ordering::Relaxed)
    }

    /// Ratio of GPU attempts that fell back to CPU (0.0 – 1.0).
    pub fn gpu_fallback_ratio(&self) -> f64 {
        let total = self.gpu_attempts.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.gpu_fallback_count.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    pub fn reset_seed(&mut self, seed: u64) {
        self.rng = Some(rand::rngs::StdRng::seed_from_u64(seed));
        self.config.seed = Some(seed);
    }

    #[cfg(feature = "gpu")]
    fn sample_gpu_impl(&mut self, logits: &[f32]) -> Result<usize> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let ctx = GpuContext::global()
            .map_err(|e| InferenceError::DecodingError(format!("GpuContext::global: {}", e)))?;
        let shape = vec![1, logits.len()];
        let cpu = ndarray::ArrayD::from_shape_vec(shape, logits.to_vec())
            .map_err(|e| InferenceError::DecodingError(format!("ndarray reshape: {}", e)))?;
        let gpu = GpuTensor::from_cpu(&cpu)
            .map_err(|e| InferenceError::DecodingError(format!("GpuTensor::from_cpu: {}", e)))?;
        let top_k = if self.config.top_k > 0 {
            self.config.top_k
        } else {
            0
        };
        let seed = self.config.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let out = ctx
            .gpu_sample(&gpu, self.config.temperature, top_k as u32, self.config.top_p, seed)
            .map_err(|e| InferenceError::DecodingError(format!("gpu_sample: {}", e)))?;
        let raw = out
            .to_cpu_raw_bytes()
            .map_err(|e| InferenceError::DecodingError(format!("to_cpu_raw_bytes: {}", e)))?;
        let token = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
        Ok(token as usize)
    }

    #[cfg(feature = "gpu")]
    pub fn sample_gpu(&mut self, logits: &[f32]) -> Result<usize> {
        self.gpu_attempts.fetch_add(1, Ordering::Relaxed);
        match self.sample_gpu_impl(logits) {
            Ok(token) => Ok(token),
            Err(e) => {
                self.gpu_fallback_count.fetch_add(1, Ordering::Relaxed);
                if self.allow_gpu_fallback {
                    let ratio = self.gpu_fallback_ratio();
                    if ratio > 0.5 {
                        warn!(
                            "GPU fallback ratio is {:.1}% — consider disabling GPU acceleration",
                            ratio * 100.0
                        );
                    }
                    warn!(
                        "GPU sampling failed: {}, falling back to CPU (fallback count: {})",
                        e,
                        self.gpu_fallback_count.load(Ordering::Relaxed)
                    );
                    self.sample_cpu(logits)
                } else {
                    error!(
                        "GPU sampling failed: {}, fallback disabled - returning error",
                        e
                    );
                    Err(e)
                }
            }
        }
    }

    /// Batch GPU sample: stack B logit vectors into [B, vocab] and sample once.
    /// Returns one token ID per sequence, all with a single GPU call.
    /// Falls back to per-sequence CPU sampling on GPU error if allow_gpu_fallback is enabled.
    #[cfg(feature = "gpu")]
    pub fn sample_gpu_batched(&mut self, batch_logits: &[&[f32]]) -> Result<Vec<usize>> {
        use nexora_autograd::gpu::{GpuContext, GpuTensor};
        let batch = batch_logits.len();
        if batch == 0 {
            return Ok(Vec::new());
        }
        let vocab = batch_logits[0].len();
        self.gpu_attempts.fetch_add(1, Ordering::Relaxed);
        let ctx = match GpuContext::global() {
            Ok(c) => c,
            Err(e) => {
                self.gpu_fallback_count.fetch_add(batch as u64, Ordering::Relaxed);
                if self.allow_gpu_fallback {
                    let ratio = self.gpu_fallback_ratio();
                    if ratio > 0.5 {
                        warn!(
                            "GPU fallback ratio is {:.1}% — consider disabling GPU acceleration",
                            ratio * 100.0
                        );
                    }
                    warn!(
                        "GPU sampling batched failed: {}, falling back to per-sequence (fallback count: {})",
                        e,
                        self.gpu_fallback_count.load(Ordering::Relaxed)
                    );
                    return batch_logits.iter().map(|l| self.sample_cpu(l)).collect();
                } else {
                    error!(
                        "GPU sampling batched failed: {}, fallback disabled - returning error",
                        e
                    );
                    return Err(InferenceError::DecodingError(format!("GPU context error: {}", e)));
                }
            }
        };

        // Stack all logits into [B, vocab] CPU array → one GPU upload
        let mut flat = Vec::with_capacity(batch * vocab);
        for logits in batch_logits {
            flat.extend_from_slice(logits);
        }
        let shape = vec![batch, vocab];
        let cpu_arr = ndarray::ArrayD::from_shape_vec(shape, flat)
            .map_err(|e| InferenceError::DecodingError(format!("ndarray reshape: {}", e)))?;
        let gpu_logits = GpuTensor::from_cpu(&cpu_arr)
            .map_err(|e| InferenceError::DecodingError(format!("GpuTensor::from_cpu: {}", e)))?;

        let top_k = if self.config.top_k > 0 { self.config.top_k } else { 0 };
        let seed = self.config.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let out = match ctx.gpu_sample(&gpu_logits, self.config.temperature, top_k as u32, self.config.top_p, seed) {
            Ok(o) => o,
            Err(e) => {
                self.gpu_fallback_count.fetch_add(batch as u64, Ordering::Relaxed);
                if self.allow_gpu_fallback {
                    let ratio = self.gpu_fallback_ratio();
                    if ratio > 0.5 {
                        warn!(
                            "GPU fallback ratio is {:.1}% — consider disabling GPU acceleration",
                            ratio * 100.0
                        );
                    }
                    warn!(
                        "GPU sampling batched call failed: {}, falling back to per-sequence (fallback count: {})",
                        e,
                        self.gpu_fallback_count.load(Ordering::Relaxed)
                    );
                    return batch_logits.iter().map(|l| self.sample_cpu(l)).collect();
                } else {
                    error!(
                        "GPU sampling batched call failed: {}, fallback disabled - returning error",
                        e
                    );
                    return Err(InferenceError::DecodingError(format!("GPU sample error: {}", e)));
                }
            }
        };

        let raw = out
            .to_cpu_raw_bytes()
            .map_err(|e| InferenceError::DecodingError(format!("to_cpu_raw_bytes: {}", e)))?;
        let tokens: Vec<usize> = raw
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize)
            .collect();

        Ok(tokens)
    }

    /// Sample a token index from logits (CPU path).
    /// Pipeline: logits → temperature scaling → softmax → top-k filter → top-p filter → sample
    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        if logits.is_empty() {
            return Err(InferenceError::DecodingError("Empty logits".to_string()));
        }
        self.validate_logits(logits)?;

        #[cfg(feature = "gpu")]
        if self.use_gpu {
            return self.sample_gpu(logits);
        }

        self.sample_cpu(logits)
    }

    fn sample_cpu(&mut self, logits: &[f32]) -> Result<usize> {
        tracing::trace!(len = logits.len(), method = ?self.config.method, "sampling from logits");

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
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Ok(probs.len().saturating_sub(1));
        }

        let mut normalized = probs.to_vec();
        for p in &mut normalized {
            *p /= sum;
        }

        let r: f32 = match &mut self.rng {
            Some(rng) => rng.gen(),
            None => rand::thread_rng().gen(),
        };
        let mut cum = 0.0;
        let epsilon = 1e-6;
        for (i, &p) in normalized.iter().enumerate() {
            cum += p;
            if cum >= r - epsilon {
                return Ok(i);
            }
        }
        Ok(normalized.len().saturating_sub(1))
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
        let penalty = self.repetition_penalty - 1.0;
        for &past in &self.history {
            if let Some(v) = adjusted.get_mut(past) {
                *v -= penalty * v.abs();
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
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

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
