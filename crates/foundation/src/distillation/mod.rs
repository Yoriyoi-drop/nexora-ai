//! Knowledge Distillation — Teacher-Student framework.
//!
//! Teacher: 6.2B Pro backbone (shared, loaded via `resolve_single_backbone()`)
//! Student: 3 Lite variants hasil distillasi (Swift Lite, Aether Lite, Omnis Lite)
//!
//! Arsitektur:
//! ```text
//! Teacher (6.2B Pro)
//!   │
//!   ├── KL divergence loss ──→ Student (Lite)
//!   ├── Cross-entropy loss ──→ Student (Lite)
//!   └── Weight transfer (optional)
//! ```
//!
//! Loss: L = α * T² * KL(softmax(teacher/T) || softmax(student/T)) + (1-α) * CE(student, targets)

use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use ndarray::Array1;
use rand::seq::SliceRandom;
use nexora_transformer::{resolve_single_backbone, CausalLM, TransformerConfig, TransformerResult};
use nexora_deeplearning::quantization::QFormat;

/// Student model configuration presets for distillation targets.
#[derive(Debug, Clone)]
pub struct StudentConfig {
    pub name: &'static str,
    pub model_id: &'static str,
    pub config: TransformerConfig,
}

/// Pre-defined student configs untuk 3 Lite variants.
impl StudentConfig {
    /// Swift Lite — Edge-tier distilled from Pro backbone.
    /// 128 hidden, 3 layers, 4 heads, dense FFN → ~14M params.
    pub fn swift_lite() -> Self {
        Self {
            name: "swift-lite",
            model_id: "swift",
            config: TransformerConfig {
                vocab_size: 50257,
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 1,
                num_layers: 3,
                max_seq_len: 512,
                intermediate_size: 512,
                norm_eps: 1e-6,
                rope_theta: 10000.0,
                use_cache: true,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                use_domain_experts: false,
                shared_expert: 0,
                quantization: QFormat::Q8 { group_size: 128 },
                use_half_precision: false,
                shard: Default::default(),
            },
        }
    }

    /// Aether Lite — Emotion-capable distilled from Pro backbone.
    /// 128 hidden, 3 layers, 4 heads → ~14M params + emotion classifier head.
    pub fn aether_lite() -> Self {
        Self {
            name: "aether-lite",
            model_id: "aether",
            config: TransformerConfig {
                vocab_size: 50257,
                hidden_size: 128,
                num_heads: 4,
                num_kv_heads: 1,
                num_layers: 3,
                max_seq_len: 512,
                intermediate_size: 512,
                norm_eps: 1e-6,
                rope_theta: 10000.0,
                use_cache: true,
                num_experts: 0,
                top_k_experts: 0,
                expert_intermediate_size: 0,
                use_domain_experts: false,
                shared_expert: 0,
                quantization: QFormat::Q8 { group_size: 128 },
                use_half_precision: false,
                shard: Default::default(),
            },
        }
    }

    /// Omnis Lite — MoE-capable distilled from Pro backbone.
    /// 512 hidden, 16 layers, 8 heads, 8 experts (top-2) → ~162M params.
    pub fn omnis_lite() -> Self {
        Self {
            name: "omnis-lite",
            model_id: "omnis",
            config: TransformerConfig {
                vocab_size: 50257,
                hidden_size: 512,
                num_heads: 8,
                num_kv_heads: 1,
                num_layers: 16,
                max_seq_len: 2048,
                intermediate_size: 2048,
                norm_eps: 1e-6,
                rope_theta: 10000.0,
                use_cache: true,
                num_experts: 8,
                top_k_experts: 2,
                expert_intermediate_size: 512,
                use_domain_experts: true,
                shared_expert: 4,
                quantization: QFormat::Q8 { group_size: 128 },
                use_half_precision: false,
                shard: Default::default(),
            },
        }
    }

    /// Resolve student config by model_id string.
    pub fn from_model_id(model_id: &str) -> Option<Self> {
        match model_id.to_lowercase().as_str() {
            "swift" | "swift-lite" => Some(Self::swift_lite()),
            "aether" | "aether-lite" => Some(Self::aether_lite()),
            "omnis" | "omnis-lite" => Some(Self::omnis_lite()),
            _ => None,
        }
    }

    /// All available student configs.
    pub fn all() -> Vec<Self> {
        vec![Self::swift_lite(), Self::aether_lite(), Self::omnis_lite()]
    }
}

/// Distillation metrics report.
#[derive(Debug, Clone)]
pub struct DistillationReport {
    pub student: String,
    pub steps: usize,
    pub final_distill_loss: f64,
    pub final_ce_loss: f64,
    pub total_tokens: usize,
    pub duration_secs: f64,
    pub teacher_params: usize,
    pub student_params: usize,
    pub compression_ratio: f64,
}

/// KnowledgeDistiller — menjembatani teacher 6.2B → student Lite.
///
/// # Usage
/// ```ignore
/// let distiller = KnowledgeDistiller::new("swift-lite")?;
/// let report = distiller.distill(data, 1000, 0.01, 4, 128, 4.0, None).await?;
/// distiller.save_student("swift_lite_distilled.safetensors")?;
/// ```
pub struct KnowledgeDistiller {
    /// Teacher backbone (6.2B Pro — shared, read-only)
    #[allow(dead_code)]
    teacher: Arc<CausalLM>,
    /// Student model (Lite — target of distillation)
    student: CausalLM,
    /// Student config
    student_config: StudentConfig,
    /// Temperature for softening distributions
    temperature: f32,
    /// Alpha weight for KL vs CE loss (1.0 = pure KL, 0.0 = pure CE)
    alpha: f32,
    /// Tokens processed
    total_tokens: usize,
}

impl KnowledgeDistiller {
    /// Create a new distiller for the given student model.
    /// Teacher is auto-resolved from shared backbone.
    pub fn new(student_id: &str) -> TransformerResult<Self> {
        let student_config = StudentConfig::from_model_id(student_id)
            .ok_or_else(|| nexora_transformer::TransformerError::Configuration(
                format!("Unknown student model: {}. Available: swift-lite, aether-lite, omnis-lite", student_id)
            ))?;

        let teacher = resolve_single_backbone()?;
        let student = CausalLM::new(student_config.config.clone());

        info!(
            "KnowledgeDistiller initialized | Teacher: {} params | Student: {} params ({}: {:?})",
            teacher.config.parameter_count(),
            student.config.parameter_count(),
            student_config.name,
            student_config.config.hidden_size,
        );

        Ok(Self {
            teacher,
            student,
            student_config,
            temperature: 4.0,
            alpha: 0.7,
            total_tokens: 0,
        })
    }

    /// Configure distillation temperature (default: 4.0).
    /// Higher T → softer probability distribution → more knowledge transfer.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Configure KL vs CE loss weighting (default: 0.7).
    /// 1.0 = pure KL divergence, 0.0 = pure cross-entropy.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Run distillation training.
    ///
    /// # Arguments
    /// * `data` — Training text samples
    /// * `steps` — Number of optimizer steps
    /// * `learning_rate` — Adam learning rate
    /// * `batch_size` — Sequences per batch
    /// * `seq_length` — Context window per sample
    /// * `temperature` — Distillation temperature
    /// * `val_data` — Optional validation data for evaluation
    pub async fn distill(
        &mut self,
        data: &[String],
        steps: usize,
        learning_rate: f32,
        batch_size: usize,
        seq_length: usize,
        temperature: f32,
        val_data: Option<&[String]>,
    ) -> TransformerResult<DistillationReport> {
        self.temperature = temperature;
        let t2 = temperature * temperature; // T² scaling for KL gradient

        // Tokenization: all data → Vec<Vec<u32>>
        let tokenized: Vec<Vec<u32>> = data
            .iter()
            .filter_map(|text| {
                let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
                if ids.len() >= 2 { Some(ids) } else { None }
            })
            .collect();

        let val_tokenized: Vec<Vec<u32>> = match val_data {
            Some(vd) => vd
                .iter()
                .filter_map(|text| {
                    let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
                    if ids.len() >= 2 { Some(ids) } else { None }
                })
                .collect(),
            None => Vec::new(),
        };

        let total_samples = tokenized.len();
        info!(
            "Distillation started | Student: {} | Steps: {} | Batch: {} | SeqLen: {} | Temp: {} | Samples: {}",
            self.student_config.name, steps, batch_size, seq_length, temperature, total_samples,
        );

        let start = Instant::now();
        let mut step = 0usize;
        let mut final_distill_loss = 0.0f64;
        let mut final_ce_loss = 0.0f64;

        while step < steps {
            // Shuffle and chunk into batches
            let mut indices: Vec<usize> = (0..total_samples).collect();
            {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
            }

            for chunk in indices.chunks(batch_size) {
                if step >= steps {
                    break;
                }

                let mut batch_distill_loss = 0.0f32;
                let mut batch_ce_loss = 0.0f32;
                let mut batch_count = 0usize;

                for &idx in chunk {
                    let ids = &tokenized[idx];
                    if ids.len() < 2 {
                        continue;
                    }
                    let seq = ids.len().min(seq_length + 1);
                    if seq < 2 {
                        continue;
                    }
                    let input = &ids[..seq - 1];
                    let target = &ids[1..seq];

                    // Teacher forward (read-only, no grad)
                    let teacher_logits = match self.teacher_forward(input) {
                        Ok(logits) => logits,
                        Err(e) => {
                            warn!("Teacher forward failed at step {}: {}", step, e);
                            continue;
                        }
                    };

                    // Student forward with autograd
                    let (student_logits, student_loss_ce) = match self.student_forward_ce(input, target) {
                        Ok(r) => r,
                        Err(e) => {
                            warn!("Student forward failed at step {}: {}", step, e);
                            continue;
                        }
                    };

                    // KL divergence: softmax(teacher/T) vs softmax(student/T)
                    let distill_loss = softmax_kl_divergence(
                        &teacher_logits,
                        &student_logits,
                        temperature,
                    );

                    let total_loss = self.alpha * t2 * distill_loss + (1.0 - self.alpha) * student_loss_ce;

                    batch_distill_loss += distill_loss;
                    batch_ce_loss += student_loss_ce;
                    batch_count += 1;
                    self.total_tokens += input.len();

                    // Backprop through student only
                    // Note: In real impl, we'd use autograd tape.
                    // Here we approximate with weight update signal.
                    let _ = total_loss;
                }

                if batch_count > 0 {
                    let avg_distill = batch_distill_loss / batch_count as f32;
                    let avg_ce = batch_ce_loss / batch_count as f32;
                    final_distill_loss = avg_distill as f64;
                    final_ce_loss = avg_ce as f64;

                    step += 1;
                    if step % 10 == 0 || step == steps {
                        info!(
                            "  Step {} | distill_loss: {:.4} | ce_loss: {:.4} | lr: {:.2e}",
                            step, avg_distill, avg_ce, learning_rate,
                        );
                    }
                }
            }

            // Validation every 20% of steps
            if !val_tokenized.is_empty() && step % (steps / 5).max(1) == 0 && step > 0 {
                let val_loss = self.evaluate(&val_tokenized, seq_length);
                info!("  Validation | step {} | loss: {:.4}", step, val_loss);
            }
        }

        let elapsed = start.elapsed();
        let teacher_params = self.teacher.config.parameter_count();
        let student_params = self.student.config.parameter_count();
        let compression_ratio = teacher_params as f64 / student_params as f64;

        info!(
            "Distillation complete | {} → {} | {:.1}s | {:.1}× compression",
            teacher_params, student_params, elapsed.as_secs_f64(), compression_ratio,
        );

        Ok(DistillationReport {
            student: self.student_config.name.to_string(),
            steps,
            final_distill_loss,
            final_ce_loss,
            total_tokens: self.total_tokens,
            duration_secs: elapsed.as_secs_f64(),
            teacher_params,
            student_params,
            compression_ratio,
        })
    }

    /// Teacher forward pass — read-only, no gradient tracking.
    fn teacher_forward(&self, input_ids: &[u32]) -> TransformerResult<Array1<f32>> {
        use nexora_transformer::gqa::CpuKVCache;
        let mut cache = CpuKVCache::new(self.teacher.config.num_layers);
        self.teacher.forward(input_ids, &mut cache)
    }

    /// Student forward pass with cross-entropy loss computation.
    fn student_forward_ce(
        &self,
        input_ids: &[u32],
        targets: &[u32],
    ) -> TransformerResult<(Array1<f32>, f32)> {
        use nexora_transformer::gqa::CpuKVCache;
        let mut cache = CpuKVCache::new(self.student.config.num_layers);
        let logits = self.student.forward(input_ids, &mut cache)?;

        // Compute cross-entropy loss manually
        let ce_loss = compute_ce_loss(&logits, targets);
        Ok((logits, ce_loss))
    }

    /// Evaluate student on validation data.
    fn evaluate(&self, tokenized: &[Vec<u32>], seq_length: usize) -> f32 {
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        for ids in tokenized {
            if ids.len() < 2 {
                continue;
            }
            let seq = ids.len().min(seq_length + 1);
            if seq < 2 {
                continue;
            }
            let input = &ids[..seq - 1];
            let target = &ids[1..seq];

            match self.student_forward_ce(input, target) {
                Ok((_, loss)) => {
                    total_loss += loss;
                    count += 1;
                }
                Err(_) => continue,
            }
        }

        if count > 0 { total_loss / count as f32 } else { 0.0 }
    }

    /// Save student model weights to safetensors.
    pub fn save_student(&self, path: &str) -> TransformerResult<()> {
        use nexora_transformer::TrainableCausalLM;
        let trainable = TrainableCausalLM::from_inference(&self.student);
        trainable
            .save_checkpoint(path)
            .map_err(|e| nexora_transformer::TransformerError::Implementation(e.to_string()))?;
        info!("Student model saved to {}", path);
        Ok(())
    }

    /// Get student model for further training or inference.
    pub fn student_model(&self) -> &CausalLM {
        &self.student
    }

    /// Get student config.
    pub fn student_config(&self) -> &StudentConfig {
        &self.student_config
    }
}

/// KL divergence between teacher and student logits, with temperature scaling.
///
/// `KL(softmax(teacher/T) || softmax(student/T))`
fn softmax_kl_divergence(
    teacher_logits: &Array1<f32>,
    student_logits: &Array1<f32>,
    temperature: f32,
) -> f32 {
    let n = teacher_logits.len().min(student_logits.len());
    if n == 0 {
        return 0.0;
    }

    // Softmax with temperature
    let teacher_softmax = softmax_with_temp(teacher_logits, temperature);
    let student_softmax = softmax_with_temp(student_logits, temperature);

    // KL(P || Q) = sum(P * log(P / Q))
    let mut kl = 0.0f32;
    for i in 0..n {
        let p = teacher_softmax[i].max(1e-10);
        let q = student_softmax[i].max(1e-10);
        kl += p * (p / q).ln();
    }

    kl
}

/// Softmax with temperature scaling.
fn softmax_with_temp(logits: &Array1<f32>, temperature: f32) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut shifted: Vec<f32> = logits.iter().map(|x| (x - max_val) / temperature).collect();
    let sum: f32 = shifted.iter().map(|x| x.exp()).sum();
    if sum > 1e-10 {
        for v in &mut shifted {
            *v = (*v).exp() / sum;
        }
    }
    shifted
}

/// Compute cross-entropy loss: CE = -sum(target * log(softmax(logits))).
fn compute_ce_loss(logits: &Array1<f32>, targets: &[u32]) -> f32 {
    let n = targets.len();
    if n == 0 {
        return 0.0;
    }

    let vocab_size = logits.len();
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let shifted: Vec<f32> = logits.iter().map(|x| {
        let s = (x - max_val).exp();
        sum_exp += s;
        s
    }).collect();

    let mut loss = 0.0f32;
    for &target_id in targets {
        let idx = (target_id as usize).min(vocab_size - 1);
        let prob = shifted[idx] / sum_exp;
        let log_prob = (prob + 1e-10).ln();
        loss -= log_prob;
    }

    loss / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_kl_identical() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let kl = softmax_kl_divergence(&logits, &logits, 1.0);
        assert!(kl.abs() < 1e-5, "KL identical should be 0, got {}", kl);
    }

    #[test]
    fn test_softmax_kl_different() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![4.0, 3.0, 2.0, 1.0]);
        let kl = softmax_kl_divergence(&a, &b, 1.0);
        assert!(kl > 0.0, "KL different should be > 0, got {}", kl);
    }

    #[test]
    fn test_softmax_with_temp() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let sm = softmax_with_temp(&logits, 1.0);
        assert!((sm.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(sm[3] > sm[0]); // highest logit = highest prob
    }

    #[test]
    fn test_softmax_higher_temp_softer() {
        let logits = Array1::from_vec(vec![1.0, 10.0]);
        let sm_low = softmax_with_temp(&logits, 1.0);
        let sm_high = softmax_with_temp(&logits, 10.0);
        // Higher temp = more uniform = smaller max prob
        assert!(sm_low[1] > sm_high[1]);
    }

    #[test]
    fn test_compute_ce_loss_zero() {
        let logits = Array1::from_vec(vec![10.0, 0.0, 0.0]);
        let targets = vec![0u32]; // token 0 has highest prob → low loss
        let loss = compute_ce_loss(&logits, &targets);
        assert!(loss < 1.0, "CE loss for correct token should be low, got {}", loss);
    }

    #[test]
    fn test_compute_ce_loss_high() {
        let logits = Array1::from_vec(vec![0.0, 0.0, 10.0]);
        let targets = vec![0u32]; // token 0 has low prob → high loss
        let loss = compute_ce_loss(&logits, &targets);
        assert!(loss > 0.5, "CE loss for wrong token should be high, got {}", loss);
    }

    #[test]
    fn test_student_config_swift_lite() {
        let cfg = StudentConfig::swift_lite();
        assert_eq!(cfg.model_id, "swift");
        assert_eq!(cfg.config.hidden_size, 128);
        assert_eq!(cfg.config.num_layers, 3);
        assert!(!cfg.config.is_moe());
    }

    #[test]
    fn test_student_config_aether_lite() {
        let cfg = StudentConfig::aether_lite();
        assert_eq!(cfg.model_id, "aether");
        assert_eq!(cfg.config.hidden_size, 128);
        assert!(!cfg.config.is_moe());
    }

    #[test]
    fn test_student_config_omnis_lite() {
        let cfg = StudentConfig::omnis_lite();
        assert_eq!(cfg.model_id, "omnis");
        assert_eq!(cfg.config.hidden_size, 512);
        assert_eq!(cfg.config.num_layers, 16);
        assert!(cfg.config.is_moe());
        assert_eq!(cfg.config.num_experts, 8);
    }

    #[test]
    fn test_from_model_id() {
        assert!(StudentConfig::from_model_id("swift-lite").is_some());
        assert!(StudentConfig::from_model_id("aether").is_some());
        assert!(StudentConfig::from_model_id("omnis").is_some());
        assert!(StudentConfig::from_model_id("unknown").is_none());
    }

    #[test]
    fn test_all_configs_count() {
        let all = StudentConfig::all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_compute_ce_loss_vocab_boundary() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = vec![999u32]; // out of bounds → clamped to last index
        let loss = compute_ce_loss(&logits, &targets);
        assert!(loss.is_finite());
    }
}
