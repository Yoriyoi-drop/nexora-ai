#![cfg_attr(not(feature = "gpu"), allow(unused))]
use std::collections::HashMap;

use tracing::info;

use crate::hallucination::{HallucinationError, PostGenCheckResult};

const TF_VOCAB_SIZE: usize = 2048;

struct TfVectorizer {
    word_to_idx: HashMap<String, usize>,
}

impl TfVectorizer {
    fn new(texts: &[String]) -> Self {
        let mut word_to_idx = HashMap::with_capacity(TF_VOCAB_SIZE);
        for text in texts {
            for word in text.split_whitespace() {
                let clean: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                if clean.is_empty() {
                    continue;
                }
                let len = word_to_idx.len();
                word_to_idx.entry(clean).or_insert(len);
                if word_to_idx.len() >= TF_VOCAB_SIZE {
                    break;
                }
            }
        }
        Self { word_to_idx }
    }

    fn vectorize(&self, texts: &[String]) -> ndarray::Array2<f32> {
        let vocab = self.word_to_idx.len();
        let mut matrix = ndarray::Array2::<f32>::zeros((texts.len(), vocab));
        for (row, text) in texts.iter().enumerate() {
            for word in text.split_whitespace() {
                let clean: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                if let Some(&idx) = self.word_to_idx.get(&clean) {
                    matrix[[row, idx]] += 1.0;
                }
            }
        }
        for mut row in matrix.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-8 {
                row /= norm;
            }
        }
        matrix
    }
}

pub fn gpu_batch_verify(
    sentences: &[String],
    sources: &[String],
) -> Result<PostGenCheckResult, HallucinationError> {
    let ctx = match nexora_deeplearning::autograd::gpu::GpuContext::global() {
        Ok(c) => c,
        Err(_) => {
            info!("GPU not available for batch verification");
            return Err(HallucinationError::Internal(
                "GPU not initialized".into(),
            ));
        }
    };

    let vectorizer = TfVectorizer::new(&{
        let mut all = sentences.to_vec();
        all.extend_from_slice(sources);
        all
    });

    let sent_features = vectorizer.vectorize(sentences);
    let src_features = vectorizer.vectorize(sources);

    if sent_features.nrows() == 0 || src_features.nrows() == 0 {
        return Err(HallucinationError::Internal(
            "Empty sentence or source features".into(),
        ));
    }

    let n_sent = sent_features.nrows();
    let n_src = src_features.nrows();
    let gpu_sent = nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&sent_features.into_dyn())
        .map_err(|e| HallucinationError::Internal(e.to_string()))?;
    let gpu_src = nexora_deeplearning::autograd::gpu::GpuTensor::from_cpu(&src_features.into_dyn())
        .map_err(|e| HallucinationError::Internal(e.to_string()))?;

    let gpu_src_t = ctx
        .transpose(&gpu_src)
        .map_err(|e| HallucinationError::Internal(e.to_string()))?;
    let sim = ctx
        .matmul(&gpu_sent, &gpu_src_t)
        .map_err(|e| HallucinationError::Internal(e.to_string()))?;

    let sim_cpu = sim
        .to_cpu()
        .map_err(|e| HallucinationError::Internal(e.to_string()))?;

    let mut high_risk_sentences = Vec::new();
    let mut verified_count = 0;

    for i in 0..n_sent {
        let mut max_sim = 0.0_f32;
        for j in 0..n_src {
            let val = sim_cpu[[i, j]];
            if val > max_sim {
                max_sim = val;
            }
        }
        if max_sim > 0.3 {
            verified_count += 1;
        } else if sentences[i].len() > 20 {
            high_risk_sentences.push(sentences[i].clone());
        }
    }

    let total_claims = n_sent;
    let source_grounding = if total_claims > 0 {
        verified_count as f32 / total_claims as f32
    } else {
        1.0
    };

    Ok(PostGenCheckResult {
        internal_consistency: 1.0,
        source_grounding,
        high_risk_sentences,
        contradiction_count: 0,
        total_claims,
        verified_claims: verified_count,
    })
}
