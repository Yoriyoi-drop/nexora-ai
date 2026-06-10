pub mod multimodal_vocab;
pub mod token_sequence;
pub mod vq_vae;

pub use multimodal_vocab::*;
pub use token_sequence::*;
pub use vq_vae::*;

use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use ndarray::ArrayD;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct NgramModel {
    order: usize,
    counts: Vec<std::collections::HashMap<Vec<usize>, usize>>,
}

impl NgramModel {
    fn new(order: usize) -> Self {
        Self {
            order,
            counts: vec![std::collections::HashMap::new(); order],
        }
    }

    fn update(&mut self, sequence: &[usize]) {
        for n in 1..=self.order.min(sequence.len()) {
            for w in sequence.windows(n) {
                *self.counts[n - 1].entry(w.to_vec()).or_insert(0) += 1;
            }
        }
    }

    fn predict(&self, context: &[usize]) -> Option<usize> {
        for n in (1..=self.order.min(context.len())).rev() {
            let ctx = &context[context.len() - n..];
            let mut candidates: Vec<(usize, usize)> = Vec::new();
            for (ngram, &count) in &self.counts[n] {
                if ngram.len() == n + 1 && &ngram[..n] == ctx {
                    candidates.push((ngram[n], count));
                }
            }
            if !candidates.is_empty() {
                candidates.sort_by(|a, b| b.1.cmp(&a.1));
                return Some(candidates[0].0);
            }
        }
        None
    }
}

pub struct UnifiedTokenizer {
    vq_vae: VectorQuantizedVAE,
    vocabulary: MultimodalVocabulary,
    sequence_processor: TokenSequenceProcessor,
    config: crate::caffeine::config::TokenizerConfig,
    tokens_processed: usize,
    ngram: NgramModel,
    rng: StdRng,
}

impl UnifiedTokenizer {
    pub fn new(config: crate::caffeine::config::TokenizerConfig) -> Result<Self> {
        let vq_vae = VectorQuantizedVAE::new(
            config.token_dim,
            config.codebook_size,
            config.num_codebooks,
            config.commitment_weight,
        )?;
        let vocabulary = MultimodalVocabulary::new(config.vocab_size, config.token_dim)?;
        let sequence_processor = TokenSequenceProcessor::new(config.vocab_size, config.max_sequence_length)?;

        Ok(Self {
            vq_vae,
            vocabulary,
            sequence_processor,
            config,
            tokens_processed: 0,
            ngram: NgramModel::new(3),
            rng: StdRng::from_entropy(),
        })
    }

    pub fn tokenize(&mut self, features: &QueryFeatures) -> Result<Vec<UnifiedToken>> {
        let mut semantic_tokens =
            self.tokenize_query_features(&features.semantic_features, ModalityType::Text)?;
        let mut spatial_tokens =
            self.tokenize_query_features(&features.spatial_features, ModalityType::Image)?;
        let mut temporal_tokens =
            self.tokenize_query_features(&features.temporal_features, ModalityType::Video)?;

        let mut all_tokens = Vec::new();
        all_tokens.append(&mut semantic_tokens);
        all_tokens.append(&mut spatial_tokens);
        all_tokens.append(&mut temporal_tokens);

        let processed_tokens = self.sequence_processor.process_sequence(all_tokens)?;
        self.tokens_processed += processed_tokens.len();

        let ids: Vec<usize> = processed_tokens.iter().map(|t| t.token_id).collect();
        self.ngram.update(&ids);

        Ok(processed_tokens)
    }

    fn tokenize_query_features(
        &mut self,
        features: &ArrayD<f32>,
        modality: ModalityType,
    ) -> Result<Vec<UnifiedToken>> {
        let shape = features.shape();
        let num_queries = shape[1];
        let embed_dim = shape[2];

        let mut tokens = Vec::new();

        for i in 0..num_queries {
            let mut query_embedding = vec![0.0f32; embed_dim];
            for d in 0..embed_dim {
                if let Some(&val) = features.get([0, i, d]) {
                    query_embedding[d] = val;
                }
            }

            let (quantized, token_ids, _) = self.vq_vae.quantize(&query_embedding)?;

            tokens.push(UnifiedToken {
                token_id: token_ids[0],
                modality,
                embedding: quantized,
                position: i,
                timestamp: None,
                spatial_coords: None,
            });
        }

        Ok(tokens)
    }

    pub fn detokenize(&mut self, tokens: &[UnifiedToken]) -> Result<QueryFeatures> {
        let mut semantic_tokens = Vec::new();
        let mut spatial_tokens = Vec::new();
        let mut temporal_tokens = Vec::new();

        for token in tokens {
            match token.modality {
                ModalityType::Text => semantic_tokens.push(token),
                ModalityType::Image => spatial_tokens.push(token),
                ModalityType::Video => temporal_tokens.push(token),
                _ => {}
            }
        }

        let semantic_owned: Vec<UnifiedToken> = semantic_tokens.into_iter().cloned().collect();
        let semantic_features = self.tokens_to_features(&semantic_owned)?;
        let spatial_owned: Vec<UnifiedToken> = spatial_tokens.into_iter().cloned().collect();
        let spatial_features = self.tokens_to_features(&spatial_owned)?;
        let temporal_owned: Vec<UnifiedToken> = temporal_tokens.into_iter().cloned().collect();
        let temporal_features = self.tokens_to_features(&temporal_owned)?;

        Ok(QueryFeatures {
            semantic_features,
            spatial_features,
            temporal_features,
            attention_weights: None,
        })
    }

    fn tokens_to_features(&mut self, tokens: &[UnifiedToken]) -> Result<ArrayD<f32>> {
        if tokens.is_empty() {
            return Ok(ArrayD::from_shape_vec(vec![1, 0, self.config.token_dim], vec![])?);
        }

        let num_tokens = tokens.len();
        let token_dim = self.config.token_dim;
        let mut features = vec![0.0f32; num_tokens * token_dim];

        for (i, token) in tokens.iter().enumerate() {
            let dequantized = self.vq_vae.dequantize(&[token.token_id])?;
            for d in 0..token_dim {
                if d < dequantized.len() {
                    features[i * token_dim + d] = dequantized[d];
                }
            }
        }

        let shape = vec![1, num_tokens, token_dim];
        Ok(ArrayD::from_shape_vec(shape, features)?)
    }

    pub fn generate_autoregressive(
        &mut self,
        prompt_tokens: &[UnifiedToken],
        max_new_tokens: usize,
    ) -> Result<Vec<UnifiedToken>> {
        let mut generated_tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            let context_tokens = &generated_tokens[generated_tokens.len().saturating_sub(512)..];
            let next_token = self.predict_next_token(context_tokens)?;
            let should_switch = self.rng.gen::<f32>() < 0.1;
            let next_token = if should_switch {
                let new_modality = self.predict_next_modality(context_tokens)?;
                UnifiedToken {
                    modality: new_modality,
                    ..next_token
                }
            } else {
                next_token
            };

            generated_tokens.push(next_token.clone());

            if next_token.token_id == self.vocabulary.get_end_token_id() {
                break;
            }
        }

        let ids: Vec<usize> = generated_tokens.iter().map(|t| t.token_id).collect();
        self.ngram.update(&ids);

        Ok(generated_tokens)
    }

    fn predict_next_token(&mut self, context_tokens: &[UnifiedToken]) -> Result<UnifiedToken> {
        if context_tokens.is_empty() {
            return Ok(UnifiedToken {
                token_id: self.vocabulary.get_start_token_id(),
                modality: ModalityType::Text,
                embedding: vec![0.0; self.config.token_dim],
                position: 0,
                timestamp: None,
                spatial_coords: None,
            });
        }

        let last = context_tokens.last().ok_or_else(|| {
            crate::caffeine::error::CaffeineError::tokenizer("No context tokens available")
        })?;

        let context_ids: Vec<usize> = context_tokens.iter().map(|t| t.token_id).collect();
        let predicted = self.ngram.predict(&context_ids).unwrap_or_else(|| {
            let last_id = last.token_id;
            let mut hash = last_id as u64;
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((hash % self.config.vocab_size as u64) as usize).max(1)
        });

        let next_modality = self.predict_next_modality(context_tokens)?;

        let quantized = self.vq_vae.quantize(&vec![predicted as f32; self.config.token_dim]).ok();
        let embedding = quantized.map(|(q, _, _)| q).unwrap_or_else(|| vec![0.0; self.config.token_dim]);

        Ok(UnifiedToken {
            token_id: predicted,
            modality: next_modality,
            embedding,
            position: last.position + 1,
            timestamp: last.timestamp.map(|t| t + 0.1),
            spatial_coords: last.spatial_coords,
        })
    }

    fn predict_next_modality(&self, context_tokens: &[UnifiedToken]) -> Result<ModalityType> {
        if context_tokens.is_empty() {
            return Ok(ModalityType::Text);
        }

        let last = context_tokens.last().ok_or_else(|| {
            crate::caffeine::error::CaffeineError::tokenizer("No context tokens available")
        })?;

        let modality_counts: std::collections::HashMap<ModalityType, usize> =
            context_tokens.iter().fold(std::collections::HashMap::new(), |mut acc, t| {
                *acc.entry(t.modality).or_insert(0) += 1;
                acc
            });

        let dominant = modality_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(m, _)| m)
            .unwrap_or(ModalityType::Text);

        if dominant == last.modality {
            Ok(match last.modality {
                ModalityType::Text => ModalityType::Image,
                ModalityType::Image => ModalityType::Video,
                ModalityType::Video => ModalityType::Audio,
                ModalityType::Audio => ModalityType::Text,
                _ => ModalityType::Text,
            })
        } else {
            Ok(dominant)
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn token_dim(&self) -> usize {
        self.config.token_dim
    }

    pub fn get_stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.config.vocab_size,
            token_dim: self.config.token_dim,
            codebook_size: self.config.codebook_size,
            num_codebooks: self.config.num_codebooks,
            total_tokens_processed: self.tokens_processed,
            compression_ratio: self.vq_vae.get_compression_ratio(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerStats {
    pub vocab_size: usize,
    pub token_dim: usize,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub total_tokens_processed: usize,
    pub compression_ratio: f32,
}
