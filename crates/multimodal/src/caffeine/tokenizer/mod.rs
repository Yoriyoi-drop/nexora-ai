//! Unified multimodal tokenizer for CAFFEINE
//!
//! Implements Stage 3: Unified Discrete Multimodal Token Space (dari MIO)
//! - VQ-VAE based tokenization
//! - Any-to-any generation support
//! - Interleaved multimodal sequences

pub mod multimodal_vocab;
pub mod token_sequence;
pub mod vq_vae;

pub use multimodal_vocab::*;
pub use token_sequence::*;
pub use vq_vae::*;

use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use ndarray::ArrayD;

/// Unified tokenizer for all modalities
pub struct UnifiedTokenizer {
    vq_vae: VectorQuantizedVAE,
    vocabulary: MultimodalVocabulary,
    sequence_processor: TokenSequenceProcessor,
    config: crate::caffeine::config::TokenizerConfig,
    tokens_processed: usize,
}

impl UnifiedTokenizer {
    /// Create new unified tokenizer
    pub fn new(config: crate::caffeine::config::TokenizerConfig) -> Result<Self> {
        let vq_vae = VectorQuantizedVAE::new(
            config.token_dim,
            config.codebook_size,
            config.num_codebooks,
            config.commitment_weight,
        )?;

        let vocabulary = MultimodalVocabulary::new(config.vocab_size, config.token_dim)?;

        let sequence_processor =
            TokenSequenceProcessor::new(config.vocab_size, config.max_sequence_length)?;

        Ok(Self {
            vq_vae,
            vocabulary,
            sequence_processor,
            config,
            tokens_processed: 0,
        })
    }

    /// Tokenize query features into unified tokens
    pub fn tokenize(&mut self, features: &QueryFeatures) -> Result<Vec<UnifiedToken>> {
        // Tokenize different query types
        let mut semantic_tokens =
            self.tokenize_query_features(&features.semantic_features, ModalityType::Text)?;

        let mut spatial_tokens =
            self.tokenize_query_features(&features.spatial_features, ModalityType::Image)?;

        let mut temporal_tokens =
            self.tokenize_query_features(&features.temporal_features, ModalityType::Video)?;

        // Combine and process sequence
        let mut all_tokens = Vec::new();
        all_tokens.append(&mut semantic_tokens);
        all_tokens.append(&mut spatial_tokens);
        all_tokens.append(&mut temporal_tokens);

        // Process sequence (handle interleaving, special tokens, etc.)
        let processed_tokens = self.sequence_processor.process_sequence(all_tokens)?;

        self.tokens_processed += processed_tokens.len();

        Ok(processed_tokens)
    }

    /// Tokenize individual query features
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
            // Extract query embedding
            let mut query_embedding = vec![0.0f32; embed_dim];
            for d in 0..embed_dim {
                if let Some(&val) = features.get([0, i, d]) {
                    query_embedding[d] = val;
                }
            }

            // Quantize using VQ-VAE
            let (quantized, token_ids, _) = self.vq_vae.quantize(&query_embedding)?;

            // Create unified token
            let unified_token = UnifiedToken {
                token_id: token_ids[0], // Use first codebook
                modality,
                embedding: quantized,
                position: i,
                timestamp: None,
                spatial_coords: None,
            };

            tokens.push(unified_token);
        }

        Ok(tokens)
    }

    /// Detokenize unified tokens back to features
    pub fn detokenize(&mut self, tokens: &[UnifiedToken]) -> Result<QueryFeatures> {
        // Group tokens by modality
        let mut semantic_tokens = Vec::new();
        let mut spatial_tokens = Vec::new();
        let mut temporal_tokens = Vec::new();

        for token in tokens {
            match token.modality {
                ModalityType::Text => semantic_tokens.push(token),
                ModalityType::Image => spatial_tokens.push(token),
                ModalityType::Video => temporal_tokens.push(token),
                _ => {} // Handle other modalities as needed
            }
        }

        // Convert back to features
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

    /// Convert tokens back to feature arrays
    fn tokens_to_features(&mut self, tokens: &[UnifiedToken]) -> Result<ArrayD<f32>> {
        if tokens.is_empty() {
            return Ok(ArrayD::from_shape_vec(
                vec![1, 0, self.config.token_dim],
                vec![],
            )?);
        }

        let num_tokens = tokens.len();
        let token_dim = self.config.token_dim;
        let mut features = vec![0.0f32; num_tokens * token_dim];

        for (i, token) in tokens.iter().enumerate() {
            // Dequantize embedding
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

    /// Generate tokens autoregressively
    pub fn generate_autoregressive(
        &mut self,
        prompt_tokens: &[UnifiedToken],
        max_new_tokens: usize,
    ) -> Result<Vec<UnifiedToken>> {
        let mut generated_tokens = prompt_tokens.to_vec();

        for _ in 0..max_new_tokens {
            // Get current context
            let context_tokens = &generated_tokens[generated_tokens.len().saturating_sub(512)..];

            // Predict next token (simplified)
            let next_token = self.predict_next_token(context_tokens)?;

            generated_tokens.push(next_token.clone());

            // Check for end token
            if next_token.token_id == self.vocabulary.get_end_token_id() {
                break;
            }
        }

        Ok(generated_tokens)
    }

    /// Predict next token using the LLM backbone (not available in this context)
    /// Falls back to bigram/unigram frequency-based prediction rather than a counter.
    fn predict_next_token(&self, context_tokens: &[UnifiedToken]) -> Result<UnifiedToken> {
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

        let last = context_tokens.last().unwrap();
        let max_id = self.config.vocab_size.max(100);

        // Build bigram frequencies from context
        let mut bigram_counts: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut unigram_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for pair in context_tokens.windows(2) {
            let prev = pair[0].token_id;
            let next = pair[1].token_id;
            *bigram_counts.entry((prev, next)).or_insert(0) += 1;
            *unigram_counts.entry(next).or_insert(0) += 1;
        }
        if !context_tokens.is_empty() {
            *unigram_counts.entry(last.token_id).or_insert(0) += 1;
        }

        // Pick next token: prefer bigram match, fall back to unigram, then heuristic
        let next_id = bigram_counts
            .iter()
            .filter(|((prev, _), _)| *prev == last.token_id)
            .max_by_key(|(_, &count)| count)
            .map(|((_, next), _)| *next)
            .or_else(|| {
                unigram_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&id, _)| id)
            })
            .unwrap_or_else(|| {
                let mod_token = match last.modality {
                    ModalityType::Text => 100,
                    ModalityType::Image => 200,
                    ModalityType::Audio => 300,
                    ModalityType::Video => 400,
                    ModalityType::Action => 500,
                };
                (last.token_id + 1).max(mod_token + 1) % max_id
            });

        Ok(UnifiedToken {
            token_id: next_id,
            modality: last.modality,
            embedding: last.embedding.clone(),
            position: last.position + 1,
            timestamp: last.timestamp.map(|t| t + 0.1),
            spatial_coords: last.spatial_coords,
        })
    }

    /// Predict next modality type
    fn predict_next_modality(&self, context_tokens: &[UnifiedToken]) -> Result<ModalityType> {
        if context_tokens.is_empty() {
            return Ok(ModalityType::Text);
        }

        // Simple pattern: cycle through modalities
        let last_modality = context_tokens
            .last()
            .ok_or_else(|| {
                crate::caffeine::error::CaffeineError::tokenizer("No context tokens available")
            })?
            .modality;

        match last_modality {
            ModalityType::Text => Ok(ModalityType::Image),
            ModalityType::Image => Ok(ModalityType::Video),
            ModalityType::Video => Ok(ModalityType::Text),
            _ => Ok(ModalityType::Text),
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get token dimension
    pub fn token_dim(&self) -> usize {
        self.config.token_dim
    }

    /// Get tokenizer statistics
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

/// Tokenizer statistics
#[derive(Debug, Clone)]
pub struct TokenizerStats {
    pub vocab_size: usize,
    pub token_dim: usize,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub total_tokens_processed: usize,
    pub compression_ratio: f32,
}
