//! FIM + Contrastive Dual Loss untuk Pretraining
//! 
//! Implementasi pretraining objective yang menggabungkan Fill-in-the-Middle (FIM)
//! dengan ContraCode untuk pemahaman semantik kode yang lebih baik.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Konfigurasi Pretraining ORACLE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePretrainingConfig {
    /// FIM probability
    pub fim_probability: f32,
    /// FIM span ratio
    pub fim_span_ratio: f32,
    /// Contrastive loss weight
    pub contrastive_weight: f32,
    /// Temperature for contrastive learning
    pub contrastive_temperature: f32,
    /// Number of negative samples for contrastive
    pub contrastive_negatives: usize,
    /// Code vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for OraclePretrainingConfig {
    fn default() -> Self {
        Self {
            fim_probability: 0.5,
            fim_span_ratio: 0.15,
            contrastive_weight: 0.1,
            contrastive_temperature: 0.1,
            contrastive_negatives: 8,
            vocab_size: 50000,
            max_seq_len: 8192,
        }
    }
}

/// Fill-in-the-Middle (FIM) Processor
pub struct FimProcessor {
    config: OraclePretrainingConfig,
    fim_tokens: FimTokens,
}

impl FimProcessor {
    pub fn new(config: OraclePretrainingConfig) -> Self {
        Self {
            fim_tokens: FimTokens::new(),
            config,
        }
    }
    
    /// Apply FIM transformation to code sequence
    pub fn apply_fim(&self, tokens: &[i32]) -> Result<FimExample> {
        if rand::random::<f32>() > self.config.fim_probability {
            // Return as-is (no FIM)
            return Ok(FimExample {
                prefix: tokens.to_vec(),
                middle: Vec::new(),
                suffix: Vec::new(),
                original: tokens.to_vec(),
                fim_type: FimType::None,
            });
        }
        
        let seq_len = tokens.len();
        let span_size = (seq_len as f32 * self.config.fim_span_ratio) as usize;
        let span_start = rand::random::<usize>() % (seq_len - span_size + 1);
        let span_end = span_start + span_size;
        
        let prefix = tokens[..span_start].to_vec();
        let middle = tokens[span_start..span_end].to_vec();
        let suffix = tokens[span_end..].to_vec();
        
        // Choose FIM type randomly
        let fim_type = match rand::random::<u32>() % 3 {
            0 => FimType::PrefixMiddleSuffix,
            1 => FimType::SuffixPrefixMiddle,
            2 => FimType::MiddlePrefixSuffix,
            _ => FimType::PrefixMiddleSuffix,
        };
        
        Ok(FimExample {
            prefix,
            middle,
            suffix,
            original: tokens.to_vec(),
            fim_type,
        })
    }
    
    /// Convert FIM example to model input
    pub fn to_model_input(&self, fim_example: &FimExample) -> Result<ModelInput> {
        match fim_example.fim_type {
            FimType::None => {
                Ok(ModelInput {
                    input_ids: fim_example.original.clone(),
                    attention_mask: vec![1; fim_example.original.len()],
                    labels: fim_example.original.clone(),
                })
            },
            FimType::PrefixMiddleSuffix => {
                let mut input_ids = fim_example.prefix.clone();
                input_ids.push(self.fim_tokens.prefix);
                input_ids.extend_from_slice(&fim_example.suffix);
                input_ids.push(self.fim_tokens.suffix);
                input_ids.extend_from_slice(&fim_example.middle);
                
                let mut labels = vec![-100; fim_example.prefix.len() + 1]; // Prefix + <PRE>
                labels.extend_from_slice(&fim_example.suffix); // Suffix
                labels.push(-100); // <SUF>
                labels.extend_from_slice(&fim_example.middle); // Middle
                
                Ok(ModelInput {
                    input_ids: input_ids.clone(),
                    attention_mask: vec![1; input_ids.len()],
                    labels,
                })
            },
            FimType::SuffixPrefixMiddle => {
                let mut input_ids = fim_example.suffix.clone();
                input_ids.push(self.fim_tokens.prefix);
                input_ids.extend_from_slice(&fim_example.prefix);
                input_ids.push(self.fim_tokens.suffix);
                input_ids.extend_from_slice(&fim_example.middle);
                
                let mut labels = vec![-100; fim_example.suffix.len() + 1]; // Suffix + <PRE>
                labels.extend_from_slice(&fim_example.prefix); // Prefix
                labels.push(-100); // <SUF>
                labels.extend_from_slice(&fim_example.middle); // Middle
                
                Ok(ModelInput {
                    input_ids: input_ids.clone(),
                    attention_mask: vec![1; input_ids.len()],
                    labels,
                })
            },
            FimType::MiddlePrefixSuffix => {
                let mut input_ids = fim_example.middle.clone();
                input_ids.push(self.fim_tokens.prefix);
                input_ids.extend_from_slice(&fim_example.prefix);
                input_ids.push(self.fim_tokens.suffix);
                input_ids.extend_from_slice(&fim_example.suffix);
                
                let mut labels = fim_example.middle.clone(); // Middle (no mask)
                labels.push(-100); // <PRE>
                labels.extend_from_slice(&fim_example.prefix); // Prefix
                labels.push(-100); // <SUF>
                labels.extend_from_slice(&fim_example.suffix); // Suffix
                
                Ok(ModelInput {
                    input_ids: input_ids.clone(),
                    attention_mask: vec![1; input_ids.len()],
                    labels,
                })
            },
        }
    }
}

/// FIM special tokens
#[derive(Debug, Clone)]
pub struct FimTokens {
    pub prefix: i32,
    pub suffix: i32,
}

impl FimTokens {
    pub fn new() -> Self {
        Self {
            prefix: 50001, // <PRE>
            suffix: 50002, // <SUF>
        }
    }
}

/// FIM example
#[derive(Debug, Clone)]
pub struct FimExample {
    pub prefix: Vec<i32>,
    pub middle: Vec<i32>,
    pub suffix: Vec<i32>,
    pub original: Vec<i32>,
    pub fim_type: FimType,
}

/// FIM transformation types
#[derive(Debug, Clone, PartialEq)]
pub enum FimType {
    None,
    PrefixMiddleSuffix,
    SuffixPrefixMiddle,
    MiddlePrefixSuffix,
}

/// Model input for training
#[derive(Debug, Clone)]
pub struct ModelInput {
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<i32>,
    pub labels: Vec<i32>,
}

/// Contrastive Code Learning
pub struct ContrastiveCodeLearning {
    config: OraclePretrainingConfig,
    code_encoder: CodeEncoder,
}

impl ContrastiveCodeLearning {
    pub fn new(config: OraclePretrainingConfig) -> Self {
        Self {
            code_encoder: CodeEncoder::new(config.vocab_size, 512),
            config,
        }
    }
    
    /// Create contrastive pairs from code snippets
    pub fn create_contrastive_pairs(&self, code_snippets: &[CodeSnippet]) -> Result<Vec<ContrastivePair>> {
        let mut pairs = Vec::new();
        
        for i in 0..code_snippets.len() {
            let anchor = &code_snippets[i];
            
            // Find semantically similar snippets
            let positives = self.find_similar_snippets(anchor, code_snippets, 3)?;
            
            // Find dissimilar snippets as negatives
            let negatives = self.find_dissimilar_snippets(anchor, code_snippets, self.config.contrastive_negatives)?;
            
            for positive in positives {
                pairs.push(ContrastivePair {
                    anchor: anchor.clone(),
                    positive,
                    negatives: negatives.clone(),
                });
            }
        }
        
        Ok(pairs)
    }
    
    /// Compute contrastive loss
    pub fn compute_contrastive_loss(&self, pairs: &[ContrastivePair]) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut num_pairs = 0;
        
        for pair in pairs {
            // Encode anchor and positive
            let anchor_embedding = self.code_encoder.encode(&pair.anchor.tokens)?;
            let positive_embedding = self.code_encoder.encode(&pair.positive.tokens)?;
            
            // Compute similarity
            let anchor_pos_sim = self.cosine_similarity(&anchor_embedding, &positive_embedding);
            
            // Compute similarities with negatives
            let mut neg_sims = Vec::new();
            for negative in &pair.negatives {
                let neg_embedding = self.code_encoder.encode(&negative.tokens)?;
                let sim = self.cosine_similarity(&anchor_embedding, &neg_embedding);
                neg_sims.push(sim);
            }
            
            // Compute InfoNCE loss
            let numerator = (anchor_pos_sim / self.config.contrastive_temperature).exp();
            let denominator = numerator + neg_sims.iter()
                .map(|&sim| (sim / self.config.contrastive_temperature).exp())
                .sum::<f32>();
            
            let loss = - (numerator / denominator).ln();
            total_loss += loss;
            num_pairs += 1;
        }
        
        Ok(if num_pairs > 0 { total_loss / num_pairs as f32 } else { 0.0 })
    }
    
    /// Find semantically similar snippets
    fn find_similar_snippets(&self, anchor: &CodeSnippet, snippets: &[CodeSnippet], k: usize) -> Result<Vec<CodeSnippet>> {
        let anchor_embedding = self.code_encoder.encode(&anchor.tokens)?;
        
        let mut similarities = Vec::new();
        for snippet in snippets {
            if snippet.id == anchor.id {
                continue; // Skip self
            }
            
            let snippet_embedding = self.code_encoder.encode(&snippet.tokens)?;
            let similarity = self.cosine_similarity(&anchor_embedding, &snippet_embedding);
            similarities.push((snippet.clone(), similarity));
        }
        
        // Sort by similarity and take top-k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(similarities.into_iter().take(k).map(|(snippet, _)| snippet).collect())
    }
    
    /// Find dissimilar snippets
    fn find_dissimilar_snippets(&self, anchor: &CodeSnippet, snippets: &[CodeSnippet], k: usize) -> Result<Vec<CodeSnippet>> {
        let anchor_embedding = self.code_encoder.encode(&anchor.tokens)?;
        
        let mut similarities = Vec::new();
        for snippet in snippets {
            if snippet.id == anchor.id {
                continue; // Skip self
            }
            
            let snippet_embedding = self.code_encoder.encode(&snippet.tokens)?;
            let similarity = self.cosine_similarity(&anchor_embedding, &snippet_embedding);
            similarities.push((snippet.clone(), similarity));
        }
        
        // Sort by similarity and take bottom-k (most dissimilar)
        similarities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(similarities.into_iter().take(k).map(|(snippet, _)| snippet).collect())
    }
    
    /// Compute cosine similarity
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Code snippet for contrastive learning
#[derive(Debug, Clone)]
pub struct CodeSnippet {
    pub id: Uuid,
    pub tokens: Vec<i32>,
    pub language: String,
    pub function_name: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl CodeSnippet {
    pub fn new(tokens: Vec<i32>, language: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            tokens,
            language,
            function_name: None,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_function_name(mut self, name: String) -> Self {
        self.function_name = Some(name);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Contrastive pair for training
#[derive(Debug, Clone)]
pub struct ContrastivePair {
    pub anchor: CodeSnippet,
    pub positive: CodeSnippet,
    pub negatives: Vec<CodeSnippet>,
}

/// Code encoder for contrastive learning
pub struct CodeEncoder {
    embedding: Array2<f32>,
    projection: Array2<f32>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl CodeEncoder {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            embedding: Array2::from_shape_fn((vocab_size, embedding_dim), |_| rand::random()),
            projection: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| rand::random()),
            vocab_size,
            embedding_dim,
        }
    }
    
    /// Encode code tokens to embedding
    pub fn encode(&self, tokens: &[i32]) -> Result<Array1<f32>> {
        let mut embedding_sum = Array1::zeros(self.embedding_dim);
        
        for &token_id in tokens {
            if token_id >= 0 && token_id < self.vocab_size as i32 {
                let token_embedding = self.embedding.slice(s![token_id as usize, ..]);
                embedding_sum += &token_embedding;
            }
        }
        
        // Average pooling
        if !tokens.is_empty() {
            embedding_sum /= tokens.len() as f32;
        }
        
        // Projection
        let projected = embedding_sum.dot(&self.projection);
        Ok(projected)
    }
    
    /// Batch encode multiple snippets
    pub fn encode_batch(&self, token_sequences: &[Vec<i32>]) -> Result<Array2<f32>> {
        let mut embeddings = Vec::new();
        
        for tokens in token_sequences {
            let embedding = self.encode(tokens)?;
            embeddings.push(embedding);
        }
        
        // Convert to 2D array
        let batch_size = embeddings.len();
        let embedding_dim = if embeddings.is_empty() { 0 } else { embeddings[0].len() };
        
        let mut result = Array2::zeros((batch_size, embedding_dim));
        for (i, embedding) in embeddings.into_iter().enumerate() {
            for (j, &val) in embedding.iter().enumerate() {
                result[[i, j]] = val;
            }
        }
        
        Ok(result)
    }
}

/// Dual Loss Calculator
pub struct DualLossCalculator {
    config: OraclePretrainingConfig,
    fim_processor: FimProcessor,
    contrastive_learner: ContrastiveCodeLearning,
}

impl DualLossCalculator {
    pub fn new(config: OraclePretrainingConfig) -> Self {
        Self {
            fim_processor: FimProcessor::new(config.clone()),
            contrastive_learner: ContrastiveCodeLearning::new(config.clone()),
            config,
        }
    }
    
    /// Compute dual loss (FIM + Contrastive)
    pub fn compute_dual_loss(&self, batch: &TrainingBatch) -> Result<DualLoss> {
        // Compute FIM loss using uniform distribution fallback
        let fim_loss = self.compute_fim_loss_uniform(batch)?;
        
        // Compute contrastive loss
        let contrastive_loss = self.compute_contrastive_loss_for_batch(batch)?;
        
        // Combine losses
        let total_loss = fim_loss + self.config.contrastive_weight * contrastive_loss;
        
        Ok(DualLoss {
            fim_loss,
            contrastive_loss,
            total_loss,
        })
    }
    
    /// Compute dual loss with actual model logits for real cross-entropy
    pub fn compute_dual_loss_with_logits(
        &self,
        batch: &TrainingBatch,
        logits: &Array3<f32>,
    ) -> Result<DualLoss> {
        // Compute FIM loss with real logits
        let fim_loss = self.compute_fim_loss(batch, logits)?;
        
        // Compute contrastive loss
        let contrastive_loss = self.compute_contrastive_loss_for_batch(batch)?;
        
        // Combine losses
        let total_loss = fim_loss + self.config.contrastive_weight * contrastive_loss;
        
        Ok(DualLoss {
            fim_loss,
            contrastive_loss,
            total_loss,
        })
    }
    
    /// Compute FIM loss for batch (uniform fallback)
    fn compute_fim_loss_uniform(&self, batch: &TrainingBatch) -> Result<f32> {
        let mut total_fim_loss = 0.0;
        let mut num_examples = 0;
        
        for example in &batch.examples {
            let fim_example = self.fim_processor.apply_fim(&example.tokens)?;
            let model_input = self.fim_processor.to_model_input(&fim_example)?;
            let loss = self.compute_cross_entropy_loss_uniform(&model_input)?;
            total_fim_loss += loss;
            num_examples += 1;
        }
        
        Ok(if num_examples > 0 { total_fim_loss / num_examples as f32 } else { 0.0 })
    }
    
    /// Compute FIM loss with actual logits for real cross-entropy
    fn compute_fim_loss(&self, batch: &TrainingBatch, logits: &Array3<f32>) -> Result<f32> {
        let mut total_fim_loss = 0.0;
        let mut num_examples = 0;
        let (batch_size, seq_len, vocab_size) = logits.dim();
        
        for (idx, example) in batch.examples.iter().enumerate() {
            if idx >= batch_size { break; }
            let fim_example = self.fim_processor.apply_fim(&example.tokens)?;
            let model_input = self.fim_processor.to_model_input(&fim_example)?;
            let logits_slice = logits.slice(s![idx, .., ..]).to_owned();
            let loss = compute_real_cross_entropy(&logits_slice, &model_input)?;
            total_fim_loss += loss;
            num_examples += 1;
        }
        
        Ok(if num_examples > 0 { total_fim_loss / num_examples as f32 } else { 0.0 })
    }
    
    /// Compute contrastive loss for batch
    fn compute_contrastive_loss_for_batch(&self, batch: &TrainingBatch) -> Result<f32> {
        // Convert batch examples to code snippets
        let code_snippets: Vec<CodeSnippet> = batch.examples.iter()
            .map(|example| CodeSnippet::new(example.tokens.clone(), example.language.clone()))
            .collect();
        
        // Create contrastive pairs
        let pairs = self.contrastive_learner.create_contrastive_pairs(&code_snippets)?;
        
        // Compute contrastive loss
        self.contrastive_learner.compute_contrastive_loss(&pairs)
    }
    
    /// Compute cross-entropy loss (uniform fallback)
    fn compute_cross_entropy_loss_uniform(&self, model_input: &ModelInput) -> Result<f32> {
        let mut loss = 0.0;
        let mut count = 0;
        
        for &label in &model_input.labels {
            if label >= 0 {
                loss += -(1.0 / self.config.vocab_size as f32 + 1e-8).ln();
                count += 1;
            }
        }
        
        Ok(if count > 0 { loss / count as f32 } else { 0.0 })
    }
}

/// Compute real cross-entropy loss from logits and labels
pub fn compute_real_cross_entropy(logits: &Array2<f32>, model_input: &ModelInput) -> Result<f32> {
    let (seq_len_logits, vocab_size) = logits.dim();
    let mut loss = 0.0;
    let mut count = 0;
    
    for (t, &label) in model_input.labels.iter().enumerate() {
        if label >= 0 && t < seq_len_logits && (label as usize) < vocab_size {
            let logit_row = logits.slice(s![t, ..]);
            let max_val = logit_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            for &v in logit_row.iter() {
                sum_exp += (v - max_val).exp();
            }
            let log_sum_exp = max_val + sum_exp.ln();
            let nll = log_sum_exp - logit_row[label as usize];
            loss += nll;
            count += 1;
        }
    }
    
    Ok(if count > 0 { loss / count as f32 } else { 0.0 })
}

/// Training batch for pretraining
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub examples: Vec<TrainingExample>,
}

/// Training example
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub tokens: Vec<i32>,
    pub language: String,
    pub metadata: HashMap<String, String>,
}

/// Dual loss components
#[derive(Debug, Clone)]
pub struct DualLoss {
    pub fim_loss: f32,
    pub contrastive_loss: f32,
    pub total_loss: f32,
}

/// Pretraining trainer
pub struct OraclePretrainer {
    config: OraclePretrainingConfig,
    dual_loss_calculator: DualLossCalculator,
    training_state: PretrainingState,
}

impl OraclePretrainer {
    pub fn new(config: OraclePretrainingConfig) -> Self {
        Self {
            dual_loss_calculator: DualLossCalculator::new(config.clone()),
            config,
            training_state: PretrainingState::new(),
        }
    }
    
    /// Training step (uniform fallback)
    pub fn training_step(&mut self, batch: &TrainingBatch) -> Result<DualLoss> {
        let loss = self.dual_loss_calculator.compute_dual_loss(batch)?;
        
        // Update training state
        self.training_state.step_count += 1;
        self.training_state.total_loss += loss.total_loss;
        self.training_state.fim_loss += loss.fim_loss;
        self.training_state.contrastive_loss += loss.contrastive_loss;
        
        Ok(loss)
    }
    
    /// Training step with actual model logits for real cross-entropy
    pub fn training_step_with_logits(
        &mut self,
        batch: &TrainingBatch,
        logits: &Array3<f32>,
    ) -> Result<DualLoss> {
        let loss = self.dual_loss_calculator.compute_dual_loss_with_logits(batch, logits)?;
        
        self.training_state.step_count += 1;
        self.training_state.total_loss += loss.total_loss;
        self.training_state.fim_loss += loss.fim_loss;
        self.training_state.contrastive_loss += loss.contrastive_loss;
        
        Ok(loss)
    }
    
    /// Get training statistics
    pub fn get_training_stats(&self) -> PretrainingStats {
        let step_count = self.training_state.step_count;
        
        PretrainingStats {
            step_count,
            avg_total_loss: self.training_state.total_loss / step_count.max(1) as f32,
            avg_fim_loss: self.training_state.fim_loss / step_count.max(1) as f32,
            avg_contrastive_loss: self.training_state.contrastive_loss / step_count.max(1) as f32,
            fim_probability: self.config.fim_probability,
            contrastive_weight: self.config.contrastive_weight,
        }
    }
    
    /// Reset training state
    pub fn reset(&mut self) {
        self.training_state = PretrainingState::new();
    }
}

/// Training state
#[derive(Debug, Clone)]
pub struct PretrainingState {
    pub step_count: usize,
    pub total_loss: f32,
    pub fim_loss: f32,
    pub contrastive_loss: f32,
}

impl PretrainingState {
    pub fn new() -> Self {
        Self {
            step_count: 0,
            total_loss: 0.0,
            fim_loss: 0.0,
            contrastive_loss: 0.0,
        }
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct PretrainingStats {
    pub step_count: usize,
    pub avg_total_loss: f32,
    pub avg_fim_loss: f32,
    pub avg_contrastive_loss: f32,
    pub fim_probability: f32,
    pub contrastive_weight: f32,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create training examples from code
    pub fn create_training_examples(code: &str, language: &str) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        
        // Simple tokenization (in practice, use proper tokenizer)
        let tokens: Vec<i32> = code.split_whitespace()
            .map(|word| word.chars().map(|c| c as i32).sum::<i32>())
            .collect();
        
        // Split into chunks
        let chunk_size = 512;
        for chunk in tokens.chunks(chunk_size) {
            examples.push(TrainingExample {
                tokens: chunk.to_vec(),
                language: language.to_string(),
                metadata: HashMap::new(),
            });
        }
        
        examples
    }
    
    /// Analyze code complexity
    pub fn analyze_code_complexity(code: &str) -> CodeComplexity {
        let lines = code.lines().count();
        let functions = code.matches("fn ").count() + code.matches("function ").count();
        let classes = code.matches("class ").count();
        let loops = code.matches("for ").count() + code.matches("while ").count();
        let conditionals = code.matches("if ").count() + code.matches("else ").count();
        
        let complexity_score = (functions * 2 + classes * 3 + loops * 2 + conditionals) as f32;
        
        CodeComplexity {
            lines,
            functions,
            classes,
            loops,
            conditionals,
            complexity_score,
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct CodeComplexity {
        pub lines: usize,
        pub functions: usize,
        pub classes: usize,
        pub loops: usize,
        pub conditionals: usize,
        pub complexity_score: f32,
    }
    
    /// Validate pretraining configuration
    pub fn validate_pretraining_config(config: &OraclePretrainingConfig) -> Result<()> {
        if config.fim_probability < 0.0 || config.fim_probability > 1.0 {
            return Err(anyhow::anyhow!("FIM probability must be between 0 and 1"));
        }
        
        if config.fim_span_ratio < 0.0 || config.fim_span_ratio > 1.0 {
            return Err(anyhow::anyhow!("FIM span ratio must be between 0 and 1"));
        }
        
        if config.contrastive_weight < 0.0 {
            return Err(anyhow::anyhow!("Contrastive weight must be non-negative"));
        }
        
        if config.contrastive_temperature <= 0.0 {
            return Err(anyhow::anyhow!("Contrastive temperature must be positive"));
        }
        
        if config.vocab_size == 0 {
            return Err(anyhow::anyhow!("Vocabulary size must be positive"));
        }
        
        if config.max_seq_len == 0 {
            return Err(anyhow::anyhow!("Max sequence length must be positive"));
        }
        
        Ok(())
    }
}
