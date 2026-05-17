//! Token sequence processor for CAFFEINE
//! 
//! Handles interleaved multimodal sequences and special tokens

use crate::caffeine::types::*;
use crate::caffeine::error::Result;
use std::collections::HashMap;

/// Token sequence processor
pub struct TokenSequenceProcessor {
    vocab_size: usize,
    max_sequence_length: usize,
    special_token_ids: HashMap<String, usize>,
}

impl TokenSequenceProcessor {
    /// Create new token sequence processor
    pub fn new(vocab_size: usize, max_sequence_length: usize) -> Result<Self> {
        let mut special_token_ids = HashMap::with_capacity(7);
        
        // Initialize special token IDs
        special_token_ids.insert("<PAD>".to_string(), 0);
        special_token_ids.insert("<UNK>".to_string(), 1);
        special_token_ids.insert("<CLS>".to_string(), 2);
        special_token_ids.insert("<SEP>".to_string(), 3);
        special_token_ids.insert("<MASK>".to_string(), 4);
        special_token_ids.insert("<START>".to_string(), 5);
        special_token_ids.insert("<END>".to_string(), 6);
        
        Ok(Self {
            vocab_size,
            max_sequence_length,
            special_token_ids,
        })
    }
    
    /// Process token sequence with special tokens and interleaving
    pub fn process_sequence(&self, tokens: Vec<UnifiedToken>) -> Result<Vec<UnifiedToken>> {
        if tokens.is_empty() {
            return Ok(tokens);
        }
        
        // Add special tokens
        let mut processed = self.add_special_tokens(tokens)?;
        
        // Handle interleaving between modalities
        processed = self.handle_interleaving(processed)?;
        
        // Truncate if too long
        if processed.len() > self.max_sequence_length {
            processed.truncate(self.max_sequence_length - 1);
            // Add end token
            if let Some(end_token_id) = self.special_token_ids.get("<END>") {
                let end_token = UnifiedToken {
                    token_id: *end_token_id,
                    modality: ModalityType::Text,
                    embedding: vec![0.0; 768],
                    position: processed.len(),
                    timestamp: None,
                    spatial_coords: None,
                };
                processed.push(end_token);
            }
        }
        
        // Update positions
        self.update_positions(&mut processed);
        
        Ok(processed)
    }
    
    /// Add special tokens to sequence
    fn add_special_tokens(&self, mut tokens: Vec<UnifiedToken>) -> Result<Vec<UnifiedToken>> {
        // Add CLS token at beginning
        if let Some(cls_token_id) = self.special_token_ids.get("<CLS>") {
            let cls_token = UnifiedToken {
                token_id: *cls_token_id,
                modality: ModalityType::Text,
                embedding: vec![0.0; 768],
                position: 0,
                timestamp: None,
                spatial_coords: None,
            };
            tokens.insert(0, cls_token);
        }
        
        // Add SEP tokens between modalities
        let mut with_sep = Vec::new();
        let mut current_modality = None;
        
        for token in tokens {
            let token_modality = token.modality;
            
            // Add SEP token if modality changes
            if let Some(ref current) = current_modality {
                if *current != token_modality {
                    if let Some(sep_token_id) = self.special_token_ids.get("<SEP>") {
                        let sep_token = UnifiedToken {
                            token_id: *sep_token_id,
                            modality: ModalityType::Text,
                            embedding: vec![0.0; 768],
                            position: with_sep.len(),
                            timestamp: None,
                            spatial_coords: None,
                        };
                        with_sep.push(sep_token);
                    }
                }
            }
            
            with_sep.push(token);
            current_modality = Some(token_modality);
        }
        
        // Add END token at end
        if let Some(end_token_id) = self.special_token_ids.get("<END>") {
            let end_token = UnifiedToken {
                token_id: *end_token_id,
                modality: ModalityType::Text,
                embedding: vec![0.0; 768],
                position: with_sep.len(),
                timestamp: None,
                spatial_coords: None,
            };
            with_sep.push(end_token);
        }
        
        Ok(with_sep)
    }
    
    /// Handle interleaving between different modalities
    fn handle_interleaving(&self, tokens: Vec<UnifiedToken>) -> Result<Vec<UnifiedToken>> {
        let mut interleaved = Vec::new();
        let mut modality_groups = HashMap::with_capacity(5);
        
        // Group tokens by modality
        for token in &tokens {
            modality_groups.entry(token.modality).or_insert_with(Vec::new).push(token.clone());
        }
        
        // Interleave modalities in a balanced way
        let modalities = vec![
            ModalityType::Text,
            ModalityType::Image,
            ModalityType::Video,
            ModalityType::Audio,
            ModalityType::Action,
        ];
        
        let mut max_group_size = 0;
        for modality in &modalities {
            if let Some(group) = modality_groups.get(modality) {
                max_group_size = max_group_size.max(group.len());
            }
        }
        
        // Interleave tokens
        for i in 0..max_group_size {
            for modality in &modalities {
                if let Some(group) = modality_groups.get(modality) {
                    if i < group.len() {
                        interleaved.push(group[i].clone());
                    }
                }
            }
        }
        
        Ok(interleaved)
    }
    
    /// Update token positions
    fn update_positions(&self, tokens: &mut [UnifiedToken]) {
        for (i, token) in tokens.iter_mut().enumerate() {
            token.position = i;
        }
    }
    
    /// Create attention mask for sequence
    pub fn create_attention_mask(&self, tokens: &[UnifiedToken]) -> Result<Vec<f32>> {
        let mut mask = vec![1.0f32; tokens.len()];
        
        // Mask out padding tokens
        if let Some(pad_token_id) = self.special_token_ids.get("<PAD>") {
            for (i, token) in tokens.iter().enumerate() {
                if token.token_id == *pad_token_id {
                    mask[i] = 0.0;
                }
            }
        }
        
        Ok(mask)
    }
    
    /// Create modality mask for sequence
    pub fn create_modality_mask(&self, tokens: &[UnifiedToken]) -> Result<HashMap<ModalityType, Vec<f32>>> {
        let mut modality_masks = HashMap::new();
        
        // Initialize masks for all modalities
        for modality in [
            ModalityType::Text,
            ModalityType::Image,
            ModalityType::Audio,
            ModalityType::Video,
            ModalityType::Action,
        ] {
            modality_masks.insert(modality, vec![0.0f32; tokens.len()]);
        }
        
        // Set mask values
        for (i, token) in tokens.iter().enumerate() {
            if let Some(mask) = modality_masks.get_mut(&token.modality) {
                mask[i] = 1.0f32;
            }
        }
        
        Ok(modality_masks)
    }
    
    /// Create position embeddings
    pub fn create_position_embeddings(&self, tokens: &[UnifiedToken], embed_dim: usize) -> Result<Vec<Vec<f32>>> {
        let mut position_embeddings = Vec::new();
        
        for token in tokens {
            let mut embedding = vec![0.0f32; embed_dim];
            
            for d in 0..embed_dim {
                if d % 2 == 0 {
                    embedding[d] = (token.position as f32 / 10000.0_f32.powf(d as f32 / embed_dim as f32)).sin();
                } else {
                    embedding[d] = (token.position as f32 / 10000.0_f32.powf(d as f32 / embed_dim as f32)).cos();
                }
            }
            
            position_embeddings.push(embedding);
        }
        
        Ok(position_embeddings)
    }
    
    /// Create modality embeddings
    pub fn create_modality_embeddings(&self, tokens: &[UnifiedToken], embed_dim: usize) -> Result<Vec<Vec<f32>>> {
        let mut modality_embeddings = Vec::new();
        
        // Create modality ID mapping
        let modality_ids = HashMap::from([
            (ModalityType::Text, 0),
            (ModalityType::Image, 1),
            (ModalityType::Audio, 2),
            (ModalityType::Video, 3),
            (ModalityType::Action, 4),
        ]);
        
        for token in tokens {
            let mut embedding = vec![0.0f32; embed_dim];
            
            if let Some(&modality_id) = modality_ids.get(&token.modality) {
                for d in 0..embed_dim {
                    embedding[d] = (modality_id as f32 * (d as f32 + 1.0) * 0.1).sin();
                }
            }
            
            modality_embeddings.push(embedding);
        }
        
        Ok(modality_embeddings)
    }
    
    /// Validate token sequence
    pub fn validate_sequence(&self, tokens: &[UnifiedToken]) -> Result<()> {
        if tokens.len() > self.max_sequence_length {
            return Err(crate::caffeine::error::CaffeineError::tokenizer(
                &format!("Sequence length {} exceeds maximum {}", tokens.len(), self.max_sequence_length)
            ));
        }
        
        for token in tokens {
            if token.token_id >= self.vocab_size {
                return Err(crate::caffeine::error::CaffeineError::tokenizer(
                    &format!("Token ID {} exceeds vocabulary size {}", token.token_id, self.vocab_size)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get sequence statistics
    pub fn get_sequence_stats(&self, tokens: &[UnifiedToken]) -> SequenceStats {
        let mut modality_counts = HashMap::new();
        let mut total_length = 0;
        
        for token in tokens {
            *modality_counts.entry(token.modality).or_insert(0) += 1;
            total_length += 1;
        }
        
        SequenceStats {
            total_length,
            modality_counts,
            has_special_tokens: tokens.iter().any(|t| self.is_special_token(t.token_id)),
            is_interleaved: self.is_interleaved_sequence(tokens),
        }
    }
    
    /// Check if token is special
    fn is_special_token(&self, token_id: usize) -> bool {
        self.special_token_ids.values().any(|&id| id == token_id)
    }
    
    /// Check if sequence is interleaved
    fn is_interleaved_sequence(&self, tokens: &[UnifiedToken]) -> bool {
        if tokens.len() < 3 {
            return false;
        }
        
        let mut modality_changes = 0;
        let mut last_modality = tokens[0].modality;
        
        for token in tokens.iter().skip(1) {
            if token.modality != last_modality && !self.is_special_token(token.token_id) {
                modality_changes += 1;
                last_modality = token.modality;
            }
        }
        
        modality_changes > 2
    }
    
    /// Get special token ID
    pub fn get_special_token_id(&self, token_name: &str) -> Option<usize> {
        self.special_token_ids.get(token_name).copied()
    }
}

/// Sequence statistics
#[derive(Debug, Clone)]
pub struct SequenceStats {
    pub total_length: usize,
    pub modality_counts: HashMap<ModalityType, usize>,
    pub has_special_tokens: bool,
    pub is_interleaved: bool,
}

/// Sequence builder for easy construction
pub struct SequenceBuilder {
    tokens: Vec<UnifiedToken>,
    processor: TokenSequenceProcessor,
}

impl SequenceBuilder {
    /// Create new sequence builder
    pub fn new(vocab_size: usize, max_sequence_length: usize) -> Result<Self> {
        let processor = TokenSequenceProcessor::new(vocab_size, max_sequence_length)?;
        
        Ok(Self {
            tokens: Vec::new(),
            processor,
        })
    }
    
    /// Add token to sequence
    pub fn add_token(mut self, token: UnifiedToken) -> Self {
        self.tokens.push(token);
        self
    }
    
    /// Add multiple tokens
    pub fn add_tokens(mut self, tokens: Vec<UnifiedToken>) -> Self {
        self.tokens.extend(tokens);
        self
    }
    
    /// Build final sequence
    pub fn build(self) -> Result<Vec<UnifiedToken>> {
        self.processor.process_sequence(self.tokens)
    }
}
