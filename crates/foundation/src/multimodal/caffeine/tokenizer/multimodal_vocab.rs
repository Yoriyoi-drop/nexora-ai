//! Multimodal vocabulary for CAFFEINE
//! 
//! Unified vocabulary for all modalities with special tokens

use crate::multimodal::caffeine::types::*;
use crate::multimodal::caffeine::error::Result;
use std::collections::HashMap;

/// Multimodal vocabulary
pub struct MultimodalVocabulary {
    vocab_size: usize,
    _token_dim: usize,
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    modality_tokens: HashMap<ModalityType, Vec<usize>>,
    special_tokens: HashMap<String, usize>,
}

impl MultimodalVocabulary {
    /// Create new multimodal vocabulary
    pub fn new(vocab_size: usize, _token_dim: usize) -> Result<Self> {
        let mut token_to_id = HashMap::with_capacity(12);
        let mut id_to_token = HashMap::with_capacity(12);
        let mut modality_tokens = HashMap::with_capacity(12);
        let mut special_tokens = HashMap::with_capacity(12);
        
        // Add special tokens
        let special_token_list = vec![
            ("<PAD>", 0),
            ("<UNK>", 1),
            ("<CLS>", 2),
            ("<SEP>", 3),
            ("<MASK>", 4),
            ("<START>", 5),
            ("<END>", 6),
            ("</TEXT>", 100),
            ("</IMAGE>", 200),
            ("</AUDIO>", 300),
            ("</VIDEO>", 400),
            ("</ACTION>", 500),
        ];
        
        for (token, id) in special_token_list {
            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
            special_tokens.insert(token.to_string(), id);
        }
        
        // Initialize modality token ranges
        modality_tokens.insert(ModalityType::Text, Vec::new());
        modality_tokens.insert(ModalityType::Image, Vec::new());
        modality_tokens.insert(ModalityType::Audio, Vec::new());
        modality_tokens.insert(ModalityType::Video, Vec::new());
        modality_tokens.insert(ModalityType::Action, Vec::new());
        
        // Add basic vocabulary for each modality
        Self::add_basic_vocabulary(&mut token_to_id, &mut id_to_token, &mut modality_tokens)?;
        
        Ok(Self {
            vocab_size,
            _token_dim,
            token_to_id,
            id_to_token,
            modality_tokens,
            special_tokens,
        })
    }
    
    /// Add basic vocabulary for each modality
    fn add_basic_vocabulary(
        token_to_id: &mut HashMap<String, usize>,
        id_to_token: &mut HashMap<usize, String>,
        modality_tokens: &mut HashMap<ModalityType, Vec<usize>>,
    ) -> Result<()> {
        // Text vocabulary (common words)
        let text_words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "under", "over",
            "hello", "world", "thanks", "please", "sorry", "yes", "no", "maybe",
        ];
        
        let mut text_ids = Vec::new();
        for (i, word) in text_words.iter().enumerate() {
            let id = 1000 + i; // Start text tokens from 1000
            token_to_id.insert(word.to_string(), id);
            id_to_token.insert(id, word.to_string());
            text_ids.push(id);
        }
        modality_tokens.insert(ModalityType::Text, text_ids);
        
        // Image vocabulary (visual concepts)
        let image_concepts = vec![
            "person", "car", "tree", "house", "sky", "cloud", "sun", "moon", "star",
            "mountain", "river", "ocean", "beach", "forest", "road", "building",
            "animal", "dog", "cat", "bird", "flower", "grass", "rock", "sand",
            "red", "blue", "green", "yellow", "black", "white", "gray", "brown",
        ];
        
        let mut image_ids = Vec::new();
        for (i, concept) in image_concepts.iter().enumerate() {
            let id = 2000 + i; // Start image tokens from 2000
            token_to_id.insert(concept.to_string(), id);
            id_to_token.insert(id, concept.to_string());
            image_ids.push(id);
        }
        modality_tokens.insert(ModalityType::Image, image_ids);
        
        // Audio vocabulary (sound concepts)
        let audio_concepts = vec![
            "music", "speech", "noise", "silence", "voice", "song", "instrument",
            "drum", "guitar", "piano", "violin", "flute", "trumpet", "saxophone",
            "loud", "quiet", "soft", "hard", "high", "low", "fast", "slow",
            "happy", "sad", "angry", "calm", "exciting", "boring", "scary",
        ];
        
        let mut audio_ids = Vec::new();
        for (i, concept) in audio_concepts.iter().enumerate() {
            let id = 3000 + i; // Start audio tokens from 3000
            token_to_id.insert(concept.to_string(), id);
            id_to_token.insert(id, concept.to_string());
            audio_ids.push(id);
        }
        modality_tokens.insert(ModalityType::Audio, audio_ids);
        
        // Video vocabulary (motion concepts)
        let video_concepts = vec![
            "moving", "still", "running", "walking", "jumping", "sitting", "standing",
            "dancing", "fighting", "playing", "working", "eating", "drinking", "sleeping",
            "fast", "slow", "smooth", "jerky", "up", "down", "left", "right",
            "forward", "backward", "zoom", "pan", "rotate", "fade", "cut", "transition",
        ];
        
        let mut video_ids = Vec::new();
        for (i, concept) in video_concepts.iter().enumerate() {
            let id = 4000 + i; // Start video tokens from 4000
            token_to_id.insert(concept.to_string(), id);
            id_to_token.insert(id, concept.to_string());
            video_ids.push(id);
        }
        modality_tokens.insert(ModalityType::Video, video_ids);
        
        // Action vocabulary (action concepts)
        let action_concepts = vec![
            "click", "type", "scroll", "drag", "drop", "swipe", "tap", "press",
            "move", "stop", "start", "pause", "resume", "cancel", "confirm", "submit",
            "open", "close", "save", "load", "delete", "copy", "paste", "cut",
            "undo", "redo", "search", "find", "replace", "select", "deselect",
        ];
        
        let mut action_ids = Vec::new();
        for (i, concept) in action_concepts.iter().enumerate() {
            let id = 5000 + i; // Start action tokens from 5000
            token_to_id.insert(concept.to_string(), id);
            id_to_token.insert(id, concept.to_string());
            action_ids.push(id);
        }
        modality_tokens.insert(ModalityType::Action, action_ids);
        
        Ok(())
    }
    
    /// Get token ID for string
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }
    
    /// Get string for token ID
    pub fn id_to_token(&self, id: usize) -> Option<&String> {
        self.id_to_token.get(&id)
    }
    
    /// Get tokens for modality
    pub fn get_modality_tokens(&self, modality: ModalityType) -> Option<&Vec<usize>> {
        self.modality_tokens.get(&modality)
    }
    
    /// Add new token to vocabulary
    pub fn add_token(&mut self, token: String, modality: ModalityType) -> Result<usize> {
        // Check if token already exists
        if let Some(&existing_id) = self.token_to_id.get(&token) {
            return Ok(existing_id);
        }
        
        // Find next available ID
        let next_id = self.token_to_id.len();
        
        if next_id >= self.vocab_size {
            return Err(crate::multimodal::caffeine::error::CaffeineError::tokenizer(
                "Vocabulary size exceeded"
            ));
        }
        
        // Add token
        self.token_to_id.insert(token.clone(), next_id);
        self.id_to_token.insert(next_id, token.clone());
        
        // Add to modality tokens
        if let Some(modality_tokens) = self.modality_tokens.get_mut(&modality) {
            modality_tokens.push(next_id);
        }
        
        Ok(next_id)
    }
    
    /// Get special token ID
    pub fn get_special_token_id(&self, token_name: &str) -> Option<usize> {
        self.special_tokens.get(token_name).copied()
    }
    
    /// Get end token ID
    pub fn get_end_token_id(&self) -> usize {
        self.special_tokens.get("<END>").copied().unwrap_or(6)
    }
    
    /// Get start token ID
    pub fn get_start_token_id(&self) -> usize {
        self.special_tokens.get("<START>").copied().unwrap_or(5)
    }
    
    /// Check if token is special
    pub fn is_special_token(&self, token_id: usize) -> bool {
        self.special_tokens.values().any(|&id| id == token_id)
    }
    
    /// Get modality for token ID
    pub fn get_token_modality(&self, token_id: usize) -> Option<ModalityType> {
        for (modality, token_ids) in &self.modality_tokens {
            if token_ids.contains(&token_id) {
                return Some(*modality);
            }
        }
        None
    }
    
    /// Get vocabulary statistics
    pub fn get_stats(&self) -> VocabStats {
        let mut modality_counts = HashMap::new();
        
        for (modality, token_ids) in &self.modality_tokens {
            modality_counts.insert(*modality, token_ids.len());
        }
        
        VocabStats {
            total_tokens: self.token_to_id.len(),
            special_tokens: self.special_tokens.len(),
            modality_counts,
            vocab_size: self.vocab_size,
        }
    }
    
    /// Export vocabulary to JSON
    pub fn export_to_json(&self) -> Result<String> {
        let export_data = serde_json::json!({
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
            "modality_tokens": self.modality_tokens
                .iter()
                .map(|(k, v)| (format!("{:?}", k), v))
                .collect::<HashMap<_, _>>()
        });
        
        Ok(export_data.to_string())
    }
    
    /// Import vocabulary from JSON
    pub fn import_from_json(&mut self, json_str: &str) -> Result<()> {
        let data: serde_json::Value = serde_json::from_str(json_str)?;
        
        if let Some(token_to_id) = data.get("token_to_id").and_then(|v| v.as_object()) {
            for (token, id) in token_to_id {
                if let Some(id_num) = id.as_u64() {
                    self.token_to_id.insert(token.clone(), id_num as usize);
                    self.id_to_token.insert(id_num as usize, token.clone());
                }
            }
        }
        
        Ok(())
    }
}

/// Vocabulary statistics
#[derive(Debug, Clone)]
pub struct VocabStats {
    pub total_tokens: usize,
    pub special_tokens: usize,
    pub modality_counts: HashMap<ModalityType, usize>,
    pub vocab_size: usize,
}

/// Vocabulary builder for easy construction
pub struct VocabularyBuilder {
    vocab_size: usize,
    token_dim: usize,
    custom_tokens: Vec<(String, ModalityType)>,
}

impl VocabularyBuilder {
    /// Create new vocabulary builder
    pub fn new(vocab_size: usize, _token_dim: usize) -> Self {
        Self {
            vocab_size,
            token_dim: _token_dim,
            custom_tokens: Vec::new(),
        }
    }
    
    /// Add custom token
    pub fn add_token(mut self, token: String, modality: ModalityType) -> Self {
        self.custom_tokens.push((token, modality));
        self
    }
    
    /// Build vocabulary
    pub fn build(self) -> Result<MultimodalVocabulary> {
        let mut vocab = MultimodalVocabulary::new(self.vocab_size, self.token_dim)?;
        
        // Add custom tokens
        for (token, modality) in self.custom_tokens {
            vocab.add_token(token, modality)?;
        }
        
        Ok(vocab)
    }
}
