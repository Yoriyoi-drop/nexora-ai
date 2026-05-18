//! Tokenizer Core - Rust implementation
//! 
//! Core tokenizer functionality with vocabulary and merge rules

use std::borrow::Cow;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use rand::Rng;
use crate::special_tokens::SpecialTokens;

/// Token pair for vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub token: String,
    pub id: u32,
    pub frequency: u64,
}

/// Merge rule for BPE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRule {
    pub left: u32,
    pub right: u32,
    pub new_id: u32,
    pub priority: f32,
    pub frequency: u64,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub num_merges: usize,
    pub special_tokens: bool,
    pub lowercase: bool,
    pub max_length: usize,
    pub dropout: Option<f32>,
    pub add_prefix_space: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            num_merges: 30000,
            special_tokens: true,
            lowercase: false,
            max_length: 512,
            dropout: None,
            add_prefix_space: false,
        }
    }
}

/// Core tokenizer implementation
#[derive(Debug, Clone)]
pub struct TokenizerCore {
    config: TokenizerConfig,
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<MergeRule>,
    special_tokens: SpecialTokens,
    next_id: u32,
}

impl TokenizerCore {
    /// Create a new tokenizer core
    pub fn new(config: TokenizerConfig) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        let special_tokens = SpecialTokens::new();
        
        // Initialize with special tokens if enabled
        let mut next_id = 0;
        if config.special_tokens {
            for special_id in special_tokens.get_all_ids() {
                if let Some(token_str) = special_tokens.get_token_str(special_id) {
                    vocab.insert(token_str.to_string(), special_id);
                    reverse_vocab.insert(special_id, token_str.to_string());
                    next_id = next_id.max(special_id + 1);
                }
            }
        }
        
        Self {
            config,
            vocab,
            reverse_vocab,
            merges: Vec::new(),
            special_tokens,
            next_id,
        }
    }
    
    /// Create tokenizer with default config
    pub fn with_default_config() -> Self {
        Self::new(TokenizerConfig::default())
    }
    
    /// Add a token to vocabulary
    pub fn add_token(&mut self, token: &str, _frequency: u64) -> Result<u32> {
        if let Some(&existing_id) = self.vocab.get(token) {
            // Update frequency if token already exists
            // Note: In a real implementation, you'd track frequencies separately
            return Ok(existing_id);
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        self.vocab.insert(token.to_string(), id);
        self.reverse_vocab.insert(id, token.to_string());
        
        Ok(id)
    }
    
    /// Get token ID from string
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    
    /// Get token string from ID
    pub fn get_token_str(&self, id: u32) -> Option<&str> {
        self.reverse_vocab.get(&id).map(|s| s.as_str())
    }
    
    /// Check if token exists in vocabulary
    pub fn has_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    /// Add a merge rule
    pub fn add_merge(&mut self, left: u32, right: u32, priority: f32, frequency: u64) -> Result<u32> {
        let new_id = self.next_id;
        self.next_id += 1;
        
        let merge_rule = MergeRule {
            left,
            right,
            new_id,
            priority,
            frequency,
        };
        
        self.merges.push(merge_rule);
        
        Ok(new_id)
    }
    
    /// Get merge rules
    pub fn get_merges(&self) -> &[MergeRule] {
        &self.merges
    }
    
    /// Tokenize text into token IDs
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(text.len());
        
        // Add prefix space if configured
        let processed_text: Cow<'_, str> = if self.config.add_prefix_space && !text.starts_with(' ') {
            Cow::Owned(format!(" {text}"))
        } else {
            Cow::Borrowed(text)
        };
        
        // Convert to lowercase if configured
        let processed_text: Cow<'_, str> = if self.config.lowercase {
            Cow::Owned(processed_text.to_lowercase())
        } else {
            processed_text
        };
        
        // Split into characters (byte-level tokenization)
        let chars: Vec<char> = processed_text.chars().collect();
        
        // Convert each character to token ID
        for ch in chars {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(s) {
                tokens.push(id);
            } else {
                // Use unknown token
                tokens.push(self.special_tokens.get_id(&crate::special_tokens::SpecialTokenID::Unk));
            }
        }
        
        // Apply BPE merges
        if !self.merges.is_empty() {
            tokens = self.apply_bpe_merges(&tokens)?;
        }
        
        // Truncate if too long
        if tokens.len() > self.config.max_length {
            tokens.truncate(self.config.max_length);
        }
        
        Ok(tokens)
    }
    
    /// Apply BPE merge rules to token sequence
    fn apply_bpe_merges(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        if tokens.len() <= 1 {
            return Ok(tokens.to_vec());
        }

        let merge_map: HashMap<(u32, u32), u32> = self.merges.iter()
            .map(|m| ((m.left, m.right), m.new_id))
            .collect();

        let mut result = tokens.to_vec();
        loop {
            let mut merged = false;
            let mut new_result = Vec::with_capacity(result.len());
            let mut i = 0;
            while i < result.len() {
                if i + 1 < result.len() {
                    if let Some(&new_id) = merge_map.get(&(result[i], result[i + 1])) {
                        new_result.push(new_id);
                        i += 2;
                        merged = true;
                        continue;
                    }
                }
                new_result.push(result[i]);
                i += 1;
            }
            result = new_result;
            if !merged {
                break;
            }
        }
        Ok(result)
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();
        
        for &id in token_ids {
            if let Some(token_str) = self.get_token_str(id) {
                result.push_str(token_str);
            } else {
                // Handle unknown token
                result.push_str(self.special_tokens.get_token(&crate::special_tokens::SpecialTokenID::Unk));
            }
        }
        
        Ok(result)
    }
    
    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
    
    /// Get configuration
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: TokenizerConfig) {
        self.config = config;
    }
    
    /// Get vocabulary statistics
    pub fn get_vocab_stats(&self) -> VocabStats {
        let total_frequency: u64 = self.vocab.values().len() as u64; // Simplified
        
        VocabStats {
            vocab_size: self.vocab.len(),
            merge_count: self.merges.len(),
            total_frequency,
            average_frequency: if self.vocab.is_empty() { 0.0 } else { total_frequency as f64 / self.vocab.len() as f64 },
        }
    }
    
    /// Save tokenizer to JSON
    pub fn save_to_json(&self) -> Result<String> {
        let tokenizer_data = TokenizerData {
            config: self.config.clone(),
            vocab: self.vocab.clone(),
            merges: self.merges.clone(),
            next_id: self.next_id,
        };
        
        serde_json::to_string_pretty(&tokenizer_data).map_err(|e| anyhow::anyhow!("Failed to serialize tokenizer: {}", e))
    }
    
    /// Load tokenizer from JSON
    pub fn load_from_json(json_str: &str) -> Result<Self> {
        let tokenizer_data: TokenizerData = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize tokenizer: {}", e))?;
        
        let mut tokenizer = Self::new(tokenizer_data.config);
        tokenizer.vocab = tokenizer_data.vocab;
        tokenizer.merges = tokenizer_data.merges;
        tokenizer.next_id = tokenizer_data.next_id;
        
        // Rebuild reverse vocab
        tokenizer.reverse_vocab.clear();
        for (token_str, &id) in &tokenizer.vocab {
            tokenizer.reverse_vocab.insert(id, token_str.clone());
        }
        
        Ok(tokenizer)
    }
    
    /// Apply dropout to tokens (for training)
    pub fn apply_dropout(&self, tokens: Vec<u32>) -> Result<Vec<u32>> {
        if let Some(dropout_rate) = self.config.dropout {
            if dropout_rate > 0.0 && dropout_rate < 1.0 {
                let mut rng = rand::thread_rng();
                let mut result = Vec::with_capacity(tokens.len());
                
                for token in tokens {
                    if rng.gen::<f64>() >= dropout_rate.into() {
                        result.push(token);
                    }
                }
                
                Ok(result)
            } else {
                Ok(tokens)
            }
        } else {
            Ok(tokens)
        }
    }
    
    /// Validate tokenizer state
    pub fn validate(&self) -> Result<()> {
        // Check if reverse vocab matches vocab
        if self.vocab.len() != self.reverse_vocab.len() {
            return Err(anyhow::anyhow!("Vocab and reverse vocab size mismatch"));
        }
        
        // Check if all IDs in reverse vocab exist in vocab
        for (&id, token_str) in &self.reverse_vocab {
            if let Some(&vocab_id) = self.vocab.get(token_str) {
                if vocab_id != id {
                    return Err(anyhow::anyhow!("ID mismatch for token '{}': vocab={}, reverse={}", token_str, vocab_id, id));
                }
            } else {
                return Err(anyhow::anyhow!("Token '{}' in reverse vocab not found in vocab", token_str));
            }
        }
        
        // Check merge rules
        for merge in &self.merges {
            if !self.reverse_vocab.contains_key(&merge.left) {
                return Err(anyhow::anyhow!("Merge rule references non-existent left token ID: {}", merge.left));
            }
            if !self.reverse_vocab.contains_key(&merge.right) {
                return Err(anyhow::anyhow!("Merge rule references non-existent right token ID: {}", merge.right));
            }
            if !self.reverse_vocab.contains_key(&merge.new_id) {
                return Err(anyhow::anyhow!("Merge rule references non-existent new token ID: {}", merge.new_id));
            }
        }
        
        Ok(())
    }
}

/// Vocabulary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabStats {
    pub vocab_size: usize,
    pub merge_count: usize,
    pub total_frequency: u64,
    pub average_frequency: f64,
}

/// Tokenizer data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenizerData {
    config: TokenizerConfig,
    vocab: HashMap<String, u32>,
    merges: Vec<MergeRule>,
    next_id: u32,
}

impl Default for TokenizerCore {
    fn default() -> Self {
        Self::with_default_config()
    }
}

/// Convenience functions
pub fn create_tokenizer() -> TokenizerCore {
    TokenizerCore::with_default_config()
}

pub fn tokenize_text(text: &str) -> Result<Vec<u32>> {
    let tokenizer = create_tokenizer();
    tokenizer.tokenize(text)
}

pub fn decode_tokens(token_ids: &[u32]) -> Result<String> {
    let tokenizer = create_tokenizer();
    tokenizer.decode(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = TokenizerCore::with_default_config();
        assert!(tokenizer.vocab_size() > 0); // Should have special tokens
    }
    
    #[test]
    fn test_add_token() {
        let mut tokenizer = TokenizerCore::with_default_config();
        let id = tokenizer.add_token("hello", 1).unwrap();
        assert!(tokenizer.has_token("hello"));
        assert_eq!(tokenizer.get_token_id("hello"), Some(id));
        assert_eq!(tokenizer.get_token_str(id), Some("hello"));
    }
    
    #[test]
    fn test_tokenize_simple() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        // Add some basic tokens
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let tokens = tokenizer.tokenize("hello").unwrap();
        assert_eq!(tokens.len(), 5); // h, e, l, l, o
    }
    
    #[test]
    fn test_decode_simple() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        // Add some basic tokens
        let h_id = tokenizer.add_token("h", 1).unwrap();
        let e_id = tokenizer.add_token("e", 1).unwrap();
        let l_id = tokenizer.add_token("l", 1).unwrap();
        let o_id = tokenizer.add_token("o", 1).unwrap();
        
        let tokens = vec![h_id, e_id, l_id, l_id, o_id];
        let text = tokenizer.decode(&tokens).unwrap();
        assert_eq!(text, "hello");
    }
    
    #[test]
    fn test_bpe_merges() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        // Add basic tokens
        let _h_id = tokenizer.add_token("h", 1).unwrap();
        let _e_id = tokenizer.add_token("e", 1).unwrap();
        let l_id = tokenizer.add_token("l", 1).unwrap();
        let _o_id = tokenizer.add_token("o", 1).unwrap();
        
        // Add merge rule for "ll" -> "ll"
        let _ll_id = tokenizer.add_token("ll", 1).unwrap();
        tokenizer.add_merge(l_id, l_id, 1.0, 1).unwrap();
        
        let tokens = tokenizer.tokenize("hello").unwrap();
        // Should merge "ll" into single token
        assert_eq!(tokens.len(), 4); // h, e, ll, o
    }
    
    #[test]
    fn test_lowercase() {
        let mut tokenizer = TokenizerCore::new(TokenizerConfig {
            lowercase: true,
            ..Default::default()
        });
        
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let tokens = tokenizer.tokenize("HELLO").unwrap();
        assert_eq!(tokens.len(), 5);
    }
    
    #[test]
    fn test_prefix_space() {
        let mut tokenizer = TokenizerCore::new(TokenizerConfig {
            add_prefix_space: true,
            ..Default::default()
        });
        
        tokenizer.add_token(" ", 1).unwrap();
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let tokens = tokenizer.tokenize("hello").unwrap();
        assert_eq!(tokens.len(), 6); // space, h, e, l, l, o
    }
    
    #[test]
    fn test_special_tokens() {
        let tokenizer = TokenizerCore::with_default_config();
        
        let unk_id = tokenizer.special_tokens().get_id(&crate::special_tokens::SpecialTokenID::Unk);
        assert!(tokenizer.get_token_str(unk_id).is_some());
        
        // Test tokenizing with unknown characters
        let tokens = tokenizer.tokenize("🤖").unwrap();
        assert_eq!(tokens[0], unk_id);
    }
    
    #[test]
    fn test_validation() {
        let tokenizer = TokenizerCore::with_default_config();
        assert!(tokenizer.validate().is_ok());
    }
    
    #[test]
    fn test_serialization() {
        let mut tokenizer = TokenizerCore::with_default_config();
        tokenizer.add_token("test", 1).unwrap();
        
        let json = tokenizer.save_to_json().unwrap();
        let loaded = TokenizerCore::load_from_json(&json).unwrap();
        
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
        assert!(loaded.has_token("test"));
    }
}
