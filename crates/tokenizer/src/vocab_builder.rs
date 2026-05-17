//! Vocabulary Builder - Rust implementation
//! 
//! Building and managing token vocabularies

use std::collections::HashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use crate::tokenizer_core::TokenizerCore;
use crate::special_tokens::SpecialTokens;

/// Vocabulary entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabEntry {
    pub token: String,
    pub id: u32,
    pub frequency: u64,
    pub score: f32,
    pub is_special: bool,
}

/// Vocabulary builder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabBuilderConfig {
    pub max_vocab_size: usize,
    pub min_frequency: u64,
    pub byte_level: bool,
    pub include_special_tokens: bool,
    pub score_by_frequency: bool,
    pub normalize_scores: bool,
    pub custom_tokens: Vec<String>,
}

impl Default for VocabBuilderConfig {
    fn default() -> Self {
        Self {
            max_vocab_size: 30000,
            min_frequency: 2,
            byte_level: true,
            include_special_tokens: true,
            score_by_frequency: true,
            normalize_scores: true,
            custom_tokens: Vec::new(),
        }
    }
}

/// Vocabulary builder
pub struct VocabBuilder {
    config: VocabBuilderConfig,
    vocab: HashMap<String, VocabEntry>,
    next_id: u32,
    _special_tokens: SpecialTokens,
}

impl VocabBuilder {
    /// Create a new vocabulary builder
    pub fn new(config: VocabBuilderConfig) -> Self {
        let mut vocab = HashMap::new();
        let mut next_id = 0;
        let special_tokens = SpecialTokens::new();
        
        // Add special tokens if enabled
        if config.include_special_tokens {
            for special_id in special_tokens.get_all_ids() {
                if let Some(token_str) = special_tokens.get_token_str(special_id) {
                    vocab.insert(token_str.to_string(), VocabEntry {
                        token: token_str.to_string(),
                        id: special_id,
                        frequency: 0,
                        score: f32::INFINITY, // Highest priority for special tokens
                        is_special: true,
                    });
                    next_id = next_id.max(special_id + 1);
                }
            }
        }
        
        // Add byte-level tokens if enabled
        if config.byte_level {
            for i in 0..256 {
                let byte_token = (i as u8 as char).to_string();
                vocab.insert(byte_token.clone(), VocabEntry {
                    token: byte_token,
                    id: next_id,
                    frequency: 0,
                    score: 0.0,
                    is_special: false,
                });
                next_id += 1;
            }
        }
        
        // Add custom tokens
        for custom_token in &config.custom_tokens {
            if !vocab.contains_key(custom_token) {
                vocab.insert(custom_token.clone(), VocabEntry {
                    token: custom_token.clone(),
                    id: next_id,
                    frequency: 0,
                    score: 0.0,
                    is_special: false,
                });
                next_id += 1;
            }
        }
        
        Self {
            config,
            vocab,
            next_id,
            _special_tokens: special_tokens,
        }
    }
    
    /// Create builder with default config
    pub fn with_default_config() -> Self {
        Self::new(VocabBuilderConfig::default())
    }
    
    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: &str, frequency: u64) -> Result<u32> {
        if let Some(entry) = self.vocab.get_mut(token) {
            // Update existing token frequency
            entry.frequency += frequency;
            Ok(entry.id)
        } else {
            // Add new token
            let id = self.next_id;
            self.next_id += 1;
            
            let score = if self.config.score_by_frequency {
                frequency as f32
            } else {
                0.0
            };
            
            self.vocab.insert(token.to_string(), VocabEntry {
                token: token.to_string(),
                id,
                frequency,
                score,
                is_special: false,
            });
            
            Ok(id)
        }
    }
    
    /// Add multiple tokens from text
    pub fn add_tokens_from_text(&mut self, text: &str) -> Result<()> {
        let tokens = self.extract_tokens(text)?;
        
        for token in tokens {
            self.add_token(&token, 1)?;
        }
        
        Ok(())
    }
    
    /// Extract tokens from text (simple whitespace-based tokenization)
    fn extract_tokens(&self, text: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        
        // Simple tokenization by whitespace
        for word in text.split_whitespace() {
            tokens.push(word.to_string());
        }
        
        // If byte-level is enabled, also add individual bytes
        if self.config.byte_level {
            for ch in text.chars() {
                let byte_token = ch.to_string();
                if !tokens.contains(&byte_token) {
                    tokens.push(byte_token);
                }
            }
        }
        
        Ok(tokens)
    }
    
    /// Build the final vocabulary
    pub fn build(mut self) -> Result<Vec<VocabEntry>> {
        // Filter by minimum frequency
        self.vocab.retain(|_, entry| {
            entry.is_special || entry.frequency >= self.config.min_frequency
        });
        
        // Calculate scores if needed
        if !self.config.score_by_frequency {
            self.calculate_scores();
        }
        
        // Normalize scores if needed
        if self.config.normalize_scores {
            self.normalize_scores();
        }
        
        // Sort by score (descending) and then by frequency (descending)
        let mut vocab_entries: Vec<VocabEntry> = self.vocab.into_values().collect();
        vocab_entries.sort_by(|a, b| {
            b.score.partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.frequency.cmp(&a.frequency))
        });
        
        // Limit vocabulary size
        if vocab_entries.len() > self.config.max_vocab_size {
            vocab_entries.truncate(self.config.max_vocab_size);
        }
        
        // Reassign IDs based on final order
        for (i, entry) in vocab_entries.iter_mut().enumerate() {
            entry.id = i as u32;
        }
        
        Ok(vocab_entries)
    }
    
    /// Calculate scores based on frequency and other factors
    fn calculate_scores(&mut self) {
        let total_frequency: u64 = self.vocab.values()
            .filter(|e| !e.is_special)
            .map(|e| e.frequency)
            .sum();
        
        for entry in self.vocab.values_mut() {
            if !entry.is_special {
                // Score based on frequency and length (shorter tokens get higher scores)
                let length_factor = 1.0 / (entry.token.len() as f32 + 1.0);
                let frequency_factor = entry.frequency as f64 / total_frequency as f64;
                entry.score = frequency_factor as f32 * length_factor;
            }
        }
    }
    
    /// Normalize scores to [0, 1] range
    fn normalize_scores(&mut self) {
        let non_special_entries: Vec<&mut VocabEntry> = self.vocab.values_mut()
            .filter(|e| !e.is_special)
            .collect();
        
        if non_special_entries.is_empty() {
            return;
        }
        
        let min_score = non_special_entries.iter()
            .map(|e| e.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        let max_score = non_special_entries.iter()
            .map(|e| e.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);
        
        if max_score > min_score {
            for entry in non_special_entries {
                entry.score = (entry.score - min_score) / (max_score - min_score);
            }
        }
    }
    
    /// Get current vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    /// Get vocabulary statistics
    pub fn get_stats(&self) -> VocabBuilderStats {
        let total_frequency: u64 = self.vocab.values().map(|e| e.frequency).sum();
        let non_special_count = self.vocab.values().filter(|e| !e.is_special).count();
        let special_count = self.vocab.values().filter(|e| e.is_special).count();
        
        let mut token_lengths = Vec::new();
        for entry in self.vocab.values() {
            if !entry.is_special {
                token_lengths.push(entry.token.len());
            }
        }
        
        token_lengths.sort_unstable();
        let avg_length = if !token_lengths.is_empty() {
            token_lengths.iter().sum::<usize>() as f64 / token_lengths.len() as f64
        } else {
            0.0
        };
        
        let median_length = if !token_lengths.is_empty() {
            token_lengths[token_lengths.len() / 2]
        } else {
            0
        };
        
        VocabBuilderStats {
            vocab_size: self.vocab.len(),
            total_frequency,
            non_special_count,
            special_count,
            avg_token_length: avg_length,
            median_token_length: median_length,
            min_frequency: self.config.min_frequency,
            max_vocab_size: self.config.max_vocab_size,
        }
    }
    
    /// Merge with another vocabulary builder
    pub fn merge(&mut self, other: VocabBuilder) -> Result<()> {
        for (_, entry) in other.vocab {
            self.add_token(&entry.token, entry.frequency)?;
        }
        Ok(())
    }
    
    /// Filter vocabulary by criteria
    pub fn filter<F>(&mut self, predicate: F) 
    where 
        F: Fn(&VocabEntry) -> bool,
    {
        self.vocab.retain(|_, entry| predicate(entry));
    }
    
    /// Export vocabulary to tokenizer
    pub fn export_to_tokenizer(&self, tokenizer: &mut TokenizerCore) -> Result<()> {
        for entry in self.vocab.values() {
            tokenizer.add_token(&entry.token, entry.frequency)?;
        }
        Ok(())
    }
    
    /// Save vocabulary to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let vocab_entries: Vec<VocabEntry> = self.vocab.values().cloned().collect();
        let json = serde_json::to_string_pretty(&vocab_entries)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load vocabulary from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let vocab_entries: Vec<VocabEntry> = serde_json::from_str(&json)?;
        
        let mut builder = Self::with_default_config();
        builder.vocab.clear();
        
        for entry in vocab_entries {
            builder.vocab.insert(entry.token.clone(), entry.clone());
            builder.next_id = builder.next_id.max(entry.id + 1);
        }
        
        Ok(builder)
    }
}

/// Vocabulary builder statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabBuilderStats {
    pub vocab_size: usize,
    pub total_frequency: u64,
    pub non_special_count: usize,
    pub special_count: usize,
    pub avg_token_length: f64,
    pub median_token_length: usize,
    pub min_frequency: u64,
    pub max_vocab_size: usize,
}

/// Convenience functions
pub fn build_vocab_from_texts(texts: &[String]) -> Result<Vec<VocabEntry>> {
    let mut builder = VocabBuilder::with_default_config();
    
    for text in texts {
        builder.add_tokens_from_text(text)?;
    }
    
    builder.build()
}

pub fn build_vocab_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<VocabEntry>> {
    let content = std::fs::read_to_string(path)?;
    let texts: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    build_vocab_from_texts(&texts)
}

pub fn create_byte_level_vocab() -> Result<Vec<VocabEntry>> {
    let config = VocabBuilderConfig {
        byte_level: true,
        include_special_tokens: true,
        ..Default::default()
    };
    
    let builder = VocabBuilder::new(config);
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vocab_builder_creation() {
        let builder = VocabBuilder::with_default_config();
        assert!(builder.vocab_size() > 0); // Should have special tokens and byte tokens
    }
    
    #[test]
    fn test_add_token() {
        let mut builder = VocabBuilder::with_default_config();
        let _id = builder.add_token("hello", 5).unwrap();
        assert!(builder.vocab.contains_key("hello"));
        
        let stats = builder.get_stats();
        assert_eq!(stats.total_frequency, 5);
    }
    
    #[test]
    fn test_add_tokens_from_text() {
        let mut builder = VocabBuilder::with_default_config();
        builder.add_tokens_from_text("hello world").unwrap();
        
        assert!(builder.vocab.contains_key("hello"));
        assert!(builder.vocab.contains_key("world"));
    }
    
    #[test]
    fn test_build_vocab() {
        let mut builder = VocabBuilder::with_default_config();
        
        builder.add_token("hello", 10).unwrap();
        builder.add_token("world", 5).unwrap();
        builder.add_token("test", 1).unwrap();
        
        let config = VocabBuilderConfig {
            min_frequency: 2,
            max_vocab_size: 10,
            ..Default::default()
        };
        
        builder.config = config;
        let vocab = builder.build().unwrap();
        
        // Should only include tokens with frequency >= 2
        assert_eq!(vocab.len(), 2); // "hello" and "world"
    }
    
    #[test]
    fn test_byte_level_vocab() {
        let vocab = create_byte_level_vocab().unwrap();
        
        // Should have 256 byte tokens + special tokens
        assert!(vocab.len() >= 256);
        
        // Check that byte tokens are present
        let byte_tokens: Vec<String> = vocab.iter()
            .filter(|e| !e.is_special && e.token.len() == 1)
            .map(|e| e.token.clone())
            .collect();
        
        assert!(byte_tokens.len() >= 256);
    }
    
    #[test]
    fn test_custom_tokens() {
        let config = VocabBuilderConfig {
            custom_tokens: vec!["<custom>".to_string(), "<test>".to_string()],
            ..Default::default()
        };
        
        let builder = VocabBuilder::new(config);
        assert!(builder.vocab.contains_key("<custom>"));
        assert!(builder.vocab.contains_key("<test>"));
    }
    
    #[test]
    fn test_vocab_stats() {
        let mut builder = VocabBuilder::with_default_config();
        
        builder.add_token("a", 1).unwrap();
        builder.add_token("ab", 2).unwrap();
        builder.add_token("abc", 3).unwrap();
        
        let stats = builder.get_stats();
        assert_eq!(stats.total_frequency, 6);
        assert!(stats.avg_token_length > 0.0);
    }
    
    #[test]
    fn test_vocab_serialization() {
        let mut builder = VocabBuilder::with_default_config();
        builder.add_token("test", 1).unwrap();
        
        // Test saving and loading
        let temp_file = "/tmp/test_vocab.json";
        builder.save_to_file(temp_file).unwrap();
        
        let loaded = VocabBuilder::load_from_file(temp_file).unwrap();
        assert!(loaded.vocab.contains_key("test"));
        
        // Clean up
        std::fs::remove_file(temp_file).ok();
    }
    
    #[test]
    fn test_build_vocab_from_texts() {
        let texts = vec![
            "hello world".to_string(),
            "hello test".to_string(),
            "world test".to_string(),
        ];
        
        let vocab = build_vocab_from_texts(&texts).unwrap();
        
        // Should contain the unique words
        let vocab_strings: Vec<String> = vocab.iter()
            .filter(|e| !e.is_special)
            .map(|e| e.token.clone())
            .collect();
        
        assert!(vocab_strings.contains(&"hello".to_string()));
        assert!(vocab_strings.contains(&"world".to_string()));
        assert!(vocab_strings.contains(&"test".to_string()));
    }
    
    #[test]
    fn test_filter_vocab() {
        let mut builder = VocabBuilder::with_default_config();
        
        builder.add_token("a", 1).unwrap();
        builder.add_token("ab", 2).unwrap();
        builder.add_token("abc", 3).unwrap();
        
        // Filter tokens with length >= 2
        builder.filter(|entry| entry.token.len() >= 2);
        
        assert!(!builder.vocab.contains_key("a"));
        assert!(builder.vocab.contains_key("ab"));
        assert!(builder.vocab.contains_key("abc"));
    }
}
