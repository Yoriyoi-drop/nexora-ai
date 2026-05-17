//! BPE (Byte Pair Encoding) Tokenizer Implementation
//! 
//! Implementasi BPE algorithm untuk text tokenization dengan vocabulary training

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// BPE Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeConfig {
    pub vocab_size: usize,
    pub special_tokens: HashMap<String, u32>,
    pub min_frequency: u32,
    pub unknown_token: String,
    pub pad_token: String,
    pub bos_token: String,
    pub eos_token: String,
}

impl Default for BpeConfig {
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<unk>".to_string(), 0);
        special_tokens.insert("<pad>".to_string(), 1);
        special_tokens.insert("<bos>".to_string(), 2);
        special_tokens.insert("<eos>".to_string(), 3);
        
        Self {
            vocab_size: 30000,
            special_tokens,
            min_frequency: 2,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
        }
    }
}

/// BPE Tokenizer
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    config: BpeConfig,
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    unicode_to_byte: HashMap<char, u8>,
    byte_to_unicode: HashMap<u8, char>,
}

impl BpeTokenizer {
    pub fn new(config: BpeConfig) -> Self {
        let mut tokenizer = Self {
            config,
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            merges: Vec::new(),
            unicode_to_byte: HashMap::new(),
            byte_to_unicode: HashMap::new(),
        };
        
        tokenizer.init_unicode_mapping();
        tokenizer
    }
    
    fn init_unicode_mapping(&mut self) {
        // Initialize Unicode to byte mapping (similar to GPT-2 tokenizer)
        let mut byte_values: Vec<u8> = (0..=255).collect();
        
        // Shift bytes to avoid conflicts with whitespace and control characters
        for i in (0..=255).rev() {
            if i == 0 || i == 32 || (i >= 127 && i <= 160) {
                byte_values.remove(i as usize);
            }
        }
        
        // Add shifted bytes to the end
        for i in (0..=255).rev() {
            if i == 0 || i == 32 || (i >= 127 && i <= 160) {
                byte_values.push(i);
            }
        }
        
        // Create mappings
        for (i, &byte_val) in byte_values.iter().enumerate() {
            let unicode_char = std::char::from_u32(i as u32).unwrap_or('\u{FFFD}'); // Replacement character
            self.unicode_to_byte.insert(unicode_char, byte_val);
            self.byte_to_unicode.insert(byte_val, unicode_char);
        }
    }
    
    /// Train BPE tokenizer dari text corpus
    pub fn train(&mut self, corpus: &str) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting BPE training with vocab size: {}", self.config.vocab_size);
        
        // Step 1: Initialize vocabulary with individual characters
        let mut vocab: HashSet<String> = HashSet::new();
        let mut word_freqs: HashMap<String, u32> = HashMap::new();
        
        // Preprocess corpus and count word frequencies
        for line in corpus.lines() {
            let processed_line = self.preprocess_line(line);
            for word in processed_line.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
                
                // Add individual characters to vocabulary
                for char in word.chars() {
                    vocab.insert(char.to_string());
                }
            }
        }
        
        info!("Initial vocabulary size: {}", vocab.len());
        
        // Step 2: Add special tokens to vocabulary
        for (token, _) in &self.config.special_tokens {
            vocab.insert(token.clone());
        }
        
        // Step 3: Learn BPE merges
        let mut merges = Vec::new();
        let mut current_vocab = vocab.clone();
        
        while current_vocab.len() < self.config.vocab_size {
            // Find most frequent pair
            if let Some((pair, freq)) = self.find_most_frequent_pair(&word_freqs, &current_vocab) {
                if freq < self.config.min_frequency {
                    break;
                }
                
                // Create new token
                let new_token = pair.0.clone() + &pair.1;
                
                // Add to vocabulary and merges
                current_vocab.insert(new_token.clone());
                merges.push(pair.clone());
                
                // Update word frequencies with new token
                self.update_word_freqs(&mut word_freqs, &pair, &new_token);
                
                debug!("Added merge: {} + {} -> {} (freq: {})", pair.0, pair.1, new_token, freq);
            } else {
                break;
            }
        }
        
        // Step 4: Create final vocabulary mapping
        self.vocab.clear();
        self.reverse_vocab.clear();
        
        for (i, token) in current_vocab.iter().enumerate() {
            self.vocab.insert(token.clone(), i as u32);
            self.reverse_vocab.insert(i as u32, token.clone());
        }
        
        self.merges = merges;
        
        info!("BPE training completed. Final vocabulary size: {}", self.vocab.len());
        info!("Total merges learned: {}", self.merges.len());
        
        Ok(())
    }
    
    fn preprocess_line(&self, line: &str) -> String {
        // Convert to lowercase and handle Unicode
        line.to_lowercase()
    }
    
    fn find_most_frequent_pair(&self, word_freqs: &HashMap<String, u32>, vocab: &HashSet<String>) -> Option<((String, String), u32)> {
        let mut pair_freqs: HashMap<(String, String), u32> = HashMap::new();
        
        for (word, freq) in word_freqs {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            
            for i in 0..chars.len() - 1 {
                let pair = (chars[i].clone(), chars[i + 1].clone());
                
                // Only consider pairs where both tokens are in vocabulary
                if vocab.contains(&pair.0) && vocab.contains(&pair.1) {
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }
        }
        
        // Find most frequent pair
        pair_freqs.into_iter()
            .max_by_key(|&(_, freq)| freq)
            .map(|(pair, freq)| (pair, freq))
    }
    
    fn update_word_freqs(&self, word_freqs: &mut HashMap<String, u32>, pair: &(String, String), new_token: &str) {
        let mut new_word_freqs = HashMap::new();
        
        for (word, freq) in word_freqs.iter() {
            let mut new_word = word.clone();
            
            // Replace all occurrences of the pair
            let pair_str = pair.0.clone() + &pair.1;
            while let Some(pos) = new_word.find(&pair_str) {
                new_word.replace_range(pos..pos + pair.0.len() + pair.1.len(), new_token);
            }
            
            if new_word != *word {
                *new_word_freqs.entry(new_word).or_insert(0) += freq;
            } else {
                *new_word_freqs.entry(new_word).or_insert(0) += freq;
            }
        }
        
        *word_freqs = new_word_freqs;
    }
    
    /// Encode text ke token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Add BOS token
        if let Some(&bos_id) = self.config.special_tokens.get(&self.config.bos_token) {
            tokens.push(bos_id);
        }
        
        // Process each word
        let processed_text = self.preprocess_line(text);
        for word in processed_text.split_whitespace() {
            let word_tokens = self.encode_word(word);
            tokens.extend(word_tokens);
        }
        
        // Add EOS token
        if let Some(&eos_id) = self.config.special_tokens.get(&self.config.eos_token) {
            tokens.push(eos_id);
        }
        
        tokens
    }
    
    fn encode_word(&self, word: &str) -> Vec<u32> {
        // Convert word to bytes
        let byte_tokens: Vec<String> = word.chars()
            .map(|c| {
                if let Some(&byte_val) = self.unicode_to_byte.get(&c) {
                    format!("{:02x}", byte_val)
                } else {
                    "00".to_string() // Unknown character
                }
            })
            .collect();
        
        // Apply BPE merges
        let mut tokens = byte_tokens;
        
        loop {
            let mut best_pair: Option<(usize, usize)> = None;
            let mut best_rank = u32::MAX;
            
            // Find best merge pair
            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                let merged = pair.0.clone() + &pair.1;
                
                if let Some(&token_id) = self.vocab.get(&merged) {
                    if token_id < best_rank {
                        best_rank = token_id;
                        best_pair = Some((i, i + 1));
                    }
                }
            }
            
            if let Some((i, j)) = best_pair {
                // Merge tokens
                let merged = tokens[i].clone() + &tokens[j];
                tokens[i] = merged;
                tokens.remove(j);
            } else {
                break;
            }
        }
        
        // Convert to token IDs
        tokens.into_iter()
            .filter_map(|token| self.vocab.get(&token).copied())
            .collect()
    }
    
    /// Decode token IDs kembali ke text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut text = String::new();
        
        for &token_id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&token_id) {
                // Skip special tokens
                if self.config.special_tokens.contains_key(token) {
                    continue;
                }
                
                // Convert token back to bytes
                let mut chars = Vec::new();
                for i in (0..token.len()).step_by(2) {
                    if i + 1 < token.len() {
                        let byte_str = &token[i..i + 2];
                        if let Ok(byte_val) = u8::from_str_radix(byte_str, 16) {
                            if let Some(&unicode_char) = self.byte_to_unicode.get(&byte_val) {
                                chars.push(unicode_char);
                            }
                        }
                    }
                }
                
                text.push_str(&chars.into_iter().collect::<String>());
            }
        }
        
        text
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    /// Get token untuk string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    
    /// Get string untuk token ID
    pub fn id_to_token(&self, id: u32) -> Option<&String> {
        self.reverse_vocab.get(&id)
    }
    
    /// Save tokenizer ke file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        
        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Save config
        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        fs::write(config_path, config_json)?;
        
        // Save vocabulary
        let vocab_path = path.join("vocab.txt");
        let mut vocab_file = fs::File::create(vocab_path)?;
        for (token, &id) in &self.vocab {
            writeln!(vocab_file, "{} {}", token, id)?;
        }
        
        // Save merges
        let merges_path = path.join("merges.txt");
        let mut merges_file = fs::File::create(merges_path)?;
        for (token1, token2) in &self.merges {
            writeln!(merges_file, "{} {}", token1, token2)?;
        }
        
        // Save Unicode mappings
        let unicode_path = path.join("unicode.json");
        let unicode_data = (
            self.unicode_to_byte.clone(),
            self.byte_to_unicode.clone()
        );
        let unicode_json = serde_json::to_string_pretty(&unicode_data)?;
        fs::write(unicode_path, unicode_json)?;
        
        info!("Tokenizer saved to: {}", path.display());
        Ok(())
    }
    
    /// Load tokenizer dari file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        
        // Load config
        let config_path = path.join("config.json");
        let config_json = fs::read_to_string(config_path)?;
        let config: BpeConfig = serde_json::from_str(&config_json)?;
        
        let mut tokenizer = Self::new(config);
        
        // Load vocabulary
        let vocab_path = path.join("vocab.txt");
        let vocab_file = BufReader::new(fs::File::open(vocab_path)?);
        for line in vocab_file.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let token = parts[0].to_string();
                if let Ok(id) = parts[1].parse::<u32>() {
                    tokenizer.vocab.insert(token.clone(), id);
                    tokenizer.reverse_vocab.insert(id, token);
                }
            }
        }
        
        // Load merges
        let merges_path = path.join("merges.txt");
        if path.exists() {
            let merges_file = BufReader::new(fs::File::open(merges_path)?);
            for line in merges_file.lines() {
                let line = line?;
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    tokenizer.merges.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
        }
        
        // Load Unicode mappings
        let unicode_path = path.join("unicode.json");
        if path.exists() {
            let unicode_json = fs::read_to_string(unicode_path)?;
            let (unicode_to_byte, byte_to_unicode): (HashMap<char, u8>, HashMap<u8, char>) = 
                serde_json::from_str(&unicode_json)?;
            tokenizer.unicode_to_byte = unicode_to_byte;
            tokenizer.byte_to_unicode = byte_to_unicode;
        }
        
        info!("Tokenizer loaded from: {}", path.display());
        Ok(tokenizer)
    }
    
    /// Get tokenizer statistics
    pub fn get_stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.vocab.len(),
            merge_count: self.merges.len(),
            special_tokens_count: self.config.special_tokens.len(),
            max_token_length: self.vocab.keys()
                .map(|t| t.len())
                .max()
                .unwrap_or(0),
        }
    }
    
    /// Add word to vocabulary
    pub fn add_word(&mut self, word: &str, _frequency: u32) -> Result<(), Box<dyn std::error::Error>> {
        if self.vocab.len() >= self.config.vocab_size {
            return Err("Vocabulary size limit reached".into());
        }
        
        if !self.vocab.contains_key(word) {
            let token_id = self.vocab.len() as u32;
            self.vocab.insert(word.to_string(), token_id);
            self.reverse_vocab.insert(token_id, word.to_string());
        }
        
        Ok(())
    }
    
    /// Get unknown token
    pub fn unknown_token(&self) -> &str {
        &self.config.unknown_token
    }
    
    /// Get pad token
    pub fn pad_token(&self) -> &str {
        &self.config.pad_token
    }
    
    /// Get bos token
    pub fn bos_token(&self) -> &str {
        &self.config.bos_token
    }
    
    /// Get eos token
    pub fn eos_token(&self) -> &str {
        &self.config.eos_token
    }
}

/// Tokenizer statistics
#[derive(Debug, Clone, Serialize)]
pub struct TokenizerStats {
    pub vocab_size: usize,
    pub merge_count: usize,
    pub special_tokens_count: usize,
    pub max_token_length: usize,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new(BpeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bpe_tokenizer_basic() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig {
            vocab_size: 100,
            special_tokens: HashMap::new(),
            min_frequency: 1,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
        });
        
        // Simple training corpus
        let corpus = "hello world hello there world wide web";
        tokenizer.train(corpus).unwrap();
        
        // Test encoding
        let tokens = tokenizer.encode("hello world");
        assert!(!tokens.is_empty());
        
        // Test decoding
        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty());
        
        // Check stats
        let stats = tokenizer.get_stats();
        assert!(stats.vocab_size > 0);
    }
    
    #[test]
    fn test_unicode_mapping() {
        let tokenizer = BpeTokenizer::new(BpeConfig::default());
        
        // Test that Unicode mapping is initialized
        assert!(!tokenizer.unicode_to_byte.is_empty());
        assert!(!tokenizer.byte_to_unicode.is_empty());
        
        // Test some basic characters
        assert!(tokenizer.unicode_to_byte.contains_key(&'a'));
        assert!(tokenizer.unicode_to_byte.contains_key(&' '));
    }
    
    #[test]
    fn test_special_tokens() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<unk>".to_string(), 0);
        special_tokens.insert("<pad>".to_string(), 1);
        
        let tokenizer = BpeTokenizer::new(BpeConfig {
            vocab_size: 1000,
            special_tokens,
            min_frequency: 1,
            unknown_token: "<unk>".to_string(),
            pad_token: "<pad>".to_string(),
            bos_token: "<bos>".to_string(),
            eos_token: "<eos>".to_string(),
        });
        
        // Test encoding with special tokens
        let tokens = tokenizer.encode("test");
        
        // Should include BOS and EOS tokens
        assert!(tokens.len() >= 2); // At least BOS and EOS
    }
    
    #[test]
    fn test_save_load() {
        let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
        
        // Train with simple corpus
        let corpus = "test training data for tokenizer";
        tokenizer.train(corpus).unwrap();
        
        // Save to temporary directory
        let temp_dir = std::env::temp_dir().join("test_tokenizer_save_load");
        let temp_path = temp_dir.to_str().unwrap();
        tokenizer.save(temp_path).unwrap();
        
        // Load from saved files
        let loaded_tokenizer = BpeTokenizer::load(temp_path).unwrap();
        
        // Check that they have the same vocab size
        assert_eq!(tokenizer.vocab_size(), loaded_tokenizer.vocab_size());
        
        // Clean up
        fs::remove_dir_all(temp_path).unwrap_or(());
    }
}
