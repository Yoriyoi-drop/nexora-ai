//! Token Packer - Rust implementation
//! 
//! Packs tokens efficiently for storage and transmission

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

use crate::DataEntry;

/// Token packer for efficient token storage and compression
#[derive(Debug, Clone)]
pub struct TokenPacker {
    config: PackerConfig,
    vocabulary: HashMap<String, u32>,
    reverse_vocabulary: HashMap<u32, String>,
    compression_stats: CompressionStats,
}

/// Packer configuration
#[derive(Debug, Clone)]
pub struct PackerConfig {
    pub max_vocabulary_size: usize,
    pub min_token_frequency: usize,
    pub enable_compression: bool,
    pub compression_level: CompressionLevel,
    pub chunk_size: usize,
    pub preserve_order: bool,
}

/// Compression levels
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionLevel {
    None,
    Light,
    Medium,
    Heavy,
}

/// Packed token data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedTokens {
    pub token_ids: Vec<u32>,
    pub metadata: PackedMetadata,
    pub compression_info: Option<CompressionInfo>,
}

/// Metadata for packed tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedMetadata {
    pub original_length: usize,
    pub vocabulary_size: usize,
    pub unique_tokens: usize,
    pub packing_version: String,
    pub created_at: u64,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    pub algorithm: String,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f32,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub total_entries_packed: usize,
    pub total_tokens_processed: usize,
    pub average_compression_ratio: f32,
    pub vocabulary_hits: usize,
    pub vocabulary_misses: usize,
}

impl TokenPacker {
    /// Create new token packer
    pub fn new(config: PackerConfig) -> Self {
        Self {
            config,
            vocabulary: HashMap::new(),
            reverse_vocabulary: HashMap::new(),
            compression_stats: CompressionStats::default(),
        }
    }
    
    /// Build vocabulary from data entries
    pub fn build_vocabulary(&mut self, entries: &[DataEntry]) -> Result<()> {
        let mut token_counts = HashMap::new();
        
        // Count token frequencies
        for entry in entries {
            let tokens = self.tokenize(&entry.content);
            for token in tokens {
                *token_counts.entry(token).or_insert(0) += 1;
            }
        }
        
        // Filter by minimum frequency and sort by count
        let mut filtered_tokens: Vec<(String, usize)> = token_counts.into_iter()
            .filter(|(_, count)| *count >= self.config.min_token_frequency)
            .collect();
        
        filtered_tokens.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Limit vocabulary size
        if filtered_tokens.len() > self.config.max_vocabulary_size {
            filtered_tokens.truncate(self.config.max_vocabulary_size);
        }
        
        // Build vocabulary mappings
        self.vocabulary.clear();
        self.reverse_vocabulary.clear();
        
        // Add special tokens
        self.vocabulary.insert("<UNK>".to_string(), 0);
        self.vocabulary.insert("<PAD>".to_string(), 1);
        self.vocabulary.insert("<BOS>".to_string(), 2);
        self.vocabulary.insert("<EOS>".to_string(), 3);
        
        self.reverse_vocabulary.insert(0, "<UNK>".to_string());
        self.reverse_vocabulary.insert(1, "<PAD>".to_string());
        self.reverse_vocabulary.insert(2, "<BOS>".to_string());
        self.reverse_vocabulary.insert(3, "<EOS>".to_string());
        
        // Add regular tokens
        let mut token_id = 4u32;
        for (token, _) in filtered_tokens {
            self.vocabulary.insert(token.clone(), token_id);
            self.reverse_vocabulary.insert(token_id, token);
            token_id += 1;
        }
        
        Ok(())
    }
    
    /// Tokenize text into tokens
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Simple whitespace tokenization with basic preprocessing
        text.split_whitespace()
            .map(|token| {
                // Basic normalization
                token.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect::<String>()
            })
            .filter(|token| !token.is_empty())
            .collect()
    }
    
    /// Convert tokens to IDs
    fn tokens_to_ids(&self, tokens: &[String]) -> Vec<u32> {
        tokens.iter()
            .map(|token| {
                *self.vocabulary.get(token).unwrap_or(&0) // <UNK> token
            })
            .collect()
    }
    
    /// Convert IDs back to tokens
    fn ids_to_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|id| {
                self.reverse_vocabulary.get(id)
                    .cloned()
                    .unwrap_or_else(|| "<UNK>".to_string())
            })
            .collect()
    }
    
    /// Pack tokens from data entry
    pub fn pack_entry(&mut self, entry: &DataEntry) -> Result<PackedTokens> {
        let tokens = self.tokenize(&entry.content);
        let token_ids = self.tokens_to_ids(&tokens);
        
        // Update statistics
        self.compression_stats.total_entries_packed += 1;
        self.compression_stats.total_tokens_processed += tokens.len();
        
        let unique_tokens = tokens.iter().collect::<HashSet<_>>().len();
        
        let metadata = PackedMetadata {
            original_length: tokens.len(),
            vocabulary_size: self.vocabulary.len(),
            unique_tokens,
            packing_version: "1.0".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after epoch")
                .as_secs(),
        };
        
        let compression_info = if self.config.enable_compression {
            Some(self.compress_token_ids(&token_ids)?)
        } else {
            None
        };
        
        Ok(PackedTokens {
            token_ids,
            metadata,
            compression_info,
        })
    }
    
    /// Unpack tokens back to text
    pub fn unpack_tokens(&self, packed: &PackedTokens) -> Result<String> {
        let tokens = self.ids_to_tokens(&packed.token_ids);
        Ok(tokens.join(" "))
    }
    
    /// Pack multiple entries
    pub fn pack_batch(&mut self, entries: &[DataEntry]) -> Result<Vec<PackedTokens>> {
        let mut packed_entries = Vec::new();
        
        for entry in entries {
            let packed = self.pack_entry(entry)?;
            packed_entries.push(packed);
        }
        
        Ok(packed_entries)
    }
    
    /// Compress token IDs using various algorithms
    fn compress_token_ids(&self, token_ids: &[u32]) -> Result<CompressionInfo> {
        let original_size = token_ids.len() * std::mem::size_of::<u32>();
        
        match self.config.compression_level {
            CompressionLevel::None => Ok(CompressionInfo {
                algorithm: "none".to_string(),
                original_size,
                compressed_size: original_size,
                compression_ratio: 1.0,
            }),
            CompressionLevel::Light => self.light_compression(token_ids, original_size),
            CompressionLevel::Medium => self.medium_compression(token_ids, original_size),
            CompressionLevel::Heavy => self.heavy_compression(token_ids, original_size),
        }
    }
    
    /// Light compression - simple run-length encoding
    fn light_compression(&self, token_ids: &[u32], original_size: usize) -> Result<CompressionInfo> {
        let mut compressed = Vec::new();
        let mut current_run = (token_ids[0], 1u32);
        
        for &token_id in &token_ids[1..] {
            if token_id == current_run.0 && current_run.1 < 255 {
                current_run.1 += 1;
            } else {
                compressed.push(current_run.0);
                compressed.push(current_run.1);
                current_run = (token_id, 1);
            }
        }
        compressed.push(current_run.0);
        compressed.push(current_run.1);
        
        let compressed_size = compressed.len() * std::mem::size_of::<u32>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        Ok(CompressionInfo {
            algorithm: "run_length".to_string(),
            original_size,
            compressed_size,
            compression_ratio,
        })
    }
    
    /// Medium compression - delta encoding
    fn medium_compression(&self, token_ids: &[u32], original_size: usize) -> Result<CompressionInfo> {
        let mut compressed = Vec::new();
        let mut prev_token = 0u32;
        
        for &token_id in token_ids {
            let delta = token_id.wrapping_sub(prev_token);
            compressed.push(delta);
            prev_token = token_id;
        }
        
        let compressed_size = compressed.len() * std::mem::size_of::<u32>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        Ok(CompressionInfo {
            algorithm: "delta".to_string(),
            original_size,
            compressed_size,
            compression_ratio,
        })
    }
    
    /// Heavy compression - variable byte encoding
    fn heavy_compression(&self, token_ids: &[u32], original_size: usize) -> Result<CompressionInfo> {
        let mut compressed = Vec::new();
        
        for &token_id in token_ids {
            let mut value = token_id;
            loop {
                let mut byte = (value & 0x7F) as u8;
                value >>= 7;
                
                if value != 0 {
                    byte |= 0x80; // Set continuation bit
                }
                
                compressed.push(byte);
                
                if value == 0 {
                    break;
                }
            }
        }
        
        let compressed_size = compressed.len();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        Ok(CompressionInfo {
            algorithm: "variable_byte".to_string(),
            original_size,
            compressed_size,
            compression_ratio,
        })
    }
    
    /// Get vocabulary statistics
    pub fn get_vocabulary_stats(&self) -> VocabularyStats {
        VocabularyStats {
            total_tokens: self.vocabulary.len(),
            special_tokens: 4, // <UNK>, <PAD>, <BOS>, <EOS>
            regular_tokens: self.vocabulary.len() - 4,
            compression_stats: self.compression_stats.clone(),
        }
    }
    
    /// Get token by ID
    pub fn get_token(&self, token_id: u32) -> Option<&str> {
        self.reverse_vocabulary.get(&token_id).map(|s| s.as_str())
    }
    
    /// Get ID by token
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.vocabulary.get(token).cloned()
    }
    
    /// Export vocabulary
    pub fn export_vocabulary(&self) -> HashMap<String, u32> {
        self.vocabulary.clone()
    }
    
    /// Import vocabulary
    pub fn import_vocabulary(&mut self, vocab: HashMap<String, u32>) -> Result<()> {
        // Rebuild reverse vocabulary
        self.reverse_vocabulary.clear();
        for (token, id) in &vocab {
            self.reverse_vocabulary.insert(*id, token.clone());
        }
        
        self.vocabulary = vocab;
        Ok(())
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.compression_stats = CompressionStats::default();
    }
}

/// Vocabulary statistics
#[derive(Debug, Clone)]
pub struct VocabularyStats {
    pub total_tokens: usize,
    pub special_tokens: usize,
    pub regular_tokens: usize,
    pub compression_stats: CompressionStats,
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            max_vocabulary_size: 50000,
            min_token_frequency: 2,
            enable_compression: true,
            compression_level: CompressionLevel::Medium,
            chunk_size: 1024,
            preserve_order: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_packing() {
        let config = PackerConfig::default();
        let mut packer = TokenPacker::new(config);
        
        let entries = vec![
            DataEntry::new("Hello world test data".to_string()),
            DataEntry::new("Processing tokens efficiently".to_string()),
        ];
        
        // Build vocabulary
        packer.build_vocabulary(&entries).unwrap();
        
        // Pack entry
        let packed = packer.pack_entry(&entries[0]).unwrap();
        assert!(!packed.token_ids.is_empty());
        
        // Unpack tokens
        let unpacked = packer.unpack_tokens(&packed).unwrap();
        assert!(!unpacked.is_empty());
    }

    #[test]
    fn test_vocabulary_building() {
        let config = PackerConfig::default();
        let mut packer = TokenPacker::new(config);
        
        let entries = vec![
            DataEntry::new("test data processing".to_string()),
            DataEntry::new("test token packing".to_string()),
            DataEntry::new("data token test".to_string()),
        ];
        
        packer.build_vocabulary(&entries).unwrap();
        
        // Check that common tokens are in vocabulary
        assert!(packer.get_token_id("test").is_some());
        assert!(packer.get_token_id("data").is_some());
        assert!(packer.get_token_id("token").is_some());
        
        // Check reverse mapping
        if let Some(test_id) = packer.get_token_id("test") {
            assert_eq!(packer.get_token(test_id), Some("test"));
        }
    }

    #[test]
    fn test_compression() {
        let config = PackerConfig {
            enable_compression: true,
            compression_level: CompressionLevel::Light,
            ..Default::default()
        };
        
        let mut packer = TokenPacker::new(config);
        
        let entries = vec![DataEntry::new("test data test data test data".to_string())];
        packer.build_vocabulary(&entries).unwrap();
        
        let packed = packer.pack_entry(&entries[0]).unwrap();
        
        // Should have compression info
        assert!(packed.compression_info.is_some());
        
        let compression_info = packed.compression_info.unwrap();
        assert_eq!(compression_info.algorithm, "run_length");
        assert!(compression_info.compression_ratio >= 1.0);
    }

    #[test]
    fn test_batch_packing() {
        let config = PackerConfig::default();
        let mut packer = TokenPacker::new(config);
        
        let entries = vec![
            DataEntry::new("First entry".to_string()),
            DataEntry::new("Second entry".to_string()),
            DataEntry::new("Third entry".to_string()),
        ];
        
        packer.build_vocabulary(&entries).unwrap();
        
        let packed_batch = packer.pack_batch(&entries).unwrap();
        assert_eq!(packed_batch.len(), 3);
        
        for packed in packed_batch {
            assert!(!packed.token_ids.is_empty());
        }
    }
}
