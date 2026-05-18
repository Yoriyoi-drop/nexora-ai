//! Context Compression - Memory optimization melalui compression
//! 
//! Implementasi context compression untuk menghemat memory usage

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{debug, trace};

/// Compressed context representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedContext {
    pub original_length: usize,
    pub compressed_length: usize,
    pub compression_ratio: f32,
    pub compression_method: CompressionMethod,
    pub data: Vec<u8>,
    pub checksum: u32,
}

impl CompressedContext {
    pub fn new(
        original_length: usize,
        compressed_length: usize,
        compression_method: CompressionMethod,
        data: Vec<u8>,
    ) -> Self {
        let compression_ratio = if original_length > 0 {
            compressed_length as f32 / original_length as f32
        } else {
            1.0
        };
        
        let checksum = Self::calculate_checksum(&data);
        
        Self {
            original_length,
            compressed_length,
            compression_ratio,
            compression_method,
            data,
            checksum,
        }
    }
    
    /// Verify checksum
    pub fn is_valid(&self) -> bool {
        self.checksum == Self::calculate_checksum(&self.data)
    }
    
    /// Calculate simple checksum
    fn calculate_checksum(data: &[u8]) -> u32 {
        data.iter().fold(0u32, |acc, &byte| {
            acc.wrapping_mul(31).wrapping_add(byte as u32)
        })
    }
}

/// Compression method enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionMethod {
    None,       // No compression
    Simple,     // Simple word-based compression
    RunLength,  // Run-length encoding
    Dictionary, // Dictionary-based compression
    Semantic,   // Semantic compression (summarization)
}

/// Context compressor dengan berbagai metode
#[derive(Debug)]
pub struct ContextCompressor {
    method: CompressionMethod,
    dictionary: HashMap<String, u16>,
    reverse_dictionary: HashMap<u16, String>,
    next_dict_id: u16,
    compression_stats: CompressionStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_compressed: usize,
    pub total_original: usize,
    pub avg_compression_ratio: f32,
    pub method_usage: HashMap<CompressionMethod, usize>,
}

impl ContextCompressor {
    pub fn new() -> Self {
        Self {
            method: CompressionMethod::Simple,
            dictionary: HashMap::new(),
            reverse_dictionary: HashMap::new(),
            next_dict_id: 0,
            compression_stats: CompressionStats::default(),
        }
    }
    
    pub fn with_method(mut self, method: CompressionMethod) -> Self {
        self.method = method;
        self
    }
    
    /// Compress context string
    pub async fn compress(&mut self, context: &str) -> Result<CompressedContext> {
        trace!("Compressing context of length: {}", context.len());
        
        let (compressed_data, method) = match self.method {
            CompressionMethod::None => {
                (context.as_bytes().to_vec(), CompressionMethod::None)
            }
            CompressionMethod::Simple => {
                (self.simple_compress(context)?, CompressionMethod::Simple)
            }
            CompressionMethod::RunLength => {
                (self.run_length_compress(context)?, CompressionMethod::RunLength)
            }
            CompressionMethod::Dictionary => {
                (self.dictionary_compress(context)?, CompressionMethod::Dictionary)
            }
            CompressionMethod::Semantic => {
                (self.semantic_compress(context)?, CompressionMethod::Semantic)
            }
        };
        
        let compressed = CompressedContext::new(
            context.len(),
            compressed_data.len(),
            method,
            compressed_data,
        );
        
        // Update statistics
        self.update_stats(&compressed);
        
        debug!("Compressed context: {} -> {} bytes (ratio: {:.2})", 
               compressed.original_length, 
               compressed.compressed_length, 
               compressed.compression_ratio);
        
        Ok(compressed)
    }
    
    /// Decompress context
    pub async fn decompress(&self, compressed: &CompressedContext) -> Result<String> {
        trace!("Decompressing context");
        
        if !compressed.is_valid() {
            return Err(anyhow::anyhow!("Invalid compressed context checksum"));
        }
        
        let decompressed = match compressed.compression_method {
            CompressionMethod::None => {
                String::from_utf8(compressed.data.clone())?
            }
            CompressionMethod::Simple => {
                self.simple_decompress(&compressed.data)?
            }
            CompressionMethod::RunLength => {
                self.run_length_decompress(&compressed.data)?
            }
            CompressionMethod::Dictionary => {
                self.dictionary_decompress(&compressed.data)?
            }
            CompressionMethod::Semantic => {
                self.semantic_decompress(&compressed.data)?
            }
        };
        
        debug!("Decompressed context: {} bytes", decompressed.len());
        Ok(decompressed)
    }
    
    /// Simple word-based compression
    fn simple_compress(&mut self, text: &str) -> Result<Vec<u8>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut compressed = Vec::with_capacity(words.len());
        
        for word in words {
            if word.len() <= 255 {
                compressed.push(word.len() as u8);
                compressed.extend_from_slice(word.as_bytes());
            } else {
                // Handle very long words
                compressed.push(255);
                compressed.extend_from_slice(&word[..255].as_bytes());
            }
        }
        
        Ok(compressed)
    }
    
    fn simple_decompress(&self, data: &[u8]) -> Result<String> {
        let mut decompressed = String::new();
        let mut i = 0;
        
        while i < data.len() {
            let word_len = data[i] as usize;
            i += 1;
            
            if i + word_len > data.len() {
                return Err(anyhow::anyhow!("Invalid compressed data"));
            }
            
            let word_bytes = &data[i..i + word_len];
            let word = String::from_utf8(word_bytes.to_vec())?;
            decompressed.push_str(&word);
            
            if i + word_len < data.len() {
                decompressed.push(' ');
            }
            
            i += word_len;
        }
        
        Ok(decompressed)
    }
    
    /// Run-length encoding compression
    fn run_length_compress(&self, text: &str) -> Result<Vec<u8>> {
        let bytes = text.as_bytes();
        let mut compressed = Vec::with_capacity(bytes.len());
        
        if bytes.is_empty() {
            return Ok(compressed);
        }
        
        let mut current_byte = bytes[0];
        let mut count = 1u8;
        
        for &byte in &bytes[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }
        
        // Add last run
        compressed.push(count);
        compressed.push(current_byte);
        
        Ok(compressed)
    }
    
    fn run_length_decompress(&self, data: &[u8]) -> Result<String> {
        if data.len() % 2 != 0 {
            return Err(anyhow::anyhow!("Invalid RLE data length"));
        }
        
        let mut decompressed = Vec::with_capacity(data.len() / 2);
        
        for chunk in data.chunks(2) {
            let count = chunk[0];
            let byte = chunk[1];
            
            for _ in 0..count {
                decompressed.push(byte);
            }
        }
        
        Ok(String::from_utf8(decompressed)?)
    }
    
    /// Dictionary-based compression
    fn dictionary_compress(&mut self, text: &str) -> Result<Vec<u8>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut compressed = Vec::with_capacity(words.len());
        
        for word in words {
            let dict_id = if let Some(&id) = self.dictionary.get(word) {
                id
            } else {
                // Add to dictionary
                let id = self.next_dict_id;
                self.dictionary.insert(word.to_string(), id);
                self.reverse_dictionary.insert(id, word.to_string());
                self.next_dict_id = self.next_dict_id.wrapping_add(1);
                id
            };
            
            // Store dictionary ID as 2 bytes
            compressed.push((dict_id >> 8) as u8);
            compressed.push((dict_id & 0xFF) as u8);
        }
        
        Ok(compressed)
    }
    
    fn dictionary_decompress(&self, data: &[u8]) -> Result<String> {
        if data.len() % 2 != 0 {
            return Err(anyhow::anyhow!("Invalid dictionary data length"));
        }
        
        let mut decompressed = String::new();
        let num_chunks = data.len() / 2;
        
        for (i, chunk) in data.chunks(2).enumerate() {
            let dict_id = ((chunk[0] as u16) << 8) | (chunk[1] as u16);
            
            if let Some(word) = self.reverse_dictionary.get(&dict_id) {
                decompressed.push_str(word);
                if i < num_chunks - 1 {
                    decompressed.push(' ');
                }
            } else {
                return Err(anyhow::anyhow!("Dictionary ID {} not found", dict_id));
            }
        }
        
        Ok(decompressed)
    }
    
    /// Semantic compression (simplified summarization)
    fn semantic_compress(&self, text: &str) -> Result<Vec<u8>> {
        // Very simple semantic compression: extract key sentences
        let sentences: Vec<&str> = text.split_terminator(&['.', '!', '?']).collect();
        let mut important_sentences = Vec::new();
        
        for sentence in &sentences {
            let sentence = sentence.trim();
            if sentence.len() > 10 { // Filter out very short sentences
                important_sentences.push(sentence.to_string());
            }
        }
        
        // If no important sentences, keep first sentence
        if important_sentences.is_empty() && !sentences.is_empty() {
            important_sentences.push(sentences[0].trim().to_string());
        }
        
        let compressed = important_sentences.join(". ").as_bytes().to_vec();
        Ok(compressed)
    }
    
    fn semantic_decompress(&self, data: &[u8]) -> Result<String> {
        Ok(String::from_utf8(data.to_vec())?)
    }
    
    /// Update compression statistics
    fn update_stats(&mut self, compressed: &CompressedContext) {
        self.compression_stats.total_compressed += compressed.compressed_length;
        self.compression_stats.total_original += compressed.original_length;
        
        if self.compression_stats.total_original > 0 {
            self.compression_stats.avg_compression_ratio = 
                self.compression_stats.total_compressed as f32 / 
                self.compression_stats.total_original as f32;
        }
        
        *self.compression_stats.method_usage
            .entry(compressed.compression_method)
            .or_insert(0) += 1;
    }
    
    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.compression_stats
    }
    
    /// Get compression ratio
    pub fn get_compression_ratio(&self) -> f32 {
        self.compression_stats.avg_compression_ratio
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.compression_stats = CompressionStats::default();
    }
    
    /// Change compression method
    pub fn set_method(&mut self, method: CompressionMethod) {
        self.method = method;
        debug!("Changed compression method to {:?}", method);
    }
    
    /// Clear dictionary
    pub fn clear_dictionary(&mut self) {
        self.dictionary.clear();
        self.reverse_dictionary.clear();
        self.next_dict_id = 0;
        debug!("Dictionary cleared");
    }
    
    /// Get dictionary size
    pub fn dictionary_size(&self) -> usize {
        self.dictionary.len()
    }
}

impl Default for ContextCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_no_compression() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::None);
        
        let text = "Hello, world!";
        let compressed = compressor.compress(text).await.unwrap();
        
        assert_eq!(compressed.compression_method, CompressionMethod::None);
        assert_eq!(compressed.compressed_length, text.len());
        assert_eq!(compressed.compression_ratio, 1.0);
        
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        assert_eq!(decompressed, text);
    }
    
    #[tokio::test]
    async fn test_simple_compression() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::Simple);
        
        let text = "Hello world this is a test";
        let compressed = compressor.compress(text).await.unwrap();
        
        assert_eq!(compressed.compression_method, CompressionMethod::Simple);
        
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        assert_eq!(decompressed, text);
    }
    
    #[tokio::test]
    async fn test_run_length_compression() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::RunLength);
        
        let text = "AAAAAAAABBBCCCCCCCC"; // Good for RLE
        let compressed = compressor.compress(text).await.unwrap();
        
        assert_eq!(compressed.compression_method, CompressionMethod::RunLength);
        assert!(compressed.compressed_length < text.len());
        
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        assert_eq!(decompressed, text);
    }
    
    #[tokio::test]
    async fn test_dictionary_compression() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::Dictionary);
        
        let text = "hello world hello test world";
        let compressed = compressor.compress(text).await.unwrap();
        
        assert_eq!(compressed.compression_method, CompressionMethod::Dictionary);
        assert_eq!(compressor.dictionary_size(), 3); // hello, world, test
        
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        assert_eq!(decompressed, text);
    }
    
    #[tokio::test]
    async fn test_semantic_compression() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::Semantic);
        
        let text = "This is important. This is not important. This is critical.";
        let compressed = compressor.compress(text).await.unwrap();
        
        assert_eq!(compressed.compression_method, CompressionMethod::Semantic);
        assert!(compressed.compressed_length < text.len());
        
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        // Semantic compression is lossy, so we check if it contains important parts
        assert!(decompressed.contains("important") || decompressed.contains("critical"));
    }
    
    #[tokio::test]
    async fn test_compression_stats() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::None);
        
        let text1 = "Hello world";
        let text2 = "AAAAAA BBBB";
        
        compressor.compress(text1).await.unwrap();
        compressor.compress(text2).await.unwrap();
        
        let stats = compressor.get_stats();
        assert_eq!(stats.total_original, text1.len() + text2.len());
        assert_eq!(stats.avg_compression_ratio, 1.0);
        assert!(stats.method_usage.len() > 0);
    }
    
    #[tokio::test]
    async fn test_compressed_context_validation() {
        let mut compressor = ContextCompressor::new();
        
        let text = "Hello world";
        let compressed = compressor.compress(text).await.unwrap();
        
        assert!(compressed.is_valid());
        
        // Corrupt the data
        let mut corrupted = compressed.clone();
        corrupted.data[0] = !corrupted.data[0];
        assert!(!corrupted.is_valid());
    }
    
    #[tokio::test]
    async fn test_compression_large_text() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::None);
        
        // Create large text
        let large_text = "This is a very long text that should be compressed. ".repeat(100);
        
        let compressed = compressor.compress(&large_text).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(large_text, decompressed);
        assert!(compressed.is_valid());
    }
    
    #[tokio::test]
    async fn test_compression_empty_string() {
        let mut compressor = ContextCompressor::new();
        
        let empty_text = "";
        let compressed = compressor.compress(empty_text).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(empty_text, decompressed);
        assert!(compressed.is_valid());
    }
    
    #[tokio::test]
    async fn test_compression_unicode_text() {
        let mut compressor = ContextCompressor::new();
        
        let unicode_text = "Hello 世界 🌍 Café naïve résumé";
        let compressed = compressor.compress(unicode_text).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(unicode_text, decompressed);
        assert!(compressed.is_valid());
    }
    
    #[tokio::test]
    async fn test_compression_special_characters() {
        let mut compressor = ContextCompressor::new();
        
        let special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let compressed = compressor.compress(special_text).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(special_text, decompressed);
        assert!(compressed.is_valid());
    }
    
    #[tokio::test]
    async fn test_compression_newlines_and_whitespace() {
        let mut compressor = ContextCompressor::new().with_method(CompressionMethod::None);
        
        let whitespace_text = "Hello\n\nWorld\t\t  \r\nTest";
        let compressed = compressor.compress(whitespace_text).await.unwrap();
        let decompressed = compressor.decompress(&compressed).await.unwrap();
        
        assert_eq!(whitespace_text, decompressed);
        assert!(compressed.is_valid());
    }
    
    #[tokio::test]
    async fn test_compression_error_handling() {
        let compressor = ContextCompressor::new();
        
        // Test decompressing invalid data
        let invalid_data = CompressedContext::new(
            100, // original_length
            4,  // compressed_length
            CompressionMethod::Simple,
            vec![1, 2, 3, 4], // Too short to be valid
        );
        
        let result = compressor.decompress(&invalid_data).await;
        assert!(result.is_err());
    }
}
