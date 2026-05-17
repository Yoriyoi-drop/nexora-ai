//! Tokenizer I/O - Rust implementation
//! 
//! Input/output operations for tokenizer

use std::io::{Read, Write, BufRead, BufReader};
use std::fs::File;
use std::path::Path;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use crate::tokenizer_core::TokenizerCore;

/// Tokenizer I/O operations
pub struct TokenizerIO;

impl TokenizerIO {
    /// Encode text to token IDs
    pub fn encode(tokenizer: &TokenizerCore, text: &str) -> Result<Vec<u32>> {
        tokenizer.tokenize(text)
    }
    
    /// Encode text to token IDs with output buffer
    pub fn encode_to_buffer(
        tokenizer: &TokenizerCore,
        text: &str,
        buffer: &mut [u32],
    ) -> Result<usize> {
        let tokens = tokenizer.tokenize(text)?;
        let count = tokens.len().min(buffer.len());
        
        for i in 0..count {
            buffer[i] = tokens[i];
        }
        
        Ok(count)
    }
    
    /// Decode token IDs to text
    pub fn decode(tokenizer: &TokenizerCore, token_ids: &[u32]) -> Result<String> {
        tokenizer.decode(token_ids)
    }
    
    /// Load tokenizer from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<TokenizerCore> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        TokenizerCore::load_from_json(&contents)
    }
    
    /// Save tokenizer to file
    pub fn save_to_file<P: AsRef<Path>>(tokenizer: &TokenizerCore, path: P) -> Result<()> {
        let json = tokenizer.save_to_json()?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    /// Load tokenizer from JSON string
    pub fn load_from_json(json_str: &str) -> Result<TokenizerCore> {
        TokenizerCore::load_from_json(json_str)
    }
    
    /// Save tokenizer to JSON string
    pub fn save_to_json(tokenizer: &TokenizerCore) -> Result<String> {
        tokenizer.save_to_json()
    }
    
    /// Encode text file to token IDs
    pub fn encode_file<P: AsRef<Path>>(
        tokenizer: &TokenizerCore,
        input_path: P,
        output_path: P,
    ) -> Result<()> {
        let input_file = File::open(input_path)?;
        let mut reader = BufReader::new(input_file);
        let output_file = File::create(output_path)?;
        let mut writer = std::io::BufWriter::new(output_file);
        
        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let tokens = tokenizer.tokenize(line.trim())?;
            
            // Write tokens as space-separated IDs
            for (i, token) in tokens.iter().enumerate() {
                if i > 0 {
                    writer.write_all(b" ")?;
                }
                writer.write_all(token.to_string().as_bytes())?;
            }
            writer.write_all(b"\n")?;
            
            line.clear();
        }
        
        Ok(())
    }
    
    /// Decode token IDs file to text
    pub fn decode_file<P: AsRef<Path>>(
        tokenizer: &TokenizerCore,
        input_path: P,
        output_path: P,
    ) -> Result<()> {
        let input_file = File::open(input_path)?;
        let mut reader = BufReader::new(input_file);
        let output_file = File::create(output_path)?;
        let mut writer = std::io::BufWriter::new(output_file);
        
        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let token_strs: Vec<&str> = line.trim().split_whitespace().collect();
            let mut token_ids = Vec::with_capacity(token_strs.len());
            
            for token_str in token_strs {
                if let Ok(id) = token_str.parse::<u32>() {
                    token_ids.push(id);
                }
            }
            
            let text = tokenizer.decode(&token_ids)?;
            writer.write_all(text.as_bytes())?;
            writer.write_all(b"\n")?;
            
            line.clear();
        }
        
        Ok(())
    }
    
    /// Batch encode multiple texts
    pub fn batch_encode(
        tokenizer: &TokenizerCore,
        texts: &[String],
    ) -> Result<Vec<Vec<u32>>> {
        let mut results = Vec::with_capacity(texts.len());
        
        for text in texts {
            let tokens = tokenizer.tokenize(text)?;
            results.push(tokens);
        }
        
        Ok(results)
    }
    
    /// Batch decode multiple token sequences
    pub fn batch_decode(
        tokenizer: &TokenizerCore,
        token_sequences: &[Vec<u32>],
    ) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(token_sequences.len());
        
        for tokens in token_sequences {
            let text = tokenizer.decode(tokens)?;
            results.push(text);
        }
        
        Ok(results)
    }
    
    /// Stream encode large text
    pub fn stream_encode<'a>(
        tokenizer: &'a TokenizerCore,
        reader: Box<dyn BufRead + 'a>,
    ) -> impl Iterator<Item = Result<Vec<u32>>> + 'a {
        TokenizerEncodeStream::new(tokenizer, reader)
    }
    
    /// Stream decode token IDs
    pub fn stream_decode<'a>(
        tokenizer: &'a TokenizerCore,
        reader: Box<dyn BufRead + 'a>,
    ) -> impl Iterator<Item = Result<String>> + 'a {
        TokenizerDecodeStream::new(tokenizer, reader)
    }
    
    /// Get tokenizer statistics
    pub fn get_stats(tokenizer: &TokenizerCore) -> Result<TokenizerStats> {
        let vocab_stats = tokenizer.get_vocab_stats();
        
        Ok(TokenizerStats {
            vocab_size: vocab_stats.vocab_size,
            merge_count: vocab_stats.merge_count,
            total_frequency: vocab_stats.total_frequency,
            average_frequency: vocab_stats.average_frequency,
        })
    }
    
    /// Validate tokenizer file
    pub fn validate_file<P: AsRef<Path>>(path: P) -> Result<bool> {
        let tokenizer = Self::load_from_file(path)?;
        tokenizer.validate().map(|_| true)
    }
    
    /// Compare two tokenizers
    pub fn compare(
        tokenizer1: &TokenizerCore,
        tokenizer2: &TokenizerCore,
    ) -> Result<TokenizerComparison> {
        let stats1 = Self::get_stats(tokenizer1)?;
        let stats2 = Self::get_stats(tokenizer2)?;
        
        let vocab_overlap = Self::calculate_vocab_overlap(tokenizer1, tokenizer2)?;
        
        Ok(TokenizerComparison {
            vocab_size_diff: stats1.vocab_size as i64 - stats2.vocab_size as i64,
            merge_count_diff: stats1.merge_count as i64 - stats2.merge_count as i64,
            vocab_overlap_ratio: vocab_overlap,
            stats1,
            stats2,
        })
    }
    
    /// Calculate vocabulary overlap between two tokenizers
    fn calculate_vocab_overlap(
        tokenizer1: &TokenizerCore,
        tokenizer2: &TokenizerCore,
    ) -> Result<f64> {
        let overlap = 0;
        
        // This is a simplified implementation
        // In practice, you'd need access to the vocabularies
        let stats1 = tokenizer1.get_vocab_stats();
        let stats2 = tokenizer2.get_vocab_stats();
        
        let total = stats1.vocab_size.max(stats2.vocab_size);
        
        if total > 0 {
            Ok(overlap as f64 / total as f64)
        } else {
            Ok(0.0)
        }
    }
}

/// Streaming encoder for large texts
struct TokenizerEncodeStream<'a> {
    tokenizer: &'a TokenizerCore,
    reader: Box<dyn BufRead + 'a>,
    buffer: String,
}

impl<'a> TokenizerEncodeStream<'a> {
    fn new(tokenizer: &'a TokenizerCore, reader: Box<dyn BufRead + 'a>) -> Self {
        Self {
            tokenizer,
            reader,
            buffer: String::new(),
        }
    }
}

impl<'a> Iterator for TokenizerEncodeStream<'a> {
    type Item = Result<Vec<u32>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();
        
        match self.reader.read_line(&mut self.buffer) {
            Ok(0) => None, // EOF
            Ok(_) => Some(self.tokenizer.tokenize(self.buffer.trim())),
            Err(e) => Some(Err(anyhow::anyhow!("Read error: {}", e))),
        }
    }
}

/// Streaming decoder for token sequences
struct TokenizerDecodeStream<'a> {
    tokenizer: &'a TokenizerCore,
    reader: Box<dyn BufRead + 'a>,
    buffer: String,
}

impl<'a> TokenizerDecodeStream<'a> {
    fn new(tokenizer: &'a TokenizerCore, reader: Box<dyn BufRead + 'a>) -> Self {
        Self {
            tokenizer,
            reader,
            buffer: String::new(),
        }
    }
}

impl<'a> Iterator for TokenizerDecodeStream<'a> {
    type Item = Result<String>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();
        
        match self.reader.read_line(&mut self.buffer) {
            Ok(0) => None, // EOF
            Ok(_) => {
                let token_strs: Vec<&str> = self.buffer.trim().split_whitespace().collect();
                let mut token_ids = Vec::with_capacity(token_strs.len());
                
                for token_str in token_strs {
                    if let Ok(id) = token_str.parse::<u32>() {
                        token_ids.push(id);
                    }
                }
                
                Some(self.tokenizer.decode(&token_ids))
            }
            Err(e) => Some(Err(anyhow::anyhow!("Read error: {}", e))),
        }
    }
}

/// Tokenizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerStats {
    pub vocab_size: usize,
    pub merge_count: usize,
    pub total_frequency: u64,
    pub average_frequency: f64,
}

/// Tokenizer comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerComparison {
    pub vocab_size_diff: i64,
    pub merge_count_diff: i64,
    pub vocab_overlap_ratio: f64,
    pub stats1: TokenizerStats,
    pub stats2: TokenizerStats,
}

/// Convenience functions
pub fn encode_text(tokenizer: &TokenizerCore, text: &str) -> Result<Vec<u32>> {
    TokenizerIO::encode(tokenizer, text)
}

pub fn decode_tokens(tokenizer: &TokenizerCore, token_ids: &[u32]) -> Result<String> {
    TokenizerIO::decode(tokenizer, token_ids)
}

pub fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<TokenizerCore> {
    TokenizerIO::load_from_file(path)
}

pub fn save_tokenizer<P: AsRef<Path>>(tokenizer: &TokenizerCore, path: P) -> Result<()> {
    TokenizerIO::save_to_file(tokenizer, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_core::TokenizerCore;
    use std::io::Cursor;
    
    #[test]
    fn test_encode_decode() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        // Add some basic tokens
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let text = "hello";
        let tokens = TokenizerIO::encode(&tokenizer, text).unwrap();
        let decoded = TokenizerIO::decode(&tokenizer, &tokens).unwrap();
        
        assert_eq!(decoded, text);
    }
    
    #[test]
    fn test_encode_to_buffer() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let text = "hello";
        let mut buffer = [0u32; 10];
        let count = TokenizerIO::encode_to_buffer(&tokenizer, text, &mut buffer).unwrap();
        
        assert_eq!(count, 5);
        assert_eq!(buffer[0], tokenizer.get_token_id("h").unwrap());
        assert_eq!(buffer[4], tokenizer.get_token_id("o").unwrap());
    }
    
    #[test]
    fn test_serialization() {
        let mut tokenizer = TokenizerCore::with_default_config();
        tokenizer.add_token("test", 1).unwrap();
        
        let json = TokenizerIO::save_to_json(&tokenizer).unwrap();
        let loaded = TokenizerIO::load_from_json(&json).unwrap();
        
        assert!(loaded.has_token("test"));
    }
    
    #[test]
    fn test_batch_operations() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let texts = vec!["hello".to_string(), "hell".to_string()];
        let encoded = TokenizerIO::batch_encode(&tokenizer, &texts).unwrap();
        let decoded = TokenizerIO::batch_decode(&tokenizer, &encoded).unwrap();
        
        assert_eq!(decoded, texts);
    }
    
    #[test]
    fn test_stream_encode() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        let input = "hello\nworld\n";
        let cursor = Cursor::new(input.as_bytes());
        let reader = Box::new(BufReader::new(cursor));
        
        let stream = TokenizerIO::stream_encode(&tokenizer, reader);
        let results: Vec<Result<Vec<u32>>> = stream.collect();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].as_ref().unwrap().len() > 0);
        assert!(results[1].as_ref().unwrap().len() > 0);
    }
    
    #[test]
    fn test_stream_decode() {
        let mut tokenizer = TokenizerCore::with_default_config();
        
        tokenizer.add_token("h", 1).unwrap();
        tokenizer.add_token("e", 1).unwrap();
        tokenizer.add_token("l", 1).unwrap();
        tokenizer.add_token("o", 1).unwrap();
        
        // Create token ID sequences
        let hello_tokens = tokenizer.tokenize("hello").unwrap();
        let world_tokens = tokenizer.tokenize("world").unwrap();
        
        let input = format!("{}\n{}\n", 
            hello_tokens.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" "),
            world_tokens.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" ")
        );
        
        let cursor = Cursor::new(input.as_bytes());
        let reader = Box::new(BufReader::new(cursor));
        
        let stream = TokenizerIO::stream_decode(&tokenizer, reader);
        let results: Vec<Result<String>> = stream.collect();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref().unwrap(), "hello");
        assert_eq!(results[1].as_ref().unwrap(), "world");
    }
    
    #[test]
    fn test_get_stats() {
        let tokenizer = TokenizerCore::with_default_config();
        let stats = TokenizerIO::get_stats(&tokenizer).unwrap();
        
        assert!(stats.vocab_size > 0); // Should have special tokens
        assert!(stats.average_frequency >= 0.0);
    }
}
