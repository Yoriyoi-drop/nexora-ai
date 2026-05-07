//! Tokenizer Backend Trait
//!
//! Defines the interface for tokenizer implementations

use async_trait::async_trait;
use crate::FoundationResult;

/// Tokenization result
#[derive(Debug, Clone)]
pub struct TokenizationResult {
    pub tokens: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub metadata: TokenizationMetadata,
}

#[derive(Debug, Clone)]
pub struct TokenizationMetadata {
    pub token_count: usize,
    pub truncated: bool,
    pub special_tokens: Vec<usize>,
}

/// Detokenization result
#[derive(Debug, Clone)]
pub struct DetokenizationResult {
    pub text: String,
    pub metadata: DetokenizationMetadata,
}

#[derive(Debug, Clone)]
pub struct DetokenizationMetadata {
    pub token_count: usize,
    pub skipped_tokens: Vec<usize>,
}

/// Tokenizer configuration
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub max_length: usize,
    pub padding: bool,
    pub truncation: bool,
}

/// Core tokenizer backend trait
#[async_trait]
pub trait TokenizerBackend: Send + Sync {
    /// Encode text to tokens
    async fn encode(&self, text: &str, config: &TokenizerConfig) -> FoundationResult<TokenizationResult>;
    
    /// Decode tokens to text
    async fn decode(&self, tokens: &[u32]) -> FoundationResult<DetokenizationResult>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get special tokens
    fn special_tokens(&self) -> HashMap<String, u32>;
    
    /// Tokenize batch of texts
    async fn encode_batch(&self, texts: &[String], config: &TokenizerConfig) -> FoundationResult<Vec<TokenizationResult>>;
}

use std::collections::HashMap;
