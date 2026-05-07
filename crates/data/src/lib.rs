//! Data Processing Components for Nexora AI
//! 
//! Rust implementation of data pipeline components

use std::collections::HashMap;

pub mod corpus_collector;
pub mod deduplicator;
pub mod domain_splitter;
pub mod pipeline;
pub mod quality_classifier;
pub mod quality_filter;
pub mod synthetic_generator;
pub mod token_mixer;
pub mod token_packer;

pub use corpus_collector::*;
pub use deduplicator::*;
pub use domain_splitter::*;
pub use pipeline::*;
pub use quality_classifier::*;
pub use quality_filter::*;
pub use synthetic_generator::*;
pub use token_mixer::*;
pub use token_packer::*;

/// Common data structures for data processing
#[derive(Debug, Clone)]
pub struct DataEntry {
    pub id: String,
    pub source_url: Option<String>,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
    pub quality_score: Option<f32>,
}

impl DataEntry {
    pub fn new(content: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            source_url: None,
            content,
            metadata: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            quality_score: None,
        }
    }
    
    pub fn with_source_url(mut self, url: String) -> Self {
        self.source_url = Some(url);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.quality_score = Some(score);
        self
    }
}
