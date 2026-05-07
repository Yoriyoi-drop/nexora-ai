//! Corpus Collector - Rust implementation
//! 
//! Collects and manages corpus data from various sources

use anyhow::Result;
use std::collections::HashMap;

use crate::DataEntry;

/// Corpus Collector for managing data entries
#[derive(Debug, Clone)]
pub struct CorpusCollector {
    source_type: String,
    entries: Vec<DataEntry>,
    capacity: usize,
}

impl CorpusCollector {
    /// Create a new corpus collector
    pub fn new(source_type: String, capacity: usize) -> Self {
        Self {
            source_type,
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Add a new entry to the corpus
    pub fn add_entry(&mut self, source_url: Option<String>, content: String, metadata: Option<HashMap<String, String>>) -> Result<()> {
        if self.entries.len() >= self.capacity {
            return Err(anyhow::anyhow!("Corpus collector capacity exceeded"));
        }
        
        let mut entry = DataEntry::new(content);
        
        if let Some(url) = source_url {
            entry = entry.with_source_url(url);
        }
        
        if let Some(meta) = metadata {
            for (key, value) in meta {
                entry = entry.with_metadata(key, value);
            }
        }
        
        self.entries.push(entry);
        Ok(())
    }
    
    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    /// Check if the corpus is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    
    /// Get the source type
    pub fn source_type(&self) -> &str {
        &self.source_type
    }
    
    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get all entries
    pub fn entries(&self) -> &[DataEntry] {
        &self.entries
    }
    
    /// Get entries by quality score threshold
    pub fn filter_by_quality(&self, min_score: f32) -> Vec<&DataEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.quality_score
                    .map(|score| score >= min_score)
                    .unwrap_or(false)
            })
            .collect()
    }
    
    /// Get entries by source URL pattern
    pub fn filter_by_source_pattern(&self, pattern: &str) -> Vec<&DataEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.source_url
                    .as_ref()
                    .map(|url| url.contains(pattern))
                    .unwrap_or(false)
            })
            .collect()
    }
    
    /// Get entries by metadata key-value pair
    pub fn filter_by_metadata(&self, key: &str, value: &str) -> Vec<&DataEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.metadata
                    .get(key)
                    .map(|v| v == value)
                    .unwrap_or(false)
            })
            .collect()
    }
    
    /// Get statistics about the corpus
    pub fn get_statistics(&self) -> CorpusStatistics {
        let total_chars: usize = self.entries.iter().map(|e| e.content.len()).sum();
        let total_words: usize = self.entries.iter()
            .map(|e| e.content.split_whitespace().count())
            .sum();
        
        let avg_quality = if self.entries.is_empty() {
            0.0
        } else {
            let scored_entries: f32 = self.entries.iter()
                .filter_map(|e| e.quality_score)
                .count() as f32;
            let total_score: f32 = self.entries.iter()
                .filter_map(|e| e.quality_score)
                .sum();
            
            if scored_entries > 0.0 {
                total_score / scored_entries
            } else {
                0.0
            }
        };
        
        CorpusStatistics {
            total_entries: self.entries.len(),
            total_chars,
            total_words,
            average_quality: avg_quality,
            source_type: self.source_type.clone(),
            capacity: self.capacity,
        }
    }
    
    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    
    /// Remove entries by quality threshold
    pub fn remove_by_quality(&mut self, min_score: f32) -> usize {
        let initial_len = self.entries.len();
        self.entries.retain(|entry| {
            entry.quality_score
                .map(|score| score >= min_score)
                .unwrap_or(true)
        });
        initial_len - self.entries.len()
    }
}

/// Statistics about the corpus
#[derive(Debug, Clone)]
pub struct CorpusStatistics {
    pub total_entries: usize,
    pub total_chars: usize,
    pub total_words: usize,
    pub average_quality: f32,
    pub source_type: String,
    pub capacity: usize,
}

impl Default for CorpusCollector {
    fn default() -> Self {
        Self::new("default".to_string(), 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_collector_basic() {
        let mut collector = CorpusCollector::new("test".to_string(), 10);
        
        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());
        
        collector.add_entry(
            Some("http://example.com".to_string()),
            "Test content".to_string(),
            None
        ).unwrap();
        
        assert_eq!(collector.len(), 1);
        assert!(!collector.is_empty());
    }

    #[test]
    fn test_quality_filtering() {
        let mut collector = CorpusCollector::new("test".to_string(), 10);
        
        let mut entry1 = DataEntry::new("Good content".to_string());
        entry1.quality_score = Some(0.8);
        collector.entries.push(entry1);
        
        let mut entry2 = DataEntry::new("Poor content".to_string());
        entry2.quality_score = Some(0.3);
        collector.entries.push(entry2);
        
        let high_quality = collector.filter_by_quality(0.5);
        assert_eq!(high_quality.len(), 1);
    }
}
