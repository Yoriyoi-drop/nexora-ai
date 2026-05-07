//! Quality Filter - Rust implementation
//! 
//! Filters data entries based on quality scores and criteria

use anyhow::Result;
use std::collections::HashMap;

use crate::{DataEntry, QualityClassifier, QualityConfig, QualityScore};

/// Quality filter with configurable criteria
pub struct QualityFilter {
    classifier: QualityClassifier,
    criteria: QualityCriteria,
    statistics: FilterStatistics,
}

/// Quality filtering criteria
pub struct QualityCriteria {
    pub min_overall_score: f32,
    pub min_algorithm_scores: HashMap<String, f32>,
    pub max_content_length: Option<usize>,
    pub min_content_length: Option<usize>,
    pub required_languages: Vec<String>,
    pub forbidden_languages: Vec<String>,
    pub allowed_sources: Vec<String>,
    pub blocked_sources: Vec<String>,
    pub custom_filters: Vec<Box<dyn CustomFilter>>,
}

/// Custom filter trait
pub trait CustomFilter: Send + Sync {
    fn name(&self) -> &str;
    fn filter(&self, entry: &DataEntry, score: &QualityScore) -> bool;
    fn reason(&self, entry: &DataEntry, score: &QualityScore) -> Option<String>;
}

/// Filter statistics
#[derive(Debug, Clone, Default)]
pub struct FilterStatistics {
    pub total_processed: usize,
    pub passed: usize,
    pub failed: usize,
    pub failure_reasons: HashMap<String, usize>,
}

/// Filter result with detailed information
#[derive(Debug, Clone)]
pub struct FilterResult {
    pub passed: bool,
    pub entry: DataEntry,
    pub score: QualityScore,
    pub reasons: Vec<String>,
}

impl QualityFilter {
    /// Create new quality filter
    pub fn new(classifier: QualityClassifier, criteria: QualityCriteria) -> Self {
        Self {
            classifier,
            criteria,
            statistics: FilterStatistics::default(),
        }
    }
    
    /// Filter a single data entry
    pub fn filter_entry(&mut self, entry: DataEntry) -> FilterResult {
        let score = self.classifier.classify(&entry);
        let mut reasons = Vec::new();
        let mut passed = true;
        
        self.statistics.total_processed += 1;
        
        // Check overall score
        if score.overall_score < self.criteria.min_overall_score {
            passed = false;
            reasons.push(format!("Overall score {:.2} below threshold {:.2}", 
                score.overall_score, self.criteria.min_overall_score));
        }
        
        // Check algorithm-specific scores
        for (algorithm, min_score) in &self.criteria.min_algorithm_scores {
            if let Some(algorithm_score) = score.algorithm_scores.get(algorithm) {
                if algorithm_score < min_score {
                    passed = false;
                    reasons.push(format!("Algorithm '{}' score {:.2} below threshold {:.2}", 
                        algorithm, algorithm_score, min_score));
                }
            }
        }
        
        // Check content length
        if let Some(max_len) = self.criteria.max_content_length {
            if entry.content.len() > max_len {
                passed = false;
                reasons.push(format!("Content length {} exceeds maximum {}", 
                    entry.content.len(), max_len));
            }
        }
        
        if let Some(min_len) = self.criteria.min_content_length {
            if entry.content.len() < min_len {
                passed = false;
                reasons.push(format!("Content length {} below minimum {}", 
                    entry.content.len(), min_len));
            }
        }
        
        // Check source URL
        if let Some(source_url) = &entry.source_url {
            if !self.criteria.allowed_sources.is_empty() && 
               !self.criteria.allowed_sources.iter().any(|allowed| source_url.contains(allowed)) {
                passed = false;
                reasons.push(format!("Source '{}' not in allowed list", source_url));
            }
            
            if self.criteria.blocked_sources.iter().any(|blocked| source_url.contains(blocked)) {
                passed = false;
                reasons.push(format!("Source '{}' in blocked list", source_url));
            }
        }
        
        // Apply custom filters
        for custom_filter in &self.criteria.custom_filters {
            if !custom_filter.filter(&entry, &score) {
                passed = false;
                if let Some(reason) = custom_filter.reason(&entry, &score) {
                    reasons.push(reason);
                } else {
                    reasons.push(format!("Failed custom filter: {}", custom_filter.name()));
                }
            }
        }
        
        if passed {
            self.statistics.passed += 1;
        } else {
            self.statistics.failed += 1;
            for reason in &reasons {
                *self.statistics.failure_reasons.entry(reason.clone()).or_insert(0) += 1;
            }
        }
        
        FilterResult {
            passed,
            entry,
            score,
            reasons,
        }
    }
    
    /// Filter multiple entries
    pub fn filter_batch(&mut self, entries: Vec<DataEntry>) -> Vec<FilterResult> {
        entries.into_iter()
            .map(|entry| self.filter_entry(entry))
            .collect()
    }
    
    /// Get only entries that passed filtering
    pub fn get_passed_entries(&mut self, entries: Vec<DataEntry>) -> Vec<DataEntry> {
        let results = self.filter_batch(entries);
        results.into_iter()
            .filter(|result| result.passed)
            .map(|result| result.entry)
            .collect()
    }
    
    /// Get entries that failed filtering with reasons
    pub fn get_failed_entries(&mut self, entries: Vec<DataEntry>) -> Vec<(DataEntry, Vec<String>)> {
        let results = self.filter_batch(entries);
        results.into_iter()
            .filter(|result| !result.passed)
            .map(|result| (result.entry, result.reasons))
            .collect()
    }
    
    /// Get filter statistics
    pub fn get_statistics(&self) -> &FilterStatistics {
        &self.statistics
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = FilterStatistics::default();
    }
    
    /// Update criteria
    pub fn update_criteria(&mut self, criteria: QualityCriteria) {
        self.criteria = criteria;
    }
    
    /// Add custom filter
    pub fn add_custom_filter(&mut self, filter: Box<dyn CustomFilter>) {
        self.criteria.custom_filters.push(filter);
    }
    
    /// Get pass rate
    pub fn get_pass_rate(&self) -> f32 {
        if self.statistics.total_processed == 0 {
            0.0
        } else {
            self.statistics.passed as f32 / self.statistics.total_processed as f32
        }
    }
}

/// Content length custom filter
#[derive(Debug, Clone)]
pub struct ContentLengthFilter {
    min_length: usize,
    max_length: usize,
}

impl ContentLengthFilter {
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self { min_length, max_length }
    }
}

impl CustomFilter for ContentLengthFilter {
    fn name(&self) -> &str {
        "content_length_filter"
    }
    
    fn filter(&self, entry: &DataEntry, _score: &QualityScore) -> bool {
        entry.content.len() >= self.min_length && entry.content.len() <= self.max_length
    }
    
    fn reason(&self, entry: &DataEntry, _score: &QualityScore) -> Option<String> {
        let length = entry.content.len();
        if length < self.min_length {
            Some(format!("Content too short: {} chars (min: {})", length, self.min_length))
        } else if length > self.max_length {
            Some(format!("Content too long: {} chars (max: {})", length, self.max_length))
        } else {
            None
        }
    }
}

/// Score variance custom filter
#[derive(Debug, Clone)]
pub struct ScoreVarianceFilter {
    max_variance: f32,
}

impl ScoreVarianceFilter {
    pub fn new(max_variance: f32) -> Self {
        Self { max_variance }
    }
}

impl CustomFilter for ScoreVarianceFilter {
    fn name(&self) -> &str {
        "score_variance_filter"
    }
    
    fn filter(&self, _entry: &DataEntry, score: &QualityScore) -> bool {
        let values: Vec<f32> = score.algorithm_scores.values().cloned().collect();
        if values.is_empty() {
            return true;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance <= self.max_variance
    }
    
    fn reason(&self, _entry: &DataEntry, score: &QualityScore) -> Option<String> {
        let values: Vec<f32> = score.algorithm_scores.values().cloned().collect();
        if values.is_empty() {
            return None;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        if variance > self.max_variance {
            Some(format!("Score variance {:.3} exceeds maximum {:.3}", variance, self.max_variance))
        } else {
            None
        }
    }
}

/// Confidence threshold custom filter
#[derive(Debug, Clone)]
pub struct ConfidenceThresholdFilter {
    min_confidence: f32,
}

impl ConfidenceThresholdFilter {
    pub fn new(min_confidence: f32) -> Self {
        Self { min_confidence }
    }
}

impl CustomFilter for ConfidenceThresholdFilter {
    fn name(&self) -> &str {
        "confidence_threshold_filter"
    }
    
    fn filter(&self, _entry: &DataEntry, score: &QualityScore) -> bool {
        score.confidence >= self.min_confidence
    }
    
    fn reason(&self, _entry: &DataEntry, score: &QualityScore) -> Option<String> {
        if score.confidence < self.min_confidence {
            Some(format!("Confidence {:.2} below threshold {:.2}", score.confidence, self.min_confidence))
        } else {
            None
        }
    }
}

impl Default for QualityCriteria {
    fn default() -> Self {
        Self {
            min_overall_score: 0.5,
            min_algorithm_scores: HashMap::new(),
            max_content_length: Some(10000),
            min_content_length: Some(10),
            required_languages: vec!["en".to_string()],
            forbidden_languages: vec![],
            allowed_sources: vec![],
            blocked_sources: vec![],
            custom_filters: vec![],
        }
    }
}

/// Quality filter builder for easy configuration
pub struct QualityFilterBuilder {
    classifier: Option<QualityClassifier>,
    criteria: QualityCriteria,
}

impl QualityFilterBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            classifier: None,
            criteria: QualityCriteria::default(),
        }
    }
    
    /// Set quality classifier
    pub fn with_classifier(mut self, classifier: QualityClassifier) -> Self {
        self.classifier = Some(classifier);
        self
    }
    
    /// Set minimum overall score
    pub fn min_overall_score(mut self, score: f32) -> Self {
        self.criteria.min_overall_score = score;
        self
    }
    
    /// Set minimum algorithm score
    pub fn min_algorithm_score(mut self, algorithm: String, score: f32) -> Self {
        self.criteria.min_algorithm_scores.insert(algorithm, score);
        self
    }
    
    /// Set content length bounds
    pub fn content_length(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.criteria.min_content_length = min;
        self.criteria.max_content_length = max;
        self
    }
    
    /// Add allowed source pattern
    pub fn allow_source(mut self, source: String) -> Self {
        self.criteria.allowed_sources.push(source);
        self
    }
    
    /// Add blocked source pattern
    pub fn block_source(mut self, source: String) -> Self {
        self.criteria.blocked_sources.push(source);
        self
    }
    
    /// Add custom filter
    pub fn add_custom_filter(mut self, filter: Box<dyn CustomFilter>) -> Self {
        self.criteria.custom_filters.push(filter);
        self
    }
    
    /// Build the filter
    pub fn build(self) -> Result<QualityFilter> {
        let classifier = self.classifier.unwrap_or_else(|| {
            QualityClassifier::new(QualityConfig::default())
        });
        
        Ok(QualityFilter::new(classifier, self.criteria))
    }
}

impl Default for QualityFilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QualityConfig;

    #[test]
    fn test_quality_filtering() {
        let config = QualityConfig::default();
        let classifier = QualityClassifier::new(config);
        let criteria = QualityCriteria::default();
        let mut filter = QualityFilter::new(classifier, criteria);
        
        let good_entry = DataEntry::new("This is a well-written English sentence with appropriate length and good quality.".to_string());
        let bad_entry = DataEntry::new("short".to_string());
        
        let good_result = filter.filter_entry(good_entry);
        let bad_result = filter.filter_entry(bad_entry);
        
        assert!(good_result.passed);
        assert!(!bad_result.passed);
        assert!(!bad_result.reasons.is_empty());
    }

    #[test]
    fn test_custom_filters() {
        let config = QualityConfig::default();
        let classifier = QualityClassifier::new(config);
        let mut criteria = QualityCriteria::default();
        
        // Add custom length filter
        criteria.custom_filters.push(Box::new(ContentLengthFilter::new(50, 200)));
        
        let mut filter = QualityFilter::new(classifier, criteria);
        
        let good_entry = DataEntry::new("This entry has appropriate length for the custom filter requirements.".to_string());
        let bad_entry = DataEntry::new("Too short".to_string());
        
        let good_result = filter.filter_entry(good_entry);
        let bad_result = filter.filter_entry(bad_entry);
        
        assert!(good_result.passed);
        assert!(!bad_result.passed);
    }

    #[test]
    fn test_filter_builder() {
        let filter = QualityFilterBuilder::new()
            .min_overall_score(0.7)
            .content_length(Some(20), Some(1000))
            .allow_source("example.com".to_string())
            .build()
            .unwrap();
        
        assert_eq!(filter.criteria.min_overall_score, 0.7);
        assert_eq!(filter.criteria.min_content_length, Some(20));
        assert_eq!(filter.criteria.max_content_length, Some(1000));
        assert!(filter.criteria.allowed_sources.contains(&"example.com".to_string()));
    }
}
