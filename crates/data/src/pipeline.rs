//! Data Pipeline - Rust implementation
//! 
//! Orchestrates data processing pipeline with multiple stages

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

use crate::{DataEntry, CorpusCollector, DataDeduplicator, DeduplicationAlgorithm};

/// Data processing pipeline with configurable stages
pub struct DataPipeline {
    _name: String,
    stages: Vec<Box<dyn PipelineStage>>,
    _config: PipelineConfig,
    statistics: PipelineStatistics,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub max_workers: usize,
    pub enable_parallel: bool,
    pub error_handling: ErrorHandling,
}

/// Error handling strategy
#[derive(Debug, Clone)]
pub enum ErrorHandling {
    StopOnError,
    SkipAndContinue,
    Retry { max_attempts: usize },
}

/// Pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    pub total_processed: usize,
    pub successful: usize,
    pub failed: usize,
    pub skipped: usize,
    pub processing_time_ms: u64,
}

/// Trait for pipeline stages
#[async_trait::async_trait]
pub trait PipelineStage: Send + Sync {
    /// Get stage name
    fn name(&self) -> &str;
    
    /// Process a single data entry
    async fn process(&self, entry: DataEntry) -> Result<DataEntry>;
    
    /// Process batch of entries
    async fn process_batch(&self, entries: Vec<DataEntry>) -> Result<Vec<DataEntry>> {
        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            match self.process(entry).await {
                Ok(processed) => results.push(processed),
                Err(e) => return Err(e),
            }
        }
        Ok(results)
    }
    
    /// Configure stage
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()>;
}

/// Collection stage - gathers data from sources
#[derive(Debug)]
pub struct CollectionStage {
    _source_type: String,
    collector: Arc<RwLock<CorpusCollector>>,
}

impl CollectionStage {
    pub fn new(source_type: String, capacity: usize) -> Self {
        Self {
            _source_type: source_type.clone(),
            collector: Arc::new(RwLock::new(CorpusCollector::new(source_type, capacity))),
        }
    }
    
    pub async fn add_entry(&self, source_url: Option<String>, content: String, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let mut collector = self.collector.write().await;
        collector.add_entry(source_url, content, metadata)
    }
    
    pub async fn get_all_entries(&self) -> Vec<DataEntry> {
        let collector = self.collector.read().await;
        collector.entries().to_vec()
    }
}

#[async_trait::async_trait]
impl PipelineStage for CollectionStage {
    fn name(&self) -> &str {
        "collection"
    }
    
    async fn process(&self, entry: DataEntry) -> Result<DataEntry> {
        // Collection stage just passes through the entry
        Ok(entry)
    }
    
    async fn process_batch(&self, entries: Vec<DataEntry>) -> Result<Vec<DataEntry>> {
        // Add all entries to collector
        for entry in &entries {
            let mut collector = self.collector.write().await;
            collector.add_entry(
                entry.source_url.clone(),
                entry.content.clone(),
                Some(entry.metadata.clone())
            )?;
        }
        Ok(entries)
    }
    
    fn configure(&mut self, _config: &HashMap<String, String>) -> Result<()> {
        // Collection stage doesn't need additional configuration
        Ok(())
    }
}

/// Deduplication stage - removes duplicate entries
#[derive(Debug)]
pub struct DeduplicationStage {
    deduplicator: Arc<RwLock<DataDeduplicator>>,
}

impl DeduplicationStage {
    pub fn new(algorithm: DeduplicationAlgorithm, similarity_threshold: f32) -> Self {
        Self {
            deduplicator: Arc::new(RwLock::new(DataDeduplicator::new(algorithm, similarity_threshold))),
        }
    }
}

#[async_trait::async_trait]
impl PipelineStage for DeduplicationStage {
    fn name(&self) -> &str {
        "deduplication"
    }
    
    async fn process(&self, entry: DataEntry) -> Result<DataEntry> {
        let mut deduplicator = self.deduplicator.write().await;
        if deduplicator.is_duplicate(&entry)? {
            return Err(anyhow::anyhow!("Duplicate entry detected"));
        }
        Ok(entry)
    }
    
    async fn process_batch(&self, entries: Vec<DataEntry>) -> Result<Vec<DataEntry>> {
        let mut deduplicator = self.deduplicator.write().await;
        let mut unique_entries = Vec::new();
        
        for entry in entries {
            if !deduplicator.is_duplicate(&entry)? {
                unique_entries.push(entry);
            }
        }
        
        Ok(unique_entries)
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(threshold) = config.get("similarity_threshold") {
            let threshold = threshold.parse::<f32>()?;
            let mut deduplicator = self.deduplicator.blocking_write();
            deduplicator.similarity_threshold = threshold;
        }
        Ok(())
    }
}

/// Quality filtering stage
pub struct QualityFilterStage {
    min_quality_score: f32,
    content_filters: Vec<Box<dyn ContentFilter>>,
}

impl QualityFilterStage {
    pub fn new(min_quality_score: f32) -> Self {
        Self {
            min_quality_score,
            content_filters: Vec::new(),
        }
    }
    
    pub fn add_filter(mut self, filter: Box<dyn ContentFilter>) -> Self {
        self.content_filters.push(filter);
        self
    }
}

#[async_trait::async_trait]
impl PipelineStage for QualityFilterStage {
    fn name(&self) -> &str {
        "quality_filter"
    }
    
    async fn process(&self, entry: DataEntry) -> Result<DataEntry> {
        // Check quality score
        if let Some(score) = entry.quality_score {
            if score < self.min_quality_score {
                return Err(anyhow::anyhow!("Quality score {} below threshold {}", score, self.min_quality_score));
            }
        }
        
        // Apply content filters
        for filter in &self.content_filters {
            if !filter.filter(&entry) {
                return Err(anyhow::anyhow!("Content filtered by {}", filter.name()));
            }
        }
        
        Ok(entry)
    }
    
    fn configure(&mut self, config: &HashMap<String, String>) -> Result<()> {
        if let Some(threshold) = config.get("min_quality_score") {
            self.min_quality_score = threshold.parse::<f32>()?;
        }
        Ok(())
    }
}

/// Trait for content filters
pub trait ContentFilter: Send + Sync {
    fn name(&self) -> &str;
    fn filter(&self, entry: &DataEntry) -> bool;
}

/// Length filter - filters entries by content length
#[derive(Debug)]
pub struct LengthFilter {
    min_length: usize,
    max_length: usize,
}

impl LengthFilter {
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self { min_length, max_length }
    }
}

impl ContentFilter for LengthFilter {
    fn name(&self) -> &str {
        "length_filter"
    }
    
    fn filter(&self, entry: &DataEntry) -> bool {
        entry.content.len() >= self.min_length && entry.content.len() <= self.max_length
    }
}

/// Language filter - filters entries by language
#[derive(Debug)]
pub struct LanguageFilter {
    allowed_languages: Vec<String>,
}

impl LanguageFilter {
    pub fn new(allowed_languages: Vec<String>) -> Self {
        Self { allowed_languages }
    }
}

impl ContentFilter for LanguageFilter {
    fn name(&self) -> &str {
        "language_filter"
    }
    
    fn filter(&self, entry: &DataEntry) -> bool {
        // Simplified language detection - in practice, use a proper language detection library
        let content = &entry.content.to_lowercase();
        let has_english = content.contains("the") || content.contains("and") || content.contains("is");
        
        if self.allowed_languages.contains(&"en".to_string()) {
            has_english
        } else {
            true // Allow if no specific language restrictions
        }
    }
}

impl DataPipeline {
    /// Create new data pipeline
    pub fn new(name: String, config: PipelineConfig) -> Self {
        Self {
            _name: name,
            stages: Vec::new(),
            _config: config,
            statistics: PipelineStatistics::default(),
        }
    }
    
    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }
    
    /// Process a single entry through all stages
    pub async fn process_entry(&mut self, mut entry: DataEntry) -> Result<DataEntry> {
        let start_time = std::time::Instant::now();
        
        for stage in &self.stages {
            match stage.process(entry).await {
                Ok(processed) => entry = processed,
                Err(e) => {
                    self.statistics.failed += 1;
                    return Err(anyhow::anyhow!("Stage '{}' failed: {}", stage.name(), e));
                }
            }
        }
        
        self.statistics.successful += 1;
        self.statistics.processing_time_ms += start_time.elapsed().as_millis() as u64;
        Ok(entry)
    }
    
    /// Process batch of entries
    pub async fn process_batch(&mut self, entries: Vec<DataEntry>) -> Result<Vec<DataEntry>> {
        let start_time = std::time::Instant::now();
        self.statistics.total_processed += entries.len();
        
        let mut current_entries = entries;
        
        for stage in &self.stages {
            let current_len = current_entries.len();
            match stage.process_batch(current_entries).await {
                Ok(processed) => current_entries = processed,
                Err(e) => {
                    self.statistics.failed += current_len;
                    return Err(anyhow::anyhow!("Stage '{}' batch failed: {}", stage.name(), e));
                }
            }
        }
        
        self.statistics.successful += current_entries.len();
        self.statistics.processing_time_ms += start_time.elapsed().as_millis() as u64;
        
        Ok(current_entries)
    }
    
    /// Get pipeline statistics
    pub fn get_statistics(&self) -> &PipelineStatistics {
        &self.statistics
    }
    
    /// Get stage names
    pub fn get_stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|stage| stage.name()).collect()
    }
    
    /// Configure specific stage
    pub fn configure_stage(&mut self, stage_name: &str, config: &HashMap<String, String>) -> Result<()> {
        for stage in &mut self.stages {
            if stage.name() == stage_name {
                return stage.configure(config);
            }
        }
        Err(anyhow::anyhow!("Stage '{}' not found", stage_name))
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = PipelineStatistics::default();
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_workers: 4,
            enable_parallel: true,
            error_handling: ErrorHandling::SkipAndContinue,
        }
    }
}

/// 4GB Budget Optimization Functions
impl DataPipeline {
    /// Create optimized pipeline for 4GB budget
    /// Focus on SFT + Reasoning, not pretraining
    pub fn build_4gb_optimized_pipeline() -> Self {
        let config = PipelineConfig {
            batch_size: 1000,  // Larger batches for efficiency
            max_workers: 8,     // More workers for parallel processing
            enable_parallel: true,
            error_handling: ErrorHandling::SkipAndContinue,
        };
        
        let pipeline = Self::new("nexora_4gb_pipeline".to_string(), config)
            // Stage 1: Collection with high priority datasets
            .add_stage(Box::new(CollectionStage::new("high_priority_sft".to_string(), 2_500_000)))
            // Stage 2: Deduplication with high threshold for SFT
            .add_stage(Box::new(DeduplicationStage::new(DeduplicationAlgorithm::SemanticSimilarity, 0.85)))
            // Stage 3: Quality filtering optimized for instruction tuning
            .add_stage(Box::new(QualityFilterStage::new(0.6)
                .add_filter(Box::new(LengthFilter::new(50, 4096)))  // Instruction length optimization
                .add_filter(Box::new(LanguageFilter::new(vec!["en".to_string()])))
            ));
        
        pipeline
    }
    
    /// Get 4GB budget dataset recommendations
    pub fn get_4gb_dataset_recommendations() -> Vec<DatasetRecommendation> {
        vec![
            // 🔴 HIGH PRIORITY - Core SFT (2.2GB)
            DatasetRecommendation {
                name: "OpenHermes-2.5".to_string(),
                priority: "high".to_string(),
                estimated_gb: 1.0,
                category: "backbone_sft".to_string(),
                description: "🔴 HIGH PRIORITY: Highest quality instruction following".to_string(),
                weight: 0.25,
                reason: "Backbone for instruction tuning - highest quality dataset".to_string(),
            },
            DatasetRecommendation {
                name: "UltraChat-200k".to_string(),
                priority: "high".to_string(),
                estimated_gb: 0.8,
                category: "conversation_sft".to_string(),
                description: "🔴 HIGH PRIORITY: Multi-turn conversations".to_string(),
                weight: 0.20,
                reason: "Essential for multi-turn conversation capabilities".to_string(),
            },
            DatasetRecommendation {
                name: "OpenOrca".to_string(),
                priority: "high".to_string(),
                estimated_gb: 0.4,
                category: "reasoning_sft".to_string(),
                description: "🔴 HIGH PRIORITY: Reasoning traces".to_string(),
                weight: 0.10,
                reason: "Critical for reasoning and chain-of-thought capabilities".to_string(),
            },
            
            // 🟡 MEDIUM PRIORITY - Coding & Reasoning (1.5GB)
            DatasetRecommendation {
                name: "CodeAlpaca-20k".to_string(),
                priority: "medium".to_string(),
                estimated_gb: 0.02,
                category: "coding_sft".to_string(),
                description: "🟡 MEDIUM: Code instruction following".to_string(),
                weight: 0.02,
                reason: "Essential for code generation capabilities".to_string(),
            },
            DatasetRecommendation {
                name: "Magicoder-OSS".to_string(),
                priority: "medium".to_string(),
                estimated_gb: 0.08,
                category: "coding_explanation".to_string(),
                description: "🟡 MEDIUM: Code with explanations".to_string(),
                weight: 0.08,
                reason: "Improves code explanation and documentation".to_string(),
            },
            DatasetRecommendation {
                name: "GSM8K".to_string(),
                priority: "medium".to_string(),
                estimated_gb: 0.006,
                category: "math_reasoning".to_string(),
                description: "🟡 MEDIUM: Mathematical reasoning".to_string(),
                weight: 0.03,
                reason: "Foundation for mathematical reasoning".to_string(),
            },
            DatasetRecommendation {
                name: "MetaMathQA".to_string(),
                priority: "medium".to_string(),
                estimated_gb: 0.2,
                category: "advanced_math".to_string(),
                description: "🟡 MEDIUM: Advanced math reasoning".to_string(),
                weight: 0.05,
                reason: "Advanced mathematical problem solving".to_string(),
            },
            DatasetRecommendation {
                name: "HH-RLHF".to_string(),
                priority: "medium".to_string(),
                estimated_gb: 0.15,
                category: "alignment".to_string(),
                description: "🟡 MEDIUM: Helpfulness/harmless alignment".to_string(),
                weight: 0.07,
                reason: "Critical for safety and alignment".to_string(),
            },
            
            // 🟢 LOW PRIORITY - Quality enhancement (0.3GB)
            DatasetRecommendation {
                name: "NoRobots".to_string(),
                priority: "low".to_string(),
                estimated_gb: 0.01,
                category: "quality_anchor".to_string(),
                description: "🟢 LOW: Quality anchor for instruction following".to_string(),
                weight: 0.10,
                reason: "Quality baseline for instruction following".to_string(),
            },
            DatasetRecommendation {
                name: "Dolly-15k".to_string(),
                priority: "low".to_string(),
                estimated_gb: 0.013,
                category: "diversity".to_string(),
                description: "🟢 LOW: Response diversity".to_string(),
                weight: 0.05,
                reason: "Adds diversity to response patterns".to_string(),
            },
            DatasetRecommendation {
                name: "UltraFeedback".to_string(),
                priority: "low".to_string(),
                estimated_gb: 0.15,
                category: "preference_learning".to_string(),
                description: "🟢 LOW: Preference alignment".to_string(),
                weight: 0.05,
                reason: "Improves response quality through preference learning".to_string(),
            },
        ]
    }
    
    /// Calculate optimal dataset mix for 4GB budget
    pub fn calculate_4gb_mix() -> BudgetMix {
        let _total_budget = 4.0;
        
        BudgetMix {
            backbone_sft: 1.0,      // 25% - OpenHermes-2.5
            conversation_sft: 0.8,   // 20% - UltraChat
            reasoning_sft: 0.4,      // 10% - OpenOrca
            coding_sft: 0.1,         // 2.5% - Code datasets
            math_reasoning: 0.206,    // 5.15% - Math datasets
            alignment: 0.15,          // 3.75% - HH-RLHF
            quality_enhancement: 0.173, // 4.3% - Quality datasets
            total_used: 3.829,        // 95.7% of budget
            remaining: 0.171,          // 4.3% buffer
        }
    }
}

/// Dataset recommendation for 4GB budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRecommendation {
    pub name: String,
    pub priority: String,
    pub estimated_gb: f64,
    pub category: String,
    pub description: String,
    pub weight: f64,
    pub reason: String,
}

/// Budget mix calculation for 4GB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetMix {
    pub backbone_sft: f64,
    pub conversation_sft: f64,
    pub reasoning_sft: f64,
    pub coding_sft: f64,
    pub math_reasoning: f64,
    pub alignment: f64,
    pub quality_enhancement: f64,
    pub total_used: f64,
    pub remaining: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeduplicationAlgorithm;

    #[tokio::test]
    async fn test_basic_pipeline() {
        let config = PipelineConfig::default();
        let mut pipeline = DataPipeline::new("test".to_string(), config)
            .add_stage(Box::new(CollectionStage::new("test".to_string(), 100)))
            .add_stage(Box::new(DeduplicationStage::new(DeduplicationAlgorithm::ExactHash, 0.0)));
        
        let entry = DataEntry::new("test content".to_string());
        let result = pipeline.process_entry(entry).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_quality_filtering() {
        let filter_stage = QualityFilterStage::new(0.5)
            .add_filter(Box::new(LengthFilter::new(10, 1000)))
            .add_filter(Box::new(LanguageFilter::new(vec!["en".to_string()])));
        
        let good_entry = DataEntry::new("This is a good English sentence with proper length.".to_string());
        let bad_entry = DataEntry::new("short".to_string());
        
        assert!(filter_stage.process(good_entry).await.is_ok());
        assert!(filter_stage.process(bad_entry).await.is_err());
    }
    
    #[test]
    fn test_4gb_recommendations() {
        let recommendations = DataPipeline::get_4gb_dataset_recommendations();
        
        // Should have 11 recommendations
        assert_eq!(recommendations.len(), 11);
        
        // Check priority distribution
        let high_priority = recommendations.iter().filter(|r| r.priority == "high").count();
        let medium_priority = recommendations.iter().filter(|r| r.priority == "medium").count();
        let low_priority = recommendations.iter().filter(|r| r.priority == "low").count();
        
        assert_eq!(high_priority, 3);   // Core SFT
        assert_eq!(medium_priority, 5);  // Coding & Reasoning
        assert_eq!(low_priority, 3);    // Quality enhancement
        
        // Check total weight sums to 1.0
        let total_weight: f64 = recommendations.iter().map(|r| r.weight).sum();
        assert!((total_weight - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_4gb_budget_mix() {
        let mix = DataPipeline::calculate_4gb_mix();
        
        // Should use less than 4GB total
        assert!(mix.total_used <= 4.0);
        
        // Should leave some buffer
        assert!(mix.remaining > 0.0);
        
        // Backbone SFT should be largest portion
        assert!(mix.backbone_sft > mix.conversation_sft);
        assert!(mix.backbone_sft > mix.reasoning_sft);
    }
    
    #[test]
    fn test_4gb_pipeline_creation() {
        let pipeline = DataPipeline::build_4gb_optimized_pipeline();
        
        // Should have 3 stages
        assert_eq!(pipeline.get_stage_names().len(), 3);
        
        // Should have optimized config
        assert_eq!(pipeline._config.batch_size, 1000);
        assert_eq!(pipeline._config.max_workers, 8);
        assert!(pipeline._config.enable_parallel);
    }
}
