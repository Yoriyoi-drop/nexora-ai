//! Data handling utilities for SPARO
//! 
//! Modul untuk mengelola data pelatihan, preprocessing, dan augmentasi

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use rand::Rng;
use rand::seq::SliceRandom;

use super::core::{ReasoningTrace, ReasoningStep, JudgeFeedback, FeedbackType};

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Maximum traces per batch
    pub max_traces_per_batch: usize,
    /// Minimum trace quality score
    pub min_quality_score: f32,
    /// Data augmentation enabled
    pub enable_augmentation: bool,
    /// Shuffle data each epoch
    pub shuffle_data: bool,
    /// Validation split ratio
    pub validation_split: f32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            max_traces_per_batch: 100,
            min_quality_score: 0.3,
            enable_augmentation: true,
            shuffle_data: true,
            validation_split: 0.2,
        }
    }
}

/// Dataset for SPARO training
pub struct SparoDataset {
    config: DatasetConfig,
    traces: Vec<ReasoningTrace>,
    feedback: Vec<JudgeFeedback>,
    validation_traces: Vec<ReasoningTrace>,
    validation_feedback: Vec<JudgeFeedback>,
}

impl SparoDataset {
    /// Create new dataset
    pub fn new(config: DatasetConfig) -> Self {
        Self {
            config,
            traces: Vec::new(),
            feedback: Vec::new(),
            validation_traces: Vec::new(),
            validation_feedback: Vec::new(),
        }
    }
    
    /// Add training data
    pub fn add_data(&mut self, traces: Vec<ReasoningTrace>, feedback: Vec<JudgeFeedback>) -> Result<()> {
        // Filter by quality
        let filtered_traces = self.filter_traces_by_quality(traces)?;
        let filtered_feedback = self.filter_feedback_by_quality(feedback)?;
        
        self.traces.extend(filtered_traces);
        self.feedback.extend(filtered_feedback);
        
        // Split validation data
        if self.config.validation_split > 0.0 {
            self.split_validation_data()?;
        }
        
        Ok(())
    }
    
    /// Get training batch
    pub fn get_training_batch(&mut self, batch_size: usize) -> Result<TrainingBatch> {
        if self.traces.is_empty() {
            return Err(anyhow::anyhow!("No training data available"));
        }
        
        // Shuffle if enabled
        if self.config.shuffle_data {
            self.shuffle_training_data();
        }
        
        // Sample batch
        let batch_size = batch_size.min(self.traces.len());
        let batch_traces: Vec<ReasoningTrace> = self.traces.drain(0..batch_size).collect();
        let batch_feedback = self.extract_feedback_for_traces(&batch_traces)?;
        
        Ok(TrainingBatch::new(batch_traces, batch_feedback, 0))
    }
    
    /// Get validation batch
    pub fn get_validation_batch(&self, batch_size: usize) -> Result<TrainingBatch> {
        if self.validation_traces.is_empty() {
            return Err(anyhow::anyhow!("No validation data available"));
        }
        
        let batch_size = batch_size.min(self.validation_traces.len());
        let start_idx = rand::random::<usize>() % (self.validation_traces.len() - batch_size + 1);
        let batch_traces = self.validation_traces[start_idx..start_idx + batch_size].to_vec();
        let batch_feedback = self.extract_feedback_for_traces(&batch_traces)?;
        
        Ok(TrainingBatch::new(batch_traces, batch_feedback, 0))
    }
    
    /// Get dataset statistics
    pub fn get_stats(&self) -> DatasetStats {
        DatasetStats {
            total_traces: self.traces.len(),
            total_feedback: self.feedback.len(),
            validation_traces: self.validation_traces.len(),
            validation_feedback: self.validation_feedback.len(),
            avg_trace_length: self.calculate_avg_trace_length(),
            feedback_distribution: self.calculate_feedback_distribution(),
        }
    }
    
    /// Augment dataset
    pub fn augment_dataset(&mut self) -> Result<()> {
        if !self.config.enable_augmentation {
            return Ok(());
        }
        
        let original_size = self.traces.len();
        
        // Generate augmented traces
        let augmented_traces = self.generate_augmented_traces(&self.traces)?;
        let augmented_feedback = self.generate_augmented_feedback(&augmented_traces)?;
        
        self.traces.extend(augmented_traces);
        self.feedback.extend(augmented_feedback);
        
        println!("Augmented dataset: {} -> {} traces", original_size, self.traces.len());
        
        Ok(())
    }
    
    // Helper methods
    fn filter_traces_by_quality(&self, traces: Vec<ReasoningTrace>) -> Result<Vec<ReasoningTrace>> {
        let filtered: Vec<ReasoningTrace> = traces
            .into_iter()
            .filter(|trace| self.calculate_trace_quality(trace) >= self.config.min_quality_score)
            .collect();
        
        Ok(filtered)
    }
    
    fn filter_feedback_by_quality(&self, feedback: Vec<JudgeFeedback>) -> Result<Vec<JudgeFeedback>> {
        let filtered: Vec<JudgeFeedback> = feedback
            .into_iter()
            .filter(|fb| {
                self.calculate_feedback_quality(fb)
                    .map(|quality| quality >= self.config.min_quality_score)
                    .unwrap_or(false)
            })
            .collect();
        
        Ok(filtered)
    }
    
    fn calculate_trace_quality(&self, trace: &ReasoningTrace) -> f32 {
        let mut quality = 0.5;
        
        // Length factor
        if trace.steps.len() >= 3 && trace.steps.len() <= 10 {
            quality += 0.2;
        }
        
        // Final answer quality
        if !trace.final_answer.is_empty() {
            quality += 0.1;
        }
        
        // Step consistency
        let step_numbers: std::collections::HashSet<_> = trace.steps.iter()
            .map(|s| s.step_number)
            .collect();
        if step_numbers.len() == trace.steps.len() {
            quality += 0.2;
        }
        
        (quality as f32).min(1.0)
    }
    
    fn calculate_feedback_quality(&self, feedback: &JudgeFeedback) -> Result<f32> {
        let mut quality = 0.5;
        
        match &feedback.feedback_type {
            FeedbackType::Independent { confidence, .. } => {
                quality += *confidence * 0.5;
            },
            FeedbackType::Pairwise { confidence, .. } => {
                quality += *confidence * 0.5;
            },
        }
        
        // Reasoning length
        if feedback.reasoning.len() > 10 {
            quality += 0.1;
        }
        
        Ok(quality.min(1.0))
    }
    
    fn split_validation_data(&mut self) -> Result<()> {
        let total_size = self.traces.len();
        let validation_size = (total_size as f32 * self.config.validation_split) as usize;
        
        if validation_size == 0 {
            return Ok(());
        }
        
        // Randomly sample validation data
        let mut indices: Vec<usize> = (0..total_size).collect();
        indices.shuffle(&mut rand::thread_rng());
        
        let validation_indices: std::collections::HashSet<_> = indices
            .into_iter()
            .take(validation_size)
            .collect();
        
        // Split traces
        let (validation_traces, training_traces): (Vec<_>, Vec<_>) = self.traces
            .drain(..)
            .enumerate()
            .partition(|(idx, _)| validation_indices.contains(idx));
        
        self.validation_traces = validation_traces.into_iter().map(|(_, trace)| trace).collect();
        self.traces = training_traces.into_iter().map(|(_, trace)| trace).collect();
        
        // Split feedback
        let (validation_feedback, training_feedback): (Vec<_>, Vec<_>) = self.feedback
            .drain(..)
            .enumerate()
            .partition(|(idx, _)| validation_indices.contains(idx));
        
        self.validation_feedback = validation_feedback.into_iter().map(|(_, fb)| fb).collect();
        self.feedback = training_feedback.into_iter().map(|(_, fb)| fb).collect();
        
        Ok(())
    }
    
    fn shuffle_training_data(&mut self) {
        let mut indices: Vec<usize> = (0..self.traces.len()).collect();
        indices.shuffle(&mut rand::thread_rng());
        
        let mut shuffled_traces = Vec::with_capacity(self.traces.len());
        for idx in indices {
            shuffled_traces.push(self.traces[idx].clone());
        }
        self.traces = shuffled_traces;
    }
    
    fn extract_feedback_for_traces(&self, traces: &[ReasoningTrace]) -> Result<Vec<JudgeFeedback>> {
        let trace_ids: std::collections::HashSet<_> = traces.iter().map(|t| t.id).collect();
        
        let relevant_feedback: Vec<JudgeFeedback> = self.feedback
            .iter()
            .filter(|fb| trace_ids.contains(&fb.trace_id))
            .cloned()
            .collect();
        
        Ok(relevant_feedback)
    }
    
    fn calculate_avg_trace_length(&self) -> f32 {
        if self.traces.is_empty() {
            return 0.0;
        }
        
        let total_steps: usize = self.traces.iter().map(|t| t.steps.len()).sum();
        total_steps as f32 / self.traces.len() as f32
    }
    
    fn calculate_feedback_distribution(&self) -> FeedbackDistribution {
        let mut independent_count = 0;
        let mut pairwise_count = 0;
        
        for feedback in &self.feedback {
            match &feedback.feedback_type {
                FeedbackType::Independent { .. } => independent_count += 1,
                FeedbackType::Pairwise { .. } => pairwise_count += 1,
            }
        }
        
        FeedbackDistribution {
            total_feedback: self.feedback.len(),
            independent_count,
            pairwise_count,
        }
    }
    
    fn generate_augmented_traces(&self, original_traces: &[ReasoningTrace]) -> Result<Vec<ReasoningTrace>> {
        let mut augmented = Vec::with_capacity(original_traces.len() * 2);
        
        for trace in original_traces {
            // Paraphrase augmentation
            let paraphrased = self.paraphrase_trace(trace)?;
            augmented.push(paraphrased);
            
            // Step reordering augmentation (if applicable)
            if trace.steps.len() > 3 {
                let reordered = self.reorder_steps(trace)?;
                augmented.push(reordered);
            }
        }
        
        Ok(augmented)
    }
    
    fn generate_augmented_feedback(&self, traces: &[ReasoningTrace]) -> Result<Vec<JudgeFeedback>> {
        let mut augmented_feedback = Vec::with_capacity(traces.len());
        
        for trace in traces {
            // Generate synthetic feedback
            let synthetic_feedback = self.generate_synthetic_feedback(trace)?;
            augmented_feedback.extend(synthetic_feedback);
        }
        
        Ok(augmented_feedback)
    }
    
    fn paraphrase_trace(&self, original: &ReasoningTrace) -> Result<ReasoningTrace> {
        let mut paraphrased_steps = Vec::with_capacity(original.steps.len());
        
        for step in &original.steps {
            let paraphrased_content = self.paraphrase_text(&step.content)?;
            
            let paraphrased_step = ReasoningStep {
                id: Uuid::new_v4(),
                content: paraphrased_content,
                step_number: step.step_number,
                timestamp: chrono::Utc::now(),
            };
            
            paraphrased_steps.push(paraphrased_step);
        }
        
        Ok(ReasoningTrace {
            id: Uuid::new_v4(),
            prompt: self.paraphrase_text(&original.prompt)?,
            steps: paraphrased_steps,
            final_answer: self.paraphrase_text(&original.final_answer)?,
            created_at: chrono::Utc::now(),
        })
    }
    
    fn reorder_steps(&self, original: &ReasoningTrace) -> Result<ReasoningTrace> {
        let mut steps = original.steps.clone();
        
        // Simple reordering: swap adjacent steps
        for i in 0..steps.len().saturating_sub(2) {
            if rand::random::<f32>() < 0.3 {
                steps.swap(i, i + 1);
            }
        }
        
        // Update step numbers
        for (i, step) in steps.iter_mut().enumerate() {
            step.step_number = i + 1;
        }
        
        Ok(ReasoningTrace {
            id: Uuid::new_v4(),
            prompt: original.prompt.clone(),
            steps,
            final_answer: original.final_answer.clone(),
            created_at: chrono::Utc::now(),
        })
    }
    
    fn paraphrase_text(&self, text: &str) -> Result<String> {
        // Simple paraphrasing: replace synonyms
        let mut paraphrased = text.to_string();
        
        let synonyms = vec![
            ("solve", "resolve"),
            ("explain", "describe"),
            ("calculate", "compute"),
            ("analyze", "examine"),
            ("because", "since"),
            ("therefore", "thus"),
            ("however", "but"),
        ];
        
        for (original, synonym) in synonyms {
            paraphrased = paraphrased.replace(original, synonym);
        }
        
        Ok(paraphrased)
    }
    
    fn generate_synthetic_feedback(&self, trace: &ReasoningTrace) -> Result<Vec<JudgeFeedback>> {
        let mut synthetic_feedback = Vec::with_capacity(trace.steps.len());
        
        for step in &trace.steps {
            let confidence = 0.5 + rand::random::<f32>() * 0.5;
            let is_good = confidence > 0.5;
            
            let feedback_type = FeedbackType::Independent {
                step_id: step.id,
                is_good,
                confidence,
            };
            
            let feedback = JudgeFeedback {
                id: Uuid::new_v4(),
                trace_id: trace.id,
                feedback_type,
                reasoning: format!("Synthetic feedback for step {} with confidence {}", 
                    step.step_number, confidence),
                created_at: chrono::Utc::now(),
            };
            
            synthetic_feedback.push(feedback);
        }
        
        Ok(synthetic_feedback)
    }
}

/// Training batch wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingBatch {
    pub id: Uuid,
    pub traces: Vec<ReasoningTrace>,
    pub feedback: Vec<JudgeFeedback>,
    pub iteration: usize,
}

impl TrainingBatch {
    pub fn new(traces: Vec<ReasoningTrace>, feedback: Vec<JudgeFeedback>, iteration: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            traces,
            feedback,
            iteration,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }
    
    pub fn size(&self) -> usize {
        self.traces.len()
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_traces: usize,
    pub total_feedback: usize,
    pub validation_traces: usize,
    pub validation_feedback: usize,
    pub avg_trace_length: f32,
    pub feedback_distribution: FeedbackDistribution,
}

/// Feedback distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackDistribution {
    pub total_feedback: usize,
    pub independent_count: usize,
    pub pairwise_count: usize,
}

/// Data processor for preprocessing
pub struct DataProcessor {
    _config: DatasetConfig,
}

impl DataProcessor {
    pub fn new(config: DatasetConfig) -> Self {
        Self { _config: config }
    }
    
    /// Preprocess traces
    pub fn preprocess_traces(&self, traces: &[ReasoningTrace]) -> Result<Vec<ReasoningTrace>> {
        let mut processed = Vec::with_capacity(traces.len());
        
        for trace in traces {
            let cleaned = self.clean_trace(trace)?;
            let normalized = self.normalize_trace(&cleaned)?;
            processed.push(normalized);
        }
        
        Ok(processed)
    }
    
    /// Clean trace by removing invalid steps
    fn clean_trace(&self, trace: &ReasoningTrace) -> Result<ReasoningTrace> {
        let cleaned_steps: Vec<ReasoningStep> = trace.steps
            .iter()
            .filter(|step| !step.content.trim().is_empty())
            .cloned()
            .collect();
        
        Ok(ReasoningTrace {
            id: trace.id,
            prompt: trace.prompt.clone(),
            steps: cleaned_steps,
            final_answer: trace.final_answer.clone(),
            created_at: trace.created_at,
        })
    }
    
    /// Normalize trace format
    fn normalize_trace(&self, trace: &ReasoningTrace) -> Result<ReasoningTrace> {
        let mut normalized_steps = Vec::with_capacity(trace.steps.len());
        
        for (i, step) in trace.steps.iter().enumerate() {
            let normalized_step = ReasoningStep {
                id: step.id,
                content: self.normalize_text(&step.content),
                step_number: i + 1,
                timestamp: step.timestamp,
            };
            normalized_steps.push(normalized_step);
        }
        
        Ok(ReasoningTrace {
            id: trace.id,
            prompt: self.normalize_text(&trace.prompt),
            steps: normalized_steps,
            final_answer: self.normalize_text(&trace.final_answer),
            created_at: trace.created_at,
        })
    }
    
    /// Normalize text format
    fn normalize_text(&self, text: &str) -> String {
        text.trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?;:".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default dataset
    pub fn create_default_dataset() -> SparoDataset {
        SparoDataset::new(DatasetConfig::default())
    }
    
    /// Generate sample data for testing
    pub fn generate_sample_data(num_traces: usize) -> (Vec<ReasoningTrace>, Vec<JudgeFeedback>) {
        let mut traces = Vec::with_capacity(num_traces);
        let mut feedback = Vec::with_capacity(num_traces);
        
        for i in 0..num_traces {
            let trace = ReasoningTrace {
                id: Uuid::new_v4(),
                prompt: format!("Sample prompt {}", i),
                steps: vec![
                    ReasoningStep {
                        id: Uuid::new_v4(),
                        content: format!("Step 1 for prompt {}", i),
                        step_number: 1,
                        timestamp: chrono::Utc::now(),
                    },
                    ReasoningStep {
                        id: Uuid::new_v4(),
                        content: format!("Step 2 for prompt {}", i),
                        step_number: 2,
                        timestamp: chrono::Utc::now(),
                    },
                ],
                final_answer: format!("Answer for prompt {}", i),
                created_at: chrono::Utc::now(),
            };
            
            // Generate feedback for each step
            for step in &trace.steps {
                let feedback_type = FeedbackType::Independent {
                    step_id: step.id,
                    is_good: i % 2 == 0,
                    confidence: 0.8,
                };
                
                let fb = JudgeFeedback {
                    id: Uuid::new_v4(),
                    trace_id: trace.id,
                    feedback_type,
                    reasoning: format!("Feedback for step {} in trace {}", step.step_number, i),
                    created_at: chrono::Utc::now(),
                };
                
                feedback.push(fb);
            }
            
            traces.push(trace);
        }
        
        (traces, feedback)
    }
    
    /// Validate dataset integrity
    pub fn validate_dataset(traces: &[ReasoningTrace], feedback: &[JudgeFeedback]) -> Result<()> {
        // Check trace validity
        for trace in traces {
            if trace.prompt.is_empty() {
                return Err(anyhow::anyhow!("Empty prompt in trace {}", trace.id));
            }
            
            if trace.steps.is_empty() {
                return Err(anyhow::anyhow!("No steps in trace {}", trace.id));
            }
            
            // Check step numbering
            let step_numbers: std::collections::HashSet<_> = trace.steps.iter()
                .map(|s| s.step_number)
                .collect();
            if step_numbers.len() != trace.steps.len() {
                return Err(anyhow::anyhow!("Duplicate step numbers in trace {}", trace.id));
            }
        }
        
        // Check feedback validity
        let trace_ids: std::collections::HashSet<_> = traces.iter().map(|t| t.id).collect();
        
        for fb in feedback {
            if !trace_ids.contains(&fb.trace_id) {
                return Err(anyhow::anyhow!("Feedback refers to non-existent trace {}", fb.trace_id));
            }
            
            match &fb.feedback_type {
                FeedbackType::Independent { step_id, confidence, .. } => {
                    if *confidence < 0.0 || *confidence > 1.0 {
                        return Err(anyhow::anyhow!("Invalid confidence in feedback {}", fb.id));
                    }
                    
                    // Check if step exists
                    let step_exists = traces.iter()
                        .any(|trace| trace.steps.iter().any(|step| step.id == *step_id));
                    
                    if !step_exists {
                        return Err(anyhow::anyhow!("Feedback refers to non-existent step {}", step_id));
                    }
                },
                FeedbackType::Pairwise { preferred, rejected, confidence, .. } => {
                    if preferred == rejected {
                        return Err(anyhow::anyhow!("Preferred and rejected cannot be the same"));
                    }
                    
                    if *confidence < 0.0 || *confidence > 1.0 {
                        return Err(anyhow::anyhow!("Invalid confidence in feedback {}", fb.id));
                    }
                },
            }
        }
        
        Ok(())
    }
}
