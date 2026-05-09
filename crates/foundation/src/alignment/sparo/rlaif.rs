//! Reinforcement Learning from AI Feedback (RLAIF) Implementation
//! 
//! RLAIF menggunakan AI cerdas sebagai pemberi umpan balik, 
//! menggantikan peran manusia yang mahal.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::core::{ReasoningStep, ReasoningTrace, JudgeFeedback, FeedbackType};

/// Konfigurasi RLAIF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlaifConfig {
    /// Model AI judge yang digunakan
    pub judge_model: String,
    /// Threshold untuk confidence AI judge
    pub confidence_threshold: f32,
    /// Maximum number of judge models
    pub max_judges: usize,
    /// Consensus threshold untuk multiple judges
    pub consensus_threshold: f32,
    /// Weight untuk AI feedback vs human feedback
    pub ai_feedback_weight: f32,
}

impl Default for RlaifConfig {
    fn default() -> Self {
        Self {
            judge_model: "gpt-4".to_string(),
            confidence_threshold: 0.7,
            max_judges: 3,
            consensus_threshold: 0.8,
            ai_feedback_weight: 1.0,
        }
    }
}

/// AI Judge interface
pub trait AiJudge {
    fn evaluate_step(&self, step: &ReasoningStep, context: &ReasoningTrace) -> Result<JudgeEvaluation>;
    fn evaluate_trace(&self, trace: &ReasoningTrace) -> Result<TraceEvaluation>;
    fn compare_steps(&self, step1: &ReasoningStep, step2: &ReasoningStep) -> Result<StepComparison>;
    fn model_name(&self) -> &str;
    fn confidence_level(&self) -> f32;
}

/// Hasil evaluasi step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeEvaluation {
    pub step_id: Uuid,
    pub quality_score: f32,
    pub confidence: f32,
    pub reasoning: String,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Hasil evaluasi trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvaluation {
    pub trace_id: Uuid,
    pub overall_score: f32,
    pub step_evaluations: Vec<JudgeEvaluation>,
    pub coherence_score: f32,
    pub correctness_score: f32,
    pub completeness_score: f32,
    pub reasoning: String,
}

/// Perbandingan antar steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepComparison {
    pub step1_id: Uuid,
    pub step2_id: Uuid,
    pub preferred_step: Uuid,
    pub confidence: f32,
    pub reasoning: String,
    pub comparison_criteria: Vec<ComparisonCriterion>,
}

/// Kriteria perbandingan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonCriterion {
    pub name: String,
    pub step1_score: f32,
    pub step2_score: f32,
    pub weight: f32,
}

/// GPT-4 Judge Implementation
pub struct Gpt4Judge {
    model_name: String,
    confidence: f32,
}

impl Gpt4Judge {
    pub fn new(confidence: f32) -> Self {
        Self {
            model_name: "gpt-4".to_string(),
            confidence,
        }
    }
}

impl AiJudge for Gpt4Judge {
    fn evaluate_step(&self, step: &ReasoningStep, context: &ReasoningTrace) -> Result<JudgeEvaluation> {
        let quality_score = self.calculate_step_quality(step, context)?;
        let confidence = self.confidence;
        
        let (strengths, weaknesses, suggestions) = self.analyze_step_content(&step.content);
        
        Ok(JudgeEvaluation {
            step_id: step.id,
            quality_score,
            confidence,
            reasoning: format!("Step {} shows {} quality with confidence {}", 
                step.step_number, quality_score, confidence),
            strengths,
            weaknesses,
            suggestions,
        })
    }
    
    fn evaluate_trace(&self, trace: &ReasoningTrace) -> Result<TraceEvaluation> {
        let mut step_evaluations = Vec::new();
        let mut total_quality = 0.0;
        
        for step in &trace.steps {
            let evaluation = self.evaluate_step(step, trace)?;
            total_quality += evaluation.quality_score;
            step_evaluations.push(evaluation);
        }
        
        let overall_score = total_quality / trace.steps.len().max(1) as f32;
        let coherence_score = self.calculate_coherence(trace)?;
        let correctness_score = self.calculate_correctness(trace)?;
        let completeness_score = self.calculate_completeness(trace)?;
        
        Ok(TraceEvaluation {
            trace_id: trace.id,
            overall_score,
            step_evaluations,
            coherence_score,
            correctness_score,
            completeness_score,
            reasoning: format!("Trace evaluation with overall score {}", overall_score),
        })
    }
    
    fn compare_steps(&self, step1: &ReasoningStep, step2: &ReasoningStep) -> Result<StepComparison> {
        let score1 = self.calculate_step_quality_simple(step1)?;
        let score2 = self.calculate_step_quality_simple(step2)?;
        
        let preferred_step = if score1 > score2 { step1.id } else { step2.id };
        let confidence = ((score1 - score2).abs() / (score1 + score2).max(0.1)).min(1.0);
        
        let comparison_criteria = vec![
            ComparisonCriterion {
                name: "clarity".to_string(),
                step1_score: self.calculate_clarity(&step1.content),
                step2_score: self.calculate_clarity(&step2.content),
                weight: 0.3,
            },
            ComparisonCriterion {
                name: "relevance".to_string(),
                step1_score: self.calculate_relevance(&step1.content),
                step2_score: self.calculate_relevance(&step2.content),
                weight: 0.4,
            },
            ComparisonCriterion {
                name: "completeness".to_string(),
                step1_score: self.calculate_completeness_score(&step1.content),
                step2_score: self.calculate_completeness_score(&step2.content),
                weight: 0.3,
            },
        ];
        
        Ok(StepComparison {
            step1_id: step1.id,
            step2_id: step2.id,
            preferred_step,
            confidence,
            reasoning: format!("Step {} preferred over {} with confidence {}", 
                preferred_step, if preferred_step == step1.id { step2.id } else { step1.id }, confidence),
            comparison_criteria,
        })
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
    
    fn confidence_level(&self) -> f32 {
        self.confidence
    }
}

impl Gpt4Judge {
    fn calculate_step_quality(&self, step: &ReasoningStep, _context: &ReasoningTrace) -> Result<f32> {
        let mut score = 0.5;
        
        // Length factor
        let length_score = if step.content.len() > 10 && step.content.len() < 200 {
            0.3
        } else {
            0.1
        };
        score += length_score;
        
        // Content quality factors
        let clarity = self.calculate_clarity(&step.content);
        let relevance = self.calculate_relevance(&step.content);
        let completeness = self.calculate_completeness_score(&step.content);
        
        score += clarity * 0.4 + relevance * 0.3 + completeness * 0.3;
        
        Ok((score as f32).min(1.0))
    }
    
    fn calculate_step_quality_simple(&self, step: &ReasoningStep) -> Result<f32> {
        let clarity = self.calculate_clarity(&step.content);
        let relevance = self.calculate_relevance(&step.content);
        let completeness = self.calculate_completeness_score(&step.content);
        
        Ok(clarity * 0.4 + relevance * 0.3 + completeness * 0.3)
    }
    
    fn calculate_clarity(&self, content: &str) -> f32 {
        let mut score = 0.5;
        
        // Check for clear structure
        if content.contains('.') || content.contains(',') {
            score += 0.2;
        }
        
        // Check for explanation words
        let explanation_words = ["because", "therefore", "since", "thus", "hence", "due to"];
        for word in &explanation_words {
            if content.to_lowercase().contains(word) {
                score += 0.1;
            }
        }
        
        // Check sentence length
        let avg_word_length = content.len() / content.split_whitespace().count().max(1);
        if avg_word_length >= 4 && avg_word_length <= 8 {
            score += 0.2;
        }
        
        (score as f32).min(1.0)
    }
    
    fn calculate_relevance(&self, content: &str) -> f32 {
        let mut score = 0.5;
        
        // Check for task-relevant keywords
        let relevant_keywords = ["solve", "answer", "result", "calculate", "determine"];
        for keyword in &relevant_keywords {
            if content.to_lowercase().contains(keyword) {
                score += 0.1;
            }
        }
        
        // Check for focus
        if !content.contains("um") && !content.contains("uh") && !content.contains("like") {
            score += 0.2;
        }
        
        (score as f32).min(1.0)
    }
    
    fn calculate_completeness_score(&self, content: &str) -> f32 {
        let mut score = 0.5;
        
        // Check for complete sentences
        if content.ends_with('.') || content.ends_with('!') || content.ends_with('?') {
            score += 0.2;
        }
        
        // Check for sufficient detail
        if content.len() > 20 {
            score += 0.2;
        }
        
        // Check for logical flow
        if content.contains("and") || content.contains("but") || content.contains("so") {
            score += 0.1;
        }
        
        (score as f32).min(1.0)
    }
    
    fn analyze_step_content(&self, content: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();
        let mut suggestions = Vec::new();
        
        // Analyze strengths
        if content.len() > 50 {
            strengths.push("Detailed explanation".to_string());
        }
        
        if content.to_lowercase().contains("because") {
            strengths.push("Provides reasoning".to_string());
        }
        
        // Analyze weaknesses
        if content.len() < 10 {
            weaknesses.push("Too brief".to_string());
        }
        
        if content.chars().filter(|c| c.is_uppercase()).count() > content.len() / 2 {
            weaknesses.push("Excessive capitalization".to_string());
        }
        
        // Generate suggestions
        if weaknesses.iter().any(|w| w.contains("brief")) {
            suggestions.push("Add more detail to your explanation".to_string());
        }
        
        if !content.contains('.') {
            suggestions.push("Use proper punctuation".to_string());
        }
        
        (strengths, weaknesses, suggestions)
    }
    
    fn calculate_coherence(&self, trace: &ReasoningTrace) -> Result<f32> {
        if trace.steps.len() < 2 {
            return Ok(0.5);
        }
        
        let mut coherence_score = 0.0;
        
        for i in 1..trace.steps.len() {
            let prev_step = &trace.steps[i - 1];
            let curr_step = &trace.steps[i];
            
            // Check logical flow between steps
            let flow_score = self.calculate_logical_flow(&prev_step.content, &curr_step.content);
            coherence_score += flow_score;
        }
        
        Ok(coherence_score / (trace.steps.len() - 1) as f32)
    }
    
    fn calculate_correctness(&self, trace: &ReasoningTrace) -> Result<f32> {
        // Simplified correctness check
        let mut correctness_score = 0.5;
        
        // Check for mathematical consistency
        for step in &trace.steps {
            if step.content.contains('=') {
                correctness_score += 0.1;
            }
        }
        
        // Check for logical consistency
        if trace.steps.len() > 1 {
            correctness_score += 0.2;
        }
        
        Ok((correctness_score as f32).min(1.0))
    }
    
    fn calculate_completeness(&self, trace: &ReasoningTrace) -> Result<f32> {
        let mut completeness_score = 0.5;
        
        // Check if trace has conclusion
        if !trace.final_answer.is_empty() {
            completeness_score += 0.3;
        }
        
        // Check step progression
        if trace.steps.iter().map(|s| s.step_number).collect::<std::collections::HashSet<_>>().len() == trace.steps.len() {
            completeness_score += 0.2;
        }
        
        Ok((completeness_score as f32).min(1.0))
    }
    
    fn calculate_logical_flow(&self, prev_content: &str, curr_content: &str) -> f32 {
        let mut flow_score = 0.5;
        
        // Check for transition words
        let transition_words = ["therefore", "however", "furthermore", "consequently"];
        for word in &transition_words {
            if curr_content.to_lowercase().contains(word) {
                flow_score += 0.2;
            }
        }
        
        // Check for topic continuity
        let prev_words: std::collections::HashSet<_> = prev_content
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
            
        let curr_words: std::collections::HashSet<_> = curr_content
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
            
        let overlap = prev_words.intersection(&curr_words).count();
        if overlap > 0 {
            flow_score += 0.1;
        }
        
        (flow_score as f32).min(1.0)
    }
}

/// RLAIF Manager
pub struct RlaifManager {
    config: RlaifConfig,
    judges: Vec<Box<dyn AiJudge>>,
}

impl RlaifManager {
    pub fn new(config: RlaifConfig) -> Self {
        let mut judges: Vec<Box<dyn AiJudge>> = Vec::new();
        
        // Add default GPT-4 judge
        judges.push(Box::new(Gpt4Judge::new(config.confidence_threshold)));
        
        Self {
            config,
            judges,
        }
    }
    
    /// Add custom judge
    pub fn add_judge(&mut self, judge: Box<dyn AiJudge>) -> Result<()> {
        if self.judges.len() >= self.config.max_judges {
            return Err(anyhow::anyhow!("Maximum number of judges reached"));
        }
        
        self.judges.push(judge);
        Ok(())
    }
    
    /// Generate feedback untuk reasoning trace
    pub fn generate_feedback(&self, trace: &ReasoningTrace) -> Result<Vec<JudgeFeedback>> {
        let mut all_feedback = Vec::new();
        
        // Get feedback from all judges
        for judge in &self.judges {
            let trace_evaluation = judge.evaluate_trace(trace)?;
            let judge_feedback = self.convert_evaluation_to_feedback(&trace_evaluation)?;
            all_feedback.extend(judge_feedback);
        }
        
        // Aggregate feedback if multiple judges
        if self.judges.len() > 1 {
            Ok(self.aggregate_feedback(all_feedback)?)
        } else {
            Ok(all_feedback)
        }
    }
    
    /// Generate pairwise comparisons
    pub fn generate_comparisons(&self, traces: &[ReasoningTrace]) -> Result<Vec<JudgeFeedback>> {
        let mut comparisons = Vec::new();
        
        // Compare all pairs of traces
        for (i, trace1) in traces.iter().enumerate() {
            for trace2 in traces.iter().skip(i + 1) {
                for judge in &self.judges {
                    let comparison = self.compare_traces(judge.as_ref(), trace1, trace2)?;
                    comparisons.push(comparison);
                }
            }
        }
        
        Ok(comparisons)
    }
    
    /// Get judge statistics
    pub fn get_judge_stats(&self) -> JudgeStats {
        let judge_names: Vec<String> = self.judges.iter()
            .map(|j| j.model_name().to_string())
            .collect();
            
        let avg_confidence = self.judges.iter()
            .map(|j| j.confidence_level())
            .sum::<f32>() / self.judges.len().max(1) as f32;
        
        JudgeStats {
            num_judges: self.judges.len(),
            judge_names,
            avg_confidence,
            consensus_threshold: self.config.consensus_threshold,
        }
    }
    
    // Helper methods
    fn convert_evaluation_to_feedback(&self, evaluation: &TraceEvaluation) -> Result<Vec<JudgeFeedback>> {
        let mut feedbacks = Vec::new();
        
        // Convert step evaluations to independent feedback
        for step_eval in &evaluation.step_evaluations {
            let feedback_type = FeedbackType::Independent {
                step_id: step_eval.step_id,
                is_good: step_eval.quality_score >= 0.5,
                confidence: step_eval.confidence,
            };
            
            let feedback = JudgeFeedback {
                id: Uuid::new_v4(),
                trace_id: evaluation.trace_id,
                feedback_type,
                reasoning: step_eval.reasoning.clone(),
                created_at: chrono::Utc::now(),
            };
            
            feedbacks.push(feedback);
        }
        
        Ok(feedbacks)
    }
    
    fn compare_traces(&self, judge: &dyn AiJudge, trace1: &ReasoningTrace, trace2: &ReasoningTrace) -> Result<JudgeFeedback> {
        // Compare overall traces
        let eval1 = judge.evaluate_trace(trace1)?;
        let eval2 = judge.evaluate_trace(trace2)?;
        
        let preferred_trace = if eval1.overall_score > eval2.overall_score {
            trace1.id
        } else {
            trace2.id
        };
        
        let rejected_trace = if preferred_trace == trace1.id { trace2.id } else { trace1.id };
        
        let confidence = ((eval1.overall_score - eval2.overall_score).abs() / 
                         (eval1.overall_score + eval2.overall_score).max(0.1)).min(1.0);
        
        let feedback_type = FeedbackType::Pairwise {
            preferred: preferred_trace,
            rejected: rejected_trace,
            confidence,
        };
        
        Ok(JudgeFeedback {
            id: Uuid::new_v4(),
            trace_id: trace1.id, // Use first trace as reference
            feedback_type,
            reasoning: format!("Trace {} preferred over trace {} with confidence {}", 
                preferred_trace, rejected_trace, confidence),
            created_at: chrono::Utc::now(),
        })
    }
    
    fn aggregate_feedback(&self, feedbacks: Vec<JudgeFeedback>) -> Result<Vec<JudgeFeedback>> {
        let mut aggregated = HashMap::new();
        
        // Group feedback by step/trace
        for feedback in feedbacks {
            let key = match &feedback.feedback_type {
                FeedbackType::Independent { step_id, .. } => format!("independent_{}", step_id),
                FeedbackType::Pairwise { preferred, rejected, .. } => format!("pairwise_{}_{}", preferred, rejected),
            };
            
            aggregated.entry(key).or_insert_with(Vec::new).push(feedback);
        }
        
        // Create consensus feedback
        let mut consensus_feedbacks = Vec::new();
        
        for (key, group_feedbacks) in aggregated {
            if let Some(consensus) = self.create_consensus_feedback(&group_feedbacks)? {
                consensus_feedbacks.push(consensus);
            }
        }
        
        Ok(consensus_feedbacks)
    }
    
    fn create_consensus_feedback(&self, feedbacks: &[JudgeFeedback]) -> Result<Option<JudgeFeedback>> {
        if feedbacks.is_empty() {
            return Ok(None);
        }
        
        // Check if consensus threshold is met
        let consensus_ratio = self.calculate_consensus_ratio(feedbacks)?;
        
        if consensus_ratio < self.config.consensus_threshold {
            return Ok(None); // No consensus
        }
        
        // Create aggregated feedback
        let first_feedback = &feedbacks[0];
        let aggregated_reasoning = feedbacks.iter()
            .map(|f| f.reasoning.clone())
            .collect::<Vec<_>>()
            .join("; ");
        
        let aggregated_feedback = match &first_feedback.feedback_type {
            FeedbackType::Independent { step_id, is_good, confidence } => {
                let avg_confidence = feedbacks.iter()
                    .map(|f| match &f.feedback_type {
                        FeedbackType::Independent { confidence, .. } => *confidence,
                        _ => 0.0,
                    })
                    .sum::<f32>() / feedbacks.len() as f32;
                
                JudgeFeedback {
                    id: Uuid::new_v4(),
                    trace_id: first_feedback.trace_id,
                    feedback_type: FeedbackType::Independent {
                        step_id: *step_id,
                        is_good: *is_good,
                        confidence: avg_confidence,
                    },
                    reasoning: aggregated_reasoning,
                    created_at: chrono::Utc::now(),
                }
            },
            FeedbackType::Pairwise { preferred, rejected, confidence } => {
                let avg_confidence = feedbacks.iter()
                    .map(|f| match &f.feedback_type {
                        FeedbackType::Pairwise { confidence, .. } => *confidence,
                        _ => 0.0,
                    })
                    .sum::<f32>() / feedbacks.len() as f32;
                
                JudgeFeedback {
                    id: Uuid::new_v4(),
                    trace_id: first_feedback.trace_id,
                    feedback_type: FeedbackType::Pairwise {
                        preferred: *preferred,
                        rejected: *rejected,
                        confidence: avg_confidence,
                    },
                    reasoning: aggregated_reasoning,
                    created_at: chrono::Utc::now(),
                }
            },
        };
        
        Ok(Some(aggregated_feedback))
    }
    
    fn calculate_consensus_ratio(&self, feedbacks: &[JudgeFeedback]) -> Result<f32> {
        if feedbacks.len() < 2 {
            return Ok(1.0); // Single feedback is consensus by definition
        }
        
        // Check if all feedbacks agree
        let first_feedback = &feedbacks[0];
        let mut agreements = 0;
        
        for feedback in feedbacks.iter().skip(1) {
            if self.feedback_agrees(first_feedback, feedback) {
                agreements += 1;
            }
        }
        
        Ok(agreements as f32 / (feedbacks.len() - 1) as f32)
    }
    
    fn feedback_agrees(&self, feedback1: &JudgeFeedback, feedback2: &JudgeFeedback) -> bool {
        match (&feedback1.feedback_type, &feedback2.feedback_type) {
            (FeedbackType::Independent { is_good: good1, .. }, 
             FeedbackType::Independent { is_good: good2, .. }) => good1 == good2,
            (FeedbackType::Pairwise { preferred: pref1, .. }, 
             FeedbackType::Pairwise { preferred: pref2, .. }) => pref1 == pref2,
            _ => false,
        }
    }
}

/// Judge statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeStats {
    pub num_judges: usize,
    pub judge_names: Vec<String>,
    pub avg_confidence: f32,
    pub consensus_threshold: f32,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default RLAIF manager
    pub fn create_default_manager() -> RlaifManager {
        RlaifManager::new(RlaifConfig::default())
    }
    
    /// Create mock judge for testing
    pub fn create_mock_judge(name: &str, confidence: f32) -> Box<dyn AiJudge> {
        Box::new(Gpt4Judge::new(confidence))
    }
    
    /// Validate judge feedback
    pub fn validate_feedback(feedbacks: &[JudgeFeedback]) -> Result<()> {
        for feedback in feedbacks {
            match &feedback.feedback_type {
                FeedbackType::Independent { step_id, confidence, .. } => {
                    if *confidence < 0.0 || *confidence > 1.0 {
                        return Err(anyhow::anyhow!("Invalid confidence value: {}", confidence));
                    }
                },
                FeedbackType::Pairwise { preferred, rejected, confidence, .. } => {
                    if preferred == rejected {
                        return Err(anyhow::anyhow!("Preferred and rejected cannot be the same"));
                    }
                    if *confidence < 0.0 || *confidence > 1.0 {
                        return Err(anyhow::anyhow!("Invalid confidence value: {}", confidence));
                    }
                },
            }
        }
        Ok(())
    }
    
    /// Analyze feedback distribution
    pub fn analyze_feedback_distribution(feedbacks: &[JudgeFeedback]) -> FeedbackDistribution {
        let mut independent_count = 0;
        let mut pairwise_count = 0;
        let mut total_confidence = 0.0;
        
        for feedback in feedbacks {
            match &feedback.feedback_type {
                FeedbackType::Independent { confidence, .. } => {
                    independent_count += 1;
                    total_confidence += confidence;
                },
                FeedbackType::Pairwise { confidence, .. } => {
                    pairwise_count += 1;
                    total_confidence += confidence;
                },
            }
        }
        
        let avg_confidence = if feedbacks.is_empty() {
            0.0
        } else {
            total_confidence / feedbacks.len() as f32
        };
        
        FeedbackDistribution {
            total_feedbacks: feedbacks.len(),
            independent_count,
            pairwise_count,
            avg_confidence,
        }
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FeedbackDistribution {
        pub total_feedbacks: usize,
        pub independent_count: usize,
        pub pairwise_count: usize,
        pub avg_confidence: f32,
    }
}
