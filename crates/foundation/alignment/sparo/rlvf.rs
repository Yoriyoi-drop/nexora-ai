//! Reinforcement Learning from Verifiable Feedback (RLVF) Implementation
//! 
//! RLVF mengevaluasi kualitas setiap langkah penalaran, bukan hanya jawaban akhir,
//! dengan feedback yang dapat diverifikasi secara otomatis.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::core::{ReasoningStep, ReasoningTrace, JudgeFeedback, FeedbackType};

/// Konfigurasi RLVF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlvfConfig {
    /// Weight untuk step-level feedback
    pub step_weight: f32,
    /// Weight untuk final answer feedback
    pub final_weight: f32,
    /// Threshold untuk verifikasi otomatis
    pub verification_threshold: f32,
    /// Maximum steps per trace
    pub max_steps: usize,
}

impl Default for RlvfConfig {
    fn default() -> Self {
        Self {
            step_weight: 0.8,
            final_weight: 0.2,
            verification_threshold: 0.7,
            max_steps: 10,
        }
    }
}

/// Jenis verifikasi yang dapat dilakukan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationType {
    /// Verifikasi matematis
    Mathematical,
    /// Verifikasi logika
    Logical,
    /// Verifikasi sintaks
    Syntactic,
    /// Verifikasi semantik
    Semantic,
    /// Verifikasi fakta
    Factual,
}

/// Hasil verifikasi untuk satu langkah
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub step_id: Uuid,
    pub verification_type: VerificationType,
    pub is_correct: bool,
    pub confidence: f32,
    pub error_message: Option<String>,
    pub verification_details: HashMap<String, String>,
}

/// Step-level feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepFeedback {
    pub step_id: Uuid,
    pub trace_id: Uuid,
    pub verification_results: Vec<VerificationResult>,
    pub overall_score: f32,
    pub feedback_text: String,
    pub is_verifiable: bool,
}

/// Verifier untuk berbagai jenis verifikasi
pub trait Verifier {
    fn verify(&self, step: &ReasoningStep, context: &ReasoningTrace) -> Result<VerificationResult>;
    fn verification_type(&self) -> VerificationType;
}

/// Mathematical Verifier
pub struct MathematicalVerifier;

impl Verifier for MathematicalVerifier {
    fn verify(&self, step: &ReasoningStep, _context: &ReasoningTrace) -> Result<VerificationResult> {
        let content = &step.content;
        
        // Extract mathematical expressions
        let expressions = self.extract_math_expressions(content);
        let mut is_correct = true;
        let mut error_messages = Vec::new();
        let mut details = HashMap::new();
        
        for expr in &expressions {
            if let Err(e) = self.evaluate_expression(expr) {
                is_correct = false;
                error_messages.push(format!("Error in expression '{}': {}", expr, e));
                details.insert(expr.clone(), "error".to_string());
            } else {
                details.insert(expr.clone(), "correct".to_string());
            }
        }
        
        Ok(VerificationResult {
            step_id: step.id,
            verification_type: VerificationType::Mathematical,
            is_correct,
            confidence: if is_correct { 0.9 } else { 0.1 },
            error_message: if error_messages.is_empty() { None } else { Some(error_messages.join("; ")) },
            verification_details: details,
        })
    }
    
    fn verification_type(&self) -> VerificationType {
        VerificationType::Mathematical
    }
}

impl MathematicalVerifier {
    fn extract_math_expressions(&self, text: &str) -> Vec<String> {
        // Simple regex to extract mathematical expressions
        // In practice, this would be more sophisticated
        let mut expressions = Vec::new();
        
        // Look for patterns like "2 + 3 = 5" or "x = 2 + 3"
        for line in text.lines() {
            if line.contains('=') || line.contains('+') || line.contains('-') || 
               line.contains('*') || line.contains('/') {
                expressions.push(line.trim().to_string());
            }
        }
        
        expressions
    }
    
    fn evaluate_expression(&self, expr: &str) -> Result<f32> {
        // Very simple expression evaluator
        // In practice, this would use a proper math parser
        
        if let Some(eq_pos) = expr.find('=') {
            let left_side = &expr[..eq_pos].trim();
            let right_side = &expr[eq_pos + 1..].trim();
            
            // Try to evaluate both sides
            let left_val = self.simple_eval(left_side)?;
            let right_val = self.simple_eval(right_side)?;
            
            if (left_val - right_val).abs() < 1e-6 {
                Ok(left_val)
            } else {
                Err(anyhow::anyhow!("Mathematical inconsistency: {} ≠ {}", left_val, right_val))
            }
        } else {
            self.simple_eval(expr)
        }
    }
    
    fn simple_eval(&self, expr: &str) -> Result<f32> {
        // Extremely simple evaluator for basic arithmetic
        let expr = expr.replace(" ", "");
        
        if let Ok(val) = expr.parse::<f32>() {
            return Ok(val);
        }
        
        // Handle basic operations
        if let Some(plus_pos) = expr.find('+') {
            let left = &expr[..plus_pos];
            let right = &expr[plus_pos + 1..];
            return Ok(self.simple_eval(left)? + self.simple_eval(right)?);
        }
        
        if let Some(minus_pos) = expr.find('-') {
            let left = &expr[..minus_pos];
            let right = &expr[minus_pos + 1..];
            return Ok(self.simple_eval(left)? - self.simple_eval(right)?);
        }
        
        Err(anyhow::anyhow!("Cannot evaluate expression: {}", expr))
    }
}

/// Logical Verifier
pub struct LogicalVerifier;

impl Verifier for LogicalVerifier {
    fn verify(&self, step: &ReasoningStep, context: &ReasoningTrace) -> Result<VerificationResult> {
        let content = &step.content;
        
        // Check for logical consistency
        let contradictions = self.find_contradictions(content, context);
        let is_correct = contradictions.is_empty();
        
        let mut details = HashMap::new();
        details.insert("contradictions".to_string(), contradictions.len().to_string());
        
        Ok(VerificationResult {
            step_id: step.id,
            verification_type: VerificationType::Logical,
            is_correct,
            confidence: if is_correct { 0.8 } else { 0.2 },
            error_message: if contradictions.is_empty() { None } else { Some(contradictions.join("; ")) },
            verification_details: details,
        })
    }
    
    fn verification_type(&self) -> VerificationType {
        VerificationType::Logical
    }
}

impl LogicalVerifier {
    fn find_contradictions(&self, current_step: &str, context: &ReasoningTrace) -> Vec<String> {
        let mut contradictions = Vec::new();
        
        // Check for contradictions with previous steps
        for prev_step in &context.steps {
            if self.check_contradiction(current_step, &prev_step.content) {
                contradictions.push(format!("Contradiction with step {}: {}", 
                    prev_step.step_number, prev_step.content));
            }
        }
        
        contradictions
    }
    
    fn check_contradiction(&self, step1: &str, step2: &str) -> bool {
        // Simple contradiction detection
        // Look for opposite statements
        
        let step1_lower = step1.to_lowercase();
        let step2_lower = step2.to_lowercase();
        
        // Check for negation patterns
        if step1_lower.contains("not") && step2_lower.contains("not") {
            return false; // Both negative, not necessarily contradictory
        }
        
        // Check for opposite claims
        if (step1_lower.contains("true") && step2_lower.contains("false")) ||
           (step1_lower.contains("yes") && step2_lower.contains("no")) ||
           (step1_lower.contains("correct") && step2_lower.contains("incorrect")) {
            return true;
        }
        
        false
    }
}

/// Syntactic Verifier
pub struct SyntacticVerifier;

impl Verifier for SyntacticVerifier {
    fn verify(&self, step: &ReasoningStep, _context: &ReasoningTrace) -> Result<VerificationResult> {
        let content = &step.content;
        
        // Check for basic syntax errors
        let syntax_errors = self.check_syntax(content);
        let is_correct = syntax_errors.is_empty();
        
        let mut details = HashMap::new();
        details.insert("error_count".to_string(), syntax_errors.len().to_string());
        
        Ok(VerificationResult {
            step_id: step.id,
            verification_type: VerificationType::Syntactic,
            is_correct,
            confidence: if is_correct { 0.95 } else { 0.3 },
            error_message: if syntax_errors.is_empty() { None } else { Some(syntax_errors.join("; ")) },
            verification_details: details,
        })
    }
    
    fn verification_type(&self) -> VerificationType {
        VerificationType::Syntactic
    }
}

impl SyntacticVerifier {
    fn check_syntax(&self, text: &str) -> Vec<String> {
        let mut errors = Vec::new();
        
        // Check for basic syntax issues
        if text.ends_with('.') && text.chars().filter(|&c| c == '.').count() > 1 {
            errors.push("Multiple periods".to_string());
        }
        
        if text.contains("  ") {
            errors.push("Double spaces".to_string());
        }
        
        // Check for unmatched parentheses
        let open_parens = text.chars().filter(|&c| c == '(').count();
        let close_parens = text.chars().filter(|&c| c == ')').count();
        if open_parens != close_parens {
            errors.push("Unmatched parentheses".to_string());
        }
        
        errors
    }
}

/// RLVF Manager
pub struct RlvfManager {
    config: RlvfConfig,
    verifiers: Vec<Box<dyn Verifier>>,
}

impl RlvfManager {
    pub fn new(config: RlvfConfig) -> Self {
        let verifiers: Vec<Box<dyn Verifier>> = vec![
            Box::new(MathematicalVerifier),
            Box::new(LogicalVerifier),
            Box::new(SyntacticVerifier),
        ];
        
        Self {
            config,
            verifiers,
        }
    }
    
    /// Add custom verifier
    pub fn add_verifier(&mut self, verifier: Box<dyn Verifier>) {
        self.verifiers.push(verifier);
    }
    
    /// Verify reasoning trace
    pub fn verify_trace(&self, trace: &ReasoningTrace) -> Result<Vec<StepFeedback>> {
        let mut step_feedbacks = Vec::new();
        
        for step in &trace.steps {
            let mut verification_results = Vec::new();
            
            for verifier in &self.verifiers {
                let result = verifier.verify(step, trace)?;
                verification_results.push(result);
            }
            
            // Calculate overall score
            let overall_score = self.calculate_overall_score(&verification_results);
            
            // Generate feedback text
            let feedback_text = self.generate_feedback_text(&verification_results);
            
            // Check if verifiable
            let is_verifiable = verification_results.iter().any(|r| r.confidence >= self.config.verification_threshold);
            
            let step_feedback = StepFeedback {
                step_id: step.id,
                trace_id: trace.id,
                verification_results,
                overall_score,
                feedback_text,
                is_verifiable,
            };
            
            step_feedbacks.push(step_feedback);
        }
        
        Ok(step_feedbacks)
    }
    
    /// Convert step feedback to judge feedback
    pub fn feedback_to_judge_feedback(&self, step_feedbacks: &[StepFeedback]) -> Vec<JudgeFeedback> {
        let mut judge_feedbacks = Vec::new();
        
        for step_feedback in step_feedbacks {
            // Create independent feedback for verifiable steps
            if step_feedback.is_verifiable {
                let feedback_type = FeedbackType::Independent {
                    step_id: step_feedback.step_id,
                    is_good: step_feedback.overall_score >= 0.5,
                    confidence: step_feedback.overall_score,
                };
                
                let judge_feedback = JudgeFeedback {
                    id: Uuid::new_v4(),
                    trace_id: step_feedback.trace_id,
                    feedback_type,
                    reasoning: step_feedback.feedback_text.clone(),
                    created_at: chrono::Utc::now(),
                };
                
                judge_feedbacks.push(judge_feedback);
            }
        }
        
        judge_feedbacks
    }
    
    /// Get verification statistics
    pub fn get_verification_stats(&self, step_feedbacks: &[StepFeedback]) -> VerificationStats {
        let total_steps = step_feedbacks.len();
        let verifiable_steps = step_feedbacks.iter().filter(|f| f.is_verifiable).count();
        let correct_steps = step_feedbacks.iter().filter(|f| f.overall_score >= 0.5).count();
        
        let avg_score = step_feedbacks.iter()
            .map(|f| f.overall_score)
            .sum::<f32>() / total_steps.max(1) as f32;
        
        let mut verification_type_counts = HashMap::new();
        for feedback in step_feedbacks {
            for result in &feedback.verification_results {
                *verification_type_counts.entry(result.verification_type.clone()).or_insert(0) += 1;
            }
        }
        
        VerificationStats {
            total_steps,
            verifiable_steps,
            correct_steps,
            verifiable_ratio: verifiable_steps as f32 / total_steps.max(1) as f32,
            accuracy: correct_steps as f32 / total_steps.max(1) as f32,
            avg_score,
            verification_type_counts,
        }
    }
    
    // Helper methods
    fn calculate_overall_score(&self, results: &[VerificationResult]) -> f32 {
        if results.is_empty() {
            return 0.5; // Neutral score
        }
        
        let weighted_score: f32 = results.iter()
            .map(|r| r.confidence * if r.is_correct { 1.0 } else { 0.0 })
            .sum();
            
        let total_confidence: f32 = results.iter()
            .map(|r| r.confidence)
            .sum();
            
        if total_confidence > 0.0 {
            weighted_score / total_confidence
        } else {
            0.5
        }
    }
    
    fn generate_feedback_text(&self, results: &[VerificationResult]) -> String {
        let mut feedback_parts = Vec::new();
        
        for result in results {
            if result.is_correct {
                feedback_parts.push(format!("{:?} check passed", result.verification_type));
            } else {
                if let Some(error) = &result.error_message {
                    feedback_parts.push(format!("{:?} error: {}", result.verification_type, error));
                } else {
                    feedback_parts.push(format!("{:?} check failed", result.verification_type));
                }
            }
        }
        
        if feedback_parts.is_empty() {
            "No verification performed".to_string()
        } else {
            feedback_parts.join("; ")
        }
    }
}

/// Verification statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStats {
    pub total_steps: usize,
    pub verifiable_steps: usize,
    pub correct_steps: usize,
    pub verifiable_ratio: f32,
    pub accuracy: f32,
    pub avg_score: f32,
    pub verification_type_counts: HashMap<VerificationType, usize>,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default RLVF manager with all verifiers
    pub fn create_default_manager() -> RlvfManager {
        RlvfManager::new(RlvfConfig::default())
    }
    
    /// Filter verifiable steps
    pub fn filter_verifiable_steps(feedbacks: &[StepFeedback]) -> Vec<&StepFeedback> {
        feedbacks.iter().filter(|f| f.is_verifiable).collect()
    }
    
    /// Aggregate verification results by type
    pub fn aggregate_by_type(results: &[VerificationResult]) -> HashMap<VerificationType, Vec<&VerificationResult>> {
        let mut aggregated = HashMap::new();
        
        for result in results {
            aggregated.entry(result.verification_type.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        aggregated
    }
}
