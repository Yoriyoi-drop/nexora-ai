//! Reasoning Module - High-level reasoning capabilities

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use nexora_foundation::FoundationResult;

/// Reasoning chain for complex problem solving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub steps: Vec<ReasoningStep>,
    pub conclusion: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: usize,
    pub premise: String,
    pub inference: String,
    pub confidence: f32,
    pub step_type: ReasoningType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningType {
    Deductive,
    Inductive,
    Abductive,
    Analogical,
    Causal,
}

/// High-level reasoning engine trait
#[async_trait]
pub trait ReasoningEngine: Send + Sync {
    /// Perform reasoning on a problem
    async fn reason(&self, problem: &str, context: serde_json::Value) -> FoundationResult<ReasoningChain>;
    
    /// Verify reasoning chain
    async fn verify(&self, chain: &ReasoningChain) -> FoundationResult<bool>;
    
    /// Get alternative reasoning paths
    async fn alternatives(&self, problem: &str, context: serde_json::Value) -> FoundationResult<Vec<ReasoningChain>>;
    
    /// Explain reasoning step
    async fn explain_step(&self, step: &ReasoningStep) -> FoundationResult<String>;
}

/// Chain-of-thought reasoning implementation
pub struct ChainOfThoughtReasoner;

impl ChainOfThoughtReasoner {
    fn decompose_problem(&self, problem: &str) -> Vec<String> {
        let mut steps = Vec::new();
        let sentences: Vec<&str> = problem
            .split(|c: char| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for sentence in &sentences {
            let parts: Vec<&str> = sentence
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            steps.extend(parts.into_iter().map(|p| p.to_string()));
        }

        if steps.is_empty() {
            steps.push(problem.to_string());
        }
        steps
    }

    fn infer_step(&self, premise: &str, step_index: usize, total_steps: usize) -> String {
        let connective = match step_index {
            0 => "Given that".to_string(),
            i if i == total_steps - 1 => "Therefore".to_string(),
            _ => "Furthermore".to_string(),
        };
        format!("{} {}, we can infer that this leads to the next logical step", connective, premise)
    }
}

#[async_trait]
impl ReasoningEngine for ChainOfThoughtReasoner {
    async fn reason(&self, problem: &str, _context: serde_json::Value) -> FoundationResult<ReasoningChain> {
        let premises = self.decompose_problem(problem);
        let total = premises.len();
        let mut steps = Vec::with_capacity(total);

        for (i, premise) in premises.iter().enumerate() {
            let inference = self.infer_step(premise, i, total);
            steps.push(ReasoningStep {
                step_id: i + 1,
                premise: premise.clone(),
                inference,
                confidence: 1.0 - (i as f32 * 0.05).min(0.5),
                step_type: match i {
                    0 => ReasoningType::Deductive,
                    i if i == total - 1 => ReasoningType::Abductive,
                    _ => ReasoningType::Inductive,
                },
            });
        }

        let conclusion = steps.last().map(|s| s.inference.clone()).unwrap_or_default();
        let avg_confidence = if steps.is_empty() {
            0.0
        } else {
            steps.iter().map(|s| s.confidence).sum::<f32>() / steps.len() as f32
        };

        Ok(ReasoningChain {
            steps,
            conclusion,
            confidence: avg_confidence,
        })
    }
    
    async fn verify(&self, chain: &ReasoningChain) -> FoundationResult<bool> {
        if chain.steps.is_empty() {
            return Ok(false);
        }
        if chain.conclusion.is_empty() {
            return Ok(false);
        }
        let valid_ids: Vec<usize> = (1..=chain.steps.len()).collect();
        let has_all_ids = chain.steps.iter().all(|s| valid_ids.contains(&s.step_id));
        let has_confidence = chain.steps.iter().all(|s| (0.0..=1.0).contains(&s.confidence));
        Ok(has_all_ids && has_confidence)
    }
    
    async fn alternatives(&self, problem: &str, _context: serde_json::Value) -> FoundationResult<Vec<ReasoningChain>> {
        let mut alternatives = Vec::new();

        // Deductive alternative
        let deductive: ReasoningChain = self.reason(problem, serde_json::Value::Null).await?;
        alternatives.push(deductive);

        // Inductive alternative — reversed premise order
        let premises = self.decompose_problem(problem);
        if premises.len() > 1 {
            let reversed: Vec<&str> = premises.iter().rev().map(|s| s.as_str()).collect();
            let reversed_problem = reversed.join(". ");
            let inductive: ReasoningChain = self.reason(&reversed_problem, serde_json::Value::Null).await?;
            alternatives.push(inductive);
        }

        Ok(alternatives)
    }
    
    async fn explain_step(&self, step: &ReasoningStep) -> FoundationResult<String> {
        Ok(format!(
            "Step {} ({:?}): From '{}', we infer: {} (confidence: {:.2})",
            step.step_id, step.step_type, step.premise, step.inference, step.confidence
        ))
    }
}
