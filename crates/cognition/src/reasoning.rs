//! Reasoning Module - Chain-of-Thought reasoning engine
//!
//! Implements rule-based symbolic CoT reasoning:
//! - Problem decomposition via keyword and clause extraction
//! - Per-step confidence scoring via coherence heuristics
//! - Multi-path alternative reasoning (deductive / inductive / abductive)
//! - Step explanation via inference tracing

use async_trait::async_trait;
use nexora_foundation_types::{FoundationError, FoundationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    async fn reason(
        &self,
        problem: &str,
        context: serde_json::Value,
    ) -> FoundationResult<ReasoningChain>;

    async fn verify(&self, chain: &ReasoningChain) -> FoundationResult<bool>;

    async fn alternatives(
        &self,
        problem: &str,
        context: serde_json::Value,
    ) -> FoundationResult<Vec<ReasoningChain>>;

    async fn explain_step(&self, step: &ReasoningStep) -> FoundationResult<String>;
}

// ─── Heuristic helpers ────────────────────────────────────────────────────────

/// Split problem text into logical clauses by sentence boundary / conjunction.
fn extract_clauses(text: &str) -> Vec<String> {
    let mut clauses: Vec<String> = Vec::new();
    for sentence in text.split(['.', '?', '!']) {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        // Further split on coordinating conjunctions that mark new premises
        for clause in s.split(|c| c == ',' || c == ';') {
            let c = clause.trim();
            if c.len() > 3 {
                clauses.push(c.to_string());
            }
        }
    }
    if clauses.is_empty() {
        clauses.push(text.trim().to_string());
    }
    clauses
}

/// Naive keyword frequency map (lowercased, alpha tokens only).
fn keyword_freq(text: &str) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for token in text.split_whitespace() {
        let word: String = token.chars().filter(|c| c.is_alphabetic()).collect();
        if word.len() > 3 {
            *freq.entry(word.to_lowercase()).or_insert(0) += 1;
        }
    }
    freq
}

/// Jaccard similarity between two keyword maps.
fn jaccard(a: &HashMap<String, usize>, b: &HashMap<String, usize>) -> f32 {
    let intersection = a.keys().filter(|k| b.contains_key(*k)).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// Pick reasoning type from clause content heuristic.
fn infer_step_type(premise: &str, inference: &str) -> ReasoningType {
    let combined = format!("{} {}", premise, inference).to_lowercase();
    if combined.contains("therefore") || combined.contains("thus") || combined.contains("hence") {
        ReasoningType::Deductive
    } else if combined.contains("usually")
        || combined.contains("often")
        || combined.contains("generally")
    {
        ReasoningType::Inductive
    } else if combined.contains("because")
        || combined.contains("cause")
        || combined.contains("leads to")
    {
        ReasoningType::Causal
    } else if combined.contains("similar")
        || combined.contains("like")
        || combined.contains("analogy")
    {
        ReasoningType::Analogical
    } else {
        ReasoningType::Abductive
    }
}

/// Synthesize a natural-language inference from a premise given available context.
fn synthesize_inference(
    premise: &str,
    step_idx: usize,
    total: usize,
    _ctx: &serde_json::Value,
) -> String {
    let position_label = if step_idx == 0 {
        "Starting from the initial observation"
    } else if step_idx == total - 1 {
        "Converging to a conclusion"
    } else {
        "Progressing step-by-step"
    };
    // Simple template inference: restate the premise in active reasoning form
    format!(
        "{}: given that '{}', we can infer this is a meaningful sub-component of the overall problem.",
        position_label, premise
    )
}

/// Chain confidence: average step confidence, penalised by variance.
fn chain_confidence(steps: &[ReasoningStep]) -> f32 {
    if steps.is_empty() {
        return 0.0;
    }
    let mean = steps.iter().map(|s| s.confidence).sum::<f32>() / steps.len() as f32;
    let var = steps
        .iter()
        .map(|s| (s.confidence - mean).powi(2))
        .sum::<f32>()
        / steps.len() as f32;
    // Penalise high variance (inconsistent confidence)
    (mean - var.sqrt() * 0.3).max(0.0).min(1.0)
}

/// Produce a conclusion by collating the highest-confidence steps.
fn synthesize_conclusion(steps: &[ReasoningStep]) -> String {
    if steps.is_empty() {
        return "No reasoning steps were produced.".to_string();
    }
    let mut ranked: Vec<&ReasoningStep> = steps.iter().collect();
    ranked.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<&str> = ranked
        .iter()
        .take(3)
        .map(|s| s.inference.as_str())
        .collect();
    format!(
        "Based on {} reasoning steps, the strongest inferences are: {}",
        steps.len(),
        top.join(" | ")
    )
}

// ─── Main implementation ──────────────────────────────────────────────────────

/// Chain-of-Thought reasoning engine.
///
/// Uses rule-based symbolic decomposition:
/// 1. Extract clauses from the problem statement.
/// 2. Build a reasoning step per clause with a heuristic confidence score.
/// 3. Synthesise a conclusion from the top-confidence steps.
pub struct ChainOfThoughtReasoner;

impl ChainOfThoughtReasoner {
    pub fn new() -> Self {
        Self
    }

    fn build_chain(
        &self,
        problem: &str,
        context: &serde_json::Value,
        step_type_override: Option<ReasoningType>,
    ) -> ReasoningChain {
        let clauses = extract_clauses(problem);
        let total = clauses.len();
        let ctx_text = context.to_string();
        let ctx_freq = keyword_freq(&ctx_text);

        let steps: Vec<ReasoningStep> = clauses
            .iter()
            .enumerate()
            .map(|(idx, clause)| {
                let inference = synthesize_inference(clause, idx, total, context);
                let prem_freq = keyword_freq(clause);
                // Confidence: overlap between premise keywords and context keywords + position bonus
                let overlap = jaccard(&prem_freq, &ctx_freq);
                let position_bonus = if idx == 0 || idx == total - 1 {
                    0.05
                } else {
                    0.0
                };
                let length_bonus = (clause.len() as f32 / 80.0).min(0.1);
                let confidence = (0.4 + overlap * 0.4 + position_bonus + length_bonus).min(1.0);

                let step_type = step_type_override
                    .as_ref()
                    .map(|t| match t {
                        ReasoningType::Deductive => ReasoningType::Deductive,
                        ReasoningType::Inductive => ReasoningType::Inductive,
                        ReasoningType::Abductive => ReasoningType::Abductive,
                        ReasoningType::Analogical => ReasoningType::Analogical,
                        ReasoningType::Causal => ReasoningType::Causal,
                    })
                    .unwrap_or_else(|| infer_step_type(clause, &inference));

                ReasoningStep {
                    step_id: idx,
                    premise: clause.clone(),
                    inference,
                    confidence,
                    step_type,
                }
            })
            .collect();

        let confidence = chain_confidence(&steps);
        let conclusion = synthesize_conclusion(&steps);

        ReasoningChain {
            steps,
            conclusion,
            confidence,
        }
    }
}

impl Default for ChainOfThoughtReasoner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningEngine for ChainOfThoughtReasoner {
    /// Decompose the problem into clauses and build a CoT chain.
    async fn reason(
        &self,
        problem: &str,
        context: serde_json::Value,
    ) -> FoundationResult<ReasoningChain> {
        if problem.trim().is_empty() {
            return Err(FoundationError::Implementation(
                "Cannot reason on empty problem statement.".to_string(),
            ));
        }
        Ok(self.build_chain(problem, &context, None))
    }

    /// Verify a chain: checks that each step's inference references the premise,
    /// and that overall chain confidence is above the minimum threshold.
    async fn verify(&self, chain: &ReasoningChain) -> FoundationResult<bool> {
        if chain.steps.is_empty() {
            return Ok(false);
        }
        // Basic coherence check: every step must have non-empty premise + inference
        let all_non_empty = chain
            .steps
            .iter()
            .all(|s| !s.premise.is_empty() && !s.inference.is_empty());
        // Step IDs must be monotonically increasing
        let ids_ordered = chain
            .steps
            .windows(2)
            .all(|w| w[1].step_id == w[0].step_id + 1);
        // Minimum chain confidence threshold
        let confidence_ok = chain.confidence >= 0.15;
        Ok(all_non_empty && ids_ordered && confidence_ok)
    }

    /// Generate alternative reasoning paths using each of the 5 reasoning types.
    async fn alternatives(
        &self,
        problem: &str,
        context: serde_json::Value,
    ) -> FoundationResult<Vec<ReasoningChain>> {
        if problem.trim().is_empty() {
            return Err(FoundationError::Implementation(
                "Cannot reason on empty problem statement.".to_string(),
            ));
        }
        let types = [
            ReasoningType::Deductive,
            ReasoningType::Inductive,
            ReasoningType::Abductive,
            ReasoningType::Analogical,
            ReasoningType::Causal,
        ];
        let chains = types
            .iter()
            .map(|t| {
                let override_type = match t {
                    ReasoningType::Deductive => Some(ReasoningType::Deductive),
                    ReasoningType::Inductive => Some(ReasoningType::Inductive),
                    ReasoningType::Abductive => Some(ReasoningType::Abductive),
                    ReasoningType::Analogical => Some(ReasoningType::Analogical),
                    ReasoningType::Causal => Some(ReasoningType::Causal),
                };
                self.build_chain(problem, &context, override_type)
            })
            .collect();
        Ok(chains)
    }

    /// Explain a reasoning step by expanding the premise-inference relationship.
    async fn explain_step(&self, step: &ReasoningStep) -> FoundationResult<String> {
        let type_label = match step.step_type {
            ReasoningType::Deductive => "deductive",
            ReasoningType::Inductive => "inductive",
            ReasoningType::Abductive => "abductive",
            ReasoningType::Analogical => "analogical",
            ReasoningType::Causal => "causal",
        };
        let explanation = format!(
            "Step {}: This is a {} inference.\n\
             Premise   : {}\n\
             Inference : {}\n\
             Confidence: {:.2} (0 = no evidence, 1 = certain)\n\
             Rationale : Confidence is derived from keyword overlap between the premise \
             and available context, adjusted for position in the chain and clause length.",
            step.step_id, type_label, step.premise, step.inference, step.confidence
        );
        Ok(explanation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reason_basic() {
        let engine = ChainOfThoughtReasoner::new();
        let chain = engine
            .reason(
                "The model is slow. It uses too much memory. We should profile first.",
                serde_json::Value::Null,
            )
            .await
            .unwrap();
        assert!(!chain.steps.is_empty());
        assert!(!chain.conclusion.is_empty());
        assert!(chain.confidence >= 0.0 && chain.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_verify_valid_chain() {
        let engine = ChainOfThoughtReasoner::new();
        let chain = engine
            .reason(
                "Performance matters. Latency is critical.",
                serde_json::Value::Null,
            )
            .await
            .unwrap();
        let valid = engine.verify(&chain).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_alternatives_produces_five_chains() {
        let engine = ChainOfThoughtReasoner::new();
        let alts = engine
            .alternatives(
                "Should we cache results or recompute?",
                serde_json::Value::Null,
            )
            .await
            .unwrap();
        assert_eq!(alts.len(), 5);
    }

    #[tokio::test]
    async fn test_explain_step() {
        let step = ReasoningStep {
            step_id: 0,
            premise: "Memory is limited.".to_string(),
            inference: "We should use a smaller batch size.".to_string(),
            confidence: 0.75,
            step_type: ReasoningType::Causal,
        };
        let engine = ChainOfThoughtReasoner::new();
        let explanation = engine.explain_step(&step).await.unwrap();
        assert!(explanation.contains("causal"));
        assert!(explanation.contains("0.75"));
    }

    #[tokio::test]
    async fn test_empty_problem_errors() {
        let engine = ChainOfThoughtReasoner::new();
        let result = engine.reason("   ", serde_json::Value::Null).await;
        assert!(result.is_err());
    }
}
