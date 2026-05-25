use std::collections::HashMap;
use nexora_shared::base_model::NxrModelResult;

#[derive(Debug, Clone, Default)]
pub struct ChainExecutorRuntimeAgent;

#[derive(Debug, Clone)]
struct ChainStep {
    description: String,
    reasoning: String,
    verification_status: VerificationStatus,
    coherence_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
enum VerificationStatus {
    Verified,
    NeedsReview,
    Failed,
}

#[derive(Debug, Clone, Default)]
struct ChainState {
    steps: Vec<ChainStep>,
    context: HashMap<String, String>,
    coherence_trace: Vec<f32>,
    artifacts: Vec<String>,
}

impl ChainState {
    fn coherence_with(&self, step_tokens: &[String]) -> f32 {
        if self.context.is_empty() {
            return 1.0;
        }
        let overlap: usize = self
            .context
            .keys()
            .filter(|k| step_tokens.iter().any(|t| k.contains(t.as_str()) || t.contains(k.as_str())))
            .count();
        (overlap as f32) / (self.context.len().max(1) as f32).min(1.0)
    }
}

impl ChainExecutorRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    /// Execute chain-of-thought reasoning by breaking decomposition into steps,
    /// applying reasoning at each step, verifying coherence.
    pub fn execute_chain(
        &self,
        decomposition: &str,
        meta_reasoning: &str,
    ) -> NxrModelResult<String> {
        let mut state = ChainState::default();

        // Parse decomposition into discrete reasoning steps
        let raw_steps: Vec<&str> = decomposition
            .split(|c: char| c == '\n' || c == '.' || c == ';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if raw_steps.is_empty() {
            return Ok("[CHAIN-EXECUTOR] No decomposition steps to execute.".to_string());
        }

        let meta_tokens: Vec<String> = meta_reasoning
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        for (i, step_text) in raw_steps.iter().enumerate() {
            let step_tokens: Vec<String> = step_text
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect();

            // Apply reasoning at this step
            let reasoning = self.apply_step_reasoning(step_text, &meta_tokens, &step_tokens);

            // Verify the step
            let (verification, coherence) =
                self.verify_step(step_text, i, &state, &step_tokens);

            state.steps.push(ChainStep {
                description: step_text.to_string(),
                reasoning,
                verification_status: verification,
                coherence_score: coherence,
            });
            state.coherence_trace.push(coherence);

            // Track key concepts as context for subsequent steps
            let Some(last_step) = state.steps.last() else { continue; };
            for concept in last_step.reasoning.split_whitespace() {
                if concept.len() > 3 {
                    state
                        .context
                        .entry(concept.to_lowercase())
                        .and_modify(|v| *v = format!("{}; ref_{}", v, i))
                        .or_insert_with(|| format!("step_{}", i));
                }
            }

            state.artifacts.push(step_text.to_string());
        }

        let avg_coherence: f32 = if state.coherence_trace.is_empty() {
            0.0
        } else {
            state.coherence_trace.iter().sum::<f32>() / state.coherence_trace.len() as f32
        };

        let verified_count = state
            .steps
            .iter()
            .filter(|s| s.verification_status == VerificationStatus::Verified)
            .count();
        let failed_count = state
            .steps
            .iter()
            .filter(|s| s.verification_status == VerificationStatus::Failed)
            .count();

        let mut result = format!(
            "[CHAIN-EXECUTOR] Chain execution complete:\n\
             - {} steps decomposed and executed\n\
             - {} steps verified, {} needs review, {} failed\n\
             - Average coherence: {:.2}\n\
             Steps:\n",
            state.steps.len(),
            verified_count,
            state.steps.iter().filter(|s| s.verification_status == VerificationStatus::NeedsReview).count(),
            failed_count,
            avg_coherence,
        );

        for step in &state.steps {
            result.push_str(&format!(
                "  [{:?}] coherence={:.2}: {}\n    -> {}\n",
                step.verification_status,
                step.coherence_score,
                step.description,
                step.reasoning,
            ));
        }

        Ok(result)
    }

    fn apply_step_reasoning(
        &self,
        step_text: &str,
        meta_tokens: &[String],
        step_tokens: &[String],
    ) -> String {
        let mut reasoning = String::new();

        // Check how the step relates to meta-reasoning
        let meta_overlap: usize = meta_tokens
            .iter()
            .filter(|mt| step_tokens.iter().any(|st| st.contains(mt.as_str())))
            .count();

        if meta_overlap > 2 {
            reasoning.push_str(&format!(
                "Strong alignment with meta-reasoning ({} overlapping concepts). ",
                meta_overlap
            ));
        } else if meta_overlap > 0 {
            reasoning.push_str(&format!(
                "Partial alignment with meta-reasoning ({} overlapping concepts). ",
                meta_overlap
            ));
        }

        // Analyze step type and apply reasoning
        if step_text.contains("==>") || step_text.contains("→") || step_text.contains("->") {
            reasoning.push_str("Transition step: propagating state across boundary. ");
        } else if step_tokens.iter().any(|t| *t == "if" || *t == "when" || *t == "unless") {
            reasoning.push_str("Conditional step: evaluating branching logic. ");
        }

        if reasoning.is_empty() {
            reasoning.push_str("Direct execution step.");
        }

        reasoning
    }

    fn verify_step(
        &self,
        _step_text: &str,
        step_index: usize,
        state: &ChainState,
        step_tokens: &[String],
    ) -> (VerificationStatus, f32) {
        if step_index == 0 {
            return (VerificationStatus::Verified, 1.0);
        }

        let coherence = state.coherence_with(step_tokens);

        if coherence > 0.3 {
            (VerificationStatus::Verified, coherence)
        } else if coherence > 0.1 {
            (VerificationStatus::NeedsReview, coherence)
        } else {
            (VerificationStatus::Failed, coherence)
        }
    }
}
