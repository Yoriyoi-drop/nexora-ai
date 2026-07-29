use crate::MetaReasoningState;
use nexora_foundation::model_core::foundation::{call_model, FoundationModel};
use nexora_shared::base_model::NxrModelError;
use nexora_shared::base_model::NxrModelResult;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default)]
pub struct MetaReasonerRuntimeAgent;

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

impl MetaReasonerRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub async fn analyze_approach(&self, decomposition: &str) -> NxrModelResult<String> {
        let prompt = format!(
            "Analyze the following problem decomposition and suggest the best approach for each sub-problem. \
             Consider different reasoning strategies and potential challenges.\n\nDecomposition:\n{decomposition}"
        );
        call_model(foundation(), &prompt, 512, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))
    }

    pub async fn analyze_problem(&self, problem: &str) -> NxrModelResult<MetaReasoningState> {
        let prompt = format!(
            "Analyze this problem step by step. Provide your reasoning chain, confidence level, \
             and any hypotheses you can form.\n\nProblem: {problem}"
        );
        let result = call_model(foundation(), &prompt, 512, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))?;
        Ok(MetaReasoningState {
            reasoning_chain: vec![crate::ReasoningStep {
                id: uuid::Uuid::new_v4(),
                step_type: crate::ReasoningStepType::MetaReasoning,
                content: result,
                confidence: 0.8,
                dependencies: Vec::new(),
                timestamp: chrono::Utc::now(),
            }],
            confidence_scores: vec![0.8],
            hypothesis_space: vec![],
            truth_arbitration: crate::TruthArbitrationState::default(),
        })
    }

    pub async fn stream_reasoning(&self, input: &str) -> NxrModelResult<Vec<String>> {
        let prompt = format!(
            "Provide step-by-step reasoning for the following. List each step on a new line starting with 'Step N:'.\n\n{input}"
        );
        let result = call_model(foundation(), &prompt, 512, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))?;
        let steps: Vec<String> = result
            .lines()
            .filter(|l| l.trim().starts_with("Step"))
            .map(|l| l.to_string())
            .collect();
        if steps.is_empty() {
            Ok(vec![result])
        } else {
            Ok(steps)
        }
    }
}
