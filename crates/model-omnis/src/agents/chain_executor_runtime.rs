use nexora_foundation::model_core::foundation::{call_model, FoundationModel};
use nexora_shared::base_model::NxrModelError;
use nexora_shared::base_model::NxrModelResult;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default)]
pub struct ChainExecutorRuntimeAgent;

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

impl ChainExecutorRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute_chain(
        &self,
        decomposition: &str,
        meta_reasoning: &str,
    ) -> NxrModelResult<String> {
        let prompt = format!(
            "Execute a chain-of-thought reasoning process based on the following decomposition \
             and meta-reasoning analysis. Work through each sub-problem step by step, \
             showing your reasoning for each step.\n\nDecomposition:\n{decomposition}\n\n\
             Meta-Reasoning Analysis:\n{meta_reasoning}"
        );
        call_model(foundation(), &prompt, 1024, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))
    }
}
