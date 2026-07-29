use nexora_foundation::model_core::foundation::{call_model, FoundationModel};
use nexora_shared::base_model::NxrModelError;
use nexora_shared::base_model::NxrModelResult;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default)]
pub struct TruthArbiterRuntimeAgent;

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

impl TruthArbiterRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub async fn arbitrate(&self, claim_a: &str, claim_b: &str) -> NxrModelResult<String> {
        let prompt = format!(
            "Arbitrate between the following two claims. Identify contradictions, \
             areas of agreement, and determine which claim is more accurate or \
             how they can be reconciled.\n\nClaim A:\n{claim_a}\n\nClaim B:\n{claim_b}"
        );
        call_model(foundation(), &prompt, 512, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))
    }
}
