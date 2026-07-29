use nexora_foundation::model_core::foundation::{call_model, FoundationModel};
use nexora_shared::base_model::NxrModelError;
use nexora_shared::base_model::NxrModelResult;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default)]
pub struct Oracle7RuntimeAgent;

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

impl Oracle7RuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub async fn decompose_problem(&self, input: &str) -> NxrModelResult<String> {
        let prompt = format!(
            "Decompose the following problem into a list of clear, sequential sub-problems. \
             Number each sub-problem and keep each one concise.\n\nProblem: {input}"
        );
        call_model(foundation(), &prompt, 512, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))
    }
}
