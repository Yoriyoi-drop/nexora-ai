use nexora_model_core::foundation::{call_model, FoundationModel};
use nexora_shared::base_model::NxrModelError;
use nexora_shared::base_model::NxrModelResult;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default)]
pub struct SynthPrimeRuntimeAgent;

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

impl SynthPrimeRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub async fn synthesize(&self, fragments: &[String]) -> NxrModelResult<String> {
        let fragments_block = fragments
            .iter()
            .enumerate()
            .map(|(i, f)| format!("Fragment {}:\n{}", i + 1, f))
            .collect::<Vec<_>>()
            .join("\n\n");
        let prompt = format!(
            "Synthesize the following fragments into a single, cohesive, well-structured response. \
             Eliminate redundancy and ensure logical flow.\n\n{fragments_block}"
        );
        call_model(foundation(), &prompt, 1024, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))
    }
}
