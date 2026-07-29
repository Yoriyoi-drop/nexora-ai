use nexora_foundation::model_core::foundation::{call_model, FoundationModel};
use nexora_shared::base_model::NxrModelError;
use nexora_shared::base_model::NxrModelResult;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default)]
pub struct WorldModelRuntimeAgent;

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

impl WorldModelRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub async fn update_context(&self, input: &str) -> NxrModelResult<String> {
        let prompt = format!(
            "Update the world model context with the following new information. \
             Identify key entities, relationships, and state changes.\n\nNew information: {input}"
        );
        call_model(foundation(), &prompt, 256, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))
    }

    pub async fn process_input(&self, input: &str) -> NxrModelResult<HashMap<String, serde_json::Value>> {
        let prompt = format!(
            "Extract structured information from the following text. Return as key-value pairs.\n\nText: {input}"
        );
        let result = call_model(foundation(), &prompt, 256, 0.7)
            .await
            .map_err(|e| NxrModelError::Internal(e))?;
        let mut map = HashMap::new();
        map.insert("extracted".to_string(), serde_json::Value::String(result));
        map.insert("status".to_string(), serde_json::Value::String("processed".to_string()));
        Ok(map)
    }
}
