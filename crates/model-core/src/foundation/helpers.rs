//! Helper utilities for interacting with NxrModel trait implementations.
//!
//! Single responsibility: provide convenience functions for model inference
//! (parameter extraction, simplified call wrappers).

use serde_json::Value;
use std::collections::HashMap;

use nexora_shared::base_model::{InputData, NxrInput, NxrModel, OutputData};

/// Convenience wrapper: build an `NxrInput` from a prompt string, call `model.infer()`,
/// and return the generated text.
pub async fn call_model<M: NxrModel<Config = Value, Metrics = Value, State = Value>>(
    model: &M,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, String> {
    let input = NxrInput {
        id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        data: InputData::Text(prompt.to_string()),
        parameters: HashMap::from([
            ("max_tokens".into(), serde_json::json!(max_tokens)),
            ("temperature".into(), serde_json::json!(temperature)),
            ("top_k".into(), serde_json::json!(50)),
        ]),
        metadata: HashMap::new(),
    };
    let output = model.infer(&input).await.map_err(|e| e.to_string())?;
    match output.data {
        OutputData::Text(t) => Ok(t),
        _ => Err("unexpected output type".into()),
    }
}

/// Extract `(max_tokens, temperature, top_k)` from an `NxrInput` parameter map.
pub fn extract_params(input: &NxrInput) -> (usize, f32, usize) {
    let max_tokens = input
        .parameters
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    let temperature = input
        .parameters
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    let top_k = input
        .parameters
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    (max_tokens, temperature, top_k)
}
