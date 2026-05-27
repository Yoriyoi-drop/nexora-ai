use crate::foundation::NxrGenesisModel;
use crate::genesis::classifier;
use crate::genesis::classifier::QualityClassifier;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

fn foundation() -> &'static NxrGenesisModel {
    static F: OnceLock<NxrGenesisModel> = OnceLock::new();
    F.get_or_init(NxrGenesisModel::new)
}

fn init_classifier() {
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let embed = model.token_embedding.clone();
            QualityClassifier::init(embed);
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_quality(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_quality_focus(text, &ids)
}

async fn call(prompt: &str, max_tokens: usize, temperature: f32) -> Result<String, String> {
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
    let output = foundation().infer(&input).await.map_err(|e| e.to_string())?;
    match output.data {
        OutputData::Text(t) => Ok(t),
        _ => Err("unexpected output type".into()),
    }
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let dimensions = classify_quality(prompt);
    let primary = dimensions.first().map(|(d, _)| d.as_str()).unwrap_or("clarity");
    let focus = classifier::refinement_focus(primary);

    let first = call(prompt, 256, 0.8).await.unwrap_or_else(|e| {
        tracing::warn!("genesis delegation call failed: {}", e);
        format!("[genesis inference error: {}]", e)
    });
    if first.is_empty() {
        return first;
    }
    let improved = call(
        &format!(
            "[Genesis refinement | focus: {primary}]\n\
             {focus}\n\n\
             Original:\n{first}\n\n\
             Refined version:"
        ),
        256,
        0.6,
    )
    .await
    .unwrap_or(first);
    improved
}
