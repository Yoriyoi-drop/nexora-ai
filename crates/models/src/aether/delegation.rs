use crate::aether::classifier;
use crate::aether::classifier::EmotionClassifier;
use crate::foundation::NxrAetherModel;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

fn foundation() -> &'static NxrAetherModel {
    static F: OnceLock<NxrAetherModel> = OnceLock::new();
    F.get_or_init(NxrAetherModel::new)
}

fn init_classifier() {
    let f = foundation();
    let guard = f.model.blocking_lock();
    if let Some(ref model) = *guard {
        let embed = model.token_embedding.clone();
        classifier::EmotionClassifier::init(embed);
    }
}

fn classify(text: &str) -> Vec<(String, f32)> {
    let f = foundation();
    let guard = f.model.blocking_lock();
    if let Some(ref model) = *guard {
        let tokenizer = f.tokenizer.as_ref();
        let ids = tokenizer.map_or_else(
            || crate::foundation::byte_encode(text),
            |tk| tk.read().encode(text),
        );
        let clf = EmotionClassifier::global();
        return clf.predict(&ids);
    }
    vec![("neutral".to_string(), 1.0)]
}

fn dominant_emotion(emotions: &[(String, f32)]) -> String {
    emotions.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(e, _)| e.clone()).unwrap_or_default()
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
    let emotions = classify(prompt);
    let dominant = dominant_emotion(&emotions);
    let framed = format!(
        "[Empathetic response | detected emotion: {dominant}]\n\
         Consider the emotional context and psychological aspects.\n\
         User input: {prompt}\n\
         Response (empathetic, emotionally aware):"
    );
    call(&framed, 512, 0.8).await.unwrap_or_default()
}