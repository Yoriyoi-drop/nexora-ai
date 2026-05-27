use crate::foundation::NxrKronosModel;
use crate::kronos::classifier;
use crate::kronos::classifier::TemporalClassifier;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

fn foundation() -> &'static NxrKronosModel {
    static F: OnceLock<NxrKronosModel> = OnceLock::new();
    F.get_or_init(NxrKronosModel::new)
}

fn init_classifier() {
    let f = foundation();
    let guard = f.model.blocking_lock();
    if let Some(ref model) = *guard {
        let embed = model.token_embedding.clone();
        TemporalClassifier::init(embed);
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_temporal(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_temporal_mode(text, &ids)
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
    let modes = classify_temporal(prompt);
    let primary = modes.first().map(|(m, _)| m.as_str()).unwrap_or("evergreen");
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC");
    let framing = classifier::temporal_framing(primary);

    let framed = format!(
        "[Kronos temporal | mode: {primary} | current time: {now}]\n\
         {framing}\n\n\
         Input: {prompt}\n\
         Temporal analysis and response:"
    );
    call(&framed, 512, 0.5).await.unwrap_or_default()
}
