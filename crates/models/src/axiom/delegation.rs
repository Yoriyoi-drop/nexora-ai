use crate::axiom::classifier;
use crate::axiom::classifier::ReasoningClassifier;
use crate::foundation::NxrAxiomModel;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

fn foundation() -> &'static NxrAxiomModel {
    static F: OnceLock<NxrAxiomModel> = OnceLock::new();
    F.get_or_init(NxrAxiomModel::new)
}

fn init_classifier() {
    let f = foundation();
    let guard = f.model.blocking_lock();
    if let Some(ref model) = *guard {
        let embed = model.token_embedding.clone();
        ReasoningClassifier::init(embed);
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_reasoning(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_reasoning_type(text, &ids)
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
    let types = classify_reasoning(prompt);
    let primary = types.first().map(|(r, _)| r.as_str()).unwrap_or("analytical");
    let focus = classifier::reasoning_prompt(primary);

    let framed = format!(
        "[Axiom reasoning | type: {primary}]\n\
         {focus}\n\n\
         Input: {prompt}\n\
         Conclusion:"
    );
    call(&framed, 512, 0.3).await.unwrap_or_default()
}
