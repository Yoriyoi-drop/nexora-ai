use crate::foundation::NxrNexumModel;
use crate::nexum::classifier;
use crate::nexum::classifier::ComplexityClassifier;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrNexumModel {
    static F: OnceLock<NxrNexumModel> = OnceLock::new();
    F.get_or_init(NxrNexumModel::new)
}

fn init_classifier() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let embed = model.token_embedding.clone();
            ComplexityClassifier::init(embed);
            let _ = INITIALIZED.set(true);
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_complexity(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_complexity(text, &ids)
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

fn decompose(text: &str, complexity: &str) -> Vec<String> {
    match complexity {
        "moderate" => {
            let parts: Vec<&str> = text.split(|c: char| c == '.' || c == ';').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if parts.len() > 3 { parts[..3].to_vec().into_iter().map(String::from).collect() } else { parts.into_iter().map(String::from).collect() }
        }
        "complex" | "multi_domain" => {
            let parts: Vec<&str> = text.split(|c: char| c == '.' || c == ';' || c == ',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            let limit = if complexity == "complex" { 5 } else { 8 };
            if parts.len() > limit { parts[..limit].to_vec().into_iter().map(String::from).collect() } else { parts.into_iter().map(String::from).collect() }
        }
        _ => vec![text.to_string()],
    }
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let levels = classify_complexity(prompt);
    let primary = levels.first().map(|(l, _)| l.as_str()).unwrap_or("simple");
    let strategy = classifier::decomposition_strategy(primary);

    if primary == "simple" {
        return call(prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    let subtasks = decompose(prompt, primary);
    if subtasks.len() <= 1 {
        return call(prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    let mut results = Vec::new();
    for (i, task) in subtasks.iter().enumerate() {
        let sub_prompt = format!(
            "[Nexum subtask {i} | complexity: {primary}]\n\
             {strategy}\n\n\
             {task}"
        );
        if let Ok(r) = call(&sub_prompt, 256, 0.6).await {
            results.push(format!("Task {i} result: {r}"));
        }
    }

    if results.len() > 1 && primary == "multi_domain" {
        let merged = results.join("\n");
        let synthesis = call(
            &format!(
                "[Nexum synthesis | multi-domain]\n\
                 Synthesize these partial results into a coherent response.\n\n\
                 {merged}",
            ),
            512,
            0.6,
        )
        .await
        .unwrap_or(merged);
        synthesis
    } else {
        results.join("\n")
    }
}
