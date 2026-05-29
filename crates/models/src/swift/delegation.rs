use crate::foundation::NxrSwiftModel;
use crate::swift::classifier;
use crate::swift::classifier::TaskClassifier;
use nexora_has_moe_ffn::Router;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrSwiftModel {
    static F: OnceLock<NxrSwiftModel> = OnceLock::new();
    F.get_or_init(NxrSwiftModel::new)
}

fn init_classifier() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let embed = model.token_embedding.clone().unwrap();
            TaskClassifier::init(embed);
            let _ = INITIALIZED.set(true);
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_task(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_task_type(text, &ids)
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
    let tasks = classify_task(prompt);
    let primary = tasks.first().map(|(t, _)| t.as_str()).unwrap_or("qa");
    let (max_tokens, temperature) = classifier::task_params(primary);

    // Phase 4: Route through MoE for latency-aware dispatch
    let moe = Router::new(768, 5, 1);
    let embed_dim = 768usize;
    let avg_vec: Vec<f32> = token_ids(prompt)
        .first()
        .map(|_| vec![0.5f32; embed_dim])
        .unwrap_or_else(|| vec![0.0; embed_dim]);
    let input_array = ndarray::Array2::from_shape_vec((1, embed_dim), avg_vec).unwrap();
    let moe_weights = moe.forward(&input_array);
    let top_expert = (0..moe_weights.shape()[1])
        .max_by(|&a, &b| {
            moe_weights[[0, a]]
                .partial_cmp(&moe_weights[[0, b]])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let expert_route = match top_expert {
        0 => "qa",
        1 => "summarize",
        2 => "translate",
        3 => "generate",
        _ => "analyze",
    };

    let framed = format!(
        "[Swift task | type: {primary} | moe_route: {expert_route}]\n\
         Process this input efficiently:\n\
         {prompt}"
    );
    call(&framed, max_tokens, temperature).await.unwrap_or_else(|e| {
        tracing::warn!("swift delegation call failed: {}", e);
        format!("[swift inference error: {}]", e)
    })
}
