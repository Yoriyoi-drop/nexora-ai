use crate::foundation::NxrNexumModel;
use crate::nexum::classifier;
use crate::nexum::classifier::ComplexityClassifier;
use nexora_oracle::verifiers::CodeVerifierManager;
use nexora_reasoning::SacaEngine;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static VERIFIER: OnceLock<CodeVerifierManager> = OnceLock::new();

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

fn quality_label(score: f32) -> &'static str {
    if score >= 0.9 { "excellent" }
    else if score >= 0.7 { "good" }
    else if score >= 0.5 { "fair" }
    else { "poor" }
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

    // Phase 4: SACA reasoning for smarter decomposition on complex tasks
    let subtasks = if primary == "complex" || primary == "multi_domain" {
        let engine = SacaEngine::new();
        match engine.reason(prompt, strategy).await {
            Ok(r) if r.conclusion.len() > 20 => {
                let tasks: Vec<String> = r.conclusion
                    .split('\n')
                    .map(|s| s.trim().to_string())
                    .filter(|s| s.len() > 15)
                    .collect();
                if tasks.len() > 1 { tasks } else { decompose(prompt, primary) }
            }
            _ => {
                tracing::warn!("nexum SACA reasoning unavailable, using naive decomposition");
                decompose(prompt, primary)
            }
        }
    } else {
        decompose(prompt, primary)
    };

    if subtasks.len() <= 1 {
        return call(prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    // Phase 4: Oracle verifier for quality checking each subtask result
    let verifier = VERIFIER.get_or_init(CodeVerifierManager::new);
    let mut results: Vec<(String, f32)> = Vec::new();

    for (i, task) in subtasks.iter().enumerate() {
        let sub_prompt = format!(
            "[Nexum subtask {i} | complexity: {primary}]\n\
             {strategy}\n\n\
             {task}"
        );
        match call(&sub_prompt, 256, 0.6).await {
            Ok(r) => {
                let quality = verifier.verify_code(&r, "text").unwrap_or(0.5);
                tracing::debug!("nexum subtask {i} quality: {:.2}", quality);
                let label = quality_label(quality);
                results.push((format!("Task {i} result ({label} quality): {r}"), quality));
            }
            Err(e) => {
                tracing::warn!("nexum subtask {i} failed: {}", e);
            }
        }
    }

    if results.is_empty() {
        return call(prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    let above_threshold = results.iter().filter(|(_, q)| *q >= 0.5).count();
    let needs_synthesis = results.len() > 1 && (primary == "multi_domain" || above_threshold < results.len());

    if needs_synthesis {
        let merged: String = results.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>().join("\n");
        let avg_quality: f32 = results.iter().map(|(_, q)| q).sum::<f32>() / results.len() as f32;
        let synthesis = call(
            &format!(
                "[Nexum synthesis | multi-domain | avg quality: {avg_quality:.2}]\n\
                 Synthesize these partial results into a coherent response.\n\
                 Some results may have lower quality — prioritize high-quality ones.\n\n\
                 {merged}",
            ),
            512,
            0.6,
        )
        .await
        .unwrap_or(merged);
        synthesis
    } else {
        results.into_iter().map(|(r, _)| r).collect::<Vec<_>>().join("\n")
    }
}
