use nexora_model_core::classifier_util;
use nexora_model_core::foundation::NxrNexumModel;
use crate::classifier;
use nexora_oracle::linters::CodeLinterManager;
use nexora_reasoning::SacaEngine;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static LINTER_MGR: OnceLock<CodeLinterManager> = OnceLock::new();

fn foundation() -> &'static NxrNexumModel {
    static F: OnceLock<NxrNexumModel> = OnceLock::new();
    F.get_or_init(NxrNexumModel::new)
}

pub fn inject_model(model_arc: Arc<CausalLM>) {
    foundation().set_model_arc(model_arc);
}

fn init_classifier() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            if let Some(ref embed) = model.token_embedding {
                classifier::init_classifier(embed.clone());
                let _ = INITIALIZED.set(true);
            }
        }
    }
}

static SACA: OnceLock<SacaEngine> = OnceLock::new();

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| nexora_model_core::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_complexity(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_complexity(text, &ids)
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
        let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
        return nexora_model_core::foundation::call_model(foundation(), &sanitized_prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    let subtasks = if primary == "complex" || primary == "multi_domain" {
        let engine = SACA.get_or_init(SacaEngine::new);
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
        let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
        return nexora_model_core::foundation::call_model(foundation(), &sanitized_prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    let linter = LINTER_MGR.get_or_init(CodeLinterManager::new);
    let mut results: Vec<(String, f32)> = Vec::new();

    for (i, task) in subtasks.iter().enumerate() {
        let sanitized_task = classifier_util::sanitize_prompt(task);
        let sub_prompt = format!(
            "[Nexum subtask {i} | complexity: {primary}]\n\
             {strategy}\n\n\
             {sanitized_task}"
        );
        match nexora_model_core::foundation::call_model(foundation(), &sub_prompt, 256, 0.6).await {
            Ok(r) => {
                let quality = linter.verify_code(&r, "text").unwrap_or(0.5);
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
        let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
        return nexora_model_core::foundation::call_model(foundation(), &sanitized_prompt, 512, 0.6).await.unwrap_or_else(|e| {
            tracing::warn!("nexum delegation call failed: {}", e);
            format!("[nexum inference error: {}]", e)
        });
    }

    let above_threshold = results.iter().filter(|(_, q)| *q >= 0.5).count();
    let needs_synthesis = results.len() > 1 && (primary == "multi_domain" || above_threshold < results.len());

    if needs_synthesis {
        let merged: String = results.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>().join("\n");
        let avg_quality: f32 = results.iter().map(|(_, q)| q).sum::<f32>() / results.len() as f32;
        let synthesis = nexora_model_core::foundation::call_model(
            foundation(),
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
