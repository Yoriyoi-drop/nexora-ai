use nexora_foundation::model_core::delegation_base;
use nexora_foundation::model_core::foundation::FoundationModel;
use crate::classifier;
use nexora_cognition::saca::SacaEngine;
use nexora_cognition::reflection::{DefaultReflector, ReflectionEngine, Action, ReflectionType};
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

const MAX_REFINEMENT_ITERATIONS: usize = 3;
const QUALITY_CONFIDENCE: f32 = 0.65;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::genesis)
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
static REFLECTOR: OnceLock<DefaultReflector> = OnceLock::new();

fn token_ids(text: &str) -> Vec<u32> {
    delegation_base::token_ids(foundation(), text)
}

fn classify_quality(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_quality_focus(text, &ids)
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let dims = classify_quality(prompt);
    let primary = dims.first().map(|(d, _)| d.as_str()).unwrap_or("clarity");
    let focus = classifier::refinement_focus(primary);

    let engine = SACA.get_or_init(SacaEngine::new);
    let mut current = match engine.reason(prompt, focus).await {
        Ok(r) if !r.conclusion.is_empty() => {
            tracing::debug!("genesis SACA generation succeeded (focus: {primary})");
            r.conclusion
        }
        _ => {
            tracing::warn!("genesis SACA reasoning unavailable, using direct generation");
            let sanitized_prompt = delegation_base::sanitize_prompt(prompt);
            delegation_base::call_model(foundation(), &sanitized_prompt, 256, 0.8).await.unwrap_or_else(|e| {
                format!("[genesis inference error: {}]", e)
            })
        }
    };

    if current.is_empty() {
        return current;
    }

    let reflector = REFLECTOR.get_or_init(DefaultReflector::new);

    for iteration in 0..MAX_REFINEMENT_ITERATIONS {
        let actions = vec![Action {
            action_type: format!("genesis_generate_{primary}"),
            input: prompt.to_string(),
            output: current.clone(),
            timestamp: now_unix(),
            success: current.len() > 20,
        }];

        let reflection = match reflector.reflect(&actions, &format!("quality refinement iteration {iteration}, focus: {primary}")).await {
            Ok(r) => r,
            Err(_) => break,
        };

        if reflection.confidence >= QUALITY_CONFIDENCE {
            tracing::debug!("genesis quality confidence met (iter {iteration}, confidence: {:.2})", reflection.confidence);
            break;
        }

        let improvements = reflector.suggest_improvements(&reflection).await.unwrap_or_default();
        let refinement_hint = improvements.first().map(|s| s.as_str()).unwrap_or("improve clarity");

        tracing::info!("genesis refinement iter {iteration}: confidence {:.2}, focus: {refinement_hint}", reflection.confidence);

        match engine.reason(&current, refinement_hint).await {
            Ok(r) if !r.conclusion.is_empty() => {
                current = r.conclusion;
            }
            _ => {
                let sanitized_current = delegation_base::sanitize_prompt(&current);
                let improved = delegation_base::call_model(
                    foundation(),
                    &format!(
                        "[Genesis refinement | iter {iteration} | focus: {refinement_hint}]\n\
                         Original:\n{sanitized_current}\n\n\
                         Refined version:"
                    ),
                    256,
                    0.6,
                )
                .await
                .unwrap_or(current.clone());
                current = improved;
            }
        }
    }

    current
}
