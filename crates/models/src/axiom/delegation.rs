use crate::axiom::classifier;
use crate::axiom::classifier::ReasoningClassifier;
use crate::foundation::NxrAxiomModel;
use nexora_reasoning::SacaEngine;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrAxiomModel {
    static F: OnceLock<NxrAxiomModel> = OnceLock::new();
    F.get_or_init(NxrAxiomModel::new)
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
                ReasoningClassifier::init(embed.clone());
                let _ = INITIALIZED.set(true);
            }
        }
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

static SACA: OnceLock<SacaEngine> = OnceLock::new();

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let types = classify_reasoning(prompt);
    let primary = types.first().map(|(r, _)| r.as_str()).unwrap_or("analytical");
    let focus = classifier::reasoning_prompt(primary);

    let engine = SACA.get_or_init(SacaEngine::new);
    match engine.reason(prompt, focus).await {
        Ok(r) if !r.conclusion.is_empty() => {
            tracing::debug!("axiom SACA reasoning succeeded (type: {primary})");
            r.conclusion
        }
        _ => {
            tracing::warn!("axiom SACA reasoning unavailable (type: {primary}), using prompt-based reasoning");
            let framed = format!(
                "[Axiom reasoning | type: {primary}]\n\
                 {focus}\n\n\
                 Input: {prompt}\n\
                 Conclusion:"
            );
            crate::foundation::call_model(foundation(), &framed, 512, 0.3).await.unwrap_or_else(|e| {
                tracing::warn!("axiom delegation call failed: {}", e);
                format!("[axiom inference error: {}]", e)
            })
        }
    }
}
