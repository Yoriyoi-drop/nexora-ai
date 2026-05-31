use crate::foundation::NxrKronosModel;
use crate::kronos::classifier;
use crate::kronos::classifier::TemporalClassifier;
use nexora_reasoning::SacaEngine;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrKronosModel {
    static F: OnceLock<NxrKronosModel> = OnceLock::new();
    F.get_or_init(NxrKronosModel::new)
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
                TemporalClassifier::init(embed.clone());
                let _ = INITIALIZED.set(true);
            }
        }
    }
}

static SACA: OnceLock<SacaEngine> = OnceLock::new();

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_temporal(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_temporal_mode(text, &ids)
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let modes = classify_temporal(prompt);
    let primary = modes.first().map(|(m, _)| m.as_str()).unwrap_or("evergreen");
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC");
    let framing = classifier::temporal_framing(primary);
    let context = format!("mode: {primary} | current time: {now}\n{framing}");

    let engine = SACA.get_or_init(SacaEngine::new);
    match engine.reason(prompt, &context).await {
        Ok(r) if !r.conclusion.is_empty() => {
            tracing::debug!("kronos SACA reasoning succeeded (mode: {primary})");
            r.conclusion
        }
        _ => {
            tracing::warn!("kronos SACA reasoning unavailable (mode: {primary}), using prompt-based reasoning");
            let framed = format!(
                "[Kronos temporal | mode: {primary} | current time: {now}]\n\
                 {framing}\n\n\
                 Input: {prompt}\n\
                 Temporal analysis and response:"
            );
            crate::foundation::call_model(foundation(), &framed, 512, 0.5).await.unwrap_or_else(|e| {
                tracing::warn!("kronos delegation call failed: {}", e);
                format!("[kronos inference error: {}]", e)
            })
        }
    }
}
