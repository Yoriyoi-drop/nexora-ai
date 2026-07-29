use nexora_foundation::model_core::delegation_base;
use nexora_foundation::model_core::foundation::FoundationModel;
use crate::classifier;
use nexora_cognition::saca::SacaEngine;
use nexora_memory::{MemoryManager, MemoryLayer};
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static MEMORY: OnceLock<MemoryManager> = OnceLock::new();

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::kronos)
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
    delegation_base::token_ids(foundation(), text)
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

    let memory = MEMORY.get_or_init(MemoryManager::new);
    let _ = memory.store(MemoryLayer::Session, &format!("input:{}", prompt), prompt).await;

    let historical = memory.retrieve(MemoryLayer::Long, &format!("kronos:{}", primary)).await.unwrap_or_default();
    let history_context = match historical {
        Some(ref h) if h.len() > 10 => format!("\n[historical context for {primary}]: {h}"),
        _ => String::new(),
    };

    let full_prompt = format!("{context}{history_context}\nInput: {prompt}\nTemporal response:");

    let engine = SACA.get_or_init(SacaEngine::new);
    let result = match engine.reason(prompt, &full_prompt).await {
        Ok(r) if !r.conclusion.is_empty() => {
            tracing::debug!("kronos SACA reasoning succeeded (mode: {primary})");
            r.conclusion
        }
        _ => {
            tracing::warn!("kronos SACA reasoning unavailable (mode: {primary}), using prompt-based reasoning");
            let sanitized_prompt = delegation_base::sanitize_prompt(prompt);
            let framed = format!(
                "[Kronos temporal | mode: {primary} | current time: {now}]\n\
                 {framing}{history_context}\n\n\
                 Input: {sanitized_prompt}\n\
                 Temporal analysis and response:"
            );
            delegation_base::call_model(foundation(), &framed, 512, 0.5).await.unwrap_or_else(|e| {
                tracing::warn!("kronos delegation call failed: {}", e);
                format!("[kronos inference error: {}]", e)
            })
        }
    };

    let _ = memory.store(MemoryLayer::Long, &format!("kronos:{}", primary), &result).await;
    let _ = memory.store(MemoryLayer::Session, &format!("output:{}", primary), &result).await;

    result
}
