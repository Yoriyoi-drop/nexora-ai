use nexora_model_core::classifier_util;
use nexora_model_core::foundation::FoundationModel;
use crate::router;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::omnis)
}

pub fn inject_model(model_arc: Arc<CausalLM>) {
    foundation().set_model_arc(model_arc);
}

fn init_router() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            if let Some(ref embed) = model.token_embedding {
                router::init_router(embed.clone());
                let _ = INITIALIZED.set(true);
            }
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer
        .map_or_else(
            || nexora_model_core::foundation::byte_encode(text),
            |tk| tk.read().encode(text),
        )
}

fn classify_domains(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    router::detect_domains(text, &ids)
}

pub async fn delegate(prompt: &str) -> String {
    init_router();
    let domains = classify_domains(prompt);

    let primary = domains.first().map(|(d, _)| d.as_str()).unwrap_or("general");
    let system = router::domain_system_prompt(primary);

    let secondary = domains.get(1).filter(|(_, p)| *p > 0.3).map(|(d, _)| d.as_str());

    let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
    let framed = format!(
        "[Omnis reasoning | domain: {primary}]\n\
         {system}\n\n\
         User input: {sanitized_prompt}\n\
         Response:"
    );

    let primary_result = nexora_model_core::foundation::call_model(foundation(), &framed, 512, 0.7).await.unwrap_or_else(|e| {
        tracing::warn!("omnis delegation call failed: {}", e);
        format!("[omnis inference error: {}]", e)
    });

    if let Some(sec_domain) = secondary {
        let sec_system = router::domain_system_prompt(sec_domain);
        let sec_framed = format!(
            "[Omnis reasoning | domain: {sec_domain}]\n\
             {sec_system}\n\n\
             User input: {sanitized_prompt}\n\
             Response:"
        );
        let sec_result = nexora_model_core::foundation::call_model(foundation(), &sec_framed, 512, 0.7).await.unwrap_or_else(|e| {
            tracing::warn!("omnis delegation call failed: {}", e);
            format!("[omnis inference error: {}]", e)
        });
        let synthesis = nexora_model_core::foundation::call_model(
            foundation(),
            &format!(
                "[Omnis synthesis | domains: {primary}, {sec_domain}]\n\
                 Synthesize these two expert perspectives into a unified response.\n\n\
                 === {primary} expert ===\n{primary_result}\n\n\
                 === {sec_domain} expert ===\n{sec_result}\n\n\
                 Unified response:"
            ),
            512,
            0.7,
        )
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("omnis synthesis call failed: {}", e);
            format!("{primary_result}\n\n{sec_result}")
        });
        synthesis
    } else {
        primary_result
    }
}
