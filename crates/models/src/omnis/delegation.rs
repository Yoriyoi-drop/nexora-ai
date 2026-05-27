use crate::foundation::NxrOmnisModel;
use crate::omnis::router;
use crate::omnis::router::OmnisMoERouter;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrOmnisModel {
    static F: OnceLock<NxrOmnisModel> = OnceLock::new();
    F.get_or_init(NxrOmnisModel::new)
}

fn init_router() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let embed = model.token_embedding.clone();
            OmnisMoERouter::init(embed);
            let _ = INITIALIZED.set(true);
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer
        .map_or_else(
            || crate::foundation::byte_encode(text),
            |tk| tk.read().encode(text),
        )
}

fn classify_domains(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    router::detect_domains(text, &ids)
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
    init_router();
    let domains = classify_domains(prompt);

    let primary = domains.first().map(|(d, _)| d.as_str()).unwrap_or("general");
    let system = router::domain_system_prompt(primary);

    let secondary = domains.get(1).filter(|(_, p)| *p > 0.3).map(|(d, _)| d.as_str());

    let framed = format!(
        "[Omnis reasoning | domain: {primary}]\n\
         {system}\n\n\
         User input: {prompt}\n\
         Response:"
    );

    let primary_result = call(&framed, 512, 0.7).await.unwrap_or_else(|e| {
        tracing::warn!("omnis delegation call failed: {}", e);
        format!("[omnis inference error: {}]", e)
    });

    if let Some(sec_domain) = secondary {
        let sec_system = router::domain_system_prompt(sec_domain);
        let sec_framed = format!(
            "[Omnis reasoning | domain: {sec_domain}]\n\
             {sec_system}\n\n\
             User input: {prompt}\n\
             Response:"
        );
        let sec_result = call(&sec_framed, 512, 0.7).await.unwrap_or_else(|e| {
            tracing::warn!("omnis delegation call failed: {}", e);
            format!("[omnis inference error: {}]", e)
        });
        let synthesis = call(
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
