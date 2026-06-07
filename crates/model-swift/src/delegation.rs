use nexora_model_core::classifier_util;
use nexora_model_core::foundation::NxrSwiftModel;
use crate::classifier;
use nexora_has_moe_ffn::Router;
use nexora_atqs::compression::AtqsCompression;
use nexora_erp::{ERPEngine, ERPConfig, CompressionMode};
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static ATQS: OnceLock<AtqsCompression> = OnceLock::new();
static ERP: OnceLock<std::sync::Mutex<ERPEngine>> = OnceLock::new();

fn foundation() -> &'static NxrSwiftModel {
    static F: OnceLock<NxrSwiftModel> = OnceLock::new();
    F.get_or_init(NxrSwiftModel::new)
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

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| nexora_model_core::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_task(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_task_type(text, &ids)
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let tasks = classify_task(prompt);
    let primary = tasks.first().map(|(t, _)| t.as_str()).unwrap_or("qa");
    let (max_tokens, temperature) = classifier::task_params(primary);

    let ids = token_ids(prompt);
    let expert_route = {
        let f = foundation();
        match f.model.lock() {
            Ok(guard) => {
                guard.as_ref().and_then(|m| m.token_embedding.as_ref()).and_then(|embed_table| {
                    let avg = classifier_util::embed_average(embed_table, &ids);
                    let embed_dim = avg.len();
                    if embed_dim == 0 { return None; }
                    let moe = Router::new(embed_dim, 5, 1);
                    let input_array = avg.into_shape((1, embed_dim)).ok()?;
                    let moe_weights = moe.forward(&input_array);
                    let top_expert = (0..moe_weights.shape()[1])
                        .max_by(|&a, &b| moe_weights[[0, a]].partial_cmp(&moe_weights[[0, b]]).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(0);
                    Some(match top_expert {
                        0 => "qa",
                        1 => "summarize",
                        2 => "translate",
                        3 => "generate",
                        _ => "analyze",
                    })
                })
            }
            Err(e) => {
                tracing::warn!("Swift model lock poisoned: {}", e);
                None
            }
        }
    };

    let edge_insight = {
        let atqs = ATQS.get_or_init(AtqsCompression::new);
        match atqs.compress(prompt.as_bytes()).await {
            Ok(cr) if cr.compression_ratio < 0.9 => {
                format!("edge_compression: {:.2}x", 1.0 / cr.compression_ratio)
            }
            _ => String::new(),
        }
    };

    let erp_insight = {
        let cfg = ERPConfig {
            compression_mode: CompressionMode::Aggressive,
            ..ERPConfig::default()
        };
        let erp = ERP.get_or_init(|| std::sync::Mutex::new(ERPEngine::new(cfg)));
        match erp.lock() {
            Ok(_engine) => "erp: aggressive".to_string(),
            Err(_) => String::new(),
        }
    };

    let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
    let optimization_tag = if !edge_insight.is_empty() || !erp_insight.is_empty() {
        format!("[{} | {}]", edge_insight, erp_insight)
    } else {
        String::new()
    };

    let framed = if let Some(route) = expert_route {
        format!(
            "[Swift task | type: {primary} | moe_route: {route}]{optimization_tag}\n\
             Process this input efficiently:\n\
             {sanitized_prompt}"
        )
    } else {
        format!(
            "[Swift task | type: {primary}]{optimization_tag}\n\
             Process this input efficiently:\n\
             {sanitized_prompt}"
        )
    };
    nexora_model_core::foundation::call_model(foundation(), &framed, max_tokens, temperature).await.unwrap_or_else(|e| {
        tracing::warn!("swift delegation call failed: {}", e);
        format!("[swift inference error: {}]", e)
    })
}
