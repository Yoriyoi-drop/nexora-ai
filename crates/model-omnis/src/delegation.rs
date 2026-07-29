use nexora_foundation::model_core::delegation_base;
use nexora_foundation::model_core::foundation::FoundationModel;
use crate::router;
use ndarray::{Array2, s};
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_oracle::{OracleBackboneConfig, OraclePool};
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static ORACLE_POOL: OnceLock<Arc<OraclePool>> = OnceLock::new();

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

fn init_oracle_pool() {
    ORACLE_POOL.get_or_init(|| {
        let config = OracleBackboneConfig {
            d_model: 1024,
            n_heads: 8,
            n_experts: 8,
            top_k: 2,
            latent_dim: 128,
            context_size: 8192,
            mlp_hidden: 4096,
            dropout: 0.0,
        };
        Arc::new(OraclePool::new_with_config(1, 50000, config))
    });
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    delegation_base::token_ids(f, text)
}

fn classify_domains(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    router::detect_domains(text, &ids)
}

fn oracle_reasoning_insight(prompt: &str, domain: &str) -> String {
    let pool = match ORACLE_POOL.get() {
        Some(p) => p,
        None => return String::new(),
    };
    let handle = match OraclePool::acquire_arc(pool) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("omnis oracle acquire failed: {e}");
            return String::new();
        }
    };
    let id = handle.id();

    let result = pool.with_backbone(id, |backbone| {
        let ids: Vec<i32> = token_ids(prompt).iter().map(|&t| t as i32).collect();
        if ids.is_empty() {
            return String::new();
        }
        let seq_len = ids.len();
        let capped = if seq_len > 2048 { &ids[..2048] } else { &ids };
        let input = match Array2::from_shape_vec((1, capped.len()), capped.to_vec()) {
            Ok(v) => v,
            Err(_) => return String::new(),
        };
        let logits = match backbone.forward(&input, None) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("omnis oracle forward failed: {e}");
                return String::new();
            }
        };
        let last = logits.slice(s![0, -1, ..]).to_vec();
        let max_l = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_l == f32::NEG_INFINITY {
            return String::new();
        }
        let exp_sum: f32 = last.iter().map(|&x| (x - max_l).exp()).sum();
        if exp_sum <= 0.0 {
            return String::new();
        }
        let probs: Vec<f32> = last.iter().map(|&x| (x - max_l).exp() / exp_sum).collect();
        let entropy: f32 = -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.log(std::f32::consts::E)).sum::<f32>() / (probs.len() as f32).ln().max(1.0);
        let clarity = (1.0 - entropy) * 100.0;
        format!(
            "[OracleDeepReasoning | domain: {domain} | confidence: {clarity:.0}% | backbone: MoE+MLA]"
        )
    });
    result.unwrap_or_default()
}

pub async fn delegate(prompt: &str) -> String {
    init_router();
    init_oracle_pool();
    let domains = classify_domains(prompt);

    let primary = domains.first().map(|(d, _)| d.as_str()).unwrap_or("general");
    let system = router::domain_system_prompt(primary);

    let secondary = domains.get(1).filter(|(_, p)| *p > 0.3).map(|(d, _)| d.as_str());

    let sanitized_prompt = delegation_base::sanitize_prompt(prompt);
    let oracle_insight = oracle_reasoning_insight(prompt, primary);
    let framed = if oracle_insight.is_empty() {
        format!(
            "[Omnis reasoning | domain: {primary}]\n\
             {system}\n\n\
             User input: {sanitized_prompt}\n\
             Response:"
        )
    } else {
        format!(
            "[Omnis reasoning | domain: {primary}]\n\
             {system}\n\
             {oracle_insight}\n\n\
             User input: {sanitized_prompt}\n\
             Response:"
        )
    };

    let primary_result = delegation_base::call_model(foundation(), &framed, 512, 0.7).await.unwrap_or_else(|e| {
        tracing::warn!("omnis delegation call failed: {}", e);
        format!("[omnis inference error: {}]", e)
    });

    if let Some(sec_domain) = secondary {
        let sec_system = router::domain_system_prompt(sec_domain);
        let sec_oracle = oracle_reasoning_insight(prompt, sec_domain);
        let sec_framed = if sec_oracle.is_empty() {
            format!(
                "[Omnis reasoning | domain: {sec_domain}]\n\
                 {sec_system}\n\n\
                 User input: {sanitized_prompt}\n\
                 Response:"
            )
        } else {
            format!(
                "[Omnis reasoning | domain: {sec_domain}]\n\
                 {sec_system}\n\
                 {sec_oracle}\n\n\
                 User input: {sanitized_prompt}\n\
                 Response:"
            )
        };
        let sec_result = delegation_base::call_model(foundation(), &sec_framed, 512, 0.7).await.unwrap_or_else(|e| {
            tracing::warn!("omnis delegation call failed: {}", e);
            format!("[omnis inference error: {}]", e)
        });
        let synthesis = delegation_base::call_model(
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
