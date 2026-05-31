use crate::foundation::NxrGenesisModel;
use crate::genesis::classifier;
use crate::genesis::classifier::QualityClassifier;
use nexora_reasoning::SacaEngine;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

const MAX_REFINEMENT_ITERATIONS: usize = 3;
const QUALITY_THRESHOLD: f32 = 0.6;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrGenesisModel {
    static F: OnceLock<NxrGenesisModel> = OnceLock::new();
    F.get_or_init(NxrGenesisModel::new)
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
                QualityClassifier::init(embed.clone());
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

fn classify_quality(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_quality_focus(text, &ids)
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
            crate::foundation::call_model(foundation(), prompt, 256, 0.8).await.unwrap_or_else(|e| {
                format!("[genesis inference error: {}]", e)
            })
        }
    };

    if current.is_empty() {
        return current;
    }

    for iteration in 0..MAX_REFINEMENT_ITERATIONS {
        let q = classify_quality(&current);
        if let Some((weakest, score)) = q.first() {
            if *score >= QUALITY_THRESHOLD {
                tracing::debug!("genesis quality threshold met (iter {iteration}, {weakest}: {score:.2})");
                break;
            }
            let refocus = classifier::refinement_focus(weakest);
            tracing::info!("genesis refinement iter {iteration}: focusing on {weakest} ({score:.2})");

            match engine.reason(&current, refocus).await {
                Ok(r) if !r.conclusion.is_empty() => {
                    current = r.conclusion;
                }
                _ => {
                    let improved = crate::foundation::call_model(
                        foundation(),
                        &format!(
                            "[Genesis refinement | focus: {weakest}]\n\
                             {refocus}\n\n\
                             Original:\n{current}\n\n\
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
    }

    current
}
