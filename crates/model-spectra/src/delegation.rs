use nexora_model_core::classifier_util;
use nexora_model_core::foundation::NxrSpectraModel;
use crate::classifier;
use nexora_multimodal::caffeine::{CaffeineConfig, CaffeineProcessor};
use nexora_multimodal::MultiModalInputs;
use nexora_multimodal::MultimodalResult;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrSpectraModel {
    static F: OnceLock<NxrSpectraModel> = OnceLock::new();
    F.get_or_init(NxrSpectraModel::new)
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

fn classify_style(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_creative_style(text, &ids)
}

fn unique_word_ratio(s: &str) -> f64 {
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let unique: HashSet<&&str> = words.iter().collect();
    unique.len() as f64 / words.len() as f64
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let styles = classify_style(prompt);
    let primary = styles.first().map(|(s, _)| s.as_str()).unwrap_or("narrative");
    let (base_temp, _) = classifier::style_params(primary);
    let framing = classifier::style_framing(primary);

    let mut mm = CaffeineProcessor::with_caffeine(CaffeineConfig::default()).unwrap_or_default();
    let multimodal = MultiModalInputs {
        text: Some(nexora_multimodal::TextInput {
            text: prompt.to_string(),
            tokens: None,
            language: "en".into(),
        }),
        image: None,
        audio: None,
        video: None,
        context: None,
    };
    let mm_result = mm.process_multimodal(&multimodal).await.unwrap_or_else(|e| {
        tracing::warn!("spectra multimodal processing failed: {}", e);
        MultimodalResult::default()
    });
    let mm_summary = mm_result.processing_summary;

    let temps = [base_temp - 0.1, base_temp, base_temp + 0.2];
    let mut results: Vec<(String, f64)> = Vec::new();
    let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
    for &t in &temps {
        let creative_prompt = format!(
            "[Spectra creative | style: {primary} | multimodal: {mm_summary} | temperature={t:.1}]\n\
             {framing}\n\n\
             Generate original content for: {sanitized_prompt}"
        );
        if let Ok(text) = nexora_model_core::foundation::call_model(foundation(), &creative_prompt, 256, t).await {
            let score = unique_word_ratio(&text);
            results.push((text, score));
        }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.into_iter().next().map(|(t, _)| t).unwrap_or_default()
}
