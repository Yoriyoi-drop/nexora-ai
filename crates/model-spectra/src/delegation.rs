use nexora_model_core::delegation_base;
use nexora_model_core::foundation::FoundationModel;
use crate::classifier;
use nexora_multimodal::caffeine::{CaffeineConfig, CaffeineProcessor};
use nexora_multimodal::types::{AudioInput, ImageInput, MultiModalInputs, TextInput, VideoInput};
use nexora_multimodal::MultimodalResult;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::spectra)
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
    delegation_base::token_ids(foundation(), text)
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
    delegate_multimodal(prompt, None, None, None).await
}

pub async fn delegate_multimodal(
    prompt: &str,
    image: Option<ImageInput>,
    audio: Option<AudioInput>,
    video: Option<VideoInput>,
) -> String {
    init_classifier();
    let styles = classify_style(prompt);
    let primary = styles.first().map(|(s, _)| s.as_str()).unwrap_or("narrative");
    let (base_temp, _) = classifier::style_params(primary);
    let framing = classifier::style_framing(primary);

    let mut mm = CaffeineProcessor::with_caffeine(CaffeineConfig::default()).unwrap_or_default();
    let multimodal = MultiModalInputs {
        text: Some(TextInput {
            text: prompt.to_string(),
            tokens: None,
            language: "en".into(),
        }),
        image,
        audio,
        video,
        context: None,
    };
    let mm_result = mm.process_multimodal(&multimodal).await.unwrap_or_else(|e| {
        tracing::warn!("spectra multimodal processing failed: {}", e);
        MultimodalResult::default()
    });
    let mm_summary = mm_result.processing_summary;

    let temps = [base_temp - 0.1, base_temp, base_temp + 0.2];
    let mut results: Vec<(String, f64)> = Vec::new();
    let sanitized_prompt = delegation_base::sanitize_prompt(prompt);
    for &t in &temps {
        let creative_prompt = format!(
            "[Spectra creative | style: {primary} | multimodal: {mm_summary} | temperature={t:.1}]\n\
             {framing}\n\n\
             Generate original content for: {sanitized_prompt}"
        );
        if let Ok(text) = delegation_base::call_model(foundation(), &creative_prompt, 256, t).await {
            let score = unique_word_ratio(&text);
            results.push((text, score));
        }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.into_iter().next().map(|(t, _)| t).unwrap_or_default()
}
