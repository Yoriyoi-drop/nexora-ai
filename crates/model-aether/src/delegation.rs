use crate::classifier;
use nexora_model_core::delegation_base;
use nexora_model_core::foundation::FoundationModel;
use nexora_multimodal::caffeine::{CaffeineConfig, CaffeineProcessor};
use nexora_multimodal::types::{AudioInput, ImageInput, MultiModalInputs, TextInput, VideoInput};
use nexora_multimodal::MultimodalResult;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::aether)
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

fn classify(text: &str) -> Vec<(String, f32)> {
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let ids = token_ids(text);
            return classifier::detect_emotions(text, &ids);
        }
    }
    vec![("neutral".to_string(), 1.0)]
}

fn dominant_emotion(emotions: &[(String, f32)]) -> String {
    emotions.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(e, _)| e.clone()).unwrap_or_default()
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
    let emotions = classify(prompt);
    let dominant = dominant_emotion(&emotions);

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
        tracing::warn!("aether multimodal processing failed: {}", e);
        MultimodalResult::default()
    });
    let mm_summary = mm_result.processing_summary;

    let sanitized_prompt = delegation_base::sanitize_prompt(prompt);
    let framed = format!(
        "[Empathetic response | detected emotion: {dominant} | multimodal: {mm_summary}]\n\
         Consider the emotional context and psychological aspects.\n\
         User input: {sanitized_prompt}\n\
         Response (empathetic, emotionally aware):"
    );
    delegation_base::call_model(foundation(), &framed, 512, 0.8).await.unwrap_or_else(|e| {
        tracing::warn!("aether delegation call failed: {}", e);
        format!("[aether inference error: {}]", e)
    })
}