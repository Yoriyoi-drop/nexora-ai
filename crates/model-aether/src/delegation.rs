use crate::classifier;
use nexora_model_core::classifier_util;
use nexora_model_core::foundation::NxrAetherModel;
use nexora_multimodal::caffeine::{CaffeineConfig, CaffeineProcessor};
use nexora_multimodal::types::{AudioInput, ImageInput, MultiModalInputs, TextInput, VideoInput};
use nexora_multimodal::MultimodalResult;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrAetherModel {
    static F: OnceLock<NxrAetherModel> = OnceLock::new();
    F.get_or_init(NxrAetherModel::new)
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

fn classify(text: &str) -> Vec<(String, f32)> {
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let tokenizer = f.tokenizer.as_ref();
            let ids = tokenizer.map_or_else(
                || nexora_model_core::foundation::byte_encode(text),
                |tk| tk.read().encode(text),
            );
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

    let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
    let framed = format!(
        "[Empathetic response | detected emotion: {dominant} | multimodal: {mm_summary}]\n\
         Consider the emotional context and psychological aspects.\n\
         User input: {sanitized_prompt}\n\
         Response (empathetic, emotionally aware):"
    );
    nexora_model_core::foundation::call_model(foundation(), &framed, 512, 0.8).await.unwrap_or_else(|e| {
        tracing::warn!("aether delegation call failed: {}", e);
        format!("[aether inference error: {}]", e)
    })
}