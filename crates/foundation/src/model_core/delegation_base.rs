use crate::model_core::foundation::FoundationModel;
use ndarray::Array2;
use std::sync::OnceLock;

pub fn token_ids(f: &FoundationModel, text: &str) -> Vec<u32> {
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(
        || crate::model_core::foundation::byte_encode(text),
        |tk| tk.read().encode(text),
    )
}

pub fn init_embedding_classifier(
    initialized: &OnceLock<bool>,
    f: &FoundationModel,
    classifier_init: impl FnOnce(ndarray::Array2<f32>),
) {
    if initialized.get().is_some() {
        return;
    }
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            if let Some(ref embed) = model.token_embedding {
                classifier_init(embed.clone());
                let _ = initialized.set(true);
            }
        }
    }
}

pub fn embed_average(embed_table: &Array2<f32>, token_ids: &[u32]) -> Vec<f32> {
    crate::model_core::classifier_util::embed_average(embed_table, token_ids).to_vec()
}

pub fn sanitize_prompt(prompt: &str) -> String {
    crate::model_core::classifier_util::sanitize_prompt(prompt)
}

pub async fn call_model(
    f: &FoundationModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, String> {
    crate::model_core::foundation::call_model(f, prompt, max_tokens, temperature).await
}
