use crate::cipher::classifier;
use crate::cipher::classifier::ThreatClassifier;
use crate::foundation::NxrCipherModel;
use nexora_oracle::CodeVerifierManager;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

fn foundation() -> &'static NxrCipherModel {
    static F: OnceLock<NxrCipherModel> = OnceLock::new();
    F.get_or_init(NxrCipherModel::new)
}

fn init_classifier() {
    let f = foundation();
    let guard = f.model.blocking_lock();
    if let Some(ref model) = *guard {
        let embed = model.token_embedding.clone();
        ThreatClassifier::init(embed);
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer.map_or_else(|| crate::foundation::byte_encode(text), |tk| tk.read().encode(text))
}

fn classify_threat(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_threat_type(text, &ids)
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
    init_classifier();
    let threats = classify_threat(prompt);
    let primary = threats.first().map(|(t, _)| t.as_str()).unwrap_or("injection");
    let focus = classifier::threat_focus(primary);

    // Phase 4: Run through Oracle security verifier
    let verifier = CodeVerifierManager::new();
    let findings = verifier
        .verify_detailed(prompt, "text")
        .ok()
        .map(|results| {
            let issues: Vec<String> = results
                .iter()
                .flat_map(|r| r.issues.iter())
                .map(|i| format!("[{}] {} (severity: {:?})", i.category, i.message, i.severity))
                .collect();
            if issues.is_empty() {
                "No security issues detected".to_string()
            } else {
                issues.join("; ")
            }
        })
        .unwrap_or_else(|| "Verification unavailable".to_string());

    let checklist = format!(
        "[Cipher security | primary threat: {primary}]\n\
         Oracle security scan: {findings}\n\
         {focus}\n\n\
         Security analysis checklist:\n\
         [1] Injection threats\n\
         [2] Data exposure\n\
         [3] Authentication bypass\n\
         [4] Input validation\n\
         [5] Access control\n\n\
         Analyze this input against the checklist:\n\
         {prompt}\n\n\
         Security findings:"
    );
    call(&checklist, 512, 0.3).await.unwrap_or_default()
}
