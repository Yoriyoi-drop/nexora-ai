use crate::cipher::classifier;
use crate::cipher::classifier::ThreatClassifier;
use crate::foundation::NxrCipherModel;
use nexora_oracle::CodeLinterManager;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrCipherModel {
    static F: OnceLock<NxrCipherModel> = OnceLock::new();
    F.get_or_init(NxrCipherModel::new)
}

fn init_classifier() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let embed = model.token_embedding.clone();
            ThreatClassifier::init(embed);
            if INITIALIZED.set(true).is_err() {
                tracing::warn!("cipher: INITIALIZED already set (race condition)");
            }
        }
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
    let primary = match threats.first() {
        Some((t, _)) => t.as_str(),
        None => {
            tracing::warn!("cipher: no threats classified, defaulting to 'injection'");
            "injection"
        }
    };

    // ENFORCEMENT: Block high-confidence dangerous threats
    if let Some((threat_type, confidence)) = threats.first() {
        let dangerous = ["injection", "xss", "auth"];
        if *confidence > 0.8 && dangerous.contains(&threat_type.as_str()) {
            tracing::warn!(
                "CIPHER SECURITY BLOCK: threat={}, confidence={:.2}, prompt={:.50}",
                threat_type,
                confidence,
                prompt
            );
            return format!(
                "[Cipher security] Request blocked: detected {} threat (confidence: {:.1}%)",
                threat_type,
                confidence * 100.0
            );
        }
    }

    let focus = classifier::threat_focus(primary);

    // Phase 4: Run through Oracle security linter
    let linter = CodeLinterManager::new();
    let findings = match linter.verify_detailed(prompt, "text") {
        Ok(results) => {
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
        }
        Err(e) => {
            tracing::warn!("cipher security linter failed: {}", e);
            "Lint unavailable".to_string()
        }
    };

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
    call(&checklist, 512, 0.3).await.unwrap_or_else(|e| {
        tracing::warn!("cipher delegation call failed: {}", e);
        format!("[cipher inference error: {}]", e)
    })
}
