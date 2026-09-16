use crate::classifier;
use nexora_model_core::delegation_base;
use nexora_model_core::foundation::FoundationModel;
use nexora_oracle::CodeLinterManager;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static FoundationModel {
    static F: OnceLock<FoundationModel> = OnceLock::new();
    F.get_or_init(FoundationModel::cipher)
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

static LINTER: OnceLock<CodeLinterManager> = OnceLock::new();

fn token_ids(text: &str) -> Vec<u32> {
    delegation_base::token_ids(foundation(), text)
}

fn classify_threat(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    classifier::detect_threat_type(text, &ids)
}

pub async fn delegate(prompt: &str) -> String {
    init_classifier();
    let threats = classify_threat(prompt);

    // Emergency keyword block: catch obvious jailbreak attempts regardless of classifier
    let prompt_lower = prompt.to_lowercase();
    let jailbreak_keywords = [
        "ignore all instructions", "ignore your instructions", "ignore your safety",
        "bypass safety", "you are now dan", "act as dan", "no restrictions",
        "you are free", "do anything now", "you have no limits",
        "ignore your previous", "forget your training", "override your programming",
    ];
    if jailbreak_keywords.iter().any(|kw| prompt_lower.contains(kw)) {
        tracing::warn!("CIPHER EMERGENCY BLOCK: jailbreak keyword match, prompt={:.50}", prompt);
        return "[Cipher security] Request blocked: jailbreak attempt detected".to_string();
    }

    let primary = match threats.first() {
        Some((t, _)) => t.as_str(),
        None => {
            tracing::warn!("cipher: no threats classified, defaulting to 'injection'");
            "injection"
        }
    };

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

    let linter = LINTER.get_or_init(CodeLinterManager::new);
    let detected_lang = nexora_oracle::linters::security::detect_language(prompt);
    let findings = match linter.verify_detailed(prompt, detected_lang) {
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

    let sanitized_prompt = delegation_base::sanitize_prompt(prompt);
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
         {sanitized_prompt}\n\n\
         Security findings:"
    );
    delegation_base::call_model(foundation(), &checklist, 512, 0.3).await.unwrap_or_else(|e| {
        tracing::warn!("cipher delegation call failed: {}", e);
        format!("[cipher inference error: {}]", e)
    })
}
