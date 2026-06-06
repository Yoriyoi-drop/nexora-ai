use nexora_model_core::classifier_util;
use nexora_model_core::foundation::NxrVortexModel;
use crate::analyzer;
use nexora_oracle::CodeLinterManager;
use std::sync::Arc;
use std::sync::OnceLock;
use nexora_transformer::CausalLM;

static INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn foundation() -> &'static NxrVortexModel {
    static F: OnceLock<NxrVortexModel> = OnceLock::new();
    F.get_or_init(NxrVortexModel::new)
}

pub fn inject_model(model_arc: Arc<CausalLM>) {
    foundation().set_model_arc(model_arc);
}

fn init_analyzer() {
    if INITIALIZED.get().is_some() {
        return;
    }
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            if let Some(ref embed) = model.token_embedding {
                analyzer::init_analyzer(embed.clone());
                let _ = INITIALIZED.set(true);
            }
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer
        .map_or_else(
            || nexora_model_core::foundation::byte_encode(text),
            |tk| tk.read().encode(text),
        )
}

fn classify_review(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    analyzer::analyze_review_type(text, &ids)
}

static LINTER: OnceLock<CodeLinterManager> = OnceLock::new();

fn run_linters(code: &str, language: &str) -> String {
    let mgr = LINTER.get_or_init(CodeLinterManager::new);
    match mgr.verify_detailed(code, language) {
        Ok(results) => {
            let mut report = String::new();
            for r in &results {
                let status = if r.passed { "PASS" } else { "FAIL" };
                report.push_str(&format!(
                    "[{}] {} — score: {:.2}\n",
                    status, r.linter_name, r.score
                ));
                for issue in &r.issues {
                    let line = issue
                        .line_number
                        .map(|l| format!(":{}", l))
                        .unwrap_or_default();
                    report.push_str(&format!(
                        "  - [{:?}] {} at {}{}\n",
                        issue.severity, issue.message, issue.rule_id, line
                    ));
                }
                for s in &r.suggestions {
                    report.push_str(&format!("  ~ Suggestion: {}\n", s));
                }
            }
            report
        }
        Err(e) => {
            tracing::warn!("vortex code linter failed: {}", e);
            format!("[Lint error: {}]", e)
        }
    }
}

pub async fn delegate(prompt: &str) -> String {
    init_analyzer();
    let categories = classify_review(prompt);
    let primary = categories.first().map(|(c, _)| c.as_str()).unwrap_or("general");
    let lang = analyzer::detect_language(prompt);

    let verification = run_linters(prompt, lang);

    let sanitized_prompt = classifier_util::sanitize_prompt(prompt);
    let framed = format!(
        "[Vortex code review | focus: {primary} | language: {lang}]\n\
         Automated code analysis results:\n\
         {verification}\n\n\
         Based on the above analysis, provide a detailed code review:\n\
         ```\n{sanitized_prompt}\n```\n\n\
         Code review:"
    );
    nexora_model_core::foundation::call_model(foundation(), &framed, 512, 0.4).await.unwrap_or_else(|e| {
        tracing::warn!("vortex delegation call failed: {}", e);
        format!("[vortex inference error: {}]", e)
    })
}
