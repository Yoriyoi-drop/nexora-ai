use crate::foundation::NxrVortexModel;
use crate::vortex::analyzer;
use crate::vortex::analyzer::CodeReviewClassifier;
use nexora_oracle::CodeVerifierManager;
use nexora_shared::base_model::{InputData, NxrInput, OutputData};
use std::collections::HashMap;
use std::sync::OnceLock;

fn foundation() -> &'static NxrVortexModel {
    static F: OnceLock<NxrVortexModel> = OnceLock::new();
    F.get_or_init(NxrVortexModel::new)
}

fn init_analyzer() {
    let f = foundation();
    if let Ok(guard) = f.model.try_lock() {
        if let Some(ref model) = *guard {
            let embed = model.token_embedding.clone();
            CodeReviewClassifier::init(embed);
        }
    }
}

fn token_ids(text: &str) -> Vec<u32> {
    let f = foundation();
    let tokenizer = f.tokenizer.as_ref();
    tokenizer
        .map_or_else(
            || crate::foundation::byte_encode(text),
            |tk| tk.read().encode(text),
        )
}

fn classify_review(text: &str) -> Vec<(String, f32)> {
    let ids = token_ids(text);
    analyzer::analyze_review_type(text, &ids)
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

fn run_verifiers(code: &str, language: &str) -> String {
    let mgr = CodeVerifierManager::new();
    match mgr.verify_detailed(code, language) {
        Ok(results) => {
            let mut report = String::new();
            for r in &results {
                let status = if r.passed { "PASS" } else { "FAIL" };
                report.push_str(&format!(
                    "[{}] {} — score: {:.2}\n",
                    status, r.verifier_name, r.score
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
            tracing::warn!("vortex code verifier failed: {}", e);
            format!("[Verification error: {}]", e)
        }
    }
}

pub async fn delegate(prompt: &str) -> String {
    init_analyzer();
    let categories = classify_review(prompt);
    let primary = categories.first().map(|(c, _)| c.as_str()).unwrap_or("general");
    let lang = analyzer::detect_language(prompt);

    let verification = run_verifiers(prompt, lang);

    let framed = format!(
        "[Vortex code review | focus: {primary} | language: {lang}]\n\
         Automated code analysis results:\n\
         {verification}\n\n\
         Based on the above analysis, provide a detailed code review:\n\
         ```\n{prompt}\n```\n\n\
         Code review:"
    );
    call(&framed, 512, 0.4).await.unwrap_or_else(|e| {
        tracing::warn!("vortex delegation call failed: {}", e);
        format!("[vortex inference error: {}]", e)
    })
}
