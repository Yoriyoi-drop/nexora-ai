use async_trait::async_trait;
use std::collections::HashSet;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction};

#[derive(Debug, Clone)]
pub struct LanguageFilter {
    pub allowed_languages: HashSet<String>,
    pub min_alpha_ratio: f64,
}

impl Default for LanguageFilter {
    fn default() -> Self {
        Self {
            allowed_languages: ["en", "id", "ms", "vi", "th", "zh", "ja", "ko", "de", "fr", "es"]
                .iter().map(|s| s.to_string()).collect(),
            min_alpha_ratio: 0.5,
        }
    }
}

impl LanguageFilter {
    pub fn new(languages: Vec<String>) -> Self {
        Self {
            allowed_languages: languages.into_iter().collect(),
            ..Default::default()
        }
    }

    fn detect_language(&self, text: &str) -> Option<String> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }

        let alpha_count = text.chars().filter(|c| c.is_alphabetic()).count() as f64;
        let total = text.len().max(1) as f64;
        let alpha_ratio = alpha_count / total;

        if alpha_ratio < self.min_alpha_ratio {
            return None;
        }

        let sample = text.chars().take(200).collect::<String>().to_lowercase();

        if sample.contains("the ") || sample.contains("and ") || sample.contains("that ") {
            return Some("en".to_string());
        }
        if sample.contains("yang ") || sample.contains("dan ") || sample.contains("di ") {
            return Some("id".to_string());
        }
        if sample.contains("der ") || sample.contains("die ") || sample.contains("und ") {
            return Some("de".to_string());
        }
        if sample.contains("le ") || sample.contains("les ") || sample.contains("des ") {
            return Some("fr".to_string());
        }
        if sample.contains("el ") || sample.contains("la ") || sample.contains("los ") {
            return Some("es".to_string());
        }
        if sample.contains(" 的") || sample.contains(" 了") || sample.contains(" 是") {
            return Some("zh".to_string());
        }
        if sample.chars().any(|c| ('\u{3040}'..='\u{309f}').contains(&c)) {
            return Some("ja".to_string());
        }
        if sample.chars().any(|c| ('\u{ac00}'..='\u{d7af}').contains(&c)) {
            return Some("ko".to_string());
        }

        let non_ascii = text.chars().filter(|c| !c.is_ascii()).count() as f64;
        if non_ascii / total < 0.1 {
            return Some("en".to_string());
        }

        None
    }
}

#[async_trait]
impl Filter for LanguageFilter {
    fn name(&self) -> &str {
        "language"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let lang = self.detect_language(&sample.text);
        let passed = match lang {
            Some(ref l) => self.allowed_languages.contains(l.as_str()),
            None => false,
        };
        let reason = if !passed {
            Some(format!("unrecognized_or_disallowed_language: {:?}", lang))
        } else {
            None
        };
        FilterResult {
            passed,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason,
            score_delta: if passed { 0.05 } else { -0.5 },
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Reject
    }
}
