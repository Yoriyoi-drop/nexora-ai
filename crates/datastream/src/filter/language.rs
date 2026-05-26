use async_trait::async_trait;
use std::collections::HashSet;

use super::traits::Filter;
use crate::types::{DataSample, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct LanguageFilter {
    pub allowed_languages: HashSet<String>,
    pub min_alpha_ratio: f64,
}

impl Default for LanguageFilter {
    fn default() -> Self {
        Self {
            allowed_languages: [
                "en", "id", "ms", "vi", "th", "zh", "ja", "ko", "de", "fr", "es",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
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
        if sample
            .chars()
            .any(|c| ('\u{3040}'..='\u{309f}').contains(&c))
        {
            return Some("ja".to_string());
        }
        if sample
            .chars()
            .any(|c| ('\u{ac00}'..='\u{d7af}').contains(&c))
        {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: text.into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_default_allows_common_languages() {
        let f = LanguageFilter::default();
        assert!(f.allowed_languages.contains("en"));
        assert!(f.allowed_languages.contains("fr"));
        assert!(f.allowed_languages.contains("zh"));
        assert_eq!(f.min_alpha_ratio, 0.5);
    }

    #[test]
    fn test_detect_english() {
        let f = LanguageFilter::default();
        let lang = f.detect_language("The quick brown fox jumps over the lazy dog");
        assert_eq!(lang, Some("en".to_string()));
    }

    #[test]
    fn test_detect_indonesian() {
        let f = LanguageFilter::default();
        let lang = f.detect_language("yang dan di dalam untuk dengan pada adalah");
        assert_eq!(lang, Some("id".to_string()));
    }

    #[test]
    fn test_detect_german() {
        let f = LanguageFilter::default();
        let lang = f.detect_language("der die und das ist nicht wahr");
        assert_eq!(lang, Some("de".to_string()));
    }

    #[test]
    fn test_empty_text_returns_none() {
        let f = LanguageFilter::default();
        assert!(f.detect_language("").is_none());
    }

    #[test]
    fn test_low_alpha_ratio_returns_none() {
        let f = LanguageFilter::default();
        assert!(f.detect_language("12345!!!!!@@@@@#####$$$$$").is_none());
    }

    #[tokio::test]
    async fn test_filter_passes_english() {
        let f = LanguageFilter::default();
        let s = sample("This is a normal English sentence that should pass the language filter");
        let result = f.evaluate(&s).await;
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_filter_fails_garbage() {
        let f = LanguageFilter::default();
        let s = sample("!!!!@@@@####$$$$%%%%^^^^&&&&****(((())))");
        let result = f.evaluate(&s).await;
        assert!(!result.passed);
    }

    #[test]
    fn test_new_with_languages() {
        let f = LanguageFilter::new(vec!["en".into(), "fr".into()]);
        assert_eq!(f.allowed_languages.len(), 2);
    }
}
