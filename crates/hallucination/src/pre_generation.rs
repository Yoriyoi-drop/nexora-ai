use crate::types::PreGenCheckResult;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct PreGenConfig {
    pub scope_threshold: f32,
    pub ambiguity_threshold: f32,
    pub context_threshold: f32,
    pub knowledge_cutoff: String,
    pub domain_list: Vec<String>,
}

impl Default for PreGenConfig {
    fn default() -> Self {
        Self {
            scope_threshold: 0.3,
            ambiguity_threshold: 0.5,
            context_threshold: 0.4,
            knowledge_cutoff: "Last updated per training data cutoff".to_string(),
            domain_list: vec![
                "coding",
                "mathematics",
                "science",
                "technology",
                "general",
                "reasoning",
                "analysis",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }
}

pub struct PreGenerationChecker {
    config: PreGenConfig,
    ambiguity_patterns: Vec<Regex>,
    specific_claim_patterns: Vec<Regex>,
    recency_patterns: Vec<Regex>,
}

impl PreGenerationChecker {
    pub fn new(config: PreGenConfig) -> Self {
        Self {
            config,
            ambiguity_patterns: vec![
                Regex::new(r"(?i)\b(mungkin|maybe|perhaps|possibly|could be|sepertinya)\b")
                    .expect("valid ambiguity keyword regex"),
                Regex::new(r"(?i)\b(tidak jelas|unclear|ambiguous|tidak pasti)\b")
                    .expect("valid uncertain keyword regex"),
                Regex::new(r"\?").expect("valid question mark regex"),
            ],
            specific_claim_patterns: vec![
                Regex::new(r"\b\d{4}\b").expect("valid year pattern regex"),
                Regex::new(r"\b\d+\.\d+%?\b").expect("valid decimal pattern regex"),
                Regex::new(r#""[^"]{10,}""#).expect("valid quote pattern regex"),
                Regex::new(r"(?i)\b(menurut|according to|research|study|penelitian)\b")
                    .expect("valid citation keyword regex"),
            ],
            recency_patterns: vec![
                Regex::new(r"(?i)\b(tahun ini|this year|recent|baru-baru|terbaru|202[4-9])\b")
                    .expect("valid recency keyword regex"),
                Regex::new(r"(?i)\b(saat ini|currently|now|sekarang)\b")
                    .expect("valid current time keyword regex"),
            ],
        }
    }

    pub fn check(
        &self,
        input: &str,
        context: Option<&str>,
    ) -> Result<PreGenCheckResult, crate::HallucinationError> {
        let in_scope = self.check_scope(input);
        let ambiguity_score = self.compute_ambiguity(input);
        let context_sufficiency = self.compute_context_sufficiency(input, context);
        let suggestions = self.generate_suggestions(ambiguity_score, context_sufficiency);

        let can_proceed = in_scope && ambiguity_score < self.config.ambiguity_threshold;

        Ok(PreGenCheckResult {
            can_proceed,
            in_scope,
            ambiguity_score,
            context_sufficiency,
            reason: if !can_proceed {
                if !in_scope {
                    "Out of scope".into()
                } else if ambiguity_score >= self.config.ambiguity_threshold {
                    format!("High ambiguity ({:.2})", ambiguity_score)
                } else {
                    format!("Insufficient context ({:.2})", context_sufficiency)
                }
            } else {
                "OK".into()
            },
            suggestions,
        })
    }

    fn check_scope(&self, input: &str) -> bool {
        let lower = input.to_lowercase();
        if self.recency_patterns.iter().any(|r| r.is_match(&lower)) {
            return false;
        }

        let out_of_scope = vec![
            "internal server",
            "proprietary",
            "confidential",
            "rahasia",
            "internal data",
            "future prediction",
            "stock price",
            "election result",
        ];
        if out_of_scope.iter().any(|k| lower.contains(k)) {
            return false;
        }

        true
    }

    fn compute_ambiguity(&self, input: &str) -> f32 {
        let mut score = 0.0f32;
        for p in &self.ambiguity_patterns {
            if p.is_match(input) {
                score += 0.2;
            }
        }
        if input.len() < 10 {
            score += 0.3;
        }
        score.min(1.0)
    }

    fn compute_context_sufficiency(&self, _input: &str, context: Option<&str>) -> f32 {
        match context {
            Some(ctx) if !ctx.is_empty() => {
                let word_count = ctx.split_whitespace().count();
                if word_count > 50 {
                    1.0
                } else if word_count > 20 {
                    0.7
                } else {
                    0.4
                }
            }
            Some(_) => 0.3,
            None => 0.1,
        }
    }

    fn generate_suggestions(&self, ambiguity: f32, context: f32) -> Vec<String> {
        let mut s = Vec::new();
        if ambiguity > 0.3 {
            s.push("Please clarify your question — it contains ambiguous terms.".to_string());
        }
        if context < 0.5 {
            s.push("Provide more context for a more accurate answer.".to_string());
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> PreGenerationChecker {
        PreGenerationChecker::new(PreGenConfig::default())
    }

    #[test]
    fn test_check_normal_input() {
        let c = checker();
        let result = c.check("What is the capital of France?", None).unwrap();
        assert!(result.can_proceed);
        assert!(result.in_scope);
        assert!(result.ambiguity_score < 0.5);
    }

    #[test]
    fn test_check_empty_input() {
        let c = checker();
        let result = c.check("", None).unwrap();
        assert!(result.ambiguity_score > 0.0);
        assert!(result.suggestions.len() > 0);
    }

    #[test]
    fn test_check_out_of_scope_proprietary() {
        let c = checker();
        let result = c.check("Tell me the confidential internal data", None).unwrap();
        assert!(!result.can_proceed);
        assert!(!result.in_scope);
        assert_eq!(result.reason, "Out of scope");
    }

    #[test]
    fn test_check_out_of_scope_rahasia() {
        let c = checker();
        let result = c.check("Apa rahasia perusahaan?", None).unwrap();
        assert!(!result.can_proceed);
        assert!(!result.in_scope);
    }

    #[test]
    fn test_check_out_of_scope_stock_price() {
        let c = checker();
        let result = c.check("What will be the stock price tomorrow?", None).unwrap();
        assert!(!result.can_proceed);
        assert!(!result.in_scope);
    }

    #[test]
    fn test_check_high_ambiguity() {
        let c = checker();
        let result = c.check("mungkin? tidak jelas", None).unwrap();
        assert!(result.ambiguity_score >= 0.5);
    }

    #[test]
    fn test_check_with_good_context() {
        let c = checker();
        let ctx_str = "word ".repeat(60);
        let result = c.check("What is Rust?", Some(&ctx_str)).unwrap();
        assert!(result.can_proceed);
        assert_eq!(result.context_sufficiency, 1.0);
    }

    #[test]
    fn test_check_with_partial_context() {
        let c = checker();
        let context = Some("short context");
        let result = c.check("What is Rust?", context).unwrap();
        assert_eq!(result.context_sufficiency, 0.4);
    }

    #[test]
    fn test_check_without_context() {
        let c = checker();
        let result = c.check("What is Rust?", None).unwrap();
        assert_eq!(result.context_sufficiency, 0.1);
    }

    #[test]
    fn test_check_recency_keyword_blocks() {
        let c = checker();
        let result = c.check("What is the latest news this year?", None).unwrap();
        assert!(!result.in_scope);
        assert!(!result.can_proceed);
    }

    #[test]
    fn test_check_recency_keyword_2025() {
        let c = checker();
        let result = c.check("What happened in 2025?", None).unwrap();
        assert!(!result.in_scope);
    }

    #[test]
    fn test_ambiguity_scoring_multiple_patterns() {
        let c = checker();
        let result = c.check("mungkin? tidak jelas", None).unwrap();
        assert!(result.ambiguity_score >= 0.6);
    }

    #[test]
    fn test_suggestions_on_ambiguous() {
        let c = checker();
        let result = c.check("mungkin", None).unwrap();
        assert!(result.suggestions.iter().any(|s| s.contains("ambiguous")));
    }

    #[test]
    fn test_suggestions_on_low_context() {
        let c = checker();
        let result = c.check("hello", Some("short")).unwrap();
        assert!(result.suggestions.iter().any(|s| s.contains("context")));
    }

    #[test]
    fn test_generate_suggestions_both_triggers() {
        let c = checker();
        let s = c.generate_suggestions(0.5, 0.3);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_generate_suggestions_none() {
        let c = checker();
        let s = c.generate_suggestions(0.0, 1.0);
        assert!(s.is_empty());
    }
}
