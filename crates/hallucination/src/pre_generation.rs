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
                "coding", "mathematics", "science", "technology",
                "general", "reasoning", "analysis",
            ].into_iter().map(String::from).collect(),
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
                Regex::new(r"(?i)\b(mungkin|maybe|perhaps|possibly|could be|sepertinya)\b").unwrap(),
                Regex::new(r"(?i)\b(tidak jelas|unclear|ambiguous|tidak pasti)\b").unwrap(),
                Regex::new(r"\?").unwrap(),
            ],
            specific_claim_patterns: vec![
                Regex::new(r"\b\d{4}\b").unwrap(),
                Regex::new(r"\b\d+\.\d+%?\b").unwrap(),
                Regex::new(r#""[^"]{10,}""#).unwrap(),
                Regex::new(r"(?i)\b(menurut|according to|research|study|penelitian)\b").unwrap(),
            ],
            recency_patterns: vec![
                Regex::new(r"(?i)\b(tahun ini|this year|recent|baru-baru|terbaru|202[4-9])\b").unwrap(),
                Regex::new(r"(?i)\b(saat ini|currently|now|sekarang)\b").unwrap(),
            ],
        }
    }

    pub fn check(&self, input: &str, context: Option<&str>) -> Result<PreGenCheckResult, crate::HallucinationError> {
        let in_scope = self.check_scope(input);
        let ambiguity_score = self.compute_ambiguity(input);
        let context_sufficiency = self.compute_context_sufficiency(input, context);
        let suggestions = self.generate_suggestions(ambiguity_score, context_sufficiency);

        let can_proceed = in_scope
            && ambiguity_score < self.config.ambiguity_threshold;

        Ok(PreGenCheckResult {
            can_proceed,
            in_scope,
            ambiguity_score,
            context_sufficiency,
            reason: if !can_proceed {
                if !in_scope { "Out of scope".into() }
                else if ambiguity_score >= self.config.ambiguity_threshold {
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
            "internal server", "proprietary", "confidential",
            "rahasia", "internal data", "future prediction",
            "stock price", "election result",
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
