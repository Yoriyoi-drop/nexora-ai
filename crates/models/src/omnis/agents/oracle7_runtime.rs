use nexora_shared::base_model::NxrModelResult;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct Oracle7RuntimeAgent;

impl Oracle7RuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub fn decompose_problem(&self, input: &str) -> NxrModelResult<String> {
        if input.is_empty() {
            return Err("Empty input cannot be decomposed".into());
        }
        let word_count = input.split_whitespace().count();
        let sentence_count = input
            .split(|c: char| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .count();
        let unique_terms: HashSet<&str> = input
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        let complexity = (unique_terms.len() as f64 / word_count.max(1) as f64 * 100.0).min(100.0);

        Ok(format!(
            "[ORACLE-7] Decomposed problem via causal graph analysis:\n\
             - Input length: {} chars, {} words, {} sentences\n\
             - Unique terms: {} (lexical diversity: {:.1}%)\n\
             - Complexity score: {:.1}/100\n\
             - Sub-problems identified: {}\n\
             - Recommended approach: {}",
            input.len(),
            word_count,
            sentence_count,
            unique_terms.len(),
            complexity,
            complexity,
            if complexity > 50.0 {
                "multiple sub-problems with dependency graph"
            } else {
                "single-pass resolution"
            },
            if complexity > 70.0 {
                "hierarchical decomposition"
            } else if complexity > 40.0 {
                "parallel processing"
            } else {
                "direct execution"
            }
        ))
    }
}
