use crate::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const COMPLEXITY_LEVELS: [&str; 4] = [
    "simple", "moderate", "complex", "multi_domain",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{COMPLEXITY_LEVELS.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

pub fn decomposition_strategy(level: &str) -> &'static str {
    match level {
        "simple" => "Single direct response — no decomposition needed.",
        "moderate" => "Split into 2-3 logical subtasks, solve sequentially.",
        "complex" => "Decompose into 3-5 subtasks with dependencies. Solve each with reasoning chain.",
        "multi_domain" => "Decompose by domain expertise. Each subtask assigned to domain expert. Synthesize results.",
        _ => "Split into manageable subtasks and solve independently.",
    }
}

pub fn detect_complexity(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(COMPLEXITY_LEVELS[0].to_string(), 1.0)],
    };
    clf.predict_sorted(token_ids, &COMPLEXITY_LEVELS, COMPLEXITY_LEVELS[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_cls(hidden: usize) -> GenericClassifier<4> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_complexity("x", &[]);
        assert_eq!(r[0].0, "simple");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(512);
        let r = cls.predict(&[], &COMPLEXITY_LEVELS, "simple");
        assert_eq!(r[0].0, "simple");
    }

    #[test]
    fn test_predict_returns_all_levels() {
        let cls = init_cls(512);
        let r = cls.predict(&[0, 1], &COMPLEXITY_LEVELS, "simple");
        assert_eq!(r.len(), COMPLEXITY_LEVELS.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
