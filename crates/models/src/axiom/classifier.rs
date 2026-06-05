use crate::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const REASONING_TYPES: [&str; 6] = [
    "deductive", "inductive", "abductive", "analogical", "causal", "analytical",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{REASONING_TYPES.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

const REASONING_PROMPTS: &[(&str, &str)] = &[
    ("deductive", "Apply deductive reasoning from general principles to specific conclusions."),
    ("inductive", "Use inductive reasoning to derive general patterns from specific observations."),
    ("abductive", "Apply abductive reasoning to infer the most likely explanation."),
    ("analogical", "Use analogical reasoning by comparing with known similar cases."),
    ("causal", "Apply causal reasoning to identify cause-effect relationships."),
    ("analytical", "Use systematic analytical reasoning breaking down the problem into components."),
];

pub fn reasoning_prompt(reasoning_type: &str) -> &'static str {
    REASONING_PROMPTS
        .iter()
        .find(|(r, _)| *r == reasoning_type)
        .map(|(_, p)| *p)
        .unwrap_or("Use systematic analytical reasoning breaking down the problem into components.")
}

pub fn detect_reasoning_type(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(REASONING_TYPES[0].to_string(), 1.0)],
    };
    clf.predict_sorted(token_ids, &REASONING_TYPES, REASONING_TYPES[0])
}

#[cfg(test)]
mod tests {
    use super::*;
fn init_cls(hidden: usize) -> GenericClassifier<6> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_reasoning_type("x", &[]);
        assert_eq!(r[0].0, "deductive");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(768);
        let r = cls.predict(&[], &REASONING_TYPES, "analytical");
        assert_eq!(r[0].0, "analytical");
    }

    #[test]
    fn test_predict_returns_all_types() {
        let cls = init_cls(768);
        let r = cls.predict(&[0, 1], &REASONING_TYPES, "analytical");
        assert_eq!(r.len(), REASONING_TYPES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
