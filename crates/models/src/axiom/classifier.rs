use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const REASONING_TYPES: [&str; 6] = [
    "deductive",
    "inductive",
    "abductive",
    "analogical",
    "causal",
    "analytical",
];

const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<ReasoningClassifier> = OnceLock::new();

pub struct ReasoningClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl ReasoningClassifier {
    pub fn global() -> &'static Self {
        CLASSIFIER.get_or_init(|| Self {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        })
    }

    pub fn init(embed_table: Array2<f32>) {
        let hidden_size = embed_table.shape()[1];
        let _ = CLASSIFIER.set(Self {
            w1: classifier_util::xavier_init(hidden_size, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, REASONING_TYPES.len()),
            b2: Array1::zeros(REASONING_TYPES.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("analytical".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        let mut results: Vec<_> = REASONING_TYPES
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
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
    let classifier = CLASSIFIER.get().unwrap_or_else(ReasoningClassifier::global);
    if token_ids.is_empty() || !classifier.is_initialized() {
        return vec![("analytical".to_string(), 1.0)];
    }
    classifier.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> ReasoningClassifier {
        ReasoningClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> ReasoningClassifier {
        ReasoningClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, REASONING_TYPES.len()),
            b2: Array1::zeros(REASONING_TYPES.len()),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!ReasoningClassifier::global().is_initialized());
    }

    #[test]
    fn test_uninit_not_initialized() {
        assert!(!uninit().is_initialized());
    }

    #[test]
    fn test_init_initialized() {
        assert!(init_cls(768).is_initialized());
    }

    #[test]
    fn test_predict_default_on_uninit() {
        let r = uninit().predict(&[1]);
        assert_eq!(r[0].0, "analytical");
    }

    #[test]
    fn test_predict_empty_ids() {
        let r = init_cls(768).predict(&[]);
        assert_eq!(r[0].0, "analytical");
    }

    #[test]
    fn test_predict_returns_all_types() {
        let r = init_cls(768).predict(&[0, 1]);
        assert_eq!(r.len(), REASONING_TYPES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_reasoning_type("x", &[]);
        assert_eq!(r[0].0, "analytical");
    }
}
