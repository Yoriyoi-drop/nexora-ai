use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const QUALITY_DIMENSIONS: [&str; 6] = [
    "clarity",
    "depth",
    "accuracy",
    "structure",
    "conciseness",
    "engagement",
];

const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<QualityClassifier> = OnceLock::new();

pub struct QualityClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl QualityClassifier {
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
            w2: classifier_util::xavier_init(HIDDEN, QUALITY_DIMENSIONS.len()),
            b2: Array1::zeros(QUALITY_DIMENSIONS.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("clarity".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        let mut results: Vec<_> = QUALITY_DIMENSIONS
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

pub fn refinement_focus(dimension: &str) -> &'static str {
    match dimension {
        "clarity" => "Improve clarity: simplify language, define terms, clarify intent.",
        "depth" => "Add depth: include more detail, examples, nuance, and thoroughness.",
        "accuracy" => "Improve accuracy: fact-check, add precision, and correct errors.",
        "structure" => "Improve structure: reorganize for logical flow, add headings, improve readability.",
        "conciseness" => "Make more concise: remove redundancy, tighten prose, focus on essentials.",
        "engagement" => "Improve engagement: add examples, rhetorical devices, and compelling language.",
        _ => "Improve overall quality: clarity, depth, accuracy, structure, conciseness, and engagement.",
    }
}

pub fn detect_quality_focus(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let classifier = CLASSIFIER.get().unwrap_or_else(QualityClassifier::global);
    if token_ids.is_empty() || !classifier.is_initialized() {
        return vec![("clarity".to_string(), 1.0)];
    }
    classifier.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> QualityClassifier {
        QualityClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> QualityClassifier {
        QualityClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, QUALITY_DIMENSIONS.len()),
            b2: Array1::zeros(QUALITY_DIMENSIONS.len()),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!QualityClassifier::global().is_initialized());
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
        assert_eq!(uninit().predict(&[1])[0].0, "clarity");
    }

    #[test]
    fn test_predict_empty_ids() {
        assert_eq!(init_cls(768).predict(&[])[0].0, "clarity");
    }

    #[test]
    fn test_predict_returns_all_dimensions() {
        let r = init_cls(768).predict(&[0, 1]);
        assert_eq!(r.len(), QUALITY_DIMENSIONS.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        assert_eq!(detect_quality_focus("x", &[])[0].0, "clarity");
    }
}
