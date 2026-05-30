use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const CREATIVE_STYLES: [&str; 6] = [
    "narrative",
    "poetic",
    "persuasive",
    "technical",
    "dialogue",
    "descriptive",
];

const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<StyleClassifier> = OnceLock::new();

pub struct StyleClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl StyleClassifier {
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
            w2: classifier_util::xavier_init(HIDDEN, CREATIVE_STYLES.len()),
            b2: Array1::zeros(CREATIVE_STYLES.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("narrative".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        let mut results: Vec<_> = CREATIVE_STYLES
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

pub fn style_params(style: &str) -> (f32, f32) {
    match style {
        "narrative" => (0.8, 0.85),
        "poetic" => (1.0, 0.9),
        "persuasive" => (0.7, 0.6),
        "technical" => (0.3, 0.3),
        "dialogue" => (0.9, 0.75),
        "descriptive" => (0.85, 0.7),
        _ => (0.7, 0.7),
    }
}

pub fn style_framing(style: &str) -> &'static str {
    match style {
        "narrative" => "Write a compelling narrative with plot, character, and setting.",
        "poetic" => "Write expressively with rhythm, imagery, and figurative language.",
        "persuasive" => "Write persuasively with arguments, evidence, and rhetorical devices.",
        "technical" => "Write with technical precision, clarity, and structured exposition.",
        "dialogue" => "Write as natural dialogue between characters with distinct voices.",
        "descriptive" => "Write vividly with rich sensory detail and immersive description.",
        _ => "Write creatively with originality and expressiveness.",
    }
}

pub fn detect_creative_style(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let classifier = CLASSIFIER.get().unwrap_or_else(StyleClassifier::global);
    if token_ids.is_empty() || !classifier.is_initialized() {
        return vec![("narrative".to_string(), 1.0)];
    }
    classifier.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> StyleClassifier {
        StyleClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> StyleClassifier {
        StyleClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, CREATIVE_STYLES.len()),
            b2: Array1::zeros(CREATIVE_STYLES.len()),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!StyleClassifier::global().is_initialized());
    }

    #[test]
    fn test_uninit_not_initialized() {
        assert!(!uninit().is_initialized());
    }

    #[test]
    fn test_init_initialized() {
        assert!(init_cls(384).is_initialized());
    }

    #[test]
    fn test_predict_default_on_uninit() {
        assert_eq!(uninit().predict(&[1])[0].0, "narrative");
    }

    #[test]
    fn test_predict_empty_ids() {
        assert_eq!(init_cls(384).predict(&[])[0].0, "narrative");
    }

    #[test]
    fn test_predict_returns_all_styles() {
        let r = init_cls(384).predict(&[0, 1]);
        assert_eq!(r.len(), CREATIVE_STYLES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        assert_eq!(detect_creative_style("x", &[])[0].0, "narrative");
    }
}
