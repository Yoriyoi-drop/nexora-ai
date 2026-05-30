use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

const NUM_EMOTIONS: usize = 8;
const HIDDEN: usize = 64;

static CLASSIFIER: OnceLock<EmotionClassifier> = OnceLock::new();

pub struct EmotionClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl EmotionClassifier {
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
            w2: classifier_util::xavier_init(HIDDEN, NUM_EMOTIONS),
            b2: Array1::zeros(NUM_EMOTIONS),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("neutral".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        EMOTION_LABELS.iter().zip(probs.iter()).map(|(label, &p)| (label.to_string(), p)).collect()
    }
}

pub const EMOTION_LABELS: [&str; NUM_EMOTIONS] = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "neutral",
];

pub fn detect_emotions(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = CLASSIFIER.get().unwrap_or_else(EmotionClassifier::global);
    if token_ids.is_empty() || !clf.is_initialized() {
        return vec![("neutral".to_string(), 1.0)];
    }
    clf.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> EmotionClassifier {
        EmotionClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> EmotionClassifier {
        EmotionClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, NUM_EMOTIONS),
            b2: Array1::zeros(NUM_EMOTIONS),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        let clf = EmotionClassifier::global();
        assert!(!clf.is_initialized());
    }

    #[test]
    fn test_uninit_not_initialized() {
        assert!(!uninit().is_initialized());
    }

    #[test]
    fn test_init_initialized() {
        assert!(init_cls(512).is_initialized());
    }

    #[test]
    fn test_predict_default_on_uninit() {
        let result = uninit().predict(&[1, 2, 3]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "neutral");
        assert!((result[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_predict_empty_ids() {
        let result = init_cls(512).predict(&[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "neutral");
    }

    #[test]
    fn test_predict_returns_all_emotions() {
        let result = init_cls(512).predict(&[0, 1, 2]);
        assert_eq!(result.len(), NUM_EMOTIONS);
        let sum: f32 = result.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let result = detect_emotions("hello", &[]);
        assert_eq!(result[0].0, "neutral");
    }
}
