use nexora_model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const NUM_EMOTIONS: usize = 8;
pub const EMOTION_LABELS: [&str; NUM_EMOTIONS] = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "neutral",
];
const HIDDEN: usize = 64;

static CLASSIFIER: OnceLock<GenericClassifier<NUM_EMOTIONS>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

pub fn detect_emotions(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(EMOTION_LABELS[0].to_string(), 1.0)],
    };
    clf.predict(token_ids, &EMOTION_LABELS, EMOTION_LABELS[0])
}

#[cfg(test)]
mod tests {
    use super::*;
fn init_cls(hidden: usize) -> GenericClassifier<NUM_EMOTIONS> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_emotions("hello", &[]);
        assert_eq!(r[0].0, "joy");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(512);
        let r = cls.predict(&[], &EMOTION_LABELS, "neutral");
        assert_eq!(r[0].0, "neutral");
    }

    #[test]
    fn test_predict_returns_all_emotions() {
        let cls = init_cls(512);
        let r = cls.predict(&[0, 1, 2], &EMOTION_LABELS, "neutral");
        assert_eq!(r.len(), NUM_EMOTIONS);
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
