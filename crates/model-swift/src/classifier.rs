use nexora_model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const TASK_TYPES: [&str; 5] = [
    "qa", "summarize", "translate", "generate", "analyze",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{TASK_TYPES.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

pub fn task_params(task_type: &str) -> (usize, f32) {
    match task_type {
        "qa" => (256, 0.3),
        "summarize" => (256, 0.4),
        "translate" => (256, 0.2),
        "generate" => (512, 0.9),
        "analyze" => (512, 0.3),
        _ => (256, 0.7),
    }
}

pub fn detect_task_type(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(TASK_TYPES[0].to_string(), 1.0)],
    };
    clf.predict_sorted(token_ids, &TASK_TYPES, TASK_TYPES[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_cls(hidden: usize) -> GenericClassifier<5> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_task_type("x", &[]);
        assert_eq!(r[0].0, "qa");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(128);
        let r = cls.predict(&[], &TASK_TYPES, "qa");
        assert_eq!(r[0].0, "qa");
    }

    #[test]
    fn test_predict_returns_all_types() {
        let cls = init_cls(128);
        let r = cls.predict(&[0, 1], &TASK_TYPES, "qa");
        assert_eq!(r.len(), TASK_TYPES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
