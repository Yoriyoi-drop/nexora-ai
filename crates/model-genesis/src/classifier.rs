use nexora_foundation::model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const QUALITY_DIMENSIONS: [&str; 6] = [
    "clarity", "depth", "accuracy", "structure", "conciseness", "engagement",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{QUALITY_DIMENSIONS.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
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
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(QUALITY_DIMENSIONS[0].to_string(), 1.0)],
    };
    clf.predict_sorted(token_ids, &QUALITY_DIMENSIONS, QUALITY_DIMENSIONS[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_cls(hidden: usize) -> GenericClassifier<6> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_quality_focus("x", &[]);
        assert_eq!(r[0].0, "clarity");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(768);
        let r = cls.predict(&[], &QUALITY_DIMENSIONS, "clarity");
        assert_eq!(r[0].0, "clarity");
    }

    #[test]
    fn test_predict_returns_all_dimensions() {
        let cls = init_cls(768);
        let r = cls.predict(&[0, 1], &QUALITY_DIMENSIONS, "clarity");
        assert_eq!(r.len(), QUALITY_DIMENSIONS.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
