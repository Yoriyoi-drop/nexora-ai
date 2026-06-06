use nexora_model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const CREATIVE_STYLES: [&str; 6] = [
    "narrative", "poetic", "persuasive", "technical", "dialogue", "descriptive",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{CREATIVE_STYLES.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
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
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(CREATIVE_STYLES[0].to_string(), 1.0)],
    };
    clf.predict_sorted(token_ids, &CREATIVE_STYLES, CREATIVE_STYLES[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_cls(hidden: usize) -> GenericClassifier<6> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_creative_style("x", &[]);
        assert_eq!(r[0].0, "narrative");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(384);
        let r = cls.predict(&[], &CREATIVE_STYLES, "narrative");
        assert_eq!(r[0].0, "narrative");
    }

    #[test]
    fn test_predict_returns_all_styles() {
        let cls = init_cls(384);
        let r = cls.predict(&[0, 1], &CREATIVE_STYLES, "narrative");
        assert_eq!(r.len(), CREATIVE_STYLES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
