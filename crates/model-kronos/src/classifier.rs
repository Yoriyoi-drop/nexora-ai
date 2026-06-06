use nexora_model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const TEMPORAL_MODES: [&str; 5] = [
    "urgent", "scheduled", "historical", "realtime", "evergreen",
];
const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<GenericClassifier<{TEMPORAL_MODES.len()}>> = OnceLock::new();

pub fn init_classifier(embed_table: Array2<f32>) {
    CLASSIFIER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

pub fn temporal_framing(mode: &str) -> &'static str {
    match mode {
        "urgent" => "Time-sensitive! Provide immediate actionable response considering urgency.",
        "scheduled" => "Consider scheduling, timelines, deadlines, and chronological planning.",
        "historical" => "Analyze historical context, past events, and temporal evolution.",
        "realtime" => "Consider real-time dynamics, current state, and live updates.",
        "evergreen" => "Provide timeless analysis applicable across temporal contexts.",
        _ => "Consider temporal context and chronological relationships.",
    }
}

pub fn detect_temporal_mode(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = match CLASSIFIER.get() {
        Some(c) => c,
        None => return vec![(TEMPORAL_MODES[0].to_string(), 1.0)],
    };
    clf.predict_sorted(token_ids, &TEMPORAL_MODES, TEMPORAL_MODES[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_cls(hidden: usize) -> GenericClassifier<5> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_detect_default_on_uninit() {
        let r = detect_temporal_mode("x", &[]);
        assert_eq!(r[0].0, "urgent");
    }

    #[test]
    fn test_predict_empty_ids() {
        let cls = init_cls(256);
        let r = cls.predict(&[], &TEMPORAL_MODES, "evergreen");
        assert_eq!(r[0].0, "evergreen");
    }

    #[test]
    fn test_predict_returns_all_modes() {
        let cls = init_cls(256);
        let r = cls.predict(&[0, 1], &TEMPORAL_MODES, "evergreen");
        assert_eq!(r.len(), TEMPORAL_MODES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
