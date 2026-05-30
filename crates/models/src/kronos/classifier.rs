use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const TEMPORAL_MODES: [&str; 5] = [
    "urgent",
    "scheduled",
    "historical",
    "realtime",
    "evergreen",
];

const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<TemporalClassifier> = OnceLock::new();

pub struct TemporalClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl TemporalClassifier {
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
            w2: classifier_util::xavier_init(HIDDEN, TEMPORAL_MODES.len()),
            b2: Array1::zeros(TEMPORAL_MODES.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("evergreen".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        let mut results: Vec<_> = TEMPORAL_MODES
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
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
    let classifier = CLASSIFIER.get().unwrap_or_else(TemporalClassifier::global);
    if token_ids.is_empty() || !classifier.is_initialized() {
        return vec![("evergreen".to_string(), 1.0)];
    }
    classifier.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> TemporalClassifier {
        TemporalClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> TemporalClassifier {
        TemporalClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, TEMPORAL_MODES.len()),
            b2: Array1::zeros(TEMPORAL_MODES.len()),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!TemporalClassifier::global().is_initialized());
    }

    #[test]
    fn test_uninit_not_initialized() {
        assert!(!uninit().is_initialized());
    }

    #[test]
    fn test_init_initialized() {
        assert!(init_cls(256).is_initialized());
    }

    #[test]
    fn test_predict_default_on_uninit() {
        assert_eq!(uninit().predict(&[1])[0].0, "evergreen");
    }

    #[test]
    fn test_predict_empty_ids() {
        assert_eq!(init_cls(256).predict(&[])[0].0, "evergreen");
    }

    #[test]
    fn test_predict_returns_all_modes() {
        let r = init_cls(256).predict(&[0, 1]);
        assert_eq!(r.len(), TEMPORAL_MODES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        assert_eq!(detect_temporal_mode("x", &[])[0].0, "evergreen");
    }
}
