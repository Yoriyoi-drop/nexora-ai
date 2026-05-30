use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const THREAT_CATEGORIES: [&str; 6] = [
    "injection",
    "xss",
    "auth",
    "crypto",
    "config",
    "network",
];

const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<ThreatClassifier> = OnceLock::new();

pub struct ThreatClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl ThreatClassifier {
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
            w2: classifier_util::xavier_init(HIDDEN, THREAT_CATEGORIES.len()),
            b2: Array1::zeros(THREAT_CATEGORIES.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("injection".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        let mut results: Vec<_> = THREAT_CATEGORIES
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

const THREAT_PROMPTS: &[(&str, &str)] = &[
    ("injection", "Focus on SQL/command injection, input sanitization, and parameterized queries."),
    ("xss", "Focus on cross-site scripting, content security policy, and output encoding."),
    ("auth", "Focus on authentication bypass, session management, and access control."),
    ("crypto", "Focus on cryptographic weaknesses, key management, and secure protocols."),
    ("config", "Focus on security misconfiguration, default credentials, and exposed secrets."),
    ("network", "Focus on network security, TLS, API security, and data-in-transit."),
];

pub fn threat_focus(threat: &str) -> &'static str {
    THREAT_PROMPTS
        .iter()
        .find(|(t, _)| *t == threat)
        .map(|(_, p)| *p)
        .unwrap_or("Focus on all security aspects including injection, XSS, auth, crypto, config, and network.")
}

pub fn detect_threat_type(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let classifier = CLASSIFIER.get().unwrap_or_else(ThreatClassifier::global);
    if token_ids.is_empty() || !classifier.is_initialized() {
        return vec![("injection".to_string(), 1.0)];
    }
    classifier.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> ThreatClassifier {
        ThreatClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> ThreatClassifier {
        ThreatClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, THREAT_CATEGORIES.len()),
            b2: Array1::zeros(THREAT_CATEGORIES.len()),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!ThreatClassifier::global().is_initialized());
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
        assert_eq!(uninit().predict(&[1])[0].0, "injection");
    }

    #[test]
    fn test_predict_empty_ids() {
        assert_eq!(init_cls(384).predict(&[])[0].0, "injection");
    }

    #[test]
    fn test_predict_returns_all_categories() {
        let r = init_cls(384).predict(&[0, 1]);
        assert_eq!(r.len(), THREAT_CATEGORIES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        assert_eq!(detect_threat_type("x", &[])[0].0, "injection");
    }
}
