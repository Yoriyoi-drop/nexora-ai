use crate::classifier_util;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const REVIEW_CATEGORIES: [&str; 6] = [
    "bugs",
    "security",
    "performance",
    "style",
    "architecture",
    "general",
];

const HIDDEN: usize = 64;

static ANALYZER: OnceLock<CodeReviewClassifier> = OnceLock::new();

pub struct CodeReviewClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl CodeReviewClassifier {
    pub fn global() -> &'static Self {
        ANALYZER.get_or_init(|| Self {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        })
    }

    pub fn init(embed_table: Array2<f32>) {
        let hidden_size = embed_table.shape()[1];
        let _ = ANALYZER.set(Self {
            w1: classifier_util::xavier_init(hidden_size, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, REVIEW_CATEGORIES.len()),
            b2: Array1::zeros(REVIEW_CATEGORIES.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("general".to_string(), 1.0)];
        }

        let avg = classifier_util::embed_average(&self.embed_table, token_ids);
        let h = classifier_util::gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = classifier_util::softmax(&logits);

        let mut results: Vec<_> = REVIEW_CATEGORIES
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

pub fn detect_language(code: &str) -> &'static str {
    let code = code.trim();
    if code.contains("fn ") && (code.contains("->") || code.contains("mut ") || code.contains("let ")) {
        "rust"
    } else if code.contains("def ") || code.contains("import ") || code.contains("class ") && code.contains(":") {
        "python"
    } else if code.contains("function ") || code.contains("const ") && code.contains("=>") || code.contains("let ") && code.contains("var ") {
        "javascript"
    } else if code.contains("public class ") || code.contains("private ") || code.contains("void ") && code.contains("{") {
        "java"
    } else if code.contains("#include") || code.contains("int main") || code.contains("std::") {
        "cpp"
    } else if code.contains("package ") || code.contains("import ") && code.contains("fmt") {
        "go"
    } else if code.contains("SELECT ") || code.contains("FROM ") || code.contains("WHERE ") {
        "sql"
    } else {
        "unknown"
    }
}

const CATEGORY_PROMPTS: &[(&str, &str)] = &[
    ("bugs", "Focus on identifying bugs, logic errors, edge cases, and runtime issues."),
    ("security", "Focus on security vulnerabilities, injection risks, authentication flaws, and unsafe patterns."),
    ("performance", "Focus on performance bottlenecks, unnecessary allocations, algorithmic complexity, and optimization opportunities."),
    ("style", "Focus on code style, naming conventions, formatting, idiomatic patterns, and maintainability."),
    ("architecture", "Focus on architectural design, coupling, cohesion, separation of concerns, and design patterns."),
    ("general", "Provide a comprehensive code review covering all aspects."),
];

pub fn category_focus(category: &str) -> &'static str {
    CATEGORY_PROMPTS
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, p)| *p)
        .unwrap_or("Provide a comprehensive code review covering all aspects.")
}

pub fn analyze_review_type(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let analyzer = ANALYZER.get().unwrap_or_else(CodeReviewClassifier::global);
    if token_ids.is_empty() || !analyzer.is_initialized() {
        return vec![("general".to_string(), 1.0)];
    }
    analyzer.predict(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uninit() -> CodeReviewClassifier {
        CodeReviewClassifier {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        }
    }

    fn init_cls(hidden: usize) -> CodeReviewClassifier {
        CodeReviewClassifier {
            embed_table: Array2::zeros((10, hidden)),
            w1: classifier_util::xavier_init(hidden, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: classifier_util::xavier_init(HIDDEN, REVIEW_CATEGORIES.len()),
            b2: Array1::zeros(REVIEW_CATEGORIES.len()),
        }
    }

    #[test]
    fn test_global_not_initialized() {
        assert!(!CodeReviewClassifier::global().is_initialized());
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
        assert_eq!(uninit().predict(&[1])[0].0, "general");
    }

    #[test]
    fn test_predict_empty_ids() {
        assert_eq!(init_cls(512).predict(&[])[0].0, "general");
    }

    #[test]
    fn test_predict_returns_all_categories() {
        let r = init_cls(512).predict(&[0, 1]);
        assert_eq!(r.len(), REVIEW_CATEGORIES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_default_on_uninit() {
        assert_eq!(analyze_review_type("x", &[])[0].0, "general");
    }

    #[test]
    fn test_detect_language_rust() {
        assert_eq!(detect_language("fn main() -> i32 { 0 }"), "rust");
    }

    #[test]
    fn test_detect_language_python() {
        assert_eq!(detect_language("def hello(): print('hi')"), "python");
    }

    #[test]
    fn test_detect_language_javascript() {
        assert_eq!(detect_language("const x = () => 42;"), "javascript");
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language("xyzzy"), "unknown");
    }
}
