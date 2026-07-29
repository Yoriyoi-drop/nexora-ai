use nexora_foundation::model_core::classifier_util::GenericClassifier;
use ndarray::Array2;
use std::sync::OnceLock;

pub const REVIEW_CATEGORIES: [&str; 6] = [
    "bugs", "security", "performance", "style", "architecture", "general",
];
const HIDDEN: usize = 64;

static ANALYZER: OnceLock<GenericClassifier<{REVIEW_CATEGORIES.len()}>> = OnceLock::new();

pub fn init_analyzer(embed_table: Array2<f32>) {
    ANALYZER.set(GenericClassifier::new(embed_table, HIDDEN)).ok();
}

pub fn detect_language(code: &str) -> &'static str {
    nexora_oracle::linters::detect_language(code)
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
    let analyzer = match ANALYZER.get() {
        Some(a) => a,
        None => return vec![(REVIEW_CATEGORIES[0].to_string(), 1.0)],
    };
    analyzer.predict_sorted(token_ids, &REVIEW_CATEGORIES, REVIEW_CATEGORIES[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_analyzer(hidden: usize) -> GenericClassifier<6> {
        GenericClassifier::new(Array2::zeros((10, hidden)), HIDDEN)
    }

    #[test]
    fn test_analyze_default_on_uninit() {
        let r = analyze_review_type("x", &[]);
        assert_eq!(r[0].0, "bugs");
    }

    #[test]
    fn test_predict_empty_ids() {
        let a = init_analyzer(512);
        let r = a.predict(&[], &REVIEW_CATEGORIES, "general");
        assert_eq!(r[0].0, "general");
    }

    #[test]
    fn test_predict_returns_all_categories() {
        let a = init_analyzer(512);
        let r = a.predict(&[0, 1], &REVIEW_CATEGORIES, "general");
        assert_eq!(r.len(), REVIEW_CATEGORIES.len());
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
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
        assert_eq!(detect_language("function foo() { return 1; }"), "javascript");
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language("xyzzy"), "text");
    }
}
