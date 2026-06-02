/// Keputusan apakah Oracle perlu aktif atau skip
#[derive(Debug, Clone)]
pub struct ActivationDecision {
    pub needs_oracle: bool,
    pub confidence: f32,
    pub reason: String,
}

/// Kategori input yang dideteksi classifier
#[derive(Debug, Clone, PartialEq)]
pub enum InputCategory {
    SimpleQA,
    GenericChat,
    CodeReview,
    ComplexReasoning,
    Planning,
    CodingComplex,
    Multimodal,
    SecurityAnalysis,
}

impl InputCategory {
    pub fn needs_oracle(&self) -> bool {
        matches!(
            self,
            InputCategory::ComplexReasoning
                | InputCategory::CodeReview
                | InputCategory::Planning
                | InputCategory::CodingComplex
                | InputCategory::Multimodal
                | InputCategory::SecurityAnalysis
        )
    }
}

/// Klasifikasi heuristik berdasarkan keyword — fallback cepat tanpa MLP.
fn classify_heuristic(text: &str) -> (InputCategory, f32) {
    let lower = text.to_lowercase();

    let security_keywords = [
        "injection", "xss", "csrf", "sql injection", "authentication",
        "authorization", "password", "encrypt", "decrypt", "cipher",
        "vulnerability", "cve", "exploit", "malware", "ransomware",
        "backdoor", "zero-day", "payload", "sanitize",
    ];
    let security_count = security_keywords
        .iter()
        .filter(|k| lower.contains(*k))
        .count();
    if security_count >= 2 {
        return (InputCategory::SecurityAnalysis, 0.85);
    }

    let reasoning_keywords = [
        "reasoning", "deduce", "infer", "logical", "implication",
        "syllogism", "contradiction", "counterfactual", "causal",
        "if and only if", "therefore", "because", "why is",
    ];
    let reasoning_count = reasoning_keywords
        .iter()
        .filter(|k| lower.contains(*k))
        .count();
    if reasoning_count >= 3 {
        return (InputCategory::ComplexReasoning, 0.75);
    }

    let code_keywords = [
        "code review", "review this code", "check my code",
        "refactor", "code quality", "bug", "fix this", "optimize",
        "performance issue", "code smell", "technical debt",
    ];
    let code_count = code_keywords
        .iter()
        .filter(|k| lower.contains(*k))
        .count();
    if code_count >= 2 {
        return (InputCategory::CodeReview, 0.70);
    }

    let coding_keywords = [
        "implement", "design pattern", "architecture", "system design",
        "algorithm", "data structure", "concurrent", "parallel",
        "distributed", "microservice", "rest api", "graphql",
        "database schema", "orm", "migration", "deploy",
    ];
    let coding_count = coding_keywords
        .iter()
        .filter(|k| lower.contains(*k))
        .count();
    if coding_count >= 3 {
        return (InputCategory::CodingComplex, 0.80);
    }

    let planning_keywords = [
        "plan", "roadmap", "milestone", "sprint", "timeline",
        "strategy", "goal", "objective", "phases", "step by step",
        "project", "task manager", "prioritize",
    ];
    if planning_keywords.iter().any(|k| lower.contains(k)) {
        return (InputCategory::Planning, 0.65);
    }

    let multimodal_keywords = [
        "image", "audio", "video", "multimodal", "pixel",
        "speech", "voice", "visual", "segmentation", "detection",
    ];
    if multimodal_keywords.iter().any(|k| lower.contains(k)) {
        return (InputCategory::Multimodal, 0.75);
    }

    let simple_count = lower.split_whitespace().count();
    if simple_count < 15 {
        return (InputCategory::SimpleQA, 0.90);
    }

    (InputCategory::GenericChat, 0.60)
}

/// Activation Classifier — menentukan apakah request perlu Oracle
pub struct OracleActivationClassifier {
    weights_loaded: bool,
    threshold: f32,
    total_requests: u64,
    oracle_activated: u64,
}

impl OracleActivationClassifier {
    pub fn new(threshold: f32) -> Self {
        Self {
            weights_loaded: false,
            threshold,
            total_requests: 0,
            oracle_activated: 0,
        }
    }

    pub fn activation_rate(&self) -> f32 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.oracle_activated as f32 / self.total_requests as f32
    }

    pub fn should_activate(&mut self, text: &str) -> ActivationDecision {
        self.total_requests += 1;
        let (category, confidence) = classify_heuristic(text);
        let needs = category.needs_oracle() && confidence >= self.threshold;

        if needs {
            self.oracle_activated += 1;
        }

        ActivationDecision {
            needs_oracle: needs,
            confidence,
            reason: format!("heuristic: {:?} conf={:.2} thresh={:.2}", category, confidence, self.threshold),
        }
    }

    pub fn skip_reason(text: &str) -> Option<String> {
        let (category, _confidence) = classify_heuristic(text);
        if category.needs_oracle() {
            None
        } else {
            Some(format!("{:?} — tidak perlu Oracle", category))
        }
    }
}

impl Default for OracleActivationClassifier {
    fn default() -> Self {
        Self::new(0.60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_qa_skip() {
        let mut classifier = OracleActivationClassifier::default();
        let decision = classifier.should_activate("Apa ibu kota Indonesia?");
        assert!(!decision.needs_oracle);
    }

    #[test]
    fn test_complex_reasoning_activate() {
        let mut classifier = OracleActivationClassifier::default();
        let decision = classifier.should_activate(
            "Jelaskan reasoning logis mengapa counterfactual ini menyebabkan \
             kontradiksi dengan inferensi kausal sebelumnya.",
        );
        assert!(decision.needs_oracle);
    }

    #[test]
    fn test_code_review_activate() {
        let mut classifier = OracleActivationClassifier::default();
        let decision = classifier.should_activate(
            "Tolong review code ini. Ada bug dan perlu refactor.",
        );
        assert!(decision.needs_oracle);
    }

    #[test]
    fn test_activation_rate_tracking() {
        let mut classifier = OracleActivationClassifier::default();
        classifier.should_activate("hai");
        classifier.should_activate("apa kabar");
        classifier.should_activate("review code ini ada bug dan perlu refactor");
        let rate = classifier.activation_rate();
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }
}
