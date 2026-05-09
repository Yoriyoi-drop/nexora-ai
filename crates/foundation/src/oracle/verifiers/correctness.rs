//! Correctness Verifier
//!
//! Implements correctness verification for code logic and bugs.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};

use super::{
    CodeIssue,
    CodeVerifier,
    IssueSeverity,
    VerificationResult,
    VerifierType,
};

/// Correctness pattern
#[derive(Debug, Clone)]
struct CorrectnessPattern {
    name: &'static str,
    pattern: &'static Lazy<Regex>,
    severity: IssueSeverity,
    description: &'static str,
    language: &'static str,
    rule_id: &'static str,
    score_penalty: f32,
}

/// Regex patterns
static DIV_ZERO_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"/\s*0+(\.0+)?\b").unwrap()
});

static INFINITE_LOOP_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"while\s*\(\s*(true|1)\s*\)").unwrap()
});

static C_UNINIT_VAR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"\b(int|float|double|char)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;"
    )
    .unwrap()
});

static JS_LOOSE_EQUALITY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?<![=!])==(?!=)").unwrap()
});

static PY_BARE_EXCEPT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"except\s*:").unwrap()
});

/// Correctness verifier
pub struct CorrectnessVerifier {
    correctness_patterns: Vec<CorrectnessPattern>,
}

impl CorrectnessVerifier {
    pub fn new() -> Self {
        Self {
            correctness_patterns: vec![
                CorrectnessPattern {
                    name: "Division By Zero",
                    pattern: &DIV_ZERO_REGEX,
                    severity: IssueSeverity::Error,
                    description: "Division by literal zero detected",
                    language: "all",
                    rule_id: "division_by_zero",
                    score_penalty: 0.30,
                },
                CorrectnessPattern {
                    name: "Potential Infinite Loop",
                    pattern: &INFINITE_LOOP_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Infinite loop detected, ensure termination logic exists",
                    language: "all",
                    rule_id: "infinite_loop",
                    score_penalty: 0.15,
                },
                CorrectnessPattern {
                    name: "Uninitialized Variable",
                    pattern: &C_UNINIT_VAR_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Variable declared without initialization",
                    language: "c",
                    rule_id: "uninitialized_variable",
                    score_penalty: 0.10,
                },
                CorrectnessPattern {
                    name: "Loose Equality",
                    pattern: &JS_LOOSE_EQUALITY_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Loose equality detected, use === instead",
                    language: "javascript",
                    rule_id: "js_loose_equality",
                    score_penalty: 0.10,
                },
                CorrectnessPattern {
                    name: "Bare Except",
                    pattern: &PY_BARE_EXCEPT_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Bare except detected",
                    language: "python",
                    rule_id: "py_bare_except",
                    score_penalty: 0.10,
                },
            ],
        }
    }

    fn add_issue(
        issues: &mut Vec<CodeIssue>,
        seen: &mut HashSet<String>,
        issue: CodeIssue,
    ) {
        let key = format!(
            "{}:{}",
            issue.rule_id,
            issue.line_number.unwrap_or(0)
        );

        if seen.insert(key) {
            issues.push(issue);
        }
    }
}

impl CodeVerifier for CorrectnessVerifier {
    fn verify(
        &self,
        code: &str,
        language: &str,
    ) -> Result<VerificationResult> {
        let mut issues = Vec::new();
        let mut seen = HashSet::new();
        let mut score: f32 = 1.0;

        // Language-specific checks
        let language_issues =
            self.check_language_specific_correctness(
                code,
                language,
            )?;

        for issue in language_issues {
            Self::add_issue(&mut issues, &mut seen, issue);
        }

        // Generic pattern checks
        for pattern in &self.correctness_patterns {
            if pattern.language != "all"
                && pattern.language != language
            {
                continue;
            }

            for mat in pattern.pattern.find_iter(code) {
                let line_number =
                    code[..mat.start()].lines().count().max(1);

                Self::add_issue(
                    &mut issues,
                    &mut seen,
                    CodeIssue {
                        severity: pattern.severity.clone(),
                        category: "Correctness".to_string(),
                        message: format!(
                            "{}: {}",
                            pattern.name,
                            pattern.description
                        ),
                        line_number: Some(line_number),
                        column_number: None,
                        rule_id: pattern.rule_id.to_string(),
                    },
                );

                score -= pattern.score_penalty;
            }
        }

        score = score.clamp(0.0, 1.0);

        let suggestions =
            self.generate_correctness_suggestions(&issues);

        let mut metrics = HashMap::new();

        metrics.insert(
            "correctness_score".to_string(),
            score,
        );

        metrics.insert(
            "correctness_issue_count".to_string(),
            issues.len() as f32,
        );

        metrics.insert(
            "critical_issue_count".to_string(),
            issues
                .iter()
                .filter(|i| {
                    matches!(i.severity, IssueSeverity::Error)
                })
                .count() as f32,
        );

        Ok(VerificationResult {
            verifier_name: "CorrectnessVerifier".to_string(),
            verifier_type: VerifierType::Correctness,
            score,
            passed: score >= 0.75,
            issues,
            suggestions,
            metrics,
        })
    }

    fn verifier_name(&self) -> &str {
        "CorrectnessVerifier"
    }

    fn verifier_type(&self) -> VerifierType {
        VerifierType::Correctness
    }

    fn check_language_specific_correctness(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();

        match language {
            "rust" => {
                let unwrap_count =
                    code.matches(".unwrap()").count();

                if unwrap_count > 3 {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Correctness".to_string(),
                        message:
                            format!("Excessive unwrap() usage ({unwrap_count})"),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_unwrap".to_string(),
                    });
                }

                if code.contains("panic!") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Correctness".to_string(),
                        message:
                            "panic! macro detected".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_panic".to_string(),
                    });
                }

                if code.contains("unsafe {") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Correctness".to_string(),
                        message:
                            "unsafe block detected, review memory safety"
                                .to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_unsafe".to_string(),
                    });
                }
            }

            "javascript" => {
                if code.contains("eval(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Error,
                        category: "Correctness".to_string(),
                        message:
                            "eval() usage detected".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_eval".to_string(),
                    });
                }
            }

            "python" => {
                if code.contains("global ") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Correctness".to_string(),
                        message:
                            "Global variable usage detected"
                                .to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_global".to_string(),
                    });
                }
            }

            _ => {}
        }

        Ok(issues)
    }

    fn generate_correctness_suggestions(
        &self,
        issues: &[CodeIssue],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if issues.iter().any(|i| {
            i.rule_id == "division_by_zero"
        }) {
            suggestions.push(
                "Validate divisor before division"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| {
            i.rule_id == "infinite_loop"
        }) {
            suggestions.push(
                "Ensure loops contain proper exit conditions"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| {
            i.rule_id == "rust_unwrap"
        }) {
            suggestions.push(
                "Use proper Result handling instead of unwrap()"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| {
            i.rule_id == "js_loose_equality"
        }) {
            suggestions.push(
                "Use strict equality operators (===)"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| {
            i.rule_id == "py_bare_except"
        }) {
            suggestions.push(
                "Catch specific exception types"
                    .to_string(),
            );
        }

        suggestions
    }
}
