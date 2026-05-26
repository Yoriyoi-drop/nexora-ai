//! Style Verifier
//!
//! Implements style verification for code formatting and best practices.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

use crate::verifiers::{CodeIssue, CodeVerifier, IssueSeverity, VerificationResult, VerifierType};

static LONG_LINE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r".{81,}").expect("valid long-line regex"));
static TRAILING_WHITESPACE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[ \t]+$").expect("valid trailing whitespace regex"));
static TAB_CHARACTER_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\t").expect("valid tab character regex"));
static MISSING_DOCUMENTATION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"fn\s+\w+\s*\([^)]*\)\s*\{[^}]*//").expect("valid missing doc regex"));
static CAMEL_CASE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(let|const|var)\s+[a-z][a-zA-Z0-9]*[A-Z]").expect("valid camel case regex")
});
static MAGIC_NUMBER_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(10|100|1000|24|60|3600)\b").expect("valid magic number regex"));
static DEEP_NESTING_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s{16,}").expect("valid deep nesting regex"));
static LARGE_FUNCTION_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"fn\s+\w+\s*\([^)]*\)\s*\{[^}]{500,}").expect("valid large function regex")
});

/// Style pattern
#[derive(Debug, Clone)]
struct StylePattern {
    name: String,
    pattern: &'static Lazy<Regex>,
    severity: IssueSeverity,
    description: String,
    language: String,
}

/// Style verifier
pub struct StyleVerifier {
    style_patterns: Vec<StylePattern>,
}

impl StyleVerifier {
    pub fn new() -> Self {
        Self {
            style_patterns: vec![
                StylePattern {
                    name: "Long Line".to_string(),
                    pattern: &LONG_LINE_REGEX,
                    severity: IssueSeverity::Style,
                    description: "Line exceeds 80 characters".to_string(),
                    language: "all".to_string(),
                },
                StylePattern {
                    name: "Trailing Whitespace".to_string(),
                    pattern: &TRAILING_WHITESPACE_REGEX,
                    severity: IssueSeverity::Style,
                    description: "Trailing whitespace detected".to_string(),
                    language: "all".to_string(),
                },
                StylePattern {
                    name: "Tab Character".to_string(),
                    pattern: &TAB_CHARACTER_REGEX,
                    severity: IssueSeverity::Style,
                    description: "Tab character detected - use spaces".to_string(),
                    language: "all".to_string(),
                },
                StylePattern {
                    name: "Missing Documentation".to_string(),
                    pattern: &MISSING_DOCUMENTATION_REGEX,
                    severity: IssueSeverity::Info,
                    description: "Function missing documentation".to_string(),
                    language: "rust".to_string(),
                },
                StylePattern {
                    name: "Camel Case Variable".to_string(),
                    pattern: &CAMEL_CASE_REGEX,
                    severity: IssueSeverity::Style,
                    description: "Variable should use snake_case".to_string(),
                    language: "rust".to_string(),
                },
                StylePattern {
                    name: "Magic Number".to_string(),
                    pattern: &MAGIC_NUMBER_REGEX,
                    severity: IssueSeverity::Info,
                    description: "Magic number detected - use named constant".to_string(),
                    language: "all".to_string(),
                },
                StylePattern {
                    name: "Deep Nesting".to_string(),
                    pattern: &DEEP_NESTING_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Deep nesting detected - consider refactoring".to_string(),
                    language: "all".to_string(),
                },
                StylePattern {
                    name: "Large Function".to_string(),
                    pattern: &LARGE_FUNCTION_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Large function detected - consider splitting".to_string(),
                    language: "all".to_string(),
                },
            ],
        }
    }
}

impl CodeVerifier for StyleVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult> {
        let mut issues = Vec::new();
        let mut score: f32 = 1.0;

        // Filter out lines starting with "const " for magic number check
        let filtered_code: String = code
            .lines()
            .filter(|l| !l.trim().starts_with("const "))
            .collect::<Vec<_>>()
            .join("\n");

        // Check language-specific style issues
        let language_issues = self.check_language_specific_style(code, language)?;
        issues.extend(language_issues);

        // Check general style patterns
        for pattern in &self.style_patterns {
            if pattern.language == "all" || pattern.language == language {
                let check_code = if pattern.name == "Magic Number" {
                    &filtered_code
                } else {
                    code
                };
                for (line_num, line) in check_code.lines().enumerate() {
                    if pattern.pattern.is_match(line) {
                        issues.push(CodeIssue {
                            severity: pattern.severity.clone(),
                            category: "Style".to_string(),
                            message: format!("{}: {}", pattern.name, pattern.description),
                            line_number: Some(line_num + 1),
                            column_number: None,
                            rule_id: pattern.name.clone(),
                        });

                        // Reduce score based on severity
                        match pattern.severity {
                            IssueSeverity::Critical => score -= 0.15,
                            IssueSeverity::Error | IssueSeverity::High => score -= 0.1,
                            IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.05,
                            IssueSeverity::Info | IssueSeverity::Low => score -= 0.02,
                            IssueSeverity::Style => score -= 0.01,
                        }
                    }
                }
            }
        }

        // Generate style suggestions
        let suggestions = self.generate_style_suggestions(&issues);

        // Calculate style metrics
        let mut metrics = HashMap::new();
        metrics.insert("style_score".to_string(), score.max(0.0));
        metrics.insert("style_issue_count".to_string(), issues.len() as f32);
        metrics.insert("line_count".to_string(), code.lines().count() as f32);
        metrics.insert(
            "max_line_length".to_string(),
            code.lines().map(|l| l.len()).max().unwrap_or(0) as f32,
        );

        Ok(VerificationResult {
            verifier_name: "StyleVerifier".to_string(),
            verifier_type: VerifierType::Style,
            score: score.max(0.0),
            passed: score >= 0.8,
            issues,
            suggestions,
            metrics,
        })
    }

    fn verifier_name(&self) -> &str {
        "StyleVerifier"
    }

    fn verifier_type(&self) -> VerifierType {
        VerifierType::Style
    }

    fn check_language_specific_style(&self, code: &str, language: &str) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();

        match language {
            "rust" => {
                // Rust-specific style checks
                if code.contains("::std::") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Explicit std:: path detected - consider importing".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_explicit_std".to_string(),
                    });
                }

                if code
                    .lines()
                    .any(|l| l.trim().starts_with("pub fn") && !l.contains("///"))
                {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Style".to_string(),
                        message: "Public function missing documentation".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_missing_docs".to_string(),
                    });
                }

                // Check for proper error handling
                if code.contains("unwrap()") && !code.contains("unwrap_or(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Style".to_string(),
                        message: "unwrap() without fallback - consider using unwrap_or()"
                            .to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_unwrap_style".to_string(),
                    });
                }
            }
            "javascript" => {
                // JavaScript-specific style checks
                if code.contains("==") && !code.contains("===") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Use strict equality (===) instead of loose equality (==)"
                            .to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_strict_equality".to_string(),
                    });
                }

                if code.contains("var ") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Use let or const instead of var".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_var_keyword".to_string(),
                    });
                }

                // Check for semicolon usage
                let lines_without_semicolon = code
                    .lines()
                    .filter(|l| {
                        l.trim().ends_with(';') == false
                            && l.trim().ends_with('{') == false
                            && l.trim().ends_with('}') == false
                            && !l.trim().is_empty()
                    })
                    .count();

                if lines_without_semicolon > 0 {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Missing semicolon detected".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_missing_semicolon".to_string(),
                    });
                }
            }
            "python" => {
                // Python-specific style checks
                if code.lines().any(|l| l.len() > 79) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Line exceeds PEP 8 recommendation of 79 characters".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_line_length".to_string(),
                    });
                }

                if code.contains("\t") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Use spaces instead of tabs (PEP 8)".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_tabs".to_string(),
                    });
                }

                // Check for proper imports
                if code.contains("from * import") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Style".to_string(),
                        message: "Wildcard import detected - import specific modules".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_wildcard_import".to_string(),
                    });
                }
            }
            _ => {
                // Generic checks for other languages
            }
        }

        Ok(issues)
    }

    fn generate_style_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();

        if issues.iter().any(|i| i.rule_id.contains("line")) {
            suggestions.push("Break long lines to improve readability".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("whitespace")) {
            suggestions
                .push("Remove trailing whitespace and configure editor to show it".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("tab")) {
            suggestions.push("Configure editor to use spaces instead of tabs".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("documentation")) {
            suggestions.push("Add proper documentation comments to functions".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("camel")) {
            suggestions.push("Use snake_case for variable names in Rust".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("magic")) {
            suggestions.push("Replace magic numbers with named constants".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("nesting")) {
            suggestions.push(
                "Consider refactoring deeply nested code into separate functions".to_string(),
            );
        }

        if issues.iter().any(|i| i.rule_id.contains("function")) {
            suggestions.push("Split large functions into smaller, focused functions".to_string());
        }

        suggestions
    }
}
