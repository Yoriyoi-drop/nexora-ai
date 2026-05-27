//! Security Pattern Detector
//!
//! Regex-based pattern matcher untuk security vulnerabilities.
//! BUKAN static analysis — menggunakan regex + string containment.
//! Rentan false positive/negative.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

use crate::linters::{CodeIssue, CodeLinter, IssueSeverity, LintResult, LinterType};

static SQL_INJECTION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)execute\s*\(").expect("valid SQL injection regex"));
static COMMAND_INJECTION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"exec\s*\(").expect("valid command injection regex"));
static HARDCODED_PASSWORD_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)password\s*=").expect("valid password assignment regex"));
static BUFFER_OVERFLOW_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(strcpy|strcat|gets|sprintf)\s*\(").expect("valid unsafe string op regex")
});
static XSS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(innerHTML|document\.write|eval\s*\()\s*\+").expect("valid XSS regex")
});
static PATH_TRAVERSAL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\.\.\/|\.\.\\|\/etc\/passwd|\/etc\/shadow)")
        .expect("valid path traversal regex")
});
static INSECURE_RANDOM_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(rand|random|Math\.random)\s*\(").expect("valid insecure random regex")
});
static WEAK_CRYPTO_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(md5|sha1|des|rc4)\s*\(").expect("valid weak crypto regex"));

/// Security vulnerability pattern
#[derive(Debug, Clone)]
struct VulnerabilityPattern {
    name: String,
    pattern: &'static Lazy<Regex>,
    severity: IssueSeverity,
    description: String,
    language: String,
}

/// Security verifier
pub struct SecurityLinter {
    vulnerability_patterns: Vec<VulnerabilityPattern>,
}

impl SecurityLinter {
    pub fn new() -> Self {
        Self {
            vulnerability_patterns: Vec::new(),
        }
    }
}

impl CodeLinter for SecurityLinter {
    fn verify(&self, code: &str, language: &str) -> Result<LintResult> {
        let mut issues = Vec::new();
        let mut score: f32 = 1.0;

        // Check language-specific security issues
        let language_issues = self.check_language_specific_security(code, language)?;
        issues.extend(language_issues);

        // Check general vulnerability patterns
        for pattern in &self.vulnerability_patterns {
            if pattern.language == "all" || pattern.language == language {
                for (line_num, line) in code.lines().enumerate() {
                    if pattern.pattern.is_match(line) {
                        issues.push(CodeIssue {
                            severity: pattern.severity.clone(),
                            category: "Security".to_string(),
                            message: format!("{}: {}", pattern.name, pattern.description),
                            line_number: Some(line_num + 1),
                            column_number: None,
                            rule_id: pattern.name.clone(),
                        });

                        // Reduce score based on severity
                        match pattern.severity {
                            IssueSeverity::Critical => score -= 0.3,
                            IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                            IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                            IssueSeverity::Info | IssueSeverity::Low => score -= 0.05,
                            IssueSeverity::Style => score -= 0.02,
                        }
                    }
                }
            }
        }

        // Generate security suggestions
        let suggestions = self.generate_security_suggestions(&issues);

        // Calculate security metrics
        let mut metrics = HashMap::new();
        metrics.insert("security_score".to_string(), score.max(0.0));
        metrics.insert("vulnerability_count".to_string(), issues.len() as f32);
        metrics.insert(
            "high_severity_count".to_string(),
            issues
                .iter()
                .filter(|i| matches!(i.severity, IssueSeverity::Error))
                .count() as f32,
        );

        Ok(LintResult {
            linter_name: "SecurityLinter".to_string(),
            linter_type: LinterType::Security,
            score: score.max(0.0),
            passed: score >= 0.7,
            issues,
            suggestions,
            metrics,
        })
    }

    fn linter_name(&self) -> &str {
        "SecurityLinter"
    }

    fn linter_type(&self) -> LinterType {
        LinterType::Security
    }

    fn check_language_specific_security(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();

        match language {
            "rust" => {
                // Rust-specific security checks
                if code.contains("unsafe") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: "Unsafe code detected - review for security implications"
                            .to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_unsafe".to_string(),
                    });
                }

                if code.contains("transmute") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: "Transmute detected - potential type confusion".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_transmute".to_string(),
                    });
                }
            }
            "javascript" => {
                // JavaScript-specific security checks
                if code.contains("eval(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Error,
                        category: "Security".to_string(),
                        message: "eval() detected - code injection risk".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_eval".to_string(),
                    });
                }

                if code.contains("innerHTML") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: "innerHTML detected - XSS risk".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_innerhtml".to_string(),
                    });
                }
            }
            "python" => {
                // Python-specific security checks
                if code.contains("exec(") || code.contains("eval(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Error,
                        category: "Security".to_string(),
                        message: "exec() or eval() detected - code injection risk".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_exec_eval".to_string(),
                    });
                }

                if code.contains("pickle.loads(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: "pickle.loads() detected - potential code execution".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_pickle".to_string(),
                    });
                }
            }
            _ => {
                tracing::warn!("Unknown language in security verifier: {language}");
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    category: "Security".to_string(),
                    message: format!("Language '{language}' is not supported by the security verifier"),
                    line_number: None,
                    column_number: None,
                    rule_id: "lang_unsupported".to_string(),
                });
            }
        }

        Ok(issues)
    }

    fn generate_security_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();

        if issues.iter().any(|i| i.rule_id.contains("sql")) {
            suggestions.push(
                "Use parameterized queries or prepared statements to prevent SQL injection"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| i.rule_id.contains("xss")) {
            suggestions
                .push("Sanitize user input and use proper escaping to prevent XSS".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("password")) {
            suggestions.push("Store passwords securely using proper hashing and salt".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("buffer")) {
            suggestions.push("Use safer string operations like strlcpy or std::string".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("eval")) {
            suggestions
                .push("Avoid using eval() with user input - use safer alternatives".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("crypto")) {
            suggestions
                .push("Use stronger cryptographic algorithms like SHA-256 or AES".to_string());
        }

        suggestions
    }
}
