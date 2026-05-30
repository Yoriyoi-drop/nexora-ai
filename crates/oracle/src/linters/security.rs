//! Security Pattern Detector
//!
//! Hybrid security linter: regex + AST-based static analysis untuk Rust code.
//! - Regex patterns: cross-language vulnerability detection (SQL injection, XSS, dll)
//! - AST analysis (syn): Rust-specific structural checks (unsafe classification,
//!   Command injection, transmute, recursive types, format! SQL injection)
//! - Comment/string-aware: skip matches in comments and string literals
//! - Multi-line detection: cross-line pattern matching untuk SQL injection

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

use crate::linters::ast_analyzer;
use crate::linters::dep_scanner;
use crate::linters::go_analyzer;
use crate::linters::java_analyzer;
use crate::linters::javascript_analyzer;
use crate::linters::python_analyzer;
use crate::linters::{CodeIssue, CodeLinter, IssueSeverity, LintResult, LinterType};

/// Detect programming language from code content using structural heuristics.
/// Returns a language identifier string: "rust", "python", "javascript", "java",
/// "go", "typescript", or "text" if unknown.
pub fn detect_language(code: &str) -> &'static str {
    let first_lines: String = code.lines().take(20).collect::<Vec<_>>().join("\n");

    // Rust: fn main, let mut, impl, use std, ->
    if first_lines.contains("fn main") || first_lines.contains("use std::")
        || first_lines.contains("let mut ") || first_lines.contains("impl ")
        || first_lines.contains("pub fn") || first_lines.contains("unsafe {")
        || code.contains("-> ") && code.contains("struct ") && code.contains("impl ")
    {
        return "rust";
    }

    // Python: def, import, from x import, # comment, no braces
    if first_lines.contains("def ") || first_lines.contains("import ")
        || first_lines.contains("from ") || first_lines.contains("class ")
        || first_lines.contains("if __name__") || first_lines.contains("__name__")
        || (first_lines.contains("print(") && !first_lines.contains('{')
            && !first_lines.contains("System.out"))
    {
        return "python";
    }

    // TypeScript: : type annotations, interface, type keyword
    if first_lines.contains(": string") || first_lines.contains(": number")
        || first_lines.contains(": boolean") || first_lines.contains("interface ")
        || first_lines.contains("as string") || first_lines.contains("as number")
        || first_lines.contains("export interface") || first_lines.contains("export type")
    {
        return "typescript";
    }

    // JavaScript: function, const/let/var, =>, require, console.log
    if first_lines.contains("function ") || first_lines.contains("const ")
        || first_lines.contains("=>") || first_lines.contains("require(")
        || first_lines.contains("console.log") || first_lines.contains("document.")
        || first_lines.contains("module.exports") || first_lines.contains("import React")
        || first_lines.contains("export default") || first_lines.contains("useState")
    {
        return "javascript";
    }

    // Java: public class, public static void main, extends, implements, @Override
    if first_lines.contains("public class") || first_lines.contains("public static void main")
        || first_lines.contains("private static final") || first_lines.contains("extends ")
        || first_lines.contains("implements ") || first_lines.contains("@Override")
        || first_lines.contains("import java.") || first_lines.contains("import javax.")
        || first_lines.contains("System.out.println") || first_lines.contains("void main")
        || first_lines.contains("public final class") || first_lines.contains("abstract class")
    {
        return "java";
    }

    // Go: package main, func main, import (, defer, :=
    if first_lines.contains("package main") || first_lines.contains("func main")
        || first_lines.contains("import (") || first_lines.contains("fmt.")
        || first_lines.contains("defer ") || first_lines.contains(":= ")
        || first_lines.contains("http.HandleFunc") || first_lines.contains("http.ListenAndServe")
        || first_lines.contains("goroutine") || first_lines.contains("go func")
    {
        return "go";
    }

    "text"
}

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
            vulnerability_patterns: vec![
                VulnerabilityPattern {
                    name: "SQL Injection".to_string(),
                    pattern: &SQL_INJECTION_REGEX,
                    severity: IssueSeverity::Critical,
                    description: "Dynamic SQL execution detected - use parameterized queries".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "Command Injection".to_string(),
                    pattern: &COMMAND_INJECTION_REGEX,
                    severity: IssueSeverity::Critical,
                    description: "Shell execution detected - validate and sanitize input".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "Hardcoded Password".to_string(),
                    pattern: &HARDCODED_PASSWORD_REGEX,
                    severity: IssueSeverity::Error,
                    description: "Hardcoded credential detected - use environment variables".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "Buffer Overflow".to_string(),
                    pattern: &BUFFER_OVERFLOW_REGEX,
                    severity: IssueSeverity::Error,
                    description: "Unsafe string copy function detected - use safe alternatives".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "XSS".to_string(),
                    pattern: &XSS_REGEX,
                    severity: IssueSeverity::Error,
                    description: "Potential XSS via unsafe DOM manipulation".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "Path Traversal".to_string(),
                    pattern: &PATH_TRAVERSAL_REGEX,
                    severity: IssueSeverity::Error,
                    description: "Path traversal pattern detected - sanitize file paths".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "Insecure Random".to_string(),
                    pattern: &INSECURE_RANDOM_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Weak random number generator - use cryptographically secure RNG".to_string(),
                    language: "all".to_string(),
                },
                VulnerabilityPattern {
                    name: "Weak Crypto".to_string(),
                    pattern: &WEAK_CRYPTO_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Weak cryptographic algorithm detected - use stronger alternatives".to_string(),
                    language: "all".to_string(),
                },
            ],
        }
    }
}

impl CodeLinter for SecurityLinter {
    fn verify(&self, code: &str, language: &str) -> Result<LintResult> {
        let mut issues = Vec::new();
        let mut score: f32 = 1.0;
        let mut metrics = HashMap::new();

        // ── AST / structural analysis per language ──────────────────
        match language {
            "rust" => {
                let ast_findings = ast_analyzer::analyze_rust_ast(code);
                for issue in &ast_findings.issues {
                    issues.push(issue.clone());
                    match issue.severity {
                        IssueSeverity::Critical => score -= 0.3,
                        IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                        IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                        IssueSeverity::Info | IssueSeverity::Low => score -= 0.05,
                        IssueSeverity::Style => score -= 0.02,
                    }
                }
                metrics.extend(ast_findings.metrics);
            }
            "python" => {
                let py_findings = python_analyzer::analyze_python(code);
                for issue in &py_findings.issues {
                    issues.push(issue.clone());
                    match issue.severity {
                        IssueSeverity::Critical => score -= 0.3,
                        IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                        IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                        _ => score -= 0.05,
                    }
                }
                metrics.extend(py_findings.metrics);
            }
            "javascript" | "typescript" => {
                let js_findings = javascript_analyzer::analyze_javascript(code);
                for issue in &js_findings.issues {
                    issues.push(issue.clone());
                    match issue.severity {
                        IssueSeverity::Critical => score -= 0.3,
                        IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                        IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                        _ => score -= 0.05,
                    }
                }
                metrics.extend(js_findings.metrics);
            }
            "java" => {
                let java_findings = java_analyzer::analyze_java(code);
                for issue in &java_findings.issues {
                    issues.push(issue.clone());
                    match issue.severity {
                        IssueSeverity::Critical => score -= 0.3,
                        IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                        IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                        _ => score -= 0.05,
                    }
                }
                metrics.extend(java_findings.metrics);
            }
            "go" => {
                let go_findings = go_analyzer::analyze_go(code);
                for issue in &go_findings.issues {
                    issues.push(issue.clone());
                    match issue.severity {
                        IssueSeverity::Critical => score -= 0.3,
                        IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                        IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                        _ => score -= 0.05,
                    }
                }
                metrics.extend(go_findings.metrics);
            }
            _ => {}
        }

        // ── Dependency vulnerability scanning ──────────────────────
        // Scan for known vulnerable dependency patterns across all code
        let dep_findings = dep_scanner::scan_dependencies(code);
        for issue in &dep_findings.issues {
            issues.push(issue.clone());
            match issue.severity {
                IssueSeverity::Critical => score -= 0.3,
                IssueSeverity::Error | IssueSeverity::High => score -= 0.2,
                IssueSeverity::Warning | IssueSeverity::Medium => score -= 0.1,
                _ => score -= 0.05,
            }
        }
        metrics.extend(dep_findings.metrics);

        // ── Comment/string-aware regex matching ─────────────────────
        // Strip comments and strings to avoid false positives
        let clean_code = ast_analyzer::strip_comments_and_strings(code);

        // Check language-specific security issues (on original code)
        let language_issues = self.check_language_specific_security(code, language)?;
        issues.extend(language_issues);

        // Check general vulnerability patterns (on clean code)
        for pattern in &self.vulnerability_patterns {
            if pattern.language == "all" || pattern.language == language {
                for (line_num, line) in clean_code.lines().enumerate() {
                    if pattern.pattern.is_match(line) {
                        issues.push(CodeIssue {
                            severity: pattern.severity.clone(),
                            category: "Security".to_string(),
                            message: format!("{}: {}", pattern.name, pattern.description),
                            line_number: Some(line_num + 1),
                            column_number: None,
                            rule_id: pattern.name.clone(),
                        });

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

        // ── Multi-line SQL injection detection ──────────────────────
        // Detects SQL query built across multiple lines via concatenation
        let lines: Vec<&str> = code.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            let has_sql_keyword = line.to_uppercase().contains("SELECT ")
                || line.to_uppercase().contains("FROM ")
                || line.to_uppercase().contains("WHERE ")
                || line.to_uppercase().contains("INSERT ")
                || line.to_uppercase().contains("DELETE ");
            if has_sql_keyword {
                let mut j = i;
                let mut combined = String::new();
                while j < lines.len() && j < i + 5 {
                    combined.push_str(lines[j]);
                    combined.push(' ');
                    if lines[j].contains(';') || lines[j].contains('"') && !lines[j].contains('\\') {
                        if combined.contains("SELECT") || combined.contains("INSERT") {
                            if combined.contains('+') || combined.contains("format!(") {
                                let clean_combined = ast_analyzer::strip_comments_and_strings(&combined);
                                if clean_combined.contains('+') || clean_combined.contains("format!(") {
                                    let is_new = !issues.iter().any(|x| x.rule_id == "ML-SQL-INJECTION"
                                        && x.line_number == Some(i + 1));
                                    if is_new {
                                        issues.push(CodeIssue {
                                            severity: IssueSeverity::Critical,
                                            category: "Security".to_string(),
                                            message: "SQL injection risk: concatenation in SQL query spanning multiple lines. Use parameterized queries.".to_string(),
                                            line_number: Some(i + 1),
                                            column_number: None,
                                            rule_id: "ML-SQL-INJECTION".to_string(),
                                        });
                                        score -= 0.3;
                                    }
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    j += 1;
                }
            }
            i += 1;
        }

        // Generate security suggestions
        let suggestions = self.generate_security_suggestions(&issues);

        metrics.insert("security_score".to_string(), score.max(0.0));
        metrics.insert("vulnerability_count".to_string(), issues.len() as f32);
        metrics.insert(
            "high_severity_count".to_string(),
            issues
                .iter()
                .filter(|i| matches!(i.severity, IssueSeverity::Error | IssueSeverity::Critical))
                .count() as f32,
        );
        metrics.insert("ast_analysis_active".to_string(), 1.0);
        let lang_detected = detect_language(code);
        metrics.insert("detected_language".to_string(), match lang_detected {
            "rust" => 1.0, "python" => 2.0, "javascript" | "typescript" => 3.0,
            "java" => 4.0, "go" => 5.0, _ => 0.0,
        });

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
                    message: format!(
                        "Language '{language}' is not supported by the security verifier"
                    ),
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

        if issues.iter().any(|i| i.rule_id.contains("AST-CMD-INJECTION")) {
            suggestions.push("Avoid std::process::Command with user input. Use a whitelist or parse input strictly".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("AST-TRANSMUTE")) {
            suggestions.push("Replace transmute with safe conversions (TryFrom, Into, bytemuck::cast) where possible".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("AST-UNSAFE-HIGH")) {
            suggestions.push("Minimize unsafe blocks — extract unsafe operations into small, audited functions with safe wrappers".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("AST-SQL-FORMAT")) {
            suggestions.push("Do not build SQL queries with format!() — use sqlx or diesel's query builder for compile-time checked queries".to_string());
        }

        if issues.iter().any(|i| i.rule_id.contains("ML-")) {
            suggestions.push("SQL concatenation across lines is hard to review — use parameterized queries instead".to_string());
        }

        suggestions
    }
}
