//! Pattern Detector Manager
//!
//! Main manager untuk mengkoordinasikan multiple pattern detectors (regex-based).
//! Catatan: Detectors ini menggunakan regex + string containment, BUKAN static analysis.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    correctness::CorrectnessLinter, performance::PerformanceLinter, security::SecurityLinter,
    style::StyleLinter,
};

/// Code linter interface
pub trait CodeLinter: Send + Sync {
    fn verify(&self, code: &str, language: &str) -> Result<LintResult>;
    fn linter_name(&self) -> &str;
    fn linter_type(&self) -> LinterType;

    // Optional methods for specific linter types
    fn check_language_specific_security(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<CodeIssue>> {
        if !language.is_empty() {
            tracing::warn!("default security linter ignores language specificity: {language}");
        }
        let mut issues = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let ln = i + 1;
            let lower = line.to_lowercase();

            if line.to_uppercase().contains("SELECT")
                && line.contains('+')
                && line.to_uppercase().contains("FROM")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "sql_injection".into(),
                    message: "SQL injection risk: string concatenation in SQL query. Use parameterized queries.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "SEC-SQL-001".into(),
                });
            }

            if lower.contains("<script")
                || lower.contains("onerror=")
                || lower.contains("onclick=")
                || lower.contains("onload=")
                || lower.contains("onmouseover=")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::High,
                    category: "xss".into(),
                    message: "Cross-site scripting (XSS) vulnerability: inline script or event handler detected.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "SEC-XSS-001".into(),
                });
            }

            if lower.contains("system(")
                || lower.contains("exec(")
                || lower.contains("shell_exec(")
                || lower.contains("os.system")
                || lower.contains("subprocess.call")
                || lower.contains("popen(")
                || lower.contains("process::new")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "command_injection".into(),
                    message: "Command injection risk: dangerous system command execution. Sanitize all inputs.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "SEC-CMD-001".into(),
                });
            }

            if line.contains('=') || line.contains(':') {
                let trimmed = line.trim();
                if !trimmed.starts_with("//")
                    && !trimmed.starts_with('#')
                    && !trimmed.starts_with("/*")
                {
                    if (lower.contains("password")
                        || lower.contains("secret_key")
                        || lower.contains("api_key")
                        || lower.contains("apikey")
                        || lower.contains("auth_token"))
                        && !lower.contains("env(")
                        && !lower.contains("getenv")
                        && !lower.contains("config")
                    {
                        issues.push(CodeIssue {
                            severity: IssueSeverity::High,
                            category: "hardcoded_secret".into(),
                            message: "Hardcoded secret detected. Store in environment variables or a secrets manager.".into(),
                            line_number: Some(ln),
                            column_number: None,
                            rule_id: "SEC-SECRET-001".into(),
                        });
                    }
                }
            }

            if lower.contains("eval(") || (lower.contains("exec(") && !lower.contains("exec_")) {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "unsafe_eval".into(),
                    message: "Unsafe eval() or exec() call — allows arbitrary code execution. Use safe parsers instead.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "SEC-EVAL-001".into(),
                });
            }

            if lower.contains("strcpy(")
                || lower.contains("strcat(")
                || lower.contains("sprintf(")
                || lower.contains("gets(")
                || lower.contains("scanf(")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::High,
                    category: "buffer_overflow".into(),
                    message: "Buffer overflow risk: unsafe C string function. Prefer strncpy/strncat/snprintf/fgets.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "SEC-BUF-001".into(),
                });
            }
        }

        Ok(issues)
    }

    fn generate_security_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        let mut seen_categories = std::collections::HashSet::new();

        for issue in issues {
            if seen_categories.insert(&issue.category) {
                match issue.category.as_str() {
                    "sql_injection" => suggestions.push(
                        "Use parameterized queries or prepared statements instead of string concatenation in SQL queries.".to_string()
                    ),
                    "xss" => suggestions.push(
                        "Sanitize user input with a context-aware encoder. Use Content-Security-Policy headers and avoid innerHTML.".to_string()
                    ),
                    "command_injection" => suggestions.push(
                        "Avoid shell commands with user input. Use library APIs instead of system()/exec(). If necessary, validate against a whitelist.".to_string()
                    ),
                    "hardcoded_secret" => suggestions.push(
                        "Move secrets to environment variables or a secure vault. Never commit secrets to version control.".to_string()
                    ),
                    "unsafe_eval" => suggestions.push(
                        "Replace eval() with a proper parser (e.g., serde_json, nom, pest). Eval is a code injection vector.".to_string()
                    ),
                    "buffer_overflow" => suggestions.push(
                        "Replace unsafe C string functions with bounded alternatives: strncpy, strncat, snprintf, fgets.".to_string()
                    ),
                    _ => {
                        tracing::warn!("unknown security suggestion category: {}", issue.category);
                    }
                }
            }
        }

        if suggestions.is_empty() {
            suggestions.push(
                "Run a dependency vulnerability scanner (e.g., cargo audit, npm audit, pip audit)."
                    .to_string(),
            );
        }

        suggestions
    }

    fn check_language_specific_performance(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<CodeIssue>> {
        if !language.is_empty() {
            tracing::warn!("default performance linter ignores language specificity: {language}");
        }
        let mut issues = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let code_str = code.to_string();

        for (i, line) in lines.iter().enumerate() {
            let ln = i + 1;

            if line.contains("for ")
                && line.contains("for ")
                && i + 1 < lines.len()
                && lines[i + 1].contains("for ")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "nested_loop".into(),
                    message: "O(n²) complexity: nested loop detected. Consider flattening or using a hash map.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "PERF-NEST-001".into(),
                });
            }

            if line.contains(".clone()") && line.contains("for ") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "redundant_clone".into(),
                    message: "Unnecessary .clone() inside loop — causes repeated allocation. Use references instead.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "PERF-CLONE-001".into(),
                });
            }

            if (line.contains("+=") || line.contains("push_str")) && line.contains("for ") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    category: "string_allocation".into(),
                    message: "String concatenation inside loop causes O(n²) allocations. Prefer collecting into a Vec and joining.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "PERF-STR-001".into(),
                });
            }
        }

        if code_str.matches(".to_string()").count() >= 5 {
            let count = code_str.matches(".to_string()").count();
            issues.push(CodeIssue {
                severity: IssueSeverity::Info,
                category: "excessive_to_string".into(),
                message: format!(
                    "{} calls to .to_string() detected. Reuse owned strings or use Cow<str>.",
                    count
                ),
                line_number: None,
                column_number: None,
                rule_id: "PERF-TOSTR-001".into(),
            });
        }

        Ok(issues)
    }

    fn calculate_complexity(&self, code: &str) -> f32 {
        let lines = code.lines().count() as f32;
        if lines < 1.0 {
            tracing::warn!("calculate_complexity: empty code, returning 1.0");
            return 1.0;
        }
        let branching_keywords = [
            " if ", "else if", "for ", "while ", "case ", "catch ", " && ", " || ", "match ",
            "when ",
        ];
        let branch_count: usize = branching_keywords
            .iter()
            .map(|kw| code.matches(kw).count())
            .sum();
        let functions = code.matches("fn ").count()
            + code.matches("def ").count()
            + code.matches("function ").count()
            + code.matches("=>").count()
            + code.matches("-> {").count();
        let nesting = code.matches('{').count().max(code.matches('}').count());
        let raw = 1.0
            + branch_count as f32 * 0.5
            + functions as f32 * 0.3
            + (nesting as f32 / lines.max(1.0));
        raw.min(50.0).max(1.0)
    }

    fn generate_performance_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for issue in issues {
            if seen.insert(&issue.category) {
                match issue.category.as_str() {
                    "nested_loop" => suggestions.push(
                        "Extract nested loops into a flat structure using hash maps or sorting for O(n log n) instead of O(n²).".to_string()
                    ),
                    "redundant_clone" => suggestions.push(
                        "Remove unnecessary .clone() calls inside loops. Use &T references instead of owned T.".to_string()
                    ),
                    "string_allocation" => suggestions.push(
                        "Replace string concatenation in loops with a Vec<String> and .join() for O(n) allocation.".to_string()
                    ),
                    "excessive_to_string" => suggestions.push(
                        "Cache .to_string() results and reuse. Consider using Cow<str> or &str where possible.".to_string()
                    ),
                    _ => {
                        tracing::warn!("unknown performance suggestion category: {}", issue.category);
                    }
                }
            }
        }

        if suggestions.is_empty() {
            suggestions.push("Profile the hot path before optimizing. Premature optimization is the root of all evil.".to_string());
        }

        suggestions
    }

    fn check_language_specific_correctness(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<CodeIssue>> {
        if !language.is_empty() {
            tracing::warn!("default correctness linter ignores language specificity: {language}");
        }
        let mut issues = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let ln = i + 1;

            if line.contains(".unwrap()")
                && !line.trim_start().starts_with("//")
                && !line.trim_start().starts_with('#')
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "null_pointer_risk".into(),
                    message: "Unwrap without safety check — will panic on None/Err. Use pattern matching or ? operator.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "CORR-UNWRAP-001".into(),
                });
            }

            if line.contains(" as ")
                && (line.contains(" as u") || line.contains(" as i") || line.contains(" as f"))
                && !line.contains("as usize")
                && !line.contains("as isize")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "type_confusion".into(),
                    message: "Numeric cast may truncate or overflow. Use try_from() for safe conversions.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "CORR-CAST-001".into(),
                });
            }

            if line.contains("<= ") && lines.len() > i + 1 {
                let next_line = lines[i + 1];
                if next_line.contains('[') || next_line.contains("get(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "off_by_one".into(),
                        message: "Loop with <= bound followed by index access — potential off-by-one error.".into(),
                        line_number: Some(ln),
                        column_number: None,
                        rule_id: "CORR-OFFBYONE-001".into(),
                    });
                }
            }

            if (line.contains("let ") && line.contains(": ") && !line.contains("= "))
                && !line.contains("fn ")
            {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "uninitialized_variable".into(),
                    message:
                        "Variable declared without initialization — may be used uninitialized."
                            .into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "CORR-UNINIT-001".into(),
                });
            }
        }

        Ok(issues)
    }

    fn generate_correctness_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for issue in issues {
            if seen.insert(&issue.category) {
                match issue.category.as_str() {
                    "null_pointer_risk" => suggestions.push(
                        "Replace .unwrap() with match, if let, or the ? operator to handle None/Err gracefully.".to_string()
                    ),
                    "type_confusion" => suggestions.push(
                        "Use TryFrom/Into for numeric conversions that can fail, instead of 'as' casts.".to_string()
                    ),
                    "off_by_one" => suggestions.push(
                        "Use '<' instead of '<=' for zero-indexed collections, or subtract 1 from the bound.".to_string()
                    ),
                    "uninitialized_variable" => suggestions.push(
                        "Initialize variables at declaration with a default value, or use Option.".to_string()
                    ),
                    _ => {
                        tracing::warn!("unknown correctness suggestion category: {}", issue.category);
                    }
                }
            }
        }

        if suggestions.is_empty() {
            suggestions.push(
                "Add property-based tests (e.g., proptest, quickcheck) to catch edge cases."
                    .to_string(),
            );
        }

        suggestions
    }

    fn check_language_specific_style(&self, code: &str, language: &str) -> Result<Vec<CodeIssue>> {
        if !language.is_empty() {
            tracing::warn!("default style linter ignores language specificity: {language}");
        }
        let mut issues = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let mut indent_size: Option<usize> = None;

        for (i, line) in lines.iter().enumerate() {
            let ln = i + 1;

            if line.len() > 100 {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Style,
                    category: "long_line".into(),
                    message: format!(
                        "Line exceeds 100 characters ({} chars). Break into multiple lines.",
                        line.len()
                    ),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "STYLE-LINE-001".into(),
                });
            }

            if line.len() > line.trim_end().len() {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Style,
                    category: "trailing_whitespace".into(),
                    message: "Trailing whitespace detected. Remove trailing spaces.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "STYLE-TRAIL-001".into(),
                });
            }

            if !line.is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                continue;
            }

            let leading_spaces = line.chars().take_while(|c| *c == ' ').count();
            let leading_tabs = line.chars().take_while(|c| *c == '\t').count();

            if leading_spaces > 0 {
                if let Some(expected) = indent_size {
                    if leading_spaces % expected != 0 {
                        issues.push(CodeIssue {
                            severity: IssueSeverity::Style,
                            category: "inconsistent_indentation".into(),
                            message: format!(
                                "Inconsistent indentation: {} spaces, expected multiple of {}.",
                                leading_spaces, expected
                            ),
                            line_number: Some(ln),
                            column_number: None,
                            rule_id: "STYLE-INDENT-001".into(),
                        });
                    }
                } else if leading_spaces > 0 {
                    indent_size = Some(leading_spaces);
                }
            }

            if leading_tabs > 0 && leading_spaces > 0 {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Style,
                    category: "mixed_indentation".into(),
                    message: "Mixed tabs and spaces in indentation. Pick one style.".into(),
                    line_number: Some(ln),
                    column_number: None,
                    rule_id: "STYLE-INDENT-002".into(),
                });
            }
        }

        let has_snake = lines
            .iter()
            .any(|l| l.contains('_') && l.chars().any(|c| c.is_ascii_lowercase()));
        let has_camel = lines
            .iter()
            .any(|l| l.chars().any(|c| c.is_ascii_uppercase()) && !l.contains('_'));

        if has_snake && has_camel {
            issues.push(CodeIssue {
                severity: IssueSeverity::Style,
                category: "naming_inconsistency".into(),
                message:
                    "Mixed snake_case and camelCase naming detected. Adopt a single convention."
                        .into(),
                line_number: None,
                column_number: None,
                rule_id: "STYLE-NAMING-001".into(),
            });
        }

        Ok(issues)
    }

    fn generate_style_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for issue in issues {
            if seen.insert(&issue.category) {
                match issue.category.as_str() {
                    "long_line" => suggestions.push(
                        "Break long lines at 100 characters. Use early returns or extract helper functions.".to_string()
                    ),
                    "trailing_whitespace" => suggestions.push(
                        "Remove trailing whitespace. Configure your editor to strip it on save.".to_string()
                    ),
                    "inconsistent_indentation" | "mixed_indentation" => suggestions.push(
                        "Use a formatter (rustfmt, black, prettier) to enforce consistent indentation.".to_string()
                    ),
                    "naming_inconsistency" => suggestions.push(
                        "Use snake_case for variables/functions, CamelCase for types, SCREAMING_SNAKE for constants.".to_string()
                    ),
                    _ => {
                        tracing::warn!("unknown style suggestion category: {}", issue.category);
                    }
                }
            }
        }

        if suggestions.is_empty() {
            suggestions.push("Run an auto-formatter to enforce a consistent style.".to_string());
        }

        suggestions
    }

    // Performance analysis methods with default implementations
    fn count_clones_efficiently(&self, code: &str) -> usize {
        let Ok(clone_regex) = regex::Regex::new(r"\.clone\(\)") else {
            return 0;
        };
        clone_regex.find_iter(code).count()
    }

    fn has_intermediate_allocation(&self, code: &str) -> bool {
        code.contains(".collect::<Vec<_>>())") && code.contains(".iter()")
    }

    fn has_inefficient_string_ops(&self, code: &str) -> bool {
        code.contains("String::new()") && code.contains(".push_str(")
    }

    fn has_inefficient_python_lists(&self, code: &str) -> bool {
        code.contains("for i in range(len(") || code.contains("list.append(")
    }

    fn has_dom_query_in_loop(&self, code: &str) -> bool {
        let dom_methods = [
            "getElementById",
            "querySelector",
            "querySelectorAll",
            "getElementsByClassName",
        ];
        dom_methods.iter().any(|method| code.contains(method)) && code.contains("for")
    }

    fn has_potential_memory_leaks(&self, code: &str) -> bool {
        code.contains("malloc") && !code.contains("free")
    }
}

/// Types of linters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum LinterType {
    Security,
    Performance,
    Correctness,
    Style,
}

/// Lint result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintResult {
    pub linter_name: String,
    pub linter_type: LinterType,
    pub score: f32,
    pub passed: bool,
    pub issues: Vec<CodeIssue>,
    pub suggestions: Vec<String>,
    pub metrics: HashMap<String, f32>,
}

/// Code issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeIssue {
    pub severity: IssueSeverity,
    pub category: String,
    pub message: String,
    pub line_number: Option<usize>,
    pub column_number: Option<usize>,
    pub rule_id: String,
}

pub use nexora_core::types::IssueSeverity;

/// Main code linter manager
pub struct CodeLinterManager {
    linters: Vec<Box<dyn CodeLinter>>,
}

impl CodeLinterManager {
    pub fn new() -> Self {
        Self {
            linters: vec![
                Box::new(SecurityLinter::new()),
                Box::new(PerformanceLinter::new()),
                Box::new(CorrectnessLinter::new()),
                Box::new(StyleLinter::new()),
            ],
        }
    }

    pub fn add_linter(&mut self, linter: Box<dyn CodeLinter>) {
        self.linters.push(linter);
    }

    pub fn verify_code(&self, code: &str, language: &str) -> Result<f32> {
        let mut total_score = 0.0;
        let mut all_issues = Vec::new();

        for linter in &self.linters {
            let result = linter.verify(code, language)?;
            total_score += result.score;
            all_issues.extend(result.issues);
        }

        let avg_score = total_score / self.linters.len() as f32;

        for issue in &all_issues {
            tracing::info!(
                "Issue: {} - {} (Line: {:?})",
                issue.category, issue.message, issue.line_number
            );
        }

        Ok(avg_score)
    }

    pub fn verify_detailed(&self, code: &str, language: &str) -> Result<Vec<LintResult>> {
        let mut results = Vec::new();

        for linter in &self.linters {
            let result = linter.verify(code, language)?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn get_security_score(&self, code: &str, language: &str) -> Result<f32> {
        for linter in &self.linters {
            if linter.linter_type() == LinterType::Security {
                let result = linter.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }

    pub fn get_performance_score(&self, code: &str, language: &str) -> Result<f32> {
        for linter in &self.linters {
            if linter.linter_type() == LinterType::Performance {
                let result = linter.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }

    pub fn get_correctness_score(&self, code: &str, language: &str) -> Result<f32> {
        for linter in &self.linters {
            if linter.linter_type() == LinterType::Correctness {
                let result = linter.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }

    pub fn get_style_score(&self, code: &str, language: &str) -> Result<f32> {
        for linter in &self.linters {
            if linter.linter_type() == LinterType::Style {
                let result = linter.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }

    pub fn get_linter_names(&self) -> Vec<String> {
        self.linters
            .iter()
            .map(|v| v.linter_name().to_string())
            .collect()
    }

    pub fn get_issues_by_severity(
        &self,
        code: &str,
        language: &str,
        severity: IssueSeverity,
    ) -> Result<Vec<CodeIssue>> {
        let mut filtered_issues = Vec::new();

        for linter in &self.linters {
            let result = linter.verify(code, language)?;
            filtered_issues.extend(
                result
                    .issues
                    .into_iter()
                    .filter(|issue| issue.severity == severity),
            );
        }

        Ok(filtered_issues)
    }

    pub fn get_summary_report(&self, code: &str, language: &str) -> Result<LintSummary> {
        let results = self.verify_detailed(code, language)?;

        let mut total_issues = 0;
        let mut error_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;
        let mut style_count = 0;

        for result in &results {
            total_issues += result.issues.len();

            for issue in &result.issues {
                match issue.severity {
                    IssueSeverity::Critical | IssueSeverity::Error => error_count += 1,
                    IssueSeverity::High | IssueSeverity::Warning => warning_count += 1,
                    IssueSeverity::Medium | IssueSeverity::Info => info_count += 1,
                    IssueSeverity::Low | IssueSeverity::Style => style_count += 1,
                }
            }
        }

        let overall_score = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;

        Ok(LintSummary {
            overall_score,
            total_issues,
            error_count,
            warning_count,
            info_count,
            style_count,
            linter_results: results,
        })
    }
}

/// Lint summary report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintSummary {
    pub overall_score: f32,
    pub total_issues: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub style_count: usize,
    pub linter_results: Vec<LintResult>,
}
