use std::collections::HashMap;
use crate::linters::ast_analyzer;
use crate::linters::{CodeIssue, IssueSeverity};

#[derive(Debug, Default)]
pub struct JsFindings {
    pub issues: Vec<CodeIssue>,
    pub metrics: HashMap<String, f32>,
}

fn is_comment_or_empty(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

pub fn analyze_javascript(code: &str) -> JsFindings {
    let mut issues = Vec::new();
    let _clean = ast_analyzer::strip_comments_and_strings(code);

    for (i, raw_line) in code.lines().enumerate() {
        let line_num = i + 1;
        if is_comment_or_empty(raw_line) { continue; }
        let trimmed = raw_line.trim();

        // ── eval() ─────────────────────────────────────────────────
        if trimmed.contains("eval(") && !trimmed.starts_with("//") {
            let arg = extract_call_arg(trimmed, "eval");
            if arg.map_or(true, |a| !is_literal_string(&a)) {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "eval() with non-literal argument — arbitrary code execution risk. Never pass user input to eval().".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "JS-EVAL".to_string(),
                });
            }
        }

        // ── Function() constructor (eval-like) ─────────────────────
        if trimmed.contains("new Function(") || trimmed.contains("new Function(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "Function() constructor detected — eval-like code execution. Use a proper parser instead.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JS-FUNCTION-CONSTRUCTOR".to_string(),
            });
        }

        // ── setTimeout/setInterval with string arg ─────────────────
        if (trimmed.contains("setTimeout(") || trimmed.contains("setInterval("))
            && (trimmed.contains("\"") || trimmed.contains("'"))
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "setTimeout/setInterval with string argument — eval-like behavior. Pass a function reference instead.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JS-TIMER-STRING".to_string(),
            });
        }

        // ── XSS via innerHTML / outerHTML / insertAdjacentHTML ────
        for method in &["innerHTML", "outerHTML", "insertAdjacentHTML"] {
            if trimmed.contains(method) && !trimmed.contains("textContent") {
                let has_concat = trimmed.contains('+') || trimmed.contains("${");
                if has_concat {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Error,
                        category: "Security".to_string(),
                        message: format!("{method} with concatenation — XSS risk. Use textContent or createElement + sanitization."),
                        line_number: Some(line_num),
                        column_number: None,
                        rule_id: "JS-XSS-INNERHTML".to_string(),
                    });
                }
            }
        }

        // ── document.write() ───────────────────────────────────────
        if trimmed.contains("document.write(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Error,
                category: "Security".to_string(),
                message: "document.write() — XSS risk. Use DOM manipulation methods instead.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JS-DOCUMENT-WRITE".to_string(),
            });
        }

        // ── SQL injection (Node.js) ────────────────────────────────
        let has_sql_kw = trimmed.to_uppercase().contains("SELECT ")
            || trimmed.to_uppercase().contains("INSERT ")
            || trimmed.to_uppercase().contains("DELETE ")
            || trimmed.to_uppercase().contains("UPDATE ");
        if has_sql_kw {
            let has_concat = trimmed.contains('+') || trimmed.contains("${") || trimmed.contains("concat(");
            if has_concat {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "Dynamic SQL query with string concatenation — SQL injection risk. Use parameterized queries.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "JS-SQL-INJECTION".to_string(),
                });
            }
        }

        // ── Math.random() for security ─────────────────────────────
        if trimmed.contains("Math.random(")
            && (trimmed.contains("password") || trimmed.contains("token")
                || trimmed.contains("secret") || trimmed.contains("crypto"))
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Error,
                category: "Security".to_string(),
                message: "Math.random() used for security context — not cryptographically secure. Use crypto.getRandomValues() or Web Crypto API.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JS-MATH-RANDOM".to_string(),
            });
        }

        // ── hardcoded secrets ──────────────────────────────────────
        let lower = trimmed.to_lowercase();
        if (lower.contains("password") || lower.contains("secret") || lower.contains("api_key")
            || lower.contains("apikey") || lower.contains("auth_token") || lower.contains("token"))
            && !lower.contains("process.env.") && !lower.contains("env(")
        {
            let has_str = trimmed.contains('"') || trimmed.contains('\'') || trimmed.contains("`");
            if has_str {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "Hardcoded secret detected — use environment variables via process.env.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "JS-HARDCODED-SECRET".to_string(),
                });
            }
        }

        // ── weak crypto ────────────────────────────────────────────
        if trimmed.contains("createHash(") && (trimmed.contains("md5") || trimmed.contains("sha1")) {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "Weak cryptographic hash (MD5/SHA1) — use SHA-256 or stronger.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JS-WEAK-CRYPTO".to_string(),
            });
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("js_issues".to_string(), issues.len() as f32);
    metrics.insert("js_high_severity".to_string(),
        issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical | IssueSeverity::Error)).count() as f32,
    );

    JsFindings { issues, metrics }
}

fn extract_call_arg(s: &str, fn_name: &str) -> Option<String> {
    let idx = s.find(&format!("{fn_name}("))?;
    let after = &s[idx + fn_name.len() + 1..];
    let mut depth = 1;
    let mut arg = String::new();
    for ch in after.chars() {
        if ch == '(' { depth += 1; }
        if ch == ')' { depth -= 1; }
        if depth == 0 { break; }
        arg.push(ch);
    }
    Some(arg.trim().to_string())
}

fn is_literal_string(s: &str) -> bool {
    let t = s.trim();
    (t.starts_with('"') && t.ends_with('"'))
        || (t.starts_with('\'') && t.ends_with('\''))
        || (t.starts_with('`') && t.ends_with('`'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_js_eval() {
        let findings = analyze_javascript(r#"eval(userInput)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JS-EVAL"));
    }

    #[test]
    fn test_js_eval_literal() {
        let findings = analyze_javascript(r#"eval("1+1")"#);
        assert!(!findings.issues.iter().any(|i| i.rule_id == "JS-EVAL"));
    }

    #[test]
    fn test_js_innerhtml_xss() {
        let findings = analyze_javascript(r#"element.innerHTML = "<div>" + userInput + "</div>";"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JS-XSS-INNERHTML"));
    }

    #[test]
    fn test_js_document_write() {
        let findings = analyze_javascript(r#"document.write(userInput)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JS-DOCUMENT-WRITE"));
    }

    #[test]
    fn test_js_sql_injection() {
        let findings = analyze_javascript(r#"const query = "SELECT * FROM users WHERE id = " + userId;"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JS-SQL-INJECTION"));
    }

    #[test]
    fn test_js_hardcoded_secret() {
        let findings = analyze_javascript(r#"const API_KEY = "sk-abc123";"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JS-HARDCODED-SECRET"));
    }

    #[test]
    fn test_js_no_false_positive_env() {
        let findings = analyze_javascript(r#"const API_KEY = process.env.API_KEY;"#);
        assert!(!findings.issues.iter().any(|i| i.rule_id == "JS-HARDCODED-SECRET"));
    }
}
