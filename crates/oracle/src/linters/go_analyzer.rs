use std::collections::HashMap;
use crate::linters::ast_analyzer;
use crate::linters::{CodeIssue, IssueSeverity};

#[derive(Debug, Default)]
pub struct GoFindings {
    pub issues: Vec<CodeIssue>,
    pub metrics: HashMap<String, f32>,
}

fn is_comment_or_empty(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

pub fn analyze_go(code: &str) -> GoFindings {
    let mut issues = Vec::new();
    let _clean = ast_analyzer::strip_comments_and_strings(code);

    for (i, raw_line) in code.lines().enumerate() {
        let line_num = i + 1;
        if is_comment_or_empty(raw_line) { continue; }
        let trimmed = raw_line.trim();

        // ── os/exec command injection ──────────────────────────────
        if trimmed.contains("exec.Command(") {
            let next: String = code.lines().skip(i).take(3).collect::<Vec<_>>().join(" ");
            if !next.contains(".CombinedOutput()") && !next.contains(".Output()") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "exec.Command() detected — command injection risk. Validate all input arguments; avoid passing user input directly.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "GO-CMD-INJECTION".to_string(),
                });
            }
        }

        // ── SQL injection (string concatenation) ───────────────────
        let has_sql_kw = trimmed.to_uppercase().contains("SELECT ")
            || trimmed.to_uppercase().contains("INSERT ")
            || trimmed.to_uppercase().contains("DELETE ")
            || trimmed.to_uppercase().contains("UPDATE ")
            || trimmed.to_uppercase().contains("DROP ");
        if has_sql_kw {
            let has_concat = trimmed.contains(" + ") || trimmed.contains("fmt.Sprintf(")
                || trimmed.contains("fmt.Sprintf(") || trimmed.contains("strings.Join(");
            if has_concat {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "Dynamic SQL with string formatting — SQL injection risk. Use parameterized queries with ? or $1 placeholders.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "GO-SQL-INJECTION".to_string(),
                });
            }
        }

        // ── HTTP without TLS ───────────────────────────────────────
        if trimmed.contains("http.ListenAndServe(") && !trimmed.contains("http.ListenAndServeTLS(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "HTTP server without TLS — data transmitted in plaintext. Use http.ListenAndServeTLS() with valid certificates.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "GO-NO-TLS".to_string(),
            });
        }

        // ── Weak crypto imports ────────────────────────────────────
        if trimmed.contains("\"crypto/md5\"") || trimmed.contains("\"crypto/sha1\"") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "Weak cryptographic hash import (crypto/md5 or crypto/sha1) — use crypto/sha256 or crypto/sha512.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "GO-WEAK-CRYPTO".to_string(),
            });
        }

        // ── math/rand used for security ────────────────────────────
        if trimmed.contains("math/rand") || trimmed.contains("rand.Int") || trimmed.contains("rand.Float64") {
            let context_lines: String = code.lines().skip(i.saturating_sub(2)).take(5).collect::<Vec<_>>().join(" ");
            let is_security_context = context_lines.contains("password") || context_lines.contains("token")
                || context_lines.contains("secret") || context_lines.contains("key");
            if is_security_context {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "math/rand used for security-sensitive context — not cryptographically secure. Use crypto/rand instead.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "GO-INSECURE-RAND".to_string(),
                });
            }
        }

        // ── Hardcoded secrets ──────────────────────────────────────
        let lower = trimmed.to_lowercase();
        if (lower.contains("password") || lower.contains("secret") || lower.contains("api_key")
            || lower.contains("apikey") || lower.contains("auth_token") || lower.contains("token"))
            && !lower.contains("os.getenv") && !lower.contains("os.Environ")
            && !lower.contains("viper.Get")
        {
            let has_str = trimmed.contains('"') || trimmed.contains('`');
            if has_str {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "Hardcoded secret detected — use environment variables or a secrets manager.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "GO-HARDCODED-SECRET".to_string(),
                });
            }
        }

        // ── Unsafe pointer arithmetic ──────────────────────────────
        if trimmed.contains("unsafe.Pointer(") || trimmed.contains("unsafe.Pointer(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "unsafe.Pointer detected — memory safety risk. Minimize unsafe pointer usage.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "GO-UNSAFE-POINTER".to_string(),
            });
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("go_issues".to_string(), issues.len() as f32);
    metrics.insert("go_high_severity".to_string(),
        issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical | IssueSeverity::Error)).count() as f32,
    );

    GoFindings { issues, metrics }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_cmd_injection() {
        let findings = analyze_go(r#"cmd := exec.Command("bash", "-c", userInput)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "GO-CMD-INJECTION"));
    }

    #[test]
    fn test_go_sql_injection() {
        let findings = analyze_go(r#"q := fmt.Sprintf("SELECT * FROM users WHERE id = %s", userId)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "GO-SQL-INJECTION"));
    }

    #[test]
    fn test_go_no_tls() {
        let findings = analyze_go(r#"log.Fatal(http.ListenAndServe(":8080", router))"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "GO-NO-TLS"));
    }

    #[test]
    fn test_go_weak_crypto() {
        let findings = analyze_go(r#"import "crypto/md5""#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "GO-WEAK-CRYPTO"));
    }

    #[test]
    fn test_go_hardcoded_secret() {
        let findings = analyze_go(r#"apiKey := "sk-abc123xyz""#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "GO-HARDCODED-SECRET"));
    }
}
