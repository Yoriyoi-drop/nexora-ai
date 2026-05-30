use std::collections::HashMap;
use crate::linters::ast_analyzer;
use crate::linters::{CodeIssue, IssueSeverity};

#[derive(Debug, Default)]
pub struct JavaFindings {
    pub issues: Vec<CodeIssue>,
    pub metrics: HashMap<String, f32>,
}

fn is_comment_or_empty(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

pub fn analyze_java(code: &str) -> JavaFindings {
    let mut issues = Vec::new();
    let _clean = ast_analyzer::strip_comments_and_strings(code);

    for (i, raw_line) in code.lines().enumerate() {
        let line_num = i + 1;
        if is_comment_or_empty(raw_line) { continue; }
        let trimmed = raw_line.trim();

        // ── Runtime.exec() / ProcessBuilder command injection ──────
        if trimmed.contains("Runtime.getRuntime().exec(") || trimmed.contains("Runtime.getRuntime().exec(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "Runtime.exec() detected — command injection risk. Use ProcessBuilder with argument list instead of shell string.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JAVA-CMD-INJECTION".to_string(),
            });
        }
        if trimmed.contains("new ProcessBuilder(") && trimmed.contains(".start()") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "ProcessBuilder with potential shell injection — validate all input arguments.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JAVA-PROCESSBUILDER".to_string(),
            });
        }

        // ── SQL injection (string concatenation) ───────────────────
        let has_sql_kw = trimmed.to_uppercase().contains("SELECT ")
            || trimmed.to_uppercase().contains("INSERT ")
            || trimmed.to_uppercase().contains("DELETE ")
            || trimmed.to_uppercase().contains("UPDATE ")
            || trimmed.to_uppercase().contains("DROP ");
        if has_sql_kw {
            let has_concat = trimmed.contains(" + ") || trimmed.contains("concat(") || trimmed.contains("String.format(");
            if has_concat {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "Dynamic SQL with string concatenation — SQL injection risk. Use PreparedStatement with ? placeholders.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "JAVA-SQL-INJECTION".to_string(),
                });
            }
        }

        // ── Deserialization (ObjectInputStream) ────────────────────
        if trimmed.contains("new ObjectInputStream(") || trimmed.contains("readObject()") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "Java deserialization via ObjectInputStream — remote code execution risk. Use a safe serialization format (JSON, protobuf).".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JAVA-DESERIALIZATION".to_string(),
            });
        }

        // ── XXE (XML parsing without protection) ──────────────────
        if trimmed.contains("DocumentBuilderFactory.newInstance()")
            || trimmed.contains("SAXParserFactory.newInstance()")
            || trimmed.contains("XMLInputFactory.newInstance()")
        {
            let next: String = code.lines().skip(i).take(5).collect::<Vec<_>>().join(" ");
            let has_xxe_protection = next.contains("setFeature(")
                && (next.contains("XMLConstants.FEATURE_SECURE_PROCESSING")
                    || next.contains("DISALLOW_DOCTYPE_DECL")
                    || next.contains("ACCESS_EXTERNAL_DTD")
                    || next.contains("ACCESS_EXTERNAL_STYLESHEET"));
            if !has_xxe_protection {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "XML parser without XXE protection — XML External Entity injection risk. Set FEATURE_SECURE_PROCESSING and disallow DOCTYPE.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "JAVA-XXE".to_string(),
                });
            }
        }

        // ── Weak cryptography ─────────────────────────────────────
        if trimmed.contains("Cipher.getInstance(")
            && (trimmed.contains("DES") || trimmed.contains("RC4") || trimmed.contains("Blowfish"))
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Error,
                category: "Security".to_string(),
                message: "Weak cipher algorithm (DES/RC4/Blowfish) — use AES/GCM or ChaCha20.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JAVA-WEAK-CRYPTO".to_string(),
            });
        }

        // ── Hardcoded secrets ──────────────────────────────────────
        let lower = trimmed.to_lowercase();
        if (lower.contains("password") || lower.contains("secret") || lower.contains("api_key")
            || lower.contains("apikey") || lower.contains("jwt_secret") || lower.contains("token"))
            && !lower.contains("system.getenv") && !lower.contains("env(")
        {
            let has_str = trimmed.contains('"') || trimmed.contains('\'');
            if has_str {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Error,
                    category: "Security".to_string(),
                    message: "Hardcoded secret detected — use environment variables or a secrets manager (Vault, AWS Secrets Manager).".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "JAVA-HARDCODED-SECRET".to_string(),
                });
            }
        }

        // ── Log injection / log forging ────────────────────────────
        if (trimmed.contains("logger.") || trimmed.contains("log.") || trimmed.contains("LOG."))
            && (trimmed.contains("+ ") || trimmed.contains(".format("))
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Security".to_string(),
                message: "Log entry with string concatenation — log injection risk. Use parameterized logging (SLF4J {} placeholders).".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "JAVA-LOG-INJECTION".to_string(),
            });
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("java_issues".to_string(), issues.len() as f32);
    metrics.insert("java_high_severity".to_string(),
        issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical | IssueSeverity::Error)).count() as f32,
    );

    JavaFindings { issues, metrics }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_java_cmd_injection() {
        let findings = analyze_java(r#"Runtime.getRuntime().exec("rm -rf /");"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JAVA-CMD-INJECTION"));
    }

    #[test]
    fn test_java_sql_injection() {
        let findings = analyze_java(r#"String q = "SELECT * FROM users WHERE id = " + userId;"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JAVA-SQL-INJECTION"));
    }

    #[test]
    fn test_java_deserialization() {
        let findings = analyze_java(r#"ObjectInputStream ois = new ObjectInputStream(input);"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JAVA-DESERIALIZATION"));
    }

    #[test]
    fn test_java_xxe() {
        let findings = analyze_java(r#"DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JAVA-XXE"));
    }

    #[test]
    fn test_java_weak_crypto() {
        let findings = analyze_java(r#"Cipher cipher = Cipher.getInstance("DES/ECB/PKCS5Padding");"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JAVA-WEAK-CRYPTO"));
    }

    #[test]
    fn test_java_hardcoded_secret() {
        let findings = analyze_java(r#"String jwtSecret = "my-super-secret-key";"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "JAVA-HARDCODED-SECRET"));
    }
}
