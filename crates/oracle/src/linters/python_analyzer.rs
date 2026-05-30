use std::collections::HashMap;
use crate::linters::ast_analyzer;
use crate::linters::{CodeIssue, IssueSeverity};

#[derive(Debug, Default)]
pub struct PyFindings {
    pub issues: Vec<CodeIssue>,
    pub metrics: HashMap<String, f32>,
}

struct IndentScope {
    start_line: usize,
    indent: usize,
}

fn detect_indent(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

fn is_comment_or_empty(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with('#')
}

pub fn analyze_python(code: &str) -> PyFindings {
    let mut issues = Vec::new();
    let clean = ast_analyzer::strip_comments_and_strings(code);
    let lines: Vec<&str> = clean.lines().collect();
    let mut scopes: Vec<IndentScope> = Vec::new();

    for (i, raw_line) in code.lines().enumerate() {
        let line_num = i + 1;
        if is_comment_or_empty(raw_line) { continue; }
        let trimmed = raw_line.trim();
        let clean_line = if i < lines.len() { lines[i] } else { "" };

        let indent = detect_indent(raw_line);
        while let Some(top) = scopes.last() {
            if indent <= top.indent { scopes.pop(); } else { break; }
        }

        if trimmed.ends_with(':') && !trimmed.starts_with('#') {
            scopes.push(IndentScope { start_line: line_num, indent });
        }

        // ── exec() / eval() ────────────────────────────────────────
        if (trimmed.starts_with("exec(") || trimmed.contains("exec("))
            && !trimmed.starts_with('#')
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "exec() detected — arbitrary code execution risk. Remove or replace with safe alternatives.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-EXEC".to_string(),
            });
        }
        if (trimmed.starts_with("eval(") || trimmed.contains("eval("))
            && !trimmed.starts_with('#')
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "eval() detected — arbitrary code execution risk. Use ast.literal_eval or a proper parser.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-EVAL".to_string(),
            });
        }

        // ── pickle deserialization ─────────────────────────────────
        if trimmed.contains("pickle.loads(") || trimmed.contains("pickle.load(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "pickle deserialization detected — remote code execution risk. Use JSON or a safe serialization format.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-PICKLE".to_string(),
            });
        }

        // ── yaml.load without SafeLoader ───────────────────────────
        if trimmed.contains("yaml.load(") && !trimmed.contains("SafeLoader") && !trimmed.contains("FullLoader") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "yaml.load() without SafeLoader — arbitrary code execution risk. Use yaml.safe_load() instead.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-YAML-LOAD".to_string(),
            });
        }

        // ── command injection via os / subprocess ──────────────────
        if trimmed.contains("os.system(") || trimmed.contains("os.popen(") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "Shell command execution detected — command injection risk. Use subprocess with argument list instead of shell=True.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-CMD-INJECTION".to_string(),
            });
        }
        if (trimmed.contains("subprocess.call(") || trimmed.contains("subprocess.Popen(")
            || trimmed.contains("subprocess.run(") || trimmed.contains("subprocess.check_output("))
            && (trimmed.contains("shell=True") || clean_line.contains("shell=True"))
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Critical,
                category: "Security".to_string(),
                message: "subprocess with shell=True — command injection risk. Pass arguments as a list without shell=True.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-SUBPROC-SHELL".to_string(),
            });
        }

        // ── SQL injection via f-strings / format / % ───────────────
        let has_sql_kw = trimmed.to_uppercase().contains("SELECT ")
            || trimmed.to_uppercase().contains("INSERT ")
            || trimmed.to_uppercase().contains("DELETE ")
            || trimmed.to_uppercase().contains("UPDATE ")
            || trimmed.to_uppercase().contains("DROP ");
        if has_sql_kw {
            let has_fmt = trimmed.contains("f\"") || trimmed.contains("f'")
                || trimmed.contains(".format(") || trimmed.contains("%s") || trimmed.contains("%(");
            if has_fmt {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Critical,
                    category: "Security".to_string(),
                    message: "Dynamic SQL query with string formatting — SQL injection risk. Use parameterized queries (e.g., ? placeholders in SQLite/psycopg2).".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "PY-SQL-INJECTION".to_string(),
                });
            }
        }

        // ── assert used for security validation ────────────────────
        if trimmed.starts_with("assert ") && scopes.len() >= 2 {
            let in_fn = scopes.iter().any(|s| {
                if s.start_line < line_num {
                    let src_line = code.lines().nth(s.start_line - 1).unwrap_or("");
                    src_line.trim().starts_with("def ")
                } else { false }
            });
            if in_fn {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "Security".to_string(),
                    message: "assert used for validation inside function — disabled with python -O. Use proper if/raise instead.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "PY-ASSERT-VALIDATION".to_string(),
                });
            }
        }

        // ── hardcoded secrets ──────────────────────────────────────
        let lower = trimmed.to_lowercase();
        if (lower.contains("password") || lower.contains("secret") || lower.contains("api_key")
            || lower.contains("apikey") || lower.contains("auth_token") || lower.contains("token"))
            && (trimmed.contains('"') || trimmed.contains('\''))
            && !lower.contains("env(") && !lower.contains("environ.get") && !lower.contains("os.getenv")
        {
            issues.push(CodeIssue {
                severity: IssueSeverity::Error,
                category: "Security".to_string(),
                message: "Hardcoded secret detected — use environment variables or a secrets manager.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-HARDCODED-SECRET".to_string(),
            });
        }

        // ── Flask debug mode ───────────────────────────────────────
        if trimmed.contains("app.run(") && trimmed.contains("debug=True") {
            issues.push(CodeIssue {
                severity: IssueSeverity::Error,
                category: "Security".to_string(),
                message: "Flask debug mode enabled in production — debugger allows arbitrary code execution.".to_string(),
                line_number: Some(line_num),
                column_number: None,
                rule_id: "PY-FLASK-DEBUG".to_string(),
            });
        }

        // ── request without TLS verification ───────────────────────
        if (trimmed.contains("requests.get(") || trimmed.contains("requests.post("))
            && !trimmed.contains("verify=")
        {
            let next_lines: String = code.lines().skip(i).take(3).collect::<Vec<_>>().join(" ");
            if !next_lines.contains("verify=") {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Warning,
                    category: "Security".to_string(),
                    message: "requests.get/post without verify — TLS certificate verification not disabled but not explicit. Set verify=True or False explicitly.".to_string(),
                    line_number: Some(line_num),
                    column_number: None,
                    rule_id: "PY-REQUESTS-VERIFY".to_string(),
                });
            }
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("py_issues".to_string(), issues.len() as f32);
    metrics.insert("py_high_severity".to_string(),
        issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical | IssueSeverity::Error)).count() as f32,
    );

    PyFindings { issues, metrics }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_exec() {
        let findings = analyze_python(r#"exec("malicious code")"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-EXEC"));
    }

    #[test]
    fn test_python_eval() {
        let findings = analyze_python(r#"eval(user_input)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-EVAL"));
    }

    #[test]
    fn test_python_pickle() {
        let findings = analyze_python(r#"data = pickle.loads(raw)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-PICKLE"));
    }

    #[test]
    fn test_python_yaml_unsafe() {
        let findings = analyze_python(r#"config = yaml.load(data)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-YAML-LOAD"));
    }

    #[test]
    fn test_python_yaml_safe() {
        let findings = analyze_python(r#"config = yaml.safe_load(data)"#);
        assert!(!findings.issues.iter().any(|i| i.rule_id == "PY-YAML-LOAD"));
    }

    #[test]
    fn test_python_sql_injection() {
        let findings = analyze_python(r#"query = f"SELECT * FROM users WHERE id = {user_id}""#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-SQL-INJECTION"));
    }

    #[test]
    fn test_python_hardcoded_secret() {
        let findings = analyze_python(r#"api_key = "sk-1234abcd""#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-HARDCODED-SECRET"));
    }

    #[test]
    fn test_python_flask_debug() {
        let findings = analyze_python(r#"app.run(host="0.0.0.0", debug=True)"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "PY-FLASK-DEBUG"));
    }

    #[test]
    fn test_python_no_false_positive_comment() {
        let findings = analyze_python("# eval is dangerous");
        assert!(!findings.issues.iter().any(|i| i.rule_id == "PY-EVAL"));
    }
}
