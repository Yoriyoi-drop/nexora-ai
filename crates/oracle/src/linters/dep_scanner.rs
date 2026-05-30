use std::collections::HashMap;
use crate::linters::{CodeIssue, IssueSeverity};

#[derive(Debug, Default)]
pub struct DepFindings {
    pub issues: Vec<CodeIssue>,
    pub metrics: HashMap<String, f32>,
}

struct VulnPattern {
    name: &'static str,
    description: &'static str,
    severity: IssueSeverity,
    rule_id: &'static str,
}

static DEP_PATTERNS: &[VulnPattern] = &[
    VulnPattern {
        name: "log4j < 2.17.0",
        description: "Log4Shell (CVE-2021-44228 / CVE-2021-45105) — JNDI injection via log messages. Upgrade to log4j 2.17.0+.",
        severity: IssueSeverity::Critical,
        rule_id: "DEP-LOG4J",
    },
    VulnPattern {
        name: "lodash < 4.17.21",
        description: "Prototype pollution in lodash (CVE-2020-8203, CVE-2021-23337). Upgrade to lodash 4.17.21+.",
        severity: IssueSeverity::High,
        rule_id: "DEP-LODASH",
    },
    VulnPattern {
        name: "nth-check < 2.1.1",
        description: "Inefficient regex in nth-check (CVE-2023-26100) — ReDoS. Upgrade to nth-check 2.1.1+.",
        severity: IssueSeverity::Warning,
        rule_id: "DEP-NTH-CHECK",
    },
    VulnPattern {
        name: "semver < 7.5.2",
        description: "ReDoS in semver (CVE-2023-26102). Upgrade to semver 7.5.2+.",
        severity: IssueSeverity::Medium,
        rule_id: "DEP-SEMVER",
    },
    VulnPattern {
        name: "crypto-js < 4.2.0",
        description: "Timing attack in crypto-js (CVE-2023-46233). Upgrade to crypto-js 4.2.0+.",
        severity: IssueSeverity::High,
        rule_id: "DEP-CRYPTO-JS",
    },
    VulnPattern {
        name: "json5 < 2.2.2",
        description: "Prototype pollution in json5 (CVE-2022-46175). Upgrade to json5 2.2.2+.",
        severity: IssueSeverity::High,
        rule_id: "DEP-JSON5",
    },
    VulnPattern {
        name: "path-parse < 1.0.7",
        description: "ReDoS in path-parse (CVE-2021-3803). Upgrade to path-parse 1.0.7+.",
        severity: IssueSeverity::Warning,
        rule_id: "DEP-PATH-PARSE",
    },
    VulnPattern {
        name: "minimatch < 3.1.2",
        description: "ReDoS in minimatch (CVE-2022-3517). Upgrade to minimatch 3.1.2+.",
        severity: IssueSeverity::Warning,
        rule_id: "DEP-MINIMATCH",
    },
    VulnPattern {
        name: "underscore < 1.13.0-2",
        description: "Arbitrary code execution in underscore (CVE-2021-23358). Upgrade to underscore 1.13.0+.",
        severity: IssueSeverity::Critical,
        rule_id: "DEP-UNDERSCORE",
    },
    VulnPattern {
        name: "axios < 1.6.0",
        description: "Server-Side Request Forgery in axios (CVE-2023-45857). Upgrade to axios 1.6.0+.",
        severity: IssueSeverity::High,
        rule_id: "DEP-AXIOS",
    },
    VulnPattern {
        name: "go-yaml < 2.2.8",
        description: "Unmarshal bomb in go-yaml (CVE-2023-45288). Upgrade to gopkg.in/yaml.v3 or yaml.v2 2.2.8+.",
        severity: IssueSeverity::High,
        rule_id: "DEP-GO-YAML",
    },
    VulnPattern {
        name: "fast-xml-parser < 4.2.4",
        description: "XXE in fast-xml-parser (CVE-2023-34104). Upgrade to fast-xml-parser 4.2.4+.",
        severity: IssueSeverity::High,
        rule_id: "DEP-FAST-XML",
    },
];

static DEP_KEYWORDS: &[(&str, &str)] = &[
    ("log4j", "DEP-LOG4J"),
    ("org.apache.logging.log4j", "DEP-LOG4J"),
    ("lodash", "DEP-LODASH"),
    ("nth-check", "DEP-NTH-CHECK"),
    ("semver@", "DEP-SEMVER"),
    ("\"semver\"", "DEP-SEMVER"),
    ("crypto-js", "DEP-CRYPTO-JS"),
    ("json5", "DEP-JSON5"),
    ("path-parse", "DEP-PATH-PARSE"),
    ("minimatch", "DEP-MINIMATCH"),
    ("underscore", "DEP-UNDERSCORE"),
    ("axios", "DEP-AXIOS"),
    ("gopkg.in/yaml", "DEP-GO-YAML"),
    ("fast-xml-parser", "DEP-FAST-XML"),
];

pub fn scan_dependencies(code: &str) -> DepFindings {
    let mut issues = Vec::new();
    let mut matched = std::collections::HashSet::new();
    let lines: Vec<&str> = code.lines().collect();

    for (keyword, rule_id) in DEP_KEYWORDS {
        let rid: &str = rule_id;
        if matched.contains(rid) { continue; }
        let mut found = false;
        let mut line_num = None;
        for (i, line) in lines.iter().enumerate() {
            if line.contains(keyword) {
                let t = line.trim();
                if !t.starts_with("//") && !t.starts_with('#') {
                    found = true;
                    line_num = Some(i + 1);
                    break;
                }
            }
        }
        if found {
            if let Some(pattern) = DEP_PATTERNS.iter().find(|p| p.rule_id == *rule_id) {
                issues.push(CodeIssue {
                    severity: pattern.severity.clone(),
                    category: "Dependency".to_string(),
                    message: format!("{} — {}", pattern.name, pattern.description),
                    line_number: line_num,
                    column_number: None,
                    rule_id: pattern.rule_id.to_string(),
                });
                matched.insert(rid.to_string());
            }
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("dep_issues".to_string(), issues.len() as f32);
    metrics.insert("dep_critical".to_string(),
        issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Critical)).count() as f32,
    );

    DepFindings { issues, metrics }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dep_log4j() {
        let findings = scan_dependencies(r#"log4j version 2.14.0"#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "DEP-LOG4J"));
    }

    #[test]
    fn test_dep_lodash() {
        let findings = scan_dependencies(r#""lodash": "^4.17.20""#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "DEP-LODASH"));
    }

    #[test]
    fn test_dep_not_in_comment() {
        let findings = scan_dependencies(r#"// log4j is used"#);
        assert!(!findings.issues.iter().any(|i| i.rule_id == "DEP-LOG4J"));
    }

    #[test]
    fn test_dep_empty() {
        let findings = scan_dependencies(r#"fn main() { println!("hello"); }"#);
        assert!(findings.issues.is_empty());
    }

    #[test]
    fn test_dep_axios() {
        let findings = scan_dependencies(r#""axios": "^1.5.0""#);
        assert!(findings.issues.iter().any(|i| i.rule_id == "DEP-AXIOS"));
    }
}
