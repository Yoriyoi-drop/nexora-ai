//! Code Verifiers untuk ORACLE
//! 
//! Verifiers khusus untuk kode yang memeriksa security,
//! efficiency, correctness, dan best practices
//! berbagai bahasa pemrograman.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;

/// Code verifier interface
pub trait CodeVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult>;
    fn verifier_name(&self) -> &str;
    fn verifier_type(&self) -> VerifierType;
}

/// Types of verifiers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VerifierType {
    Security,
    Performance,
    Correctness,
    Style,
    BestPractices,
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub verifier_name: String,
    pub verifier_type: VerifierType,
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

/// Issue severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
    Style,
}

/// Security verifier
pub struct SecurityVerifier {
    vulnerability_patterns: Vec<VulnerabilityPattern>,
}

impl SecurityVerifier {
    pub fn new() -> Self {
        Self {
            vulnerability_patterns: vec![
                VulnerabilityPattern {
                    name: "SQL Injection".to_string(),
                    pattern: Regex::new(r"(?i)(SELECT|INSERT|UPDATE|DELETE).*\+.*\s*(FROM|INTO|SET|WHERE)").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential SQL injection vulnerability".to_string(),
                    cwe_id: "CWE-89".to_string(),
                },
                VulnerabilityPattern {
                    name: "Command Injection".to_string(),
                    pattern: Regex::new(r"(?i)(system|exec|shell|subprocess\.call).*\+.*\s*(\(|\")").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential command injection vulnerability".to_string(),
                    cwe_id: "CWE-78".to_string(),
                },
                VulnerabilityPattern {
                    name: "Cross-Site Scripting".to_string(),
                    pattern: Regex::new(r"(?i)(innerHTML|document\.write|eval\(|\.innerHTML\s*=).*\+.*\s*(<|\"|')").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential XSS vulnerability".to_string(),
                    cwe_id: "CWE-79".to_string(),
                },
                VulnerabilityPattern {
                    name: "Hardcoded Credentials".to_string(),
                    pattern: Regex::new(r"(?i)(password|passwd|secret|api_key|token)\s*=\s*[\"'][^\"']+[\"']").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Hardcoded credentials detected".to_string(),
                    cwe_id: "CWE-798".to_string(),
                },
                VulnerabilityPattern {
                    name: "Path Traversal".to_string(),
                    pattern: Regex::new(r"(?i)(\.\.|\.\.\/|\.\.\\).*\s*(\+|\$)").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential path traversal vulnerability".to_string(),
                    cwe_id: "CWE-22".to_string(),
                },
                VulnerabilityPattern {
                    name: "Insecure Randomness".to_string(),
                    pattern: Regex::new(r"(?i)(random|rand)\s*\(\s*(time\(\)|date\(\)|seed\()").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Insecure random number generation".to_string(),
                    cwe_id: "CWE-338".to_string(),
                },
                VulnerabilityPattern {
                    name: "Buffer Overflow".to_string(),
                    pattern: Regex::new(r"(?i)(strcpy|strcat|gets|sprintf)\s*\(").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential buffer overflow vulnerability".to_string(),
                    cwe_id: "CWE-120".to_string(),
                },
                VulnerabilityPattern {
                    name: "Race Condition".to_string(),
                    pattern: Regex::new(r"(?i)(shared|global)\s+.*\s*(\+\+|--)").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Potential race condition".to_string(),
                    cwe_id: "CWE-362".to_string(),
                },
            ],
        }
    }
}

impl CodeVerifier for SecurityVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult> {
        let mut issues = Vec::new();
        let mut score = 1.0;
        
        // Check against vulnerability patterns
        for pattern in &self.vulnerability_patterns {
            for (line_num, line) in code.lines().enumerate() {
                if let Some(mat) = pattern.pattern.find(line) {
                    let severity_penalty = match pattern.severity {
                        IssueSeverity::Error => 0.3,
                        IssueSeverity::Warning => 0.1,
                        IssueSeverity::Info => 0.05,
                        IssueSeverity::Style => 0.02,
                    };
                    
                    score -= severity_penalty;
                    
                    issues.push(CodeIssue {
                        severity: pattern.severity.clone(),
                        category: "Security".to_string(),
                        message: format!("{}: {}", pattern.name, pattern.description),
                        line_number: Some(line_num + 1),
                        column_number: Some(mat.start()),
                        rule_id: pattern.cwe_id.clone(),
                    });
                }
            }
        }
        
        // Language-specific security checks
        let language_issues = self.check_language_specific_security(code, language)?;
        issues.extend(language_issues);
        
        // Calculate final score
        score = score.max(0.0);
        
        let mut metrics = HashMap::new();
        metrics.insert("security_score".to_string(), score);
        metrics.insert("vulnerability_count".to_string(), issues.len() as f32);
        
        Ok(VerificationResult {
            verifier_name: "SecurityVerifier".to_string(),
            verifier_type: VerifierType::Security,
            score,
            passed: score >= 0.7,
            issues,
            suggestions: self.generate_security_suggestions(&issues),
            metrics,
        })
    }
    
    fn verifier_name(&self) -> &str {
        "SecurityVerifier"
    }
    
    fn verifier_type(&self) -> VerifierType {
        VerifierType::Security
    }
    
    fn check_language_specific_security(&self, code: &str, language: &str) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();
        
        match language.to_lowercase().as_str() {
            "python" => {
                // Python-specific security checks
                if code.contains("eval(") || code.contains("exec(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Error,
                        category: "Security".to_string(),
                        message: "Use of eval() or exec() functions".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-SEC-001".to_string(),
                    });
                }
                
                if code.contains("input(") && !code.contains("strip(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: "Unvalidated input() usage".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-SEC-002".to_string(),
                    });
                }
            },
            "javascript" => {
                // JavaScript-specific security checks
                if code.contains("innerHTML") || code.contains("document.write") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Security".to_string(),
                        message: "Potential XSS vulnerability".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "JS-SEC-001".to_string(),
                    });
                }
            },
            "java" => {
                // Java-specific security checks
                if code.contains("System.out.println") && code.contains("input") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Security".to_string(),
                        message: "Debug output with user input".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "JAVA-SEC-001".to_string(),
                    });
                }
            },
            _ => {}
        }
        
        Ok(issues)
    }
    
    fn generate_security_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if issues.iter().any(|i| i.rule_id.contains("SQL")) {
            suggestions.push("Use parameterized queries or prepared statements".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("XSS")) {
            suggestions.push("Sanitize user input before rendering".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("eval")) {
            suggestions.push("Avoid using eval() with user input".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("password")) {
            suggestions.push("Store credentials securely using environment variables".to_string());
        }
        
        suggestions
    }
}

/// Performance verifier
pub struct PerformanceVerifier {
    performance_patterns: Vec<PerformancePattern>,
}

impl PerformanceVerifier {
    pub fn new() -> Self {
        Self {
            performance_patterns: vec![
                PerformancePattern {
                    name: "Nested Loops".to_string(),
                    pattern: Regex::new(r"for\s*\([^)]*\)\s*\{[^}]*for\s*\(").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Nested loops can cause performance issues".to_string(),
                    impact_score: 0.3,
                },
                PerformancePattern {
                    name: "Inefficient String Concatenation".to_string(),
                    pattern: Regex::new(r"\+\s*\".*\"").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Inefficient string concatenation in loops".to_string(),
                    impact_score: 0.2,
                },
                PerformancePattern {
                    name: "Memory Leak Pattern".to_string(),
                    pattern: Regex::new(r"(?i)(new|malloc)\s*\([^)]*\)\s*(?!.*delete|free)").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Potential memory leak".to_string(),
                    impact_score: 0.4,
                },
                PerformancePattern {
                    name: "Synchronous I/O in Loop".to_string(),
                    pattern: Regex::new(r"(?i)(read|write|open)\s*\([^)]*\).*for\s*\(").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Synchronous I/O operations in loop".to_string(),
                    impact_score: 0.5,
                },
            ],
        }
    }
}

impl CodeVerifier for PerformanceVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult> {
        let mut issues = Vec::new();
        let mut score = 1.0;
        
        // Check against performance patterns
        for pattern in &self.performance_patterns {
            for (line_num, line) in code.lines().enumerate() {
                if pattern.pattern.is_match(line) {
                    score -= pattern.impact_score;
                    
                    issues.push(CodeIssue {
                        severity: pattern.severity.clone(),
                        category: "Performance".to_string(),
                        message: format!("{}: {}", pattern.name, pattern.description),
                        line_number: Some(line_num + 1),
                        column_number: None,
                        rule_id: format!("PERF-{}", pattern.name.replace(" ", "_")),
                    });
                }
            }
        }
        
        // Language-specific performance checks
        let language_issues = self.check_language_specific_performance(code, language)?;
        issues.extend(language_issues);
        
        // Calculate complexity-based performance impact
        let complexity = self.calculate_complexity(code);
        if complexity > 10.0 {
            score -= 0.2;
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: "Performance".to_string(),
                message: format!("High cyclomatic complexity: {:.1}", complexity),
                line_number: None,
                column_number: None,
                rule_id: "PERF-COMPLEXITY".to_string(),
            });
        }
        
        score = score.max(0.0);
        
        let mut metrics = HashMap::new();
        metrics.insert("performance_score".to_string(), score);
        metrics.insert("complexity".to_string(), complexity);
        metrics.insert("performance_issues".to_string(), issues.len() as f32);
        
        Ok(VerificationResult {
            verifier_name: "PerformanceVerifier".to_string(),
            verifier_type: VerifierType::Performance,
            score,
            passed: score >= 0.6,
            issues,
            suggestions: self.generate_performance_suggestions(&issues),
            metrics,
        })
    }
    
    fn verifier_name(&self) -> &str {
        "PerformanceVerifier"
    }
    
    fn verifier_type(&self) -> VerifierType {
        VerifierType::Performance
    }
    
    fn check_language_specific_performance(&self, code: &str, language: &str) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();
        
        match language.to_lowercase().as_str() {
            "python" => {
                // Python-specific performance checks
                if code.contains("for i in range(len(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        message: "Use enumerate() instead of range(len())".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-PERF-001".to_string(),
                    });
                }
                
                if code.contains("global ") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Performance".to_string(),
                        message: "Global variables can impact performance".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-PERF-002".to_string(),
                    });
                }
            },
            "javascript" => {
                // JavaScript-specific performance checks
                if code.contains("for (var i = 0; i <") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Performance".to_string(),
                        message: "Consider using forEach() or for...of".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "JS-PERF-001".to_string(),
                    });
                }
            },
            _ => {}
        }
        
        Ok(issues)
    }
    
    fn calculate_complexity(&self, code: &str) -> f32 {
        let mut complexity = 1.0;
        
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("if ") || trimmed.starts_with("elif ") || 
               trimmed.starts_with("for ") || trimmed.starts_with("while ") ||
               trimmed.starts_with("case ") || trimmed.contains("?") {
                complexity += 1.0;
            }
        }
        
        complexity
    }
    
    fn generate_performance_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if issues.iter().any(|i| i.rule_id.contains("NESTED")) {
            suggestions.push("Consider refactoring nested loops into separate functions".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("COMPLEXITY")) {
            suggestions.push("Break down complex functions into smaller ones".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("STRING")) {
            suggestions.push("Use string builders or template literals".to_string());
        }
        
        suggestions
    }
}

/// Correctness verifier
pub struct CorrectnessVerifier {
    correctness_patterns: Vec<CorrectnessPattern>,
}

impl CorrectnessVerifier {
    pub fn new() -> Self {
        Self {
            correctness_patterns: vec![
                CorrectnessPattern {
                    name: "Null Pointer Dereference".to_string(),
                    pattern: Regex::new(r"\*\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?!\s*=\s*NULL|\s*=\s*null|\s*=\s*nullptr)").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential null pointer dereference".to_string(),
                },
                CorrectnessPattern {
                    name: "Division by Zero".to_string(),
                    pattern: Regex::new(r"/\s*(?!\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?:\s*[<>!=]\s*[0-9]+))").unwrap(),
                    severity: IssueSeverity::Error,
                    description: "Potential division by zero".to_string(),
                },
                CorrectnessPattern {
                    name: "Uninitialized Variable".to_string(),
                    pattern: Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*\s*(?!\s*=\s*[^=])").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Potentially uninitialized variable".to_string(),
                },
                CorrectnessPattern {
                    name: "Resource Leak".to_string(),
                    pattern: Regex::new(r"(?i)(fopen|open|malloc|new)\s*\([^)]*\)\s*(?!.*(?:fclose|close|free|delete))").unwrap(),
                    severity: IssueSeverity::Warning,
                    description: "Potential resource leak".to_string(),
                },
            ],
        }
    }
}

impl CodeVerifier for CorrectnessVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult> {
        let mut issues = Vec::new();
        let mut score = 1.0;
        
        // Check against correctness patterns
        for pattern in &self.correctness_patterns {
            for (line_num, line) in code.lines().enumerate() {
                if pattern.pattern.is_match(line) {
                    score -= 0.2;
                    
                    issues.push(CodeIssue {
                        severity: pattern.severity.clone(),
                        category: "Correctness".to_string(),
                        message: format!("{}: {}", pattern.name, pattern.description),
                        line_number: Some(line_num + 1),
                        column_number: None,
                        rule_id: format!("CORRECT-{}", pattern.name.replace(" ", "_")),
                    });
                }
            }
        }
        
        // Language-specific correctness checks
        let language_issues = self.check_language_specific_correctness(code, language)?;
        issues.extend(language_issues);
        
        score = score.max(0.0);
        
        let mut metrics = HashMap::new();
        metrics.insert("correctness_score".to_string(), score);
        metrics.insert("correctness_issues".to_string(), issues.len() as f32);
        
        Ok(VerificationResult {
            verifier_name: "CorrectnessVerifier".to_string(),
            verifier_type: VerifierType::Correctness,
            score,
            passed: score >= 0.7,
            issues,
            suggestions: self.generate_correctness_suggestions(&issues),
            metrics,
        })
    }
    
    fn verifier_name(&self) -> &str {
        "CorrectnessVerifier"
    }
    
    fn verifier_type(&self) -> VerifierType {
        VerifierType::Correctness
    }
    
    fn check_language_specific_correctness(&self, code: &str, language: &str) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();
        
        match language.to_lowercase().as_str() {
            "python" => {
                // Python-specific correctness checks
                if code.contains("== None") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Correctness".to_string(),
                        message: "Use 'is None' instead of '== None'".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-CORRECT-001".to_string(),
                    });
                }
                
                if code.contains("except:") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Correctness".to_string(),
                        message: "Bare except clause catches all exceptions".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-CORRECT-002".to_string(),
                    });
                }
            },
            "java" => {
                // Java-specific correctness checks
                if code.contains("==") && code.contains("String") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Error,
                        category: "Correctness".to_string(),
                        message: "Use .equals() for string comparison".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "JAVA-CORRECT-001".to_string(),
                    });
                }
            },
            _ => {}
        }
        
        Ok(issues)
    }
    
    fn generate_correctness_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if issues.iter().any(|i| i.rule_id.contains("NULL")) {
            suggestions.push("Always check for null before dereferencing pointers".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("DIVISION")) {
            suggestions.push("Check divisor before performing division".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("RESOURCE")) {
            suggestions.push("Ensure proper resource cleanup".to_string());
        }
        
        suggestions
    }
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
                    name: "Long Lines".to_string(),
                    pattern: Regex::new(r".{121,}").unwrap(),
                    severity: IssueSeverity::Style,
                    description: "Line exceeds 120 characters".to_string(),
                },
                StylePattern {
                    name: "Trailing Whitespace".to_string(),
                    pattern: Regex::new(r"[ \t]+$").unwrap(),
                    severity: IssueSeverity::Style,
                    description: "Trailing whitespace".to_string(),
                },
                StylePattern {
                    name: "Inconsistent Indentation".to_string(),
                    pattern: Regex::new(r"^(?:    )*\t").unwrap(),
                    severity: IssueSeverity::Style,
                    description: "Mixed tabs and spaces".to_string(),
                },
                StylePattern {
                    name: "Missing Documentation".to_string(),
                    pattern: Regex::new(r"(def|function|class)\s+\w+.*\{[^}]*//").unwrap(),
                    severity: IssueSeverity::Info,
                    description: "Function without documentation".to_string(),
                },
            ],
        }
    }
}

impl CodeVerifier for StyleVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult> {
        let mut issues = Vec::new();
        let mut score = 1.0;
        
        // Check against style patterns
        for pattern in &self.style_patterns {
            for (line_num, line) in code.lines().enumerate() {
                if pattern.pattern.is_match(line) {
                    score -= 0.05;
                    
                    issues.push(CodeIssue {
                        severity: pattern.severity.clone(),
                        category: "Style".to_string(),
                        message: format!("{}: {}", pattern.name, pattern.description),
                        line_number: Some(line_num + 1),
                        column_number: None,
                        rule_id: format!("STYLE-{}", pattern.name.replace(" ", "_")),
                    });
                }
            }
        }
        
        // Language-specific style checks
        let language_issues = self.check_language_specific_style(code, language)?;
        issues.extend(language_issues);
        
        score = score.max(0.0);
        
        let mut metrics = HashMap::new();
        metrics.insert("style_score".to_string(), score);
        metrics.insert("style_issues".to_string(), issues.len() as f32);
        
        Ok(VerificationResult {
            verifier_name: "StyleVerifier".to_string(),
            verifier_type: VerifierType::Style,
            score,
            passed: score >= 0.8,
            issues,
            suggestions: self.generate_style_suggestions(&issues),
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
        
        match language.to_lowercase().as_str() {
            "python" => {
                // Python PEP 8 checks
                if code.contains("camelCase") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Use snake_case for variable names".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "PY-STYLE-001".to_string(),
                    });
                }
            },
            "java" => {
                // Java naming convention checks
                if code.contains("method_name") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Style,
                        category: "Style".to_string(),
                        message: "Use camelCase for method names".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "JAVA-STYLE-001".to_string(),
                    });
                }
            },
            _ => {}
        }
        
        Ok(issues)
    }
    
    fn generate_style_suggestions(&self, issues: &[CodeIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if issues.iter().any(|i| i.rule_id.contains("LINES")) {
            suggestions.push("Break long lines into multiple shorter lines".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("WHITESPACE")) {
            suggestions.push("Remove trailing whitespace and configure editor to show it".to_string());
        }
        
        if issues.iter().any(|i| i.rule_id.contains("INDENTATION")) {
            suggestions.push("Use consistent indentation (tabs or spaces)".to_string());
        }
        
        suggestions
    }
}

/// Main code verifier
pub struct CodeVerifier {
    verifiers: Vec<Box<dyn CodeVerifier>>,
}

impl CodeVerifier {
    pub fn new() -> Self {
        Self {
            verifiers: vec![
                Box::new(SecurityVerifier::new()),
                Box::new(PerformanceVerifier::new()),
                Box::new(CorrectnessVerifier::new()),
                Box::new(StyleVerifier::new()),
            ],
        }
    }
    
    pub fn add_verifier(&mut self, verifier: Box<dyn CodeVerifier>) {
        self.verifiers.push(verifier);
    }
    
    pub fn verify_code(&self, code: &str, language: &str) -> Result<f32> {
        let mut total_score = 0.0;
        let mut all_issues = Vec::new();
        
        for verifier in &self.verifiers {
            let result = verifier.verify(code, language)?;
            total_score += result.score;
            all_issues.extend(result.issues);
        }
        
        let avg_score = total_score / self.verifiers.len() as f32;
        
        // Log issues for debugging
        for issue in &all_issues {
            println!("Issue: {} - {} (Line: {:?})", 
                issue.category, issue.message, issue.line_number);
        }
        
        Ok(avg_score)
    }
    
    pub fn verify_detailed(&self, code: &str, language: &str) -> Result<Vec<VerificationResult>> {
        let mut results = Vec::new();
        
        for verifier in &self.verifiers {
            let result = verifier.verify(code, language)?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    pub fn get_verifier_summary(&self) -> Vec<String> {
        self.verifiers.iter()
            .map(|v| format!("{}: {:?}", v.verifier_name(), v.verifier_type()))
            .collect()
    }
}

/// Supporting structures
#[derive(Debug, Clone)]
struct VulnerabilityPattern {
    name: String,
    pattern: Regex,
    severity: IssueSeverity,
    description: String,
    cwe_id: String,
}

#[derive(Debug, Clone)]
struct PerformancePattern {
    name: String,
    pattern: Regex,
    severity: IssueSeverity,
    description: String,
    impact_score: f32,
}

#[derive(Debug, Clone)]
struct CorrectnessPattern {
    name: String,
    pattern: Regex,
    severity: IssueSeverity,
    description: String,
}

#[derive(Debug, Clone)]
struct StylePattern {
    name: String,
    pattern: Regex,
    severity: IssueSeverity,
    description: String,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default verifier set
    pub fn create_default_verifiers() -> CodeVerifier {
        CodeVerifier::new()
    }
    
    /// Analyze verification results
    pub fn analyze_results(results: &[VerificationResult]) -> VerificationAnalysis {
        let mut total_score = 0.0;
        let mut passed_count = 0;
        let mut total_issues = 0;
        let mut issues_by_type = HashMap::new();
        
        for result in results {
            total_score += result.score;
            if result.passed {
                passed_count += 1;
            }
            total_issues += result.issues.len();
            
            let count = issues_by_type.entry(result.verifier_type.clone()).or_insert(0);
            *count += result.issues.len();
        }
        
        let avg_score = total_score / results.len() as f32;
        let pass_rate = passed_count as f32 / results.len() as f32;
        
        VerificationAnalysis {
            overall_score: avg_score,
            pass_rate,
            total_issues,
            issues_by_type,
            recommendations: generate_recommendations(results),
        }
    }
    
    fn generate_recommendations(results: &[VerificationResult]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Security recommendations
        if let Some(security_result) = results.iter().find(|r| r.verifier_type == VerifierType::Security) {
            if security_result.score < 0.7 {
                recommendations.push("Review and fix security vulnerabilities immediately".to_string());
            }
        }
        
        // Performance recommendations
        if let Some(performance_result) = results.iter().find(|r| r.verifier_type == VerifierType::Performance) {
            if performance_result.score < 0.6 {
                recommendations.push("Optimize performance bottlenecks and complex code".to_string());
            }
        }
        
        // Correctness recommendations
        if let Some(correctness_result) = results.iter().find(|r| r.verifier_type == VerifierType::Correctness) {
            if correctness_result.score < 0.7 {
                recommendations.push("Fix correctness issues before deployment".to_string());
            }
        }
        
        // Style recommendations
        if let Some(style_result) = results.iter().find(|r| r.verifier_type == VerifierType::Style) {
            if style_result.score < 0.8 {
                recommendations.push("Improve code style and consistency".to_string());
            }
        }
        
        recommendations
    }
    
    #[derive(Debug, Clone)]
    pub struct VerificationAnalysis {
        pub overall_score: f32,
        pub pass_rate: f32,
        pub total_issues: usize,
        pub issues_by_type: HashMap<VerifierType, usize>,
        pub recommendations: Vec<String>,
    }
}
