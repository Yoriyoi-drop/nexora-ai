//! Performance Verifier
//!
//! Implements performance verification for code efficiency and optimization.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::{
    CodeIssue,
    CodeVerifier,
    IssueSeverity,
    VerificationResult,
    VerifierType,
};

/// Performance thresholds configuration
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub time_limits_ms: Vec<u64>,
    pub time_scores: Vec<f32>,
    pub memory_limits_mb: Vec<f32>,
    pub memory_scores: Vec<f32>,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            time_limits_ms: vec![10, 100, 1000, 10000],
            time_scores: vec![1.0, 0.8, 0.6, 0.4, 0.2],
            memory_limits_mb: vec![1.0, 10.0, 100.0, 1000.0],
            memory_scores: vec![1.0, 0.8, 0.6, 0.4, 0.2],
        }
    }
}

impl PerformanceThresholds {
    pub fn get_time_score(&self, time_ms: u64) -> f32 {
        for (i, &limit) in self.time_limits_ms.iter().enumerate() {
            if time_ms < limit {
                return self.time_scores[i];
            }
        }
        *self.time_scores.last().unwrap_or(&0.2)
    }

    pub fn get_memory_score(&self, memory_mb: f32) -> f32 {
        for (i, &limit) in self.memory_limits_mb.iter().enumerate() {
            if memory_mb < limit {
                return self.memory_scores[i];
            }
        }
        *self.memory_scores.last().unwrap_or(&0.2)
    }
}

/// Performance pattern
#[derive(Debug, Clone)]
struct PerformancePattern {
    name: &'static str,
    pattern: &'static Lazy<Regex>,
    severity: IssueSeverity,
    description: &'static str,
    language: &'static str,
    performance_impact: f32,
    rule_id: &'static str,
}

/// Regex patterns with proper error handling
static NESTED_LOOP_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)for\s*\(.*?\)\s*\{.*?for\s*\(")
        .expect("Invalid nested loop regex pattern")
});

static PYTHON_STRING_CONCAT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"\w+\s*\+=\s*["']"#)
        .expect("Invalid Python string concat regex pattern")
});

static C_MALLOC_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\bmalloc\s*\(")
        .expect("Invalid malloc regex pattern")
});

static BLOCKING_IO_LOOP_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)(for|while).*(read|recv|input)\s*\(")
        .expect("Invalid blocking I/O regex pattern")
});

static INEFFICIENT_SORT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(bubble_sort|selection_sort)\b")
        .expect("Invalid inefficient sort regex pattern")
});

/// Performance verifier
pub struct PerformanceVerifier {
    performance_patterns: Vec<PerformancePattern>,
}

impl PerformanceVerifier {
    pub fn new() -> Self {
        Self {
            performance_patterns: vec![
                PerformancePattern {
                    name: "Nested Loops",
                    pattern: &NESTED_LOOP_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Nested loops detected, potential O(n²) complexity",
                    language: "all",
                    performance_impact: 0.25,
                    rule_id: "nested_loops",
                },
                PerformancePattern {
                    name: "Inefficient String Concatenation",
                    pattern: &PYTHON_STRING_CONCAT_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Repeated string concatenation detected",
                    language: "python",
                    performance_impact: 0.15,
                    rule_id: "string_concat",
                },
                PerformancePattern {
                    name: "Potential Memory Leak",
                    pattern: &C_MALLOC_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "malloc detected, ensure free() is called",
                    language: "c",
                    performance_impact: 0.30,
                    rule_id: "memory_leak",
                },
                PerformancePattern {
                    name: "Blocking I/O In Loop",
                    pattern: &BLOCKING_IO_LOOP_REGEX,
                    severity: IssueSeverity::Warning,
                    description: "Blocking I/O inside loop detected",
                    language: "all",
                    performance_impact: 0.35,
                    rule_id: "blocking_io_loop",
                },
                PerformancePattern {
                    name: "Inefficient Sorting",
                    pattern: &INEFFICIENT_SORT_REGEX,
                    severity: IssueSeverity::Info,
                    description: "Inefficient sorting algorithm detected",
                    language: "all",
                    performance_impact: 0.15,
                    rule_id: "inefficient_sort",
                },
            ],
        }
    }

    /// Validate input code before processing
    fn validate_input(code: &str) -> Result<()> {
        if code.is_empty() {
            return Err(anyhow::anyhow!("Code input cannot be empty"));
        }
        if code.len() > 1_000_000 {
            return Err(anyhow::anyhow!("Code input too large (max 1MB)"));
        }
        Ok(())
    }

    /// Add issue with deduplication
    fn add_issue(
        issues: &mut Vec<CodeIssue>,
        seen: &mut HashSet<String>,
        issue: CodeIssue,
    ) {
        let key = format!(
            "{}:{}:{:?}",
            issue.rule_id,
            issue.line_number.unwrap_or(0),
            issue.severity
        );

        if seen.insert(key) {
            issues.push(issue);
        }
    }
}

impl CodeVerifier for PerformanceVerifier {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult> {
        // Validate input
        Self::validate_input(code)?;
        
        let mut issues = Vec::new();
        let mut seen = HashSet::new();
        let mut score: f32 = 1.0;

        // Language-specific checks
        let language_issues =
            self.check_language_specific_performance(code, language)?;

        for issue in language_issues {
            Self::add_issue(&mut issues, &mut seen, issue);
        }

        // General pattern checks
        for pattern in &self.performance_patterns {
            if pattern.language != "all" && pattern.language != language {
                continue;
            }

            for mat in pattern.pattern.find_iter(code) {
                let line_number = code[..mat.start()]
                    .lines()
                    .count()
                    .max(1);

                Self::add_issue(
                    &mut issues,
                    &mut seen,
                    CodeIssue {
                        severity: pattern.severity.clone(),
                        category: "Performance".to_string(),
                        message: format!(
                            "{}: {}",
                            pattern.name,
                            pattern.description
                        ),
                        line_number: Some(line_number),
                        column_number: None,
                        rule_id: pattern.rule_id.to_string(),
                    },
                );

                score -= pattern.performance_impact;
            }
        }

        score = score.clamp(0.0, 1.0);

        let suggestions =
            self.generate_performance_suggestions(&issues);

        let mut metrics = HashMap::new();
        metrics.insert("performance_score".to_string(), score);
        metrics.insert(
            "performance_issue_count".to_string(),
            issues.len() as f32,
        );
        metrics.insert(
            "high_impact_count".to_string(),
            issues
                .iter()
                .filter(|i| matches!(i.severity, IssueSeverity::Error))
                .count() as f32,
        );

        Ok(VerificationResult {
            verifier_name: "PerformanceVerifier".to_string(),
            verifier_type: VerifierType::Performance,
            score,
            passed: score >= 0.7,
            issues,
            suggestions,
            metrics,
        })
    }

    fn verifier_name(&self) -> &str {
        "PerformanceVerifier"
    }

    fn verifier_type(&self) -> VerifierType {
        VerifierType::Performance
    }

    fn check_language_specific_performance(
        &self,
        code: &str,
        language: &str,
    ) -> Result<Vec<CodeIssue>> {
        let mut issues = Vec::new();

        match language {
            "rust" => {
                // More sophisticated clone detection
                let clone_count = self.count_clones_efficiently(code);
                
                if clone_count > 5 {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        message: format!(
                            "Excessive cloning detected ({} clones). Consider using references or borrowing",
                            clone_count
                        ),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_excessive_clone".to_string(),
                    });
                }

                // Better pattern matching for intermediate allocations
                if self.has_intermediate_allocation(code) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Performance".to_string(),
                        message: "Intermediate Vec allocation detected. Consider using iterators or pre-allocation".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_intermediate_alloc".to_string(),
                    });
                }
                
                // Check for inefficient string operations
                if self.has_inefficient_string_ops(code) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        message: "Inefficient string operations detected. Consider using String::with_capacity or format!".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "rust_inefficient_string".to_string(),
                    });
                }
            }

            "python" => {
                // Enhanced Python pattern detection
                if code.contains("range(len(") {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Info,
                        category: "Performance".to_string(),
                        message: "Consider using enumerate() instead of range(len())".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_range_len".to_string(),
                    });
                }
                
                // Check for inefficient list operations
                if self.has_inefficient_python_lists(code) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        message: "Inefficient list operations detected. Consider using list comprehensions or generators".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "py_inefficient_lists".to_string(),
                    });
                }
            }

            "javascript" => {
                // Enhanced JavaScript pattern detection
                if self.has_dom_query_in_loop(code) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        message: "DOM query inside loop may hurt performance. Cache DOM queries outside loops".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_dom_query_loop".to_string(),
                    });
                }
                
                // Check for memory leaks
                if self.has_potential_memory_leaks(code) {
                    issues.push(CodeIssue {
                        severity: IssueSeverity::Warning,
                        category: "Performance".to_string(),
                        message: "Potential memory leak detected. Ensure proper cleanup of event listeners and timers".to_string(),
                        line_number: None,
                        column_number: None,
                        rule_id: "js_memory_leak".to_string(),
                    });
                }
            }

            _ => {}
        }

        Ok(issues)
    }
    
    /// Count clones efficiently using regex
    fn count_clones_efficiently(&self, code: &str) -> usize {
        let Ok(clone_regex) = regex::Regex::new(r"\.clone\(\)") else { return 0 };
        clone_regex.find_iter(code).count()
    }
    
    /// Check for intermediate allocations
    fn has_intermediate_allocation(&self, code: &str) -> bool {
        code.contains(".collect::<Vec<_>>())") && code.contains(".iter()")
    }
    
    /// Check for inefficient string operations
    fn has_inefficient_string_ops(&self, code: &str) -> bool {
        code.contains("String::new()") && code.contains(".push_str(")
    }
    
    /// Check for inefficient Python list operations
    fn has_inefficient_python_lists(&self, code: &str) -> bool {
        code.contains("for i in range(len(") || code.contains("list.append(")
    }
    
    /// Check for DOM queries in loops
    fn has_dom_query_in_loop(&self, code: &str) -> bool {
        let dom_methods = ["getElementById", "querySelector", "querySelectorAll", "getElementsByClassName"];
        dom_methods.iter().any(|method| code.contains(method)) && code.contains("for")
    }
    
    /// Check for potential memory leaks
    fn has_potential_memory_leaks(&self, code: &str) -> bool {
        code.contains("addEventListener") && !code.contains("removeEventListener")
    }

    fn generate_performance_suggestions(
        &self,
        issues: &[CodeIssue],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if issues.iter().any(|i| i.rule_id == "nested_loops") {
            suggestions.push(
                "Consider hash maps, indexing, or caching to reduce nested iteration"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| i.rule_id == "string_concat") {
            suggestions.push(
                "Use buffered string builders or join operations"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| i.rule_id == "memory_leak") {
            suggestions.push(
                "Ensure allocated memory is properly freed"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| i.rule_id == "blocking_io_loop") {
            suggestions.push(
                "Move blocking I/O outside loops or batch operations"
                    .to_string(),
            );
        }

        if issues.iter().any(|i| i.rule_id == "rust_excessive_clone") {
            suggestions.push(
                "Use borrowing (&T) instead of cloning where possible"
                    .to_string(),
            );
        }

        suggestions
    }
}
