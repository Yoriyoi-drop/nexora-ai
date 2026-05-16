//! Code Verifier Manager
//! 
//! Main manager for coordinating multiple code verifiers.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{security::SecurityVerifier, performance::PerformanceVerifier, correctness::CorrectnessVerifier, style::StyleVerifier};

/// Code verifier interface
pub trait CodeVerifier: Send + Sync {
    fn verify(&self, code: &str, language: &str) -> Result<VerificationResult>;
    fn verifier_name(&self) -> &str;
    fn verifier_type(&self) -> VerifierType;
    
    // Optional methods for specific verifiers
    fn check_language_specific_security(&self, _code: &str, _language: &str) -> Result<Vec<CodeIssue>> {
        Ok(Vec::new())
    }
    
    fn generate_security_suggestions(&self, _issues: &[CodeIssue]) -> Vec<String> {
        Vec::new()
    }
    
    fn check_language_specific_performance(&self, _code: &str, _language: &str) -> Result<Vec<CodeIssue>> {
        Ok(Vec::new())
    }
    
    fn calculate_complexity(&self, _code: &str) -> f32 {
        1.0
    }
    
    fn generate_performance_suggestions(&self, _issues: &[CodeIssue]) -> Vec<String> {
        Vec::new()
    }
    
    fn check_language_specific_correctness(&self, _code: &str, _language: &str) -> Result<Vec<CodeIssue>> {
        Ok(Vec::new())
    }
    
    fn generate_correctness_suggestions(&self, _issues: &[CodeIssue]) -> Vec<String> {
        Vec::new()
    }
    
    fn check_language_specific_style(&self, _code: &str, _language: &str) -> Result<Vec<CodeIssue>> {
        Ok(Vec::new())
    }
    
    fn generate_style_suggestions(&self, _issues: &[CodeIssue]) -> Vec<String> {
        Vec::new()
    }
    
    // Performance analysis methods with default implementations
    fn count_clones_efficiently(&self, code: &str) -> usize {
        let clone_regex = regex::Regex::new(r"\.clone\(\)").unwrap_or_else(|_| regex::Regex::new(r"").unwrap());
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
        let dom_methods = ["getElementById", "querySelector", "querySelectorAll", "getElementsByClassName"];
        dom_methods.iter().any(|method| code.contains(method)) && code.contains("for")
    }
    
    fn has_potential_memory_leaks(&self, code: &str) -> bool {
        code.contains("malloc") && !code.contains("free")
    }
}

/// Types of verifiers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
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

/// Main code verifier manager
pub struct CodeVerifierManager {
    verifiers: Vec<Box<dyn CodeVerifier>>,
}

impl CodeVerifierManager {
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
    
    pub fn get_security_score(&self, code: &str, language: &str) -> Result<f32> {
        for verifier in &self.verifiers {
            if verifier.verifier_type() == VerifierType::Security {
                let result = verifier.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }
    
    pub fn get_performance_score(&self, code: &str, language: &str) -> Result<f32> {
        for verifier in &self.verifiers {
            if verifier.verifier_type() == VerifierType::Performance {
                let result = verifier.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }
    
    pub fn get_correctness_score(&self, code: &str, language: &str) -> Result<f32> {
        for verifier in &self.verifiers {
            if verifier.verifier_type() == VerifierType::Correctness {
                let result = verifier.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }
    
    pub fn get_style_score(&self, code: &str, language: &str) -> Result<f32> {
        for verifier in &self.verifiers {
            if verifier.verifier_type() == VerifierType::Style {
                let result = verifier.verify(code, language)?;
                return Ok(result.score);
            }
        }
        Ok(1.0)
    }
    
    pub fn get_verifier_names(&self) -> Vec<String> {
        self.verifiers.iter()
            .map(|v| v.verifier_name().to_string())
            .collect()
    }
    
    pub fn get_issues_by_severity(&self, code: &str, language: &str, severity: IssueSeverity) -> Result<Vec<CodeIssue>> {
        let mut filtered_issues = Vec::new();
        
        for verifier in &self.verifiers {
            let result = verifier.verify(code, language)?;
            filtered_issues.extend(
                result.issues
                    .into_iter()
                    .filter(|issue| issue.severity == severity)
            );
        }
        
        Ok(filtered_issues)
    }
    
    pub fn get_summary_report(&self, code: &str, language: &str) -> Result<VerificationSummary> {
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
                    IssueSeverity::Error => error_count += 1,
                    IssueSeverity::Warning => warning_count += 1,
                    IssueSeverity::Info => info_count += 1,
                    IssueSeverity::Style => style_count += 1,
                }
            }
        }
        
        let overall_score = results.iter()
            .map(|r| r.score)
            .sum::<f32>() / results.len() as f32;
        
        Ok(VerificationSummary {
            overall_score,
            total_issues,
            error_count,
            warning_count,
            info_count,
            style_count,
            verifier_results: results,
        })
    }
}

/// Verification summary report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub overall_score: f32,
    pub total_issues: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub style_count: usize,
    pub verifier_results: Vec<VerificationResult>,
}
