//! Fix Generator
//! 
//! Generates fixes for common code issues based on error analysis.

use crate::reasoning::saca::{types::*, error::*};
use super::analyzer::ErrorAnalysis;

/// Fix generator for code issues
pub struct FixGenerator;

impl FixGenerator {
    pub fn new() -> Self {
        Self
    }
    
    /// Generate fixes for a candidate based on error analysis
    pub async fn generate_fixes(
        &self,
        candidate: &SamplingCandidate,
        error_analysis: &ErrorAnalysis,
    ) -> SACAResult<Vec<FixSuggestion>> {
        let mut fixes = Vec::new();
        
        // Generate fixes based on error types
        for (i, error_type) in error_analysis.error_types.iter().enumerate() {
            let fix = match error_type.as_str() {
                "SyntaxError" => self.fix_syntax_errors(&candidate.implementation),
                "UnwrapError" => self.fix_missing_semicolons(&candidate.implementation),
                "Panic" => self.add_result_error_handling(&candidate.implementation),
                _ => None,
            };
            
            if let Some(fix_suggestion) = fix {
                let confidence = error_analysis.confidence_scores.get(i).unwrap_or(&0.5);
                fixes.push(FixSuggestion {
                    description: format!("Fix for {}", error_type),
                    fixed_code: fix_suggestion,
                    confidence: *confidence,
                });
            }
        }
        
        // Generate fixes based on root causes
        for (i, root_cause) in error_analysis.root_causes.iter().enumerate() {
            let fix = match root_cause.as_str() {
                "Array access without bounds checking" => {
                    self.add_bounds_checking(&candidate.implementation)
                },
                "Null pointer dereference" => {
                    self.add_comprehensive_null_checks(&candidate.implementation)
                },
                _ => None,
            };
            
            if let Some(fix_suggestion) = fix {
                let confidence = error_analysis.confidence_scores.get(i).unwrap_or(&0.5);
                fixes.push(FixSuggestion {
                    description: format!("Fix for {}", root_cause),
                    fixed_code: fix_suggestion,
                    confidence: *confidence,
                });
            }
        }
        
        Ok(fixes)
    }
    
    /// Fix missing semicolons with proper parsing
    fn fix_missing_semicolons(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines = Vec::new();
        
        for line in lines {
            let trimmed = line.trim();
            
            // Skip comments and empty lines
            if trimmed.starts_with("//") || trimmed.is_empty() {
                fixed_lines.push(line.to_string());
                continue;
            }
            
            // Check if line needs semicolon
            let needs_semicolon = !trimmed.ends_with(';')
                && !trimmed.ends_with('{')
                && !trimmed.ends_with('}')
                && !trimmed.starts_with("fn ")
                && !trimmed.starts_with("struct ")
                && !trimmed.starts_with("enum ")
                && !trimmed.starts_with("impl ")
                && !trimmed.starts_with("mod ")
                && !trimmed.starts_with("use ")
                && !trimmed.starts_with("pub ");
            
            if needs_semicolon {
                fixed_lines.push(format!("{};", line));
            } else {
                fixed_lines.push(line.to_string());
            }
        }
        
        Some(fixed_lines.join("\n"))
    }
    
    /// Fix bracket mismatches with proper counting
    fn _fix_bracket_mismatches(&self, code: &str) -> Option<String> {
        let mut result = code.to_string();
        let mut brace_count = 0;
        let mut paren_count = 0;
        let mut bracket_count = 0;
        
        for ch in code.chars() {
            match ch {
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                '[' => bracket_count += 1,
                ']' => bracket_count -= 1,
                _ => {}
            }
        }
        
        // Add missing closing brackets
        while brace_count > 0 {
            result.push('}');
            brace_count -= 1;
        }
        while paren_count > 0 {
            result.push(')');
            paren_count -= 1;
        }
        while bracket_count > 0 {
            result.push(']');
            bracket_count -= 1;
        }
        
        Some(result)
    }
    
    /// Add proper Result error handling
    fn add_result_error_handling(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines = Vec::new();
        
        for line in lines {
            let trimmed = line.trim();
            
            // Look for potential Result operations without proper handling
            if trimmed.contains("File::open") || trimmed.contains("parse()") || trimmed.contains("write!") {
                // Wrap in proper error handling
                let indent = " ".repeat(line.len() - trimmed.len());
                fixed_lines.push(format!("{}match {} {{", indent, trimmed));
                fixed_lines.push(format!("{}    Ok(value) => value,", indent));
                fixed_lines.push(format!("{}    Err(e) => return Err(SACAError::ExecuteError(format!(\"Operation failed: {{}}\", e))),", indent));
                fixed_lines.push(format!("{}}}", indent));
            } else {
                fixed_lines.push(line.to_string());
            }
        }
        
        Some(fixed_lines.join("\n"))
    }
    
    /// Add bounds checking for array access
    fn add_bounds_checking(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines = Vec::new();
        
        for line in lines {
            let trimmed = line.trim();
            
            // Look for potential array access without bounds checking
            if trimmed.contains('[') && trimmed.contains(']') && !trimmed.contains("len()") {
                // Add bounds checking
                let indent = " ".repeat(line.len() - trimmed.len());
                fixed_lines.push(format!("{}if index < array.len() {{", indent));
                fixed_lines.push(format!("{}    {}", indent, trimmed));
                fixed_lines.push(format!("{}}} else {{", indent));
                fixed_lines.push(format!("{}    return Err(SACAError::ExecuteError(\"Index out of bounds\".to_string()));", indent));
                fixed_lines.push(format!("{}}}", indent));
            } else {
                fixed_lines.push(line.to_string());
            }
        }
        
        Some(fixed_lines.join("\n"))
    }
    
    /// Add comprehensive null/option checks
    fn add_comprehensive_null_checks(&self, code: &str) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let mut fixed_lines = Vec::new();
        
        for line in lines {
            let trimmed = line.trim();
            
            // Look for potential None/Option access
            if trimmed.contains(".unwrap()") || trimmed.contains(".expect(") {
                let indent = " ".repeat(line.len() - trimmed.len());
                
                if trimmed.contains(".unwrap()") {
                    let safe_line = trimmed.replace(".unwrap()", "");
                    fixed_lines.push(format!("{}if let Some(value) = {} {{", indent, safe_line));
                    fixed_lines.push(format!("{}    value", indent));
                    fixed_lines.push(format!("{}}} else {{", indent));
                    fixed_lines.push(format!("{}    return Err(SACAError::ExecuteError(\"Unexpected None value\".to_string()));", indent));
                    fixed_lines.push(format!("{}}}", indent));
                } else {
                    fixed_lines.push(line.to_string());
                }
            } else {
                fixed_lines.push(line.to_string());
            }
        }
        
        Some(fixed_lines.join("\n"))
    }
    
    /// Fix syntax errors with basic heuristics
    fn fix_syntax_errors(&self, code: &str) -> Option<String> {
        let mut result = code.to_string();
        
        // Fix common syntax issues
        result = result.replace("fn  ", "fn ");
        result = result.replace("  {", " {");
        result = result.replace(",,", ",");
        result = result.replace(";;", ";");
        
        Some(result)
    }
}

/// Fix suggestion
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    pub description: String,
    pub fixed_code: String,
    pub confidence: f32,
}
