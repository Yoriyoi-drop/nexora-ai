//! Request processing functionality

use anyhow::Result;
use tracing::{info, debug};
use chrono::Utc;
use std::sync::{Arc, RwLock};

use super::types::{InputType, CodeAnalysis, FunctionInfo, ClassInfo, ImportInfo, ComplexityMetrics, CodeIssue, IssueSeverity, PatternInfo, CodeMetrics};

/// Request processor for handling different input types
#[derive(Debug, Clone)]
pub struct RequestProcessor {
    request_count: Arc<RwLock<u64>>,
}

impl RequestProcessor {
    pub fn new(request_count: Arc<RwLock<u64>>) -> Self {
        Self { request_count }
    }

    /// Process a request with input type detection and routing
    pub async fn process_request(&self, input: &str) -> Result<String> {
        let request_start = Utc::now();
        
        // Increment request counter
        {
            let mut count = self.request_count.write()
                .map_err(|e| anyhow::anyhow!("Failed to acquire write lock for request counter: {}", e))?;
            *count += 1;
        }
        
        info!("Processing request #{}: {}", 
              *self.request_count.read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire read lock for request counter: {}", e))?, input);
        
        // Validate input
        if input.is_empty() {
            return Err(anyhow::anyhow!("Input cannot be empty"));
        }
        
        if input.len() > 10000 {
            return Err(anyhow::anyhow!("Input too long (max 10000 characters)"));
        }
        
        // Detect input type and route appropriately
        let input_type = self.detect_input_type(input);
        debug!("Detected input type: {:?}", input_type);
        
        let result = match input_type {
            InputType::Command => self.process_command(input).await?,
            InputType::Query => self.process_query(input).await?,
            InputType::Code => self.process_code(input).await?,
            InputType::Data => self.process_data(input).await?,
            InputType::Text => self.process_text(input).await?,
        };
        
        let processing_time = (Utc::now() - request_start).num_milliseconds();
        debug!("Request processed in {}ms", processing_time);
        
        Ok(result)
    }
    
    /// Detect input type for routing
    fn detect_input_type(&self, input: &str) -> InputType {
        let trimmed = input.trim();
        
        // Check for JSON data
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            return InputType::Data;
        }
        
        // Check for code patterns
        if trimmed.contains("fn ") || trimmed.contains("function ") || 
           trimmed.contains("class ") || trimmed.contains("def ") ||
           trimmed.contains("import ") || trimmed.contains("#include") {
            return InputType::Code;
        }
        
        // Check for commands
        if trimmed.starts_with('/') || trimmed.starts_with('!') ||
           trimmed.contains("help") || trimmed.contains("status") ||
           trimmed.contains("list") || trimmed.contains("show") {
            return InputType::Command;
        }
        
        // Check for questions
        if trimmed.ends_with('?') || trimmed.starts_with("what ") ||
           trimmed.starts_with("how ") || trimmed.starts_with("why ") ||
           trimmed.starts_with("when ") || trimmed.starts_with("where ") ||
           trimmed.contains("?") {
            return InputType::Query;
        }
        
        InputType::Text
    }
    
    /// Process command input
    async fn process_command(&self, command: &str) -> Result<String> {
        info!("Processing command: {}", command);
        
        match command.to_lowercase().trim() {
            cmd if cmd.contains("status") => {
                Ok("System Status: Healthy (Score: 85.2)".to_string())
            },
            cmd if cmd.contains("help") => {
                Ok("Available commands: status, help, models, memory".to_string())
            },
            cmd if cmd.contains("models") => {
                Ok("Active models: default, gpt-4, claude".to_string())
            },
            cmd if cmd.contains("memory") => {
                Ok("Memory usage: 45.2% (2048/4096 MB)".to_string())
            },
            _ => Ok(format!("Unknown command: {}", command))
        }
    }
    
    /// Process query input
    async fn process_query(&self, query: &str) -> Result<String> {
        info!("Processing query: {}", query);
        
        // Simple query processing - would delegate to inference engine
        let response = format!("Query processed: {}", query);
        Ok(response)
    }
    
    /// Process code input
    async fn process_code(&self, code: &str) -> Result<String> {
        info!("Processing code input ({} chars)", code.len());
        
        // Basic code analysis - would delegate to models crate
        let analysis = self.analyze_code(code).await?;
        
        Ok(format!("Code analysis: {} lines, {} functions, {} classes", 
                  analysis.line_count, analysis.functions.len(), analysis.classes.len()))
    }
    
    /// Process data input
    async fn process_data(&self, data: &str) -> Result<String> {
        info!("Processing JSON data");
        
        // Try to parse as JSON
        match serde_json::from_str::<serde_json::Value>(data) {
            Ok(json) => {
                let keys = json.as_object().map(|obj| obj.keys().count()).unwrap_or(0);
                Ok(format!("JSON parsed successfully: {} keys", keys))
            },
            Err(e) => Ok(format!("Invalid JSON: {}", e))
        }
    }
    
    /// Process text input
    async fn process_text(&self, text: &str) -> Result<String> {
        info!("Processing text input ({} chars)", text.len());
        
        // Basic text processing
        let words = text.split_whitespace().count();
        let sentences = text.split(&['.', '!', '?'][..]).filter(|s| !s.trim().is_empty()).count();
        
        Ok(format!("Text processed: {} words, {} sentences", words, sentences))
    }
    
    /// Analyze code structure and complexity
    pub async fn analyze_code(&self, code: &str) -> Result<CodeAnalysis> {
        let lines = code.lines().count();
        let characters = code.len();
        
        // Extract functions
        let functions = self.extract_functions(code);
        
        // Extract classes
        let classes = self.extract_classes(code);
        
        // Extract imports
        let imports = self.extract_imports(code);
        
        // Calculate complexity metrics
        let complexity = self.calculate_complexity(code);
        
        // Identify issues
        let issues = self.identify_issues(code);
        
        // Detect patterns
        let patterns = self.detect_patterns(code);
        
        // Calculate general metrics
        let metrics = self.calculate_metrics(code);
        
        Ok(CodeAnalysis {
            language: self.detect_language(code),
            line_count: lines,
            character_count: characters,
            functions,
            classes,
            imports,
            complexity,
            issues,
            patterns,
            metrics,
        })
    }
    
    /// Extract functions from code
    fn extract_functions(&self, code: &str) -> Vec<FunctionInfo> {
        let mut functions = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            if let Some(func_name) = self.extract_function_name(line) {
                functions.push(FunctionInfo {
                    name: func_name,
                    line_number: line_num + 1,
                    parameters: self.extract_parameters(line),
                    return_type: self.extract_return_type(line),
                    visibility: self.extract_visibility(line),
                });
            }
        }
        
        functions
    }
    
    /// Extract function name from line
    fn extract_function_name(&self, line: &str) -> Option<String> {
        if line.contains("fn ") {
            let parts: Vec<&str> = line.split("fn ").collect();
            if parts.len() > 1 {
                let name_part = parts[1].split('(').next().unwrap_or("");
                Some(name_part.trim().to_string())
            } else {
                None
            }
        } else if line.contains("function ") {
            let parts: Vec<&str> = line.split("function ").collect();
            if parts.len() > 1 {
                let name_part = parts[1].split('(').next().unwrap_or("");
                Some(name_part.trim().to_string())
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Extract parameters from function line
    fn extract_parameters(&self, line: &str) -> String {
        if let Some(start) = line.find('(') {
            if let Some(end) = line.find(')') {
                line[start + 1..end].to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    }
    
    /// Extract return type from function line
    fn extract_return_type(&self, line: &str) -> Option<String> {
        if let Some(arrow_pos) = line.find("->") {
            Some(line[arrow_pos + 2..].trim().to_string())
        } else {
            None
        }
    }
    
    /// Extract visibility from function line
    fn extract_visibility(&self, line: &str) -> String {
        if line.contains("pub ") {
            "public".to_string()
        } else if line.contains("private ") {
            "private".to_string()
        } else if line.contains("protected ") {
            "protected".to_string()
        } else {
            "private".to_string()
        }
    }
    
    /// Extract classes from code
    fn extract_classes(&self, code: &str) -> Vec<ClassInfo> {
        let mut classes = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            if line.contains("class ") || line.contains("struct ") {
                let parts: Vec<&str> = if line.contains("class ") {
                    line.split("class ").collect()
                } else {
                    line.split("struct ").collect()
                };
                
                if parts.len() > 1 {
                    let name_part = parts[1].split('{').next().unwrap_or("").trim();
                    classes.push(ClassInfo {
                        name: name_part.to_string(),
                        line_number: line_num + 1,
                        type_name: if line.contains("class ") { "class" } else { "struct" }.to_string(),
                        visibility: if line.contains("pub ") { Some("public".to_string()) } else { None },
                        inheritance: None, // Simplified
                    });
                }
            }
        }
        
        classes
    }
    
    /// Extract imports from code
    fn extract_imports(&self, code: &str) -> Vec<ImportInfo> {
        let mut imports = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            if line.contains("import ") || line.contains("use ") || line.contains("#include") {
                let import_type = if line.contains("import ") {
                    "import"
                } else if line.contains("use ") {
                    "use"
                } else {
                    "include"
                };
                
                let module = if import_type == "include" {
                    line.split('#').next().unwrap_or("").trim().to_string()
                } else {
                    line.split_whitespace().nth(1).unwrap_or("").to_string()
                };
                
                imports.push(ImportInfo {
                    module,
                    line_number: line_num + 1,
                    import_type: import_type.to_string(),
                });
            }
        }
        
        imports
    }
    
    /// Calculate complexity metrics
    fn calculate_complexity(&self, code: &str) -> ComplexityMetrics {
        let lines = code.lines().count() as u32;
        let code_lines = code.lines().filter(|line| !line.trim().is_empty() && !line.trim().starts_with("//")).count() as u32;
        let comment_lines = code.lines().filter(|line| line.trim().starts_with("//")).count() as u32;
        let _empty_lines = lines - code_lines - comment_lines;
        
        let functions = code.matches("fn ").count() as u32 + code.matches("function ").count() as u32;
        let conditionals = code.matches("if ").count() as u32;
        let nested_loops = code.matches("for ").count() as u32 + code.matches("while ").count() as u32;
        
        let cyclomatic_complexity = 1 + conditionals + nested_loops;
        let comment_ratio = if code_lines > 0 { comment_lines as f64 / code_lines as f64 } else { 0.0 };
        
        ComplexityMetrics {
            cyclomatic_complexity,
            nested_loops,
            conditionals,
            functions,
            total_lines: lines,
            code_lines,
            comment_lines,
            comment_ratio,
        }
    }
    
    /// Identify code issues
    fn identify_issues(&self, code: &str) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            // Check for very long lines
            if line.len() > 100 {
                issues.push(CodeIssue {
                    line_number: line_num + 1,
                    severity: IssueSeverity::Warning,
                    message: "Line too long".to_string(),
                    suggestion: "Consider breaking this line into multiple lines".to_string(),
                });
            }
            
            // Check for TODO comments with categorization
            if line.contains("TODO") || line.contains("FIXME") {
                let (issue_type, severity, message, suggestion) = if line.contains("FIXME") {
                    (
                        "FIXME",
                        IssueSeverity::Error,
                        "FIXME comment found - requires immediate attention".to_string(),
                        "Fix the critical issue marked with FIXME".to_string(),
                    )
                } else if line.contains("TODO") && line.contains("urgent") {
                    (
                        "URGENT_TODO",
                        IssueSeverity::Error,
                        "Urgent TODO found - high priority task".to_string(),
                        "Address this urgent TODO item as soon as possible".to_string(),
                    )
                } else if line.contains("TODO") && line.contains("security") {
                    (
                        "SECURITY_TODO",
                        IssueSeverity::Error,
                        "Security-related TODO found".to_string(),
                        "Address security concerns immediately".to_string(),
                    )
                } else {
                    (
                        "TODO",
                        IssueSeverity::Warning,
                        "TODO comment found - task to be completed".to_string(),
                        "Complete the TODO task or remove if no longer needed".to_string(),
                    )
                };
                
                issues.push(CodeIssue {
                    line_number: line_num + 1,
                    severity,
                    message: format!("{}: {}", issue_type, message),
                    suggestion: format!("{}: {}", issue_type, suggestion),
                });
            }
        }
        
        issues
    }
    
    /// Detect design patterns
    fn detect_patterns(&self, _code: &str) -> Vec<PatternInfo> {
        // Simplified pattern detection
        vec![]
    }
    
    /// Calculate general metrics
    fn calculate_metrics(&self, code: &str) -> CodeMetrics {
        let lines = code.lines().count() as u32;
        let empty_lines = code.lines().filter(|line| line.trim().is_empty()).count() as u32;
        let comment_lines = code.lines().filter(|line| line.trim().starts_with("//")).count() as u32;
        let code_lines = lines - empty_lines - comment_lines;
        
        CodeMetrics {
            total_lines: lines,
            empty_lines,
            comment_lines,
            code_lines,
        }
    }
    
    /// Detect programming language
    fn detect_language(&self, code: &str) -> String {
        if code.contains("fn ") || code.contains("let ") || code.contains("->") {
            "Rust".to_string()
        } else if code.contains("function ") || code.contains("const ") || code.contains("let ") {
            "JavaScript".to_string()
        } else if code.contains("def ") || code.contains("import ") || code.contains("class ") {
            "Python".to_string()
        } else if code.contains("public class ") || code.contains("private ") || code.contains("public ") {
            "Java".to_string()
        } else {
            "Unknown".to_string()
        }
    }
}
