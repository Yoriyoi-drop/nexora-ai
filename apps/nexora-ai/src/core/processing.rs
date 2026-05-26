//! Request processing functionality

use crate::error::{NexoraError, NexoraResult};
use chrono::Utc;
use nexora_foundation::shared::{
    base_model::{InputData, NxrInput},
    model_identity::NxrModelId,
    model_registry::NxrModelRegistry,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info};

use super::types::{
    ClassInfo, CodeAnalysis, CodeIssue, CodeMetrics, ComplexityMetrics, FunctionInfo, ImportInfo,
    InputType, IssueSeverity, PatternInfo,
};

/// Request processor for handling different input types
#[derive(Debug, Clone)]
pub struct RequestProcessor {
    request_count: Arc<AtomicU64>,
    registry: Arc<NxrModelRegistry>,
    active_model_id: NxrModelId,
}

impl RequestProcessor {
    pub fn new(
        request_count: Arc<AtomicU64>,
        registry: Arc<NxrModelRegistry>,
        active_model_id: NxrModelId,
    ) -> Self {
        Self {
            request_count,
            registry,
            active_model_id,
        }
    }

    /// Process a request with input type detection and routing
    pub async fn process_request(&self, input: &str) -> NexoraResult<String> {
        let request_start = Utc::now();

        // Validate input
        if input.is_empty() {
            return Err(NexoraError::validation("input", "Input cannot be empty"));
        }

        if input.len() > 10000 {
            return Err(NexoraError::validation(
                "input",
                "Input too long (max 10000 characters)",
            ));
        }

        // Detect input type and route appropriately
        let input_type = self.detect_input_type(input);

        // Increment request counter (atomic operation - no lock needed)
        let current_count = self.request_count.fetch_add(1, Ordering::Relaxed) + 1;

        info!(
            "🔄 Processing request #{}: input_len={}, type={:?}",
            current_count,
            input.len(),
            input_type
        );
        debug!("Detected input type: {:?}", input_type);

        let result = match input_type {
            InputType::Command => self.process_command(input).await?,
            InputType::Query => self.process_query(input).await?,
            InputType::Code => self.process_code(input).await?,
            InputType::Data => self.process_data(input).await?,
            InputType::Text => self.process_text(input).await?,
            InputType::Internal => self.process_text(input).await?,
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
        if trimmed.contains("fn ")
            || trimmed.contains("function ")
            || trimmed.contains("class ")
            || trimmed.contains("def ")
            || trimmed.contains("import ")
            || trimmed.contains("#include")
        {
            return InputType::Code;
        }

        // Check for commands
        if trimmed.starts_with('/')
            || trimmed.starts_with('!')
            || trimmed.contains("help")
            || trimmed.contains("status")
            || trimmed.contains("list")
            || trimmed.contains("show")
        {
            return InputType::Command;
        }

        // Check for questions
        if trimmed.ends_with('?')
            || trimmed.starts_with("what ")
            || trimmed.starts_with("how ")
            || trimmed.starts_with("why ")
            || trimmed.starts_with("when ")
            || trimmed.starts_with("where ")
            || trimmed.contains("?")
        {
            return InputType::Query;
        }

        InputType::Text
    }

    /// Process command input via model inference
    async fn process_command(&self, command: &str) -> NexoraResult<String> {
        info!("Processing command: {}", command);

        let response = self.model_generate(command).await?;
        Ok(response)
    }

    async fn model_generate(&self, input: &str) -> NexoraResult<String> {
        let model = self
            .registry
            .get_model(&self.active_model_id)
            .await
            .map_err(|e| {
                NexoraError::model(format!(
                    "Model {} not available: {}",
                    self.active_model_id, e
                ))
            })?;

        let nxr_input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(input.to_string()),
            parameters: [
                ("max_tokens".to_string(), serde_json::json!(256)),
                ("temperature".to_string(), serde_json::json!(0.7)),
            ]
            .into(),
            metadata: std::collections::HashMap::new(),
        };

        let output = model.infer(&nxr_input).await.map_err(|e| {
            NexoraError::model(format!(
                "Inference failed for '{}': {}",
                self.active_model_id, e
            ))
        })?;

        match output.data {
            nexora_foundation::shared::base_model::OutputData::Text(text) => Ok(text),
            _ => Ok(format!("{:?}", output.data)),
        }
    }

    /// Process query input via model inference
    async fn process_query(&self, query: &str) -> NexoraResult<String> {
        info!("Processing query: {}", query);
        self.model_generate(query).await
    }

    /// Process code input
    async fn process_code(&self, code: &str) -> NexoraResult<String> {
        info!("Processing code input ({} chars)", code.len());

        // Basic code analysis - would delegate to models crate
        let analysis = self.analyze_code(code).await?;

        Ok(format!(
            "Code analysis: {} lines, {} functions, {} classes",
            analysis.line_count,
            analysis.functions.len(),
            analysis.classes.len()
        ))
    }

    /// Process data input
    async fn process_data(&self, data: &str) -> NexoraResult<String> {
        info!("Processing JSON data");

        // Try to parse as JSON
        if data.len() > 10_000_000 {
            return Ok("JSON input too large (max 10MB)".to_string());
        }
        match serde_json::from_str::<serde_json::Value>(data) {
            Ok(json) => {
                let keys = json.as_object().map(|obj| obj.keys().count()).unwrap_or(0);
                Ok(format!("JSON parsed successfully: {} keys", keys))
            }
            Err(e) => Ok(format!("Invalid JSON: {}", e)),
        }
    }

    /// Process text input
    async fn process_text(&self, text: &str) -> NexoraResult<String> {
        info!("Processing text input ({} chars)", text.len());

        // Basic text processing
        let words = text.split_whitespace().count();
        let sentences = text
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .count();

        Ok(format!(
            "Text processed: {} words, {} sentences",
            words, sentences
        ))
    }

    /// Analyze code structure and complexity
    pub async fn analyze_code(&self, code: &str) -> NexoraResult<CodeAnalysis> {
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
        let lines: Vec<&str> = code.lines().collect();
        let mut functions = Vec::with_capacity(lines.len());

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
        let lines: Vec<&str> = code.lines().collect();
        let mut classes = Vec::with_capacity(lines.len());

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
                        type_name: if line.contains("class ") {
                            "class"
                        } else {
                            "struct"
                        }
                        .to_string(),
                        visibility: if line.contains("pub ") {
                            Some("public".to_string())
                        } else {
                            None
                        },
                        inheritance: None, // Simplified
                    });
                }
            }
        }

        classes
    }

    /// Extract imports from code
    fn extract_imports(&self, code: &str) -> Vec<ImportInfo> {
        let lines: Vec<&str> = code.lines().collect();
        let mut imports = Vec::with_capacity(lines.len());

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
        let code_lines = code
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.trim().starts_with("//"))
            .count() as u32;
        let comment_lines = code
            .lines()
            .filter(|line| line.trim().starts_with("//"))
            .count() as u32;
        let _empty_lines = lines - code_lines - comment_lines;

        let functions =
            code.matches("fn ").count() as u32 + code.matches("function ").count() as u32;
        let conditionals = code.matches("if ").count() as u32;
        let nested_loops =
            code.matches("for ").count() as u32 + code.matches("while ").count() as u32;

        let cyclomatic_complexity = 1 + conditionals + nested_loops;
        let comment_ratio = if code_lines > 0 {
            comment_lines as f64 / code_lines as f64
        } else {
            0.0
        };

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
        let lines: Vec<&str> = code.lines().collect();
        let mut issues = Vec::with_capacity(lines.len());

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

            // Check for unfinished code markers with categorization
            if line.contains("UNFINISHED") || line.contains("FIX_ME") || line.contains("HACK") {
                let (issue_type, severity, message, suggestion) = if line.contains("FIX_ME") {
                    (
                        "FIX_ME",
                        IssueSeverity::Error,
                        "FIX_ME comment found - requires immediate attention".to_string(),
                        "Fix the critical issue marked with FIX_ME".to_string(),
                    )
                } else if line.contains("UNFINISHED") && line.contains("urgent") {
                    (
                        "URGENT_UNFINISHED",
                        IssueSeverity::Error,
                        "Urgent UNFINISHED found - high priority task".to_string(),
                        "Address this urgent unfinished item as soon as possible".to_string(),
                    )
                } else if line.contains("UNFINISHED") && line.contains("security") {
                    (
                        "SECURITY_UNFINISHED",
                        IssueSeverity::Error,
                        "Security-related UNFINISHED found".to_string(),
                        "Address security concerns immediately".to_string(),
                    )
                } else {
                    (
                        "UNFINISHED",
                        IssueSeverity::Warning,
                        "UNFINISHED comment found - task to be completed".to_string(),
                        "Complete the unfinished task or remove if no longer needed".to_string(),
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

    /// Detect design patterns in code
    fn detect_patterns(&self, code: &str) -> Vec<PatternInfo> {
        let mut patterns = Vec::new();

        // Singleton: static instance method with private constructor
        if code.contains("fn instance(")
            || (code.contains("static") && code.contains("OnceLock") && code.contains("new("))
        {
            patterns.push(PatternInfo {
                name: "Singleton".into(),
                confidence: 0.8,
                description:
                    "Static instance accessor pattern detected (OnceLock/static + constructor)"
                        .into(),
            });
        }

        // Factory: methods returning trait objects or named *Factory
        if code.contains("trait") && code.contains("Box<dyn")
            || code.contains("impl") && code.contains("Factory")
        {
            patterns.push(PatternInfo {
                name: "Factory".into(),
                confidence: 0.7,
                description: "Trait-based factory or Factory-named type detected".into(),
            });
        }

        // Builder: methods returning Self + build/into method
        if code.contains("fn ")
            && code.contains("-> Self")
            && code.contains("self")
            && (code.contains("fn build(")
                || code.contains("fn into(")
                || code.contains("fn finalize("))
        {
            patterns.push(PatternInfo {
                name: "Builder".into(),
                confidence: 0.8,
                description:
                    "Chainable builder pattern: methods return Self with terminal build/finalize"
                        .into(),
            });
        }

        // Observer/Event: subscribe/notify/emit pattern
        if (code.contains("fn subscribe") || code.contains("fn notify") || code.contains("fn emit"))
            || (code.contains("add_listener") || code.contains("on_event"))
        {
            patterns.push(PatternInfo {
                name: "Observer/Event".into(),
                confidence: 0.7,
                description: "Subscribe/notify or event listener pattern detected".into(),
            });
        }

        // Strategy: trait with multiple implementations or Box<dyn Fn>
        if (code.contains("Box<dyn") && (code.contains("Fn(") || code.contains("FnMut(")))
            || (code.contains("trait") && code.contains("impl") && code.contains("fn execute("))
        {
            patterns.push(PatternInfo {
                name: "Strategy".into(),
                confidence: 0.6,
                description: "Trait dispatch or Box<dyn Fn> strategy pattern detected".into(),
            });
        }

        // Adapter: wrapping foreign types
        if (code.contains("struct") && code.contains("Wrapper"))
            || (code.contains(".0") && code.contains("impl"))
        {
            patterns.push(PatternInfo {
                name: "Adapter/Wrapper".into(),
                confidence: 0.6,
                description: "Wrapper/newtype or adapter pattern detected (tuple struct wrapping)"
                    .into(),
            });
        }

        // Iterator: IntoIterator impl or next() method
        if code.contains("impl Iterator")
            || code.contains("impl IntoIterator")
            || (code.contains("fn next(") && code.contains("Option<"))
        {
            patterns.push(PatternInfo {
                name: "Iterator".into(),
                confidence: 0.9,
                description: "Iterator trait implementation or next() method detected".into(),
            });
        }

        // Decorator: wrapping function calls or middleware
        if code.contains("middleware") || (code.contains("wrap") && code.contains("fn")) {
            patterns.push(PatternInfo {
                name: "Decorator/Middleware".into(),
                confidence: 0.5,
                description: "Function wrapping or middleware pattern detected".into(),
            });
        }

        patterns
    }

    /// Calculate general metrics
    fn calculate_metrics(&self, code: &str) -> CodeMetrics {
        let lines = code.lines().count() as u32;
        let empty_lines = code.lines().filter(|line| line.trim().is_empty()).count() as u32;
        let comment_lines = code
            .lines()
            .filter(|line| line.trim().starts_with("//"))
            .count() as u32;
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
        let trimmed = code.trim();
        let lines: Vec<&str> = trimmed.lines().collect();
        let first_lines: Vec<&str> = lines.iter().take(10).map(|l| l.trim()).collect();
        let joined = first_lines.join(" ");

        if joined.contains("fn ")
            && (joined.contains("->") || joined.contains("let mut") || joined.contains("println!"))
        {
            "Rust".to_string()
        } else if joined.contains("use ") || joined.contains("mod ") || joined.contains("impl ") {
            "Rust".to_string()
        } else if joined.contains("def ") || joined.contains("import ") || joined.contains("from ")
        {
            "Python".to_string()
        } else if joined.contains("function ") || joined.contains("const ") || joined.contains("=>")
        {
            if joined.contains(": ") || joined.contains("interface ") {
                "TypeScript".to_string()
            } else {
                "JavaScript".to_string()
            }
        } else if joined.contains("public class ")
            || joined.contains("private ")
            || joined.contains("protected ")
        {
            "Java".to_string()
        } else if joined.contains("#include")
            || joined.contains("int main")
            || joined.contains("std::")
        {
            "C++".to_string()
        } else if joined.contains("package ") || joined.contains("func ") || joined.contains("go ")
        {
            "Go".to_string()
        } else if trimmed.starts_with("--")
            || joined.contains("local ")
            || joined.contains("function(")
        {
            "Lua".to_string()
        } else {
            "Unknown".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_foundation::shared::model_registry::NxrModelRegistry;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn make_processor() -> RequestProcessor {
        let request_count = Arc::new(AtomicU64::new(0));
        let registry = Arc::new(NxrModelRegistry::new());
        RequestProcessor::new(request_count, registry, NxrModelId::Omnis)
    }

    #[tokio::test]
    async fn test_process_request_validation() {
        let processor = make_processor();

        // Test empty input
        let result = processor.process_request("").await;
        assert!(result.is_err());

        // Test input too long
        let long_input = "a".repeat(10001);
        let result = processor.process_request(&long_input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_detect_input_type() {
        let processor = make_processor();

        // Test JSON data
        let input_type = processor.detect_input_type("{\"key\": \"value\"}");
        assert_eq!(input_type, InputType::Data);

        // Test code
        let input_type = processor.detect_input_type("fn main() {}");
        assert_eq!(input_type, InputType::Code);

        // Test command
        let input_type = processor.detect_input_type("/help");
        assert_eq!(input_type, InputType::Command);

        // Test query
        let input_type = processor.detect_input_type("What is Rust?");
        assert_eq!(input_type, InputType::Query);

        // Test text
        let input_type = processor.detect_input_type("Hello world");
        assert_eq!(input_type, InputType::Text);
    }

    #[tokio::test]
    async fn test_process_command_without_model() {
        let processor = make_processor();

        // Registry is empty — should return model-not-found error, not placeholder
        let result = processor.process_command("/status").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not available") || err.contains("NXR-OMNIS"));
    }

    #[tokio::test]
    async fn test_process_query_without_model() {
        let processor = make_processor();

        // Registry is empty — should return model-not-found error, not placeholder
        let result = processor.process_query("What is Rust?").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not available") || err.contains("NXR-OMNIS"));
    }

    #[tokio::test]
    async fn test_process_code() {
        let processor = make_processor();

        let code = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let result = processor.process_code(code).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.contains("Code analysis"));
        assert!(response.contains("lines"));
    }

    #[tokio::test]
    async fn test_process_data() {
        let processor = make_processor();

        // Test valid JSON
        let json = "{\"name\": \"test\", \"value\": 123}";
        let result = processor.process_data(json).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.contains("JSON parsed successfully"));

        // Test invalid JSON
        let invalid_json = "{invalid json}";
        let result = processor.process_data(invalid_json).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.contains("Invalid JSON"));
    }

    #[tokio::test]
    async fn test_process_text() {
        let processor = make_processor();

        let text = "Hello world! How are you today?";
        let result = processor.process_text(text).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.contains("Text processed"));
        assert!(response.contains("words"));
        assert!(response.contains("sentences"));
    }

    #[tokio::test]
    async fn test_analyze_code() {
        let processor = make_processor();

        let code = r#"
pub struct Test {
    value: i32,
}

impl Test {
    pub fn new(value: i32) -> Self {
        Self { value }
    }
    
    pub fn get_value(&self) -> i32 {
        self.value
    }
}
"#;

        let result = processor.analyze_code(code).await;
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.language, "Rust");
        assert!(analysis.line_count > 0);
        assert!(analysis.functions.len() > 0);
        assert!(analysis.classes.len() > 0);
        assert!(analysis.character_count > 0);
    }

    #[test]
    fn test_extract_functions() {
        let processor = make_processor();

        let code = r#"
fn main() {}
pub fn test() -> i32 { 42 }
private function helper() {}
"#;

        let functions = processor.extract_functions(code);
        assert_eq!(functions.len(), 3);

        let function_names: Vec<String> = functions.iter().map(|f| f.name.clone()).collect();
        assert!(function_names.contains(&"main".to_string()));
        assert!(function_names.contains(&"test".to_string()));
        assert!(function_names.contains(&"helper".to_string()));
    }

    #[test]
    fn test_extract_classes() {
        let processor = make_processor();

        let code = r#"
pub struct TestStruct {
    value: i32,
}

class TestClass {
    constructor() {}
}
"#;

        let classes = processor.extract_classes(code);
        assert_eq!(classes.len(), 2);

        let class_names: Vec<String> = classes.iter().map(|c| c.name.clone()).collect();
        assert!(class_names.contains(&"TestStruct".to_string()));
        assert!(class_names.contains(&"TestClass".to_string()));
    }

    #[test]
    fn test_extract_imports() {
        let processor = make_processor();

        let code = r#"
use std::collections::HashMap;
import React from 'react';
#include <stdio.h>
"#;

        let imports = processor.extract_imports(code);
        assert_eq!(imports.len(), 3);

        let import_types: Vec<String> = imports.iter().map(|i| i.import_type.clone()).collect();
        assert!(import_types.contains(&"use".to_string()));
        assert!(import_types.contains(&"import".to_string()));
        assert!(import_types.contains(&"include".to_string()));
    }

    #[test]
    fn test_calculate_complexity() {
        let processor = make_processor();

        let code = r#"
fn test() {
    if condition {
        for i in 0..10 {
            // nested loop
            for j in 0..5 {
                println!("{}", i * j);
            }
        }
    }
}
// This is a comment
"#;

        let complexity = processor.calculate_complexity(code);
        assert!(complexity.cyclomatic_complexity > 0);
        assert!(complexity.nested_loops > 0);
        assert!(complexity.conditionals > 0);
        assert!(complexity.functions > 0);
        assert!(complexity.comment_lines > 0);
        assert!(complexity.code_lines > 0);
    }

    #[test]
    fn test_detect_language() {
        let processor = make_processor();

        let rust_code = "fn main() -> Result<(), Box<dyn Error>> { Ok(()) }";
        assert_eq!(processor.detect_language(rust_code), "Rust");

        let python_code = "def main():\n    print('Hello, world!')";
        assert_eq!(processor.detect_language(python_code), "Python");

        let js_code = "function main() { console.log('Hello, world!'); }";
        assert_eq!(processor.detect_language(js_code), "JavaScript");

        let java_code = "public class Main { public static void main(String[] args) {} }";
        assert_eq!(processor.detect_language(java_code), "Java");

        let unknown_code = "some random text";
        assert_eq!(processor.detect_language(unknown_code), "Unknown");
    }

    #[test]
    fn test_identify_issues() {
        let processor = make_processor();

        let code = r#"
fn test() {
    // This is a very long line that exceeds the 100 character limit and should trigger a warning
    println!("Hello");
    // UNFINISHED: implement this function
    // FIX_ME: critical issue
    // UNFINISHED: urgent task
    // UNFINISHED: security vulnerability
}
"#;

        let issues = processor.identify_issues(code);
        assert!(issues.len() > 0);

        let issue_types: Vec<IssueSeverity> = issues.iter().map(|i| i.severity.clone()).collect();
        assert!(issue_types.contains(&IssueSeverity::Warning));
        assert!(issue_types.contains(&IssueSeverity::Error));

        let issue_messages: Vec<String> = issues.iter().map(|i| i.message.clone()).collect();
        assert!(issue_messages
            .iter()
            .any(|msg| msg.contains("Line too long")));
        assert!(issue_messages.iter().any(|msg| msg.contains("UNFINISHED")));
        assert!(issue_messages.iter().any(|msg| msg.contains("FIX_ME")));
    }
}
