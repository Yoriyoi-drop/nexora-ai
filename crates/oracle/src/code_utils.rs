//! Code-specific Utilities dan Verifiers
//! 
//! Utilities untuk processing, parsing, dan verifikasi kode
//! berbagai bahasa pemrograman dengan fokus pada security,
//! efficiency, dan code quality.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;

/// Code tokenizer untuk berbagai bahasa
pub struct CodeTokenizer {
    vocab: HashMap<String, i32>,
    reverse_vocab: HashMap<i32, String>,
    special_tokens: SpecialTokens,
}

impl CodeTokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            special_tokens: SpecialTokens::new(),
        };
        
        // Initialize basic vocabulary
        tokenizer.initialize_basic_vocab();
        tokenizer
    }
    
    /// Tokenize code string
    pub fn tokenize(&self, code: &str) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        
        // Add special tokens
        tokens.push(self.special_tokens.bos);
        
        // Simple tokenization by splitting on whitespace and punctuation
        let cleaned_code = self.clean_code(code);
        let words = self.extract_tokens(&cleaned_code);
        
        for word in words {
            if let Some(&token_id) = self.vocab.get(&word) {
                tokens.push(token_id);
            } else {
                // Handle unknown tokens
                tokens.push(self.special_tokens.unk);
            }
        }
        
        tokens.push(self.special_tokens.eos);
        Ok(tokens)
    }
    
    /// Decode tokens back to code
    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        let mut words = Vec::new();
        
        for &token_id in tokens {
            if let Some(word) = self.reverse_vocab.get(&token_id) {
                words.push(word.clone());
            }
        }
        
        Ok(words.join(" "))
    }
    
    /// Clean code for tokenization
    fn clean_code(&self, code: &str) -> String {
        // Remove comments (simplified)
        let no_comments = self.remove_comments(code);
        
        // Normalize whitespace
        let normalized = no_comments
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        
        normalized
    }
    
    /// Remove comments from code
    fn remove_comments(&self, code: &str) -> String {
        let mut result = String::new();
        let mut in_string = false;
        let mut in_comment = false;
        let mut chars = code.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                '"' if !in_comment => {
                    in_string = !in_string;
                    result.push(ch);
                },
                '/' if !in_string && !in_comment => {
                    if let Some(&next_ch) = chars.peek() {
                        if next_ch == '/' {
                            // Single line comment
                            in_comment = true;
                            chars.next(); // Consume the second '/'
                            continue;
                        } else if next_ch == '*' {
                            // Multi-line comment
                            in_comment = true;
                            chars.next(); // Consume the '*'
                            continue;
                        }
                    }
                    result.push(ch);
                },
                '\n' if in_comment => {
                    in_comment = false;
                    result.push(ch);
                },
                '*' if in_comment => {
                    if let Some(&next_ch) = chars.peek() {
                        if next_ch == '/' {
                            // End of multi-line comment
                            in_comment = false;
                            chars.next(); // Consume the '/'
                            continue;
                        }
                    }
                },
                _ if !in_comment => {
                    result.push(ch);
                },
                _ => {} // Skip characters in comments
            }
        }
        
        result
    }
    
    /// Extract tokens from cleaned code
    fn extract_tokens(&self, code: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut chars = code.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                ch if ch.is_alphanumeric() || ch == '_' => {
                    current_token.push(ch);
                },
                ch if ch.is_whitespace() => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                },
                '(' | ')' | '{' | '}' | '[' | ']' | ';' | ',' | '.' | '=' | '+' | '-' | '*' | '/' | '%' | '<' | '>' | '&' | '|' | '!' | '?' | ':' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                    tokens.push(ch.to_string());
                },
                _ => {
                    current_token.push(ch);
                }
            }
        }
        
        if !current_token.is_empty() {
            tokens.push(current_token);
        }
        
        tokens
    }
    
    /// Initialize basic vocabulary
    fn initialize_basic_vocab(&mut self) {
        let basic_tokens = vec![
            // Keywords
            "def", "function", "class", "if", "else", "elif", "for", "while", "return", "break", "continue", "pass", "import", "from", "as", "try", "except", "finally", "with", "yield", "lambda", "global", "nonlocal", "assert", "del", "raise",
            // Types
            "int", "float", "str", "bool", "list", "dict", "tuple", "set", "None", "True", "False",
            // Operators
            "+", "-", "*", "/", "%", "**", "//", "=", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not", "in", "is",
            // Built-in functions
            "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted", "sum", "min", "max", "abs", "round", "int", "float", "str", "bool", "list", "dict", "tuple", "set", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
            // Common identifiers
            "self", "cls", "super", "init", "main", "args", "kwargs", "config", "data", "result", "output", "input", "value", "key", "item", "index", "count", "length", "size", "width", "height", "x", "y", "z", "i", "j", "k", "n", "m", "a", "b", "c", "d", "e", "f", "g", "h", "temp", "tmp", "buf", "buffer",
            // Literals
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000", "\"", "'", "''", "\"\"\"", "'''",
            // Punctuation
            "(", ")", "{", "}", "[", "]", ";", ",", ".", ":", "->", "::", "...",
        ];
        
        for (i, token) in basic_tokens.iter().enumerate() {
            self.vocab.insert(token.to_string(), i as i32);
            self.reverse_vocab.insert(i as i32, token.to_string());
        }
        
        // Add special tokens
        let special_start = self.vocab.len();
        self.vocab.insert(self.special_tokens.bos.to_string(), special_start as i32);
        self.vocab.insert(self.special_tokens.eos.to_string(), (special_start + 1) as i32);
        self.vocab.insert(self.special_tokens.pad.to_string(), (special_start + 2) as i32);
        self.vocab.insert(self.special_tokens.unk.to_string(), (special_start + 3) as i32);
        
        self.reverse_vocab.insert(special_start as i32, self.special_tokens.bos.to_string());
        self.reverse_vocab.insert((special_start + 1) as i32, self.special_tokens.eos.to_string());
        self.reverse_vocab.insert((special_start + 2) as i32, self.special_tokens.pad.to_string());
        self.reverse_vocab.insert((special_start + 3) as i32, self.special_tokens.unk.to_string());
    }
}

/// Special tokens for code tokenization
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: i32,  // Beginning of sequence
    pub eos: i32,  // End of sequence
    pub pad: i32,  // Padding
    pub unk: i32,  // Unknown
}

impl SpecialTokens {
    pub fn new() -> Self {
        Self {
            bos: 50000,
            eos: 50001,
            pad: 50002,
            unk: 50003,
        }
    }
}

/// Code parser untuk berbagai bahasa
pub struct CodeParser {
    language: String,
}

impl CodeParser {
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_lowercase(),
        }
    }
    
    /// Parse code into AST-like structure
    pub fn parse(&self, code: &str) -> Result<CodeAst> {
        match self.language.as_str() {
            "python" => self.parse_python(code),
            "javascript" | "js" => self.parse_javascript(code),
            "java" => self.parse_java(code),
            "cpp" | "c++" | "c" => self.parse_cpp(code),
            _ => self.parse_generic(code),
        }
    }
    
    /// Parse Python code
    fn parse_python(&self, code: &str) -> Result<CodeAst> {
        let mut ast = CodeAst::new("python".to_string());
        
        // Extract functions
        let function_regex = Regex::new(r"def\s+(\w+)\s*\(([^)]*)\)\s*:")
            .map_err(|e| anyhow::anyhow!("Failed to create Python function regex: {}", e))?;
        for cap in function_regex.captures_iter(code) {
            let name = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture function name"))?.as_str().to_string();
            let params = cap.get(2).ok_or_else(|| anyhow::anyhow!("Failed to capture function parameters"))?.as_str().to_string();
            ast.add_function(CodeFunction {
                name,
                parameters: params,
                line_start: 0, // Would need proper line tracking
                line_end: 0,
                complexity: 1.0,
            });
        }
        
        // Extract classes
        let class_regex = Regex::new(r"class\s+(\w+)\s*(\([^)]*\))?\s*:")
            .map_err(|e| anyhow::anyhow!("Failed to create Python class regex: {}", e))?;
        for cap in class_regex.captures_iter(code) {
            let name = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture class name"))?.as_str().to_string();
            let inheritance = cap.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();
            ast.add_class(CodeClass {
                name,
                inheritance,
                line_start: 0,
                line_end: 0,
                methods: Vec::new(),
            });
        }
        
        Ok(ast)
    }
    
    /// Parse JavaScript code
    fn parse_javascript(&self, code: &str) -> Result<CodeAst> {
        let mut ast = CodeAst::new("javascript".to_string());
        
        // Extract functions
        let function_regex = Regex::new(r"function\s+(\w+)\s*\(([^)]*)\)\s*\{")
            .map_err(|e| anyhow::anyhow!("Failed to create JavaScript function regex: {}", e))?;
        for cap in function_regex.captures_iter(code) {
            let name = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture function name"))?.as_str().to_string();
            let params = cap.get(2).ok_or_else(|| anyhow::anyhow!("Failed to capture function parameters"))?.as_str().to_string();
            ast.add_function(CodeFunction {
                name,
                parameters: params,
                line_start: 0,
                line_end: 0,
                complexity: 1.0,
            });
        }
        
        // Extract classes (ES6)
        let class_regex = Regex::new(r"class\s+(\w+)\s*(\s+extends\s+(\w+))?\s*\{")
            .map_err(|e| anyhow::anyhow!("Failed to create JavaScript class regex: {}", e))?;
        for cap in class_regex.captures_iter(code) {
            let name = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture class name"))?.as_str().to_string();
            let inheritance = cap.get(3).map(|m| m.as_str().to_string()).unwrap_or_default();
            ast.add_class(CodeClass {
                name,
                inheritance,
                line_start: 0,
                line_end: 0,
                methods: Vec::new(),
            });
        }
        
        Ok(ast)
    }
    
    /// Parse Java code
    fn parse_java(&self, code: &str) -> Result<CodeAst> {
        let mut ast = CodeAst::new("java".to_string());
        
        // Extract methods
        let method_regex = Regex::new(r"(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{")
            .map_err(|e| anyhow::anyhow!("Failed to create Java method regex: {}", e))?;
        for cap in method_regex.captures_iter(code) {
            let return_type = cap.get(4).ok_or_else(|| anyhow::anyhow!("Failed to capture return type"))?.as_str().to_string();
            let name = cap.get(5).ok_or_else(|| anyhow::anyhow!("Failed to capture method name"))?.as_str().to_string();
            let params = cap.get(6).ok_or_else(|| anyhow::anyhow!("Failed to capture method parameters"))?.as_str().to_string();
            ast.add_function(CodeFunction {
                name: format!("{}: {}", name, return_type),
                parameters: params,
                line_start: 0,
                line_end: 0,
                complexity: 1.0,
            });
        }
        
        // Extract classes
        let class_regex = Regex::new(r"(public\s+)?class\s+(\w+)\s*(\s+extends\s+(\w+))?\s*\{")
            .map_err(|e| anyhow::anyhow!("Failed to create Java class regex: {}", e))?;
        for cap in class_regex.captures_iter(code) {
            let name = cap.get(2).ok_or_else(|| anyhow::anyhow!("Failed to capture class name"))?.as_str().to_string();
            let inheritance = cap.get(4).map(|m| m.as_str().to_string()).unwrap_or_default();
            ast.add_class(CodeClass {
                name,
                inheritance,
                line_start: 0,
                line_end: 0,
                methods: Vec::new(),
            });
        }
        
        Ok(ast)
    }
    
    /// Parse C++ code
    fn parse_cpp(&self, code: &str) -> Result<CodeAst> {
        let mut ast = CodeAst::new("cpp".to_string());
        
        // Extract functions
        let function_regex = Regex::new(r"(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{")
            .map_err(|e| anyhow::anyhow!("Failed to create C++ function regex: {}", e))?;
        for cap in function_regex.captures_iter(code) {
            let return_type = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture return type"))?.as_str().to_string();
            let name = cap.get(2).ok_or_else(|| anyhow::anyhow!("Failed to capture function name"))?.as_str().to_string();
            let params = cap.get(3).ok_or_else(|| anyhow::anyhow!("Failed to capture function parameters"))?.as_str().to_string();
            ast.add_function(CodeFunction {
                name: format!("{}: {}", name, return_type),
                parameters: params,
                line_start: 0,
                line_end: 0,
                complexity: 1.0,
            });
        }
        
        // Extract classes
        let class_regex = Regex::new(r"class\s+(\w+)\s*(\s*:\s*(public|private|protected)\s+(\w+))?\s*\{")
            .map_err(|e| anyhow::anyhow!("Failed to create C++ class regex: {}", e))?;
        for cap in class_regex.captures_iter(code) {
            let name = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture class name"))?.as_str().to_string();
            let inheritance = cap.get(4).map(|m| m.as_str().to_string()).unwrap_or_default();
            ast.add_class(CodeClass {
                name,
                inheritance,
                line_start: 0,
                line_end: 0,
                methods: Vec::new(),
            });
        }
        
        Ok(ast)
    }
    
    /// Parse generic code (fallback)
    fn parse_generic(&self, code: &str) -> Result<CodeAst> {
        let mut ast = CodeAst::new(self.language.clone());
        
        // Simple function detection
        let function_regex = Regex::new(r"(\w+)\s*\(")
            .map_err(|e| anyhow::anyhow!("Failed to create generic function regex: {}", e))?;
        for cap in function_regex.captures_iter(code) {
            let name = cap.get(1).ok_or_else(|| anyhow::anyhow!("Failed to capture function name"))?.as_str().to_string();
            ast.add_function(CodeFunction {
                name,
                parameters: String::new(),
                line_start: 0,
                line_end: 0,
                complexity: 1.0,
            });
        }
        
        Ok(ast)
    }
}

/// Abstract Syntax Tree representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAst {
    pub language: String,
    pub functions: Vec<CodeFunction>,
    pub classes: Vec<CodeClass>,
    pub imports: Vec<String>,
    pub complexity: f32,
}

impl CodeAst {
    pub fn new(language: String) -> Self {
        Self {
            language,
            functions: Vec::new(),
            classes: Vec::new(),
            imports: Vec::new(),
            complexity: 1.0,
        }
    }
    
    pub fn add_function(&mut self, function: CodeFunction) {
        self.functions.push(function);
    }
    
    pub fn add_class(&mut self, class: CodeClass) {
        self.classes.push(class);
    }
    
    pub fn add_import(&mut self, import: String) {
        self.imports.push(import);
    }
    
    pub fn calculate_complexity(&mut self) {
        let mut total_complexity = 0.0;
        
        for function in &self.functions {
            total_complexity += function.complexity;
        }
        
        for class in &self.classes {
            total_complexity += class.methods.len() as f32;
        }
        
        self.complexity = total_complexity;
    }
}

/// Code function representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFunction {
    pub name: String,
    pub parameters: String,
    pub line_start: usize,
    pub line_end: usize,
    pub complexity: f32,
}

/// Code class representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeClass {
    pub name: String,
    pub inheritance: String,
    pub line_start: usize,
    pub line_end: usize,
    pub methods: Vec<String>,
}

/// Code formatter
pub struct CodeFormatter {
    language: String,
    indent_size: usize,
    use_tabs: bool,
}

impl CodeFormatter {
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_lowercase(),
            indent_size: 4,
            use_tabs: false,
        }
    }
    
    /// Format code according to language conventions
    pub fn format(&self, code: &str) -> Result<String> {
        match self.language.as_str() {
            "python" => self.format_python(code),
            "javascript" | "js" => self.format_javascript(code),
            "java" => self.format_java(code),
            "cpp" | "c++" | "c" => self.format_cpp(code),
            _ => Ok(code.to_string()), // Return as-is for unknown languages
        }
    }
    
    /// Format Python code
    fn format_python(&self, code: &str) -> Result<String> {
        let mut formatted = String::new();
        let mut indent_level: usize = 0;
        let mut lines = code.lines();
        
        while let Some(line) = lines.next() {
            let trimmed = line.trim();
            
            // Handle dedentation
            if trimmed.starts_with("elif ") || trimmed.starts_with("else:") || trimmed.starts_with("except ") || trimmed.starts_with("finally:") {
                indent_level = indent_level.saturating_sub(1);
            }
            
            // Add indentation
            let indent = if self.use_tabs {
                "\t".repeat(indent_level)
            } else {
                " ".repeat(indent_level * self.indent_size)
            };
            
            formatted.push_str(&format!("{}{}\n", indent, trimmed));
            
            // Handle indentation
            if trimmed.starts_with("def ") || trimmed.starts_with("class ") || trimmed.starts_with("if ") || trimmed.starts_with("elif ") || trimmed.starts_with("else:") || trimmed.starts_with("for ") || trimmed.starts_with("while ") || trimmed.starts_with("try:") || trimmed.starts_with("except ") || trimmed.starts_with("finally:") || trimmed.starts_with("with ") {
                if !trimmed.starts_with("elif ") && !trimmed.starts_with("else:") && !trimmed.starts_with("except ") && !trimmed.starts_with("finally:") {
                    indent_level += 1;
                }
            }
        }
        
        Ok(formatted)
    }
    
    /// Format JavaScript code
    fn format_javascript(&self, code: &str) -> Result<String> {
        let mut formatted = String::new();
        let mut indent_level: usize = 0;
        let mut lines = code.lines();
        
        while let Some(line) = lines.next() {
            let trimmed = line.trim();
            
            // Handle dedentation
            if trimmed.starts_with("}") || trimmed.starts_with("]") || trimmed.starts_with(")") {
                indent_level = indent_level.saturating_sub(1);
            }
            
            // Add indentation
            let indent = if self.use_tabs {
                "\t".repeat(indent_level)
            } else {
                " ".repeat(indent_level * self.indent_size)
            };
            
            formatted.push_str(&format!("{}{}\n", indent, trimmed));
            
            // Handle indentation
            if trimmed.ends_with("{") || trimmed.ends_with("[") || trimmed.ends_with("(") {
                indent_level += 1;
            }
        }
        
        Ok(formatted)
    }
    
    /// Format Java code
    fn format_java(&self, code: &str) -> Result<String> {
        // Similar to JavaScript but with Java-specific rules
        self.format_javascript(code)
    }
    
    /// Format C++ code
    fn format_cpp(&self, code: &str) -> Result<String> {
        // Similar to JavaScript but with C++-specific rules
        self.format_javascript(code)
    }
}

/// Code metrics calculator
pub struct CodeMetrics {
    language: String,
}

impl CodeMetrics {
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_lowercase(),
        }
    }
    
    /// Calculate comprehensive code metrics
    pub fn calculate_metrics(&self, code: &str) -> Result<CodeMetricsResult> {
        let lines = code.lines().count();
        let non_empty_lines = code.lines().filter(|line| !line.trim().is_empty()).count();
        let comment_lines = self.count_comment_lines(code);
        let code_lines = non_empty_lines - comment_lines;
        
        let functions = self.count_functions(code);
        let classes = self.count_classes(code);
        let complexity = self.calculate_cyclomatic_complexity(code);
        let maintainability = self.calculate_maintainability_index(code_lines, complexity);
        
        Ok(CodeMetricsResult {
            language: self.language.clone(),
            total_lines: lines,
            code_lines,
            comment_lines,
            empty_lines: lines - non_empty_lines,
            functions,
            classes,
            complexity,
            maintainability_index: maintainability,
            technical_debt: self.estimate_technical_debt(complexity, code_lines),
        })
    }
    
    /// Count comment lines
    fn count_comment_lines(&self, code: &str) -> usize {
        let mut count = 0;
        let mut in_block_comment = false;
        
        for line in code.lines() {
            let trimmed = line.trim();
            
            if self.language == "python" {
                if trimmed.starts_with("#") {
                    count += 1;
                }
            } else {
                // Handle C-style comments
                if trimmed.starts_with("/*") {
                    in_block_comment = true;
                }
                if trimmed.starts_with("//") || in_block_comment {
                    count += 1;
                }
                if trimmed.ends_with("*/") {
                    in_block_comment = false;
                }
            }
        }
        
        count
    }
    
    /// Count functions
    fn count_functions(&self, code: &str) -> usize {
        let mut count = 0;
        
        match self.language.as_str() {
            "python" => {
                let regex = Regex::new(r"def\s+\w+\s*\(").expect("valid regex");
                count = regex.find_iter(code).count();
            },
            "javascript" | "js" => {
                let regex = Regex::new(r"function\s+\w+\s*\(").expect("valid regex");
                count = regex.find_iter(code).count();
            },
            "java" => {
                let regex = Regex::new(r"\w+\s+\w+\s*\(").expect("valid regex");
                count = regex.find_iter(code).count();
            },
            "cpp" | "c++" | "c" => {
                let regex = Regex::new(r"\w+\s+\w+\s*\(").expect("valid regex");
                count = regex.find_iter(code).count();
            },
            _ => {}
        }
        
        count
    }
    
    /// Count classes
    fn count_classes(&self, code: &str) -> usize {
        let mut count = 0;
        
        match self.language.as_str() {
            "python" => {
                let regex = Regex::new(r"class\s+\w+").expect("valid regex");
                count = regex.find_iter(code).count();
            },
            "javascript" | "js" | "java" | "cpp" | "c++" => {
                let regex = Regex::new(r"class\s+\w+").expect("valid regex");
                count = regex.find_iter(code).count();
            },
            _ => {}
        }
        
        count
    }
    
    /// Calculate cyclomatic complexity
    fn calculate_cyclomatic_complexity(&self, code: &str) -> f32 {
        let mut complexity = 1.0; // Base complexity
        
        let decision_patterns = vec![
            r"\bif\b", r"\belif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", 
            r"\bcase\b", r"\bcatch\b", r"\?\s*[^:]*:", r"\|\|", r"\&\&"
        ];
        
        for pattern in decision_patterns {
            let regex = Regex::new(pattern)
                .unwrap_or_else(|_| Regex::new("").expect("empty regex is always valid"));
            complexity += regex.find_iter(code).count() as f32;
        }
        
        complexity
    }
    
    /// Calculate maintainability index
    fn calculate_maintainability_index(&self, code_lines: usize, complexity: f32) -> f32 {
        if code_lines == 0 {
            return 100.0;
        }
        
        let volume = code_lines as f32 * (2.0_f32).log2();
        let difficulty = complexity / 2.0;
        let effort = difficulty * volume;
        let time = effort / 18.0;
        let bugs = volume / 3000.0;
        
        let mi = 171.0 - 5.2 * time.ln() - 0.23 * effort.ln() + 16.2 * bugs.ln();
        mi.max(0.0)
    }
    
    /// Estimate technical debt
    fn estimate_technical_debt(&self, complexity: f32, code_lines: usize) -> f32 {
        // Simplified technical debt estimation
        let complexity_ratio = complexity / code_lines.max(1) as f32;
        let debt_hours = complexity_ratio * code_lines as f32 * 0.1; // 0.1 hours per complexity point per line
        
        debt_hours
    }
}

/// Code metrics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMetricsResult {
    pub language: String,
    pub total_lines: usize,
    pub code_lines: usize,
    pub comment_lines: usize,
    pub empty_lines: usize,
    pub functions: usize,
    pub classes: usize,
    pub complexity: f32,
    pub maintainability_index: f32,
    pub technical_debt: f32,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Detect programming language from code
    pub fn detect_language(code: &str) -> String {
        if code.contains("def ") && code.contains(":") {
            "python".to_string()
        } else if code.contains("function ") || code.contains("=>") {
            "javascript".to_string()
        } else if code.contains("public class ") || code.contains("import java.") {
            "java".to_string()
        } else if code.contains("#include") || code.contains("std::") {
            "cpp".to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    /// Extract imports from code
    pub fn extract_imports(code: &str, language: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        match language {
            "python" => {
                let regex = Regex::new(r"(?:import|from)\s+([^\s\n]+)")
                    .unwrap_or_else(|_| Regex::new("").expect("empty regex is always valid"));
                for cap in regex.captures_iter(code) {
                    if let Some(import) = cap.get(1) {
                        imports.push(import.as_str().to_string());
                    }
                }
            },
            "javascript" => {
                let regex = Regex::new(r"(?:import|require)\s+([^\s\n;]+)")
                    .unwrap_or_else(|_| Regex::new("").expect("empty regex is always valid"));
                for cap in regex.captures_iter(code) {
                    if let Some(import) = cap.get(1) {
                        imports.push(import.as_str().to_string());
                    }
                }
            },
            "java" => {
                let regex = Regex::new(r"import\s+([^\s\n;]+)")
                    .unwrap_or_else(|_| Regex::new("").expect("empty regex is always valid"));
                for cap in regex.captures_iter(code) {
                    if let Some(import) = cap.get(1) {
                        imports.push(import.as_str().to_string());
                    }
                }
            },
            _ => {}
        }
        
        imports
    }
    
    /// Validate code syntax (basic)
    pub fn validate_syntax(code: &str, language: &str) -> SyntaxValidationResult {
        let mut result = SyntaxValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        };
        
        // Basic syntax checks
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
        
        if brace_count != 0 {
            result.is_valid = false;
            result.errors.push(format!("Unmatched braces: {}", brace_count));
        }
        
        if paren_count != 0 {
            result.is_valid = false;
            result.errors.push(format!("Unmatched parentheses: {}", paren_count));
        }
        
        if bracket_count != 0 {
            result.is_valid = false;
            result.errors.push(format!("Unmatched brackets: {}", bracket_count));
        }
        
        // Language-specific checks
        match language {
            "python" => {
                if code.contains("    ") && code.contains("\t") {
                    result.warnings.push("Mixed tabs and spaces detected".to_string());
                }
            },
            _ => {}
        }
        
        result
    }
    
    #[derive(Debug, Clone)]
    pub struct SyntaxValidationResult {
        pub is_valid: bool,
        pub errors: Vec<String>,
        pub warnings: Vec<String>,
    }
}
