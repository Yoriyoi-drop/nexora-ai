//! Controller Analysis - Advanced analysis functions untuk code, memory, dan reasoning

use crate::error::CoreResult;

/// Advanced analysis implementations
pub struct AnalysisProcessor;

impl AnalysisProcessor {
    /// Advanced code analysis dengan comprehensive language detection
    pub async fn analyze_code_request_advanced(input: &str) -> CoreResult<String> {
        let mut analysis = Vec::new();
        
        let language = Self::detect_programming_language(input);
        analysis.push(format!("🔍 Detected Language: {}", language));
        
        let complexity = Self::calculate_code_complexity(input);
        analysis.push(format!("📊 Complexity Score: {:.2}/10", complexity));
        
        let functions = Self::extract_functions(input, &language);
        analysis.push(format!("🔧 Functions Found: {}", functions.len()));
        for func in functions.iter().take(5) {
            analysis.push(format!("  - {}", func));
        }
        
        let security_issues = Self::analyze_security_issues(input);
        if !security_issues.is_empty() {
            analysis.push("⚠️  Security Issues:".to_string());
            for issue in security_issues.iter().take(3) {
                analysis.push(format!("  - {}", issue));
            }
        }
        
        let perf_suggestions = Self::analyze_performance(input);
        if !perf_suggestions.is_empty() {
            analysis.push("⚡ Performance Suggestions:".to_string());
            for suggestion in perf_suggestions.iter().take(3) {
                analysis.push(format!("  - {}", suggestion));
            }
        }
        
        Ok(analysis.join("\n"))
    }

    /// Detect programming language dengan advanced patterns
    fn detect_programming_language(code: &str) -> String {
        let language_patterns = [
            ("rust", vec!["fn ", "let ", "mut ", "impl ", "struct ", "enum ", "use ", "mod "]),
            ("python", vec!["def ", "import ", "from ", "class ", "self.", "elif ", "try:"]),
            ("javascript", vec!["function ", "const ", "let ", "=>", "async ", "await ", "export "]),
            ("typescript", vec!["interface ", "type ", "as ", "enum ", "declare ", "namespace "]),
            ("java", vec!["public class", "private ", "public ", "static ", "void ", "import "]),
            ("cpp", vec!["#include", "std::", "cout", "cin", "namespace", "class ", "template "]),
            ("go", vec!["func ", "package ", "import ", "go ", "chan ", "defer "]),
        ];
        
        let mut scores = std::collections::HashMap::new();
        
        for (lang, patterns) in &language_patterns {
            let mut score = 0;
            for pattern in patterns {
                score += code.matches(pattern).count() * pattern.len();
            }
            scores.insert(lang.to_string(), score);
        }
        
        scores
            .into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(lang, _)| if lang == "rust" { "Rust".to_string() } else { lang })
            .unwrap_or_else(|| "Unknown".to_string())
    }

    /// Calculate code complexity score
    fn calculate_code_complexity(code: &str) -> f32 {
        let mut complexity = 0.0;
        
        complexity += code.lines().count() as f32 * 0.1;
        
        let nesting_depth = Self::calculate_nesting_depth(code);
        complexity += nesting_depth as f32 * 0.5;
        
        let control_flow = code.matches("if").count() + code.matches("while").count() + code.matches("for").count();
        complexity += control_flow as f32 * 0.3;
        
        let function_calls = code.matches("(").count();
        complexity += function_calls as f32 * 0.05;
        
        complexity.min(10.0)
    }

    /// Calculate maximum nesting depth
    fn calculate_nesting_depth(code: &str) -> usize {
        let mut max_depth: i32 = 0;
        let mut current_depth: i32 = 0;
        
        for line in code.lines() {
            let line = line.trim();
            if line.contains('{') || line.contains('[') || line.contains('(') {
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            }
            if line.contains('}') || line.contains(']') || line.contains(')') {
                current_depth = current_depth.saturating_sub(1);
            }
        }
        
        max_depth as usize
    }

    /// Extract functions dari code
    fn extract_functions(code: &str, language: &str) -> Vec<String> {
        let mut functions = Vec::new();
        
        match language {
            "Rust" => {
                for line in code.lines() {
                    if let Some(func) = line.trim().strip_prefix("fn ") {
                        if let Some(name) = func.split('(').next() {
                            functions.push(name.trim().to_string());
                        }
                    }
                }
            }
            "Python" => {
                for line in code.lines() {
                    if let Some(func) = line.trim().strip_prefix("def ") {
                        if let Some(name) = func.split('(').next() {
                            functions.push(name.trim().to_string());
                        }
                    }
                }
            }
            "JavaScript" | "TypeScript" => {
                for line in code.lines() {
                    if line.contains("function ") {
                        if let Some(func) = line.split("function ").nth(1) {
                            if let Some(name) = func.split('(').next() {
                                functions.push(name.trim().to_string());
                            }
                        }
                    } else if line.contains("=") && line.contains("=>") {
                        if let Some(name) = line.split('=').next() {
                            functions.push(name.trim().to_string());
                        }
                    }
                }
            }
            _ => {}
        }
        
        functions
    }

    /// Analyze security issues
    fn analyze_security_issues(code: &str) -> Vec<String> {
        let mut issues = Vec::new();
        
        let security_patterns = [
            ("eval(", "Use of eval() can lead to code injection"),
            ("exec(", "Use of exec() can lead to code injection"),
            ("system(", "Direct system call detected"),
            ("shell_exec(", "Shell execution detected"),
            ("password", "Hardcoded password detected"),
            ("secret", "Hardcoded secret detected"),
            ("token", "Hardcoded token detected"),
            ("sql", "Potential SQL injection risk"),
        ];
        
        for (pattern, message) in &security_patterns {
            if code.to_lowercase().contains(pattern) {
                issues.push(message.to_string());
            }
        }
        
        issues
    }

    /// Analyze performance issues
    fn analyze_performance(code: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        let performance_patterns = [
            ("for.*for", "Nested loops detected - consider optimization"),
            ("while.*while", "Nested while loops detected"),
            ("O(n^2)", "Quadratic complexity detected"),
            ("malloc", "Manual memory allocation - consider RAII"),
            ("new.*delete", "Manual memory management detected"),
        ];
        
        for (pattern, message) in &performance_patterns {
            if code.contains(pattern.split('*').next().unwrap_or(pattern)) {
                suggestions.push(message.to_string());
            }
        }
        
        suggestions
    }

    /// Advanced memory analysis
    pub async fn process_memory_request_advanced(input: &str) -> CoreResult<String> {
        let mut analysis = Vec::new();
        
        let memory_type = Self::categorize_memory_request(input);
        analysis.push(format!("🧠 Memory Type: {}", memory_type));
        
        let priority = Self::assess_memory_priority(input);
        analysis.push(format!("📊 Priority Level: {}", priority));
        
        let retention = Self::calculate_retention_score(input);
        analysis.push(format!("⏱️  Retention Score: {:.2}/10", retention));
        
        let associations = Self::extract_memory_associations(input);
        if !associations.is_empty() {
            analysis.push("🔗 Associated Concepts:".to_string());
            for assoc in associations.iter().take(5) {
                analysis.push(format!("  - {}", assoc));
            }
        }
        
        let optimizations = Self::suggest_memory_optimizations(input);
        if !optimizations.is_empty() {
            analysis.push("⚡ Optimization Suggestions:".to_string());
            for opt in optimizations.iter().take(3) {
                analysis.push(format!("  - {}", opt));
            }
        }
        
        Ok(analysis.join("\n"))
    }

    fn categorize_memory_request(input: &str) -> String {
        let input_lower = input.to_lowercase();
        
        if input_lower.contains("remember") || input_lower.contains("store") {
            "Storage Request".to_string()
        } else if input_lower.contains("recall") || input_lower.contains("retrieve") || input_lower.contains("get") {
            "Retrieval Request".to_string()
        } else if input_lower.contains("forget") || input_lower.contains("delete") {
            "Deletion Request".to_string()
        } else if input_lower.contains("update") || input_lower.contains("modify") {
            "Update Request".to_string()
        } else if input_lower.contains("search") || input_lower.contains("find") {
            "Search Request".to_string()
        } else {
            "General Memory Operation".to_string()
        }
    }

    fn assess_memory_priority(input: &str) -> String {
        let input_lower = input.to_lowercase();
        
        if input_lower.contains("urgent") || input_lower.contains("important") {
            "High".to_string()
        } else if input_lower.contains("normal") || input_lower.contains("regular") {
            "Medium".to_string()
        } else if input_lower.contains("low") || input_lower.contains("background") {
            "Low".to_string()
        } else {
            let complexity = input.len() as f32;
            if complexity > 500.0 {
                "High".to_string()
            } else if complexity > 100.0 {
                "Medium".to_string()
            } else {
                "Low".to_string()
            }
        }
    }

    fn calculate_retention_score(input: &str) -> f32 {
        let mut score = 5.0;
        
        if input.contains("struct") || input.contains("class") || input.contains("function") {
            score += 2.0;
        }
        
        if input.chars().any(|c| c.is_ascii_digit()) {
            score += 1.0;
        }
        
        let unique_words: std::collections::HashSet<_> = input.split_whitespace().collect();
        score += (unique_words.len() as f32 / 10.0).min(2.0);
        
        score.min(10.0)
    }

    fn extract_memory_associations(input: &str) -> Vec<String> {
        let mut associations = Vec::new();
        
        let association_keywords = [
            ("code", "Programming"),
            ("function", "Software Development"),
            ("data", "Information Management"),
            ("algorithm", "Computer Science"),
            ("design", "Architecture"),
            ("test", "Quality Assurance"),
            ("debug", "Troubleshooting"),
            ("optimize", "Performance"),
        ];
        
        let input_lower = input.to_lowercase();
        for (keyword, association) in &association_keywords {
            if input_lower.contains(keyword) {
                associations.push(association.to_string());
            }
        }
        
        associations
    }

    fn suggest_memory_optimizations(input: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if input.len() > 1000 {
            suggestions.push("Consider compressing large memory entries".to_string());
        }
        
        if input.contains("repetitive") {
            suggestions.push("Use deduplication for repetitive content".to_string());
        }
        
        if input.contains("frequent") {
            suggestions.push("Consider caching frequently accessed memories".to_string());
        }
        
        suggestions
    }

    /// Advanced logical reasoning analysis
    pub async fn apply_logical_reasoning_advanced(input: &str) -> CoreResult<String> {
        let mut analysis = Vec::new();
        
        let reasoning_type = Self::detect_reasoning_type(input);
        analysis.push(format!("🧠 Reasoning Type: {}", reasoning_type));
        
        let structure = Self::analyze_logical_structure(input);
        analysis.push(format!("🏗️  Logical Structure: {}", structure));
        
        let inferences = Self::extract_inference_chain(input);
        if !inferences.is_empty() {
            analysis.push("🔗 Inference Chain:".to_string());
            for (i, inference) in inferences.iter().enumerate().take(5) {
                analysis.push(format!("  {}. {}", i + 1, inference));
            }
        }
        
        let fallacies = Self::detect_logical_fallacies(input);
        if !fallacies.is_empty() {
            analysis.push("⚠️  Potential Fallacies:".to_string());
            for fallacy in fallacies.iter().take(3) {
                analysis.push(format!("  - {}", fallacy));
            }
        }
        
        let confidence = Self::calculate_reasoning_confidence(input);
        analysis.push(format!("📊 Reasoning Confidence: {:.2}%", confidence * 100.0));
        
        let improvements = Self::suggest_reasoning_improvements(input);
        if !improvements.is_empty() {
            analysis.push("💡 Reasoning Improvements:".to_string());
            for improvement in improvements.iter().take(3) {
                analysis.push(format!("  - {}", improvement));
            }
        }
        
        Ok(analysis.join("\n"))
    }

    fn detect_reasoning_type(input: &str) -> String {
        let input_lower = input.to_lowercase();
        
        if input_lower.contains("if") && input_lower.contains("then") {
            "Conditional Reasoning".to_string()
        } else if input_lower.contains("because") || input_lower.contains("therefore") || input_lower.contains("thus") {
            "Causal Reasoning".to_string()
        } else if input_lower.contains("all") || input_lower.contains("every") || input_lower.contains("always") {
            "Deductive Reasoning".to_string()
        } else if input_lower.contains("some") || input_lower.contains("might") || input_lower.contains("could") {
            "Inductive Reasoning".to_string()
        } else if input_lower.contains("similar") || input_lower.contains("like") || input_lower.contains("as") {
            "Analogical Reasoning".to_string()
        } else if input_lower.contains("better") || input_lower.contains("worse") || input_lower.contains("best") {
            "Comparative Reasoning".to_string()
        } else {
            "General Reasoning".to_string()
        }
    }

    fn analyze_logical_structure(input: &str) -> String {
        let premise_count = input.matches("because").count() + input.matches("since").count();
        let conclusion_count = input.matches("therefore").count() + input.matches("thus").count();
        let condition_count = input.matches("if").count();
        
        if premise_count > 0 && conclusion_count > 0 {
            format!("Structured Argument ({} premises, {} conclusions)", premise_count, conclusion_count)
        } else if condition_count > 0 {
            format!("Conditional Structure ({} conditions)", condition_count)
        } else if premise_count > 0 {
            format!("Premise-based ({} premises)", premise_count)
        } else {
            "Unstructured Reasoning".to_string()
        }
    }

    fn extract_inference_chain(input: &str) -> Vec<String> {
        let mut inferences = Vec::new();
        
        let sentences: Vec<&str> = input.split(&['.', '!', '?'][..]).collect();
        
        for (i, sentence) in sentences.iter().enumerate() {
            let sentence = sentence.trim();
            if !sentence.is_empty() {
                if sentence.contains("therefore") || sentence.contains("thus") {
                    inferences.push(format!("Conclusion: {}", sentence));
                } else if sentence.contains("because") || sentence.contains("since") {
                    inferences.push(format!("Premise: {}", sentence));
                } else if sentence.contains("if") {
                    inferences.push(format!("Condition: {}", sentence));
                } else if i > 0 {
                    inferences.push(format!("Statement: {}", sentence));
                }
            }
        }
        
        inferences
    }

    fn detect_logical_fallacies(input: &str) -> Vec<String> {
        let mut fallacies = Vec::new();
        let input_lower = input.to_lowercase();
        
        let fallacy_patterns = [
            ("always", "Hasty Generalization - using 'always' without sufficient evidence"),
            ("never", "Hasty Generalization - using 'never' without sufficient evidence"),
            ("obviously", "Appeal to Common Sense - assuming something is obvious"),
            ("everyone knows", "Bandwagon Fallacy - appealing to popular opinion"),
            ("straw man", "Straw Man Argument - misrepresenting opponent's position"),
            ("false dilemma", "False Dilemma - presenting only two options when more exist"),
            ("slippery slope", "Slippery Slope - claiming one event will lead to extreme consequences"),
        ];
        
        for (pattern, description) in &fallacy_patterns {
            if input_lower.contains(pattern) {
                fallacies.push(description.to_string());
            }
        }
        
        fallacies
    }

    fn calculate_reasoning_confidence(input: &str) -> f32 {
        let mut confidence: f32 = 0.5;
        
        if input.contains("because") || input.contains("therefore") {
            confidence += 0.2;
        }
        
        if input.contains("if") && input.contains("then") {
            confidence += 0.2;
        }
        
        if input.to_lowercase().contains("always") || input.to_lowercase().contains("never") {
            confidence -= 0.1;
        }
        
        if input.len() > 200 {
            confidence += 0.1;
        }
        
        confidence.min(1.0).max(0.0)
    }

    fn suggest_reasoning_improvements(input: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if !input.contains("because") && !input.contains("therefore") {
            suggestions.push("Add explicit reasoning connectors (because, therefore)".to_string());
        }
        
        if input.to_lowercase().contains("always") || input.to_lowercase().contains("never") {
            suggestions.push("Consider using more nuanced language instead of absolutes".to_string());
        }
        
        if input.len() < 100 {
            suggestions.push("Provide more detailed explanations for your reasoning".to_string());
        }
        
        suggestions
    }
}
