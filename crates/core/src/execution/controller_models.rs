//! Controller Models - Model processing implementations untuk specialist models

use crate::error::CoreResult;
use crate::types::ModelId;

/// Model processing implementations
pub struct ModelProcessor;

impl ModelProcessor {
    pub async fn process_coding(input: &str) -> CoreResult<String> {
        let mut result = String::from("CODE ANALYSIS:\n");
        
        // Direct language detection
        let rust_score = input.matches("fn ").count() + input.matches("let ").count() + input.matches("mut ").count();
        let python_score = input.matches("def ").count() + input.matches("import ").count();
        let js_score = input.matches("function ").count() + input.matches("const ").count();
        
        let language = if rust_score > python_score && rust_score > js_score {
            "Rust"
        } else if python_score > js_score {
            "Python"
        } else if js_score > 0 {
            "JavaScript"
        } else {
            "Unknown"
        };
        
        result.push_str(&format!("Language: {}\n", language));
        result.push_str(&format!("Lines: {}\n", input.lines().count()));
        
        // Direct function extraction
        let mut functions = Vec::new();
        for line in input.lines() {
            let line = line.trim();
            if let Some(func_part) = line.strip_prefix("fn ") {
                if let Some(name) = func_part.split('(').next() {
                    functions.push(name.trim());
                }
            } else if let Some(func_part) = line.strip_prefix("def ") {
                if let Some(name) = func_part.split('(').next() {
                    functions.push(name.trim());
                }
            }
        }
        
        result.push_str(&format!("Functions: {}\n", functions.len()));
        for func in functions.iter().take(5) {
            result.push_str(&format!("  - {}\n", func));
        }
        
        // Direct security check
        let mut security_issues = Vec::with_capacity(2);
        if input.to_lowercase().contains("eval(") {
            security_issues.push("eval() detected - potential security risk");
        }
        if input.to_lowercase().contains("password") {
            security_issues.push("hardcoded password detected");
        }
        
        if !security_issues.is_empty() {
            result.push_str("SECURITY ISSUES:\n");
            for issue in security_issues {
                result.push_str(&format!("  - {}\n", issue));
            }
        }
        
        Ok(result)
    }

    pub async fn process_memory(input: &str) -> CoreResult<String> {
        let mut result = String::from("MEMORY PROCESSING:\n");
        
        // Direct categorization
        let input_lower = input.to_lowercase();
        let category = if input_lower.contains("remember") || input_lower.contains("store") {
            "Storage Request"
        } else if input_lower.contains("recall") || input_lower.contains("get") {
            "Retrieval Request"
        } else if input_lower.contains("forget") || input_lower.contains("delete") {
            "Deletion Request"
        } else {
            "General Memory Operation"
        };
        
        result.push_str(&format!("Category: {}\n", category));
        result.push_str(&format!("Input Length: {} chars\n", input.len()));
        
        // Direct word count
        let word_count = input.split_whitespace().count();
        result.push_str(&format!("Word Count: {}\n", word_count));
        
        // Direct keyword extraction
        let mut keywords = Vec::with_capacity(5);
        let important_words = ["important", "urgent", "critical", "normal", "low"];
        for word in important_words {
            if input_lower.contains(word) {
                keywords.push(word);
            }
        }
        
        if !keywords.is_empty() {
            result.push_str("Keywords: ");
            result.push_str(&keywords.join(", "));
            result.push_str("\n");
        }
        
        Ok(result)
    }

    pub async fn process_logic(input: &str) -> CoreResult<String> {
        let mut result = String::from("LOGICAL REASONING:\n");
        
        // Direct reasoning type detection
        let input_lower = input.to_lowercase();
        let reasoning_type = if input_lower.contains("if") && input_lower.contains("then") {
            "Conditional Reasoning"
        } else if input_lower.contains("because") || input_lower.contains("therefore") {
            "Causal Reasoning"
        } else if input_lower.contains("all") || input_lower.contains("every") {
            "Deductive Reasoning"
        } else if input_lower.contains("some") || input_lower.contains("might") {
            "Inductive Reasoning"
        } else {
            "General Reasoning"
        };
        
        result.push_str(&format!("Type: {}\n", reasoning_type));
        
        // Direct structure analysis
        let premise_count = input_lower.matches("because").count() + input_lower.matches("since").count();
        let conclusion_count = input_lower.matches("therefore").count() + input_lower.matches("thus").count();
        let condition_count = input_lower.matches("if").count();
        
        result.push_str(&format!("Premises: {}\n", premise_count));
        result.push_str(&format!("Conclusions: {}\n", conclusion_count));
        result.push_str(&format!("Conditions: {}\n", condition_count));
        
        // Direct fallacy detection
        let mut fallacies = Vec::with_capacity(3);
        if input_lower.contains("always") {
            fallacies.push("Hasty Generalization (always)");
        }
        if input_lower.contains("never") {
            fallacies.push("Hasty Generalization (never)");
        }
        if input_lower.contains("obviously") {
            fallacies.push("Appeal to Common Sense");
        }
        
        if !fallacies.is_empty() {
            result.push_str("Potential Fallacies:\n");
            for fallacy in fallacies {
                result.push_str(&format!("  - {}\n", fallacy));
            }
        }
        
        Ok(result)
    }

    pub async fn process_planner(input: &str) -> CoreResult<String> {
        let mut result = String::from("PLANNING:\n");
        
        // Direct step breakdown
        let words: Vec<&str> = input.split_whitespace().collect();
        let mut steps = Vec::with_capacity(8);
        
        for (i, word) in words.iter().enumerate().take(8) {
            steps.push(format!("Step {}: Process '{}'", i + 1, word));
        }
        
        for step in steps {
            result.push_str(&format!("{}\n", step));
        }
        
        // Direct priority assessment
        let input_lower = input.to_lowercase();
        let priority = if input_lower.contains("urgent") || input_lower.contains("critical") {
            "HIGH"
        } else if input_lower.contains("normal") || input_lower.contains("regular") {
            "MEDIUM"
        } else {
            "LOW"
        };
        
        result.push_str(&format!("Priority: {}\n", priority));
        result.push_str(&format!("Complexity: {} words\n", words.len()));
        
        Ok(result)
    }

    pub async fn process_ranking(input: &str) -> CoreResult<String> {
        let mut result = String::from("RANKING:\n");
        
        // Direct word frequency analysis
        let words: Vec<&str> = input.split_whitespace().collect();
        let mut word_counts = std::collections::HashMap::new();
        
        for word in &words {
            *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
        }
        
        // Convert to vec and sort
        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        
        result.push_str("Word Frequency Ranking:\n");
        for (word, count) in sorted_words.iter().take(10) {
            result.push_str(&format!("  {}: {}\n", word, count));
        }
        
        // Direct length-based ranking
        let mut long_words: Vec<_> = words.iter().filter(|w| w.len() > 5).collect();
        long_words.sort_by(|a, b| b.len().cmp(&a.len()));
        
        if !long_words.is_empty() {
            result.push_str("Longest Words:\n");
            for word in long_words.iter().take(5) {
                result.push_str(&format!("  {} ({} chars)\n", word, word.len()));
            }
        }
        
        Ok(result)
    }

    pub async fn process_retrieval(input: &str) -> CoreResult<String> {
        let mut result = String::from("INFORMATION RETRIEVAL:\n");
        
        // Direct keyword extraction
        let words: Vec<&str> = input.split_whitespace().collect();
        let mut keywords = Vec::with_capacity(10);
        
        for word in words.iter().take(10) {
            let word_lower = word.to_lowercase();
            if word.len() > 3 && !word_lower.contains("the") && !word_lower.contains("and") && !word_lower.contains("for") {
                keywords.push(word.to_string());
            }
        }
        
        result.push_str("Extracted Keywords:\n");
        for keyword in keywords.iter().take(8) {
            result.push_str(&format!("  - {}\n", keyword));
        }
        
        // Direct content analysis
        let input_lower = input.to_lowercase();
        let content_type = if input_lower.contains("code") || input_lower.contains("function") {
            "Programming Content"
        } else if input_lower.contains("data") || input_lower.contains("information") {
            "Data Content"
        } else if input_lower.contains("help") || input_lower.contains("question") {
            "Help Request"
        } else {
            "General Content"
        };
        
        result.push_str(&format!("Content Type: {}\n", content_type));
        result.push_str(&format!("Total Words: {}\n", words.len()));
        
        Ok(result)
    }

    pub async fn process_validator(input: &str) -> CoreResult<String> {
        let mut result = String::from("VALIDATION:\n");
        
        // Direct input checks
        let mut issues = Vec::with_capacity(5);
        
        if input.is_empty() {
            issues.push("Input is empty");
        }
        
        if input.len() > 10000 {
            issues.push("Input too long (>10000 chars)");
        }
        
        if input.chars().any(|c| c.is_control()) {
            issues.push("Contains control characters");
        }
        
        // Direct language validation
        let input_lower = input.to_lowercase();
        if input_lower.contains("select") && input_lower.contains("from") {
            issues.push("Potential SQL injection pattern");
        }
        
        if input_lower.contains("<script>") {
            issues.push("Potential XSS pattern");
        }
        
        if issues.is_empty() {
            result.push_str("Status: PASSED\n");
            result.push_str("No issues detected\n");
        } else {
            result.push_str("Status: FAILED\n");
            result.push_str("Issues found:\n");
            for issue in issues {
                result.push_str(&format!("  - {}\n", issue));
            }
        }
        
        result.push_str(&format!("Input Length: {} chars\n", input.len()));
        result.push_str(&format!("Word Count: {}\n", input.split_whitespace().count()));
        
        Ok(result)
    }

    pub async fn process_personality(input: &str) -> CoreResult<String> {
        let mut result = String::from("PERSONALITY RESPONSE:\n");
        
        // Direct tone detection
        let input_lower = input.to_lowercase();
        let tone = if input_lower.contains("help") || input_lower.contains("please") {
            "Helpful and Supportive"
        } else if input_lower.contains("question") || input_lower.contains("?") {
            "Inquisitive and Analytical"
        } else if input_lower.contains("urgent") || input_lower.contains("quick") {
            "Efficient and Direct"
        } else if input_lower.contains("thank") || input_lower.contains("appreciate") {
            "Grateful and Positive"
        } else {
            "Neutral and Professional"
        };
        
        result.push_str(&format!("Detected Tone: {}\n", tone));
        
        // Direct response generation
        let response = match tone {
            "Helpful and Supportive" => "I'm here to help! Let me assist you with your request.",
            "Inquisitive and Analytical" => "That's an interesting question. Let me analyze this carefully.",
            "Efficient and Direct" => "I'll provide a direct and efficient response.",
            "Grateful and Positive" => "Thank you for your input! I appreciate your approach.",
            _ => "I'll process this in a professional manner.",
        };
        
        result.push_str(&format!("Response: {}\n", response));
        result.push_str(&format!("Input Analysis: {} chars, {} words\n", input.len(), input.split_whitespace().count()));
        
        Ok(result)
    }

    pub async fn process_optimizer(input: &str) -> CoreResult<String> {
        let mut result = String::from("OPTIMIZATION:\n");
        
        // Direct optimization analysis
        let mut suggestions = Vec::with_capacity(4);
        
        if input.len() > 500 {
            suggestions.push("Consider breaking down into smaller parts");
        }
        
        if input.split_whitespace().count() > 100 {
            suggestions.push("Consider simplifying the language");
        }
        
        // Direct pattern optimization
        let input_lower = input.to_lowercase();
        if input_lower.matches("for").count() > 3 {
            suggestions.push("Multiple loops detected - consider consolidation");
        }
        
        if input_lower.matches("if").count() > 5 {
            suggestions.push("Many conditions - consider restructuring logic");
        }
        
        result.push_str("Optimization Analysis:\n");
        result.push_str(&format!("Input Size: {} chars\n", input.len()));
        result.push_str(&format!("Word Count: {}\n", input.split_whitespace().count()));
        result.push_str(&format!("Loop Count: {}\n", input_lower.matches("for").count()));
        result.push_str(&format!("Condition Count: {}\n", input_lower.matches("if").count()));
        
        if suggestions.is_empty() {
            result.push_str("Status: Already optimized\n");
        } else {
            result.push_str("Suggestions:\n");
            for suggestion in suggestions {
                result.push_str(&format!("  - {}\n", suggestion));
            }
        }
        
        Ok(result)
    }

    pub async fn process_controller(input: &str) -> CoreResult<String> {
        let mut result = String::from("CONTROLLER ANALYSIS:\n");
        
        result.push_str(&format!("Input Length: {} characters\n", input.len()));
        result.push_str(&format!("Word Count: {}\n", input.split_whitespace().count()));
        result.push_str(&format!("Line Count: {}\n", input.lines().count()));
        
        // Direct character analysis
        let letter_count = input.chars().filter(|c| c.is_alphabetic()).count();
        let digit_count = input.chars().filter(|c| c.is_ascii_digit()).count();
        let special_count = input.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
        
        result.push_str(&format!("Letters: {}\n", letter_count));
        result.push_str(&format!("Digits: {}\n", digit_count));
        result.push_str(&format!("Special Characters: {}\n", special_count));
        
        // Direct complexity score
        let complexity = (input.len() as f32 / 100.0 + input.lines().count() as f32 / 10.0).min(10.0);
        result.push_str(&format!("Complexity Score: {:.2}/10\n", complexity));
        
        Ok(result)
    }

    pub async fn process_with_model(model_id: ModelId, input: &str) -> CoreResult<String> {
        match model_id {
            ModelId::Coding => Self::process_coding(input).await,
            ModelId::Memory => Self::process_memory(input).await,
            ModelId::Logic => Self::process_logic(input).await,
            ModelId::Planner => Self::process_planner(input).await,
            ModelId::Ranking => Self::process_ranking(input).await,
            ModelId::Retrieval => Self::process_retrieval(input).await,
            ModelId::Validator => Self::process_validator(input).await,
            ModelId::Personality => Self::process_personality(input).await,
            ModelId::Optimizer => Self::process_optimizer(input).await,
            ModelId::Controller => Self::process_controller(input).await,
        }
    }
}
