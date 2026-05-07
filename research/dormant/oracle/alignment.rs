//! DPO untuk Code Preferences Alignment
//! 
//! Implementasi Direct Preference Optimization yang disesuaikan khusus
//! untuk alignment model bahasa pada preferensi kode yang clean, efisien, dan aman.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Konfigurasi Code DPO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDpoConfig {
    /// Learning rate for DPO
    pub learning_rate: f32,
    /// Beta parameter for DPO
    pub beta: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Code-specific weighting
    pub code_weight: f32,
    /// Security weight for alignment
    pub security_weight: f32,
    /// Efficiency weight for alignment
    pub efficiency_weight: f32,
    /// Regularization strength
    pub regularization_strength: f32,
}

impl Default for CodeDpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            beta: 0.1,
            max_seq_len: 8192,
            code_weight: 0.4,
            security_weight: 0.3,
            efficiency_weight: 0.3,
            regularization_strength: 0.01,
        }
    }
}

/// Code preference pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePreferencePair {
    pub id: Uuid,
    pub prompt: String,
    pub chosen_code: String,
    pub rejected_code: String,
    pub chosen_language: String,
    pub rejected_language: String,
    pub chosen_logprob: f32,
    pub rejected_logprob: f32,
    pub reference_chosen_logprob: f32,
    pub reference_rejected_logprob: f32,
    pub security_score: f32,
    pub efficiency_score: f32,
    pub code_quality_score: f32,
}

/// Code DPO Trainer
pub struct CodeDpoTrainer {
    config: CodeDpoConfig,
    model: CodeModel,
    reference_model: CodeModel,
    code_analyzer: CodeAnalyzer,
    security_analyzer: SecurityAnalyzer,
    efficiency_analyzer: EfficiencyAnalyzer,
}

impl CodeDpoTrainer {
    pub fn new(
        config: CodeDpoConfig,
        model: CodeModel,
        reference_model: CodeModel,
    ) -> Self {
        Self {
            config,
            model,
            reference_model,
            code_analyzer: CodeAnalyzer::new(),
            security_analyzer: SecurityAnalyzer::new(),
            efficiency_analyzer: EfficiencyAnalyzer::new(),
        }
    }
    
    /// Training step for code DPO
    pub fn training_step(&mut self, preference_pairs: &[CodePreferencePair]) -> Result<DpoLoss> {
        let mut total_loss = 0.0;
        let mut security_loss = 0.0;
        let mut efficiency_loss = 0.0;
        let mut code_quality_loss = 0.0;
        
        for pair in preference_pairs {
            // Compute DPO loss
            let dpo_loss = self.compute_dpo_loss(pair)?;
            
            // Compute code-specific losses
            let security_loss_pair = self.compute_security_loss(pair)?;
            let efficiency_loss_pair = self.compute_efficiency_loss(pair)?;
            let code_quality_loss_pair = self.compute_code_quality_loss(pair)?;
            
            // Combine losses
            let combined_loss = dpo_loss + 
                self.config.security_weight * security_loss_pair +
                self.config.efficiency_weight * efficiency_loss_pair +
                self.config.code_weight * code_quality_loss_pair;
            
            total_loss += combined_loss;
            security_loss += security_loss_pair;
            efficiency_loss += efficiency_loss_pair;
            code_quality_loss += code_quality_loss_pair;
        }
        
        let avg_loss = total_loss / preference_pairs.len() as f32;
        let avg_security_loss = security_loss / preference_pairs.len() as f32;
        let avg_efficiency_loss = efficiency_loss / preference_pairs.len() as f32;
        let avg_code_quality_loss = code_quality_loss / preference_pairs.len() as f32;
        
        // Update model parameters (simplified)
        self.update_model_parameters(avg_loss)?;
        
        Ok(DpoLoss {
            total_loss: avg_loss,
            security_loss: avg_security_loss,
            efficiency_loss: avg_efficiency_loss,
            code_quality_loss: avg_code_quality_loss,
        })
    }
    
    /// Compute DPO loss for a preference pair
    fn compute_dpo_loss(&self, pair: &CodePreferencePair) -> Result<f32> {
        let pi_lograt = pair.chosen_logprob - pair.rejected_logprob;
        let ref_lograt = pair.reference_chosen_logprob - pair.reference_rejected_logprob;
        
        let ratio = pi_lograt - ref_lograt;
        let sigmoid_input = self.config.beta * ratio;
        let sigmoid_input_clamped = sigmoid_input.clamp(-20.0, 20.0);
        
        let loss = -sigmoid_input_clamped.ln_1p();
        
        // Add regularization
        let regularization = self.config.regularization_strength * (pi_lograt * pi_lograt);
        
        Ok(loss + regularization)
    }
    
    /// Compute security alignment loss
    fn compute_security_loss(&self, pair: &CodePreferencePair) -> Result<f32> {
        let chosen_security = self.security_analyzer.analyze_security(&pair.chosen_code)?;
        let rejected_security = self.security_analyzer.analyze_security(&pair.rejected_code)?;
        
        // Penalize if chosen code is less secure than rejected code
        let security_diff = rejected_security - chosen_security;
        let security_loss = if security_diff > 0.0 {
            security_diff * security_diff // Quadratic penalty
        } else {
            0.0
        };
        
        Ok(security_loss)
    }
    
    /// Compute efficiency alignment loss
    fn compute_efficiency_loss(&self, pair: &CodePreferencePair) -> Result<f32> {
        let chosen_efficiency = self.efficiency_analyzer.analyze_efficiency(&pair.chosen_code)?;
        let rejected_efficiency = self.efficiency_analyzer.analyze_efficiency(&pair.rejected_code)?;
        
        // Penalize if chosen code is less efficient than rejected code
        let efficiency_diff = rejected_efficiency - chosen_efficiency;
        let efficiency_loss = if efficiency_diff > 0.0 {
            efficiency_diff * efficiency_diff // Quadratic penalty
        } else {
            0.0
        };
        
        Ok(efficiency_loss)
    }
    
    /// Compute code quality alignment loss
    fn compute_code_quality_loss(&self, pair: &CodePreferencePair) -> Result<f32> {
        let chosen_quality = self.code_analyzer.analyze_quality(&pair.chosen_code)?;
        let rejected_quality = self.code_analyzer.analyze_quality(&pair.rejected_code)?;
        
        // Penalize if chosen code has lower quality than rejected code
        let quality_diff = rejected_quality - chosen_quality;
        let quality_loss = if quality_diff > 0.0 {
            quality_diff * quality_diff // Quadratic penalty
        } else {
            0.0
        };
        
        Ok(quality_loss)
    }
    
    /// Update model parameters
    fn update_model_parameters(&mut self, loss: f32) -> Result<()> {
        // Simplified parameter update
        // In practice, this would update neural network parameters
        let gradient = loss * self.config.learning_rate;
        
        // Apply gradient (placeholder)
        println!("Updating model parameters with gradient: {:.6}", gradient);
        
        Ok(())
    }
    
    /// Generate code preferences from model outputs
    pub fn generate_preferences(&self, prompt: &str, num_samples: usize) -> Result<Vec<CodePreferencePair>> {
        let mut preferences = Vec::new();
        
        for _ in 0..num_samples {
            // Generate two code samples
            let code1 = self.model.generate_code(prompt)?;
            let code2 = self.model.generate_code(prompt)?;
            
            // Analyze both samples
            let quality1 = self.code_analyzer.analyze_quality(&code1)?;
            let quality2 = self.code_analyzer.analyze_quality(&code2)?;
            
            let security1 = self.security_analyzer.analyze_security(&code1)?;
            let security2 = self.security_analyzer.analyze_security(&code2)?;
            
            let efficiency1 = self.efficiency_analyzer.analyze_efficiency(&code1)?;
            let efficiency2 = self.efficiency_analyzer.analyze_efficiency(&code2)?;
            
            // Determine preference based on combined score
            let score1 = quality1 + security1 + efficiency1;
            let score2 = quality2 + security2 + efficiency2;
            
            let (chosen_code, rejected_code, chosen_score, rejected_score) = if score1 > score2 {
                (&code1, &code2, score1, score2)
            } else {
                (&code2, &code1, score2, score1)
            };
            
            // Compute log probabilities (simplified)
            let chosen_logprob = self.compute_log_probability(prompt, chosen_code)?;
            let rejected_logprob = self.compute_log_probability(prompt, rejected_code)?;
            let ref_chosen_logprob = self.reference_model.compute_log_probability(prompt, chosen_code)?;
            let ref_rejected_logprob = self.reference_model.compute_log_probability(prompt, rejected_code)?;
            
            let preference = CodePreferencePair {
                id: Uuid::new_v4(),
                prompt: prompt.to_string(),
                chosen_code: chosen_code.clone(),
                rejected_code: rejected_code.clone(),
                chosen_language: "python".to_string(), // Simplified
                rejected_language: "python".to_string(),
                chosen_logprob,
                rejected_logprob,
                reference_chosen_logprob: ref_chosen_logprob,
                reference_rejected_logprob: ref_rejected_logprob,
                security_score: self.security_analyzer.analyze_security(chosen_code)?,
                efficiency_score: self.efficiency_analyzer.analyze_efficiency(chosen_code)?,
                code_quality_score: self.code_analyzer.analyze_quality(chosen_code)?,
            };
            
            preferences.push(preference);
        }
        
        Ok(preferences)
    }
    
    /// Compute log probability (simplified)
    fn compute_log_probability(&self, prompt: &str, code: &str) -> Result<f32> {
        let combined = format!("{} {}", prompt, code);
        let hash = format!("{:x}", md5::compute(combined.as_bytes()));
        let hash_value = u64::from_str_radix(&hash[..8], 16).unwrap_or(0);
        let prob = (hash_value as f32 / u64::MAX as f32).ln();
        Ok(prob)
    }
    
    /// Get alignment statistics
    pub fn get_alignment_stats(&self, preference_pairs: &[CodePreferencePair]) -> AlignmentStats {
        let total_pairs = preference_pairs.len();
        
        let avg_security_score = preference_pairs.iter()
            .map(|p| p.security_score)
            .sum::<f32>() / total_pairs.max(1) as f32;
        
        let avg_efficiency_score = preference_pairs.iter()
            .map(|p| p.efficiency_score)
            .sum::<f32>() / total_pairs.max(1) as f32;
        
        let avg_code_quality_score = preference_pairs.iter()
            .map(|p| p.code_quality_score)
            .sum::<f32>() / total_pairs.max(1) as f32;
        
        let security_improvements = preference_pairs.iter()
            .filter(|p| p.security_score > 0.5)
            .count();
        
        let efficiency_improvements = preference_pairs.iter()
            .filter(|p| p.efficiency_score > 0.5)
            .count();
        
        let quality_improvements = preference_pairs.iter()
            .filter(|p| p.code_quality_score > 0.5)
            .count();
        
        AlignmentStats {
            total_pairs,
            avg_security_score,
            avg_efficiency_score,
            avg_code_quality_score,
            security_improvement_rate: security_improvements as f32 / total_pairs.max(1) as f32,
            efficiency_improvement_rate: efficiency_improvements as f32 / total_pairs.max(1) as f32,
            quality_improvement_rate: quality_improvements as f32 / total_pairs.max(1) as f32,
        }
    }
}

/// Code model (simplified)
pub struct CodeModel {
    vocab_size: usize,
    max_seq_len: usize,
}

impl CodeModel {
    pub fn new(vocab_size: usize, max_seq_len: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len,
        }
    }
    
    pub fn generate_code(&self, prompt: &str) -> Result<String> {
        // Simplified code generation
        let code_templates = vec![
            "def function():\n    pass",
            "function() {\n    // implementation\n}",
            "class MyClass:\n    def __init__(self):\n        pass",
            "// TODO: implement function",
            "return result",
        ];
        
        let template = code_templates[rand::random::<usize>() % code_templates.len()];
        Ok(format!("{}: {}", prompt, template))
    }
    
    pub fn compute_log_probability(&self, prompt: &str, code: &str) -> Result<f32> {
        let combined = format!("{} {}", prompt, code);
        let hash = format!("{:x}", md5::compute(combined.as_bytes()));
        let hash_value = u64::from_str_radix(&hash[..8], 16).unwrap_or(0);
        let prob = (hash_value as f32 / u64::MAX as f32).ln();
        Ok(prob)
    }
}

/// Code analyzer
pub struct CodeAnalyzer {
    quality_metrics: Vec<String>,
}

impl CodeAnalyzer {
    pub fn new() -> Self {
        Self {
            quality_metrics: vec![
                "lines_of_code".to_string(),
                "cyclomatic_complexity".to_string(),
                "maintainability_index".to_string(),
                "code_duplication".to_string(),
                "naming_conventions".to_string(),
            ],
        }
    }
    
    pub fn analyze_quality(&self, code: &str) -> Result<f32> {
        let mut quality_score = 0.5; // Base score
        
        // Analyze code quality metrics
        let lines = code.lines().count();
        if lines > 0 && lines < 1000 {
            quality_score += 0.1; // Reasonable length
        }
        
        // Check for good practices
        if code.contains("def ") || code.contains("function ") {
            quality_score += 0.1; // Has functions
        }
        
        if code.contains("class ") {
            quality_score += 0.05; // Has classes
        }
        
        // Check for comments
        let comment_lines = code.lines()
            .filter(|line| line.trim().starts_with("#") || line.trim().starts_with("//"))
            .count();
        if comment_lines > 0 {
            quality_score += 0.1; // Has comments
        }
        
        // Check for error handling
        if code.contains("try ") && code.contains("catch ") || code.contains("except ") {
            quality_score += 0.1; // Has error handling
        }
        
        // Penalize bad practices
        if code.contains("eval(") || code.contains("exec(") {
            quality_score -= 0.2; // Dangerous functions
        }
        
        if code.lines().any(|line| line.trim().len() > 120) {
            quality_score -= 0.1; // Long lines
        }
        
        Ok(quality_score.clamp(0.0, 1.0))
    }
    
    pub fn get_quality_metrics(&self, code: &str) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        
        metrics.insert("lines_of_code".to_string(), code.lines().count() as f32);
        metrics.insert("cyclomatic_complexity".to_string(), self.calculate_complexity(code));
        metrics.insert("maintainability_index".to_string(), self.calculate_maintainability(code));
        metrics.insert("code_duplication".to_string(), self.calculate_duplication(code));
        metrics.insert("naming_conventions".to_string(), self.check_naming_conventions(code));
        
        metrics
    }
    
    fn calculate_complexity(&self, code: &str) -> f32 {
        let mut complexity = 1.0; // Base complexity
        
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
    
    fn calculate_maintainability(&self, code: &str) -> f32 {
        let lines = code.lines().count() as f32;
        let complexity = self.calculate_complexity(code);
        
        // Simplified maintainability index
        (171.0 - 5.2 * (complexity).ln() - 0.23 * complexity - 16.2 * (lines).ln()).max(0.0)
    }
    
    fn calculate_duplication(&self, code: &str) -> f32 {
        let lines: Vec<&str> = code.lines().collect();
        let mut duplicates = 0;
        
        for (i, line1) in lines.iter().enumerate() {
            for line2 in lines.iter().skip(i + 1) {
                if line1.trim() == line2.trim() && !line1.trim().is_empty() {
                    duplicates += 1;
                    break;
                }
            }
        }
        
        if lines.is_empty() {
            0.0
        } else {
            duplicates as f32 / lines.len() as f32
        }
    }
    
    fn check_naming_conventions(&self, code: &str) -> f32 {
        let mut score = 1.0;
        
        // Check for snake_case in Python
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("def ") {
                let func_name = trimmed.split_whitespace().nth(1).unwrap_or("");
                if !func_name.is_empty() && func_name.contains("_") {
                    score += 0.1;
                }
            }
        }
        
        score.clamp(0.0, 1.0)
    }
}

/// Security analyzer
pub struct SecurityAnalyzer {
    vulnerability_patterns: Vec<String>,
}

impl SecurityAnalyzer {
    pub fn new() -> Self {
        Self {
            vulnerability_patterns: vec![
                "eval(".to_string(),
                "exec(".to_string(),
                "system(".to_string(),
                "subprocess.call".to_string(),
                "os.system".to_string(),
                "pickle.loads".to_string(),
                "input()".to_string(),
                "raw_input()".to_string(),
                "shell=True".to_string(),
                "password".to_string(),
                "secret".to_string(),
                "token".to_string(),
                "api_key".to_string(),
            ],
        }
    }
    
    pub fn analyze_security(&self, code: &str) -> Result<f32> {
        let mut security_score = 1.0; // Start with perfect score
        
        // Check for vulnerable patterns
        for pattern in &self.vulnerability_patterns {
            if code.contains(pattern) {
                security_score -= 0.1;
            }
        }
        
        // Check for input validation
        if code.contains("input(") || code.contains("raw_input(") {
            if !code.contains("strip(") && !code.contains("validate(") {
                security_score -= 0.2; // Unvalidated input
            }
        }
        
        // Check for SQL injection patterns
        if code.contains("SELECT") && code.contains("input(") {
            security_score -= 0.3;
        }
        
        // Check for XSS patterns
        if code.contains("innerHTML") || code.contains("document.write") {
            security_score -= 0.2;
        }
        
        // Check for hardcoded credentials
        if code.to_lowercase().contains("password") || 
           code.to_lowercase().contains("secret") ||
           code.to_lowercase().contains("api_key") {
            security_score -= 0.3;
        }
        
        Ok(security_score.clamp(0.0, 1.0))
    }
    
    pub fn get_vulnerabilities(&self, code: &str) -> Vec<String> {
        let mut vulnerabilities = Vec::new();
        
        for pattern in &self.vulnerability_patterns {
            if code.contains(pattern) {
                vulnerabilities.push(format!("Potential vulnerability: {}", pattern));
            }
        }
        
        vulnerabilities
    }
}

/// Efficiency analyzer
pub struct EfficiencyAnalyzer {
    efficiency_patterns: HashMap<String, f32>,
}

impl EfficiencyAnalyzer {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Good patterns
        patterns.insert("list comprehension".to_string(), 0.1);
        patterns.insert("generator".to_string(), 0.1);
        patterns.insert("cache".to_string(), 0.1);
        patterns.insert("memoize".to_string(), 0.1);
        patterns.insert("lazy evaluation".to_string(), 0.1);
        
        // Bad patterns
        patterns.insert("nested loops".to_string(), -0.1);
        patterns.insert("global variable".to_string(), -0.1);
        patterns.insert("recursion depth".to_string(), -0.1);
        patterns.insert("memory leak".to_string(), -0.2);
        
        Self { efficiency_patterns: patterns }
    }
    
    pub fn analyze_efficiency(&self, code: &str) -> Result<f32> {
        let mut efficiency_score = 0.5; // Base score
        
        // Check for efficient patterns
        for (pattern, score) in &self.efficiency_patterns {
            if code.contains(pattern) {
                efficiency_score += score;
            }
        }
        
        // Check algorithmic complexity
        let complexity_score = self.analyze_algorithmic_complexity(code);
        efficiency_score += complexity_score;
        
        // Check for memory usage patterns
        let memory_score = self.analyze_memory_usage(code);
        efficiency_score += memory_score;
        
        Ok(efficiency_score.clamp(0.0, 1.0))
    }
    
    fn analyze_algorithmic_complexity(&self, code: &str) -> f32 {
        let mut score = 0.0;
        
        // Check for nested loops
        let lines: Vec<&str> = code.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("for ") || line.trim().starts_with("while ") {
                // Check if there's another loop inside this loop
                for nested_line in lines.iter().skip(i + 1) {
                    if nested_line.trim().starts_with("    for ") || 
                       nested_line.trim().starts_with("    while ") {
                        score -= 0.1; // Nested loop detected
                        break;
                    }
                }
            }
        }
        
        score
    }
    
    fn analyze_memory_usage(&self, code: &str) -> f32 {
        let mut score = 0.0;
        
        // Check for memory-efficient patterns
        if code.contains("yield") || code.contains("generator") {
            score += 0.1; // Generator pattern
        }
        
        if code.contains("del ") || code.contains("free(") {
            score += 0.05; // Memory cleanup
        }
        
        // Check for memory-intensive patterns
        if code.contains("deepcopy") || code.contains("clone()") {
            score -= 0.05; // Potential memory overhead
        }
        
        score
    }
    
    pub fn get_efficiency_metrics(&self, code: &str) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        
        metrics.insert("algorithmic_complexity".to_string(), self.analyze_algorithmic_complexity(code));
        metrics.insert("memory_efficiency".to_string(), self.analyze_memory_usage(code));
        metrics.insert("time_complexity".to_string(), self.estimate_time_complexity(code));
        
        metrics
    }
    
    fn estimate_time_complexity(&self, code: &str) -> f32 {
        let mut complexity = 1.0; // O(1) baseline
        
        // Count nested loops
        let lines: Vec<&str> = code.lines().collect();
        let mut nested_level = 0;
        let max_nested = 0;
        
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("for ") || trimmed.starts_with("while ") {
                nested_level += 1;
                max_nested = max_nested.max(nested_level);
            }
            if trimmed == "}" || trimmed == "end" {
                nested_level = nested_level.saturating_sub(1);
            }
        }
        
        // Estimate complexity based on nesting
        complexity = match max_nested {
            0 => 1.0,    // O(1)
            1 => 2.0,    // O(n)
            2 => 4.0,    // O(n²)
            3 => 8.0,    // O(n³)
            _ => 16.0,   // O(n^k) where k >= 4
        };
        
        complexity
    }
}

/// DPO loss components
#[derive(Debug, Clone)]
pub struct DpoLoss {
    pub total_loss: f32,
    pub security_loss: f32,
    pub efficiency_loss: f32,
    pub code_quality_loss: f32,
}

/// Alignment statistics
#[derive(Debug, Clone)]
pub struct AlignmentStats {
    pub total_pairs: usize,
    pub avg_security_score: f32,
    pub avg_efficiency_score: f32,
    pub avg_code_quality_score: f32,
    pub security_improvement_rate: f32,
    pub efficiency_improvement_rate: f32,
    pub quality_improvement_rate: f32,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create code preference pairs from examples
    pub fn create_preference_pairs(
        prompts: &[String],
        good_codes: &[String],
        bad_codes: &[String],
    ) -> Vec<CodePreferencePair> {
        let mut pairs = Vec::new();
        
        for (i, prompt) in prompts.iter().enumerate() {
            if i < good_codes.len() && i < bad_codes.len() {
                let pair = CodePreferencePair {
                    id: Uuid::new_v4(),
                    prompt: prompt.clone(),
                    chosen_code: good_codes[i].clone(),
                    rejected_code: bad_codes[i].clone(),
                    chosen_language: "python".to_string(),
                    rejected_language: "python".to_string(),
                    chosen_logprob: -1.0,
                    rejected_logprob: -2.0,
                    reference_chosen_logprob: -1.1,
                    reference_rejected_logprob: -2.1,
                    security_score: 0.8,
                    efficiency_score: 0.7,
                    code_quality_score: 0.9,
                };
                pairs.push(pair);
            }
        }
        
        pairs
    }
    
    /// Validate code preference pairs
    pub fn validate_preference_pairs(pairs: &[CodePreferencePair]) -> Result<()> {
        for pair in pairs {
            if pair.chosen_code.is_empty() || pair.rejected_code.is_empty() {
                return Err(anyhow::anyhow!("Empty code in preference pair"));
            }
            
            if pair.prompt.is_empty() {
                return Err(anyhow::anyhow!("Empty prompt in preference pair"));
            }
            
            if pair.chosen_code == pair.rejected_code {
                return Err(anyhow::anyhow!("Chosen and rejected code are identical"));
            }
        }
        
        Ok(())
    }
    
    /// Analyze alignment progress
    pub fn analyze_alignment_progress(
        current_stats: &AlignmentStats,
        previous_stats: &AlignmentStats,
    ) -> AlignmentProgress {
        let security_improvement = current_stats.avg_security_score - previous_stats.avg_security_score;
        let efficiency_improvement = current_stats.avg_efficiency_score - previous_stats.avg_efficiency_score;
        let quality_improvement = current_stats.avg_code_quality_score - previous_stats.avg_code_quality_score;
        
        AlignmentProgress {
            security_improvement,
            efficiency_improvement,
            quality_improvement,
            overall_improvement: (security_improvement + efficiency_improvement + quality_improvement) / 3.0,
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct AlignmentProgress {
        pub security_improvement: f32,
        pub efficiency_improvement: f32,
        pub quality_improvement: f32,
        pub overall_improvement: f32,
    }
}
