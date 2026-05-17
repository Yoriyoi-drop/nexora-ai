//! Intent detection untuk Nexora Core

use crate::types::{InputData, IntentType, IntentResult};
use crate::error::CoreResult;
use std::collections::HashMap;
use tracing::{debug, info};

/// Intent detector untuk menganalisis intent dari input user
pub struct IntentDetector {
    confidence_threshold: f32,
    keyword_weights: HashMap<String, (IntentType, f32, f32)>, // keyword -> (intent, score, priority_weight)
    reflection_keywords: Vec<String>,
}

impl IntentDetector {
    pub fn new() -> Self {
        let mut detector = Self {
            confidence_threshold: 0.62,
            keyword_weights: HashMap::new(),
            reflection_keywords: vec![
                "perbaiki".to_string(),
                "perjelas".to_string(),
                "reflection".to_string(),
                "refleksi".to_string(),
                "introspeksi".to_string(),
                "merefleksikan".to_string(),
                "pikir ulang".to_string(),
                "penjelasan".to_string(),
                "jelaskan".to_string(),
                "mengklarifikasi".to_string(),
            ],
        };
        
        detector.init_keyword_weights();
        detector
    }
    
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }
    
    /// Initialize keyword weights untuk intent detection
    fn init_keyword_weights(&mut self) {
        // CODING - High priority for technical keywords
        self.add_keyword("buat", IntentType::Coding, 0.9, 1.0);
        self.add_keyword("code", IntentType::Coding, 0.9, 1.2);
        self.add_keyword("program", IntentType::Coding, 0.8, 1.0);
        self.add_keyword("coding", IntentType::Coding, 0.9, 1.2);
        self.add_keyword("compiler", IntentType::Coding, 0.8, 1.3);
        self.add_keyword("fungsi", IntentType::Coding, 0.8, 1.1);
        self.add_keyword("rekursif", IntentType::Coding, 0.9, 1.2);
        
        // MEMORY - Lower priority when combined with technical terms
        self.add_keyword("simpan", IntentType::Memory, 0.9, 0.8);
        self.add_keyword("memory", IntentType::Memory, 0.9, 0.6);  // Lower priority - often context word
        self.add_keyword("ingat", IntentType::Memory, 0.8, 0.9);
        self.add_keyword("remember", IntentType::Memory, 0.8, 0.9);
        self.add_keyword("store", IntentType::Memory, 0.8, 0.8);
        
        // DEBUGGING - Very high priority for bug-related terms
        self.add_keyword("debug", IntentType::Debugging, 0.9, 1.3);
        self.add_keyword("error", IntentType::Debugging, 0.8, 1.1);
        self.add_keyword("bug", IntentType::Debugging, 0.9, 1.4);  // Highest priority - clear indicator
        self.add_keyword("fix", IntentType::Debugging, 0.7, 1.2);
        self.add_keyword("issue", IntentType::Debugging, 0.7, 1.0);
        self.add_keyword("leak", IntentType::Debugging, 0.8, 1.3);
        
        // PLANNING - Architecture and design terms
        self.add_keyword("plan", IntentType::Planning, 0.9, 1.0);
        self.add_keyword("planning", IntentType::Planning, 0.9, 1.1);
        self.add_keyword("rancang", IntentType::Planning, 0.9, 1.2);
        self.add_keyword("arsitektur", IntentType::Planning, 0.9, 1.3);  // Architecture keyword
        self.add_keyword("desain", IntentType::Planning, 0.8, 1.1);
        self.add_keyword("struktur", IntentType::Planning, 0.8, 1.2);
        self.add_keyword("workflow", IntentType::Planning, 0.8, 1.1);
        
        // REASONING - Logic and analysis terms
        self.add_keyword("logic", IntentType::Reasoning, 0.9, 1.2);
        self.add_keyword("logika", IntentType::Reasoning, 0.9, 1.3);  // Indonesian logic
        self.add_keyword("reasoning", IntentType::Reasoning, 0.9, 1.1);
        self.add_keyword("think", IntentType::Reasoning, 0.7, 0.9);
        self.add_keyword("analyze", IntentType::Reasoning, 0.8, 1.0);
        self.add_keyword("matematika", IntentType::Reasoning, 0.8, 1.2);
        self.add_keyword("inferensi", IntentType::Reasoning, 0.8, 1.1);
        
        // RANKING - Evaluation and comparison terms
        self.add_keyword("rank", IntentType::Ranking, 0.9, 1.2);
        self.add_keyword("ranking", IntentType::Ranking, 0.9, 1.1);
        self.add_keyword("evaluate", IntentType::Ranking, 0.8, 1.0);
        self.add_keyword("compare", IntentType::Ranking, 0.7, 1.0);
        self.add_keyword("prioritas", IntentType::Ranking, 0.8, 1.1);
        self.add_keyword("urutkan", IntentType::Ranking, 0.8, 1.2);
        
        // RETRIEVAL - Search and find terms
        self.add_keyword("cari", IntentType::Retrieval, 0.9, 1.1);
        self.add_keyword("find", IntentType::Retrieval, 0.9, 1.0);
        self.add_keyword("search", IntentType::Retrieval, 0.9, 1.1);
        self.add_keyword("retrieve", IntentType::Retrieval, 0.8, 1.0);
        self.add_keyword("referensi", IntentType::Retrieval, 0.8, 1.2);
        self.add_keyword("api", IntentType::Retrieval, 0.8, 1.1);
        
        // VALIDATION - Check and verify terms
        self.add_keyword("validasi", IntentType::Validation, 0.9, 1.2);
        self.add_keyword("validation", IntentType::Validation, 0.9, 1.1);
        self.add_keyword("check", IntentType::Validation, 0.8, 1.0);
        self.add_keyword("verify", IntentType::Validation, 0.8, 1.1);
        self.add_keyword("cek", IntentType::Validation, 0.8, 1.2);
        
        // PERSONALITY - Style and tone terms
        self.add_keyword("style", IntentType::Personality, 0.8, 1.1);
        self.add_keyword("personality", IntentType::Personality, 0.9, 1.2);
        self.add_keyword("tone", IntentType::Personality, 0.7, 1.0);
        self.add_keyword("gaya", IntentType::Personality, 0.8, 1.2);
        self.add_keyword("karakter", IntentType::Personality, 0.8, 1.1);
        
        // OPTIMIZATION - Performance and improvement terms
        self.add_keyword("optimasi", IntentType::Optimization, 0.9, 1.3);
        self.add_keyword("optimization", IntentType::Optimization, 0.9, 1.2);
        self.add_keyword("improve", IntentType::Optimization, 0.8, 1.0);
        self.add_keyword("enhance", IntentType::Optimization, 0.8, 1.0);
        self.add_keyword("cepat", IntentType::Optimization, 0.8, 1.1);
        self.add_keyword("performa", IntentType::Optimization, 0.8, 1.2);
        self.add_keyword("tingkatkan", IntentType::Optimization, 0.8, 1.1);
        
        // Integration Layer Mapping - Bridge existing Nexora components
        // Episodic Memory integration
        self.add_keyword("episodic", IntentType::Memory, 0.8, 1.2);
        self.add_keyword("episodic memory", IntentType::Memory, 0.9, 1.3);
        self.add_keyword("episode", IntentType::Memory, 0.7, 1.1);
        self.add_keyword("kenangan", IntentType::Memory, 0.8, 1.2);
        self.add_keyword("pengalaman", IntentType::Memory, 0.8, 1.1);
        self.add_keyword("memori", IntentType::Memory, 0.9, 1.0);
        
        // Reflection Engine integration  
        self.add_keyword("reflection", IntentType::Reasoning, 0.9, 1.3);
        self.add_keyword("refleksi", IntentType::Reasoning, 0.9, 1.3);
        self.add_keyword("introspeksi", IntentType::Reasoning, 0.8, 1.2);
        self.add_keyword("self-reflection", IntentType::Reasoning, 0.9, 1.4);
        self.add_keyword("merefleksikan", IntentType::Reasoning, 0.8, 1.2);
        self.add_keyword("pikir ulang", IntentType::Reasoning, 0.8, 1.1);
        
        // Self-Evaluation integration
        self.add_keyword("evaluasi", IntentType::Validation, 0.9, 1.3);
        self.add_keyword("evaluation", IntentType::Validation, 0.9, 1.2);
        self.add_keyword("self-evaluation", IntentType::Validation, 0.9, 1.4);
        self.add_keyword("mengevaluasi", IntentType::Validation, 0.8, 1.2);
        self.add_keyword("penilaian", IntentType::Validation, 0.8, 1.1);
        self.add_keyword("assessment", IntentType::Validation, 0.8, 1.1);
        
        // Conversation State integration
        self.add_keyword("conversation", IntentType::Reasoning, 0.8, 1.1);
        self.add_keyword("percakapan", IntentType::Reasoning, 0.8, 1.2);
        self.add_keyword("dialog", IntentType::Reasoning, 0.7, 1.1);
        self.add_keyword("state", IntentType::Reasoning, 0.7, 1.0);
        self.add_keyword("konteks", IntentType::Reasoning, 0.8, 1.1);
        self.add_keyword("context", IntentType::Reasoning, 0.8, 1.0);
        
        // Context Compression integration
        self.add_keyword("compression", IntentType::Optimization, 0.8, 1.2);
        self.add_keyword("kompresi", IntentType::Optimization, 0.8, 1.2);
        self.add_keyword("compress", IntentType::Optimization, 0.8, 1.1);
        self.add_keyword("ringkas", IntentType::Optimization, 0.8, 1.1);
        self.add_keyword("summary", IntentType::Optimization, 0.8, 1.1);
        self.add_keyword("meringkas", IntentType::Optimization, 0.8, 1.2);
        
        // Internal Scratchpad integration
        self.add_keyword("scratchpad", IntentType::Planning, 0.8, 1.2);
        self.add_keyword("draft", IntentType::Planning, 0.7, 1.1);
        self.add_keyword("sketch", IntentType::Planning, 0.7, 1.1);
        self.add_keyword("kerja", IntentType::Planning, 0.7, 1.0);
        self.add_keyword("working", IntentType::Planning, 0.7, 1.0);
        self.add_keyword("temporer", IntentType::Planning, 0.7, 1.0);
        
        // Goal System integration
        self.add_keyword("goal", IntentType::Planning, 0.9, 1.3);
        self.add_keyword("tujuan", IntentType::Planning, 0.9, 1.3);
        self.add_keyword("target", IntentType::Planning, 0.8, 1.2);
        self.add_keyword("objektif", IntentType::Planning, 0.8, 1.2);
        self.add_keyword("objective", IntentType::Planning, 0.8, 1.1);
        self.add_keyword("misi", IntentType::Planning, 0.8, 1.2);
    }
    
    fn add_keyword(&mut self, keyword: &str, intent: IntentType, score: f32, priority_weight: f32) {
        self.keyword_weights.insert(keyword.to_string(), (intent, score, priority_weight));
    }
    
    /// Detect intent dari input data
    pub async fn detect_intent(&self, input_data: &InputData) -> CoreResult<IntentResult> {
        debug!("Detecting intent for input: {}", &input_data.raw_input[..input_data.raw_input.len().min(50)]);
        
        let mut result = IntentResult::new();
        
        // Check if this is a reflection input
        let is_reflection_input = self.is_reflection_input(&input_data.raw_input);
        
        if is_reflection_input {
            // Direct intent assignment for reflection inputs
            self.add_reflection_intents(&mut result);
        } else {
            // Normal detection for non-reflection inputs
            self.detect_normal_intents(&input_data.raw_input, &mut result);
        }
        
        // Generate reasoning
        self.generate_reasoning(&mut result);
        
        info!("Intent detected: primary={:?}, multi_intent={:?}, intents_count={}", 
              result.primary_intent, result.is_multi_intent, result.intents.len());
        
        Ok(result)
    }
    
    fn is_reflection_input(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        self.reflection_keywords.iter().any(|keyword| input_lower.contains(keyword))
    }
    
    fn add_reflection_intents(&self, result: &mut IntentResult) {
        // Assign reflection-specific intents with proper confidences
        result.add_intent(IntentType::Reasoning, 0.85);
        result.add_intent(IntentType::Validation, 0.80);
        result.add_intent(IntentType::Memory, 0.75);
        result.primary_intent = IntentType::Reasoning; // Reasoning is primary for reflection
        result.intent_reasoning = "Reflection input detected: Reasoning (0.85), Validation (0.80), Memory (0.75). Primary intent selected by highest confidence.".to_string();
    }
    
    fn detect_normal_intents(&self, input: &str, result: &mut IntentResult) {
        let input_lower = input.to_lowercase();
        let mut intent_scores: HashMap<IntentType, f32> = HashMap::with_capacity(self.keyword_weights.len());
        
        // Calculate weighted scores for each intent
        for (keyword, (intent_type, score, priority_weight)) in &self.keyword_weights {
            if input_lower.contains(keyword) {
                let weighted_score = score * priority_weight;
                *intent_scores.entry(*intent_type).or_insert(0.0) += weighted_score;
            }
        }
        
        // Add intents above threshold
        let mut max_confidence = 0.0;
        let mut max_intent = IntentType::Unknown;
        
        for (intent_type, score) in intent_scores {
            // Apply base confidence and length factor
            let confidence = self.calculate_confidence(input, intent_type, score);
            
            if confidence > self.confidence_threshold {
                result.add_intent(intent_type, confidence);
                
                if confidence > max_confidence {
                    max_confidence = confidence;
                    max_intent = intent_type;
                }
            }
        }
        
        // EPISODIC MEMORY PRIORITY OVERRIDE
        if self.has_episodic_memory_keywords(input) {
            let memory_confidence = result.get_confidence(IntentType::Memory);
            if memory_confidence > 0.0 {
                result.primary_intent = IntentType::Memory;
                max_intent = IntentType::Memory;
            }
        }
        
        // Set primary intent
        if result.primary_intent == IntentType::Unknown && max_intent != IntentType::Unknown {
            result.primary_intent = max_intent;
        }
    }
    
    fn calculate_confidence(&self, input: &str, intent: IntentType, score: f32) -> f32 {
        let input_len = input.len();
        let mut base_confidence = 0.4; // Higher base confidence for better detection
        
        // Length factor - longer inputs usually have more context
        if input_len > 50 {
            base_confidence += 0.15;
        } else if input_len > 20 {
            base_confidence += 0.1;
        }
        
        // Special boost for reflection/integration keywords
        if self.is_reflection_input(input) {
            match intent {
                IntentType::Reasoning | IntentType::Validation | IntentType::Memory => {
                    base_confidence += 0.4; // Higher boost for reflection-allowed intents
                }
                _ => {
                    base_confidence -= 0.2; // Penalty for disallowed intents in reflection
                }
            }
        }
        
        // Apply keyword score
        base_confidence += score * 0.6; // Max 0.6f boost from scoring
        
        // Cap confidence at 1.0
        base_confidence.clamp(0.0, 1.0)
    }
    
    fn has_episodic_memory_keywords(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        input_lower.contains("ingat") || 
        input_lower.contains("episodic") ||
        input_lower.contains("kenangan") || 
        input_lower.contains("pengalaman") ||
        input_lower.contains("recall") || 
        input_lower.contains("remember")
    }
    
    fn generate_reasoning(&self, result: &mut IntentResult) {
        if result.intent_reasoning.is_empty() {
            if result.intents.is_empty() {
                result.intent_reasoning = format!("No intents detected above threshold {}", self.confidence_threshold);
                result.primary_intent = IntentType::Unknown;
            } else if result.is_multi_intent {
                let mut reasoning = format!("Detected {} intents: ", result.intents.len());
                for (i, intent_score) in result.intents.iter().enumerate() {
                    if i == 0 {
                        reasoning.push_str(&format!("{} ({:.2})", intent_score.intent_type.name(), intent_score.confidence));
                    } else {
                        reasoning.push_str(&format!(", {} ({:.2})", intent_score.intent_type.name(), intent_score.confidence));
                    }
                }
                reasoning.push_str(". Primary intent selected by highest confidence.");
                result.intent_reasoning = reasoning;
            } else {
                result.intent_reasoning = format!(
                    "Single intent detected: {} ({:.2})",
                    result.primary_intent.name(),
                    result.intents.first().map(|s| s.confidence).unwrap_or(0.0)
                );
            }
        }
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_coding_intent_detection() {
        let detector = IntentDetector::new();
        let input_data = InputData::new("buat fungsi rekursif".to_string(), crate::types::InputType::Text);
        
        let result = detector.detect_intent(&input_data).await.unwrap();
        assert_eq!(result.primary_intent, IntentType::Coding);
        assert!(result.get_confidence(IntentType::Coding) > 0.5);
    }
    
    #[tokio::test]
    async fn test_debugging_intent_detection() {
        let detector = IntentDetector::new();
        let input_data = InputData::new("fix bug memory leak".to_string(), crate::types::InputType::Text);
        
        let result = detector.detect_intent(&input_data).await.unwrap();
        assert_eq!(result.primary_intent, IntentType::Debugging);
        assert!(result.get_confidence(IntentType::Debugging) > 0.5);
    }
    
    #[tokio::test]
    async fn test_reflection_intent_detection() {
        let detector = IntentDetector::new();
        let input_data = InputData::new("perbaiki penjelasan".to_string(), crate::types::InputType::Text);
        
        let result = detector.detect_intent(&input_data).await.unwrap();
        assert_eq!(result.primary_intent, IntentType::Reasoning);
        assert!(result.is_multi_intent);
        assert!(result.get_confidence(IntentType::Reasoning) > 0.8);
    }
    
    #[tokio::test]
    async fn test_episodic_memory_priority() {
        let detector = IntentDetector::new();
        let input_data = InputData::new("ingat pengalaman episodic".to_string(), crate::types::InputType::Text);
        
        let result = detector.detect_intent(&input_data).await.unwrap();
        assert_eq!(result.primary_intent, IntentType::Memory);
    }
}
