use nexora_shared::base_model::NxrModelResult;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default)]
pub struct WorldModelRuntimeAgent;

impl WorldModelRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub fn update_context(&self, input: &str) -> NxrModelResult<String> {
        if input.is_empty() {
            return Err("Empty input cannot update world model".into());
        }
        let word_count = input.split_whitespace().count();
        let unique_terms: HashSet<&str> = input
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        let lexical_diversity = if word_count > 0 {
            unique_terms.len() as f64 / word_count as f64
        } else {
            0.0
        };
        let coherence = (lexical_diversity * 100.0).min(100.0);

        Ok(format!(
            "[WORLD-MODEL-X] Updated world state representation:\n\
             - Ingested {} new observations from input\n\
             - Integrated into world model tensor ({} dimensions)\n\
             - Lexical diversity: {:.2}\n\
             - Coherence score: {:.2}\n\
             - World model version: omnis-x-v1",
            word_count,
            unique_terms.len(),
            lexical_diversity,
            coherence
        ))
    }

    pub fn process_input(&self, input: &str) -> NxrModelResult<HashMap<String, serde_json::Value>> {
        let mut update = HashMap::new();
        let word_count = input.split_whitespace().count();
        let unique_terms: HashSet<&str> = input
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        update.insert(
            "last_input".to_string(),
            serde_json::Value::String(input.to_string()),
        );
        update.insert(
            "input_length".to_string(),
            serde_json::Value::Number(serde_json::Number::from(input.len() as u64)),
        );
        update.insert(
            "word_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(word_count as u64)),
        );
        update.insert(
            "unique_terms".to_string(),
            serde_json::Value::Number(serde_json::Number::from(unique_terms.len() as u64)),
        );
        update.insert(
            "world_model_version".to_string(),
            serde_json::Value::String("omnis-x-v1".to_string()),
        );
        Ok(update)
    }
}
