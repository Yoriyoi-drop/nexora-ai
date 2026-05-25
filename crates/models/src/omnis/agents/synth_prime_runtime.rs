use nexora_shared::base_model::NxrModelResult;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct SynthPrimeRuntimeAgent;

impl SynthPrimeRuntimeAgent {
    pub fn new() -> Self {
        Self
    }

    pub fn synthesize(&self, fragments: &[String]) -> NxrModelResult<String> {
        if fragments.is_empty() {
            return Err("No fragments to synthesize".into());
        }
        let total_chars: usize = fragments.iter().map(|f| f.len()).sum();
        let avg_length = total_chars as f64 / fragments.len() as f64;
        let combined = fragments.join(" ");
        let words: Vec<&str> = combined.split_whitespace().collect();
        let coherence = if words.len() > 5 {
            let unique: HashSet<&&str> = words.iter().collect();
            (unique.len() as f64 / words.len() as f64 * 100.0).min(100.0)
        } else {
            0.0
        };

        Ok(format!(
            "[SYNTH-PRIME] Synthesis complete:\n\
             - Fragments combined: {}\n\
             - Total length: {} chars, {} words\n\
             - Average fragment length: {:.1} chars\n\
             - Coherence score: {:.1}%\n\
             - Output preview: {}...",
            fragments.len(),
            total_chars,
            words.len(),
            avg_length,
            coherence,
            &combined.chars().take(200).collect::<String>()
        ))
    }
}
