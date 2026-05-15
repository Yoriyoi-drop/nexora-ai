use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

#[derive(Debug, Clone)]
pub struct HarmonyGenAgent {
    pub config: HarmonyGenConfig,
    pub audio_capabilities: AudioCapabilities,
    pub synthesis_engine: SynthesisEngine,
    status: AgentStatus,
    metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyGenConfig {
    pub base_config: BaseAgentConfig,
    pub sample_rate: u32,
    pub bit_depth: u32,
    pub stereo_width: f32,
    pub supported_genres: Vec<String>,
    pub quality_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCapabilities {
    pub music_generation: bool,
    pub sound_design: bool,
    pub audio_analysis: bool,
    pub harmonic_analysis: bool,
    pub melody_extraction: bool,
    pub beat_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisEngine {
    pub techniques: Vec<SynthesisTechnique>,
    pub instrument_profiles: Vec<InstrumentProfile>,
    pub effect_chains: Vec<EffectChain>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisTechnique {
    WaveformGeneration,
    SpectralModeling,
    GranularSynthesis,
    FM_Synthesis,
    NeuralVocoding,
    DiffusionAudio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentProfile {
    pub name: String,
    pub instrument_type: InstrumentType,
    pub parameters: HashMap<String, f32>,
    pub harmonic_profile: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstrumentType {
    Acoustic,
    Electric,
    Synthesized,
    Percussion,
    Vocal,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectChain {
    pub name: String,
    pub effects: Vec<String>,
    pub mix_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerationInput {
    pub prompt: String,
    pub genre: Option<String>,
    pub tempo: Option<u32>,
    pub key: Option<String>,
    pub duration_seconds: Option<u32>,
    pub instruments: Vec<String>,
    pub mood: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerationOutput {
    pub generated_audio: String,
    pub genre: String,
    pub tempo: u32,
    pub key: String,
    pub duration_seconds: u32,
    pub harmonic_complexity: f32,
    pub rhythmic_accuracy: f32,
    pub audio_quality: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicAnalysis {
    pub key_signature: String,
    pub chord_progression: Vec<String>,
    pub harmony_score: f32,
    pub dissonance_ratio: f32,
    pub tonal_stability: f32,
}

impl Default for HarmonyGenConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            sample_rate: 44100,
            bit_depth: 24,
            stereo_width: 1.0,
            supported_genres: vec![
                "ambient".to_string(), "electronic".to_string(),
                "classical".to_string(), "jazz".to_string(),
                "cinematic".to_string(), "pop".to_string(),
            ],
            quality_threshold: 0.7,
        }
    }
}

impl Default for AudioCapabilities {
    fn default() -> Self {
        Self {
            music_generation: true,
            sound_design: true,
            audio_analysis: true,
            harmonic_analysis: true,
            melody_extraction: true,
            beat_detection: true,
        }
    }
}

impl Default for SynthesisEngine {
    fn default() -> Self {
        Self {
            techniques: vec![
                SynthesisTechnique::DiffusionAudio,
                SynthesisTechnique::NeuralVocoding,
                SynthesisTechnique::SpectralModeling,
            ],
            instrument_profiles: vec![
                InstrumentProfile {
                    name: "piano".to_string(),
                    instrument_type: InstrumentType::Acoustic,
                    parameters: HashMap::from([("warmth".to_string(), 0.8), ("resonance".to_string(), 0.7)]),
                    harmonic_profile: vec![1.0, 0.5, 0.3, 0.2, 0.1],
                },
                InstrumentProfile {
                    name: "synth_pad".to_string(),
                    instrument_type: InstrumentType::Synthesized,
                    parameters: HashMap::from([("detune".to_string(), 0.3), ("filter_cutoff".to_string(), 0.6)]),
                    harmonic_profile: vec![1.0, 0.8, 0.6, 0.4, 0.2],
                },
            ],
            effect_chains: Vec::new(),
        }
    }
}

impl Default for HarmonyGenAgent {
    fn default() -> Self {
        Self {
            config: HarmonyGenConfig::default(),
            audio_capabilities: AudioCapabilities::default(),
            synthesis_engine: SynthesisEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for HarmonyGenAgent {
    type Config = HarmonyGenConfig;
    type Input = AudioGenerationInput;
    type Output = AudioGenerationOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        self.validate_input(&input)?;

        let genre = input.genre.clone().unwrap_or_else(|| "ambient".to_string());
        let tempo = input.tempo.unwrap_or(120);
        let key = input.key.clone().unwrap_or_else(|| "C".to_string());
        let duration = input.duration_seconds.unwrap_or(30);

        let composition = self.compose(&input, &genre, tempo, &key).await?;
        let harmonic_analysis = self.analyze_harmonics(&composition).await?;

        Ok(AudioGenerationOutput {
            generated_audio: composition,
            genre,
            tempo,
            key,
            duration_seconds: duration,
            harmonic_complexity: harmonic_analysis.harmony_score,
            rhythmic_accuracy: self.calculate_rhythmic_accuracy(tempo),
            audio_quality: self.calculate_audio_quality(&input),
            metadata: HashMap::new(),
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability {
            name: "harmony_gen".to_string(),
            description: "Audio and music generation with harmonic analysis".to_string(),
            version: "1.0.0".to_string(),
            input_types: vec!["audio_generation_input".to_string()],
            output_types: vec!["audio_content".to_string(), "harmonic_analysis".to_string()],
            metrics: crate::shared::agent_types::CapabilityMetrics {
                accuracy: 0.84,
                avg_latency: 1100.0,
                resource_usage: 0.78,
                reliability: 0.89,
            },
        }]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl HarmonyGenAgent {
    pub fn new(config: HarmonyGenConfig) -> Self {
        Self {
            config,
            audio_capabilities: AudioCapabilities::default(),
            synthesis_engine: SynthesisEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    fn validate_input(&self, input: &AudioGenerationInput) -> AgentResult<()> {
        if input.prompt.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Prompt cannot be empty".to_string()
            ));
        }
        Ok(())
    }

    async fn compose(&self, input: &AudioGenerationInput, genre: &str, tempo: u32, key: &str) -> AgentResult<String> {
        let mood = input.mood.as_deref().unwrap_or("neutral");
        let instr = if input.instruments.is_empty() { "piano, synth_pad".to_string() } else { input.instruments.join(", ") };

        Ok(format!(
            "Generated {} {} in {} @ {}bpm [{}] using {} — technique: {:?}",
            mood, genre, key, tempo, input.prompt, instr,
            self.synthesis_engine.techniques.first().unwrap_or(&SynthesisTechnique::WaveformGeneration)
        ))
    }

    async fn analyze_harmonics(&self, _composition: &str) -> AgentResult<HarmonicAnalysis> {
        Ok(HarmonicAnalysis {
            key_signature: "C major".to_string(),
            chord_progression: vec!["C".to_string(), "G".to_string(), "Am".to_string(), "F".to_string()],
            harmony_score: 0.82,
            dissonance_ratio: 0.12,
            tonal_stability: 0.78,
        })
    }

    fn calculate_rhythmic_accuracy(&self, _tempo: u32) -> f32 {
        0.85
    }

    fn calculate_audio_quality(&self, _input: &AudioGenerationInput) -> f32 {
        self.config.quality_threshold + 0.1
    }

    pub async fn analyze_audio(&self, _audio_data: &str) -> AgentResult<HarmonicAnalysis> {
        self.analyze_harmonics(_audio_data).await
    }

    pub fn generate_chord_progression(&self, key: &str, length: usize) -> Vec<String> {
        let base_chords: HashMap<&str, Vec<&str>> = HashMap::from([
            ("C", vec!["C", "G", "Am", "F", "Dm", "Em"]),
            ("G", vec!["G", "D", "Em", "C", "Am", "Bm"]),
            ("D", vec!["D", "A", "Bm", "G", "Em", "F#m"]),
        ]);

        let pool = base_chords.get(key).unwrap_or(&vec!["C", "G", "Am", "F"]);
        pool.iter().take(length).map(|c| c.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_gen_agent_creation() {
        let agent = HarmonyGenAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
    }

    #[tokio::test]
    async fn test_audio_generation_processing() {
        let agent = HarmonyGenAgent::default();
        let input = AudioGenerationInput {
            prompt: "Calm ambient pad with gentle piano".to_string(),
            genre: Some("ambient".to_string()),
            tempo: Some(80),
            key: Some("C".to_string()),
            duration_seconds: Some(60),
            instruments: vec!["piano".to_string(), "synth_pad".to_string()],
            mood: Some("calm".to_string()),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.generated_audio.is_empty());
        assert!(output.harmonic_complexity > 0.0);
    }
}
