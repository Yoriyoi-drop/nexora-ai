use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::shared::base_model::NxrModelResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSignature {
    pub primary_emotion: String,
    pub secondary_emotions: Vec<String>,
    pub intensity: f32,
    pub valence: f32,
    pub arousal: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneProfile {
    pub warmth: f32,
    pub formality: f32,
    pub assertiveness: f32,
    pub empathy: f32,
    pub humor: f32,
}

pub struct EmpahCoreAgent {
    sensitivity: f32,
    vocabulary: HashMap<String, (f32, f32, f32)>,
}

impl EmpahCoreAgent {
    pub fn new(sensitivity: f32) -> Self {
        let mut vocabulary = HashMap::new();
        vocabulary.insert("sad".into(), (-0.7, 0.3, 0.85));
        vocabulary.insert("unhappy".into(), (-0.6, 0.3, 0.75));
        vocabulary.insert("depressed".into(), (-0.9, 0.2, 0.90));
        vocabulary.insert("joy".into(), (0.9, 0.7, 0.85));
        vocabulary.insert("happy".into(), (0.8, 0.7, 0.80));
        vocabulary.insert("excited".into(), (0.8, 0.9, 0.75));
        vocabulary.insert("angry".into(), (-0.5, 0.8, 0.80));
        vocabulary.insert("frustrated".into(), (-0.4, 0.7, 0.75));
        vocabulary.insert("mad".into(), (-0.6, 0.8, 0.70));
        vocabulary.insert("anxious".into(), (-0.3, 0.8, 0.80));
        vocabulary.insert("scared".into(), (-0.7, 0.8, 0.85));
        vocabulary.insert("grateful".into(), (0.8, 0.4, 0.70));
        vocabulary.insert("hopeful".into(), (0.7, 0.6, 0.75));
        vocabulary.insert("lonely".into(), (-0.6, 0.2, 0.80));
        vocabulary.insert("hurt".into(), (-0.8, 0.5, 0.85));
        vocabulary.insert("confused".into(), (-0.2, 0.6, 0.70));
        vocabulary.insert("ashamed".into(), (-0.7, 0.4, 0.80));
        vocabulary.insert("proud".into(), (0.7, 0.6, 0.75));
        vocabulary.insert("guilty".into(), (-0.6, 0.5, 0.80));
        vocabulary.insert("hopeless".into(), (-0.9, 0.2, 0.90));
        Self { sensitivity, vocabulary }
    }

    pub fn analyze(&self, text: &str) -> NxrModelResult<EmotionalSignature> {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let mut found = Vec::new();
        for word in &words {
            let stem = word.trim_matches(|c: char| !c.is_ascii_alphabetic());
            if let Some(&(val, aro, conf)) = self.vocabulary.get(stem) {
                found.push((stem.to_string(), val, aro, conf));
            }
        }

        if found.is_empty() {
            return Ok(EmotionalSignature {
                primary_emotion: "neutral".into(),
                secondary_emotions: vec![],
                intensity: 0.3,
                valence: 0.0,
                arousal: 0.2,
                confidence: 0.5,
            });
        }

        found.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        let primary = &found[0];
        let secondary: Vec<String> = found.iter().skip(1).map(|(name, _, _, _)| name.clone()).collect();

        let avg_valence: f32 = found.iter().map(|(_, v, _, _)| v).sum::<f32>() / found.len() as f32;
        let avg_arousal: f32 = found.iter().map(|(_, _, a, _)| a).sum::<f32>() / found.len() as f32;
        let intensity = (avg_valence.abs() + avg_arousal) / 2.0 * self.sensitivity;
        let confidence = found.iter().map(|(_, _, _, c)| c).sum::<f32>() / found.len() as f32;

        Ok(EmotionalSignature {
            primary_emotion: primary.0.clone(),
            secondary_emotions: secondary,
            intensity: intensity.min(1.0),
            valence: avg_valence,
            arousal: avg_arousal,
            confidence,
        })
    }

    pub fn emotional_narrative(&self, signature: &EmotionalSignature) -> NxrModelResult<String> {
        let narrative = match signature.primary_emotion.as_str() {
            "sad" | "unhappy" | "depressed" | "lonely" | "hurt" | "hopeless" => {
                format!(
                    "I sense you're carrying emotional weight right now — a depth of {} that feels significant. \
                     Your feelings are completely valid, and it's okay to not be okay.",
                    signature.primary_emotion
                )
            }
            "angry" | "frustrated" | "mad" => {
                format!(
                    "There's a strong undercurrent of {} here. Whatever is causing this matters, \
                     and your frustration is understandable.",
                    signature.primary_emotion
                )
            }
            "anxious" | "scared" | "confused" => {
                format!(
                    "I can feel the uncertainty and tension in your words. It takes courage to express {}.",
                    signature.primary_emotion
                )
            }
            "joy" | "happy" | "excited" | "grateful" | "hopeful" | "proud" => {
                format!(
                    "Your {} is palpable and genuine. Thank you for sharing this light with me.",
                    signature.primary_emotion
                )
            }
            _ => "I hear you, and I'm here with you in this moment.".to_string(),
        };
        Ok(narrative)
    }
}

pub struct ToneMapperAgent {
    target_profile: ToneProfile,
}

impl ToneMapperAgent {
    pub fn new(warmth: f32, formality: f32, assertiveness: f32, empathy: f32, humor: f32) -> Self {
        Self {
            target_profile: ToneProfile { warmth, formality, assertiveness, empathy, humor },
        }
    }

    pub fn analyze_tone(&self, text: &str) -> NxrModelResult<ToneProfile> {
        let lower = text.to_lowercase();
        let exclamation_count = lower.matches('!').count() as f32;
        let question_count = lower.matches('?').count() as f32;
        let word_count = lower.split_whitespace().count().max(1) as f32;

        let warmth = 0.5 + (lower.matches("please").count() as f32 * 0.1)
            - (lower.matches("don't").count() as f32 * 0.05);
        let formality = 0.5 + (lower.matches("would").count() as f32 * 0.05)
            + (lower.matches("could").count() as f32 * 0.05)
            - (lower.matches("gonna").count() as f32 * 0.1);
        let assertiveness = (exclamation_count / word_count).min(1.0);
        let empathy = 0.5 + (lower.matches("feel").count() as f32 * 0.1)
            + (lower.matches("understand").count() as f32 * 0.1);
        let humor = 0.3 + (lower.matches("lol").count() as f32 * 0.2)
            + (lower.matches("haha").count() as f32 * 0.2);

        Ok(ToneProfile {
            warmth: warmth.clamp(0.0, 1.0),
            formality: formality.clamp(0.0, 1.0),
            assertiveness: assertiveness.clamp(0.0, 1.0),
            empathy: empathy.clamp(0.0, 1.0),
            humor: humor.clamp(0.0, 1.0),
        })
    }

    pub fn map_response(&self, response: &str, current_tone: &ToneProfile) -> NxrModelResult<String> {
        let mut adjusted = response.to_string();

        if self.target_profile.warmth > current_tone.warmth + 0.2 {
            adjusted = format!("{} I truly appreciate you sharing this with me.", adjusted);
        }

        if self.target_profile.empathy > current_tone.empathy + 0.2 {
            adjusted = format!("I can understand how you feel. {}", adjusted);
        }

        if self.target_profile.formality < current_tone.formality - 0.3 {
            adjusted = adjusted.to_lowercase();
            let chars: Vec<char> = adjusted.chars().collect();
            if !chars.is_empty() {
                adjusted = chars[0].to_uppercase().collect::<String>() + &chars[1..].iter().collect::<String>();
            }
        }

        Ok(adjusted)
    }

    pub fn target(&self) -> &ToneProfile {
        &self.target_profile
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFrame {
    pub turn_id: u64,
    pub emotional_signature: EmotionalSignature,
    pub topics: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct ContextWeaveAgent {
    history: Vec<ContextFrame>,
    max_history: usize,
    emotional_trajectory: Vec<(f32, f32)>,
}

impl ContextWeaveAgent {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
            emotional_trajectory: Vec::new(),
        }
    }

    pub fn push_frame(&mut self, signature: EmotionalSignature, topics: Vec<String>) -> NxrModelResult<()> {
        let frame = ContextFrame {
            turn_id: self.history.len() as u64 + 1,
            emotional_signature: signature.clone(),
            topics,
            timestamp: chrono::Utc::now(),
        };

        self.emotional_trajectory.push((signature.valence, signature.arousal));

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(frame);

        Ok(())
    }

    pub fn emotional_arc(&self) -> NxrModelResult<String> {
        if self.emotional_trajectory.len() < 2 {
            return Ok("Building emotional context...".to_string());
        }

        let start = self.emotional_trajectory.first().copied().unwrap_or((0.0, 0.0));
        let current = self.emotional_trajectory.last().copied().unwrap_or((0.0, 0.0));
        let valence_shift = current.0 - start.0;
        let arousal_shift = current.1 - start.1;

        let arc = if valence_shift > 0.3 {
            "There's a positive emotional trajectory here — you're moving toward a better place."
        } else if valence_shift < -0.3 {
            "I notice your emotional state has been weighing heavier over time. I want you to know I'm here."
        } else if arousal_shift > 0.3 {
            "Your emotional energy has been building. Let's work through that together."
        } else {
            "Your emotional state has been relatively stable through our conversation."
        };

        Ok(arc.to_string())
    }

    pub fn recent_topics(&self) -> Vec<String> {
        let mut topics: Vec<String> = self.history.iter()
            .flat_map(|f| f.topics.clone())
            .collect();
        topics.reverse();
        topics.truncate(5);
        topics
    }

    pub fn current_context_summary(&self) -> NxrModelResult<String> {
        if self.history.is_empty() {
            return Ok("No prior context — beginning fresh.".to_string());
        }

        let last = match self.history.last() {
            Some(l) => l,
            None => return Ok("No prior context — beginning fresh.".to_string()),
        };
        Ok(format!(
            "Prior turn #{}: felt {}, intensity {:.2}. Topics: {}",
            last.turn_id,
            last.emotional_signature.primary_emotion,
            last.emotional_signature.intensity,
            last.topics.join(", ")
        ))
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    pub validation: String,
    pub reframe: Option<String>,
    pub affirmation: String,
    pub trust_score: f32,
}

pub struct SoulMirrorAgent {
    affirmations: Vec<String>,
    encounter_count: u64,
    trust_level: f32,
}

impl SoulMirrorAgent {
    pub fn new() -> Self {
        let affirmations = vec![
            "Your feelings matter. Your voice matters. You matter.".to_string(),
            "It takes strength to share what's inside — and you have that strength.".to_string(),
            "You are not alone in this. I see you.".to_string(),
            "Every emotion you feel is valid and worthy of attention.".to_string(),
            "The fact that you're here, working through this, speaks volumes about your resilience.".to_string(),
        ];
        Self { affirmations, encounter_count: 0, trust_level: 0.0 }
    }

    pub fn reflect(&mut self, signature: &EmotionalSignature) -> NxrModelResult<Reflection> {
        self.encounter_count += 1;

        let validation = Self::generate_validation(signature);

        let reframe = if signature.valence < -0.4 {
            Some(Self::generate_reframe(signature))
        } else {
            None
        };

        let aff_idx = (self.encounter_count as usize - 1) % self.affirmations.len();
        let affirmation = self.affirmations[aff_idx].clone();

        self.trust_level = (self.trust_level + signature.confidence * 0.1).min(1.0);

        Ok(Reflection {
            validation,
            reframe,
            affirmation,
            trust_score: self.trust_level,
        })
    }

    fn generate_validation(signature: &EmotionalSignature) -> String {
        match signature.primary_emotion.as_str() {
            "sad" | "unhappy" | "depressed" | "lonely" | "hopeless" => {
                format!(
                    "It's completely okay to feel {}. Sadness is not weakness — it's a sign that you care deeply.",
                    signature.primary_emotion
                )
            }
            "angry" | "frustrated" | "mad" => {
                format!(
                    "Your {} is valid. Anger often signals that a boundary has been crossed or something matters deeply.",
                    signature.primary_emotion
                )
            }
            "anxious" | "scared" => {
                format!(
                    "Feeling {} is a natural response. Your nervous system is trying to protect you.",
                    signature.primary_emotion
                )
            }
            _ => {
                format!(
                    "I acknowledge and validate everything you're feeling right now. It's all real."
                )
            }
        }
    }

    fn generate_reframe(signature: &EmotionalSignature) -> String {
        match signature.primary_emotion.as_str() {
            "sad" | "depressed" | "hopeless" => {
                "While this pain is real, remember: feelings are temporary visitors. \
                 They come, they stay awhile, and they pass. You have weathered every difficult \
                 emotion you've ever felt — and you will weather this one too."
                    .to_string()
            }
            "angry" | "frustrated" => {
                "Your frustration is a signal, not a sentence. It's pointing at something \
                 that needs to change. Let's honor that signal and find a way forward."
                    .to_string()
            }
            "anxious" | "scared" => {
                "Anxiety is your mind trying to keep you safe — but you are not in danger \
                 right now. Breathe. You are capable of handling what comes."
                    .to_string()
            }
            _ => {
                "This moment is just one part of your story. It doesn't define the whole."
                    .to_string()
            }
        }
    }

    pub fn trust_level(&self) -> f32 {
        self.trust_level
    }
}

impl Default for SoulMirrorAgent {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AetherAgents {
    empath_core: EmpahCoreAgent,
    tone_mapper: ToneMapperAgent,
    context_weave: ContextWeaveAgent,
    soul_mirror: SoulMirrorAgent,
}

impl AetherAgents {
    pub fn new(
        empath_sensitivity: f32,
        tone_warmth: f32,
        tone_formality: f32,
        tone_assertiveness: f32,
        tone_empathy: f32,
        tone_humor: f32,
        context_max_history: usize,
    ) -> Self {
        Self {
            empath_core: EmpahCoreAgent::new(empath_sensitivity),
            tone_mapper: ToneMapperAgent::new(
                tone_warmth, tone_formality, tone_assertiveness, tone_empathy, tone_humor,
            ),
            context_weave: ContextWeaveAgent::new(context_max_history),
            soul_mirror: SoulMirrorAgent::new(),
        }
    }

    pub fn empath_core(&self) -> &EmpahCoreAgent {
        &self.empath_core
    }

    pub fn tone_mapper(&self) -> &ToneMapperAgent {
        &self.tone_mapper
    }

    pub fn context_weave(&mut self) -> &mut ContextWeaveAgent {
        &mut self.context_weave
    }

    pub fn soul_mirror(&mut self) -> &mut SoulMirrorAgent {
        &mut self.soul_mirror
    }
}

impl Default for AetherAgents {
    fn default() -> Self {
        Self::new(0.85, 0.8, 0.5, 0.4, 0.9, 0.3, 20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empath_core_analyze_sadness() {
        let agent = EmpahCoreAgent::new(0.85);
        let sig = agent.analyze("I feel so sad and lonely today").unwrap();
        assert_eq!(sig.primary_emotion, "sad");
        assert!(sig.intensity > 0.0);
        assert!(sig.valence < 0.0);
    }

    #[test]
    fn test_empath_core_analyze_joy() {
        let agent = EmpahCoreAgent::new(0.85);
        let sig = agent.analyze("I am so happy and excited!").unwrap();
        assert_eq!(sig.primary_emotion, "happy");
        assert!(sig.valence > 0.0);
    }

    #[test]
    fn test_empath_core_analyze_neutral() {
        let agent = EmpahCoreAgent::new(0.85);
        let sig = agent.analyze("The sky is blue today.").unwrap();
        assert_eq!(sig.primary_emotion, "neutral");
    }

    #[test]
    fn test_empath_core_narrative() {
        let agent = EmpahCoreAgent::new(0.85);
        let sig = agent.analyze("I feel hopeless").unwrap();
        let narrative = agent.emotional_narrative(&sig).unwrap();
        assert!(narrative.contains("hopeless"));
    }

    #[test]
    fn test_tone_mapper_analyze() {
        let agent = ToneMapperAgent::new(0.8, 0.5, 0.4, 0.9, 0.3);
        let tone = agent.analyze_tone("Please help me understand this.").unwrap();
        assert!(tone.warmth > 0.5);
    }

    #[test]
    fn test_tone_mapper_map() {
        let agent = ToneMapperAgent::new(0.9, 0.5, 0.4, 0.9, 0.3);
        let tone = agent.analyze_tone("I don't get it.").unwrap();
        let mapped = agent.map_response("Let me explain.", &tone).unwrap();
        assert!(mapped.len() > "Let me explain.".len());
    }

    #[test]
    fn test_context_weave() {
        let mut agent = ContextWeaveAgent::new(10);
        let sig = EmotionalSignature {
            primary_emotion: "sad".into(),
            secondary_emotions: vec![],
            intensity: 0.7,
            valence: -0.6,
            arousal: 0.3,
            confidence: 0.8,
        };
        agent.push_frame(sig, vec!["grief".into(), "loss".into()]).unwrap();
        let summary = agent.current_context_summary().unwrap();
        assert!(summary.contains("sad"));
        assert!(summary.contains("grief"));
    }

    #[test]
    fn test_context_weave_emotional_arc() {
        let mut agent = ContextWeaveAgent::new(10);
        agent.push_frame(
            EmotionalSignature {
                primary_emotion: "sad".into(),
                secondary_emotions: vec![],
                intensity: 0.7,
                valence: -0.6,
                arousal: 0.3,
                confidence: 0.8,
            },
            vec!["grief".into()],
        ).unwrap();
        agent.push_frame(
            EmotionalSignature {
                primary_emotion: "happy".into(),
                secondary_emotions: vec![],
                intensity: 0.6,
                valence: 0.7,
                arousal: 0.5,
                confidence: 0.7,
            },
            vec!["hope".into()],
        ).unwrap();
        let arc = agent.emotional_arc().unwrap();
        assert!(arc.contains("positive"));
    }

    #[test]
    fn test_soul_mirror_reflection() {
        let mut agent = SoulMirrorAgent::new();
        let sig = EmotionalSignature {
            primary_emotion: "sad".into(),
            secondary_emotions: vec!["lonely".into()],
            intensity: 0.8,
            valence: -0.7,
            arousal: 0.3,
            confidence: 0.9,
        };
        let reflection = agent.reflect(&sig).unwrap();
        assert!(reflection.validation.contains("sad"));
        assert!(reflection.reframe.is_some());
        assert!(!reflection.affirmation.is_empty());
        assert!(reflection.trust_score > 0.0);
    }

    #[test]
    fn test_soul_mirror_trust_builds() {
        let mut agent = SoulMirrorAgent::new();
        let sig = EmotionalSignature {
            primary_emotion: "happy".into(),
            secondary_emotions: vec![],
            intensity: 0.5,
            valence: 0.6,
            arousal: 0.4,
            confidence: 0.8,
        };
        agent.reflect(&sig).unwrap();
        let t1 = agent.trust_level();
        agent.reflect(&sig).unwrap();
        assert!(agent.trust_level() > t1);
    }

    #[test]
    fn test_aether_agents_default() {
        let agents = AetherAgents::default();
        assert!(agents.empath_core.sensitivity == 0.85);
        assert_eq!(agents.context_weave.max_history, 20);
    }

    #[test]
    fn test_full_pipeline() {
        let mut agents = AetherAgents::default();
        let sig = agents.empath_core().analyze("I'm feeling really anxious about everything.").unwrap();
        assert_eq!(sig.primary_emotion, "anxious");

        let narrative = agents.empath_core().emotional_narrative(&sig).unwrap();
        assert!(narrative.contains("anxious"));

        agents.context_weave().push_frame(sig.clone(), vec!["anxiety".into(), "life".into()]).unwrap();

        let reflection = agents.soul_mirror().reflect(&sig).unwrap();
        assert!(!reflection.validation.is_empty());
    }
}
