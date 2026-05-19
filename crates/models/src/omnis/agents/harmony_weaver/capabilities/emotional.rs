use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligenceCapabilities {
    pub emotion_detection: bool,
    pub sentiment_analysis: bool,
    pub empathy_generation: bool,
    pub emotional_regulation_support: bool,
    pub emotional_awareness: bool,
}

impl Default for EmotionalIntelligenceCapabilities {
    fn default() -> Self {
        Self {
            emotion_detection: true,
            sentiment_analysis: true,
            empathy_generation: true,
            emotional_regulation_support: true,
            emotional_awareness: true,
        }
    }
}

impl EmotionalIntelligenceCapabilities {
    pub fn score(&self) -> f32 {
        let mut score = 0.0;
        if self.emotion_detection { score += 0.2; }
        if self.sentiment_analysis { score += 0.2; }
        if self.empathy_generation { score += 0.2; }
        if self.emotional_regulation_support { score += 0.2; }
        if self.emotional_awareness { score += 0.2; }
        score
    }
}
