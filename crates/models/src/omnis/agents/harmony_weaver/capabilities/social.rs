use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialHarmonyOptimization {
    pub group_cohesion_analysis: bool,
    pub communication_pattern_detection: bool,
    pub power_dynamic_assessment: bool,
    pub social_network_mapping: bool,
    pub conflict_prediction: bool,
}

impl Default for SocialHarmonyOptimization {
    fn default() -> Self {
        Self {
            group_cohesion_analysis: true,
            communication_pattern_detection: true,
            power_dynamic_assessment: true,
            social_network_mapping: true,
            conflict_prediction: true,
        }
    }
}

impl SocialHarmonyOptimization {
    pub fn effectiveness_score(&self) -> f32 {
        let mut score = 0.0;
        if self.group_cohesion_analysis { score += 0.2; }
        if self.communication_pattern_detection { score += 0.2; }
        if self.power_dynamic_assessment { score += 0.2; }
        if self.social_network_mapping { score += 0.2; }
        if self.conflict_prediction { score += 0.2; }
        score
    }
}
