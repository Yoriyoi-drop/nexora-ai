use crate::foundation::models::omnis::agents::harmony_weaver::types::analysis::HarmonyAssessment;

#[derive(Debug, Clone)]
pub struct HarmonyOptimizer {
    pub assessment: Option<HarmonyAssessment>,
    pub optimization_threshold: f32,
    pub improvement_potential: f32,
    pub recommended_actions: Vec<String>,
}

impl Default for HarmonyOptimizer {
    fn default() -> Self {
        Self {
            assessment: None,
            optimization_threshold: 0.7,
            improvement_potential: 0.0,
            recommended_actions: vec![],
        }
    }
}

impl HarmonyOptimizer {
    pub fn from_assessment(assessment: HarmonyAssessment) -> Self {
        let improvement_potential = 1.0 - assessment.harmony_score;
        Self {
            assessment: Some(assessment),
            optimization_threshold: 0.7,
            improvement_potential,
            recommended_actions: vec![],
        }
    }

    pub fn needs_optimization(&self) -> bool {
        self.assessment
            .as_ref()
            .map(|a| a.harmony_score < self.optimization_threshold)
            .unwrap_or(true)
    }

    pub fn generate_optimization_plan(&mut self) -> Vec<String> {
        if let Some(ref assessment) = self.assessment {
            if assessment.harmony_score < 0.5 {
                self.recommended_actions.push("Critical: Immediate intervention required to improve group harmony".to_string());
            } else if assessment.harmony_score < 0.7 {
                self.recommended_actions.push("Moderate: Targeted improvements in communication patterns needed".to_string());
            } else {
                self.recommended_actions.push("Maintenance: Continue monitoring and reinforcing positive dynamics".to_string());
            }

            if assessment.emotional_landscape.emotional_stability < 0.5 {
                self.recommended_actions.push("Address emotional volatility through regulation exercises".to_string());
            }

            if assessment.social_dynamics.group_cohesion < 0.6 {
                self.recommended_actions.push("Strengthen group cohesion through collaborative activities".to_string());
            }
        }
        self.recommended_actions.clone()
    }
}
