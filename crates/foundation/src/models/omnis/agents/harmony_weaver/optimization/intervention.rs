use crate::foundation::models::omnis::agents::harmony_weaver::types::results::InterventionOpportunity;

#[derive(Debug, Clone)]
pub struct InterventionPlanner {
    pub intervention_opportunities: Vec<InterventionOpportunity>,
    pub planning_threshold: f32,
}

impl Default for InterventionPlanner {
    fn default() -> Self {
        Self {
            intervention_opportunities: vec![],
            planning_threshold: 0.6,
        }
    }
}

impl InterventionPlanner {
    pub fn add_opportunity(&mut self, opportunity: InterventionOpportunity) {
        self.intervention_opportunities.push(opportunity);
    }

    pub fn prioritize(&self) -> Vec<InterventionOpportunity> {
        let mut sorted = self.intervention_opportunities.clone();
        sorted.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().filter(|o| o.impact_score >= self.planning_threshold).collect()
    }

    pub fn generate_timeline(&self, opportunities: &[InterventionOpportunity]) -> Vec<String> {
        opportunities.iter().enumerate().map(|(i, opp)| {
            format!("Phase {}: {} (impact: {:.1}%)", i + 1, opp.description, opp.impact_score * 100.0)
        }).collect()
    }
}
