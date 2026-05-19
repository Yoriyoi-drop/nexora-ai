use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    pub mediation_capability: bool,
    pub negotiation_facilitation: bool,
    pub root_cause_analysis: bool,
    pub de_escalation_techniques: bool,
    pub restorative_practices: bool,
}

impl Default for ConflictResolution {
    fn default() -> Self {
        Self {
            mediation_capability: true,
            negotiation_facilitation: true,
            root_cause_analysis: true,
            de_escalation_techniques: true,
            restorative_practices: true,
        }
    }
}

impl ConflictResolution {
    pub fn capability_score(&self) -> f32 {
        let mut score = 0.0;
        if self.mediation_capability { score += 0.2; }
        if self.negotiation_facilitation { score += 0.2; }
        if self.root_cause_analysis { score += 0.2; }
        if self.de_escalation_techniques { score += 0.2; }
        if self.restorative_practices { score += 0.2; }
        score
    }
}
