use crate::foundation::models::omnis::agents::harmony_weaver::types::results::IdentifiedConflict;

#[derive(Debug, Clone)]
pub struct ConflictPlanner {
    pub identified_conflicts: Vec<IdentifiedConflict>,
    pub resolution_success_threshold: f32,
}

impl Default for ConflictPlanner {
    fn default() -> Self {
        Self {
            identified_conflicts: vec![],
            resolution_success_threshold: 0.7,
        }
    }
}

impl ConflictPlanner {
    pub fn add_conflict(&mut self, conflict: IdentifiedConflict) {
        self.identified_conflicts.push(conflict);
    }

    pub fn triage_conflicts(&self) -> Vec<IdentifiedConflict> {
        let mut sorted = self.identified_conflicts.clone();
        sorted.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    pub fn calculate_overall_success_probability(&self) -> f32 {
        if self.identified_conflicts.is_empty() {
            return 1.0;
        }
        let total_severity: f32 = self.identified_conflicts.iter().map(|c| c.severity).sum();
        let avg_severity = total_severity / self.identified_conflicts.len() as f32;
        (1.0 - avg_severity).max(0.0).min(1.0)
    }
}
