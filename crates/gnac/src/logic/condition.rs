use uuid::Uuid;

/// Condition Node — eksekusi berdasarkan kondisi
#[derive(Debug, Clone)]
pub struct ConditionNode {
    pub id: Uuid,
    pub name: String,
    pub condition_type: ConditionType,
    pub threshold: f64,
    pub true_branch: Option<Uuid>,
    pub false_branch: Option<Uuid>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConditionType {
    GreaterThan,
    LessThan,
    EqualTo,
    Range { min: f64, max: f64 },
    GradientNorm,
    LossPlateau { patience: usize },
}

impl ConditionNode {
    pub fn new(name: &str, condition_type: ConditionType, threshold: f64) -> Self {
        ConditionNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            condition_type,
            threshold,
            true_branch: None,
            false_branch: None,
        }
    }

    pub fn evaluate(&self, value: f64) -> bool {
        match self.condition_type {
            ConditionType::GreaterThan => value > self.threshold,
            ConditionType::LessThan => value < self.threshold,
            ConditionType::EqualTo => (value - self.threshold).abs() < 1e-6,
            ConditionType::Range { min, max } => value >= min && value <= max,
            ConditionType::GradientNorm => value > self.threshold,
            ConditionType::LossPlateau { .. } => value > self.threshold,
        }
    }
}
