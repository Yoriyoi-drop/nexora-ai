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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_greater_than() {
        let c = ConditionNode::new("gt", ConditionType::GreaterThan, 10.0);
        assert!(c.evaluate(11.0));
        assert!(!c.evaluate(10.0));
    }

    #[test]
    fn test_condition_less_than() {
        let c = ConditionNode::new("lt", ConditionType::LessThan, 10.0);
        assert!(c.evaluate(5.0));
        assert!(!c.evaluate(10.0));
    }

    #[test]
    fn test_condition_equal_to() {
        let c = ConditionNode::new("eq", ConditionType::EqualTo, 10.0);
        assert!(c.evaluate(10.0));
        assert!(!c.evaluate(10.1));
    }

    #[test]
    fn test_condition_range() {
        let c = ConditionNode::new(
            "range",
            ConditionType::Range {
                min: 0.0,
                max: 10.0,
            },
            0.0,
        );
        assert!(c.evaluate(5.0));
        assert!(c.evaluate(0.0));
        assert!(c.evaluate(10.0));
        assert!(!c.evaluate(-1.0));
        assert!(!c.evaluate(11.0));
    }

    #[test]
    fn test_condition_loss_plateau() {
        let c = ConditionNode::new("plateau", ConditionType::LossPlateau { patience: 5 }, 0.01);
        assert!(c.evaluate(0.1));
        assert!(!c.evaluate(0.001));
    }
}
