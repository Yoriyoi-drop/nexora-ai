use uuid::Uuid;

/// Reinforcement Feedback Node — RL-based reward feedback loop
#[derive(Debug, Clone)]
pub struct RLFeedbackNode {
    pub id: Uuid,
    pub name: String,
    pub cumulative_reward: f64,
    pub discount_factor: f64,
    pub learning_rate: f64,
    pub episode_count: usize,
}

impl RLFeedbackNode {
    pub fn new(name: &str, discount_factor: f64, learning_rate: f64) -> Self {
        RLFeedbackNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            cumulative_reward: 0.0,
            discount_factor,
            learning_rate,
            episode_count: 0,
        }
    }

    /// Apply reward ke model
    pub fn apply_reward(&mut self, reward: f64) -> f64 {
        self.cumulative_reward = self.cumulative_reward * self.discount_factor + reward;
        self.episode_count += 1;
        self.cumulative_reward
    }

    /// Hitung policy gradient via REINFORCE
    pub fn policy_gradient(&self, log_prob: f64) -> f64 {
        self.learning_rate * self.cumulative_reward * log_prob
    }

    pub fn reset(&mut self) {
        self.cumulative_reward = 0.0;
        self.episode_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_feedback_new() {
        let rl = RLFeedbackNode::new("rl", 0.9, 0.01);
        assert_eq!(rl.cumulative_reward, 0.0);
        assert_eq!(rl.discount_factor, 0.9);
    }

    #[test]
    fn test_apply_reward() {
        let mut rl = RLFeedbackNode::new("rl", 0.9, 0.01);
        rl.apply_reward(10.0);
        assert!((rl.cumulative_reward - 10.0).abs() < 1e-5);
        assert_eq!(rl.episode_count, 1);
    }

    #[test]
    fn test_policy_gradient() {
        let mut rl = RLFeedbackNode::new("rl", 1.0, 0.1);
        rl.apply_reward(5.0);
        let pg = rl.policy_gradient(2.0);
        assert!((pg - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_reset() {
        let mut rl = RLFeedbackNode::new("rl", 0.9, 0.01);
        rl.apply_reward(10.0);
        rl.reset();
        assert_eq!(rl.cumulative_reward, 0.0);
        assert_eq!(rl.episode_count, 0);
    }
}
