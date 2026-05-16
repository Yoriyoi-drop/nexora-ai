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

    /// Hitung policy gradient (simulated)
    pub fn policy_gradient(&self, log_prob: f64) -> f64 {
        self.learning_rate * self.cumulative_reward * log_prob
    }

    pub fn reset(&mut self) {
        self.cumulative_reward = 0.0;
        self.episode_count = 0;
    }
}
