//! Logic Flow & Training Dynamics
//!
//! Node khusus untuk logika pelatihan: Condition Node, Recurrent Loop Node,
//! Adaptive Scheduler Node, Reinforcement Feedback Node, Context Memory Node.
//! Memungkinkan adaptive training, curriculum learning, GAN loop, meta-learning, RL pipeline.

pub mod condition;
pub mod context_memory;
pub mod loop_node;
pub mod rl_feedback;
pub mod scheduler_node;

pub use condition::*;
pub use context_memory::*;
pub use loop_node::*;
pub use rl_feedback::*;
pub use scheduler_node::*;

/// Tipe logic node
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicNodeType {
    Condition,
    RecurrentLoop,
    AdaptiveScheduler,
    RLFeedback,
    ContextMemory,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logic_node_type_debug() {
        let t = LogicNodeType::Condition;
        assert!(!format!("{:?}", t).is_empty());
    }
}
