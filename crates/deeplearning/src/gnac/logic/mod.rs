//! Logic Flow & Training Dynamics
//!
//! Node khusus untuk logika pelatihan: Condition Node, Recurrent Loop Node,
//! Adaptive Scheduler Node, Reinforcement Feedback Node, Context Memory Node.
//! Memungkinkan adaptive training, curriculum learning, GAN loop, meta-learning, RL pipeline.

pub mod condition;
pub mod loop_node;
pub mod scheduler_node;
pub mod rl_feedback;
pub mod context_memory;

pub use condition::*;
pub use loop_node::*;
pub use scheduler_node::*;
pub use rl_feedback::*;
pub use context_memory::*;

use crate::DLResult;

/// Tipe logic node
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicNodeType {
    Condition,
    RecurrentLoop,
    AdaptiveScheduler,
    RLFeedback,
    ContextMemory,
}
