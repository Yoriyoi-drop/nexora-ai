//! SPARO - Self-Play Aligned Reasoning via Prospect-Theoretic Stepwise Optimization
//! 
//! Framework alignment AI yang menggabungkan 6 teknik inovatif:
//! - DPO: Direct Preference Optimization
//! - KTO: Kahneman-Tversky Optimization  
//! - IPO: Identity Preference Optimization
//! - RLVF: Reinforcement Learning from Verifiable Feedback
//! - SPIN: Self-Play with Instruction Following
//! - RLAIF: Reinforcement Learning from AI Feedback

pub mod dpo;
pub mod kto;
pub mod ipo;
pub mod rlvf;
pub mod spin;
pub mod rlaif;
pub mod core;
pub mod trainer;
pub mod data;

// Re-export main components
pub use core::*;
pub use trainer::*;
pub use data::*;

/// Prelude module untuk import umum
pub mod prelude {
    pub use super::dpo::*;
    pub use super::kto::*;
    pub use super::ipo::*;
    pub use super::rlvf::*;
    pub use super::spin::*;
    pub use super::rlaif::*;
    pub use super::core::*;
    pub use super::trainer::*;
    pub use super::data::*;
}
