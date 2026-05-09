//! NXR-GENESIS Agents Module
//! 
//! Individual agent implementations for creation and innovation

pub mod creation_architect;
pub mod genesis_prime;
pub mod innovation_catalyst;
pub mod origin_forge;
pub mod prime_creator;
pub mod system_breeder;

// Re-export all agents
pub use creation_architect::*;
pub use genesis_prime::*;
pub use innovation_catalyst::*;
pub use origin_forge::*;
pub use prime_creator::*;
pub use system_breeder::*;
