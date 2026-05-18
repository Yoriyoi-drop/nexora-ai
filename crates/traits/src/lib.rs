// Foundation trait definitions for Nexora AI
// 
// Modular framework interfaces and trait definitions

pub mod core_traits;
pub mod tensor_traits;
pub mod model_traits;

// Re-export main traits
pub use core_traits::*;
pub use tensor_traits::*;
pub use model_traits::*;
