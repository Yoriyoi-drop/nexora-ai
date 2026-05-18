pub mod config;
pub mod rms_norm;
pub mod rope;
pub mod gqa;
pub mod swiglu;
pub mod block;
pub mod model;
pub mod trainable;
pub mod mtp;
pub mod safetensors;

pub use config::TransformerConfig;
pub use model::CausalLM;
pub use gqa::{KVCacheEntry, PagedCacheReader};
pub use rms_norm::RMSNorm;
pub use rope::RoPE;
pub use trainable::TrainableCausalLM;
pub use mtp::{MTPConfig, MTPHeads, MTPInference};

pub use model::sample_token;

pub type TransformerResult<T> = std::result::Result<T, TransformerError>;

#[derive(Debug, thiserror::Error)]
pub enum TransformerError {
    #[error("Implementation error: {0}")]
    Implementation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Not implemented")]
    NotImplemented,
}
