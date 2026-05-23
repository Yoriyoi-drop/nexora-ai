pub mod block;
pub mod config;
pub mod gqa;
pub mod model;
pub mod mtp;
pub mod rms_norm;
pub mod rope;
pub mod safetensors;
pub mod swiglu;
pub mod trainable;

pub use config::TransformerConfig;
pub use gqa::{KVCacheEntry, KVCacheProvider, CpuKVCache, PagedCacheReader};
#[cfg(feature = "gpu")]
pub use gqa::{GpuKVCache, GpuKVCacheEntry};
pub use model::{CausalLM, LayerInjector};
pub use mtp::{MTPConfig, MTPHeads, MTPInference};
pub use rms_norm::RMSNorm;
pub use rope::RoPE;
pub use trainable::TrainableCausalLM;

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
