pub mod backbone_registry;
pub mod block;
pub mod kv_cache_compression;
pub mod lazy_weights;
pub mod nccl_collective;
pub mod sharded;
pub mod config;
pub mod gqa;
pub mod model;
pub mod mtp;
pub mod observer;
pub mod quantized;
pub mod rms_norm;
pub mod rope;
pub mod safetensors;
pub mod swiglu;
pub mod trainable;

pub use backbone_registry::{resolve_tier_backbone, resolve_tier_backbone_with_config, clear_all_backbones, has_tier_backbone, tier_parameter_count, unload_tier_backbone, get_loaded_tiers, tier_vram_estimate_mb};
pub use config::TransformerConfig;
pub use gqa::{CpuKVCache, KVCacheEntry, KVCacheProvider, PagedCacheReader};
#[cfg(feature = "gpu")]
pub use gqa::{GpuKVCache, GpuKVCacheEntry};
pub use model::{CausalLM, LayerInjector};
pub use observer::{WeightNotifier, WeightObserver};
pub use mtp::{MTPConfig, MTPHeads, MTPInference};
pub use rms_norm::RMSNorm;
pub use rope::RoPE;
pub use trainable::TrainableCausalLM;

pub use model::sample_token;

// ─── F16 conversion utilities ────────────────────────────────────────────────

pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0 {
        sign | (mant >> 13) as u16
    } else if exp == 255 {
        sign | (0x7c00 | ((mant >> 13) as u16))
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            sign | (0x7c00 | ((mant >> 13) as u16))
        } else if new_exp <= 0 {
            sign | ((mant | 0x800000) >> (13 - new_exp + 1)) as u16
        } else {
            sign | ((new_exp as u16) << 10) | (mant >> 13) as u16
        }
    }
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            f32::from_bits(sign) // flush subnormals to zero
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xff << 23) | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp - 15 + 127) << 23) | (mant << 13))
    }
}

pub fn pack_f32_slice_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

/// CPU f16 matmul: out[m][n] = sum_k x[m][k] * f16_weights[n][k]
/// f16_weights is stored row-major with shape (rows, cols).
/// Corresponds to: out = x @ w^T where w has shape (rows, cols) and x has shape (m, cols)
/// Parallelized over output rows via rayon — works for both batch>1 and batch=1.
pub fn matmul_f16_cpu(x: &ndarray::Array2<f32>, f16_weights: &[u16], rows: usize, cols: usize) -> ndarray::Array2<f32> {
    let m = x.shape()[0];
    let mut out = ndarray::Array2::zeros((m, rows));
    let out_data = match out.as_slice_mut() {
        Some(s) => s,
        None => return out, // fallback: return unmodified zeros
    };
    use rayon::prelude::*;
    let num_cpus = rayon::current_num_threads();
    let chunk = std::cmp::max(1, (m * rows) / (num_cpus * 4));
    let x_contig = if x.is_standard_layout() { x } else { &x.to_owned() };
    let x_data = match x_contig.as_slice() {
        Some(s) => s,
        None => return out, // fallback: return unmodified zeros
    };
    out_data.par_chunks_mut(chunk).enumerate().for_each(|(start_idx, vals)| {
        for (offset, val) in vals.iter_mut().enumerate() {
            let flat_idx = start_idx * chunk + offset;
            let i = flat_idx / rows;
            let j = flat_idx % rows;
            let x_row_start = i * cols;
            let w_row_start = j * cols;
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += x_data[x_row_start + k] * f16_bits_to_f32(f16_weights[w_row_start + k]);
            }
            *val = sum;
        }
    });
    out
}

pub type TransformerResult<T> = std::result::Result<T, TransformerError>;

#[derive(Debug, thiserror::Error)]
pub enum TransformerError {
    #[error("Implementation error: {0}")]
    Implementation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
