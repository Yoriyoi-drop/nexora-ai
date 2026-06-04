use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use ndarray::{Array1, Array2, ArrayD};
use lru::LruCache;
use tracing::info;

use crate::block::TransformerBlock;
use crate::config::TransformerConfig;
use crate::safetensors::io::SafetensorsHeader;
use crate::{TransformerError, TransformerResult};

/// On-disk block weights loaded on demand.
#[derive(Debug, Clone)]
pub struct BlockOnDisk {
    pub attention_norm_weight: Array1<f32>,
    pub ffn_norm_weight: Array1<f32>,
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
}

/// Lazy loader for transformer weights stored in a safetensors file.
///
/// Parses the header once during construction, then loads individual
/// tensors on demand using file offsets. A small LRU cache keeps
/// recently used blocks in memory.
#[derive(Debug)]
pub struct LazyWeightLoader {
    /// Path to the safetensors file.
    path: String,
    /// Raw file bytes (memory-mapped equivalent).
    raw: Vec<u8>,
    /// Parsed header with tensor names → offsets.
    header: SafetensorsHeader,
    /// Offset where data begins (after 8-byte length + header JSON).
    data_offset: usize,
    /// Cached block weights (LRU, default 2 blocks).
    block_cache: Mutex<LruCache<usize, BlockOnDisk>>,
    /// Cached non-block tensors (embedding, norm, lm_head).
    singleton_cache: Mutex<HashMap<String, ArrayD<f32>>>,
}

impl LazyWeightLoader {
    /// Create a new lazy weight loader from a safetensors file.
    /// Loads and parses the header immediately; tensors are loaded on demand.
    pub fn new(path: impl AsRef<Path>, cache_blocks: usize) -> TransformerResult<Self> {
        let path = path.as_ref().to_string_lossy().to_string();
        let raw = std::fs::read(&path)?;

        if raw.len() < 8 {
            return Err(TransformerError::Implementation(
                "File too small".into(),
            ));
        }

        let header_len = u64::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
        ]) as usize;

        let header_end = 8 + header_len;
        if header_end > raw.len() {
            return Err(TransformerError::Implementation(
                "Header exceeds file size".into(),
            ));
        }

        let header_json = std::str::from_utf8(&raw[8..header_end])
            .map_err(|e| TransformerError::Implementation(format!("UTF-8 error: {}", e)))?;

        let header: SafetensorsHeader = serde_json::from_str(header_json)
            .map_err(|e| TransformerError::Implementation(format!("JSON parse: {}", e)))?;

        let num_tensors = header.tensors.len();
        info!(
            "LazyWeightLoader: {} tensors in {}, cache={} blocks",
            num_tensors, path, cache_blocks
        );

        Ok(Self {
            path,
            raw,
            header,
            data_offset: header_end,
            block_cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(cache_blocks.max(1)).unwrap_or(std::num::NonZeroUsize::new(2).unwrap()),
            )),
            singleton_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Check if a tensor name exists in the file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.header.tensors.contains_key(name)
    }

    /// Load a single tensor by name from the file (no caching for ad-hoc loads).
    pub fn load_tensor(&self, name: &str) -> TransformerResult<ArrayD<f32>> {
        let entry = self.header.tensors.get(name).ok_or_else(|| {
            TransformerError::Implementation(format!("Tensor '{}' not found in {}", name, self.path))
        })?;

        let start = self.data_offset + entry.data_offsets[0];
        let end = self.data_offset + entry.data_offsets[1];
        if end > self.raw.len() {
            return Err(TransformerError::Implementation(format!(
                "Data offset out of range for tensor '{}'", name
            )));
        }
        let bytes = &self.raw[start..end];
        let total: usize = entry.shape.iter().product();

        let floats = match entry.dtype.as_str() {
            "F32" => {
                let mut v = Vec::with_capacity(total);
                for chunk in bytes.chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                v
            }
            "F16" => crate::safetensors::io::f16_bytes_to_f32_slice(bytes),
            other => {
                return Err(TransformerError::Implementation(format!(
                    "Unsupported dtype '{}' for tensor '{}'", other, name
                )));
            }
        };

        ArrayD::from_shape_vec(entry.shape.clone(), floats)
            .map_err(|e| TransformerError::Implementation(format!("Shape error: {}", e)))
    }

    /// Load a 2D tensor (weight matrix).
    fn load_2d(&self, name: &str) -> TransformerResult<Array2<f32>> {
        let arr = self.load_tensor(name)?;
        arr.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TransformerError::Implementation(format!("Expected 2D tensor '{}': {}", name, e)))
    }

    /// Load a 1D tensor (norm weight).
    fn load_1d(&self, name: &str) -> TransformerResult<Array1<f32>> {
        let arr = self.load_tensor(name)?;
        arr.into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| TransformerError::Implementation(format!("Expected 1D tensor '{}': {}", name, e)))
    }

    /// Load or retrieve from singleton cache a non-block tensor.
    fn get_singleton(&self, name: &str) -> TransformerResult<ArrayD<f32>> {
        let mut cache = self.singleton_cache.lock().map_err(|e| {
            TransformerError::Implementation(format!("Singleton cache lock: {}", e))
        })?;
        if let Some(cached) = cache.get(name) {
            return Ok(cached.clone());
        }
        let tensor = self.load_tensor(name)?;
        cache.insert(name.to_string(), tensor.clone());
        Ok(tensor)
    }

    /// Load token_embedding (always cached, never evicted).
    pub fn get_token_embedding(&self) -> TransformerResult<Array2<f32>> {
        let arr = self.get_singleton("token_embedding")?;
        arr.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TransformerError::Implementation(format!("token_embedding not 2D: {}", e)))
    }

    /// Load lm_head (always cached, never evicted).
    pub fn get_lm_head(&self) -> TransformerResult<Array2<f32>> {
        let arr = self.get_singleton("lm_head")?;
        arr.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| TransformerError::Implementation(format!("lm_head not 2D: {}", e)))
    }

    /// Load norm.weight (always cached, never evicted).
    pub fn get_norm_weight(&self) -> TransformerResult<Array1<f32>> {
        let arr = self.get_singleton("norm.weight")?;
        arr.into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| TransformerError::Implementation(format!("norm.weight not 1D: {}", e)))
    }

    /// Load all seven weight tensors for a block from the file.
    fn load_block_from_file(&self, idx: usize) -> TransformerResult<BlockOnDisk> {
        let p = format!("blocks.{}.", idx);
        Ok(BlockOnDisk {
            attention_norm_weight: self.load_1d(&format!("{}attention_norm.weight", p))?,
            ffn_norm_weight: self.load_1d(&format!("{}ffn_norm.weight", p))?,
            wq: self.load_2d(&format!("{}attention.wq", p))?,
            wk: self.load_2d(&format!("{}attention.wk", p))?,
            wv: self.load_2d(&format!("{}attention.wv", p))?,
            wo: self.load_2d(&format!("{}attention.wo", p))?,
            w1: self.load_2d(&format!("{}ffn.w1", p))?,
            w2: self.load_2d(&format!("{}ffn.w2", p))?,
            w3: self.load_2d(&format!("{}ffn.w3", p))?,
        })
    }

    /// Get a block's weights, loading from disk if not cached.
    /// Loaded blocks stay in the LRU cache.
    pub fn get_block(&self, idx: usize) -> TransformerResult<BlockOnDisk> {
        let mut cache = self.block_cache.lock().map_err(|e| {
            TransformerError::Implementation(format!("Block cache lock: {}", e))
        })?;
        if let Some(block) = cache.get(&idx) {
            return Ok(block.clone());
        }
        let block = self.load_block_from_file(idx)?;
        cache.put(idx, block.clone());
        Ok(block)
    }

    /// Evict a specific block from cache.
    pub fn evict_block(&self, idx: usize) {
        if let Ok(mut cache) = self.block_cache.lock() {
            cache.pop(&idx);
        }
    }

    /// Clear all cached blocks (but keep singleton cache).
    pub fn evict_all_blocks(&self) {
        if let Ok(mut cache) = self.block_cache.lock() {
            cache.clear();
        }
    }

    /// Apply a block's weights from the lazy loader into a TransformerBlock.
    /// This is the main entry point for lazy loading during forward pass.
    pub fn apply_block_weights(
        &self,
        idx: usize,
        block: &mut TransformerBlock,
    ) -> TransformerResult<()> {
        let disk = self.get_block(idx)?;
        block.attention_norm.weight = Some(disk.attention_norm_weight);
        block.ffn_norm.weight = Some(disk.ffn_norm_weight);
        block.attention.wq = Some(disk.wq);
        block.attention.wk = Some(disk.wk);
        block.attention.wv = Some(disk.wv);
        block.attention.wo = Some(disk.wo);
        block.ffn.w1 = Some(disk.w1);
        block.ffn.w2 = Some(disk.w2);
        block.ffn.w3 = Some(disk.w3);
        Ok(())
    }
}

/// Load essential weights (embedding, lm_head, norm) from the lazy loader
/// into a CausalLM model. Returns a new CausalLM if successful.
pub fn load_lazy_into_causal_lm(
    config: TransformerConfig,
    loader: &LazyWeightLoader,
) -> TransformerResult<super::model::CausalLM> {
    
    use crate::rope::RoPE;

    let mut model = super::model::CausalLM::new_empty(config.clone());

    model.token_embedding = Some(loader.get_token_embedding()?);
    model.lm_head = Some(loader.get_lm_head()?);
    model.norm.weight = Some(loader.get_norm_weight()?);

    // Pre-compute RoPE
    let rope = RoPE::new(config.head_dim(), config.max_seq_len, config.rope_theta);
    let (cos_full, sin_full) = rope.precompute_freqs_cis();
    let half = config.head_dim() / 2;
    model.precomputed_cos = cos_full
        .into_shape(config.max_seq_len * half)
        .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));
    model.precomputed_sin = sin_full
        .into_shape(config.max_seq_len * half)
        .unwrap_or_else(|_| Array1::zeros(config.max_seq_len * half));

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_loader_nonexistent_file() {
        let result = LazyWeightLoader::new("/tmp/nonexistent_xyz.safetensors", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_loader_bad_file() {
        let path = "/tmp/test_lazy_bad.safetensors";
        std::fs::write(path, &[0u8; 4]).unwrap();
        let result = LazyWeightLoader::new(path, 2);
        assert!(result.is_err());
        let _ = std::fs::remove_file(path);
    }
}
