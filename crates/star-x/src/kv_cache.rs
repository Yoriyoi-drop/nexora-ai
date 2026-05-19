//! KV Cache untuk STAR-X
//!
//! Implementasi Key-Value cache untuk incremental decoding:
//! - Menghindari O(n²) recomputation
//! - Incremental attention computation
//! - Memory-efficient caching
//! - Streaming inference support

use crate::{DLResult, DeepLearningError};
use ndarray::{Array2, Array1, Axis, s};
use std::collections::VecDeque;

/// Key-Value Cache untuk attention computation
#[derive(Debug, Clone)]
pub struct KVCache {
    // Cached keys and values
    cached_keys: Array2<f32>,
    cached_values: Array2<f32>,
    
    // Cache configuration
    max_cache_size: usize,
    head_dim: usize,
    num_heads: usize,
    
    // Current sequence length
    seq_len: usize,
    
    // Cache statistics
    cache_hits: usize,
    cache_misses: usize,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(max_cache_size: usize, head_dim: usize, num_heads: usize) -> Self {
        Self {
            cached_keys: Array2::zeros((0, head_dim * num_heads)),
            cached_values: Array2::zeros((0, head_dim * num_heads)),
            max_cache_size,
            head_dim,
            num_heads,
            seq_len: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    /// Add new key-value pair to cache
    pub fn append(&mut self, key: Array1<f32>, value: Array1<f32>) -> DLResult<()> {
        if self.seq_len >= self.max_cache_size {
            self.evict_oldest()?;
        }
        
        // Convert to 2D arrays for consistency
        let key_2d = key.into_shape((1, self.head_dim * self.num_heads))?;
        let value_2d = value.into_shape((1, self.head_dim * self.num_heads))?;
        
        // Append to cache
        if self.seq_len == 0 {
            self.cached_keys = key_2d;
            self.cached_values = value_2d;
        } else {
            let mut new_keys = Array2::zeros((self.seq_len + 1, self.head_dim * self.num_heads));
            let mut new_values = Array2::zeros((self.seq_len + 1, self.head_dim * self.num_heads));
            
            new_keys.slice_mut(s![0..self.seq_len, ..]).assign(&self.cached_keys);
            new_keys.slice_mut(s![self.seq_len, ..]).assign(&key_2d.slice(s![0, ..]));
            
            new_values.slice_mut(s![0..self.seq_len, ..]).assign(&self.cached_values);
            new_values.slice_mut(s![self.seq_len, ..]).assign(&value_2d.slice(s![0, ..]));
            
            self.cached_keys = new_keys;
            self.cached_values = new_values;
        }
        
        self.seq_len += 1;
        self.cache_misses += 1; // New computation
        
        Ok(())
    }
    
    /// Get cached keys and values
    pub fn get_cached(&self) -> (&Array2<f32>, &Array2<f32>) {
        (&self.cached_keys, &self.cached_values)
    }
    
    /// Compute attention with cached keys/values
    pub fn compute_attention(&mut self, query: &Array1<f32>) -> DLResult<Array1<f32>> {
        if self.seq_len == 0 {
            return Ok(Array1::zeros(self.head_dim * self.num_heads));
        }
        
        let query_2d = query.view().into_shape((1, self.head_dim * self.num_heads))?;
        
        // Compute attention scores: query @ keys^T
        let attention_scores = query_2d.dot(&self.cached_keys.t());
        
        // Apply softmax
        let softmax_scores = self.softmax(&attention_scores.view().into_shape(self.seq_len)?.to_owned())?;
        
        // Compute output: weights @ values
        let output = softmax_scores.dot(&self.cached_values);
        
        self.cache_hits += 1;
        Ok(output.into_shape(self.head_dim * self.num_heads)?)
    }
    
    /// Reset cache
    pub fn reset(&mut self) {
        self.cached_keys = Array2::zeros((0, self.head_dim * self.num_heads));
        self.cached_values = Array2::zeros((0, self.head_dim * self.num_heads));
        self.seq_len = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            seq_len: self.seq_len,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            hit_rate: if self.cache_hits + self.cache_misses > 0 {
                self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
            } else {
                0.0
            },
            memory_usage: self.seq_len * self.head_dim * self.num_heads * 2 * std::mem::size_of::<f32>(),
        }
    }
    
    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_cache_size
    }
    
    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    
    // Private methods
    
    fn evict_oldest(&mut self) -> DLResult<()> {
        if self.seq_len <= 1 {
            return Ok(());
        }
        
        // Remove oldest entry (first row)
        let new_keys = self.cached_keys.slice_axis(Axis(0), ndarray::Slice::from(1..)).to_owned();
        let new_values = self.cached_values.slice_axis(Axis(0), ndarray::Slice::from(1..)).to_owned();
        
        self.cached_keys = new_keys;
        self.cached_values = new_values;
        self.seq_len -= 1;
        
        Ok(())
    }
    
    fn softmax(&self, scores: &Array1<f32>) -> DLResult<Array1<f32>> {
        // Find max for numerical stability
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp and sum
        let exp_scores: Array1<f32> = scores.mapv(|x| (x - max_score).exp());
        let sum_exp = exp_scores.sum();
        
        if sum_exp == 0.0 {
            return Err(DeepLearningError::Computation {
                reason: "Softmax denominator is zero".to_string(),
            });
        }
        
        Ok(exp_scores / sum_exp)
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub seq_len: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f32,
    pub memory_usage: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KVCache[seq={}, hits={}, misses={}, hit_rate={:.2}%, memory={}KB]",
            self.seq_len,
            self.cache_hits,
            self.cache_misses,
            self.hit_rate * 100.0,
            self.memory_usage / 1024
        )
    }
}

/// Streaming KV Cache for long sequences
#[derive(Debug, Clone)]
pub struct StreamingKVCache {
    // Multiple cache chunks for streaming
    chunks: VecDeque<KVCache>,
    chunk_size: usize,
    max_chunks: usize,
    head_dim: usize,
    num_heads: usize,
    
    // Current position
    current_seq_len: usize,
}

impl StreamingKVCache {
    /// Create new streaming KV cache
    pub fn new(chunk_size: usize, max_chunks: usize, head_dim: usize, num_heads: usize) -> Self {
        Self {
            chunks: VecDeque::new(),
            chunk_size,
            max_chunks,
            head_dim,
            num_heads,
            current_seq_len: 0,
        }
    }
    
    /// Append new key-value pair
    pub fn append(&mut self, key: Array1<f32>, value: Array1<f32>) -> DLResult<()> {
        // Get or create current chunk
        if self.chunks.is_empty() || self.chunks.back().expect("non-empty chunks").is_full() {
            if self.chunks.len() >= self.max_chunks {
                self.chunks.pop_front(); // Remove oldest chunk
            }
            self.chunks.push_back(KVCache::new(self.chunk_size, self.head_dim, self.num_heads));
        }
        
        if let Some(current_chunk) = self.chunks.back_mut() {
            current_chunk.append(key, value)?;
        }
        
        self.current_seq_len += 1;
        Ok(())
    }
    
    /// Compute attention across all chunks
    pub fn compute_attention(&mut self, query: &Array1<f32>) -> DLResult<Array1<f32>> {
        let mut output = Array1::zeros(self.head_dim * self.num_heads);
        let mut total_weight = 0.0;
        
        for chunk in &mut self.chunks {
            if chunk.seq_len() > 0 {
                let chunk_output = chunk.compute_attention(query)?;
                output += &chunk_output;
                total_weight += chunk.seq_len() as f32;
            }
        }
        
        if total_weight > 0.0 {
            output /= total_weight;
        }
        
        Ok(output)
    }
    
    /// Reset all chunks
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.current_seq_len = 0;
    }
    
    /// Get streaming statistics
    pub fn get_streaming_stats(&self) -> StreamingCacheStats {
        let total_hits: usize = self.chunks.iter().map(|c| c.get_stats().cache_hits).sum();
        let total_misses: usize = self.chunks.iter().map(|c| c.get_stats().cache_misses).sum();
        let total_memory: usize = self.chunks.iter().map(|c| c.get_stats().memory_usage).sum();
        
        StreamingCacheStats {
            total_seq_len: self.current_seq_len,
            num_chunks: self.chunks.len(),
            total_hits,
            total_misses,
            hit_rate: if total_hits + total_misses > 0 {
                total_hits as f32 / (total_hits + total_misses) as f32
            } else {
                0.0
            },
            total_memory_usage: total_memory,
        }
    }
}

/// Streaming cache statistics
#[derive(Debug, Clone)]
pub struct StreamingCacheStats {
    pub total_seq_len: usize,
    pub num_chunks: usize,
    pub total_hits: usize,
    pub total_misses: usize,
    pub hit_rate: f32,
    pub total_memory_usage: usize,
}

impl std::fmt::Display for StreamingCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StreamingKVCache[seq={}, chunks={}, hits={}, misses={}, hit_rate={:.2}%, memory={}KB]",
            self.total_seq_len,
            self.num_chunks,
            self.total_hits,
            self.total_misses,
            self.hit_rate * 100.0,
            self.total_memory_usage / 1024
        )
    }
}
