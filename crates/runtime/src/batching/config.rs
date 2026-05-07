//! Configuration for batching system

/// Configuration untuk batching
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time for batch formation (ms)
    pub max_wait_time_ms: u64,
    /// Enable dynamic batching
    pub enable_dynamic_batching: bool,
    /// Minimum batch size for dynamic batching
    pub min_batch_size: usize,
    /// Batch padding strategy
    pub padding_strategy: PaddingStrategy,
    /// Enable batch sorting by sequence length
    pub enable_length_sorting: bool,
    /// Maximum sequence length difference in batch
    pub max_length_diff: usize,
}

/// Padding strategy untuk batching
#[derive(Debug, Clone, PartialEq)]
pub enum PaddingStrategy {
    /// Pad to longest sequence
    PadToLongest,
    /// Pad to power of 2
    PadToPowerOf2,
    /// Pad to multiple of
    PadToMultiple(usize),
    /// No padding (sequences must be same length)
    NoPadding,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait_time_ms: 50,
            enable_dynamic_batching: true,
            min_batch_size: 2,
            padding_strategy: PaddingStrategy::PadToLongest,
            enable_length_sorting: true,
            max_length_diff: 128,
        }
    }
}

impl BatchConfig {
    /// Create new batch config with custom parameters
    pub fn new(max_batch_size: usize, max_wait_time_ms: u64) -> Self {
        Self {
            max_batch_size,
            max_wait_time_ms,
            ..Default::default()
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".to_string());
        }
        
        if self.max_wait_time_ms == 0 {
            return Err("max_wait_time_ms must be > 0".to_string());
        }
        
        if self.enable_dynamic_batching && self.min_batch_size >= self.max_batch_size {
            return Err("min_batch_size must be < max_batch_size when dynamic batching is enabled".to_string());
        }
        
        if let PaddingStrategy::PadToMultiple(multiple) = &self.padding_strategy {
            if *multiple == 0 {
                return Err("PadToMultiple must be > 0".to_string());
            }
        }
        
        Ok(())
    }
}
