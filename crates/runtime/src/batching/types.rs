//! Types for batching system

use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde_json::Value;

/// Batch item
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Request ID
    pub request_id: Uuid,
    /// Input sequence
    pub input_sequence: Vec<u32>,
    /// Target sequence (optional)
    pub target_sequence: Option<Vec<u32>>,
    /// Sequence length
    pub sequence_length: usize,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Timestamp when request was received
    pub received_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

impl BatchItem {
    /// Create new batch item
    pub fn new(
        request_id: Uuid,
        input_sequence: Vec<u32>,
        target_sequence: Option<Vec<u32>>,
        priority: u8,
        metadata: HashMap<String, Value>,
    ) -> Self {
        let sequence_length = input_sequence.len();
        
        Self {
            request_id,
            input_sequence,
            target_sequence,
            sequence_length,
            priority,
            received_at: Utc::now(),
            metadata,
        }
    }
    
    /// Get the padded length based on strategy
    pub fn get_padded_length(&self, max_length: usize, strategy: &super::config::PaddingStrategy) -> usize {
        use super::config::PaddingStrategy;
        
        match strategy {
            PaddingStrategy::PadToLongest => max_length,
            PaddingStrategy::PadToPowerOf2 => {
                let mut power_of_2 = 1;
                while power_of_2 < max_length {
                    power_of_2 *= 2;
                }
                power_of_2
            }
            PaddingStrategy::PadToMultiple(multiple) => {
                ((max_length + multiple - 1) / multiple) * multiple
            }
            PaddingStrategy::NoPadding => self.sequence_length,
        }
    }
}

/// Processing batch
#[derive(Debug, Clone)]
pub struct Batch {
    /// Batch ID
    pub batch_id: Uuid,
    /// Batch items
    pub items: Vec<BatchItem>,
    /// Maximum sequence length in batch
    pub max_sequence_length: usize,
    /// Total tokens in batch
    pub total_tokens: usize,
    /// Batch creation timestamp
    pub created_at: DateTime<Utc>,
    /// Batch processing start timestamp
    pub processing_started_at: Option<DateTime<Utc>>,
    /// Batch processing end timestamp
    pub processing_ended_at: Option<DateTime<Utc>>,
}

impl Batch {
    /// Create new batch
    pub fn new(items: Vec<BatchItem>, config: &super::config::BatchConfig) -> Result<Self, String> {
        if items.is_empty() {
            return Err("Cannot create empty batch".to_string());
        }
        
        if items.len() > config.max_batch_size {
            return Err(format!("Batch size {} exceeds maximum {}", items.len(), config.max_batch_size));
        }
        
        let max_sequence_length = items.iter()
            .map(|item| item.sequence_length)
            .max()
            .unwrap_or(0);
        
        let total_tokens = items.iter()
            .map(|item| item.sequence_length)
            .sum();
        
        let batch_id = Uuid::new_v4();
        
        Ok(Batch {
            batch_id,
            items,
            max_sequence_length,
            total_tokens,
            created_at: Utc::now(),
            processing_started_at: None,
            processing_ended_at: None,
        })
    }
    
    /// Start processing
    pub fn start_processing(&mut self) {
        self.processing_started_at = Some(Utc::now());
    }
    
    /// End processing
    pub fn end_processing(&mut self) {
        self.processing_ended_at = Some(Utc::now());
    }
    
    /// Get processing duration
    pub fn get_processing_duration(&self) -> Option<chrono::Duration> {
        if let (Some(start), Some(end)) = (self.processing_started_at, self.processing_ended_at) {
            Some(end - start)
        } else {
            None
        }
    }
    
    /// Check if batch is complete
    pub fn is_complete(&self) -> bool {
        self.processing_ended_at.is_some()
    }
    
    /// Get average sequence length
    pub fn get_avg_sequence_length(&self) -> f64 {
        if self.items.is_empty() {
            0.0
        } else {
            self.total_tokens as f64 / self.items.len() as f64
        }
    }
}

/// Batch statistics
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches created
    pub total_batches: u64,
    /// Total requests processed
    pub total_requests: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Total tokens processed
    pub total_tokens_processed: u64,
    /// Batches currently in progress
    pub batches_in_progress: u64,
    /// Failed batches
    pub failed_batches: u64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl BatchStats {
    /// Update statistics with completed batch
    pub fn update_with_completed_batch(&mut self, batch: &Batch) {
        self.total_batches += 1;
        self.total_requests += batch.items.len() as u64;
        self.total_tokens_processed += batch.total_tokens as u64;
        
        // Update average batch size
        self.avg_batch_size = self.total_requests as f64 / self.total_batches as f64;
        
        // Update average processing time
        if let Some(duration) = batch.get_processing_duration() {
            let duration_ms = duration.num_milliseconds() as f64;
            self.avg_processing_time_ms = 
                (self.avg_processing_time_ms * (self.total_batches - 1) as f64 + duration_ms) / self.total_batches as f64;
        }
        
        self.last_updated = Utc::now();
    }
    
    /// Increment in-progress batches
    pub fn increment_in_progress(&mut self) {
        self.batches_in_progress += 1;
        self.last_updated = Utc::now();
    }
    
    /// Decrement in-progress batches
    pub fn decrement_in_progress(&mut self) {
        if self.batches_in_progress > 0 {
            self.batches_in_progress -= 1;
        }
        self.last_updated = Utc::now();
    }
    
    /// Increment failed batches
    pub fn increment_failed(&mut self) {
        self.failed_batches += 1;
        self.last_updated = Utc::now();
    }
    
    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
