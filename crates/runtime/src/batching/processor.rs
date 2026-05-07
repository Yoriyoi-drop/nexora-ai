//! Batch processor implementation

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use tracing::{debug, info, error};

use crate::{Result, InferenceError, InferenceRequest};

use super::config::BatchConfig;
use super::types::{BatchItem, Batch, BatchStats};

/// Processor state
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessorState {
    /// Processor tidak diinisialisasi
    Uninitialized,
    /// Processor sedang diinisialisasi
    Initializing,
    /// Processor siap untuk memproses
    Ready,
    /// Processor sedang memproses
    Processing,
    /// Processor di-shutdown
    Shutdown,
}

/// Batch processor
pub struct BatchProcessor {
    /// Configuration
    config: BatchConfig,
    /// Pending requests queue
    pending_queue: Arc<RwLock<VecDeque<BatchItem>>>,
    /// Active batches
    active_batches: Arc<RwLock<HashMap<Uuid, Batch>>>,
    /// Completed batches
    completed_batches: Arc<RwLock<VecDeque<Batch>>>,
    /// Statistics
    stats: Arc<RwLock<BatchStats>>,
    /// State
    state: Arc<RwLock<ProcessorState>>,
    /// Sender for batch completion notifications
    batch_sender: mpsc::UnboundedSender<Batch>,
    /// Receiver for batch completion notifications
    batch_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<Batch>>>>,
}

impl Clone for BatchProcessor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            pending_queue: Arc::clone(&self.pending_queue),
            active_batches: Arc::clone(&self.active_batches),
            completed_batches: Arc::clone(&self.completed_batches),
            stats: Arc::clone(&self.stats),
            state: Arc::clone(&self.state),
            batch_sender: self.batch_sender.clone(),
            batch_receiver: Arc::clone(&self.batch_receiver),
        }
    }
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(config: BatchConfig) -> Self {
        let (batch_sender, batch_receiver) = mpsc::unbounded_channel();
        
        Self {
            config,
            pending_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_batches: Arc::new(RwLock::new(HashMap::new())),
            completed_batches: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(BatchStats::default())),
            state: Arc::new(RwLock::new(ProcessorState::Uninitialized)),
            batch_sender,
            batch_receiver: Arc::new(RwLock::new(Some(batch_receiver))),
        }
    }
    
    /// Initialize the batch processor
    pub async fn initialize(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if *state != ProcessorState::Uninitialized {
            return Err(InferenceError::InvalidState("Processor already initialized".to_string()).into());
        }
        
        *state = ProcessorState::Initializing;
        
        // Validate configuration
        self.config.validate()
            .map_err(|e| InferenceError::InvalidConfig(e))?;
        
        // Start background processing task
        self.start_processing_task().await?;
        
        *state = ProcessorState::Ready;
        info!("Batch processor initialized successfully");
        
        Ok(())
    }
    
    /// Add request to processing queue
    pub async fn add_request(&self, request: InferenceRequest) -> Result<()> {
        let state = self.state.read().await;
        if *state != ProcessorState::Ready {
            return Err(InferenceError::InvalidState("Processor not ready".to_string()).into());
        }
        drop(state);
        
        let batch_item = BatchItem::new(
            request.request_id,
            request.input_tokens,
            request.target_tokens,
            request.priority,
            request.metadata,
        );
        
        let mut queue = self.pending_queue.write().await;
        queue.push_back(batch_item);
        drop(queue);
        
        debug!("Added request {} to pending queue", request.request_id);
        
        // Try to form a batch immediately if conditions are met
        self.try_form_batch().await?;
        
        Ok(())
    }
    
    /// Get current statistics
    pub async fn get_stats(&self) -> BatchStats {
        self.stats.read().await.clone()
    }
    
    /// Get current state
    pub async fn get_state(&self) -> ProcessorState {
        self.state.read().await.clone()
    }
    
    /// Get pending queue size
    pub async fn get_pending_queue_size(&self) -> usize {
        self.pending_queue.read().await.len()
    }
    
    /// Get active batches count
    pub async fn get_active_batches_count(&self) -> usize {
        self.active_batches.read().await.len()
    }
    
    /// Try to form a batch from pending requests
    async fn try_form_batch(&self) -> Result<()> {
        let mut queue = self.pending_queue.write().await;
        
        if queue.len() < self.config.min_batch_size && !self.config.enable_dynamic_batching {
            return Ok(());
        }
        
        // Collect items for batch
        let mut batch_items = Vec::new();
        let max_batch_size = if self.config.enable_dynamic_batching {
            self.config.max_batch_size
        } else {
            self.config.max_batch_size.min(queue.len())
        };
        
        // Sort by priority if enabled
        if self.config.enable_length_sorting {
            let mut items: Vec<_> = queue.iter().cloned().collect();
            items.sort_by(|a, b| {
                b.sequence_length.cmp(&a.sequence_length)
                    .then(b.priority.cmp(&a.priority))
            });
            
            for item in items.iter().take(max_batch_size) {
                if let Some(pos) = queue.iter().position(|x| x.request_id == item.request_id) {
                    if let Some(removed_item) = queue.remove(pos) {
                        batch_items.push(removed_item);
                    }
                }
                if batch_items.len() >= max_batch_size {
                    break;
                }
            }
        } else {
            while batch_items.len() < max_batch_size && !queue.is_empty() {
                if let Some(item) = queue.pop_front() {
                    batch_items.push(item);
                }
            }
        }
        
        drop(queue);
        
        if batch_items.len() >= self.config.min_batch_size {
            let batch = Batch::new(batch_items, &self.config)
                .map_err(|e| InferenceError::BatchError(e))?;
            
            let batch_id = batch.batch_id;
            
            // Add to active batches
            let mut active_batches = self.active_batches.write().await;
            active_batches.insert(batch_id, batch.clone());
            drop(active_batches);
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.increment_in_progress();
            drop(stats);
            
            // Send batch for processing
            if let Err(e) = self.batch_sender.send(batch.clone()) {
                error!("Failed to send batch for processing: {}", e);
                
                // Remove from active batches on error
                let mut active_batches = self.active_batches.write().await;
                active_batches.remove(&batch_id);
                drop(active_batches);
                
                // Update stats
                let mut stats = self.stats.write().await;
                stats.decrement_in_progress();
                stats.increment_failed();
                drop(stats);
            }
            
            info!("Formed batch {} with {} items", batch_id, batch.items.len());
        }
        
        Ok(())
    }
    
    /// Process completed batch
    async fn process_completed_batch(&self, batch: Batch) -> Result<()> {
        let batch_id = batch.batch_id;
        
        // Remove from active batches
        let mut active_batches = self.active_batches.write().await;
        if let Some(mut batch) = active_batches.remove(&batch_id) {
            batch.end_processing();
            
            // Add to completed batches
            let mut completed_batches = self.completed_batches.write().await;
            completed_batches.push_back(batch.clone());
            
            // Keep only last 100 completed batches
            while completed_batches.len() > 100 {
                completed_batches.pop_front();
            }
            drop(completed_batches);
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.update_with_completed_batch(&batch);
            stats.decrement_in_progress();
            drop(stats);
            
            info!("Completed batch {} processing", batch_id);
        }
        
        Ok(())
    }
    
    /// Start background processing task
    async fn start_processing_task(&self) -> Result<()> {
        let processor = self.clone();
        
        tokio::spawn(async move {
            let mut receiver = {
                let mut receiver_lock = processor.batch_receiver.write().await;
                receiver_lock.take()
            };
            
            if let Some(ref mut rx) = receiver {
                while let Some(batch) = rx.recv().await {
                    if let Err(e) = processor.process_completed_batch(batch).await {
                        error!("Error processing completed batch: {}", e);
                    }
                    
                    // Try to form new batch after processing
                    if let Err(e) = processor.try_form_batch().await {
                        error!("Error forming new batch: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Shutdown the batch processor
    pub async fn shutdown(&self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = ProcessorState::Shutdown;
        drop(state);
        
        info!("Batch processor shutdown completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processor_state_transitions() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config);
        
        // Test initial state
        assert_eq!(processor.state.blocking_read().clone(), ProcessorState::Uninitialized);
    }
    
    #[test]
    fn test_batch_config_validation() {
        let mut config = BatchConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid max_batch_size
        config.max_batch_size = 0;
        assert!(config.validate().is_err());
        
        // Test invalid max_wait_time_ms
        config.max_batch_size = 8;
        config.max_wait_time_ms = 0;
        assert!(config.validate().is_err());
        
        // Test invalid min_batch_size
        config.max_wait_time_ms = 50;
        config.min_batch_size = 8;
        assert!(config.validate().is_err());
    }
}
