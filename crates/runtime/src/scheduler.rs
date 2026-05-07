//! Request Scheduler
//! 
//! Multi-request scheduling untuk inference engine.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use tracing::{debug, info, warn};
use chrono::{DateTime, Utc};

use crate::{Result, InferenceError, InferenceRequest, InferenceResponse};

/// Scheduling strategy
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingStrategy {
    /// First-In-First-Out
    FIFO,
    /// Priority-based scheduling
    Priority,
    /// Shortest Job First
    SJF,
    /// Round Robin
    RoundRobin,
    /// Fair scheduling
    Fair,
}

/// Queued request information
#[derive(Debug, Clone)]
pub struct QueuedRequest {
    /// Request data
    pub request: InferenceRequest,
    /// Response channel sender
    pub response_tx: mpsc::UnboundedSender<InferenceResponse>,
    /// Queue timestamp
    pub queued_at: DateTime<Utc>,
    /// Priority level
    pub priority: u8,
    /// Estimated processing time (ms)
    pub estimated_time_ms: u64,
    /// Request metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Request status
#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    /// Request di queue
    Queued,
    /// Request sedang diproses
    Processing,
    /// Request selesai
    Completed,
    /// Request gagal
    Failed(String),
    /// Request di-cancel
    Cancelled,
    /// Request timeout
    Timeout,
}

/// Request scheduler
pub struct RequestScheduler {
    /// Maximum concurrent requests
    max_concurrent_requests: usize,
    /// Scheduling strategy
    strategy: SchedulingStrategy,
    /// Request queue
    request_queue: Arc<RwLock<VecDeque<QueuedRequest>>>,
    /// Active requests
    active_requests: Arc<RwLock<HashMap<Uuid, RequestInfo>>>,
    /// Request status tracking
    request_status: Arc<RwLock<HashMap<Uuid, RequestStatus>>>,
    /// Response channels for completed requests
    response_channels: Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<InferenceResponse>>>>,
    /// Scheduling statistics
    stats: Arc<RwLock<SchedulerStats>>,
    /// Scheduler state
    state: Arc<RwLock<SchedulerState>>,
}

/// Request information
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Request ID
    pub request_id: Uuid,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// Processing time so far (ms)
    pub processing_time_ms: u64,
    /// Request priority
    pub priority: u8,
    /// Session ID (if any)
    pub session_id: Option<Uuid>,
    /// Request metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total requests received
    pub total_requests: u64,
    /// Requests queued
    pub queued_requests: u64,
    /// Requests processed
    pub processed_requests: u64,
    /// Requests failed
    pub failed_requests: u64,
    /// Requests cancelled
    pub cancelled_requests: u64,
    /// Current queue length
    pub current_queue_length: usize,
    /// Current active requests
    pub current_active_requests: usize,
    /// Average queue time (ms)
    pub avg_queue_time_ms: f64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Throughput (requests/second)
    pub throughput_rps: f64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Scheduler state
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerState {
    /// Scheduler tidak diinisialisasi
    Uninitialized,
    /// Scheduler sedang diinisialisasi
    Initializing,
    /// Scheduler siap
    Ready,
    /// Scheduler sedang shutdown
    ShuttingDown,
    /// Scheduler sudah shutdown
    Shutdown,
}

impl RequestScheduler {
    /// Create new scheduler
    pub fn new(max_concurrent_requests: usize) -> Self {
        Self {
            max_concurrent_requests,
            strategy: SchedulingStrategy::FIFO,
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            request_status: Arc::new(RwLock::new(HashMap::new())),
            response_channels: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
            state: Arc::new(RwLock::new(SchedulerState::Uninitialized)),
        }
    }
    
    /// Set scheduling strategy
    pub fn with_strategy(mut self, strategy: SchedulingStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    
    /// Initialize scheduler
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing request scheduler");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::Initializing;
        }
        
        // Initialize statistics
        {
            let mut stats = self.stats.write().await;
            stats.last_updated = Utc::now();
        }
        
        // Update state to ready
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::Ready;
        }
        
        info!("Request scheduler initialized successfully");
        Ok(())
    }
    
    /// Submit request to scheduler
    pub async fn submit_request(&self, request: InferenceRequest, response_tx: mpsc::UnboundedSender<InferenceResponse>) -> Result<()> {
        debug!("Submitting request to scheduler: {}", request.request_id);
        
        // Check scheduler state
        {
            let state = self.state.read().await;
            match *state {
                SchedulerState::Ready => {},
                _ => {
                    return Err(InferenceError::InternalError("Scheduler not ready".to_string()));
                }
            }
        }
        
        // Create queued request
        let queued_request = QueuedRequest {
            request: request.clone(),
            response_tx,
            queued_at: Utc::now(),
            priority: self.calculate_priority(&request),
            estimated_time_ms: self.estimate_processing_time(&request),
            metadata: request.metadata.clone(),
        };
        
        // Store response channel
        {
            let mut response_channels = self.response_channels.write().await;
            response_channels.insert(request.request_id, queued_request.response_tx.clone());
        }
        
        // Update request status
        {
            let mut request_status = self.request_status.write().await;
            request_status.insert(request.request_id, RequestStatus::Queued);
        }
        
        // Add to queue
        {
            let mut queue = self.request_queue.write().await;
            self.insert_into_queue(&mut queue, queued_request);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.queued_requests += 1;
            stats.current_queue_length = {
                let queue = self.request_queue.read().await;
                queue.len()
            };
            stats.last_updated = Utc::now();
        }
        
        // Try to process queue
        self.process_queue().await?;
        
        debug!("Request {} queued successfully", request.request_id);
        Ok(())
    }
    
    /// Get next request to process
    pub async fn get_next_request(&self) -> Result<Option<QueuedRequest>> {
        debug!("Getting next request from scheduler");
        
        // Check if we can process more requests
        let active_count = {
            let active_requests = self.active_requests.read().await;
            active_requests.len()
        };
        
        if active_count >= self.max_concurrent_requests {
            debug!("Maximum concurrent requests reached ({})", active_count);
            return Ok(None);
        }
        
        // Get next request from queue
        let mut queue = self.request_queue.write().await;
        if let Some(queued_request) = queue.pop_front() {
            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.current_queue_length = queue.len();
            }
            
            Ok(Some(queued_request))
        } else {
            Ok(None)
        }
    }
    
    /// Start processing request
    pub async fn start_request(&self, request_id: Uuid) -> Result<()> {
        debug!("Starting request processing: {}", request_id);
        
        // Update request status
        {
            let mut request_status = self.request_status.write().await;
            if let Some(status) = request_status.get_mut(&request_id) {
                *status = RequestStatus::Processing;
            }
        }
        
        // Add to active requests
        let request_info = RequestInfo {
            request_id,
            started_at: Utc::now(),
            processing_time_ms: 0,
            priority: 0, // Will be set from queued request
            session_id: None,
            metadata: HashMap::new(),
        };
        
        {
            let mut active_requests = self.active_requests.write().await;
            active_requests.insert(request_id, request_info);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.current_active_requests = {
                let active_requests = self.active_requests.read().await;
                active_requests.len()
            };
        }
        
        debug!("Request {} started processing", request_id);
        Ok(())
    }
    
    /// Complete request
    pub async fn complete_request(&self, request_id: Uuid) -> Result<()> {
        debug!("Completing request: {}", request_id);
        
        // Get request info for statistics
        let processing_time = {
            let mut active_requests = self.active_requests.write().await;
            if let Some(info) = active_requests.remove(&request_id) {
                let processing_time = (Utc::now() - info.started_at).num_milliseconds() as u64;
                processing_time
            } else {
                warn!("Request {} not found in active requests", request_id);
                0
            }
        };
        
        // Update request status
        {
            let mut request_status = self.request_status.write().await;
            if let Some(status) = request_status.get_mut(&request_id) {
                *status = RequestStatus::Completed;
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.processed_requests += 1;
            stats.current_active_requests = {
                let active_requests = self.active_requests.read().await;
                active_requests.len()
            };
            
            // Update average processing time
            if stats.processed_requests > 0 {
                stats.avg_processing_time_ms = (stats.avg_processing_time_ms * (stats.processed_requests - 1) as f64 + 
                                             processing_time as f64) / stats.processed_requests as f64;
            }
            
            stats.last_updated = Utc::now();
        }
        
        // Clean up response channel
        {
            let mut response_channels = self.response_channels.write().await;
            response_channels.remove(&request_id);
        }
        
        // Try to process next request in queue
        self.process_queue().await?;
        
        debug!("Request {} completed successfully", request_id);
        Ok(())
    }
    
    /// Cancel request
    pub async fn cancel_request(&self, request_id: Uuid) -> Result<bool> {
        debug!("Cancelling request: {}", request_id);
        
        let mut cancelled = false;
        
        // Check if request is in queue
        {
            let mut queue = self.request_queue.write().await;
            if let Some(pos) = queue.iter().position(|req| req.request.request_id == request_id) {
                queue.remove(pos);
                cancelled = true;
            }
        }
        
        // Check if request is active
        if !cancelled {
            let mut active_requests = self.active_requests.write().await;
            if active_requests.remove(&request_id).is_some() {
                cancelled = true;
            }
        }
        
        if cancelled {
            // Update request status
            {
                let mut request_status = self.request_status.write().await;
                request_status.insert(request_id, RequestStatus::Cancelled);
            }
            
            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.cancelled_requests += 1;
                stats.current_queue_length = {
                    let queue = self.request_queue.read().await;
                    queue.len()
                };
                stats.current_active_requests = {
                    let active_requests = self.active_requests.read().await;
                    active_requests.len()
                };
            }
            
            // Clean up response channel
            {
                let mut response_channels = self.response_channels.write().await;
                response_channels.remove(&request_id);
            }
            
            debug!("Request {} cancelled successfully", request_id);
            Ok(true)
        } else {
            debug!("Request {} not found for cancellation", request_id);
            Ok(false)
        }
    }
    
    /// Send response for request
    pub async fn send_response(&self, request_id: Uuid, response: InferenceResponse) -> Result<()> {
        debug!("Sending response for request: {}", request_id);
        
        let response_channels = self.response_channels.read().await;
        if let Some(response_tx) = response_channels.get(&request_id) {
            if let Err(_) = response_tx.send(response) {
                warn!("Failed to send response for request {}: channel closed", request_id);
                return Err(InferenceError::InternalError("Response channel closed".to_string()));
            }
            Ok(())
        } else {
            Err(InferenceError::InternalError("Response channel not found".to_string()))
        }
    }
    
    /// Get request status
    pub async fn get_request_status(&self, request_id: Uuid) -> Result<Option<RequestStatus>> {
        let request_status = self.request_status.read().await;
        Ok(request_status.get(&request_id).cloned())
    }
    
    /// Get scheduler statistics
    pub async fn get_stats(&self) -> SchedulerStats {
        self.stats.read().await.clone()
    }
    
    /// Shutdown scheduler
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down request scheduler");
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::ShuttingDown;
        }
        
        // Cancel all queued requests
        {
            let mut queue = self.request_queue.write().await;
            let queued_count = queue.len();
            queue.clear();
            
            let mut stats = self.stats.write().await;
            stats.cancelled_requests += queued_count as u64;
        }
        
        // Cancel all active requests
        {
            let active_requests: Vec<Uuid> = {
                let active = self.active_requests.read().await;
                active.keys().cloned().collect()
            };
            
            for request_id in active_requests {
                let _ = self.cancel_request(request_id).await;
            }
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::Shutdown;
        }
        
        info!("Request scheduler shutdown complete");
        Ok(())
    }
    
    /// Process queue - try to start queued requests
    async fn process_queue(&self) -> Result<()> {
        while let Some(queued_request) = self.get_next_request().await? {
            // Start processing the request
            self.start_request(queued_request.request.request_id).await?;
            
            // Send request to processing engine (this would be handled by the engine)
            // For now, we just mark it as started
        }
        
        Ok(())
    }
    
    /// Insert request into queue based on strategy
    fn insert_into_queue(&self, queue: &mut VecDeque<QueuedRequest>, request: QueuedRequest) {
        match self.strategy {
            SchedulingStrategy::FIFO => {
                queue.push_back(request);
            }
            
            SchedulingStrategy::Priority => {
                // Insert based on priority (higher priority first)
                let mut inserted = false;
                for (i, existing) in queue.iter().enumerate() {
                    if request.priority > existing.priority {
                        queue.insert(i, request.clone());
                        inserted = true;
                        break;
                    }
                }
                if !inserted {
                    queue.push_back(request);
                }
            }
            
            SchedulingStrategy::SJF => {
                // Insert based on estimated time (shorter first)
                let mut inserted = false;
                for (i, existing) in queue.iter().enumerate() {
                    if request.estimated_time_ms < existing.estimated_time_ms {
                        queue.insert(i, request.clone());
                        inserted = true;
                        break;
                    }
                }
                if !inserted {
                    queue.push_back(request);
                }
            }
            
            SchedulingStrategy::RoundRobin => {
                // Simple round robin - just add to back
                queue.push_back(request);
            }
            
            SchedulingStrategy::Fair => {
                // Fair scheduling - consider session distribution
                // For now, use FIFO
                queue.push_back(request);
            }
        }
    }
    
    /// Calculate request priority
    fn calculate_priority(&self, request: &InferenceRequest) -> u8 {
        // Simple priority calculation based on request characteristics
        let mut priority = 50; // Base priority
        
        // Higher priority for shorter requests
        if request.max_tokens < 50 {
            priority += 20;
        } else if request.max_tokens > 500 {
            priority -= 20;
        }
        
        // Higher priority for streaming requests
        if request.streaming {
            priority += 10;
        }
        
        // Consider temperature (lower temp = higher priority for deterministic requests)
        if request.temperature < 0.1 {
            priority += 15;
        }
        
        // Clamp to valid range
        priority.clamp(0, 100)
    }
    
    /// Estimate processing time
    fn estimate_processing_time(&self, request: &InferenceRequest) -> u64 {
        // Simple estimation based on token count and temperature
        let base_time_per_token = 50; // ms per token
        let temperature_factor = if request.temperature > 0.0 { 1.5 } else { 1.0 };
        
        (request.max_tokens as u64 * base_time_per_token * temperature_factor as u64).max(100)
    }
}
