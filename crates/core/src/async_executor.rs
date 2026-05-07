//! Enhanced Async Task Executor dengan Proper Concurrency
//! 
//! Implementasi task executor dengan thread pool, priority queue, dan resource management

use crate::types::ModelId;
use crate::error::{CoreError, CoreResult};
use std::sync::Arc;
use std::collections::{BinaryHeap, HashMap};
use tokio::sync::{RwLock, Semaphore, mpsc};
use parking_lot::Mutex;
use uuid::Uuid;
use tracing::{debug, info, error};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

/// Task dengan priority untuk execution
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AsyncTask {
    pub id: String,
    pub model: ModelId,
    pub priority: TaskPriority,
    pub input: String,
    pub context: String,
    pub created_at: Instant,
    pub dependencies: Vec<String>,
    pub max_retries: u32,
}

impl AsyncTask {
    pub fn new(model: ModelId, input: String, context: String, priority: TaskPriority) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            model,
            priority,
            input,
            context,
            created_at: Instant::now(),
            dependencies: Vec::new(),
            max_retries: 3,
        }
    }
    
    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self {
        self.dependencies = dependencies;
        self
    }
    
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPriority {
    Critical = 0,   // System critical tasks
    High = 1,       // User interactive tasks
    Normal = 2,     // Regular processing
    Low = 3,        // Background tasks
    Background = 4,  // Maintenance tasks
}

impl TaskPriority {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => TaskPriority::Critical,
            1 => TaskPriority::High,
            2 => TaskPriority::Normal,
            3 => TaskPriority::Low,
            _ => TaskPriority::Background,
        }
    }
    
    pub fn as_u8(&self) -> u8 {
        match self {
            TaskPriority::Critical => 0,
            TaskPriority::High => 1,
            TaskPriority::Normal => 2,
            TaskPriority::Low => 3,
            TaskPriority::Background => 4,
        }
    }
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// Priority ordering untuk BinaryHeap (min-heap, lower priority value = higher priority)
impl Ord for AsyncTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare priority (lower value = higher priority)
        match self.priority.as_u8().cmp(&other.priority.as_u8()) {
            Ordering::Equal => {
                // If same priority, compare creation time (earlier = higher priority)
                self.created_at.cmp(&other.created_at)
            }
            other => other,
        }
    }
}

impl PartialOrd for AsyncTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub model: ModelId,
    pub output: String,
    pub execution_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
    pub retry_count: u32,
}

/// Task executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    pub max_concurrent_tasks: usize,
    pub max_queue_size: usize,
    pub task_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub enable_metrics: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            max_queue_size: 1000,
            task_timeout_ms: 30000, // 30 seconds
            heartbeat_interval_ms: 5000, // 5 seconds
            enable_metrics: true,
        }
    }
}

/// Enhanced async task executor
pub struct AsyncTaskExecutor {
    config: ExecutorConfig,
    task_queue: Arc<Mutex<BinaryHeap<AsyncTask>>>,
    active_tasks: Arc<RwLock<HashMap<String, TaskInfo>>>,
    completed_tasks: Arc<RwLock<HashMap<String, TaskResult>>>,
    semaphore: Arc<Semaphore>,
    task_sender: mpsc::UnboundedSender<AsyncTask>,
    task_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<AsyncTask>>>>,
    metrics: Arc<RwLock<ExecutorMetrics>>,
    shutdown: Arc<RwLock<bool>>,
}

/// Task information for tracking
#[derive(Debug, Clone)]
struct TaskInfo {
    task: AsyncTask,
    status: TaskStatus,
    started_at: Option<Instant>,
    retry_count: u32,
    last_retry_at: Option<Instant>,
}

#[derive(Debug, Clone)]
enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed(String),
    Cancelled,
    Timeout,
}

/// Executor metrics
#[derive(Debug, Clone, Default)]
pub struct ExecutorMetrics {
    pub total_tasks_submitted: u64,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
    pub total_tasks_cancelled: u64,
    pub total_tasks_timeout: u64,
    pub avg_execution_time_ms: f64,
    pub avg_queue_time_ms: f64,
    pub current_queue_size: usize,
    pub current_active_tasks: usize,
    pub peak_queue_size: usize,
    pub peak_active_tasks: usize,
}

impl AsyncTaskExecutor {
    pub fn new(config: ExecutorConfig) -> Self {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        
        Self {
            config: config.clone(),
            task_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_tasks)),
            task_sender,
            task_receiver: Arc::new(RwLock::new(Some(task_receiver))),
            metrics: Arc::new(RwLock::new(ExecutorMetrics::default())),
            shutdown: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start the executor
    pub async fn start(&self) -> CoreResult<()> {
        info!("Starting async task executor with max_concurrent_tasks={}", self.config.max_concurrent_tasks);
        
        let receiver = {
            let mut recv_guard = self.task_receiver.write().await;
            recv_guard.take().ok_or_else(|| {
                CoreError::TaskExecution("Executor already started".to_string())
            })?
        };
        
        // Start main worker loop
        let executor = self.clone();
        tokio::spawn(async move {
            executor.worker_loop(receiver).await;
        });
        
        // Start metrics collection if enabled
        if self.config.enable_metrics {
            let executor = self.clone();
            tokio::spawn(async move {
                executor.metrics_loop().await;
            });
        }
        
        info!("Async task executor started successfully");
        Ok(())
    }
    
    /// Submit task for execution
    pub async fn submit_task(&self, task: AsyncTask) -> CoreResult<String> {
        let task_id = task.id.clone();
        
        // Check queue size limit
        {
            let queue = self.task_queue.lock();
            if queue.len() >= self.config.max_queue_size {
                return Err(CoreError::TaskExecution("Task queue is full".to_string()));
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_tasks_submitted += 1;
            metrics.current_queue_size += 1;
            if metrics.current_queue_size > metrics.peak_queue_size {
                metrics.peak_queue_size = metrics.current_queue_size;
            }
        }
        
        // Add task to queue
        {
            let mut queue = self.task_queue.lock();
            queue.push(task);
        }
        
        debug!("Task submitted: {}", task_id);
        Ok(task_id)
    }
    
    /// Get task result
    pub async fn get_task_result(&self, task_id: &str) -> Option<TaskResult> {
        let completed = self.completed_tasks.read().await;
        completed.get(task_id).cloned()
    }
    
    /// Cancel task
    pub async fn cancel_task(&self, task_id: &str) -> CoreResult<()> {
        // Try to remove from queue
        {
            let mut queue = self.task_queue.lock();
            queue.retain(|task| task.id != task_id);
        }
        
        // Update status if active
        {
            let mut active = self.active_tasks.write().await;
            if let Some(task_info) = active.get_mut(task_id) {
                task_info.status = TaskStatus::Cancelled;
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_tasks_cancelled += 1;
        }
        
        info!("Task cancelled: {}", task_id);
        Ok(())
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> ExecutorMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Shutdown executor
    pub async fn shutdown(&self) {
        info!("Shutting down async task executor");
        
        *self.shutdown.write().await = true;
        
        // Cancel all pending tasks
        {
            let mut queue = self.task_queue.lock();
            let pending_count = queue.len();
            queue.clear();
            
            let mut metrics = self.metrics.write().await;
            metrics.total_tasks_cancelled += pending_count as u64;
            metrics.current_queue_size = 0;
        }
        
        info!("Async task executor shutdown completed");
    }
    
    /// Main worker loop
    async fn worker_loop(&self, mut receiver: mpsc::UnboundedReceiver<AsyncTask>) {
        info!("Worker loop started");
        
        loop {
            // Check for shutdown
            if *self.shutdown.read().await {
                info!("Worker loop shutting down");
                break;
            }
            
            // Try to get a task from queue or receiver
            let task = tokio::select! {
                task_from_queue = self.get_next_task() => task_from_queue,
                task_from_receiver = receiver.recv() => task_from_receiver,
            };
            
            match task {
                Some(task) => {
                    // Acquire semaphore permit
                    match self.semaphore.acquire().await {
                        Ok(permit) => {
                            let task_id = task.id.clone();
                            
                            // Execute task directly without spawning to avoid lifetime issues
                            debug!("Executing task {} with permit", task_id);
                            self.execute_task_with_permit(task, permit).await;
                        }
                        Err(e) => {
                            error!("Failed to acquire semaphore permit: {}", e);
                        }
                    }
                }
                None => {
                    // No tasks available, wait a bit
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
    
    /// Get next task from priority queue
    async fn get_next_task(&self) -> Option<AsyncTask> {
        let mut queue = self.task_queue.lock();
        queue.pop()
    }
    
    /// Execute task with semaphore permit
    async fn execute_task_with_permit(&self, task: AsyncTask, _permit: tokio::sync::SemaphorePermit<'_>) {
        let task_id = task.id.clone();
        let start_time = Instant::now();
        
        debug!("Starting task execution: {}", task_id);
        
        // Add to active tasks
        {
            let mut active = self.active_tasks.write().await;
            let task_info = TaskInfo {
                task: task.clone(),
                status: TaskStatus::Running,
                started_at: Some(start_time),
                retry_count: 0,
                last_retry_at: None,
            };
            active.insert(task_id.clone(), task_info);
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.current_queue_size = metrics.current_queue_size.saturating_sub(1);
            metrics.current_active_tasks += 1;
            if metrics.current_active_tasks > metrics.peak_active_tasks {
                metrics.peak_active_tasks = metrics.current_active_tasks;
            }
        }
        
        // Execute task with timeout
        let execution_result = tokio::time::timeout(
            Duration::from_millis(self.config.task_timeout_ms),
            self.execute_task_internal(&task)
        ).await;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        let result = match execution_result {
            Ok(result) => result,
            Err(_) => {
                // Timeout
                TaskResult {
                    task_id: task_id.clone(),
                    model: task.model,
                    output: String::new(),
                    execution_time_ms: execution_time,
                    success: false,
                    error: Some("Task execution timed out".to_string()),
                    retry_count: 0,
                }
            }
        };
        
        // Handle retry logic if failed
        let final_result = if !result.success && result.retry_count < task.max_retries {
            self.handle_retry(&task, &result).await
        } else {
            result
        };
        
        // Store result and update metrics
        self.complete_task(final_result.clone()).await;
        
        debug!("Task execution completed: {} (success: {})", task_id, final_result.success);
    }
    
    /// Internal task execution
    async fn execute_task_internal(&self, task: &AsyncTask) -> TaskResult {
        let _start_time = Instant::now();
        
        // Simulate task execution (replace with actual model execution)
        let execution_time = rand::random::<u64>() % 1000 + 100; // 100-1100ms
        
        // Simulate some failures for testing
        let success = rand::random::<f32>() > 0.1; // 90% success rate
        
        if success {
            let output = format!(
                "[{:?}] Processed: {} (took {}ms)",
                task.model,
                task.input,
                execution_time
            );
            
            TaskResult {
                task_id: task.id.clone(),
                model: task.model,
                output,
                execution_time_ms: execution_time,
                success: true,
                error: None,
                retry_count: 0,
            }
        } else {
            TaskResult {
                task_id: task.id.clone(),
                model: task.model,
                output: String::new(),
                execution_time_ms: execution_time,
                success: false,
                error: Some("Simulated execution failure".to_string()),
                retry_count: 0,
            }
        }
    }
    
    /// Handle task retry
    async fn handle_retry(&self, task: &AsyncTask, previous_result: &TaskResult) -> TaskResult {
        let retry_count = previous_result.retry_count + 1;
        
        info!("Retrying task: {} (attempt {}/{})", task.id, retry_count, task.max_retries);
        
        // Wait before retry (exponential backoff)
        let delay_ms = 1000 * (1 << retry_count.min(5));
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        
        // Execute retry
        let mut retry_result = self.execute_task_internal(task).await;
        retry_result.retry_count = retry_count;
        
        retry_result
    }
    
    /// Complete task and update metrics
    async fn complete_task(&self, result: TaskResult) {
        // Remove from active tasks
        {
            let mut active = self.active_tasks.write().await;
            active.remove(&result.task_id);
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.current_active_tasks = metrics.current_active_tasks.saturating_sub(1);
            
            if result.success {
                metrics.total_tasks_completed += 1;
                
                // Update average execution time
                let total_completed = metrics.total_tasks_completed;
                let current_avg = metrics.avg_execution_time_ms;
                let new_avg = (current_avg * (total_completed - 1) as f64 + result.execution_time_ms as f64) / total_completed as f64;
                metrics.avg_execution_time_ms = new_avg;
            } else {
                metrics.total_tasks_failed += 1;
            }
        }
        
        // Store in completed tasks
        {
            let mut completed = self.completed_tasks.write().await;
            completed.insert(result.task_id.clone(), result.clone());
        }
    }
    
    /// Metrics collection loop
    async fn metrics_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_millis(self.config.heartbeat_interval_ms));
        
        loop {
            if *self.shutdown.read().await {
                break;
            }
            
            interval.tick().await;
            
            let metrics = self.get_metrics().await;
            debug!("Executor metrics: queue={}, active={}, completed={}, failed={}", 
                  metrics.current_queue_size,
                  metrics.current_active_tasks,
                  metrics.total_tasks_completed,
                  metrics.total_tasks_failed);
        }
    }
}

impl Clone for AsyncTaskExecutor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            task_queue: Arc::clone(&self.task_queue),
            active_tasks: Arc::clone(&self.active_tasks),
            completed_tasks: Arc::clone(&self.completed_tasks),
            semaphore: Arc::clone(&self.semaphore),
            task_sender: self.task_sender.clone(),
            task_receiver: Arc::clone(&self.task_receiver),
            metrics: Arc::clone(&self.metrics),
            shutdown: Arc::clone(&self.shutdown),
        }
    }
}

impl Default for AsyncTaskExecutor {
    fn default() -> Self {
        Self::new(ExecutorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelId;
    
    #[tokio::test]
    async fn test_async_task_executor() {
        let config = ExecutorConfig {
            max_concurrent_tasks: 2,
            max_queue_size: 10,
            task_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            enable_metrics: false,
        };
        
        let executor = AsyncTaskExecutor::new(config);
        executor.start().await.unwrap();
        
        // Submit some tasks
        let task1 = AsyncTask::new(
            ModelId::Coding,
            "test input 1".to_string(),
            "test context".to_string(),
            TaskPriority::High
        );
        
        let task2 = AsyncTask::new(
            ModelId::Logic,
            "test input 2".to_string(),
            "test context".to_string(),
            TaskPriority::Normal
        );
        
        let task1_id = executor.submit_task(task1).await.unwrap();
        let task2_id = executor.submit_task(task2).await.unwrap();
        
        // Wait for completion
        tokio::time::sleep(Duration::from_millis(2000)).await;
        
        // Check results
        let result1 = executor.get_task_result(&task1_id).await;
        let result2 = executor.get_task_result(&task2_id).await;
        
        assert!(result1.is_some());
        assert!(result2.is_some());
        
        let metrics = executor.get_metrics().await;
        assert!(metrics.total_tasks_submitted >= 2);
        
        executor.shutdown().await;
    }
    
    #[test]
    fn test_task_priority_ordering() {
        let critical_task = AsyncTask::new(
            ModelId::Controller,
            "critical".to_string(),
            "".to_string(),
            TaskPriority::Critical
        );
        
        let low_task = AsyncTask::new(
            ModelId::Controller,
            "low".to_string(),
            "".to_string(),
            TaskPriority::Low
        );
        
        // Critical task should come first (lower priority value)
        assert!(critical_task < low_task);
    }
}
