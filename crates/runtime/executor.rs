//! Async Task Executor
//! 
//! Shared async execution utilities for AI frameworks

use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use anyhow::Result;
use tracing::{debug, warn};

/// Async task executor with concurrency control
pub struct AsyncTaskExecutor {
    name: String,
    semaphore: Arc<Semaphore>,
    active_tasks: Arc<RwLock<usize>>,
    max_concurrent: usize,
}

impl AsyncTaskExecutor {
    /// Create new async task executor
    pub fn new(name: String, max_concurrent: usize) -> Self {
        Self {
            name,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            active_tasks: Arc::new(RwLock::new(0)),
            max_concurrent,
        }
    }
    
    /// Execute task with concurrency control
    pub async fn execute<F, Fut, T>(&self, task: F) -> Result<JoinHandle<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Acquire semaphore permit
        let permit = self.semaphore.acquire().await?;
        
        // Update active task count
        {
            let mut count = self.active_tasks.write().await;
            *count += 1;
            debug!("Executor {}: Active tasks: {}/{}", self.name, *count, self.max_concurrent);
        }
        
        let active_tasks = self.active_tasks.clone();
        let executor_name = self.name.clone();
        
        // Spawn task
        let handle = tokio::spawn(async move {
            let result = task().await;
            
            // Update active task count
            {
                let mut count = active_tasks.write().await;
                *count = count.saturating_sub(1);
                debug!("Executor {}: Task completed. Active tasks: {}", executor_name, count);
            }
            
            drop(permit); // Release semaphore permit
            result
        });
        
        Ok(handle)
    }
    
    /// Execute task with timeout
    pub async fn execute_with_timeout<F, Fut, T>(
        &self,
        task: F,
        timeout: std::time::Duration,
    ) -> Result<JoinHandle<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let permit = self.semaphore.acquire().await?;
        
        let active_tasks = self.active_tasks.clone();
        let executor_name = self.name.clone();
        
        let handle = tokio::spawn(async move {
            let result = tokio::time::timeout(timeout, task()).await;
            
            let final_result = match result {
                Ok(inner_result) => inner_result,
                Err(_) => {
                    warn!("Executor {}: Task timed out after {:?}", executor_name, timeout);
                    Err(anyhow::anyhow!("Task timed out"))
                }
            };
            
            // Update active task count
            {
                let mut count = active_tasks.write().await;
                *count = count.saturating_sub(1);
            }
            
            drop(permit);
            final_result
        });
        
        Ok(handle)
    }
    
    /// Get current active task count
    pub async fn active_task_count(&self) -> usize {
        *self.active_tasks.read().await
    }
    
    /// Get maximum concurrent tasks
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
    
    /// Get executor name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Check if executor is at capacity
    pub async fn is_at_capacity(&self) -> bool {
        self.active_task_count().await >= self.max_concurrent
    }
    
    /// Wait for all active tasks to complete
    pub async fn wait_for_completion(&self) {
        while self.active_task_count().await > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}

impl Default for AsyncTaskExecutor {
    fn default() -> Self {
        Self::new("default".to_string(), num_cpus::get())
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Priority-based task executor
pub struct PriorityTaskExecutor {
    executors: Vec<AsyncTaskExecutor>,
}

impl PriorityTaskExecutor {
    /// Create new priority task executor
    pub fn new(base_name: String, max_concurrent_per_priority: usize) -> Self {
        let executors = vec![
            AsyncTaskExecutor::new(format!("{}-low", base_name), max_concurrent_per_priority * 2),
            AsyncTaskExecutor::new(format!("{}-normal", base_name), max_concurrent_per_priority),
            AsyncTaskExecutor::new(format!("{}-high", base_name), max_concurrent_per_priority / 2),
            AsyncTaskExecutor::new(format!("{}-critical", base_name), max_concurrent_per_priority / 4),
        ];
        
        Self { executors }
    }
    
    /// Execute task with priority
    pub async fn execute_with_priority<F, Fut, T>(
        &self,
        task: F,
        priority: TaskPriority,
    ) -> Result<JoinHandle<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let executor = &self.executors[priority as usize];
        executor.execute(task).await
    }
    
    /// Get executor for specific priority
    pub fn get_executor(&self, priority: TaskPriority) -> &AsyncTaskExecutor {
        &self.executors[priority as usize]
    }
    
    /// Get statistics for all priorities
    pub async fn get_statistics(&self) -> Vec<(TaskPriority, usize, usize)> {
        let mut stats = Vec::new();
        for (i, executor) in self.executors.iter().enumerate() {
            let priority = match i {
                0 => TaskPriority::Low,
                1 => TaskPriority::Normal,
                2 => TaskPriority::High,
                3 => TaskPriority::Critical,
                _ => TaskPriority::Normal,
            };
            
            stats.push((
                priority,
                executor.active_task_count().await,
                executor.max_concurrent(),
            ));
        }
        stats
    }
}

impl Default for PriorityTaskExecutor {
    fn default() -> Self {
        Self::new("priority".to_string(), num_cpus::get())
    }
}
