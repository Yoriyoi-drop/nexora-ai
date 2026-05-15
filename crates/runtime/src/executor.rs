//! Task Executor Module
//! 
//! Provides async task execution capabilities for the runtime

use crate::Result;

/// Task executor for running async operations
pub struct TaskExecutor {
    task_count: std::sync::atomic::AtomicU64,
}

impl TaskExecutor {
    /// Create new task executor
    pub fn new() -> Self {
        Self {
            task_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get total tasks executed
    pub fn task_count(&self) -> u64 {
        self.task_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Execute a task with retry support
    pub async fn execute_with_retry<T, F>(&self, task: T, retries: u32) -> Result<()>
    where
        T: Fn() -> F + Send + Sync,
        F: std::future::Future<Output = Result<()>> + Send,
    {
        for attempt in 0..=retries {
            match task().await {
                Ok(()) => {
                    self.task_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
                Err(e) if attempt < retries => {
                    tracing::warn!("Task failed (attempt {}/{}): {}", attempt + 1, retries + 1, e);
                    tokio::time::sleep(std::time::Duration::from_millis(100 * 2u64.pow(attempt))).await;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}



impl Default for TaskExecutor {
    fn default() -> Self {
        Self::new()
    }
}
