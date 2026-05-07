//! Task Scheduler
//! 
//! Shared task scheduling utilities for AI frameworks

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tokio::time::{Duration, Instant};
use anyhow::Result;
use tracing::{debug, warn};
use crate::runtime::executor::{AsyncTaskExecutor, TaskPriority};

/// Scheduled task with metadata
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub id: String,
    pub priority: TaskPriority,
    pub scheduled_time: Instant,
    pub max_retries: usize,
    pub retry_count: usize,
    pub timeout: Option<Duration>,
}

impl ScheduledTask {
    /// Create new scheduled task
    pub fn new(id: String, priority: TaskPriority) -> Self {
        Self {
            id,
            priority,
            scheduled_time: Instant::now(),
            max_retries: 3,
            retry_count: 0,
            timeout: None,
        }
    }
    
    /// Set scheduled time
    pub fn with_scheduled_time(mut self, time: Instant) -> Self {
        self.scheduled_time = time;
        self
    }
    
    /// Set max retries
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
    
    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    /// Check if task should be executed now
    pub fn should_execute(&self) -> bool {
        self.scheduled_time <= Instant::now()
    }
    
    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
    
    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for max-heap (higher priority first)
        other.priority.cmp(&self.priority)
            .then_with(|| other.scheduled_time.cmp(&self.scheduled_time))
    }
}

/// Task scheduler with priority queue
pub struct TaskScheduler {
    name: String,
    pending_tasks: Arc<Mutex<BinaryHeap<ScheduledTask>>>,
    running_tasks: Arc<RwLock<HashMap<String, ScheduledTask>>>,
    completed_tasks: Arc<RwLock<VecDeque<String>>>,
    executor: Arc<AsyncTaskExecutor>,
    max_completed_tasks: usize,
}

impl TaskScheduler {
    /// Create new task scheduler
    pub fn new(name: String, max_concurrent: usize) -> Self {
        Self {
            name,
            pending_tasks: Arc::new(Mutex::new(BinaryHeap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(VecDeque::new())),
            executor: Arc::new(AsyncTaskExecutor::new(format!("{}-scheduler", name), max_concurrent)),
            max_completed_tasks: 1000,
        }
    }
    
    /// Schedule task for execution
    pub async fn schedule(&self, task: ScheduledTask) -> Result<()> {
        {
            let mut pending = self.pending_tasks.lock().await;
            pending.push(task.clone());
        }
        
        debug!("Scheduler {}: Scheduled task {} with priority {:?}", 
               self.name, task.id, task.priority);
        
        // Try to execute pending tasks
        self.try_execute_pending().await;
        
        Ok(())
    }
    
    /// Schedule task with delay
    pub async fn schedule_with_delay(&self, task: ScheduledTask, delay: Duration) -> Result<()> {
        let delayed_task = task.with_scheduled_time(Instant::now() + delay);
        self.schedule(delayed_task).await
    }
    
    /// Cancel scheduled task
    pub async fn cancel(&self, task_id: &str) -> Result<bool> {
        // Remove from pending tasks
        {
            let mut pending = self.pending_tasks.lock().await;
            let original_len = pending.len();
            pending.retain(|task| task.id != task_id);
            
            if pending.len() < original_len {
                debug!("Scheduler {}: Cancelled pending task {}", self.name, task_id);
                return Ok(true);
            }
        }
        
        // Remove from running tasks
        {
            let mut running = self.running_tasks.write().await;
            if running.remove(task_id).is_some() {
                debug!("Scheduler {}: Cancelled running task {}", self.name, task_id);
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        // Check pending tasks
        {
            let pending = self.pending_tasks.lock().await;
            if let Some(task) = pending.iter().find(|t| t.id == task_id) {
                return Some(TaskStatus::Pending(task.clone()));
            }
        }
        
        // Check running tasks
        {
            let running = self.running_tasks.read().await;
            if let Some(task) = running.get(task_id) {
                return Some(TaskStatus::Running(task.clone()));
            }
        }
        
        // Check completed tasks
        {
            let completed = self.completed_tasks.read().await;
            if completed.contains(&task_id.to_string()) {
                return Some(TaskStatus::Completed);
            }
        }
        
        None
    }
    
    /// Get scheduler statistics
    pub async fn get_statistics(&self) -> SchedulerStatistics {
        let pending_count = {
            let pending = self.pending_tasks.lock().await;
            pending.len()
        };
        
        let running_count = {
            let running = self.running_tasks.read().await;
            running.len()
        };
        
        let completed_count = {
            let completed = self.completed_tasks.read().await;
            completed.len()
        };
        
        SchedulerStatistics {
            name: self.name.clone(),
            pending_tasks: pending_count,
            running_tasks: running_count,
            completed_tasks: completed_count,
            max_concurrent: self.executor.max_concurrent(),
            active_executors: self.executor.active_task_count().await,
        }
    }
    
    /// Try to execute pending tasks
    async fn try_execute_pending(&self) {
        // Get tasks that should be executed now
        let tasks_to_execute = {
            let mut pending = self.pending_tasks.lock().await;
            let mut ready_tasks = Vec::new();
            
            while let Some(task) = pending.peek() {
                if task.should_execute() {
                    if let Some(task) = pending.pop() {
                        ready_tasks.push(task);
                    }
                } else {
                    break; // Tasks are ordered by time, so we can stop
                }
            }
            
            ready_tasks
        };
        
        // Execute ready tasks
        for task in tasks_to_execute {
            self.execute_task(task).await;
        }
    }
    
    /// Execute a single task
    async fn execute_task(&self, task: ScheduledTask) {
        let task_id = task.id.clone();
        let running_tasks = self.running_tasks.clone();
        let completed_tasks = self.completed_tasks.clone();
        let scheduler_name = self.name.clone();
        
        // Add to running tasks
        {
            let mut running = running_tasks.write().await;
            running.insert(task_id.clone(), task.clone());
        }
        
        debug!("Scheduler {}: Executing task {}", scheduler_name, task_id);
        
        // Execute task (placeholder - actual task execution would be implemented by caller)
        let executor = self.executor.clone();
        let _ = executor.execute(move || async move {
            // Simulate task execution
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            // Remove from running tasks and add to completed
            {
                let mut running = running_tasks.write().await;
                running.remove(&task_id);
            }
            
            {
                let mut completed = completed_tasks.write().await;
                completed.push_back(task_id.clone());
                
                // Limit completed tasks history
                if completed.len() > 1000 {
                    completed.pop_front();
                }
            }
            
            debug!("Scheduler {}: Completed task {}", scheduler_name, task_id);
            Ok::<(), anyhow::Error>(())
        }).await;
    }
    
    /// Start background scheduler loop
    pub async fn start_background_loop(&self) -> Result<tokio::task::JoinHandle<()>> {
        let pending_tasks = self.pending_tasks.clone();
        let scheduler_name = self.name.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                // Check for tasks ready to execute
                let tasks_to_execute = {
                    let mut pending = pending_tasks.lock().await;
                    let mut ready_tasks = Vec::new();
                    
                    while let Some(task) = pending.peek() {
                        if task.should_execute() {
                            if let Some(task) = pending.pop() {
                                ready_tasks.push(task);
                            }
                        } else {
                            break;
                        }
                    }
                    
                    ready_tasks
                };
                
                if !tasks_to_execute.is_empty() {
                    debug!("Scheduler {}: Found {} tasks ready for execution", 
                           scheduler_name, tasks_to_execute.len());
                }
                
                // Process ready tasks (implementation would depend on specific use case)
                for task in tasks_to_execute {
                    debug!("Scheduler {}: Task {} is ready", scheduler_name, task.id);
                }
            }
        });
        
        Ok(handle)
    }
}

/// Task status
#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending(ScheduledTask),
    Running(ScheduledTask),
    Completed,
    Failed,
    Cancelled,
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    pub name: String,
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub completed_tasks: usize,
    pub max_concurrent: usize,
    pub active_executors: usize,
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new("default".to_string(), num_cpus::get())
    }
}
