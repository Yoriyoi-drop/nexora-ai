//! Task distribution dan execution untuk Nexora Core

use crate::types::{ModelId, TaskExecution};
use crate::error::CoreResult;
use std::collections::HashMap;
use uuid::Uuid;
use tracing::{debug, info, warn};

/// Task manager untuk mengelola distribusi dan eksekusi task
pub struct TaskManager {
    active_tasks: HashMap<String, TaskExecution>,
    max_concurrent_tasks: usize,
}

impl TaskManager {
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            active_tasks: HashMap::new(),
            max_concurrent_tasks,
        }
    }
    
    /// Create dan assign task ke model
    pub async fn create_task(&mut self, model: ModelId, description: String, input: String) -> CoreResult<String> {
        if self.active_tasks.len() >= self.max_concurrent_tasks {
            return Err(crate::error::CoreError::TaskLimitExceeded {
                current: self.active_tasks.len(),
                max: self.max_concurrent_tasks,
            });
        }
        
        let task_id = Uuid::new_v4().to_string();
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        let task = TaskExecution {
            task_id: task_id.clone(),
            assigned_model: model,
            task_description: description,
            task_input: input,
            is_completed: false,
            was_successful: false,
            task_output: None,
            start_time,
            end_time: 0,
            retry_count: 0,
        };
        
        self.active_tasks.insert(task_id.clone(), task);
        
        debug!("Task created: id={}, model={:?}", task_id, model);
        info!("Active tasks count: {}", self.active_tasks.len());
        
        Ok(task_id)
    }
    
    /// Execute task pada model (simplified)
    pub async fn execute_task(&mut self, task_id: &str) -> CoreResult<String> {
        let task = self.active_tasks.get_mut(task_id)
            .ok_or_else(|| crate::error::CoreError::TaskExecution(format!("Task not found: {}", task_id)))?;
        
        debug!("Executing task: id={}, model={:?}", task_id, task.assigned_model);
        
        // Simulate task execution
        let output = format!(
            "Task executed by {:?}: {}",
            task.assigned_model,
            task.task_description
        );
        
        // Update task status
        task.task_output = Some(output.clone());
        task.is_completed = true;
        task.was_successful = true;
        task.end_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        info!("Task completed successfully: id={}", task_id);
        Ok(output)
    }
    
    /// Get task by ID
    pub fn get_task(&self, task_id: &str) -> Option<&TaskExecution> {
        self.active_tasks.get(task_id)
    }
    
    /// Get all active tasks
    pub fn get_active_tasks(&self) -> Vec<&TaskExecution> {
        self.active_tasks.values().collect()
    }
    
    /// Complete task dan remove dari active tasks
    pub async fn complete_task(&mut self, task_id: &str, success: bool) -> CoreResult<()> {
        if let Some(task) = self.active_tasks.get_mut(task_id) {
            task.is_completed = true;
            task.was_successful = success;
            task.end_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            
            if success {
                info!("Task completed successfully: id={}", task_id);
            } else {
                warn!("Task completed with failure: id={}", task_id);
            }
        }
        
        Ok(())
    }
    
    /// Remove completed tasks
    pub fn cleanup_completed(&mut self) {
        let completed_tasks: Vec<String> = self.active_tasks
            .iter()
            .filter(|(_, task)| task.is_completed)
            .map(|(id, _)| id.clone())
            .collect();
        
        for task_id in completed_tasks {
            self.active_tasks.remove(&task_id);
        }
        
        if !self.active_tasks.is_empty() {
            debug!("Cleaned up completed tasks, remaining: {}", self.active_tasks.len());
        }
    }
    
    /// Get active task count
    pub fn active_task_count(&self) -> usize {
        self.active_tasks.len()
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new(10)
    }
}

// Extend TaskExecution dari types.rs
impl TaskExecution {
    pub fn duration_ms(&self) -> u64 {
        if self.end_time > 0 && self.start_time > 0 {
            self.end_time - self.start_time
        } else {
            0
        }
    }
    
    pub fn is_running(&self) -> bool {
        !self.is_completed && self.start_time > 0
    }
}
