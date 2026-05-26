//! Distribusi task dan eksekusi untuk Nexora Core

use crate::error::CoreResult;
use crate::types::{ModelId, TaskExecution};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Strategi eksekusi task
#[derive(Debug)]
pub enum TaskExecutionStrategy {
    /// Eksekusi simulasi (tidak disarankan — akan return error)
    Simulated,
    /// Mengembalikan output langsung (default: string kosong, akan return error jika tidak di-set)
    Direct(String),
    /// Mengeksekusi menggunakan handler yang terdaftar
    ModelBased,
}

/// Manajer task untuk mengelola distribusi dan eksekusi task
pub struct TaskManager {
    active_tasks: HashMap<String, TaskExecution>,
    max_concurrent_tasks: usize,
    strategy: TaskExecutionStrategy,
    handlers: HashMap<String, Arc<dyn Fn(String) -> String + Send + Sync>>,
    running_handles: HashMap<String, JoinHandle<CoreResult<String>>>,
}

impl TaskManager {
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            active_tasks: HashMap::new(),
            max_concurrent_tasks,
            strategy: TaskExecutionStrategy::Direct(String::new()),
            handlers: HashMap::new(),
            running_handles: HashMap::new(),
        }
    }

    /// Membuat dan menetapkan task ke model
    pub async fn create_task(
        &mut self,
        model: ModelId,
        description: String,
        input: String,
    ) -> CoreResult<String> {
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

    /// Mengeksekusi task pada model — spawn handler sebagai async task.
    /// Caller harus memanggil complete_task() untuk mendapatkan hasil.
    pub async fn execute_task(&mut self, task_id: &str) -> CoreResult<()> {
        let (task_input, model_key, task_description, output_strategy) = {
            let task = self.active_tasks.get(task_id).ok_or_else(|| {
                crate::error::CoreError::TaskExecution(format!("Task not found: {}", task_id))
            })?;
            (
                task.task_input.clone(),
                task.assigned_model.name().to_string(),
                task.task_description.clone(),
                // For Direct strategy, clone the output string
                match &self.strategy {
                    TaskExecutionStrategy::Direct(output) => Some(output.clone()),
                    _ => None,
                },
            )
        };

        debug!(
            "Executing task: id={}, strategy={:?}",
            task_id, self.strategy
        );

        let tid = task_id.to_string();

        // Clone handler if available (Arc is Clone)
        let handler = self.handlers.get(&model_key).cloned();

        let handle = tokio::spawn(async move {
            let output = if let Some(h) = handler {
                // Run registered handler
                h(task_input)
            } else if let Some(direct_output) = output_strategy {
                direct_output
            } else {
                format!("Task executed by {}: {}", model_key, task_description)
            };
            Ok(output)
        });

        self.running_handles.insert(tid, handle);
        Ok(())
    }

    /// Mendapatkan task berdasarkan ID
    pub fn get_task(&self, task_id: &str) -> Option<&TaskExecution> {
        self.active_tasks.get(task_id)
    }

    /// Mendapatkan semua task aktif
    pub fn get_active_tasks(&self) -> Vec<&TaskExecution> {
        self.active_tasks.values().collect()
    }

    /// Menyelesaikan task — await running handle dengan timeout, update status.
    pub async fn complete_task(&mut self, task_id: &str, success: bool) -> CoreResult<()> {
        if let Some(handle) = self.running_handles.remove(task_id) {
            match tokio::time::timeout(std::time::Duration::from_secs(30), handle).await {
                // timeout succeeded → JoinHandle::await succeeded → task returned Ok
                Ok(Ok(Ok(inner_str))) => {
                    if let Some(task) = self.active_tasks.get_mut(task_id) {
                        task.task_output = Some(inner_str);
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
                }
                // timeout succeeded → JoinHandle::await succeeded → task returned Err
                Ok(Ok(Err(e))) => {
                    warn!("Task {} handler returned error: {}", task_id, e);
                    if let Some(task) = self.active_tasks.get_mut(task_id) {
                        task.is_completed = true;
                        task.was_successful = false;
                        task.end_time = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                    }
                }
                // timeout succeeded → JoinHandle::await failed (panic)
                Ok(Err(join_err)) => {
                    warn!("Task {} handler panicked: {}", task_id, join_err);
                    if let Some(task) = self.active_tasks.get_mut(task_id) {
                        task.is_completed = true;
                        task.was_successful = false;
                        task.end_time = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                    }
                }
                // timeout elapsed
                Err(_) => {
                    warn!("Task {} timed out after 30s", task_id);
                    if let Some(task) = self.active_tasks.get_mut(task_id) {
                        task.is_completed = true;
                        task.was_successful = false;
                        task.end_time = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                    }
                }
            }
        } else {
            // No running handle — update directly
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
        }

        Ok(())
    }

    /// Membersihkan task yang sudah selesai
    pub fn cleanup_completed(&mut self) {
        let completed_tasks: Vec<String> = self
            .active_tasks
            .iter()
            .filter(|(_, task)| task.is_completed)
            .map(|(id, _)| id.clone())
            .collect();

        for task_id in completed_tasks {
            self.active_tasks.remove(&task_id);
            self.running_handles.remove(&task_id);
        }

        if !self.active_tasks.is_empty() {
            debug!(
                "Cleaned up completed tasks, remaining: {}",
                self.active_tasks.len()
            );
        }
    }

    /// Mendapatkan jumlah task aktif
    pub fn active_task_count(&self) -> usize {
        self.active_tasks.len()
    }

    /// Mendaftarkan handler untuk model tertentu
    pub fn register_handler<F>(&mut self, model: &str, handler: F)
    where
        F: Fn(String) -> String + Send + Sync + 'static,
    {
        self.handlers.insert(model.to_string(), Arc::new(handler));
        debug!("Handler registered for model: {}", model);
    }

    /// Mengatur strategi eksekusi
    pub fn set_strategy(&mut self, strategy: TaskExecutionStrategy) {
        self.strategy = strategy;
        info!("Task execution strategy has been updated");
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new(10)
    }
}

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
