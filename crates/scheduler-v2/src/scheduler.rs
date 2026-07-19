use dashmap::DashSet;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Notify;
use tracing::{error, info};
use uuid::Uuid;

use crate::dag::{Dag, DagTask, TaskId, TaskStatus};
use crate::deadline::DeadlineScheduler;
use crate::gpu_aware::GpuAwareScheduler;
use crate::numa_aware::NumaAwareScheduler;
use crate::priority_queue::TaskPriorityQueue;
use crate::work_stealing::WorkStealingScheduler;

pub struct DagScheduler {
    dags: RwLock<HashMap<Uuid, Arc<Dag>>>,
    priority_queue: Arc<TaskPriorityQueue>,
    deadline_scheduler: Arc<DeadlineScheduler>,
    gpu_aware: Arc<GpuAwareScheduler>,
    numa_aware: Arc<NumaAwareScheduler>,
    work_stealing: Arc<WorkStealingScheduler>,
    completed: DashSet<TaskId>,
    in_progress: DashSet<TaskId>,
    running: AtomicBool,
    notify: Arc<Notify>,
    worker_count: usize,
}

impl DagScheduler {
    pub fn new(worker_count: usize, gpu_count: usize) -> Self {
        let gpu_mem = vec![16 * 1024 * 1024 * 1024; gpu_count.max(1)];
        Self {
            dags: RwLock::new(HashMap::new()),
            priority_queue: Arc::new(TaskPriorityQueue::new()),
            deadline_scheduler: Arc::new(DeadlineScheduler::new()),
            gpu_aware: Arc::new(GpuAwareScheduler::new(gpu_count, gpu_mem)),
            numa_aware: Arc::new(NumaAwareScheduler::new(4, vec![])),
            work_stealing: Arc::new(WorkStealingScheduler::new(worker_count)),
            completed: DashSet::new(),
            in_progress: DashSet::new(),
            running: AtomicBool::new(false),
            notify: Arc::new(Notify::new()),
            worker_count,
        }
    }

    pub fn submit_dag(&self, dag: Dag) -> Arc<Dag> {
        let id = dag.id;
        let dag = Arc::new(dag);
        self.dags.write().insert(id, dag.clone());
        info!("DAG submitted: id={}, name={}, tasks={}", id, dag.name, dag.task_count());

        let sorted = dag.topological_sort();
        for task_id in &sorted {
            let task = dag.tasks.get(task_id).map(|t| t.read().clone());
            if let Some(t) = task {
                if t.dependencies.is_empty() {
                    self.enqueue_task(dag.clone(), &t);
                }
            }
        }

        self.notify.notify_waiters();
        dag
    }

    fn enqueue_task(&self, dag: Arc<Dag>, task: &DagTask) {
        let id = task.id;
        if let Some(dag_task) = dag.tasks.get(&id) {
            dag_task.write().status = TaskStatus::Ready;
        }
        self.priority_queue.push(id, task.priority);
        self.work_stealing.push_global(id);
        if let Some(deadline) = task.deadline_ms {
            let deadline_instant = std::time::Instant::now()
                + std::time::Duration::from_millis(deadline);
            self.deadline_scheduler.add(id, deadline_instant, task.priority);
        }
        self.notify.notify_waiters();
    }

    pub async fn start(self: Arc<Self>) {
        self.running.store(true, Ordering::SeqCst);
        info!("DagScheduler started with {} workers", self.worker_count);

        for i in 0..self.worker_count {
            Self::spawn_worker(self.clone(), i);
        }
    }

    fn spawn_worker(scheduler: Arc<Self>, worker_idx: usize) {
        tokio::spawn(async move {
            loop {
                if let Some(task_id) = scheduler.work_stealing.pop(worker_idx) {
                    let result = scheduler.execute_task(task_id).await;
                    match result {
                        Ok(_) => {
                            scheduler.completed.insert(task_id);
                            scheduler.in_progress.remove(&task_id);
                            scheduler.handle_task_completion(task_id);
                        }
                        Err(e) => {
                            error!("task {} failed: {}", task_id, e);
                            scheduler.in_progress.remove(&task_id);
                        }
                    }
                } else {
                    tokio::select! {
                        _ = scheduler.notify.notified() => {}
                        _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
                    }
                }
            }
        });
    }

    async fn execute_task(&self, task_id: TaskId) -> Result<(), String> {
        self.in_progress.insert(task_id);

        let task = {
            let dags = self.dags.read();
            let mut found = None;
            for dag in dags.values() {
                if let Some(t) = dag.tasks.get(&task_id) {
                    found = Some(t.read().clone());
                    break;
                }
            }
            found.ok_or_else(|| format!("task {} not found", task_id))?
        };

        if let Some(dag) = self.find_dag_by_task(&task_id) {
            if let Some(t) = dag.tasks.get(&task_id) {
                t.write().status = TaskStatus::Running;
                t.write().started_at = Some(chrono::Utc::now());
            }
        }

        if task.requires_gpu {
            self.gpu_aware.assign_gpu(task_id, task.required_memory_mb * 1024 * 1024);
        }
        self.numa_aware.assign_node(task_id, task.required_memory_mb, task.numa_node);

        tokio::time::sleep(std::time::Duration::from_millis(
            task.estimated_duration_ms.min(50),
        ))
        .await;

        if let Some(dag) = self.find_dag_by_task(&task_id) {
            if let Some(t) = dag.tasks.get(&task_id) {
                t.write().status = TaskStatus::Completed;
                t.write().result = Some(task.payload.clone());
                t.write().completed_at = Some(chrono::Utc::now());
            }
        }

        self.gpu_aware.release_gpu(&task_id);
        self.numa_aware.release_node(&task_id, task.required_memory_mb);

        Ok(())
    }

    fn handle_task_completion(&self, task_id: TaskId) {
        let dags = self.dags.read();
        for dag in dags.values() {
            if let Some(task) = dag.tasks.get(&task_id) {
                let task = task.read();
                for dependent_id in &task.dependents {
                    if let Some(dep) = dag.tasks.get(dependent_id) {
                        let dep_task = dep.read();
                        let all_deps_done = dep_task.dependencies.iter().all(|d| self.completed.contains(d));
                        if all_deps_done && dep_task.status == TaskStatus::Pending {
                            drop(dep_task);
                            self.enqueue_task(dag.clone(), &dep.read());
                        }
                    }
                }
            }
        }
        for dag in dags.values() {
            if dag.all_completed() {
                info!("DAG completed: id={}, name={}", dag.id, dag.name);
            }
        }
    }

    fn find_dag_by_task(&self, task_id: &TaskId) -> Option<Arc<Dag>> {
        let dags = self.dags.read();
        for dag in dags.values() {
            if dag.tasks.contains_key(task_id) {
                return Some(dag.clone());
            }
        }
        None
    }

    pub fn dag_status(&self, dag_id: &Uuid) -> Option<DagStatus> {
        let dags = self.dags.read();
        let dag = dags.get(dag_id)?;
        Some(DagStatus {
            id: dag.id,
            name: dag.name.clone(),
            total: dag.task_count(),
            completed: dag.completed_count(),
            pending: dag.task_count() - dag.completed_count() - self.in_progress.len(),
            in_progress: self.in_progress.len(),
            has_failures: dag.has_failures(),
        })
    }

    pub fn all_dag_status(&self) -> Vec<DagStatus> {
        let dags = self.dags.read();
        dags.values()
            .map(|dag| DagStatus {
                id: dag.id,
                name: dag.name.clone(),
                total: dag.task_count(),
                completed: dag.completed_count(),
                pending: dag.task_count() - dag.completed_count() - self.in_progress.len(),
                in_progress: self.in_progress.len(),
                has_failures: dag.has_failures(),
            })
            .collect()
    }

    pub fn gpu_utilization(&self) -> Vec<f64> {
        self.gpu_aware.gpu_utilization()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DagStatus {
    pub id: Uuid,
    pub name: String,
    pub total: usize,
    pub completed: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub has_failures: bool,
}
