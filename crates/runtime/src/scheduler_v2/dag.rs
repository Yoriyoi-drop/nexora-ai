use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

pub type TaskId = Uuid;
pub type DagId = Uuid;

/// Status eksekusi task dalam DAG
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TaskStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DagTask {
    pub id: TaskId,
    pub name: String,
    pub dag_id: DagId,
    pub dependencies: Vec<TaskId>,
    pub dependents: Vec<TaskId>,
    pub priority: u32,
    pub estimated_duration_ms: u64,
    pub deadline_ms: Option<u64>,
    pub requires_gpu: bool,
    pub required_memory_mb: u64,
    pub numa_node: Option<u32>,
    pub status: TaskStatus,
    pub payload: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl DagTask {
    pub fn new(dag_id: DagId, name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            dag_id,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            priority: 50,
            estimated_duration_ms: 100,
            deadline_ms: None,
            requires_gpu: false,
            required_memory_mb: 0,
            numa_node: None,
            status: TaskStatus::Pending,
            payload: serde_json::Value::Null,
            result: None,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
        }
    }

    pub fn depends_on(&mut self, task_id: TaskId) -> &mut Self {
        self.dependencies.push(task_id);
        self
    }

    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    pub fn with_deadline(mut self, ms: u64) -> Self {
        self.deadline_ms = Some(ms);
        self
    }

    pub fn with_gpu(mut self) -> Self {
        self.requires_gpu = true;
        self
    }

    pub fn with_numa(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }

    pub fn ready(&self) -> bool {
        self.status == TaskStatus::Ready
    }

    pub fn completed(&self) -> bool {
        self.status == TaskStatus::Completed
    }

    pub fn failed(&self) -> bool {
        matches!(self.status, TaskStatus::Failed(_))
    }
}

/// Directed Acyclic Graph untuk eksekusi task
pub struct Dag {
    pub id: DagId,
    pub name: String,
    pub tasks: DashMap<TaskId, Arc<RwLock<DagTask>>>,
    pub entry_points: RwLock<Vec<TaskId>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Dag {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            tasks: DashMap::new(),
            entry_points: RwLock::new(Vec::new()),
            created_at: chrono::Utc::now(),
        }
    }

    pub fn add_task(&self, task: DagTask) -> Arc<RwLock<DagTask>> {
        let id = task.id;
        let task = Arc::new(RwLock::new(task));
        self.tasks.insert(id, task.clone());
        task
    }

    /// Tambah dependency: `task` depends on `depends_on`
    pub fn add_dependency(&self, task_id: &TaskId, depends_on: &TaskId) {
        if let Some(task) = self.tasks.get(task_id) {
            task.write().dependencies.push(*depends_on);
        }
        if let Some(dep) = self.tasks.get(depends_on) {
            dep.write().dependents.push(*task_id);
        }
    }

    /// Topological sort untuk menentukan urutan eksekusi
    pub fn topological_sort(&self) -> Vec<TaskId> {
        let mut in_degree: HashMap<TaskId, usize> = HashMap::new();
        let mut graph: HashMap<TaskId, Vec<TaskId>> = HashMap::new();

        for entry in self.tasks.iter() {
            let task = entry.value().read();
            in_degree.entry(task.id).or_insert(0);
            for dep in &task.dependencies {
                graph.entry(*dep).or_default().push(task.id);
                *in_degree.entry(task.id).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<TaskId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut sorted = Vec::with_capacity(in_degree.len());
        while let Some(id) = queue.pop_front() {
            sorted.push(id);
            if let Some(neighbors) = graph.get(&id) {
                for n in neighbors {
                    if let Some(deg) = in_degree.get_mut(n) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(*n);
                        }
                    }
                }
            }
        }
        sorted
    }

    /// Task mana yang siap dijalankan (semua dependency sudah completed)
    pub fn get_ready_tasks(&self, completed: &HashSet<TaskId>) -> Vec<TaskId> {
        let mut ready = Vec::new();
        for entry in self.tasks.iter() {
            let task = entry.value().read();
            if task.status != TaskStatus::Pending && task.status != TaskStatus::Ready {
                continue;
            }
            let all_deps_done = task.dependencies.iter().all(|d| completed.contains(d));
            if all_deps_done {
                ready.push(task.id);
            }
        }
        ready.sort_by(|a, b| {
            let ta = self.tasks.get(a).map(|t| t.read().priority).unwrap_or(50);
            let tb = self.tasks.get(b).map(|t| t.read().priority).unwrap_or(50);
            tb.cmp(&ta)
        });
        ready
    }

    pub fn all_completed(&self) -> bool {
        self.tasks.iter().all(|e| {
            let t = e.value().read();
            t.completed() || matches!(t.status, TaskStatus::Failed(_))
        })
    }

    pub fn has_failures(&self) -> bool {
        self.tasks.iter().any(|e| e.value().read().failed())
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn completed_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|e| e.value().read().completed())
            .count()
    }

    pub fn gpu_task_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|e| e.value().read().requires_gpu)
            .count()
    }
}
