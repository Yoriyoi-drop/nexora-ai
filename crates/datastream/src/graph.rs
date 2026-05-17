use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info};

use crate::filter::Filter;
use crate::types::{DataSample, FilterResult, FilterAction, PipelineMetrics, FilterMetric};

type FilterArc = Arc<dyn Filter>;

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub filter: FilterArc,
    pub depends_on: Vec<String>,
    pub children: Vec<String>,
    pub concurrent: bool,
    pub priority: u8,
}

#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    pub nodes: HashMap<String, GraphNode>,
    pub entry_points: Vec<String>,
    pub exit_points: Vec<String>,
    pub metrics: Arc<RwLock<PipelineMetrics>>,
    cached_order: Arc<RwLock<Option<Vec<String>>>>,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
            metrics: Arc::new(RwLock::new(PipelineMetrics::default())),
            cached_order: Arc::new(RwLock::new(None)),
        }
    }

    pub fn add_node(&mut self, id: &str, filter: FilterArc, depends_on: Vec<String>, concurrent: bool, priority: u8) {
        let children = Vec::new();
        let node = GraphNode {
            id: id.to_string(),
            filter,
            depends_on,
            children,
            concurrent,
            priority,
        };
        self.nodes.insert(id.to_string(), node);
    }

    pub fn finalize(&mut self) {
        let mut is_entry = HashSet::with_capacity(self.nodes.len());
        let mut has_deps = HashSet::with_capacity(self.nodes.len());
        let mut child_updates: Vec<(String, String)> = Vec::with_capacity(self.nodes.len());

        for (id, node) in &self.nodes {
            if node.depends_on.is_empty() {
                is_entry.insert(id.clone());
            }
            for dep in &node.depends_on {
                has_deps.insert(dep.clone());
                child_updates.push((dep.clone(), id.clone()));
            }
        }

        for (parent_id, child_id) in child_updates {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.push(child_id);
            }
        }

        self.entry_points = self.nodes.keys()
            .filter(|k| !has_deps.contains(k.as_str()) || is_entry.contains(k.as_str()))
            .cloned()
            .collect();

        self.exit_points = self.nodes.keys()
            .filter(|k| {
                self.nodes.get(k.as_str())
                    .map(|n| n.children.is_empty())
                    .unwrap_or(true)
            })
            .cloned()
            .collect();

        let order = self.topological_order_inner();
        *self.cached_order.write() = Some(order);

        info!(
            "Execution graph finalized: {} nodes, {} entry, {} exit",
            self.nodes.len(),
            self.entry_points.len(),
            self.exit_points.len()
        );
    }

    fn topological_order_inner(&self) -> Vec<String> {
        let mut in_degree: HashMap<String, usize> = self.nodes.keys()
            .map(|k| (k.clone(), self.nodes[k].depends_on.len()))
            .collect();

        let mut queue: VecDeque<String> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut result = Vec::with_capacity(self.nodes.len());
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id.clone());
            if let Some(node) = self.nodes.get(&node_id) {
                for child in &node.children {
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(child.clone());
                        }
                    }
                }
            }
        }
        result
    }

    pub fn topological_order(&self) -> Vec<String> {
        self.cached_order.read().clone().unwrap_or_else(|| self.topological_order_inner())
    }

    pub async fn execute(
        &self,
        sample: DataSample,
        cancel: tokio::sync::watch::Receiver<bool>,
    ) -> ExecutionResult {
        let order = self.topological_order();
        let mut results: Vec<FilterResult> = Vec::with_capacity(order.len());
        let start = std::time::Instant::now();

        for node_id in &order {
            if *cancel.borrow() {
                return ExecutionResult::Cancelled;
            }

            let node = match self.nodes.get(node_id) {
                Some(n) => n,
                None => continue,
            };

            let filter_result = node.filter.filter(&sample).await;
            debug!("Filter '{}': passed={}", node_id, filter_result.passed);

            {
                let mut metrics = self.metrics.write();
                metrics.samples_in += 1;
                let entry = metrics.filter_breakdown.entry(node_id.clone())
                    .or_insert_with(|| FilterMetric {
                        processed: 0, passed: 0, rejected: 0, avg_latency_us: 0.0,
                    });
                entry.processed += 1;
                if filter_result.passed {
                    entry.passed += 1;
                } else {
                    entry.rejected += 1;
                }
            }

            let passed = filter_result.passed;
            let reason = filter_result.reason.clone();
            results.push(filter_result);

            if !passed {
                let action = node.filter.action();
                match action {
                    FilterAction::Reject => {
                        return ExecutionResult::Rejected {
                            sample,
                            results,
                            filter_name: node_id.clone(),
                            reason,
                        };
                    }
                    FilterAction::Reroute(_) => {
                        return ExecutionResult::Rerouted {
                            sample,
                            results,
                            filter_name: node_id.clone(),
                            reason,
                        };
                    }
                    FilterAction::Flag | FilterAction::Accept => {}
                }
            }
        }

        let elapsed = start.elapsed();
        {
            let mut metrics = self.metrics.write();
            metrics.samples_accepted += 1;
            metrics.total_latency_ms += elapsed.as_millis() as u64;
        }

        ExecutionResult::Accepted { sample, results }
    }

    pub async fn execute_parallel(&self, samples: Vec<DataSample>) -> Vec<ExecutionResult> {
        use futures::future::join_all;

        let (cancel_tx, _) = tokio::sync::watch::channel(false);

        let tasks: Vec<_> = samples.into_iter()
            .map(|sample| {
                let cancel = cancel_tx.subscribe();
                async move { self.execute(sample, cancel).await }
            })
            .collect();

        join_all(tasks).await
    }

    pub fn run_rayon(&self, samples: Vec<DataSample>) -> Vec<ExecutionResult>
    where
        Self: Sync,
    {
        let (cancel_tx, _) = tokio::sync::watch::channel(false);
        let results: Vec<ExecutionResult> = samples.into_iter()
            .map(|sample| {
                let cancel = cancel_tx.subscribe();
                let rt = tokio::runtime::Handle::current();
                rt.block_on(self.execute(sample, cancel))
            })
            .collect();
        results
    }
}

#[derive(Debug)]
pub enum ExecutionResult {
    Accepted {
        sample: DataSample,
        results: Vec<FilterResult>,
    },
    Rejected {
        sample: DataSample,
        results: Vec<FilterResult>,
        filter_name: String,
        reason: Option<String>,
    },
    Rerouted {
        sample: DataSample,
        results: Vec<FilterResult>,
        filter_name: String,
        reason: Option<String>,
    },
    Cancelled,
}

impl ExecutionResult {
    pub fn is_accepted(&self) -> bool {
        matches!(self, ExecutionResult::Accepted { .. })
    }

    pub fn sample(&self) -> Option<&DataSample> {
        match self {
            ExecutionResult::Accepted { sample, .. } => Some(sample),
            ExecutionResult::Rejected { sample, .. } => Some(sample),
            ExecutionResult::Rerouted { sample, .. } => Some(sample),
            ExecutionResult::Cancelled => None,
        }
    }

    pub fn score(&self) -> f64 {
        match self {
            ExecutionResult::Accepted { results, .. } => {
                results.iter().map(|r| r.score_delta).sum()
            }
            _ => 0.0,
        }
    }
}
