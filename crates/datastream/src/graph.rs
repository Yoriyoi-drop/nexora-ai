use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::filter::Filter;
use crate::types::{DataSample, FilterAction, FilterMetric, FilterResult, PipelineMetrics};
use nexora_common::retry::RetryConfig;

const GRAPH_EXEC_CONCURRENT: usize = 64;

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
    semaphore: Arc<Semaphore>,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
            metrics: Arc::new(RwLock::new(PipelineMetrics::default())),
            cached_order: Arc::new(RwLock::new(None)),
            semaphore: Arc::new(Semaphore::new(GRAPH_EXEC_CONCURRENT)),
        }
    }

    pub fn add_node(
        &mut self,
        id: &str,
        filter: FilterArc,
        depends_on: Vec<String>,
        concurrent: bool,
        priority: u8,
    ) {
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

        self.entry_points = is_entry.iter().cloned().collect();

        self.exit_points = self
            .nodes
            .keys()
            .filter(|k| {
                self.nodes
                    .get(k.as_str())
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
        let mut in_degree: HashMap<String, usize> = self
            .nodes
            .keys()
            .map(|k| (k.clone(), self.nodes[k].depends_on.len()))
            .collect();

        let mut queue: VecDeque<String> = in_degree
            .iter()
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

        if result.len() != self.nodes.len() {
            let cycle_nodes: Vec<&String> = self
                .nodes
                .keys()
                .filter(|id| !result.contains(*id))
                .collect();
            warn!(
                "Cycle detected in execution graph — {} nodes are unreachable: {:?}",
                cycle_nodes.len(),
                cycle_nodes
            );

            // Break the cycle by removing one edge
            let mut broken = false;
            for node_id in &cycle_nodes {
                if let Some(node) = self.nodes.get(node_id.as_str()) {
                    for dep in &node.depends_on {
                        if cycle_nodes.contains(&dep) {
                            if let Some(deg) = in_degree.get_mut(*node_id) {
                                *deg = deg.saturating_sub(1);
                                warn!("Breaking cycle: removing edge {} -> {}", dep, node_id);
                                if *deg == 0 {
                                    queue.push_back((*node_id).clone());
                                }
                                broken = true;
                            }
                            break;
                        }
                    }
                }
                if broken {
                    break;
                }
            }

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
        }

        result
    }

    pub fn topological_order(&self) -> Vec<String> {
        self.cached_order
            .read()
            .clone()
            .unwrap_or_else(|| self.topological_order_inner())
    }

    pub async fn execute(
        &self,
        sample: DataSample,
        cancel: tokio::sync::watch::Receiver<bool>,
    ) -> ExecutionResult {
        let order = self.topological_order();
        let mut results: Vec<FilterResult> = Vec::with_capacity(order.len());
        let start = std::time::Instant::now();

        // Group nodes by topological depth for parallel execution
        let mut node_depths: HashMap<String, usize> = HashMap::new();
        for node_id in order.iter() {
            let node = match self.nodes.get(node_id.as_str()) {
                Some(n) => n,
                None => return ExecutionResult::Cancelled,
            };
            let depth = node
                .depends_on
                .iter()
                .filter_map(|dep| node_depths.get(dep.as_str()))
                .max()
                .copied()
                .unwrap_or(0)
                + 1;
            node_depths.insert(node_id.clone(), depth);
        }

        let max_depth = node_depths.values().max().copied().unwrap_or(0);
        let mut sample = Some(sample);
        for depth in 0..=max_depth {
            if *cancel.borrow() {
                return ExecutionResult::Cancelled;
            }

            let level_nodes: Vec<String> = node_depths
                .iter()
                .filter(|(_, &d)| d == depth)
                .map(|(id, _)| id.clone())
                .collect();

            if level_nodes.is_empty() {
                continue;
            }

            let sample_ref = match sample.take() {
                Some(s) => s,
                None => return ExecutionResult::Cancelled,
            };
            let sample_arc = std::sync::Arc::new(sample_ref);
            let mut filter_results: Vec<(String, FilterResult)> =
                Vec::with_capacity(level_nodes.len());

            let mut handles = Vec::with_capacity(level_nodes.len());
            if level_nodes.len() > 1 {
                for node_id in &level_nodes {
                    let permit = self.semaphore.clone().acquire_owned().await;
                    let permit = match permit {
                        Ok(p) => p,
                        Err(_) => return ExecutionResult::Cancelled,
                    };
                    let filter = self.nodes[node_id].filter.clone();
                    let sample = sample_arc.clone();
                    let node_id = node_id.clone();
                    handles.push(tokio::spawn(async move {
                        let _permit = permit;
                        let result = RetryConfig::default()
                            .retry(|| {
                                let filter = filter.clone();
                                let sample = sample.clone();
                                async move { Ok(filter.filter(&*sample).await) }
                            })
                            .await
                            .unwrap_or_else(|e| {
                                warn!("Filter '{}' failed after retries: {}", node_id, e);
                                FilterResult {
                                    passed: false,
                                    sample_id: sample.id,
                                    filter_name: filter.name().to_string(),
                                    reason: Some(e),
                                    score_delta: 0.0,
                                }
                            });
                        (node_id, result)
                    }));
                }
            } else {
                let node_id = level_nodes[0].clone();
                let permit = self.semaphore.clone().acquire_owned().await;
                let permit = match permit {
                    Ok(p) => p,
                    Err(_) => return ExecutionResult::Cancelled,
                };
                let filter = self.nodes[&node_id].filter.clone();
                let sample = sample_arc.clone();
                handles.push(tokio::spawn(async move {
                    let _permit = permit;
                    let result = RetryConfig::default()
                        .retry(|| {
                            let filter = filter.clone();
                            let sample = sample.clone();
                            async move { Ok::<_, String>(filter.filter(&*sample).await) }
                        })
                        .await
                        .unwrap_or_else(|e| {
                            warn!("Filter '{}' failed after retries: {}", node_id, e);
                            FilterResult {
                                passed: false,
                                sample_id: sample.id,
                                filter_name: filter.name().to_string(),
                                reason: Some(e),
                                score_delta: 0.0,
                            }
                        });
                    (node_id, result)
                }));
            }
            for handle in handles {
                match handle.await {
                    Ok(result) => filter_results.push(result),
                    Err(e) => {
                        warn!("Filter task join error: {:?}, cancelling execution", e);
                        return ExecutionResult::Cancelled;
                    }
                }
            }

            sample =
                Some(std::sync::Arc::try_unwrap(sample_arc).unwrap_or_else(|arc| (*arc).clone()));

            for (node_id, filter_result) in &filter_results {
                debug!("Filter '{}': passed={}", node_id, filter_result.passed);

                {
                    let mut metrics = self.metrics.write();
                    metrics.samples_in += 1;
                    let entry = metrics
                        .filter_breakdown
                        .entry(node_id.clone())
                        .or_insert_with(|| FilterMetric {
                            processed: 0,
                            passed: 0,
                            rejected: 0,
                            avg_latency_us: 0.0,
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
                results.push(filter_result.clone());

                if !passed {
                    let sample = match sample.take() {
                        Some(s) => s,
                        None => return ExecutionResult::Cancelled,
                    };
                    let node = &self.nodes[node_id.as_str()];
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
        }

        let elapsed = start.elapsed();
        {
            let mut metrics = self.metrics.write();
            metrics.samples_accepted += 1;
            metrics.total_latency_ms += elapsed.as_millis() as u64;
        }

        let sample = match sample.take() {
            Some(s) => s,
            None => return ExecutionResult::Cancelled,
        };
        ExecutionResult::Accepted { sample, results }
    }

    pub async fn execute_parallel(&self, samples: Vec<DataSample>) -> Vec<ExecutionResult> {
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        let tasks: Vec<_> = samples
            .into_iter()
            .map(|sample| {
                let cancel = cancel_rx.clone();
                async move { self.execute(sample, cancel).await }
            })
            .collect();

        let results = futures::future::join_all(tasks).await;
        for result in &results {
            if matches!(result, ExecutionResult::Cancelled) {
                tracing::error!("Pipeline task was cancelled");
                let _ = cancel_tx.send(true);
                break;
            }
        }
        results
    }

    pub async fn run_parallel(&self, samples: Vec<DataSample>) -> Vec<ExecutionResult> {
        let (cancel_tx, _) = tokio::sync::watch::channel(false);
        let handles: Vec<_> = samples
            .into_iter()
            .map(|sample| {
                let cancel = cancel_tx.subscribe();
                let this = self.clone();
                tokio::spawn(async move { this.execute(sample, cancel).await })
            })
            .collect();
        futures::future::join_all(handles)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};
    use async_trait::async_trait;
    use uuid::Uuid;

    fn dummy_filter(name: &str, always_pass: bool) -> FilterArc {
        let name = name.to_string();
        Arc::new(DummyFilter { name, always_pass })
    }

    #[derive(Debug)]
    struct DummyFilter {
        name: String,
        always_pass: bool,
    }

    #[async_trait]
    impl Filter for DummyFilter {
        fn name(&self) -> &str {
            &self.name
        }
        fn action(&self) -> FilterAction {
            if self.always_pass {
                FilterAction::Accept
            } else {
                FilterAction::Reject
            }
        }
        async fn evaluate(&self, sample: &DataSample) -> FilterResult {
            FilterResult {
                passed: self.always_pass,
                sample_id: sample.id,
                filter_name: self.name.clone(),
                reason: None,
                score_delta: if self.always_pass { 0.1 } else { -0.5 },
            }
        }
    }

    fn sample() -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: "test".into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_graph_new() {
        let g = ExecutionGraph::new();
        assert!(g.nodes.is_empty());
        assert!(g.entry_points.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut g = ExecutionGraph::new();
        g.add_node("filter1", dummy_filter("f1", true), vec![], false, 0);
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn test_finalize_empty_graph() {
        let mut g = ExecutionGraph::new();
        g.finalize();
        assert!(g.entry_points.is_empty());
    }

    #[test]
    fn test_finalize_single_node() {
        let mut g = ExecutionGraph::new();
        g.add_node("only", dummy_filter("only", true), vec![], false, 0);
        g.finalize();
        assert_eq!(g.entry_points, vec!["only"]);
        assert_eq!(g.exit_points, vec!["only"]);
    }

    #[test]
    fn test_finalize_with_dependency() {
        let mut g = ExecutionGraph::new();
        g.add_node("a", dummy_filter("a", true), vec![], false, 0);
        g.add_node("b", dummy_filter("b", true), vec!["a".into()], false, 0);
        g.finalize();
        assert_eq!(g.entry_points, vec!["a"]);
        assert_eq!(g.exit_points, vec!["b"]);
    }

    #[test]
    fn test_topological_order() {
        let mut g = ExecutionGraph::new();
        g.add_node("a", dummy_filter("a", true), vec![], false, 0);
        g.add_node("b", dummy_filter("b", true), vec!["a".into()], false, 0);
        g.add_node("c", dummy_filter("c", true), vec!["a".into()], false, 0);
        g.finalize();
        let order = g.topological_order();
        assert_eq!(order[0], "a");
        assert!(order.contains(&"b".to_string()));
        assert!(order.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn test_execute_accepts() {
        let mut g = ExecutionGraph::new();
        g.add_node("pass", dummy_filter("pass", true), vec![], false, 0);
        g.finalize();
        let (_, rx) = tokio::sync::watch::channel(false);
        let result = g.execute(sample(), rx).await;
        assert!(result.is_accepted());
    }

    #[tokio::test]
    async fn test_execute_rejects() {
        let mut g = ExecutionGraph::new();
        g.add_node("fail", dummy_filter("fail", false), vec![], false, 0);
        g.finalize();
        let (_, rx) = tokio::sync::watch::channel(false);
        let result = g.execute(sample(), rx).await;
        assert!(!result.is_accepted());
        match &result {
            ExecutionResult::Rejected { filter_name, .. } => {
                assert_eq!(filter_name, "fail");
            }
            other => {
                panic!("Expected ExecutionResult::Rejected, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_execution_result_score() {
        let sample = sample();
        let results = vec![FilterResult {
            passed: true,
            sample_id: sample.id,
            filter_name: "test".into(),
            reason: None,
            score_delta: 0.5,
        }];
        let accepted = ExecutionResult::Accepted { sample, results };
        assert_eq!(accepted.score(), 0.5);
    }

    #[test]
    fn test_execution_result_cancelled_score() {
        assert_eq!(ExecutionResult::Cancelled.score(), 0.0);
    }

    #[test]
    fn test_execution_result_sample() {
        let s = sample();
        let accepted = ExecutionResult::Accepted {
            sample: s.clone(),
            results: vec![],
        };
        assert!(accepted.sample().is_some());
        assert_eq!(ExecutionResult::Cancelled.sample(), None);
    }

    #[test]
    fn test_graph_node_debug() {
        let node = GraphNode {
            id: "test".into(),
            filter: dummy_filter("test", true),
            depends_on: vec![],
            children: vec![],
            concurrent: false,
            priority: 1,
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("test"));
    }
}
