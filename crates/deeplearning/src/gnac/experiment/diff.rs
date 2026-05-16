use crate::gnac::canvas::NeuralGraph;
use crate::gnac::experiment::Experiment;
use std::collections::HashSet;

/// Graf Diff
#[derive(Debug, Clone)]
pub struct GraphDiff {
    pub added_nodes: Vec<String>,
    pub removed_nodes: Vec<String>,
    pub modified_nodes: Vec<String>,
    pub added_edges: Vec<String>,
    pub removed_edges: Vec<String>,
    pub param_change: i64,
    pub flop_change: i64,
    pub summary: String,
}

impl GraphDiff {
    /// Diff antara dua graf
    pub fn between(before: &NeuralGraph, after: &NeuralGraph) -> Self {
        let before_nodes: HashSet<_> = before.nodes.keys().collect();
        let after_nodes: HashSet<_> = after.nodes.keys().collect();

        let added: Vec<_> = after_nodes.difference(&before_nodes)
            .map(|id| after.get_node(id).map(|n| n.name.clone()).unwrap_or_default())
            .collect();

        let removed: Vec<_> = before_nodes.difference(&after_nodes)
            .map(|id| before.get_node(id).map(|n| n.name.clone()).unwrap_or_default())
            .collect();

        let modified: Vec<_> = before_nodes.intersection(&after_nodes)
            .filter(|id| {
                let before_node = before.get_node(id);
                let after_node = after.get_node(id);
                before_node.map(|b| b.metadata.params_count) != after_node.map(|a| a.metadata.params_count)
            })
            .map(|id| after.get_node(id).map(|n| n.name.clone()).unwrap_or_default())
            .collect();

        let param_change = after.total_params() as i64 - before.total_params() as i64;
        let flop_change = after.total_flops() as i64 - before.total_flops() as i64;

        let summary = format!(
            "Diff: +{} nodes, -{} nodes, {} modified, params: {:+}, FLOPs: {:+}",
            added.len(), removed.len(), modified.len(), param_change, flop_change
        );

        GraphDiff {
            added_nodes: added,
            removed_nodes: removed,
            modified_nodes: modified,
            added_edges: Vec::new(),
            removed_edges: Vec::new(),
            param_change,
            flop_change,
            summary,
        }
    }

    /// Diff antara dua experiment
    pub fn between_experiments(exp_a: &Experiment, exp_b: &Experiment) -> Self {
        Self::between(&exp_a.graph_snapshot, &exp_b.graph_snapshot)
    }
}
