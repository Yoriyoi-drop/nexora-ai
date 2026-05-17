use crate::gnac::{DLResult, NodeType};
use crate::gnac::canvas::{GraphNode, GraphEdge, ZoomLevel};
use std::collections::HashMap;
use uuid::Uuid;

/// Graf utama GNAC — merepresentasikan seluruh arsitektur neural
#[derive(Debug, Clone)]
pub struct NeuralGraph {
    pub id: Uuid,
    pub name: String,
    pub nodes: HashMap<Uuid, GraphNode>,
    pub edges: HashMap<Uuid, GraphEdge>,
    pub version: u64,
    pub zoom_level: ZoomLevel,
}

impl NeuralGraph {
    pub fn new(name: &str) -> Self {
        NeuralGraph {
            id: Uuid::new_v4(),
            name: name.to_string(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            version: 0,
            zoom_level: ZoomLevel::System,
        }
    }

    pub fn add_node(&mut self, node: GraphNode) -> Uuid {
        let id = node.id;
        self.nodes.insert(id, node);
        self.version += 1;
        id
    }

    pub fn remove_node(&mut self, node_id: &Uuid) {
        self.nodes.remove(node_id);
        self.edges.retain(|_, e| e.source_node != *node_id && e.target_node != *node_id);
        self.version += 1;
    }

    pub fn add_edge(&mut self, edge: GraphEdge) -> DLResult<Uuid> {
        let id = edge.id;
        // Validasi source dan target node exist
        if !self.nodes.contains_key(&edge.source_node) {
            return Err(crate::DeepLearningError::Configuration {
                reason: format!("Source node {} not found", edge.source_node),
            });
        }
        if !self.nodes.contains_key(&edge.target_node) {
            return Err(crate::DeepLearningError::Configuration {
                reason: format!("Target node {} not found", edge.target_node),
            });
        }
        self.edges.insert(id, edge);
        self.version += 1;
        Ok(id)
    }

    pub fn remove_edge(&mut self, edge_id: &Uuid) {
        self.edges.remove(edge_id);
        self.version += 1;
    }

    pub fn get_node(&self, id: &Uuid) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: &Uuid) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    pub fn get_input_nodes(&self) -> Vec<&GraphNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == NodeType::Input)
            .collect()
    }

    pub fn get_output_nodes(&self) -> Vec<&GraphNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == NodeType::Output)
            .collect()
    }

    pub fn get_node_connections(&self, node_id: &Uuid) -> (Vec<&GraphEdge>, Vec<&GraphEdge>) {
        let incoming: Vec<_> = self.edges.values().filter(|e| e.target_node == *node_id).collect();
        let outgoing: Vec<_> = self.edges.values().filter(|e| e.source_node == *node_id).collect();
        (incoming, outgoing)
    }

    pub fn topological_order(&self) -> DLResult<Vec<Uuid>> {
        let mut in_degree: HashMap<Uuid, usize> = HashMap::new();
        for node_id in self.nodes.keys() {
            in_degree.entry(*node_id).or_insert(0);
        }
        for edge in self.edges.values() {
            *in_degree.entry(edge.target_node).or_insert(0) += 1;
        }

        let mut queue: Vec<Uuid> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(node) = queue.pop() {
            order.push(node);
            for edge in self.edges.values() {
                if edge.source_node == node {
                    if let Some(deg) = in_degree.get_mut(&edge.target_node) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(edge.target_node);
                        }
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(crate::DeepLearningError::Computation {
                reason: "Graph contains a cycle".to_string(),
            });
        }

        Ok(order)
    }

    pub fn set_zoom_level(&mut self, level: ZoomLevel) {
        self.zoom_level = level;
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn total_flops(&self) -> u64 {
        self.nodes.values().map(|n| n.metadata.flops).sum()
    }

    pub fn total_params(&self) -> usize {
        self.nodes.values().map(|n| n.metadata.params_count).sum()
    }
}
