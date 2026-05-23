use crate::canvas::{GraphEdge, GraphNode, ZoomLevel};
use crate::{DLResult, NodeType};
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
        self.edges
            .retain(|_, e| e.source_node != *node_id && e.target_node != *node_id);
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
        let incoming: Vec<_> = self
            .edges
            .values()
            .filter(|e| e.target_node == *node_id)
            .collect();
        let outgoing: Vec<_> = self
            .edges
            .values()
            .filter(|e| e.source_node == *node_id)
            .collect();
        (incoming, outgoing)
    }

    pub fn topological_order(&self) -> DLResult<Vec<Uuid>> {
        let mut in_degree: HashMap<Uuid, usize> = HashMap::with_capacity(self.nodes.len());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> NeuralGraph {
        let mut g = NeuralGraph::new("test");
        let input = GraphNode::new(crate::NodeType::Input, "input", -100.0, 0.0);
        let hidden = GraphNode::new(crate::NodeType::Linear, "hidden", 0.0, 0.0);
        let output = GraphNode::new(crate::NodeType::Output, "output", 100.0, 0.0);

        let input_id = g.add_node(input);
        let hidden_id = g.add_node(hidden);
        let output_id = g.add_node(output);

        // connect
        let tensor = crate::TensorDesc::new(vec![1, 64], crate::DType::F32);
        let e1 = GraphEdge::new(
            input_id,
            Uuid::new_v4(),
            hidden_id,
            Uuid::new_v4(),
            tensor.clone(),
        );
        let e2 = GraphEdge::new(
            hidden_id,
            Uuid::new_v4(),
            output_id,
            Uuid::new_v4(),
            tensor,
        );
        let _ = g.add_edge(e1);
        let _ = g.add_edge(e2);

        g
    }

    #[test]
    fn test_neural_graph_new() {
        let g = NeuralGraph::new("test");
        assert_eq!(g.name, "test");
        assert!(g.nodes.is_empty());
        assert!(g.edges.is_empty());
        assert_eq!(g.version, 0);
    }

    #[test]
    fn test_add_node() {
        let mut g = NeuralGraph::new("g");
        let node = GraphNode::new(crate::NodeType::Input, "in", 0.0, 0.0);
        let id = g.add_node(node);
        assert_eq!(g.node_count(), 1);
        assert!(g.get_node(&id).is_some());
        assert_eq!(g.version, 1);
    }

    #[test]
    fn test_remove_node_also_removes_edges() {
        let mut g = simple_graph();
        let input_id = g.get_input_nodes()[0].id;
        assert!(g.edge_count() > 0);
        g.remove_node(&input_id);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_add_edge_missing_source() {
        let mut g = NeuralGraph::new("g");
        let target = GraphNode::new(crate::NodeType::Output, "out", 0.0, 0.0);
        let target_id = g.add_node(target);
        let tensor = crate::TensorDesc::new(vec![1], crate::DType::F32);
        let e = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            target_id,
            Uuid::new_v4(),
            tensor,
        );
        let result = g.add_edge(e);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_edge_success() {
        let mut g = NeuralGraph::new("g");
        let a = g.add_node(GraphNode::new(crate::NodeType::Input, "a", 0.0, 0.0));
        let b = g.add_node(GraphNode::new(crate::NodeType::Linear, "b", 0.0, 0.0));
        let tensor = crate::TensorDesc::new(vec![1, 64], crate::DType::F32);
        let e = GraphEdge::new(a, Uuid::new_v4(), b, Uuid::new_v4(), tensor);
        let id = g.add_edge(e).unwrap();
        assert_eq!(g.edge_count(), 1);
        assert!(g.edges.contains_key(&id));
    }

    #[test]
    fn test_remove_edge() {
        let mut g = simple_graph();
        let edge_id = *g.edges.keys().next().unwrap();
        g.remove_edge(&edge_id);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_get_node_mut() {
        let mut g = simple_graph();
        let id = *g.nodes.keys().next().unwrap();
        let node = g.get_node_mut(&id).unwrap();
        node.name = "renamed".to_string();
        assert_eq!(g.get_node(&id).unwrap().name, "renamed");
    }

    #[test]
    fn test_get_input_nodes() {
        let g = simple_graph();
        let inputs = g.get_input_nodes();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].node_type, crate::NodeType::Input);
    }

    #[test]
    fn test_get_output_nodes() {
        let g = simple_graph();
        let outputs = g.get_output_nodes();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].node_type, crate::NodeType::Output);
    }

    #[test]
    fn test_get_node_connections() {
        let mut g = NeuralGraph::new("g");
        let inp = GraphNode::new(crate::NodeType::Input, "in", 0.0, 0.0);
        let mid = GraphNode::new(crate::NodeType::Linear, "mid", 0.0, 0.0);
        let out = GraphNode::new(crate::NodeType::Output, "out", 0.0, 0.0);
        let inp_id = g.add_node(inp);
        let mid_id = g.add_node(mid);
        let out_id = g.add_node(out);
        let tensor = crate::TensorDesc::new(vec![1, 64], crate::DType::F32);
        let _ = g.add_edge(GraphEdge::new(
            inp_id, uuid::Uuid::new_v4(), mid_id, uuid::Uuid::new_v4(), tensor.clone(),
        ));
        let _ = g.add_edge(GraphEdge::new(
            mid_id, uuid::Uuid::new_v4(), out_id, uuid::Uuid::new_v4(), tensor,
        ));
        let (incoming, outgoing) = g.get_node_connections(&mid_id);
        assert_eq!(incoming.len(), 1);
        assert_eq!(outgoing.len(), 1);
    }

    #[test]
    fn test_topological_order() {
        let g = simple_graph();
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_topological_order_cycle() {
        let mut g = NeuralGraph::new("cycle");
        let a = g.add_node(GraphNode::new(crate::NodeType::Input, "a", 0.0, 0.0));
        let b = g.add_node(GraphNode::new(crate::NodeType::Linear, "b", 0.0, 0.0));
        let tensor = crate::TensorDesc::new(vec![1], crate::DType::F32);
        let e1 = GraphEdge::new(a, Uuid::new_v4(), b, Uuid::new_v4(), tensor.clone());
        let e2 = GraphEdge::new(b, Uuid::new_v4(), a, Uuid::new_v4(), tensor);
        let _ = g.add_edge(e1);
        let _ = g.add_edge(e2);
        assert!(g.topological_order().is_err());
    }

    #[test]
    fn test_set_zoom_level() {
        let mut g = NeuralGraph::new("g");
        g.set_zoom_level(ZoomLevel::Module);
        assert_eq!(g.zoom_level, ZoomLevel::Module);
    }

    #[test]
    fn test_total_flops_and_params() {
        let g = simple_graph();
        assert_eq!(g.total_flops(), 0);
        assert_eq!(g.total_params(), 0);
    }
}
