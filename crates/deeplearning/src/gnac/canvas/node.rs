use crate::gnac::{NodeType, TensorDesc, DType, HealthStatus};
use crate::gnac::canvas::{CanvasPosition, ZoomLevel, PortDescriptor, PortDirection};
use std::collections::HashMap;
use uuid::Uuid;

/// Node dalam graf GNAC
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: Uuid,
    pub name: String,
    pub node_type: NodeType,
    pub position: CanvasPosition,
    pub zoom_level: ZoomLevel,
    pub inputs: Vec<PortDescriptor>,
    pub outputs: Vec<PortDescriptor>,
    pub params: NodeParams,
    pub health: HealthStatus,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone)]
pub struct NodeParams {
    pub hyperparameters: HashMap<String, f64>,
    pub string_params: HashMap<String, String>,
    pub enabled: bool,
    pub precision_override: Option<DType>,
}

impl NodeParams {
    pub fn new() -> Self {
        NodeParams {
            hyperparameters: HashMap::new(),
            string_params: HashMap::new(),
            enabled: true,
            precision_override: None,
        }
    }

    pub fn with_hparam(mut self, key: &str, value: f64) -> Self {
        self.hyperparameters.insert(key.to_string(), value);
        self
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.string_params.insert(key.to_string(), value.to_string());
        self
    }
}

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub flops: u64,
    pub params_count: usize,
    pub activation_size: usize,
    pub description: String,
    pub tags: Vec<String>,
}

impl GraphNode {
    pub fn new(node_type: NodeType, name: &str, x: f64, y: f64) -> Self {
        let inputs = create_default_inputs(&node_type);
        let outputs = create_default_outputs(&node_type);

        GraphNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            node_type,
            position: CanvasPosition::new(x, y),
            zoom_level: ZoomLevel::Tensor,
            inputs,
            outputs,
            params: NodeParams::new(),
            health: HealthStatus::Healthy,
            metadata: NodeMetadata {
                flops: 0,
                params_count: 0,
                activation_size: 0,
                description: String::new(),
                tags: Vec::new(),
            },
        }
    }

    pub fn set_health(&mut self, status: HealthStatus) {
        self.health = status;
    }

    pub fn toggle_enabled(&mut self) {
        self.params.enabled = !self.params.enabled;
    }
}

fn create_default_inputs(node_type: &NodeType) -> Vec<PortDescriptor> {
    match node_type {
        NodeType::Conv2D | NodeType::Conv1D | NodeType::Conv3D => {
            vec![
                PortDescriptor::new("input", PortDirection::Input, TensorDesc::new(vec![1, 3, 224, 224], DType::F32)),
                PortDescriptor::new("weight", PortDirection::Input, TensorDesc::new(vec![64, 3, 3, 3], DType::F32)),
            ]
        }
        NodeType::SelfAttention | NodeType::MultiHeadAttention => {
            vec![
                PortDescriptor::new("query", PortDirection::Input, TensorDesc::new(vec![1, 128, 768], DType::F32)),
                PortDescriptor::new("key", PortDirection::Input, TensorDesc::new(vec![1, 128, 768], DType::F32)),
                PortDescriptor::new("value", PortDirection::Input, TensorDesc::new(vec![1, 128, 768], DType::F32)),
            ]
        }
        NodeType::Linear => {
            vec![
                PortDescriptor::new("input", PortDirection::Input, TensorDesc::new(vec![1, 768], DType::F32)),
                PortDescriptor::new("weight", PortDirection::Input, TensorDesc::new(vec![768, 3072], DType::F32)),
            ]
        }
        NodeType::Input => {
            vec![]
        }
        _ => {
            vec![
                PortDescriptor::new("input", PortDirection::Input, TensorDesc::new(vec![1, 64], DType::F32)),
            ]
        }
    }
}

fn create_default_outputs(node_type: &NodeType) -> Vec<PortDescriptor> {
    match node_type {
        NodeType::Conv2D => {
            vec![
                PortDescriptor::new("output", PortDirection::Output, TensorDesc::new(vec![1, 64, 224, 224], DType::F32)),
            ]
        }
        NodeType::SelfAttention | NodeType::MultiHeadAttention => {
            vec![
                PortDescriptor::new("output", PortDirection::Output, TensorDesc::new(vec![1, 128, 768], DType::F32)),
                PortDescriptor::new("attention_weights", PortDirection::Output, TensorDesc::new(vec![1, 12, 128, 128], DType::F32)),
            ]
        }
        NodeType::Linear => {
            vec![
                PortDescriptor::new("output", PortDirection::Output, TensorDesc::new(vec![1, 3072], DType::F32)),
            ]
        }
        NodeType::Output => {
            vec![]
        }
        _ => {
            vec![
                PortDescriptor::new("output", PortDirection::Output, TensorDesc::new(vec![1, 64], DType::F32)),
            ]
        }
    }
}
