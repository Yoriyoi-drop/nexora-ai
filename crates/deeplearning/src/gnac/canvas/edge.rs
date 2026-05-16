use crate::gnac::TensorDesc;
use crate::gnac::canvas::CanvasPosition;
use uuid::Uuid;

/// Status gradient pada SmartTensor edge
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientStatus {
    Stable,
    Exploding(f32),
    Vanishing(f32),
    Saturated,
}

/// Edge dalam graf GNAC — merepresentasikan SmartTensor
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub id: Uuid,
    pub source_node: Uuid,
    pub source_port: Uuid,
    pub target_node: Uuid,
    pub target_port: Uuid,
    pub tensor: TensorDesc,
    pub gradient: GradientStatus,
    pub entropy_score: f32,
    pub activation_distribution: ActivationStats,
    pub bandwidth_estimate: f64,
    pub memory_cost: usize,
    /// Routing path untuk feature-level routing
    pub feature_slice: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity: f32,
}

impl ActivationStats {
    pub fn new() -> Self {
        ActivationStats {
            mean: 0.0,
            std: 1.0,
            min: 0.0,
            max: 0.0,
            sparsity: 0.0,
        }
    }
}

impl GraphEdge {
    pub fn new(
        source_node: Uuid,
        source_port: Uuid,
        target_node: Uuid,
        target_port: Uuid,
        tensor: TensorDesc,
    ) -> Self {
        let memory_cost = tensor.numel * 4;
        GraphEdge {
            id: Uuid::new_v4(),
            source_node,
            source_port,
            target_node,
            target_port,
            tensor,
            gradient: GradientStatus::Stable,
            entropy_score: 0.0,
            activation_distribution: ActivationStats::new(),
            bandwidth_estimate: 0.0,
            memory_cost,
            feature_slice: None,
        }
    }

    pub fn with_feature_slice(mut self, start: usize, end: usize) -> Self {
        self.feature_slice = Some((start, end));
        self
    }

    pub fn update_gradient(&mut self, grad_norm: f32) {
        if grad_norm > 10.0 {
            self.gradient = GradientStatus::Exploding(grad_norm);
        } else if grad_norm < 1e-6 {
            self.gradient = GradientStatus::Vanishing(grad_norm);
        } else {
            self.gradient = GradientStatus::Stable;
        }
    }
}
