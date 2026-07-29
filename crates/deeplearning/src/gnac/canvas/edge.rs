use crate::gnac::TensorDesc;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnac::TensorDesc;

    fn edge() -> GraphEdge {
        let src = Uuid::new_v4();
        let tgt = Uuid::new_v4();
        let tensor = TensorDesc::new(vec![1, 64], crate::gnac::DType::F32);
        GraphEdge::new(src, Uuid::new_v4(), tgt, Uuid::new_v4(), tensor)
    }

    #[test]
    fn test_graph_edge_new() {
        let e = edge();
        assert_eq!(e.gradient, GradientStatus::Stable);
        assert!(e.feature_slice.is_none());
    }

    #[test]
    fn test_graph_edge_with_feature_slice() {
        let e = edge().with_feature_slice(0, 32);
        assert_eq!(e.feature_slice, Some((0, 32)));
    }

    #[test]
    fn test_update_gradient_stable() {
        let mut e = edge();
        e.update_gradient(1.0);
        assert_eq!(e.gradient, GradientStatus::Stable);
    }

    #[test]
    fn test_update_gradient_exploding() {
        let mut e = edge();
        e.update_gradient(100.0);
        assert!(matches!(e.gradient, GradientStatus::Exploding(n) if (n - 100.0).abs() < 1e-5));
    }

    #[test]
    fn test_update_gradient_vanishing() {
        let mut e = edge();
        e.update_gradient(1e-10);
        assert!(matches!(e.gradient, GradientStatus::Vanishing(_)));
    }

    #[test]
    fn test_activation_stats_new() {
        let stats = ActivationStats::new();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 1.0);
        assert_eq!(stats.sparsity, 0.0);
    }

    #[test]
    fn test_gradient_status_clone() {
        let g = GradientStatus::Exploding(5.0);
        let h = g.clone();
        assert!(matches!(h, GradientStatus::Exploding(v) if (v - 5.0).abs() < 1e-5));
    }
}
