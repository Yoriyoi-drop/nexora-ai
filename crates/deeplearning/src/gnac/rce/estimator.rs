use crate::gnac::canvas::NeuralGraph;
use crate::gnac::rce::ResourceReport;
use crate::NodeType;

/// Estimator resource untuk graf neural
pub struct ResourceEstimator;

impl ResourceEstimator {
    /// Estimasi resource untuk seluruh graf
    pub fn estimate(graph: &NeuralGraph) -> ResourceReport {
        let mut report = ResourceReport::new();

        for node in graph.nodes.values() {
            let (node_flops, node_params, node_activation) = Self::estimate_node(&node.node_type);
            report.total_flops += node_flops;
            report.parameter_count += node_params;
            report.activation_memory_mb += node_activation as f64 / 1_000_000.0;
        }

        // VRAM = weights + activations + gradients (optimizer states 2x weights)
        let weights_mb = (report.parameter_count * 4) as f64 / 1_000_000.0;
        let optimizer_mb = weights_mb * 2.0;
        report.total_vram_mb = weights_mb + report.activation_memory_mb + optimizer_mb;

        // Latensi estimasi: 1 FLOP ≈ 1ns pada GPU modern
        report.inference_latency_ms = report.total_flops as f64 * 1e-6;

        // Bandwidth
        report.tensor_bandwidth_gbs = report.activation_memory_mb / (report.inference_latency_ms.max(1.0) / 1000.0) / 1000.0;

        // Cloud cost
        report.estimated_cloud_cost_per_hour = ResourceReport::estimate_cloud_cost(report.total_flops, report.total_vram_mb);

        report
    }

    fn estimate_node(node_type: &NodeType) -> (u64, usize, usize) {
        match node_type {
            NodeType::Conv2D => (500_000_000, 64 * 3 * 3 * 3 + 64, 64 * 224 * 224 * 4),
            NodeType::SelfAttention | NodeType::MultiHeadAttention => (200_000_000, 3 * 768 * 768, 128 * 768 * 4),
            NodeType::Linear => (2 * 768 * 3072 as u64, 768 * 3072 + 3072, 3072 * 4),
            NodeType::LayerNorm | NodeType::RMSNorm => (2 * 768, 768 * 2, 768 * 4),
            NodeType::ReLU | NodeType::GELU => (768, 0, 768 * 4),
            NodeType::Dropout => (0, 0, 768 * 4),
            NodeType::Embedding => (0, 50000 * 768, 768 * 4),
            NodeType::Input | NodeType::Output => (0, 0, 0),
            NodeType::Concat => (0, 0, 0),
            _ => (1_000_000, 1000, 4096),
        }
    }
}
