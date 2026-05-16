use crate::gnac::canvas::NeuralGraph;
use crate::gnac::swarm::SwarmConfig;
use crate::gnac::rce::ResourceEstimator;

/// Hitung fitness: J(A) = α·Acc − β·Latency − γ·Memory − δ·Energy
pub fn compute_fitness(graph: &NeuralGraph, accuracy: f32, config: &SwarmConfig) -> f32 {
    let report = ResourceEstimator::estimate(graph);

    let normalized_accuracy = accuracy; // 0.0 - 1.0
    let normalized_latency = (report.inference_latency_ms as f32 / 1000.0).min(1.0);
    let normalized_memory = (report.total_vram_mb as f32 / 80_000.0).min(1.0);
    let normalized_energy = (report.estimated_energy_joules as f32 / 500.0).min(1.0);

    config.alpha * normalized_accuracy
        - config.beta * normalized_latency
        - config.gamma * normalized_memory
        - config.delta * normalized_energy
}

/// Hitung accuracy estimasi berdasarkan arsitektur
pub fn estimate_accuracy(graph: &NeuralGraph) -> f32 {
    let node_count = graph.node_count();
    let param_count = graph.total_params();

    // Model heuristic: lebih banyak parameter = akurasi lebih tinggi (sampai batas)
    let param_factor = (param_count as f32 / 1_000_000.0).min(1.0);

    // Terlalu sedikit atau terlalu banyak node menurunkan akurasi
    let depth_factor = if node_count < 3 {
        0.3
    } else if node_count > 100 {
        0.5
    } else {
        1.0 - (node_count as f32 - 10.0).abs() / 100.0
    };

    (param_factor * 0.6 + depth_factor * 0.4).clamp(0.0, 1.0)
}
