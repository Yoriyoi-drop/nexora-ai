use crate::gnac::canvas::NeuralGraph;
use crate::gnac::experiment::{Experiment, ExperimentConfig};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Snapshot — ambil state reproducible
pub struct ExperimentSnapshot;

impl ExperimentSnapshot {
    /// Buat snapshot baru dari graf
    pub fn capture(graph: &NeuralGraph, name: &str) -> Experiment {
        let config = ExperimentConfig {
            seed: rand::random(),
            dataset_fingerprint: String::new(),
            optimizer: "adam".to_string(),
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            runtime_env: std::env::consts::ARCH.to_string(),
            compiler_config: "default".to_string(),
        };

        Experiment::new(name, graph.clone(), config)
    }

    /// Hitung hash unik untuk sebuah graf
    pub fn graph_hash(graph: &NeuralGraph) -> String {
        let mut hasher = DefaultHasher::new();
        graph.name.hash(&mut hasher);
        graph.node_count().hash(&mut hasher);
        graph.edge_count().hash(&mut hasher);
        graph.total_flops().hash(&mut hasher);
        graph.total_params().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}
