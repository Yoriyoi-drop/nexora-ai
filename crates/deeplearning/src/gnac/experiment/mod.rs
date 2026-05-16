//! Deterministic Experiment System
//!
//! Reproducibility penuh: setiap eksperimen menyimpan graph hash,
//! dataset fingerprint, optimizer state, seed random, runtime environment,
//! dan compiler configuration. Dilengkapi Graph Diff Viewer.

pub mod snapshot;
pub mod diff;

pub use snapshot::*;
pub use diff::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Representasi satu eksperimen
#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: Uuid,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub graph_snapshot: NeuralGraph,
    pub config: ExperimentConfig,
    pub metrics: ExperimentMetrics,
}

#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub seed: u64,
    pub dataset_fingerprint: String,
    pub optimizer: String,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub runtime_env: String,
    pub compiler_config: String,
}

#[derive(Debug, Clone, Default)]
pub struct ExperimentMetrics {
    pub final_accuracy: f32,
    pub final_loss: f32,
    pub training_time_secs: f64,
    pub peak_vram_mb: f64,
    pub total_flops: u64,
}

impl Experiment {
    pub fn new(name: &str, graph: NeuralGraph, config: ExperimentConfig) -> Self {
        Experiment {
            id: Uuid::new_v4(),
            name: name.to_string(),
            created_at: Utc::now(),
            graph_snapshot: graph,
            config,
            metrics: ExperimentMetrics::default(),
        }
    }
}
