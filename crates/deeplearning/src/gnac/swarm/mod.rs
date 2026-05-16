//! Swarm Agent: Constrained Neural Architecture Search
//!
//! Berbeda dari AutoML tradisional, Swarm Agent bekerja pada constrained
//! visual search space. Pengguna menentukan batas topologi, node yang
//! diizinkan, lokasi fusion, dan batas kompleksitas. Swarm Agent tidak
//! dapat menghasilkan arsitektur di luar ruang visual yang sudah didefinisikan.
//!
//! Objective: J(A) = α·Acc(A) − β·Latency(A) − γ·Memory(A) − δ·Energy(A)

pub mod search;
pub mod pruning;
pub mod objective;

pub use search::*;
pub use pruning::*;
pub use objective::*;

use crate::DLResult;
use crate::gnac::canvas::NeuralGraph;

/// Konfigurasi Swarm Agent
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    pub max_iterations: usize,
    pub population_size: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub alpha: f32, // accuracy weight
    pub beta: f32,  // latency penalty
    pub gamma: f32, // memory penalty
    pub delta: f32, // energy penalty
}

impl Default for SwarmConfig {
    fn default() -> Self {
        SwarmConfig {
            max_iterations: 100,
            population_size: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            alpha: 1.0,
            beta: 0.1,
            gamma: 0.05,
            delta: 0.01,
        }
    }
}
