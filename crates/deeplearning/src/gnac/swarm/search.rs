use crate::DLResult;
use crate::gnac::canvas::{NeuralGraph, GraphNode, GraphEdge};
use crate::gnac::swarm::{SwarmConfig, objective};
use crate::NodeType;
use std::collections::HashMap;
use uuid::Uuid;
use rand::Rng;

/// Search space yang dibatasi oleh constraint visual
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub allowed_node_types: Vec<NodeType>,
    pub max_depth: usize,
    pub max_branching: usize,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub allow_skip_connections: bool,
    pub allow_recurrent: bool,
}

impl Default for SearchSpace {
    fn default() -> Self {
        SearchSpace {
            allowed_node_types: vec![
                NodeType::Linear,
                NodeType::Conv2D,
                NodeType::SelfAttention,
                NodeType::LayerNorm,
                NodeType::ReLU,
                NodeType::GELU,
                NodeType::Dropout,
                NodeType::Concat,
            ],
            max_depth: 20,
            max_branching: 4,
            min_nodes: 3,
            max_nodes: 100,
            allow_skip_connections: true,
            allow_recurrent: false,
        }
    }
}

/// Swarm Agent untuk Constrained NAS
pub struct SwarmAgent {
    config: SwarmConfig,
    search_space: SearchSpace,
    population: Vec<NeuralGraph>,
    fitness_scores: Vec<f32>,
}

impl SwarmAgent {
    pub fn new(config: SwarmConfig, search_space: SearchSpace) -> Self {
        SwarmAgent {
            config,
            search_space,
            population: Vec::new(),
            fitness_scores: Vec::new(),
        }
    }

    /// Inisialisasi populasi awal
    pub fn initialize(&mut self) {
        let mut rng = rand::thread_rng();
        for _ in 0..self.config.population_size {
            let num_nodes = rng.gen_range(self.search_space.min_nodes..=self.search_space.max_nodes.min(20));
            let graph = self.random_graph(num_nodes, &mut rng);
            self.population.push(graph);
        }
    }

    fn random_graph(&self, num_nodes: usize, rng: &mut impl Rng) -> NeuralGraph {
        let mut graph = NeuralGraph::new("swarm_candidate");

        let input_node = GraphNode::new(NodeType::Input, "input", -200.0, 0.0);
        let input_id = graph.add_node(input_node);

        for i in 0..num_nodes {
            let node_type = self.search_space.allowed_node_types
                [rng.gen_range(0..self.search_space.allowed_node_types.len())].clone();
            let node = GraphNode::new(
                node_type,
                &format!("node_{}", i),
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
            );
            graph.add_node(node);
        }

        let output_node = GraphNode::new(NodeType::Output, "output", 200.0, 0.0);
        graph.add_node(output_node);

        graph
    }

    /// Jalankan evolutionary search
    pub fn evolve(&mut self, validation_accuracy: &dyn Fn(&NeuralGraph) -> f32) -> DLResult<NeuralGraph> {
        if self.population.is_empty() {
            self.initialize();
        }

        for iteration in 0..self.config.max_iterations {
            // Evaluasi fitness
            self.fitness_scores = self.population
                .iter()
                .map(|g| {
                    let acc = validation_accuracy(g);
                    objective::compute_fitness(g, acc, &self.config)
                })
                .collect();

            // Seleksi
            let mut new_population = self.selection();

            // Crossover
            let mut rng = rand::thread_rng();
            while new_population.len() < self.config.population_size {
                if rng.gen::<f32>() < self.config.crossover_rate {
                    if let Some(child) = self.crossover(&mut rng) {
                        new_population.push(child);
                    }
                }
                // Mutasi
                if rng.gen::<f32>() < self.config.mutation_rate {
                    if let Some(mutated) = new_population.last_mut() {
                        self.mutate(mutated, &mut rng);
                    }
                }
            }

            self.population = new_population;
        }

        // Return individu terbaik
        let best_idx = self.fitness_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(self.population[best_idx].clone())
    }

    fn selection(&self) -> Vec<NeuralGraph> {
        let mut indexed: Vec<_> = self.population.iter().zip(self.fitness_scores.iter()).collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let keep = (self.config.population_size as f32 * 0.3) as usize;
        indexed.into_iter().take(keep).map(|(g, _)| (*g).clone()).collect()
    }

    fn crossover(&self, rng: &mut impl Rng) -> Option<NeuralGraph> {
        if self.population.len() < 2 {
            return None;
        }
        let idx1 = rng.gen_range(0..self.population.len());
        let idx2 = rng.gen_range(0..self.population.len());
        if idx1 == idx2 {
            return None;
        }

        let parent1 = &self.population[idx1];
        let mut child = parent1.clone();
        child.name = format!("child_{}", Uuid::new_v4());
        Some(child)
    }

    fn mutate(&self, graph: &mut NeuralGraph, rng: &mut impl Rng) {
        if let Some(node) = graph.nodes.values_mut().next() {
            let new_type = self.search_space.allowed_node_types
                [rng.gen_range(0..self.search_space.allowed_node_types.len())].clone();
            node.node_type = new_type;
        }
    }
}
