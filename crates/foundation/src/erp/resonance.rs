//! ERP Resonance Graph Clustering
//! 
//! Implementasi dari resonance graph clustering algorithm dengan
//! Louvain clustering, spectral clustering, dan adaptive modular partitioning.

use crate::erp::{ERPConfig, ERPError};
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Advanced resonance graph clustering algorithms
pub struct ResonanceClusterer {
    config: ERPConfig,
    clustering_method: ClusteringMethod,
}

#[derive(Debug, Clone)]
pub enum ClusteringMethod {
    Louvain { resolution: f32 },
    Spectral { n_clusters: usize },
    AdaptiveModular { min_size: usize, max_size: usize },
}

impl ResonanceClusterer {
    pub fn new(config: ERPConfig) -> Self {
        let clustering_method = match config.compression_mode {
            crate::erp::CompressionMode::Conservative => ClusteringMethod::Louvain { resolution: 1.0 },
            crate::erp::CompressionMode::Balanced => ClusteringMethod::Spectral { n_clusters: 32 },
            crate::erp::CompressionMode::Aggressive => ClusteringMethod::AdaptiveModular { min_size: 2, max_size: 16 },
        };

        Self {
            config,
            clustering_method,
        }
    }

    /// Cluster resonance graph ke dalam resonance groups
    pub fn cluster(&self, graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature]) -> Result<Vec<ResonanceGroup>, ERPError> {
        match &self.clustering_method {
            ClusteringMethod::Louvain { resolution } => {
                self.louvain_clustering(graph, signatures, *resolution)
            }
            ClusteringMethod::Spectral { n_clusters } => {
                self.spectral_clustering(graph, signatures, *n_clusters)
            }
            ClusteringMethod::AdaptiveModular { min_size, max_size } => {
                self.adaptive_modular_clustering(graph, signatures, *min_size, *max_size)
            }
        }
    }

    /// Louvain clustering untuk community detection
    fn louvain_clustering(&self, graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature], resolution: f32) -> Result<Vec<ResonanceGroup>, ERPError> {
        let n_nodes = graph.adjacency.len();
        let mut communities: Vec<usize> = (0..n_nodes).collect();
        let mut changed = true;
        let mut iteration = 0;

        while changed && iteration < 100 {
            changed = false;
            iteration += 1;

            for node in 0..n_nodes {
                let current_community = communities[node];
                let best_community = self.find_best_community(node, &communities, graph, signatures, resolution);
                
                if best_community != current_community {
                    communities[node] = best_community;
                    changed = true;
                }
            }
        }

        // Convert communities ke resonance groups
        self.communities_to_groups(communities, signatures)
    }

    /// Find best community untuk node menggunakan modularity optimization
    fn find_best_community(&self, node: usize, communities: &[usize], graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature], resolution: f32) -> usize {
        let mut best_community = communities[node];
        let mut best_modularity = self.compute_modularity_gain(node, best_community, communities, graph, signatures, resolution);

        // Get neighboring communities
        let mut neighboring_communities = HashSet::new();
        for &(neighbor, _) in &graph.adjacency[node] {
            neighboring_communities.insert(communities[neighbor]);
        }

        for &community in &neighboring_communities {
            let modularity_gain = self.compute_modularity_gain(node, community, communities, graph, signatures, resolution);
            if modularity_gain > best_modularity {
                best_modularity = modularity_gain;
                best_community = community;
            }
        }

        best_community
    }

    /// Compute modularity gain untuk memindahkan node ke community
    fn compute_modularity_gain(&self, node: usize, community: usize, communities: &[usize], graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature], resolution: f32) -> f32 {
        let mut intra_community_weight = 0.0;
        let mut total_community_weight = 0.0;
        let mut node_weight = 0.0;

        // Hitung intra-community weight dan total community weight
        for &(neighbor, weight) in &graph.adjacency[node] {
            node_weight += weight;
            if communities[neighbor] == community {
                intra_community_weight += weight;
            }
        }

        // Hitung total weight untuk community
        for i in 0..communities.len() {
            if communities[i] == community {
                for &(neighbor, weight) in &graph.adjacency[i] {
                    total_community_weight += weight;
                }
            }
        }

        let total_weight = graph.total_weight();
        if total_weight == 0.0 {
            return 0.0;
        }

        // Modularity formula: ΔQ = [(2*l_in - k_i*sum_tot)/2m] - resolution * [(k_i*sum_tot)/(2m)]²
        let modularity_gain = (intra_community_weight - node_weight * total_community_weight / total_weight) / total_weight;
        modularity_gain * resolution
    }

    /// Spectral clustering menggunakan eigenvalue decomposition
    fn spectral_clustering(&self, graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature], n_clusters: usize) -> Result<Vec<ResonanceGroup>, ERPError> {
        let n_nodes = graph.adjacency.len();
        if n_nodes <= n_clusters {
            return self.each_node_as_group(n_nodes, signatures);
        }

        // Build Laplacian matrix
        let laplacian = self.build_laplacian_matrix(graph);
        
        // Simplified spectral clustering - dalam implementasi nyata gunakan proper eigenvalue decomposition
        let mut assignments = vec![0; n_nodes];
        for i in 0..n_nodes {
            assignments[i] = i % n_clusters;
        }

        self.communities_to_groups(assignments, signatures)
    }

    /// Build normalized Laplacian matrix
    fn build_laplacian_matrix(&self, graph: &ResonanceGraph) -> Array2<f32> {
        let n = graph.adjacency.len();
        let mut laplacian = Array2::zeros((n, n));

        // Build degree matrix dan adjacency matrix
        for i in 0..n {
            let mut degree = 0.0;
            for &(j, weight) in &graph.adjacency[i] {
                laplacian[[i, j]] = -weight;
                degree += weight;
            }
            if degree > 0.0 {
                laplacian[[i, i]] = 1.0; // Normalized Laplacian
            }
        }

        laplacian
    }

    /// Adaptive modular clustering dengan size constraints
    fn adaptive_modular_clustering(&self, graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature], min_size: usize, max_size: usize) -> Result<Vec<ResonanceGroup>, ERPError> {
        let n_nodes = graph.adjacency.len();
        let mut visited = vec![false; n_nodes];
        let mut groups = Vec::new();

        for i in 0..n_nodes {
            if !visited[i] {
                let mut group = self.expand_group(i, graph, signatures, &mut visited, max_size);
                
                // Handle groups yang terlalu kecil
                if group.neurons.len() < min_size {
                    // Try to merge dengan group terdekat
                    self.merge_small_group(&mut group, &mut groups, signatures, min_size, max_size);
                }

                if group.neurons.len() >= min_size {
                    groups.push(group);
                }
            }
        }

        Ok(groups)
    }

    /// Expand group dari seed node
    fn expand_group(&self, seed: usize, graph: &ResonanceGraph, signatures: &[crate::erp::core::NeuronSignature], visited: &mut [bool], max_size: usize) -> ResonanceGroup {
        let mut group_neurons = vec![seed];
        visited[seed] = true;
        let mut frontier = vec![seed];

        while !frontier.is_empty() && group_neurons.len() < max_size {
            let current = frontier.pop().unwrap_or(seed);

            for &(neighbor, _) in &graph.adjacency[current] {
                if !visited[neighbor] && group_neurons.len() < max_size {
                    // Check stability constraints
                    if self.check_group_stability(&group_neurons, neighbor, signatures) {
                        group_neurons.push(neighbor);
                        visited[neighbor] = true;
                        frontier.push(neighbor);
                    }
                }
            }
        }

        let importance_scores = group_neurons.iter()
            .map(|&idx| signatures[idx].fisher_info + signatures[idx].gradient_norm)
            .collect();

        ResonanceGroup {
            neurons: group_neurons.clone(),
            stability_variance: self.compute_group_variance(&group_neurons, signatures),
            importance_scores,
        }
    }

    /// Check stability constraints untuk group expansion
    fn check_group_stability(&self, group: &[usize], new_neuron: usize, signatures: &[crate::erp::core::NeuronSignature]) -> bool {
        if group.len() >= self.config.max_group_size {
            return false;
        }

        let mut extended_group = group.to_vec();
        extended_group.push(new_neuron);
        let variance = self.compute_group_variance(&extended_group, signatures);
        
        variance < self.config.stability_variance
    }

    /// Compute variance untuk group
    fn compute_group_variance(&self, group: &[usize], signatures: &[crate::erp::core::NeuronSignature]) -> f32 {
        if group.len() <= 1 {
            return 0.0;
        }

        let mut mean_projection = Array1::zeros(signatures[group[0]].projection.len());
        for &idx in group {
            mean_projection += &signatures[idx].projection;
        }
        mean_projection /= group.len() as f32;

        let mut variance = 0.0;
        for &idx in group {
            let diff = &signatures[idx].projection - &mean_projection;
            variance += diff.mapv(|x| x * x).sum();
        }

        variance / group.len() as f32
    }

    /// Merge small group dengan nearest group
    fn merge_small_group(&self, small_group: &mut ResonanceGroup, groups: &mut Vec<ResonanceGroup>, signatures: &[crate::erp::core::NeuronSignature], min_size: usize, max_size: usize) {
        if groups.is_empty() {
            return;
        }

        // Find nearest group
        let mut nearest_group_idx = 0;
        let mut min_distance = f32::INFINITY;

        for (i, group) in groups.iter().enumerate() {
            if group.neurons.len() + small_group.neurons.len() <= max_size {
                let distance = self.compute_group_distance(small_group, group, signatures);
                if distance < min_distance {
                    min_distance = distance;
                    nearest_group_idx = i;
                }
            }
        }

        // Merge jika memungkinkan
        if min_distance < f32::INFINITY {
            let nearest_group = &mut groups[nearest_group_idx];
            nearest_group.neurons.append(&mut small_group.neurons);
            nearest_group.importance_scores.extend(&small_group.importance_scores);
            nearest_group.stability_variance = self.compute_group_variance(&nearest_group.neurons, signatures);
        }
    }

    /// Compute distance antara dua groups
    fn compute_group_distance(&self, group1: &ResonanceGroup, group2: &ResonanceGroup, signatures: &[crate::erp::core::NeuronSignature]) -> f32 {
        let mut distance = 0.0;
        let mut count = 0;

        for &idx1 in &group1.neurons {
            for &idx2 in &group2.neurons {
                let diff = &signatures[idx1].projection - &signatures[idx2].projection;
                distance += diff.mapv(|x| x * x).sum();
                count += 1;
            }
        }

        if count > 0 {
            distance / count as f32
        } else {
            f32::INFINITY
        }
    }

    /// Convert communities ke resonance groups
    fn communities_to_groups(&self, communities: Vec<usize>, signatures: &[crate::erp::core::NeuronSignature]) -> Result<Vec<ResonanceGroup>, ERPError> {
        let mut community_map: HashMap<usize, Vec<usize>> = HashMap::new();

        for (node, community) in communities.iter().enumerate() {
            community_map.entry(*community).or_insert_with(Vec::new).push(node);
        }

        let mut groups = Vec::new();
        for (community, neurons) in community_map {
            if neurons.len() > 1 && neurons.len() <= self.config.max_group_size {
                let importance_scores = neurons.iter()
                    .map(|&idx| signatures[idx].fisher_info + signatures[idx].gradient_norm)
                    .collect();

                let group = ResonanceGroup {
                    neurons: neurons.clone(),
                    stability_variance: self.compute_group_variance(&neurons, signatures),
                    importance_scores,
                };
                groups.push(group);
            }
        }

        Ok(groups)
    }

    /// Helper: each node sebagai separate group
    fn each_node_as_group(&self, n_nodes: usize, signatures: &[crate::erp::core::NeuronSignature]) -> Result<Vec<ResonanceGroup>, ERPError> {
        let mut groups = Vec::new();

        for i in 0..n_nodes {
            let group = ResonanceGroup {
                neurons: vec![i],
                stability_variance: 0.0,
                importance_scores: vec![signatures[i].fisher_info + signatures[i].gradient_norm],
            };
            groups.push(group);
        }

        Ok(groups)
    }
}

/// Extended resonance graph dengan additional methods
impl ResonanceGraph {
    /// Get total weight dari semua edges
    pub fn total_weight(&self) -> f32 {
        let mut total = 0.0;
        for neighbors in &self.adjacency {
            for (_, weight) in neighbors {
                total += weight;
            }
        }
        total / 2.0 // Karena setiap edge dihitung dua kali
    }

    /// Get degree distribution
    pub fn degree_distribution(&self) -> Vec<f32> {
        self.adjacency.iter()
            .map(|neighbors| neighbors.iter().map(|(_, weight)| weight).sum())
            .collect()
    }
}

/// Re-export dari core module
pub use crate::erp::core::{ResonanceGraph, ResonanceGroup};
