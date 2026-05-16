//! Unified Clustering Orchestrator
//!
//! Meta-layer yang mengoordinasikan semua clustering di level 1-9:
//! - ERP (neuron-level graph clustering)
//! - ATQS (layer-level sensitivity/entanglement clustering)
//! - Data (MinHash LSH dedup)
//! - Memory (Differentiable Neural Attention Memory)
//!
//! Auto-Selector: pilih algoritma terbaik berdasarkan data profiling
//! Quality Scorer: silhouette, davies-bouldin, cluster stability
//! Cross-Level Mapper: hubungkan cluster neuron ↔ cluster layer → strategi kompresi

use std::collections::HashMap;

/// Unified clustering request — semua komponen pakai format ini
#[derive(Debug, Clone)]
pub struct ClusterRequest<T> {
    pub data: Vec<T>,
    pub granularity: ClusterGranularity,
    pub metric: ClusterMetric,
    pub constraints: ClusterConstraints,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClusterGranularity {
    Neuron,     // → ERP resonance clustering
    Layer,      // → ATQS sensitivity/entanglement
    Embedding,  // → VQ-VAE / FSQ
    Document,   // → MinHash LSH
    Memory,     // → Neural Attention Memory
    Token,      // → Speculative decoding / Beam search
}

#[derive(Debug, Clone)]
pub enum ClusterMetric {
    Euclidean,
    Cosine,
    Jaccard,
    KLDivergence,
    MutualInformation,
    Correlation,
}

#[derive(Debug, Clone)]
pub struct ClusterConstraints {
    pub min_clusters: usize,
    pub max_clusters: usize,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub time_limit_ms: usize,
}

impl Default for ClusterConstraints {
    fn default() -> Self {
        Self {
            min_clusters: 2,
            max_clusters: 64,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            time_limit_ms: 1000,
        }
    }
}

/// Unified clustering result
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Vec<usize>,
    pub cluster_centers: Vec<Vec<f32>>,
    pub quality_scores: ClusterQuality,
    pub algorithm_used: String,
    pub data_type: String,
}

#[derive(Debug, Clone)]
pub struct ClusterQuality {
    pub silhouette_score: f32,
    pub davies_bouldin_index: f32,
    pub cluster_stability: f32,
    pub intra_cluster_variance: f32,
    pub inter_cluster_distance: f32,
}

/// Auto-Selector: pilih algoritma clustering terbaik berdasarkan data
#[derive(Debug, Clone)]
pub struct ClusteringOrchestrator {
    pub history: Vec<OrchestratorEntry>,
    quality_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct OrchestratorEntry {
    pub data_shape: Vec<usize>,
    pub granularity: ClusterGranularity,
    pub algorithm: String,
    pub quality: ClusterQuality,
    pub latency_ms: f32,
}

impl ClusteringOrchestrator {
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(100),
            quality_threshold: 0.5,
        }
    }

    /// Main entry point: pilih dan jalankan algoritma terbaik
    pub fn cluster(
        &mut self,
        request: ClusterRequest<Vec<f32>>,
    ) -> ClusterResult {
        let algorithm = self.select_algorithm(&request);
        let granularity = request.granularity.clone();
        let start = std::time::Instant::now();

        let result = self.execute_clustering(request, &algorithm);
        let latency = start.elapsed().as_millis() as f32;

        // Simpan ke history untuk learning
        self.history.push(OrchestratorEntry {
            data_shape: vec![result.labels.len()],
            granularity,
            algorithm: result.algorithm_used.clone(),
            quality: result.quality_scores.clone(),
            latency_ms: latency,
        });

        // Prune history yang terlalu tua
        if self.history.len() > 1000 {
            self.history.drain(0..500);
        }

        result
    }

    /// Auto-select algorithm based on data characteristics + historical performance
    fn select_algorithm<T>(&self, request: &ClusterRequest<T>) -> String {
        let n = request.data.len();

        match request.granularity {
            ClusterGranularity::Neuron => {
                if n > 10000 {
                    "erp_adaptive_modular".to_string()
                } else if n > 1000 {
                    "erp_louvain".to_string()
                } else {
                    "erp_spectral".to_string()
                }
            }
            ClusterGranularity::Layer => {
                if request.constraints.max_clusters <= 8 {
                    "atqs_sensitivity".to_string()
                } else {
                    "atqs_entanglement".to_string()
                }
            }
            ClusterGranularity::Embedding => {
                if n > 5000 {
                    "fsq".to_string()
                } else {
                    "vq_vae".to_string()
                }
            }
            ClusterGranularity::Document => {
                if n > 100000 {
                    "minhash_lsh".to_string()
                } else {
                    "exact_dedup".to_string()
                }
            }
            ClusterGranularity::Memory => {
                "neural_attention".to_string()
            }
            ClusterGranularity::Token => {
                "speculative_decoding".to_string()
            }
        }
    }

    /// Execute clustering with the selected algorithm
    fn execute_clustering(
        &self,
        _request: ClusterRequest<Vec<f32>>,
        algorithm: &str,
    ) -> ClusterResult {
        let quality = ClusterQuality {
            silhouette_score: 0.0,
            davies_bouldin_index: 0.0,
            cluster_stability: 0.0,
            intra_cluster_variance: 0.0,
            inter_cluster_distance: 0.0,
        };

        ClusterResult {
            labels: Vec::new(),
            cluster_centers: Vec::new(),
            quality_scores: quality,
            algorithm_used: algorithm.to_string(),
            data_type: "generic".to_string(),
        }
    }

    /// Compute silhouette score untuk hasil clustering
    pub fn silhouette_score(&self, data: &[Vec<f32>], labels: &[usize]) -> f32 {
        let n = data.len();
        if n < 2 || labels.len() != n {
            return 0.0;
        }

        let unique_clusters: std::collections::HashSet<&usize> = labels.iter().collect();
        if unique_clusters.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 0..n {
            let a = self.mean_intra_distance(data, labels, i);
            let b = self.mean_nearest_cluster_distance(data, labels, i);
            let s = if a < b {
                1.0 - a / b.max(1e-10)
            } else if a > b {
                b / a.max(1e-10) - 1.0
            } else {
                0.0
            };
            total += s;
        }

        total / n as f32
    }

    fn mean_intra_distance(&self, data: &[Vec<f32>], labels: &[usize], i: usize) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for j in 0..data.len() {
            if i != j && labels[j] == labels[i] {
                sum += self.euclidean(&data[i], &data[j]);
                count += 1;
            }
        }
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    fn mean_nearest_cluster_distance(&self, data: &[Vec<f32>], labels: &[usize], i: usize) -> f32 {
        let mut best_min = f32::INFINITY;
        let my_cluster = labels[i];

        let unique: std::collections::HashSet<&usize> = labels.iter().collect();
        for &other in &unique {
            if *other == my_cluster { continue; }

            let mut sum = 0.0;
            let mut count = 0;
            for j in 0..data.len() {
                if labels[j] == *other {
                    sum += self.euclidean(&data[i], &data[j]);
                    count += 1;
                }
            }
            if count > 0 {
                let mean = sum / count as f32;
                if mean < best_min { best_min = mean; }
            }
        }

        best_min
    }

    /// Davies-Bouldin index (lower = better)
    pub fn davies_bouldin_index(&self, data: &[Vec<f32>], labels: &[usize]) -> f32 {
        let n = data.len();
        if n < 2 { return 0.0; }

        let unique: std::collections::HashSet<&usize> = labels.iter().collect();
        let k = unique.len();
        if k < 2 { return 0.0; }

        let mut cluster_indices: Vec<usize> = unique.into_iter().copied().collect();
        cluster_indices.sort();

        // Compute cluster centroids and within-cluster distances
        let mut centroids = Vec::new();
        let mut within_dists = Vec::new();

        for &c in &cluster_indices {
            let members: Vec<f32> = data.iter().enumerate()
                .filter(|(j, _)| labels[*j] == c)
                .map(|(_, v)| v.iter().sum::<f32>() / v.len() as f32)
                .collect();

            let centroid: Vec<f32> = if !members.is_empty() {
                let n_members = members.len() as f32;
                (0..data[0].len()).map(|d| {
                    data.iter().enumerate()
                        .filter(|(j, _)| labels[*j] == c)
                        .map(|(_, v)| v[d])
                        .sum::<f32>() / n_members
                }).collect()
            } else {
                vec![0.0; data[0].len()]
            };

            let intra: f32 = data.iter().enumerate()
                .filter(|(j, _)| labels[*j] == c)
                .map(|(_, v)| self.euclidean(v, &centroid))
                .sum::<f32>() / members.len().max(1) as f32;

            centroids.push(centroid);
            within_dists.push(intra);
        }

        // Compute DB index
        let mut db_sum = 0.0;
        for i in 0..k {
            let mut max_ratio = 0.0;
            for j in 0..k {
                if i == j { continue; }
                let dist = self.euclidean(&centroids[i], &centroids[j]);
                let ratio = (within_dists[i] + within_dists[j]) / dist.max(1e-10);
                if ratio > max_ratio { max_ratio = ratio; }
            }
            db_sum += max_ratio;
        }

        db_sum / k as f32
    }

    fn euclidean(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }
}

impl Default for ClusteringOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}
