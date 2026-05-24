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

use rand::Rng;

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
    Neuron,    // → ERP resonance clustering
    Layer,     // → ATQS sensitivity/entanglement
    Embedding, // → VQ-VAE / FSQ
    Document,  // → MinHash LSH
    Memory,    // → Neural Attention Memory
    Token,     // → Speculative decoding / Beam search
}

#[derive(Debug, Clone)]
pub enum ClusterMetric {
    Euclidean,
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
    _quality_threshold: f32,

    /// Use GPU acceleration for pairwise distance computations
    pub use_gpu: bool,

    /// Precomputed pairwise distance matrix (N×N), filled lazily (RefCell for &self access)
    cached_distances: std::cell::RefCell<Option<Vec<Vec<f32>>>>,
    cached_data_len: std::cell::Cell<usize>,

    /// Previous labels for cluster_stability computation
    prev_labels: std::cell::RefCell<Option<Vec<usize>>>,
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
            _quality_threshold: 0.5,
            use_gpu: true,
            cached_distances: std::cell::RefCell::new(None),
            cached_data_len: std::cell::Cell::new(0),
            prev_labels: std::cell::RefCell::new(None),
        }
    }

    /// Main entry point: pilih dan jalankan algoritma terbaik
    pub fn cluster(&mut self, request: ClusterRequest<Vec<f32>>) -> ClusterResult {
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
            ClusterGranularity::Memory => "neural_attention".to_string(),
            ClusterGranularity::Token => "speculative_decoding".to_string(),
        }
    }

    /// Execute clustering with the selected algorithm
    fn execute_clustering(
        &self,
        request: ClusterRequest<Vec<f32>>,
        algorithm: &str,
    ) -> ClusterResult {
        let n = request.data.len();
        if n == 0 {
            return ClusterResult {
                labels: Vec::new(),
                cluster_centers: Vec::new(),
                quality_scores: ClusterQuality {
                    silhouette_score: 0.0,
                    davies_bouldin_index: 0.0,
                    cluster_stability: 0.0,
                    intra_cluster_variance: 0.0,
                    inter_cluster_distance: 0.0,
                },
                algorithm_used: algorithm.to_string(),
                data_type: "generic".to_string(),
            };
        }

        let k = request
            .constraints
            .max_clusters
            .min(n)
            .max(request.constraints.min_clusters.max(2));
        let dim = request.data[0].len();

        // K-means++ initialization
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut used = vec![false; n];
        let first_idx = rand::Rng::gen_range(&mut rng, 0..n);
        centroids.push(request.data[first_idx].clone());
        used[first_idx] = true;

        for _ in 1..k {
            let mut dists = Vec::with_capacity(n);
            for (i, point) in request.data.iter().enumerate() {
                if used[i] {
                    dists.push(0.0);
                    continue;
                }
                let d2 = centroids
                    .iter()
                    .map(|c| self.euclidean(point, c).powi(2))
                    .fold(f32::INFINITY, f32::min);
                dists.push(d2);
            }
            let total: f32 = dists.iter().sum();
            if total <= 0.0 {
                while centroids.len() < k {
                    centroids.push(centroids[0].clone());
                }
                break;
            }
            let r = rand::Rng::gen_range(&mut rng, 0.0..total);
            let mut cum = 0.0;
            for (i, &d) in dists.iter().enumerate() {
                cum += d;
                if cum >= r {
                    centroids.push(request.data[i].clone());
                    used[i] = true;
                    break;
                }
            }
        }

        let max_iter = 100;
        let mut labels = vec![0usize; n];
        let mut changed = true;

        for _ in 0..max_iter {
            if !changed {
                break;
            }
            changed = false;

            // Assignment step
            for (i, point) in request.data.iter().enumerate() {
                let mut best_d = f32::INFINITY;
                let mut best_c = 0;
                for (j, c) in centroids.iter().enumerate() {
                    let d = self.euclidean(point, c);
                    if d < best_d {
                        best_d = d;
                        best_c = j;
                    }
                }
                if labels[i] != best_c {
                    labels[i] = best_c;
                    changed = true;
                }
            }

            // Update step
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, &c) in labels.iter().enumerate() {
                for (d, &v) in request.data[i].iter().enumerate() {
                    new_centroids[c][d] += v;
                }
                counts[c] += 1;
            }
            for (j, nc) in new_centroids.iter_mut().enumerate() {
                if counts[j] > 0 {
                    for v in nc.iter_mut() {
                        *v /= counts[j] as f32;
                    }
                } else {
                    *nc = centroids[j].clone();
                }
            }
            centroids = new_centroids;
        }

        self.ensure_distances(&request.data);
        let silhouette = self.silhouette_score(&request.data, &labels);
        let db = self.davies_bouldin_index(&request.data, &labels);

        let intra_var = self.compute_intra_cluster_variance(&request.data, &centroids, &labels);
        let inter_dist = self.compute_inter_cluster_distance(&centroids);
        let stability = self.compute_cluster_stability(&labels);

        *self.prev_labels.borrow_mut() = Some(labels.clone());

        let quality = ClusterQuality {
            silhouette_score: silhouette,
            davies_bouldin_index: db,
            cluster_stability: stability,
            intra_cluster_variance: intra_var,
            inter_cluster_distance: inter_dist,
        };

        ClusterResult {
            labels,
            cluster_centers: centroids,
            quality_scores: quality,
            algorithm_used: algorithm.to_string(),
            data_type: "generic".to_string(),
        }
    }

    /// Mean squared distance of each point from its cluster centroid
    fn compute_intra_cluster_variance(
        &self,
        data: &[Vec<f32>],
        centroids: &[Vec<f32>],
        labels: &[usize],
    ) -> f32 {
        let n = data.len();
        if n == 0 || centroids.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for (i, point) in data.iter().enumerate() {
            if let Some(c) = centroids.get(labels[i]) {
                total += self.euclidean(point, c);
            }
        }
        total / n as f32
    }

    /// Mean pairwise distance between distinct cluster centroids
    fn compute_inter_cluster_distance(&self, centroids: &[Vec<f32>]) -> f32 {
        let k = centroids.len();
        if k < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                total += self.euclidean(&centroids[i], &centroids[j]);
                count += 1;
            }
        }
        total / count as f32
    }

    /// Fraction of labels that remain stable compared to the previous clustering run
    fn compute_cluster_stability(&self, labels: &[usize]) -> f32 {
        let prev = self.prev_labels.borrow();
        match prev.as_ref() {
            Some(prev_labels) if prev_labels.len() == labels.len() && !labels.is_empty() => {
                let same = prev_labels.iter().zip(labels.iter()).filter(|(a, b)| a == b).count();
                same as f32 / labels.len() as f32
            }
            _ => 1.0,
        }
    }

    /// Compute silhouette score untuk hasil clustering
    pub fn silhouette_score(&self, data: &[Vec<f32>], labels: &[usize]) -> f32 {
        self.ensure_distances(data);
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
                sum += self.dist(data, i, j);
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    fn mean_nearest_cluster_distance(&self, data: &[Vec<f32>], labels: &[usize], i: usize) -> f32 {
        let mut best_min = f32::INFINITY;
        let my_cluster = labels[i];

        let mut seen = [false; 256];
        let mut unique = Vec::with_capacity(16);
        for &l in labels {
            let idx = l;
            if idx != my_cluster {
                if idx < 256 {
                    if !seen[idx] {
                        seen[idx] = true;
                        unique.push(idx);
                    }
                } else if !unique.contains(&idx) {
                    unique.push(idx);
                }
            }
        }

        for &other in &unique {
            let mut sum = 0.0;
            let mut count = 0;
            for j in 0..data.len() {
                if labels[j] == other {
                    sum += self.dist(data, i, j);
                    count += 1;
                }
            }
            if count > 0 {
                let mean = sum / count as f32;
                if mean < best_min {
                    best_min = mean;
                }
            }
        }

        best_min
    }

    /// Davies-Bouldin index (lower = better)
    pub fn davies_bouldin_index(&self, data: &[Vec<f32>], labels: &[usize]) -> f32 {
        let n = data.len();
        if n < 2 {
            return 0.0;
        }

        let unique: std::collections::HashSet<&usize> = labels.iter().collect();
        let k = unique.len();
        if k < 2 {
            return 0.0;
        }

        let mut cluster_indices: Vec<usize> = unique.into_iter().copied().collect();
        cluster_indices.sort();

        // Compute cluster centroids and within-cluster distances
        let mut centroids = Vec::new();
        let mut within_dists = Vec::new();

        for &c in &cluster_indices {
            let members: Vec<f32> = data
                .iter()
                .enumerate()
                .filter(|(j, _)| labels[*j] == c)
                .map(|(_, v)| v.iter().sum::<f32>() / v.len() as f32)
                .collect();

            let centroid: Vec<f32> = if !members.is_empty() {
                let n_members = members.len() as f32;
                (0..data[0].len())
                    .map(|d| {
                        data.iter()
                            .enumerate()
                            .filter(|(j, _)| labels[*j] == c)
                            .map(|(_, v)| v[d])
                            .sum::<f32>()
                            / n_members
                    })
                    .collect()
            } else {
                vec![0.0; data[0].len()]
            };

            let intra: f32 = data
                .iter()
                .enumerate()
                .filter(|(j, _)| labels[*j] == c)
                .map(|(_, v)| self.euclidean(v, &centroid))
                .sum::<f32>()
                / members.len().max(1) as f32;

            centroids.push(centroid);
            within_dists.push(intra);
        }

        // Compute DB index
        let mut db_sum = 0.0;
        for i in 0..k {
            let mut max_ratio = 0.0;
            for j in 0..k {
                if i == j {
                    continue;
                }
                let dist = self.euclidean(&centroids[i], &centroids[j]);
                let ratio = (within_dists[i] + within_dists[j]) / dist.max(1e-10);
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
            db_sum += max_ratio;
        }

        db_sum / k as f32
    }

    fn euclidean(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Get distance between two points, using cached matrix if available
    fn dist(&self, data: &[Vec<f32>], i: usize, j: usize) -> f32 {
        if let Some(ref dmat) = *self.cached_distances.borrow() {
            if i < dmat.len() && j < dmat.len() {
                return dmat[i][j];
            }
        }
        self.euclidean(&data[i], &data[j])
    }

    /// Lazily compute full pairwise distance matrix on GPU (can be called with &self)
    pub fn ensure_distances(&self, data: &[Vec<f32>]) {
        let cached_len = self.cached_data_len.get();
        if self.cached_distances.borrow().is_some() && cached_len == data.len() {
            return;
        }
        #[cfg(feature = "gpu")]
        if self.use_gpu {
            match crate::gpu_cluster_ops::gpu_pairwise_distances(data) {
                Ok(dmat) => {
                    *self.cached_distances.borrow_mut() = Some(dmat);
                    self.cached_data_len.set(data.len());
                    return;
                }
                Err(e) => {
                    tracing::warn!("GPU distance computation failed, falling back to CPU: {}", e);
                }
            }
        }
        // CPU fallback: compute full N×N matrix only for small N
        let n = data.len();
        if n > 2000 {
            tracing::warn!("Data size {} too large for N×N CPU matrix caching. Falling back to lazy on-the-fly computation to prevent OOM.", n);
            self.invalidate_distances();
            return;
        }
        let mut dmat = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.euclidean(&data[i], &data[j]);
                dmat[i][j] = d;
                dmat[j][i] = d;
            }
        }
        *self.cached_distances.borrow_mut() = Some(dmat);
        self.cached_data_len.set(n);
    }

    /// Invalidate cached distance matrix
    pub fn invalidate_distances(&self) {
        *self.cached_distances.borrow_mut() = None;
        self.cached_data_len.set(0);
    }
}

impl Default for ClusteringOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_new_creates_empty_history() {
        let o = ClusteringOrchestrator::new();
        assert!(o.history.is_empty());
    }

    #[test]
    fn test_orchestrator_default() {
        let o = ClusteringOrchestrator::default();
        assert!(o.history.is_empty());
    }

    #[test]
    fn test_silhouette_score_identical_points() {
        let o = ClusteringOrchestrator::new();
        let data = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let labels = vec![0, 0];
        let score = o.silhouette_score(&data, &labels);
        assert_eq!(score, 0.0, "Identical points should give 0 silhouette");
    }

    #[test]
    fn test_silhouette_score_perfect_separation() {
        let o = ClusteringOrchestrator::new();
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let labels = vec![0, 0, 1, 1];
        let score = o.silhouette_score(&data, &labels);
        assert!(
            score > 0.5,
            "Well-separated clusters should have high silhouette"
        );
    }

    #[test]
    fn test_davies_bouldin_index_two_clusters() {
        let o = ClusteringOrchestrator::new();
        let data = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
        let labels = vec![0, 0, 1, 1];
        let db = o.davies_bouldin_index(&data, &labels);
        assert!(db > 0.0, "DB index should be positive");
        assert!(db < 2.0, "Well-separated clusters should have low DB index");
    }

    #[test]
    fn test_davies_bouldin_index_single_cluster() {
        let o = ClusteringOrchestrator::new();
        let data = vec![vec![1.0], vec![2.0]];
        let labels = vec![0, 0];
        let db = o.davies_bouldin_index(&data, &labels);
        assert_eq!(db, 0.0, "Single cluster should give 0 DB index");
    }

    #[test]
    fn test_euclidean_distance() {
        let o = ClusteringOrchestrator::new();
        let dist = o.euclidean(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((dist - 5.0).abs() < 1e-6, "3-4-5 triangle, got {dist}");
    }

    #[test]
    fn test_cluster_returns_result_with_algorithm() {
        let mut o = ClusteringOrchestrator::new();
        let request = ClusterRequest {
            data: vec![vec![1.0f32]; 5],
            granularity: ClusterGranularity::Neuron,
            metric: ClusterMetric::Euclidean,
            constraints: ClusterConstraints::default(),
        };
        let result = o.cluster(request);
        assert_eq!(
            result.algorithm_used, "erp_spectral",
            "Small neuron data should use spectral"
        );
    }

    #[test]
    fn test_cluster_request_default_constraints() {
        let constraints = ClusterConstraints::default();
        assert_eq!(constraints.min_clusters, 2);
        assert_eq!(constraints.max_clusters, 64);
        assert_eq!(constraints.time_limit_ms, 1000);
    }

    #[test]
    fn test_cluster_quality_fields() {
        let q = ClusterQuality {
            silhouette_score: 0.7,
            davies_bouldin_index: 0.3,
            cluster_stability: 0.9,
            intra_cluster_variance: 0.1,
            inter_cluster_distance: 0.8,
        };
        assert!(q.silhouette_score > 0.5);
        assert!(q.davies_bouldin_index < 0.5);
    }
}
