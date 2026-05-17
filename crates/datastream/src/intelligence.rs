use std::sync::Arc;
use parking_lot::RwLock;

use crate::types::{DataSample, Domain, CurriculumLevel, TrustScoreMap};

pub struct DatasetIntelligenceCore {
    pub trust_scores: TrustScoreMap,
    pub domain_statistics: Arc<RwLock<std::collections::HashMap<Domain, u64>>>,
    pub curriculum_progress: Arc<RwLock<std::collections::HashMap<CurriculumLevel, u64>>>,
    pub source_reputation: Arc<RwLock<std::collections::HashMap<String, f64>>>,
    pub quality_distribution: Arc<RwLock<Vec<f64>>>,
    pub samples_processed: Arc<RwLock<u64>>,
}

impl Default for DatasetIntelligenceCore {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetIntelligenceCore {
    pub fn new() -> Self {
        Self {
            trust_scores: TrustScoreMap::default(),
            domain_statistics: Arc::new(RwLock::new(std::collections::HashMap::new())),
            curriculum_progress: Arc::new(RwLock::new(std::collections::HashMap::new())),
            source_reputation: Arc::new(RwLock::new(std::collections::HashMap::new())),
            quality_distribution: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            samples_processed: Arc::new(RwLock::new(0)),
        }
    }

    pub fn score_sample(&self, sample: &DataSample) -> f64 {
        let mut score = 0.5;

        let trust = self.trust_scores.0.get(&sample.source.name.to_lowercase())
            .copied()
            .unwrap_or(0.5);
        score += trust * 0.2;

        let length_factor = (sample.text.len() as f64 / 1000.0).min(1.0) * 0.1;
        score += length_factor;

        if let Some(stats) = Some(&sample.stats) {
            if stats.entropy > 3.0 && stats.entropy < 7.0 {
                score += 0.1;
            }
            let quality = stats.quality_score;
            score += quality * 0.2;
        }

        let primary_domain = sample.domains.first();
        if let Some(domain) = primary_domain {
            let domain_score = match domain {
                Domain::Science | Domain::Math => 0.15,
                Domain::Code | Domain::Reasoning => 0.1,
                Domain::Knowledge | Domain::Architecture => 0.05,
                _ => 0.0,
            };
            score += domain_score;
        }

        score.clamp(0.0, 1.0)
    }

    pub fn route_by_domain(&self, sample: &mut DataSample, domain: Domain) {
        if !sample.domains.contains(&domain) {
            sample.domains.push(domain);
        }

        let mut stats = self.domain_statistics.write();
        *stats.entry(domain).or_insert(0) += 1;
    }

    pub fn curriculum_level(&self, domain: Domain, progress: f64) -> CurriculumLevel {
        let domain_level = domain.curriculum_level() as f64;
        let adjusted = (domain_level + progress * 3.0).round().max(1.0).min(6.0) as u8;
        match adjusted {
            1 => CurriculumLevel::BasicGrammar,
            2 => CurriculumLevel::BasicInstruction,
            3 => CurriculumLevel::MediumReasoning,
            4 => CurriculumLevel::ChainOfThought,
            5 => CurriculumLevel::AgenticPlanning,
            _ => CurriculumLevel::MultiHopLogic,
        }
    }

    pub fn update_quality_distribution(&self, quality: f64) {
        let mut dist = self.quality_distribution.write();
        dist.push(quality);
        if dist.len() > 100_000 {
            dist.drain(0..50_000);
        }
    }

    pub fn quality_percentile(&self, percentile: f64) -> f64 {
        let dist = self.quality_distribution.read();
        if dist.is_empty() {
            return 0.5;
        }
        let mut sorted = dist.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * sorted.len() as f64) as usize;
        sorted.get(idx.min(sorted.len().saturating_sub(1))).copied().unwrap_or(0.5)
    }

    pub fn data_fingerprint(&self, sample: &DataSample) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        sample.source.name.hash(&mut hasher);
        sample.text.len().hash(&mut hasher);
        sample.text.chars().take(100).for_each(|c| c.hash(&mut hasher));
        format!("{:x}", hasher.finish())
    }

    pub fn source_reputation(&self, source_name: &str) -> f64 {
        let rep = self.source_reputation.read();
        rep.get(source_name).copied().unwrap_or(0.5)
    }

    pub fn update_source_reputation(&self, source_name: &str, delta: f64) {
        let mut rep = self.source_reputation.write();
        let current = rep.entry(source_name.to_string()).or_insert(0.5);
        *current = (*current + delta * 0.01).clamp(0.0, 1.0);
    }

    pub fn report(&self) -> IntelligenceReport {
        let samples = *self.samples_processed.read();
        let domains = self.domain_statistics.read().clone();
        let curriculum = self.curriculum_progress.read().clone();

        let avg_quality = {
            let dist = self.quality_distribution.read();
            if dist.is_empty() { 0.0 } else { dist.iter().sum::<f64>() / dist.len() as f64 }
        };

        IntelligenceReport {
            samples_processed: samples,
            domain_breakdown: domains,
            curriculum_progress: curriculum,
            average_quality: avg_quality,
            quality_p50: self.quality_percentile(50.0),
            quality_p95: self.quality_percentile(95.0),
        }
    }
}

pub struct IntelligenceReport {
    pub samples_processed: u64,
    pub domain_breakdown: std::collections::HashMap<Domain, u64>,
    pub curriculum_progress: std::collections::HashMap<CurriculumLevel, u64>,
    pub average_quality: f64,
    pub quality_p50: f64,
    pub quality_p95: f64,
}
