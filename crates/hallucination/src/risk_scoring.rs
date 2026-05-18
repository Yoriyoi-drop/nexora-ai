use crate::types::{
    PreGenCheckResult, InGenCheckResult, PostGenCheckResult,
    RiskLevel, RiskScore,
};
use crate::GuardAction;

#[derive(Debug, Clone)]
pub struct RiskConfig {
    pub specificity_weight: f32,
    pub domain_niche_weight: f32,
    pub citation_weight: f32,
    pub contradiction_weight: f32,
    pub recency_weight: f32,
    pub confidence_threshold: f32,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            specificity_weight: 0.25,
            domain_niche_weight: 0.15,
            citation_weight: 0.25,
            contradiction_weight: 0.20,
            recency_weight: 0.15,
            confidence_threshold: 0.7,
        }
    }
}

pub struct RiskScorer {
    config: RiskConfig,
}

impl RiskScorer {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    pub fn compute(
        &self,
        pre: &PreGenCheckResult,
        _in_gen: &InGenCheckResult,
        post: &PostGenCheckResult,
    ) -> f32 {
        let specificity_score = self.compute_specificity(post);
        let domain_niche_score = self.compute_domain_niche(pre);
        let citation_score = self.compute_citation(post);
        let contradiction_score = self.compute_contradiction(post);
        let recency_score = self.compute_recency(pre);

        let total = specificity_score * self.config.specificity_weight
            + domain_niche_score * self.config.domain_niche_weight
            + citation_score * self.config.citation_weight
            + contradiction_score * self.config.contradiction_weight
            + recency_score * self.config.recency_weight;

        total.min(1.0).max(0.0)
    }

    pub fn classify(&self, score: f32) -> RiskLevel {
        if score >= 0.8 {
            RiskLevel::Critical
        } else if score >= 0.5 {
            RiskLevel::High
        } else if score >= 0.25 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    pub fn decide_action(&self, level: RiskLevel) -> GuardAction {
        match level {
            RiskLevel::Low => GuardAction::Pass,
            RiskLevel::Medium => GuardAction::PassWithDisclaimer,
            RiskLevel::High => GuardAction::FlagForReview,
            RiskLevel::Critical => GuardAction::Blocked,
        }
    }

    pub fn breakdown(&self, pre: &PreGenCheckResult, _in: &InGenCheckResult, post: &PostGenCheckResult) -> RiskScore {
        RiskScore {
            total: self.compute(pre, _in, post),
            specificity_score: self.compute_specificity(post),
            domain_niche_score: self.compute_domain_niche(pre),
            citation_score: self.compute_citation(post),
            contradiction_score: self.compute_contradiction(post),
            recency_score: self.compute_recency(pre),
            confidence: self.config.confidence_threshold,
        }
    }

    fn compute_specificity(&self, post: &PostGenCheckResult) -> f32 {
        if post.total_claims == 0 {
            return 0.0;
        }
        let high_risk_ratio = post.high_risk_sentences.len() as f32 / post.total_claims as f32;
        (high_risk_ratio * 0.8).min(1.0)
    }

    fn compute_domain_niche(&self, _pre: &PreGenCheckResult) -> f32 {
        0.2
    }

    fn compute_citation(&self, post: &PostGenCheckResult) -> f32 {
        if post.total_claims == 0 {
            return 0.0;
        }
        let unverified = post.total_claims.saturating_sub(post.verified_claims);
        let ratio = unverified as f32 / post.total_claims as f32;
        (ratio * 0.7).min(1.0)
    }

    fn compute_contradiction(&self, post: &PostGenCheckResult) -> f32 {
        if post.total_claims == 0 {
            return 0.0;
        }
        let ratio = post.contradiction_count as f32 / post.total_claims.max(1) as f32;
        (ratio * 1.0).min(1.0)
    }

    fn compute_recency(&self, _pre: &PreGenCheckResult) -> f32 {
        0.1
    }
}
