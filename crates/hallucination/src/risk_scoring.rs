use crate::types::{InGenCheckResult, PostGenCheckResult, PreGenCheckResult, RiskLevel, RiskScore};
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

    pub fn breakdown(
        &self,
        pre: &PreGenCheckResult,
        _in: &InGenCheckResult,
        post: &PostGenCheckResult,
    ) -> RiskScore {
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

    fn compute_domain_niche(&self, pre: &PreGenCheckResult) -> f32 {
        // Domain niche risk: out-of-scope + high ambiguity + low context sufficiency
        let scope_risk = if !pre.in_scope { 0.4 } else { 0.0 };
        let ambiguity_risk = pre.ambiguity_score * 0.3;
        let context_risk = (1.0 - pre.context_sufficiency) * 0.3;
        (scope_risk + ambiguity_risk + context_risk).min(1.0)
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

    fn compute_recency(&self, pre: &PreGenCheckResult) -> f32 {
        // Recency risk: low context sufficiency suggests stale knowledge,
        // high ambiguity suggests the topic lacks recent clear references
        let ambiguity_risk = pre.ambiguity_score * 0.3;
        let context_risk = (1.0 - pre.context_sufficiency) * 0.7;
        (ambiguity_risk + context_risk).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InGenCheckResult, PostGenCheckResult, PreGenCheckResult};

    fn default_pre() -> PreGenCheckResult {
        PreGenCheckResult {
            can_proceed: true,
            in_scope: true,
            ambiguity_score: 0.0,
            context_sufficiency: 1.0,
            reason: "OK".into(),
            suggestions: vec![],
        }
    }

    fn default_in_gen() -> InGenCheckResult {
        InGenCheckResult {
            uncertainty_score: 0.0,
            enhanced_prompt: String::new(),
            requires_cot: false,
            knowledge_boundary: String::new(),
        }
    }

    fn default_post() -> PostGenCheckResult {
        PostGenCheckResult {
            internal_consistency: 1.0,
            source_grounding: 1.0,
            high_risk_sentences: vec![],
            contradiction_count: 0,
            total_claims: 0,
            verified_claims: 0,
        }
    }

    fn scorer() -> RiskScorer {
        RiskScorer::new(RiskConfig::default())
    }

    #[test]
    fn test_classify_low() {
        let s = scorer();
        assert_eq!(s.classify(0.0), RiskLevel::Low);
        assert_eq!(s.classify(0.24), RiskLevel::Low);
    }

    #[test]
    fn test_classify_medium() {
        let s = scorer();
        assert_eq!(s.classify(0.25), RiskLevel::Medium);
        assert_eq!(s.classify(0.49), RiskLevel::Medium);
    }

    #[test]
    fn test_classify_high() {
        let s = scorer();
        assert_eq!(s.classify(0.5), RiskLevel::High);
        assert_eq!(s.classify(0.79), RiskLevel::High);
    }

    #[test]
    fn test_classify_critical() {
        let s = scorer();
        assert_eq!(s.classify(0.8), RiskLevel::Critical);
        assert_eq!(s.classify(1.0), RiskLevel::Critical);
    }

    #[test]
    fn test_decide_action_low() {
        let s = scorer();
        assert_eq!(s.decide_action(RiskLevel::Low), GuardAction::Pass);
    }

    #[test]
    fn test_decide_action_medium() {
        let s = scorer();
        assert_eq!(
            s.decide_action(RiskLevel::Medium),
            GuardAction::PassWithDisclaimer
        );
    }

    #[test]
    fn test_decide_action_high() {
        let s = scorer();
        assert_eq!(s.decide_action(RiskLevel::High), GuardAction::FlagForReview);
    }

    #[test]
    fn test_decide_action_critical() {
        let s = scorer();
        assert_eq!(s.decide_action(RiskLevel::Critical), GuardAction::Blocked);
    }

    #[test]
    fn test_compute_no_claims_zero_score() {
        let s = scorer();
        let score = s.compute(&default_pre(), &default_in_gen(), &default_post());
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_compute_high_risk_claims() {
        let s = scorer();
        let pre = default_pre();
        let post = PostGenCheckResult {
            total_claims: 10,
            high_risk_sentences: vec!["risky".into(); 5],
            ..default_post()
        };
        let score = s.compute(&pre, &default_in_gen(), &post);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_compute_contradiction_increases_score() {
        let s = scorer();
        let pre = default_pre();
        let post_no_contra = PostGenCheckResult {
            total_claims: 10,
            contradiction_count: 0,
            ..default_post()
        };
        let post_contra = PostGenCheckResult {
            total_claims: 10,
            contradiction_count: 5,
            ..default_post()
        };
        let score_no = s.compute(&pre, &default_in_gen(), &post_no_contra);
        let score_yes = s.compute(&pre, &default_in_gen(), &post_contra);
        assert!(score_yes > score_no);
    }

    #[test]
    fn test_compute_out_of_scope_increases_score() {
        let s = scorer();
        let pre_out = PreGenCheckResult {
            in_scope: false,
            ..default_pre()
        };
        let pre_in = default_pre();
        let score_out = s.compute(&pre_out, &default_in_gen(), &default_post());
        let score_in = s.compute(&pre_in, &default_in_gen(), &default_post());
        assert!(score_out > score_in);
    }

    #[test]
    fn test_compute_score_clamped() {
        let s = scorer();
        let pre = PreGenCheckResult {
            in_scope: false,
            ambiguity_score: 1.0,
            context_sufficiency: 0.0,
            ..default_pre()
        };
        let post = PostGenCheckResult {
            total_claims: 100,
            high_risk_sentences: vec!["x".into(); 100],
            contradiction_count: 100,
            verified_claims: 0,
            ..default_post()
        };
        let score = s.compute(&pre, &default_in_gen(), &post);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_breakdown_returns_all_components() {
        let s = scorer();
        let breakdown = s.breakdown(&default_pre(), &default_in_gen(), &default_post());
        assert_eq!(breakdown.total, 0.0);
        assert_eq!(breakdown.specificity_score, 0.0);
        assert_eq!(breakdown.domain_niche_score, 0.0);
        assert_eq!(breakdown.contradiction_score, 0.0);
    }
}
