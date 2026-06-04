use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{CurriculumLevel, DataSample, Domain, FilterAction, FilterResult};

#[derive(Debug, Clone)]
pub struct CurriculumRanker {
    pub curriculum: Vec<(CurriculumLevel, Vec<Domain>, usize)>,
}

impl Default for CurriculumRanker {
    fn default() -> Self {
        Self {
            curriculum: vec![
                (
                    CurriculumLevel::BasicGrammar,
                    vec![Domain::Conversation, Domain::Instruction],
                    10_000,
                ),
                (
                    CurriculumLevel::BasicInstruction,
                    vec![Domain::Knowledge, Domain::Creative, Domain::General],
                    50_000,
                ),
                (
                    CurriculumLevel::MediumReasoning,
                    vec![Domain::Code, Domain::Memory, Domain::Math],
                    100_000,
                ),
                (
                    CurriculumLevel::ChainOfThought,
                    vec![Domain::Science, Domain::Architecture],
                    200_000,
                ),
                (
                    CurriculumLevel::AgenticPlanning,
                    vec![Domain::Reasoning],
                    500_000,
                ),
                (
                    CurriculumLevel::MultiHopLogic,
                    vec![Domain::Planning],
                    1_000_000,
                ),
            ],
        }
    }
}

impl CurriculumRanker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rank(&self, _sample: &DataSample, domain: Domain) -> (CurriculumLevel, u8) {
        let domain_level = domain.curriculum_level();

        let level = self
            .curriculum
            .iter()
            .enumerate()
            .find(|(_, (_, domains, _))| domains.contains(&domain))
            .map(|(i, (level, _, _))| (*level, i as u8))
            .unwrap_or_else(|| {
                let fallback = (domain_level as u8).saturating_sub(1).max(1);
                let level = match fallback {
                    1 => CurriculumLevel::BasicGrammar,
                    2 => CurriculumLevel::BasicInstruction,
                    3 => CurriculumLevel::MediumReasoning,
                    4 => CurriculumLevel::ChainOfThought,
                    5 => CurriculumLevel::AgenticPlanning,
                    _ => CurriculumLevel::MultiHopLogic,
                };
                (level, fallback.saturating_sub(1))
            });

        (level.0, level.1 + 1)
    }
}

#[async_trait]
impl Filter for CurriculumRanker {
    fn name(&self) -> &str {
        "curriculum_ranker"
    }

    async fn evaluate(&self, sample: &DataSample) -> FilterResult {
        let primary_domain = sample.domains.first().cloned().unwrap_or(Domain::General);
        let (_level, _rank) = self.rank(sample, primary_domain);

        let score_delta = match _rank {
            1 => -0.2,
            2 => -0.1,
            3 => 0.0,
            4 => 0.1,
            5 => 0.2,
            _ => 0.3,
        };

        FilterResult {
            passed: true,
            sample_id: sample.id,
            filter_name: self.name().to_string(),
            reason: Some(format!("curriculum_level={:?}, rank={}", _level, _rank)),
            score_delta,
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceCategory, SourceInfo};
    use uuid::Uuid;

    fn sample(text: &str, domain: Domain) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: text.into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![domain],
            score: None,
            curriculum_level: None,
        }
    }

    #[test]
    fn test_default_curriculum() {
        let c = CurriculumRanker::default();
        assert_eq!(c.curriculum.len(), 6);
    }

    #[test]
    fn test_rank_conversation() {
        let c = CurriculumRanker::default();
        let s = sample("hello", Domain::Conversation);
        let (level, rank) = c.rank(&s, Domain::Conversation);
        assert_eq!(level, CurriculumLevel::BasicGrammar);
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_rank_planning() {
        let c = CurriculumRanker::default();
        let s = sample("plan", Domain::Planning);
        let (level, rank) = c.rank(&s, Domain::Planning);
        assert_eq!(level, CurriculumLevel::MultiHopLogic);
        assert_eq!(rank, 6);
    }

    #[tokio::test]
    async fn test_evaluate_assigns_level() {
        let c = CurriculumRanker::default();
        let s = sample("code", Domain::Code);
        let result = c.evaluate(&s).await;
        assert!(result.passed);
        assert!(result.reason.unwrap().contains("curriculum_level"));
    }

    #[tokio::test]
    async fn test_evaluate_general_domain() {
        let c = CurriculumRanker::default();
        let s = sample("general content here", Domain::General);
        let result = c.evaluate(&s).await;
        assert!(result.passed);
    }

    #[test]
    fn test_action_is_accept() {
        let c = CurriculumRanker::default();
        assert_eq!(c.action(), FilterAction::Accept);
    }
}
