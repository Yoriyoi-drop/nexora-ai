use async_trait::async_trait;

use super::traits::Filter;
use crate::types::{DataSample, FilterResult, FilterAction, Domain, CurriculumLevel};

#[derive(Debug, Clone)]
pub struct CurriculumRanker {
    pub curriculum: Vec<(CurriculumLevel, Vec<Domain>, usize)>,
}

impl Default for CurriculumRanker {
    fn default() -> Self {
        Self {
            curriculum: vec![
                (CurriculumLevel::BasicGrammar, vec![Domain::Conversation, Domain::Instruction], 10_000),
                (CurriculumLevel::BasicInstruction, vec![Domain::Knowledge, Domain::Creative, Domain::General], 50_000),
                (CurriculumLevel::MediumReasoning, vec![Domain::Code, Domain::Memory, Domain::Math], 100_000),
                (CurriculumLevel::ChainOfThought, vec![Domain::Science, Domain::Architecture], 200_000),
                (CurriculumLevel::AgenticPlanning, vec![Domain::Reasoning], 500_000),
                (CurriculumLevel::MultiHopLogic, vec![Domain::Planning], 1_000_000),
            ],
        }
    }
}

impl CurriculumRanker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rank(&self, sample: &DataSample, domain: Domain) -> (CurriculumLevel, u8) {
        let domain_level = domain.curriculum_level();

        let level = self.curriculum.iter()
            .enumerate()
            .find(|(_, (_, domains, _))| domains.contains(&domain))
            .map(|(i, (level, _, _))| (level.clone(), i as u8))
            .unwrap_or_else(|| {
                let fallback = domain_level.saturating_sub(1).max(1);
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
            reason: Some(format!("curriculum_level={}, rank={}", _level as u8, _rank)),
            score_delta,
        }
    }

    fn action(&self) -> FilterAction {
        FilterAction::Accept
    }
}
