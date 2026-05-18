pub mod types;
pub mod pre_generation;
pub mod in_generation;
pub mod post_generation;
pub mod risk_scoring;
pub mod system_prompt;
pub mod monitoring;

pub use types::*;
pub use pre_generation::*;
pub use in_generation::*;
pub use post_generation::*;
pub use risk_scoring::*;
pub use system_prompt::*;
pub use monitoring::*;

#[derive(Debug, thiserror::Error)]
pub enum HallucinationError {
    #[error("Risk scoring error: {0}")]
    RiskScoring(String),
    #[error("Pre-generation check failed: {0}")]
    PreGeneration(String),
    #[error("Post-generation verification failed: {0}")]
    PostGeneration(String),
    #[error("Monitoring error: {0}")]
    Monitoring(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, HallucinationError>;

pub struct HallucinationGuard {
    pub pre_gen: PreGenerationChecker,
    pub in_gen: InGenerationGuard,
    pub post_gen: PostGenerationVerifier,
    pub risk: RiskScorer,
    pub prompts: SystemPromptManager,
    pub monitor: Monitor,
}

impl HallucinationGuard {
    pub fn new(config: GuardConfig) -> Self {
        Self {
            pre_gen: PreGenerationChecker::new(config.pre_gen_config),
            in_gen: InGenerationGuard::new(config.in_gen_config),
            post_gen: PostGenerationVerifier::new(config.post_gen_config),
            risk: RiskScorer::new(config.risk_config),
            prompts: SystemPromptManager::new(),
            monitor: Monitor::new(config.monitor_config),
        }
    }

    pub async fn run_pipeline(
        &self,
        input: &str,
        context: Option<&str>,
        sources: Option<Vec<String>>,
    ) -> Result<PipelineResult> {
        let begin = std::time::Instant::now();

        let pre_check = self.pre_gen.check(input, context)?;
        if !pre_check.can_proceed {
            self.monitor.record("pre_blocked", input);
            return Ok(PipelineResult {
                action: GuardAction::Blocked,
                risk_level: RiskLevel::Critical,
                pre_check,
                in_gen_check: None,
                post_check: None,
                score: 1.0,
                latency_ms: begin.elapsed().as_millis() as u64,
            });
        }

        let uncertainty = self.in_gen.compute_uncertainty(input, context);
        let enhanced_prompt = if uncertainty > 0.5 {
            self.prompts.wrap_with_uncertainty(input, uncertainty)
        } else {
            input.to_string()
        };

        let in_check = InGenCheckResult {
            uncertainty_score: uncertainty,
            enhanced_prompt,
            requires_cot: uncertainty > 0.3,
            knowledge_boundary: crate::system_prompt::SystemPromptManager::knowledge_boundary(),
        };

        let post_check = self.post_gen.verify(input, sources).await?;

        let score = self.risk.compute(
            &pre_check,
            &in_check,
            &post_check,
        );

        let risk_level = self.risk.classify(score);
        let action = self.risk.decide_action(risk_level.clone());

        self.monitor.record(&format!("{:?}", action), input);

        Ok(PipelineResult {
            action,
            risk_level,
            pre_check,
            in_gen_check: Some(in_check),
            post_check: Some(post_check),
            score,
            latency_ms: begin.elapsed().as_millis() as u64,
        })
    }
}

pub struct GuardConfig {
    pub pre_gen_config: PreGenConfig,
    pub in_gen_config: InGenConfig,
    pub post_gen_config: PostGenConfig,
    pub risk_config: RiskConfig,
    pub monitor_config: MonitorConfig,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            pre_gen_config: PreGenConfig::default(),
            in_gen_config: InGenConfig::default(),
            post_gen_config: PostGenConfig::default(),
            risk_config: RiskConfig::default(),
            monitor_config: MonitorConfig::default(),
        }
    }
}

#[derive(Debug)]
pub struct PipelineResult {
    pub action: GuardAction,
    pub risk_level: RiskLevel,
    pub pre_check: PreGenCheckResult,
    pub in_gen_check: Option<InGenCheckResult>,
    pub post_check: Option<PostGenCheckResult>,
    pub score: f32,
    pub latency_ms: u64,
}

#[derive(Debug, PartialEq)]
pub enum GuardAction {
    Pass,
    PassWithDisclaimer,
    FlagForReview,
    Blocked,
}
