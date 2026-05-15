pub mod oracle7_runtime;
pub mod synth_prime_runtime;
pub mod meta_reasoner_runtime;
pub mod world_model_x_runtime;
pub mod truth_arbiter_runtime;
pub mod chain_executor_runtime;

pub use oracle7_runtime::Oracle7RuntimeAgent;
pub use synth_prime_runtime::SynthPrimeRuntimeAgent;
pub use meta_reasoner_runtime::MetaReasonerRuntimeAgent;
pub use world_model_x_runtime::WorldModelRuntimeAgent;
pub use truth_arbiter_runtime::TruthArbiterRuntimeAgent;
pub use chain_executor_runtime::ChainExecutorRuntimeAgent;

use crate::shared::base_model::NxrModelResult;

#[derive(Debug, Clone, Default)]
pub struct OmnisAgents {
    oracle_7: Oracle7RuntimeAgent,
    meta_reasoner: MetaReasonerRuntimeAgent,
    world_model_x: WorldModelRuntimeAgent,
    chain_executor: ChainExecutorRuntimeAgent,
    truth_arbiter: TruthArbiterRuntimeAgent,
    synth_prime: SynthPrimeRuntimeAgent,
}

impl OmnisAgents {
    pub fn new(_config: &super::config::OmnisConfig) -> Self {
        Self::default()
    }

    pub async fn initialize(&self, _config: &super::config::OmnisConfig) -> Result<(), String> {
        Ok(())
    }

    pub async fn validate(&self) -> Result<(), String> {
        Ok(())
    }

    pub fn oracle_7(&self) -> &Oracle7RuntimeAgent {
        &self.oracle_7
    }

    pub fn meta_reasoner(&self) -> &MetaReasonerRuntimeAgent {
        &self.meta_reasoner
    }

    pub fn world_model_x(&self) -> &WorldModelRuntimeAgent {
        &self.world_model_x
    }

    pub fn chain_executor(&self) -> &ChainExecutorRuntimeAgent {
        &self.chain_executor
    }

    pub fn truth_arbiter(&self) -> &TruthArbiterRuntimeAgent {
        &self.truth_arbiter
    }

    pub fn synth_prime(&self) -> &SynthPrimeRuntimeAgent {
        &self.synth_prime
    }
}
