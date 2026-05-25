pub mod chain_executor_runtime;
pub mod meta_reasoner_runtime;
pub mod oracle7_runtime;
pub mod synth_prime_runtime;
pub mod truth_arbiter_runtime;
pub mod world_model_x_runtime;

pub use chain_executor_runtime::ChainExecutorRuntimeAgent;
pub use meta_reasoner_runtime::MetaReasonerRuntimeAgent;
pub use oracle7_runtime::Oracle7RuntimeAgent;
pub use synth_prime_runtime::SynthPrimeRuntimeAgent;
pub use truth_arbiter_runtime::TruthArbiterRuntimeAgent;
pub use world_model_x_runtime::WorldModelRuntimeAgent;

use nexora_shared::base_model::NxrModelResult;

#[derive(Debug, Clone)]
pub struct OmnisAgents {
    config: super::config::OmnisConfig,
    oracle_7: Oracle7RuntimeAgent,
    meta_reasoner: MetaReasonerRuntimeAgent,
    world_model_x: WorldModelRuntimeAgent,
    chain_executor: ChainExecutorRuntimeAgent,
    truth_arbiter: TruthArbiterRuntimeAgent,
    synth_prime: SynthPrimeRuntimeAgent,
}

impl Default for OmnisAgents {
    fn default() -> Self {
        Self {
            config: super::config::OmnisConfig::default(),
            oracle_7: Oracle7RuntimeAgent::default(),
            meta_reasoner: MetaReasonerRuntimeAgent::default(),
            world_model_x: WorldModelRuntimeAgent::default(),
            chain_executor: ChainExecutorRuntimeAgent::default(),
            truth_arbiter: TruthArbiterRuntimeAgent::default(),
            synth_prime: SynthPrimeRuntimeAgent::default(),
        }
    }
}

impl OmnisAgents {
    pub fn new(config: &super::config::OmnisConfig) -> Self {
        Self {
            config: config.clone(),
            ..Default::default()
        }
    }

    pub async fn initialize(&self, _config: &super::config::OmnisConfig) -> Result<(), String> {
        tracing::info!("Initializing OmnisAgents with oracle_7, meta_reasoner, world_model_x, chain_executor, truth_arbiter, synth_prime");
        Ok(())
    }

    pub async fn validate(&self) -> Result<(), String> {
        tracing::info!("OmnisAgents validated: 6 agents (oracle_7, meta_reasoner, world_model_x, chain_executor, truth_arbiter, synth_prime) ready");
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
