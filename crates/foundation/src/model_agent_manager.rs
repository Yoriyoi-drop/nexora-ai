use std::sync::OnceLock;
use tracing::{info, warn};

use nexora_models::aether::agents::AetherAgents;
use nexora_models::axiom::agents::AxiomAgents;
use nexora_models::cipher::agents::CipherAgents;
use nexora_models::genesis::agents::GenesisAgents;
use nexora_models::kronos::agents::KronosAgents;
use nexora_models::nexum::agents::NexumAgents;
use nexora_models::omnis::agents::OmnisAgents;
use nexora_models::spectra::agents::SpectraAgents;
use nexora_models::swift::agents::SwiftAgents;

use nexora_models::axiom::config::AxiomConfig;
use nexora_models::cipher::config::CipherConfig;
use nexora_models::genesis::config::GenesisConfig;
use nexora_models::kronos::config::KronosConfig;
use nexora_models::omnis::config::OmnisConfig;
use nexora_models::spectra::config::SpectraConfig;
use nexora_models::swift::config::SwiftConfig;

pub struct ModelAgentManager {
    pub omnis: OmnisAgents,
    pub swift: SwiftAgents,
    pub genesis: GenesisAgents,
    pub nexum: NexumAgents,
    pub axiom: AxiomAgents,
    pub kronos: KronosAgents,
    pub cipher: CipherAgents,
    pub aether: AetherAgents,
    pub spectra: SpectraAgents,
}

impl ModelAgentManager {
    pub async fn new() -> Self {
        let omnis = OmnisAgents::new(&OmnisConfig::default());
        info!("NXR-OMNIS agents constructed (oracle_7, meta_reasoner, world_model_x, chain_executor, truth_arbiter, synth_prime)");

        let swift = SwiftAgents::new(&SwiftConfig::default());
        info!("NXR-SWIFT agents constructed");

        let genesis = GenesisAgents::new(&GenesisConfig::default());
        info!("NXR-GENESIS agents constructed");

        let nexum = NexumAgents::new();
        info!("NXR-NEXUM agents constructed");

        let axiom = AxiomAgents::new(&AxiomConfig::default());
        info!("NXR-AXIOM agents constructed");

        let kronos = KronosAgents::new(&KronosConfig::default());
        info!("NXR-KRONOS agents constructed");

        let cipher = CipherAgents::new(&CipherConfig::default());
        info!("NXR-CIPHER agents constructed");

        let aether = AetherAgents::default();
        info!("NXR-AETHER agents constructed");

        let spectra = SpectraAgents::new(&SpectraConfig::default());
        info!("NXR-SPECTRA agents constructed");

        info!("NXR-VORTEX agents active");

        Self {
            omnis,
            swift,
            genesis,
            nexum,
            axiom,
            kronos,
            cipher,
            aether,
            spectra,
        }
    }
}

static MODEL_AGENTS: OnceLock<ModelAgentManager> = OnceLock::new();

pub fn global_model_agents() -> Option<&'static ModelAgentManager> {
    MODEL_AGENTS.get()
}

pub async fn init_model_agents() {
    let mgr = ModelAgentManager::new().await;
    if MODEL_AGENTS.set(mgr).is_err() {
        warn!("ModelAgentManager already initialized");
    } else {
        info!("All model-specific agents active ✓");
    }
}
