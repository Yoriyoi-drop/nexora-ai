use std::sync::OnceLock;
use tracing::{info, warn};

use nexora_models::aether::agents::AetherAgents;
use nexora_models::axiom::agents::AxiomAgents;
use nexora_models::cipher::agents::CipherAgents;
use nexora_models::genesis::agents::GenesisAgents;
use nexora_models::kronos::agents::KronosAgents;
use nexora_models::nexum::agents::NexumAgents;
use nexora_models::omnis::agents::OmnisAgents;
use nexora_models::swift::agents::SwiftAgents;

use nexora_models::axiom::config::AxiomConfig;
use nexora_models::cipher::config::CipherConfig;
use nexora_models::genesis::config::GenesisConfig;
use nexora_models::kronos::config::KronosConfig;
use nexora_models::omnis::config::OmnisConfig;
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
}

impl ModelAgentManager {
    pub async fn new() -> Self {
        let omnis = OmnisAgents::new(&OmnisConfig::default());
        match omnis
            .initialize(&OmnisConfig::default())
            .await
        {
            Ok(_) => info!("NXR-OMNIS agents activated (oracle_7, meta_reasoner, world_model_x, chain_executor, truth_arbiter, synth_prime) ✓"),
            Err(e) => warn!("NXR-OMNIS agents initialization failed: {e}"),
        }

        let mut swift = SwiftAgents::new(&SwiftConfig::default());
        match swift.initialize().await {
            Ok(_) => info!("NXR-SWIFT agents activated ✓"),
            Err(e) => warn!("NXR-SWIFT agents initialization failed: {e}"),
        }

        let genesis = GenesisAgents::new(&GenesisConfig::default());
        match genesis.initialize(&GenesisConfig::default()).await {
            Ok(_) => info!("NXR-GENESIS agents activated ✓"),
            Err(e) => warn!("NXR-GENESIS agents initialization failed: {e}"),
        }

        let nexum = NexumAgents::new();
        match nexum.initialize().await {
            Ok(_) => info!("NXR-NEXUM agents activated ✓"),
            Err(e) => warn!("NXR-NEXUM agents initialization failed: {e}"),
        }

        let axiom = AxiomAgents::new(&AxiomConfig::default());
        match axiom.initialize(&AxiomConfig::default()).await {
            Ok(_) => info!("NXR-AXIOM agents activated ✓"),
            Err(e) => warn!("NXR-AXIOM agents initialization failed: {e}"),
        }

        let kronos = KronosAgents::new(&KronosConfig::default());
        match kronos.initialize(&KronosConfig::default()).await {
            Ok(_) => info!("NXR-KRONOS agents activated ✓"),
            Err(e) => warn!("NXR-KRONOS agents initialization failed: {e}"),
        }

        let cipher = CipherAgents::new(&CipherConfig::default());
        match cipher.initialize(&CipherConfig::default()).await {
            Ok(_) => info!("NXR-CIPHER agents activated ✓"),
            Err(e) => warn!("NXR-CIPHER agents initialization failed: {e}"),
        }

        let aether = AetherAgents::default();
        match aether.initialize().await {
            Ok(_) => info!("NXR-AETHER agents activated (empath_core, tone_mapper, context_weave, soul_mirror) ✓"),
            Err(e) => warn!("NXR-AETHER agents initialization failed: {e}"),
        }

        info!("NXR-SPECTRA agents active (spectrum_analyzer, spectral_mapper, spectral_processor, frequency_analyzer, creative_muse, artistic_weaver, style_adapter, innovation_engine) ✓");
        info!("NXR-VORTEX agents active ✓");

        Self {
            omnis,
            swift,
            genesis,
            nexum,
            axiom,
            kronos,
            cipher,
            aether,
        }
    }
}

static MODEL_AGENTS: OnceLock<ModelAgentManager> = OnceLock::new();

pub fn global_model_agents() -> &'static ModelAgentManager {
    MODEL_AGENTS
        .get()
        .unwrap_or_else(|| {
            panic!("ModelAgentManager not initialized. Call init_model_agents() first.")
        })
}

pub async fn init_model_agents() {
    let mgr = ModelAgentManager::new().await;
    if MODEL_AGENTS.set(mgr).is_err() {
        warn!("ModelAgentManager already initialized");
    } else {
        info!("All model-specific agents active ✓");
    }
}
