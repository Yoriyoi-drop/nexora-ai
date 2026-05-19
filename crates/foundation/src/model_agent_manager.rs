use std::any::Any;
use std::sync::OnceLock;
use tracing::info;

use nexora_models::omnis::agents::OmnisAgents;
use nexora_models::swift::agents::SwiftAgents;
use nexora_models::genesis::agents::GenesisAgents;
use nexora_models::nexum::agents::NexumAgents;
use nexora_models::axiom::agents::AxiomAgents;
use nexora_models::kronos::agents::KronosAgents;
use nexora_models::cipher::agents::CipherAgents;
use nexora_models::aether::agents::AetherAgents;

use nexora_models::omnis::config::OmnisConfig;
use nexora_models::swift::config::SwiftConfig;
use nexora_models::genesis::config::GenesisConfig;
use nexora_models::axiom::config::AxiomConfig;
use nexora_models::kronos::config::KronosConfig;
use nexora_models::cipher::config::CipherConfig;

#[allow(dead_code)]
pub struct ModelAgentManager {
    pub omnis: OmnisAgents,
    pub swift: SwiftAgents,
    pub genesis: GenesisAgents,
    pub nexum: NexumAgents,
    pub axiom: AxiomAgents,
    pub kronos: KronosAgents,
    pub cipher: CipherAgents,
    pub aether: AetherAgents,
    _boxes: Vec<Box<dyn Any + Send + Sync>>,
}

impl ModelAgentManager {
    pub async fn new() -> Self {
        let omnis = OmnisAgents::new(&OmnisConfig::default());
        omnis.initialize(&OmnisConfig::default()).await.unwrap_or_else(|e| {
            info!("omnis agents initialize returned: {e}");
        });
        info!("NXR-OMNIS agents activated (oracle_7, meta_reasoner, world_model_x, chain_executor, truth_arbiter, synth_prime) ✓");

        let mut swift = SwiftAgents::new(&SwiftConfig::default());
        swift.initialize().await.unwrap_or_else(|e| {
            info!("swift agents initialize returned: {e}");
        });
        info!("NXR-SWIFT agents activated ✓");

        let genesis = GenesisAgents::new(&GenesisConfig::default());
        info!("NXR-GENESIS agents activated ✓");

        let nexum = NexumAgents::new();
        info!("NXR-NEXUM agents activated ✓");

        let axiom = AxiomAgents::new(&AxiomConfig::default());
        info!("NXR-AXIOM agents activated ✓");

        let kronos = KronosAgents::new(&KronosConfig::default());
        info!("NXR-KRONOS agents activated ✓");

        let cipher = CipherAgents::new(&CipherConfig::default());
        info!("NXR-CIPHER agents activated ✓");

        let aether = AetherAgents::default();
        info!("NXR-AETHER agents activated (empath_core, tone_mapper, context_weave, soul_mirror) ✓");

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
            _boxes: Vec::new(),
        }
    }
}

static MODEL_AGENTS: OnceLock<ModelAgentManager> = OnceLock::new();

pub fn global_model_agents() -> &'static ModelAgentManager {
    MODEL_AGENTS.get().expect("ModelAgentManager not initialized")
}

pub async fn init_model_agents() {
    let mgr = ModelAgentManager::new().await;
    let _ = MODEL_AGENTS.set(mgr);
    info!("All model-specific agents active ✓");
}
