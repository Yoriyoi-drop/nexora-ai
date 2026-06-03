use async_trait::async_trait;
use std::sync::Arc;

use nexora_core::types::{
    ContextInfo, IntentType, ModelId, SpecialistModel as CoreSpecialistModel,
};
use nexora_shared::model_identity::NxrModelId;

// ── NXR Specialist Trait ───────────────────────────────────────────────

/// Unified interface for NXR model delegation — bridges model crate delegation
/// to the core routing system.
#[async_trait]
pub trait NxrSpecialist: Send + Sync {
    fn model_id(&self) -> NxrModelId;
    fn display_name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    async fn delegate(&self, prompt: &str) -> String;
}

// ── Concrete specialist per model ──────────────────────────────────────

macro_rules! define_specialist {
    ($name:ident, $id:ident, $display:expr, $desc:expr, $mod:ident) => {
        pub struct $name;

        #[async_trait]
        impl NxrSpecialist for $name {
            fn model_id(&self) -> NxrModelId { NxrModelId::$id }
            fn display_name(&self) -> &'static str { $display }
            fn description(&self) -> &'static str { $desc }
            async fn delegate(&self, prompt: &str) -> String {
                crate::$mod::delegation::delegate(prompt).await
            }
        }
    };
}

define_specialist!(OmnisSpecialist, Omnis, "Omnis", "Omniscient Reasoning — expert routing", omnis);
define_specialist!(VortexSpecialist, Vortex, "Vortex", "Code-Specialized Review — Oracle verifiers", vortex);
define_specialist!(AetherSpecialist, Aether, "Aether", "Emotional Intelligence — emotion + Caffeine", aether);
define_specialist!(SpectraSpecialist, Spectra, "Spectra", "Multimodal Creative — 3-temp + Caffeine", spectra);
define_specialist!(NexumSpecialist, Nexum, "Nexum", "Multi-Agent Orchestrator — SACA decomposition", nexum);
define_specialist!(AxiomSpecialist, Axiom, "Axiom", "Autonomous Decision Maker — SACA 6-phase", axiom);
define_specialist!(CipherSpecialist, Cipher, "Cipher", "Cybersecurity — threat + Oracle security", cipher);
define_specialist!(SwiftSpecialist, Swift, "Swift", "Ultra Lightweight — MoE task routing", swift);
define_specialist!(KronosSpecialist, Kronos, "Kronos", "Knowledge Management — temporal reasoning", kronos);
define_specialist!(GenesisSpecialist, Genesis, "Genesis", "Self-Improving Prototype — iterative refinement", genesis);

/// Returns all 10 NXR model specialists, wired to their delegation agents.
pub fn all_specialists() -> Vec<Box<dyn NxrSpecialist>> {
    vec![
        Box::new(OmnisSpecialist),
        Box::new(VortexSpecialist),
        Box::new(AetherSpecialist),
        Box::new(SpectraSpecialist),
        Box::new(NexumSpecialist),
        Box::new(AxiomSpecialist),
        Box::new(CipherSpecialist),
        Box::new(SwiftSpecialist),
        Box::new(KronosSpecialist),
        Box::new(GenesisSpecialist),
    ]
}

// ── Bridge to core SpecialistModel ─────────────────────────────────────

/// Maps NXR model → core ModelId + IntentType.
fn nxr_to_core_model(raw: NxrModelId) -> (ModelId, IntentType) {
    match raw {
        NxrModelId::Omnis   => (ModelId::Reasoning, IntentType::Reasoning),
        NxrModelId::Vortex  => (ModelId::Coding, IntentType::Coding),
        NxrModelId::Aether  => (ModelId::Personality, IntentType::Personality),
        NxrModelId::Spectra => (ModelId::Personality, IntentType::Personality),
        NxrModelId::Nexum   => (ModelId::Planner, IntentType::Planning),
        NxrModelId::Axiom   => (ModelId::Logic, IntentType::Reasoning),
        NxrModelId::Cipher  => (ModelId::Validator, IntentType::Validation),
        NxrModelId::Swift   => (ModelId::Optimizer, IntentType::Optimization),
        NxrModelId::Kronos  => (ModelId::Retrieval, IntentType::Retrieval),
        NxrModelId::Genesis => (ModelId::Optimizer, IntentType::Optimization),
    }
}

/// Wraps an NxrSpecialist into the core SpecialistModel trait so it can be
/// registered into `CoreController::register_specialist_model()`.
pub struct NxrCoreSpecialistBridge {
    inner: Box<dyn NxrSpecialist>,
    model_id: ModelId,
    intent: IntentType,
}

impl NxrCoreSpecialistBridge {
    pub fn new(specialist: Box<dyn NxrSpecialist>) -> Self {
        let (model_id, intent) = nxr_to_core_model(specialist.model_id());
        Self { inner: specialist, model_id, intent }
    }

    /// Register all 10 NXR specialists into any `HasSpecialistRegistry`.
    pub fn register_all(registry: &mut impl HasSpecialistRegistry) {
        for spec in all_specialists() {
            let bridge = NxrCoreSpecialistBridge::new(spec);
            let name = bridge.inner.display_name().to_string();
            registry.register_specialist_model(&name, Box::new(bridge));
        }
    }
}

/// Trait abstracting the part of CoreController that accepts specialist models.
pub trait HasSpecialistRegistry {
    fn register_specialist_model(&mut self, name: &str, model: Box<dyn CoreSpecialistModel>);
}

impl HasSpecialistRegistry for nexora_core::controller::CoreController {
    fn register_specialist_model(&mut self, name: &str, model: Box<dyn CoreSpecialistModel>) {
        self.register_specialist_model(name, model);
    }
}

#[async_trait]
impl CoreSpecialistModel for NxrCoreSpecialistBridge {
    async fn process(
        &self,
        input: &str,
        _context: &ContextInfo,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.inner.delegate(input).await)
    }

    fn model_id(&self) -> ModelId {
        self.model_id
    }

    fn can_handle(&self, intent: IntentType) -> bool {
        self.intent == intent
    }
}
