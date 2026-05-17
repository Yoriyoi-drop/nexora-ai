use std::sync::Arc;
use parking_lot::RwLock as PRwLock;

use crate::erp::{ERPEngine, ERPConfig};
use crate::vogp::{VOGPPlus, VOGPConfig};
use crate::atqs::compression::AtqsCompression;
use crate::has_moe_ffn::{HasMoeFFN, HasMoeFFNConfig};

use super::tokenizer_integration::NxrTokenizerRef;
use super::deeplearning_integration::{DeepLearningEngine, DeepLearningConfig};
use super::gnac_integration::{GnacEngine, GnacIntegrationConfig};

pub struct FoundationComponents {
    pub tokenizer: NxrTokenizerRef,
    pub erp: PRwLock<ERPEngine>,
    pub vogp: PRwLock<VOGPPlus>,
    pub atqs: AtqsCompression,
    pub moe: HasMoeFFN,
    pub dl_engine: DeepLearningEngine,
    pub gnac_engine: GnacEngine,
}

impl FoundationComponents {
    pub fn new() -> Self {
        Self {
            tokenizer: Arc::new(PRwLock::new(
                nexora_tokenizer::BpeTokenizer::default()
            )),
            erp: PRwLock::new(ERPEngine::new(ERPConfig::default())),
            vogp: PRwLock::new(VOGPPlus::new()),
            atqs: AtqsCompression::new(),
            moe: HasMoeFFN::default(),
            dl_engine: DeepLearningEngine::new(DeepLearningConfig::hybrid())
                .expect("Failed to initialize DeepLearningEngine with hybrid config"),
            gnac_engine: GnacEngine::new(GnacIntegrationConfig::default()),
        }
    }

    pub fn with_moe_config(mut self, moe_config: HasMoeFFNConfig) -> Self {
        self.moe = HasMoeFFN::new(moe_config);
        self
    }

    pub fn with_erp_config(mut self, erp_config: ERPConfig) -> Self {
        self.erp = PRwLock::new(ERPEngine::new(erp_config));
        self
    }

    pub fn with_vogp_config(mut self, vogp_config: VOGPConfig) -> Self {
        self.vogp = PRwLock::new(VOGPPlus::with_config(vogp_config));
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: NxrTokenizerRef) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Returns a `Result` so callers can handle DL engine initialization failures.
    pub fn with_dl_config(mut self, dl_config: DeepLearningConfig) -> Result<Self, String> {
        self.dl_engine = DeepLearningEngine::new(dl_config)
            .map_err(|e| format!("Failed to initialize DeepLearningEngine: {e}"))?;
        Ok(self)
    }

    pub fn with_gnac_config(mut self, gnac_config: GnacIntegrationConfig) -> Self {
        self.gnac_engine = GnacEngine::new(gnac_config);
        self
    }
}

impl Default for FoundationComponents {
    fn default() -> Self {
        Self::new()
    }
}
