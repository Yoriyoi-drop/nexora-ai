use std::sync::Arc;
use parking_lot::RwLock as PRwLock;

use crate::erp::{ERPEngine, ERPConfig};
use crate::vogp::{VOGPPlus, VOGPConfig};
use crate::atqs::compression::AtqsCompression;
use crate::has_moe_ffn::{HasMoeFFN, HasMoeFFNConfig};

use super::tokenizer_integration::NxrTokenizerRef;

pub struct FoundationComponents {
    pub tokenizer: NxrTokenizerRef,
    pub erp: PRwLock<ERPEngine>,
    pub vogp: PRwLock<VOGPPlus>,
    pub atqs: AtqsCompression,
    pub moe: HasMoeFFN,
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
}

impl Default for FoundationComponents {
    fn default() -> Self {
        Self::new()
    }
}
