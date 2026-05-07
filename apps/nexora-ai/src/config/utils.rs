//! Utilities configuration

use serde::{Deserialize, Serialize};

/// Utils configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilsConfig {
    pub enable_crypto: bool,
    pub enable_text_processing: bool,
    pub enable_file_operations: bool,
    pub crypto_algorithm: String,
    pub text_processing_language: String,
    pub file_operations_max_size_mb: usize,
}

impl Default for UtilsConfig {
    fn default() -> Self {
        Self {
            enable_crypto: true,
            enable_text_processing: true,
            enable_file_operations: true,
            crypto_algorithm: "aes-256-gcm".to_string(),
            text_processing_language: "en".to_string(),
            file_operations_max_size_mb: 100,
        }
    }
}
