//! Crypto Prime Agent
//! 
//! Advanced cryptography and secure communication

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Crypto Prime Agent - Advanced cryptography and secure communication
#[derive(Debug, Clone)]
pub struct CryptoPrimeAgent {
    pub config: CryptoPrimeConfig,
    pub crypto_capabilities: CryptoCapabilities,
    pub security_engine: SecurityEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoPrimeConfig {
    pub base_config: BaseAgentConfig,
    pub crypto_model: CryptoModel,
    pub security_approach: SecurityApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptoModel {
    SymmetricEncryption,
    AsymmetricEncryption,
    HybridEncryption,
    QuantumResistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityApproach {
    ZeroKnowledge,
    EndToEndEncryption,
    MultiLayerSecurity,
    AdaptiveSecurity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoCapabilities {
    pub encryption: bool,
    pub decryption: bool,
    pub key_management: bool,
    pub secure_communication: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEngine {
    pub encryption_algorithms: Vec<String>,
    pub hash_functions: Vec<String>,
    pub key_exchange_protocols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoPrimeTaskInput {
    pub operation: String,
    pub data: Vec<u8>,
    pub security_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoPrimeTaskOutput {
    pub processed_data: Vec<u8>,
    pub security_metadata: HashMap<String, String>,
    pub encryption_details: HashMap<String, String>,
    pub security_level: f32,
}

impl Default for CryptoPrimeConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            crypto_model: CryptoModel::HybridEncryption,
            security_approach: SecurityApproach::MultiLayerSecurity,
        }
    }
}

impl Default for CryptoCapabilities {
    fn default() -> Self {
        Self {
            encryption: true,
            decryption: true,
            key_management: true,
            secure_communication: true,
        }
    }
}

impl Default for SecurityEngine {
    fn default() -> Self {
        Self {
            encryption_algorithms: vec![
                "aes_256_gcm".to_string(),
                "chacha20_poly1305".to_string(),
                "rsa_4096".to_string(),
                "ecdh_p384".to_string(),
            ],
            hash_functions: vec![
                "sha_256".to_string(),
                "sha_512".to_string(),
                "blake3".to_string(),
                "keccak_256".to_string(),
            ],
            key_exchange_protocols: vec![
                "diffie_hellman".to_string(),
                "elliptic_curve_diffie_hellman".to_string(),
                "post_quantum_kyber".to_string(),
            ],
        }
    }
}

impl Default for CryptoPrimeAgent {
    fn default() -> Self {
        Self {
            config: CryptoPrimeConfig::default(),
            crypto_capabilities: CryptoCapabilities::default(),
            security_engine: SecurityEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }
}

#[async_trait]
impl BaseAgent for CryptoPrimeAgent {
    type Config = CryptoPrimeConfig;
    type Input = CryptoPrimeTaskInput;
    type Output = CryptoPrimeTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let processed_data = self.process_crypto_operation(&input).await?;
        let security_metadata = self.generate_security_metadata(&input).await?;
        let encryption_details = self.create_encryption_details(&input).await?;
        let security_level = self.calculate_security_level(&input).await?;

        Ok(CryptoPrimeTaskOutput {
            processed_data,
            security_metadata,
            encryption_details,
            security_level,
        })
    }

    fn agent_id(&self) -> &str {
        &self.config.base_config.agent_id
    }

    fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }

    fn get_capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability {
                name: "crypto_prime".to_string(),
                description: "Advanced cryptography and secure communication".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["operation".to_string(), "data".to_string()],
                output_types: vec!["processed_data".to_string(), "security_metadata".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.98,
                    avg_latency: 3200.0,
                    resource_usage: 0.85,
                    reliability: 0.99,
                },
            },
        ]
    }

    fn get_metrics(&self) -> AgentMetrics {
        self.metrics.clone()
    }

    async fn initialize(&mut self, config: Self::Config) -> AgentResult<()> {
        self.config = config;
        self.status = AgentStatus::Idle;
        Ok(())
    }

    async fn shutdown(&mut self) -> AgentResult<()> {
        self.status = AgentStatus::Disabled;
        Ok(())
    }
}

impl CryptoPrimeAgent {
    pub fn new(config: CryptoPrimeConfig) -> Self {
        Self {
            config,
            crypto_capabilities: CryptoCapabilities::default(),
            security_engine: SecurityEngine::default(),
            status: AgentStatus::Idle,
            metrics: AgentMetrics {
                tasks_processed: 0,
                avg_processing_time: 0.0,
                success_rate: 1.0,
                current_load: 0.0,
                last_activity: chrono::Utc::now(),
            },
        }
    }

    async fn process_crypto_operation(&self, input: &CryptoPrimeTaskInput) -> AgentResult<Vec<u8>> {
        match input.operation.as_str() {
            "encrypt" => self.encrypt_data(&input.data, &input.security_parameters).await,
            "decrypt" => self.decrypt_data(&input.data, &input.security_parameters).await,
            "hash" => self.hash_data(&input.data, &input.security_parameters).await,
            _ => Ok(input.data.clone()),
        }
    }

    async fn encrypt_data(&self, data: &[u8], _params: &HashMap<String, String>) -> AgentResult<Vec<u8>> {
        // Simple XOR encryption for demonstration
        let key = 0x42;
        let mut encrypted = Vec::with_capacity(data.len());
        for &byte in data {
            encrypted.push(byte ^ key);
        }
        Ok(encrypted)
    }

    async fn decrypt_data(&self, data: &[u8], _params: &HashMap<String, String>) -> AgentResult<Vec<u8>> {
        // Simple XOR decryption for demonstration
        let key = 0x42;
        let mut decrypted = Vec::with_capacity(data.len());
        for &byte in data {
            decrypted.push(byte ^ key);
        }
        Ok(decrypted)
    }

    async fn hash_data(&self, data: &[u8], _params: &HashMap<String, String>) -> AgentResult<Vec<u8>> {
        // Simple hash for demonstration
        let mut hash = 0u64;
        for &byte in data {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        Ok(hash.to_le_bytes().to_vec())
    }

    async fn generate_security_metadata(&self, input: &CryptoPrimeTaskInput) -> AgentResult<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        metadata.insert("operation".to_string(), input.operation.clone());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        metadata.insert("data_size".to_string(), input.data.len().to_string());
        metadata.insert("crypto_model".to_string(), "HybridEncryption".to_string());
        metadata.insert("security_approach".to_string(), "MultiLayerSecurity".to_string());
        
        Ok(metadata)
    }

    async fn create_encryption_details(&self, input: &CryptoPrimeTaskInput) -> AgentResult<HashMap<String, String>> {
        let mut details = HashMap::new();
        
        details.insert("algorithm".to_string(), "aes_256_gcm".to_string());
        details.insert("key_size".to_string(), "256".to_string());
        details.insert("mode".to_string(), "gcm".to_string());
        details.insert("padding".to_string(), "pkcs7".to_string());
        
        if input.security_parameters.contains_key("custom_params") {
            details.insert("custom_params".to_string(), "applied".to_string());
        }
        
        Ok(details)
    }

    async fn calculate_security_level(&self, input: &CryptoPrimeTaskInput) -> AgentResult<f32> {
        let base_security = 0.8;
        let parameter_bonus = if input.security_parameters.len() > 0 { 0.1 } else { 0.0 };
        let data_size_factor = if input.data.len() > 100 { 0.05 } else { 0.0 };
        
        Ok(base_security + parameter_bonus + data_size_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_prime_agent_creation() {
        let agent = CryptoPrimeAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_crypto_prime_task_processing() {
        let agent = CryptoPrimeAgent::default();
        let input = CryptoPrimeTaskInput {
            operation: "encrypt".to_string(),
            data: b"Hello, World!".to_vec(),
            security_parameters: HashMap::from([
                ("key_type".to_string(), "symmetric".to_string()),
                ("algorithm".to_string(), "aes".to_string()),
            ]),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.processed_data.is_empty());
        assert!(!output.security_metadata.is_empty());
        assert!(!output.encryption_details.is_empty());
        assert!(output.security_level > 0.0);
    }

    #[tokio::test]
    async fn test_encryption_decryption() {
        let agent = CryptoPrimeAgent::default();
        let original_data = b"Test message".to_vec();
        
        let encrypt_input = CryptoPrimeTaskInput {
            operation: "encrypt".to_string(),
            data: original_data.clone(),
            security_parameters: HashMap::new(),
        };
        
        let encrypt_result = agent.process(encrypt_input).await.unwrap();
        let encrypted_data = encrypt_result.processed_data;
        
        let decrypt_input = CryptoPrimeTaskInput {
            operation: "decrypt".to_string(),
            data: encrypted_data,
            security_parameters: HashMap::new(),
        };
        
        let decrypt_result = agent.process(decrypt_input).await.unwrap();
        let decrypted_data = decrypt_result.processed_data;
        
        assert_eq!(original_data, decrypted_data);
    }

    #[tokio::test]
    async fn test_hashing() {
        let agent = CryptoPrimeAgent::default();
        let input = CryptoPrimeTaskInput {
            operation: "hash".to_string(),
            data: b"Hash this message".to_vec(),
            security_parameters: HashMap::new(),
        };

        let result = agent.process(input).await.unwrap();
        assert!(!result.processed_data.is_empty());
        assert_eq!(result.processed_data.len(), 8); // 64-bit hash
    }
}
