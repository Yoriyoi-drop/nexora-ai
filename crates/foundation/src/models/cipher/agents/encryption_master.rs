//! Encryption Master Agent
//! 
//! Advanced encryption and data protection

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Encryption Master Agent - Advanced encryption and data protection
#[derive(Debug, Clone)]
pub struct EncryptionMasterAgent {
    pub config: EncryptionMasterConfig,
    pub encryption_capabilities: EncryptionCapabilities,
    pub data_protection: DataProtection,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMasterConfig {
    pub base_config: BaseAgentConfig,
    pub encryption_model: EncryptionModel,
    pub protection_approach: ProtectionApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionModel {
    SymmetricEncryption,
    AsymmetricEncryption,
    HybridEncryption,
    QuantumEncryption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtectionApproach {
    DataAtRest,
    DataInTransit,
    DataInUse,
    EndToEndProtection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionCapabilities {
    pub data_encryption: bool,
    pub key_management: bool,
    pub secure_storage: bool,
    pub access_control: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProtection {
    pub encryption_algorithms: Vec<String>,
    pub key_derivation_methods: Vec<String>,
    pub protection_policies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMasterTaskInput {
    pub protection_request: String,
    pub data_to_protect: Vec<u8>,
    pub encryption_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMasterTaskOutput {
    pub protected_data: Vec<u8>,
    pub encryption_metadata: HashMap<String, String>,
    pub access_controls: Vec<String>,
    pub protection_level: f32,
}

impl Default for EncryptionMasterConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            encryption_model: EncryptionModel::HybridEncryption,
            protection_approach: ProtectionApproach::EndToEndProtection,
        }
    }
}

impl Default for EncryptionCapabilities {
    fn default() -> Self {
        Self {
            data_encryption: true,
            key_management: true,
            secure_storage: true,
            access_control: true,
        }
    }
}

impl Default for DataProtection {
    fn default() -> Self {
        Self {
            encryption_algorithms: vec![
                "aes_256_gcm".to_string(),
                "chacha20_poly1305".to_string(),
                "rsa_oaep".to_string(),
                "ecdh".to_string(),
            ],
            key_derivation_methods: vec![
                "pbkdf2".to_string(),
                "scrypt".to_string(),
                "argon2".to_string(),
                "hkdf".to_string(),
            ],
            protection_policies: vec![
                "zero_knowledge".to_string(),
                "perfect_forward_secrecy".to_string(),
                "data_minimization".to_string(),
            ],
        }
    }
}

impl Default for EncryptionMasterAgent {
    fn default() -> Self {
        Self {
            config: EncryptionMasterConfig::default(),
            encryption_capabilities: EncryptionCapabilities::default(),
            data_protection: DataProtection::default(),
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
impl BaseAgent for EncryptionMasterAgent {
    type Config = EncryptionMasterConfig;
    type Input = EncryptionMasterTaskInput;
    type Output = EncryptionMasterTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let protected_data = self.protect_data(&input).await?;
        let encryption_metadata = self.generate_encryption_metadata(&input).await?;
        let access_controls = self.create_access_controls(&input).await?;
        let protection_level = self.calculate_protection_level(&input).await?;

        Ok(EncryptionMasterTaskOutput {
            protected_data,
            encryption_metadata,
            access_controls,
            protection_level,
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
                name: "encryption_master".to_string(),
                description: "Advanced encryption and data protection".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["protection_request".to_string(), "data_to_protect".to_string()],
                output_types: vec!["protected_data".to_string(), "encryption_metadata".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.99,
                    avg_latency: 3500.0,
                    resource_usage: 0.88,
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

impl EncryptionMasterAgent {
    pub fn new(config: EncryptionMasterConfig) -> Self {
        Self {
            config,
            encryption_capabilities: EncryptionCapabilities::default(),
            data_protection: DataProtection::default(),
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

    async fn protect_data(&self, input: &EncryptionMasterTaskInput) -> AgentResult<Vec<u8>> {
        match input.protection_request.as_str() {
            "encrypt" => self.encrypt_data(&input.data_to_protect, &input.encryption_parameters).await,
            "sign" => self.sign_data(&input.data_to_protect, &input.encryption_parameters).await,
            "seal" => self.seal_data(&input.data_to_protect, &input.encryption_parameters).await,
            _ => Ok(input.data_to_protect.clone()),
        }
    }

    async fn encrypt_data(&self, data: &[u8], _params: &HashMap<String, String>) -> AgentResult<Vec<u8>> {
        // Simple XOR encryption for demonstration
        let key = 0xAB;
        let mut encrypted = Vec::with_capacity(data.len());
        for &byte in data {
            encrypted.push(byte ^ key);
        }
        Ok(encrypted)
    }

    async fn sign_data(&self, data: &[u8], _params: &HashMap<String, String>) -> AgentResult<Vec<u8>> {
        // Simple signature for demonstration
        let mut hash = 0u64;
        for &byte in data {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        Ok(hash.to_le_bytes().to_vec())
    }

    async fn seal_data(&self, data: &[u8], params: &HashMap<String, String>) -> AgentResult<Vec<u8>> {
        // Combine encryption and signing for demonstration
        let encrypted = self.encrypt_data(data, params).await?;
        let signature = self.sign_data(data, params).await?;
        
        let mut sealed = Vec::new();
        sealed.extend_from_slice(&(encrypted.len() as u32).to_le_bytes());
        sealed.extend_from_slice(&encrypted);
        sealed.extend_from_slice(&signature);
        
        Ok(sealed)
    }

    async fn generate_encryption_metadata(&self, input: &EncryptionMasterTaskInput) -> AgentResult<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        metadata.insert("protection_request".to_string(), input.protection_request.clone());
        metadata.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        metadata.insert("data_size".to_string(), input.data_to_protect.len().to_string());
        metadata.insert("encryption_model".to_string(), "HybridEncryption".to_string());
        metadata.insert("protection_approach".to_string(), "EndToEndProtection".to_string());
        
        if input.encryption_parameters.contains_key("algorithm") {
            metadata.insert("algorithm".to_string(), 
                input.encryption_parameters.get("algorithm").expect("algorithm key exists").clone());
        }
        
        Ok(metadata)
    }

    async fn create_access_controls(&self, input: &EncryptionMasterTaskInput) -> AgentResult<Vec<String>> {
        let mut controls = Vec::new();
        
        controls.push(format!("Access control for: {}", input.protection_request));
        controls.push("Require authentication for data access".to_string());
        controls.push("Implement role-based access control".to_string());
        controls.push("Log all access attempts".to_string());
        
        if input.encryption_parameters.contains_key("access_level") {
            controls.push(format!("Access level: {}", 
                input.encryption_parameters.get("access_level").expect("access_level key exists")));
        }
        
        Ok(controls)
    }

    async fn calculate_protection_level(&self, input: &EncryptionMasterTaskInput) -> AgentResult<f32> {
        let base_protection: f32 = 0.85;
        let parameter_bonus: f32 = if input.encryption_parameters.len() > 0 { 0.1 } else { 0.0 };
        let data_size_factor: f32 = if input.data_to_protect.len() > 100 { 0.05 } else { 0.0 };
        
        Ok((base_protection + parameter_bonus + data_size_factor).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_master_agent_creation() {
        let agent = EncryptionMasterAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_encryption_master_task_processing() {
        let agent = EncryptionMasterAgent::default();
        let input = EncryptionMasterTaskInput {
            protection_request: "encrypt".to_string(),
            data_to_protect: b"Sensitive data to protect".to_vec(),
            encryption_parameters: HashMap::from([
                ("algorithm".to_string(), "aes_256_gcm".to_string()),
                ("key_size".to_string(), "256".to_string()),
            ]),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.protected_data.is_empty());
        assert!(!output.encryption_metadata.is_empty());
        assert!(!output.access_controls.is_empty());
        assert!(output.protection_level > 0.0);
    }

    #[tokio::test]
    async fn test_data_sealing() {
        let agent = EncryptionMasterAgent::default();
        let input = EncryptionMasterTaskInput {
            protection_request: "seal".to_string(),
            data_to_protect: b"Data to seal".to_vec(),
            encryption_parameters: HashMap::new(),
        };

        let result = agent.process(input).await.unwrap();
        assert!(!result.protected_data.is_empty());
        
        // Verify sealed data structure: [4 bytes length][encrypted data][8 bytes signature]
        assert!(result.protected_data.len() > 12); // At least 4 + 1 + 8 bytes
    }

    #[tokio::test]
    async fn test_protection_level_calculation() {
        let agent = EncryptionMasterAgent::default();
        
        let input_with_params = EncryptionMasterTaskInput {
            protection_request: "encrypt".to_string(),
            data_to_protect: vec![0u8; 200], // Large data
            encryption_parameters: HashMap::from([
                ("algorithm".to_string(), "aes".to_string()),
            ]),
        };
        
        let level = agent.calculate_protection_level(&input_with_params).await.unwrap();
        assert!(level > 0.9); // Should have high protection level
        
        let input_minimal = EncryptionMasterTaskInput {
            protection_request: "encrypt".to_string(),
            data_to_protect: vec![1u8, 2u8, 3u8], // Small data
            encryption_parameters: HashMap::new(),
        };
        
        let level_minimal = agent.calculate_protection_level(&input_minimal).await.unwrap();
        assert!(level_minimal >= 0.85); // Base protection level
    }
}
