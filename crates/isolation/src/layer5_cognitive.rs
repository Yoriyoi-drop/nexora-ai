use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveIsolationLayer {
    memory_isolation: MemoryIsolationMap,
    chain_isolation: ReasoningChainIsolation,
    auto_quarantine: AutoQuarantineConfig,
    anomaly_detector: AnomalyDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIsolationMap {
    agent_memory_regions: HashMap<Uuid, MemoryRegion>,
    shared_regions: HashMap<String, MemoryRegion>,
    isolation_enforcement: IsolationEnforcement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub entries: Vec<MemoryEntry>,
    pub access_control: MemoryAccessControl,
    pub integrity_hash: String,
    pub is_quarantined: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub key: String,
    pub value: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub checksum: u64,
    pub access_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessControl {
    pub owner_agent: Uuid,
    pub read_allowed_agents: HashSet<Uuid>,
    pub write_allowed_agents: HashSet<Uuid>,
    pub world_readable: bool,
    pub world_writable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationEnforcement {
    Strict,
    Relaxed,
    AuditOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChainIsolation {
    pub chain_registry: HashMap<Uuid, ReasoningChain>,
    pub max_depth: u32,
    pub prevent_cross_chain_contamination: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub parent_chain: Option<Uuid>,
    pub steps: Vec<ReasoningStep>,
    pub isolation_scope: ChainIsolationScope,
    pub integrity_checked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: Uuid,
    pub input_hash: String,
    pub output_hash: String,
    pub reasoning_type: String,
    pub confidence: f64,
    pub token_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChainIsolationScope {
    Private,
    SharedWithinMode(String),
    CrossMode(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoQuarantineConfig {
    pub enabled: bool,
    pub anomaly_threshold: f64,
    pub corruption_threshold: f64,
    pub max_anomalies_before_quarantine: u32,
    pub notify_on_quarantine: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetector {
    pub per_agent_scores: HashMap<Uuid, AnomalyScore>,
    pub global_anomaly_threshold: f64,
    pub detection_window_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyScore {
    pub agent_id: Uuid,
    pub score: f64,
    pub reasons: Vec<String>,
    pub detected_at: chrono::DateTime<chrono::Utc>,
    pub auto_quarantined: bool,
}

pub type SharedCognitiveIsolation = Arc<RwLock<CognitiveIsolationLayer>>;

impl CognitiveIsolationLayer {
    pub fn new() -> Self {
        Self {
            memory_isolation: MemoryIsolationMap {
                agent_memory_regions: HashMap::new(),
                shared_regions: HashMap::new(),
                isolation_enforcement: IsolationEnforcement::Strict,
            },
            chain_isolation: ReasoningChainIsolation {
                chain_registry: HashMap::new(),
                max_depth: 10,
                prevent_cross_chain_contamination: true,
            },
            auto_quarantine: AutoQuarantineConfig {
                enabled: true,
                anomaly_threshold: 0.7,
                corruption_threshold: 0.9,
                max_anomalies_before_quarantine: 3,
                notify_on_quarantine: true,
            },
            anomaly_detector: AnomalyDetector {
                per_agent_scores: HashMap::new(),
                global_anomaly_threshold: 0.7,
                detection_window_seconds: 300,
            },
        }
    }

    pub fn create_memory_region(&mut self, agent_id: Uuid) -> Uuid {
        let region = MemoryRegion {
            id: Uuid::new_v4(),
            agent_id,
            entries: Vec::new(),
            access_control: MemoryAccessControl {
                owner_agent: agent_id,
                read_allowed_agents: HashSet::from([agent_id]),
                write_allowed_agents: HashSet::from([agent_id]),
                world_readable: false,
                world_writable: false,
            },
            integrity_hash: String::new(),
            is_quarantined: false,
        };
        let id = region.id;
        self.memory_isolation.agent_memory_regions.insert(id, region);
        id
    }

    pub fn write_memory(
        &mut self,
        region_id: Uuid,
        agent_id: Uuid,
        key: &str,
        value: Vec<u8>,
    ) -> Result<(), CognitiveError> {
        let region = self.memory_isolation.agent_memory_regions.get_mut(&region_id)
            .ok_or(CognitiveError::MemoryRegionNotFound(region_id))?;

        if region.is_quarantined {
            return Err(CognitiveError::RegionQuarantined(region_id));
        }

        if !region.access_control.write_allowed_agents.contains(&agent_id)
            && !region.access_control.world_writable
        {
            return Err(CognitiveError::WriteAccessDenied { agent_id, region_id });
        }

        let checksum = simple_checksum(&value);
        region.entries.push(MemoryEntry {
            id: Uuid::new_v4(),
            key: key.to_string(),
            value,
            timestamp: chrono::Utc::now(),
            checksum,
            access_count: 0,
        });

        region.integrity_hash = format!("{:x}", checksum);
        Ok(())
    }

    pub fn read_memory(
        &mut self,
        region_id: Uuid,
        agent_id: Uuid,
        key: &str,
    ) -> Result<Option<Vec<u8>>, CognitiveError> {
        let region = self.memory_isolation.agent_memory_regions.get_mut(&region_id)
            .ok_or(CognitiveError::MemoryRegionNotFound(region_id))?;

        if !region.access_control.read_allowed_agents.contains(&agent_id)
            && !region.access_control.world_readable
        {
            return Err(CognitiveError::ReadAccessDenied { agent_id, region_id });
        }

        if let Some(entry) = region.entries.iter_mut().find(|e| e.key == key) {
            entry.access_count += 1;
            Ok(Some(entry.value.clone()))
        } else {
            Ok(None)
        }
    }

    pub fn start_reasoning_chain(
        &mut self,
        agent_id: Uuid,
        parent: Option<Uuid>,
        scope: ChainIsolationScope,
    ) -> Uuid {
        let chain = ReasoningChain {
            id: Uuid::new_v4(),
            agent_id,
            parent_chain: parent,
            steps: Vec::new(),
            isolation_scope: scope,
            integrity_checked: false,
        };
        let id = chain.id;
        self.chain_isolation.chain_registry.insert(id, chain);
        id
    }

    pub fn add_reasoning_step(
        &mut self,
        chain_id: Uuid,
        input: &str,
        output: &str,
        reasoning_type: &str,
        confidence: f64,
    ) -> Result<ReasoningStep, CognitiveError> {
        let parent_chain = self.chain_isolation.chain_registry.get(&chain_id)
            .and_then(|c| c.parent_chain);

        if self.chain_isolation.prevent_cross_chain_contamination {
            if let Some(parent) = parent_chain {
                if !self.chain_isolation.chain_registry.contains_key(&parent) {
                    return Err(CognitiveError::OrphanChain(chain_id));
                }
            }
        }

        let chain = self.chain_isolation.chain_registry.get_mut(&chain_id)
            .ok_or(CognitiveError::ReasoningChainNotFound(chain_id))?;

        let step = ReasoningStep {
            step_id: Uuid::new_v4(),
            input_hash: simple_hash(input),
            output_hash: simple_hash(output),
            reasoning_type: reasoning_type.to_string(),
            confidence,
            token_count: (input.len() + output.len()) as u64,
        };
        chain.steps.push(step.clone());
        Ok(step)
    }

    pub fn detect_anomaly(&mut self, agent_id: Uuid, score: f64, reason: &str) {
        let entry = self.anomaly_detector.per_agent_scores.entry(agent_id).or_insert(AnomalyScore {
            agent_id,
            score: 0.0,
            reasons: Vec::new(),
            detected_at: chrono::Utc::now(),
            auto_quarantined: false,
        });
        entry.score = score;
        entry.reasons.push(reason.to_string());
        entry.detected_at = chrono::Utc::now();

        if self.auto_quarantine.enabled && score >= self.auto_quarantine.anomaly_threshold {
            let anomaly_count = entry.reasons.len() as u32;
            if anomaly_count >= self.auto_quarantine.max_anomalies_before_quarantine {
                entry.auto_quarantined = true;
                for (_, region) in self.memory_isolation.agent_memory_regions.iter_mut() {
                    if region.agent_id == agent_id {
                        region.is_quarantined = true;
                    }
                }
            }
        }
    }

    pub fn quarantine_agent_memory(&mut self, agent_id: Uuid) -> Vec<Uuid> {
        let mut quarantined = Vec::new();
        for (id, region) in self.memory_isolation.agent_memory_regions.iter_mut() {
            if region.agent_id == agent_id {
                region.is_quarantined = true;
                quarantined.push(*id);
            }
        }
        quarantined
    }

    pub fn verify_memory_integrity(&self, region_id: Uuid) -> Result<bool, CognitiveError> {
        let region = self.memory_isolation.agent_memory_regions.get(&region_id)
            .ok_or(CognitiveError::MemoryRegionNotFound(region_id))?;
        let computed: u64 = region.entries.iter().map(|e| e.checksum).sum();
        let expected: u64 = u64::from_str_radix(&region.integrity_hash, 16).unwrap_or(0);
        Ok(computed == expected)
    }
}

fn simple_checksum(data: &[u8]) -> u64 {
    data.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

fn simple_hash(s: &str) -> String {
    format!("{:x}", s.bytes().fold(0u64, |acc, b| acc.wrapping_mul(127).wrapping_add(b as u64)))
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum CognitiveError {
    #[error("Memory region not found: {0}")]
    MemoryRegionNotFound(Uuid),
    #[error("Region is quarantined: {0}")]
    RegionQuarantined(Uuid),
    #[error("Write access denied: agent {agent_id} -> region {region_id}")]
    WriteAccessDenied { agent_id: Uuid, region_id: Uuid },
    #[error("Read access denied: agent {agent_id} -> region {region_id}")]
    ReadAccessDenied { agent_id: Uuid, region_id: Uuid },
    #[error("Reasoning chain not found: {0}")]
    ReasoningChainNotFound(Uuid),
    #[error("Orphan reasoning chain: {0}")]
    OrphanChain(Uuid),
    #[error("Cross-chain contamination detected")]
    CrossChainContamination,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_isolation_write_read() {
        let mut cognitive = CognitiveIsolationLayer::new();
        let agent_id = Uuid::new_v4();
        let region_id = cognitive.create_memory_region(agent_id);

        cognitive.write_memory(region_id, agent_id, "secret", b"data".to_vec()).unwrap();
        let read = cognitive.read_memory(region_id, agent_id, "secret").unwrap();
        assert_eq!(read, Some(b"data".to_vec()));
    }

    #[test]
    fn test_other_agent_cannot_read() {
        let mut cognitive = CognitiveIsolationLayer::new();
        let agent_a = Uuid::new_v4();
        let agent_b = Uuid::new_v4();
        let region_id = cognitive.create_memory_region(agent_a);

        let result = cognitive.read_memory(region_id, agent_b, "secret");
        assert!(result.is_err());
    }

    #[test]
    fn test_anomaly_auto_quarantine() {
        let mut cognitive = CognitiveIsolationLayer::new();
        let agent_id = Uuid::new_v4();
        let region_id = cognitive.create_memory_region(agent_id);

        for i in 0..5 {
            cognitive.detect_anomaly(agent_id, 0.9, &format!("anomaly {i}"));
        }

        let result = cognitive.write_memory(region_id, agent_id, "test", b"data".to_vec());
        assert!(result.is_err());
        assert!(matches!(result, Err(CognitiveError::RegionQuarantined(_))));
    }
}
