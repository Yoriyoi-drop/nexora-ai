use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::{
    model_identity::NxrModelId,
    capability_spec::{CapabilityDomain, CapabilityLevel, CapabilityVector},
    base_model::NxrModelError,
};

static LOCKED_CAPABILITIES: std::sync::OnceLock<Vec<(CapabilityDomain, CapabilityLevel)>> = std::sync::OnceLock::new();

fn locked_capabilities() -> &'static Vec<(CapabilityDomain, CapabilityLevel)> {
    LOCKED_CAPABILITIES.get_or_init(|| {
        vec![
            (CapabilityDomain::Alignment, CapabilityLevel::Basic),
            (CapabilityDomain::Security, CapabilityLevel::Basic),
        ]
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyGate {
    pub gate_id: Uuid,
    pub alignment_enabled: bool,
    pub sandbox_enabled: bool,
    pub consent_required: bool,
    pub audit_enabled: bool,
    pub feature_gates: HashMap<String, bool>,
}

impl Default for SafetyGate {
    fn default() -> Self {
        Self {
            gate_id: Uuid::new_v4(),
            alignment_enabled: true,
            sandbox_enabled: true,
            consent_required: true,
            audit_enabled: true,
            feature_gates: HashMap::from([
                ("sparo_alignment".to_string(), true),
                ("sandbox_isolation".to_string(), true),
                ("consent_layer".to_string(), true),
                ("audit_trail".to_string(), true),
                ("capability_locks".to_string(), true),
            ]),
        }
    }
}

impl SafetyGate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn check_alignment(&self) -> Result<(), NxrModelError> {
        if !self.alignment_enabled {
            return Err(NxrModelError::Configuration(
                "SPARO alignment gate is disabled - all models require alignment".to_string()
            ));
        }
        Ok(())
    }

    pub fn check_sandbox(&self) -> Result<(), NxrModelError> {
        if !self.sandbox_enabled {
            return Err(NxrModelError::Configuration(
                "Sandbox isolation is disabled - Cipher and Genesis require sandbox".to_string()
            ));
        }
        Ok(())
    }

    pub fn check_consent(&self, consent_token: Option<&str>) -> Result<(), NxrModelError> {
        if !self.consent_required {
            return Ok(());
        }
        match consent_token {
            Some(token) if !token.is_empty() => Ok(()),
            _ => Err(NxrModelError::Configuration(
                "Consent token required for psychological profiling".to_string()
            )),
        }
    }

    pub fn check_audit(&self) -> Result<(), NxrModelError> {
        if !self.audit_enabled {
            return Err(NxrModelError::Configuration(
                "Audit trail is disabled - multi-agent decisions require audit".to_string()
            ));
        }
        Ok(())
    }

    pub fn with_feature(mut self, key: &str, enabled: bool) -> Self {
        self.feature_gates.insert(key.to_string(), enabled);
        self
    }

    pub fn is_feature_enabled(&self, key: &str) -> bool {
        self.feature_gates.get(key).copied().unwrap_or(false)
    }

    pub fn required_models() -> Vec<NxrModelId> {
        vec![NxrModelId::Cipher, NxrModelId::Genesis]
    }

    pub fn sandbox_models() -> Vec<NxrModelId> {
        vec![NxrModelId::Cipher, NxrModelId::Genesis]
    }

    pub fn consent_models() -> Vec<NxrModelId> {
        vec![NxrModelId::Aether]
    }

    pub fn audit_models() -> Vec<NxrModelId> {
        vec![NxrModelId::Nexum]
    }
}

pub struct CapabilityLock {
    locked: Arc<AtomicBool>,
    locked_capabilities: Arc<RwLock<HashMap<CapabilityDomain, CapabilityLevel>>>,
}

impl CapabilityLock {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        for (domain, level) in locked_capabilities() {
            map.insert(domain.clone(), *level);
        }
        Self {
            locked: Arc::new(AtomicBool::new(true)),
            locked_capabilities: Arc::new(RwLock::new(map)),
        }
    }

    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::SeqCst)
    }

    pub async fn enforce(&self, vector: &mut CapabilityVector) -> Result<(), NxrModelError> {
        let locked = self.locked_capabilities.read().await;
        for (domain, min_level) in locked.iter() {
            if let Some(spec) = vector.get_capability(domain) {
                if !spec.meets_minimum(*min_level) {
                    return Err(NxrModelError::Configuration(
                        format!("Capability {:?} cannot be below {:?} - this is a locked safety requirement", domain, min_level)
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for CapabilityLock {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentToken {
    pub token: String,
    pub model_id: NxrModelId,
    pub scope: ConsentScope,
    pub issued_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentScope {
    PsychologicalProfiling,
    EmotionalAnalysis,
    BehavioralTracking,
    All,
}

impl ConsentToken {
    pub fn is_valid(&self) -> bool {
        chrono::Utc::now() < self.expires_at
    }

    pub fn covers_scope(&self, required: &ConsentScope) -> bool {
        matches!(self.scope, ConsentScope::All) || std::mem::discriminant(&self.scope) == std::mem::discriminant(required)
    }
}

pub struct AuditTrail {
    enabled: Arc<AtomicBool>,
    entries: Arc<RwLock<Vec<AuditEntry>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub model_id: NxrModelId,
    pub action: String,
    pub actor: String,
    pub decision: String,
    pub metadata: HashMap<String, String>,
}

impl AuditTrail {
    pub fn new() -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(true)),
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    pub async fn record(&self, model_id: NxrModelId, action: &str, actor: &str, decision: &str) {
        if !self.is_enabled() {
            return;
        }
        let entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            model_id,
            action: action.to_string(),
            actor: actor.to_string(),
            decision: decision.to_string(),
            metadata: HashMap::new(),
        };
        self.entries.write().await.push(entry);
    }

    pub async fn record_with_metadata(
        &self,
        model_id: NxrModelId,
        action: &str,
        actor: &str,
        decision: &str,
        metadata: HashMap<String, String>,
    ) {
        if !self.is_enabled() {
            return;
        }
        let entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            model_id,
            action: action.to_string(),
            actor: actor.to_string(),
            decision: decision.to_string(),
            metadata,
        };
        self.entries.write().await.push(entry);
    }

    pub async fn recent_entries(&self, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries.iter().rev().take(count).cloned().collect()
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GlobalSafetyGate {
    pub gate: SafetyGate,
    pub cap_lock: CapabilityLock,
    pub audit: AuditTrail,
}

impl GlobalSafetyGate {
    pub fn new() -> Self {
        Self {
            gate: SafetyGate::new(),
            cap_lock: CapabilityLock::new(),
            audit: AuditTrail::new(),
        }
    }

    pub async fn enforce_all(&self, model_id: NxrModelId, capabilities: &mut CapabilityVector) -> Result<(), NxrModelError> {
        self.gate.check_alignment()?;

        if SafetyGate::sandbox_models().contains(&model_id) {
            self.gate.check_sandbox()?;
        }

        self.cap_lock.enforce(capabilities).await?;

        Ok(())
    }

    pub async fn pre_inference_check(&self, model_id: NxrModelId, consent: Option<&str>) -> Result<(), NxrModelError> {
        self.gate.check_alignment()?;

        if SafetyGate::sandbox_models().contains(&model_id) {
            self.gate.check_sandbox()?;
        }

        if SafetyGate::consent_models().contains(&model_id) {
            self.gate.check_consent(consent)?;
        }

        if SafetyGate::audit_models().contains(&model_id) {
            self.gate.check_audit()?;
        }

        Ok(())
    }
}

impl Default for GlobalSafetyGate {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL_SAFETY: std::sync::OnceLock<Arc<GlobalSafetyGate>> = std::sync::OnceLock::new();

pub fn global_safety() -> Arc<GlobalSafetyGate> {
    GLOBAL_SAFETY.get_or_init(|| Arc::new(GlobalSafetyGate::new())).clone()
}
