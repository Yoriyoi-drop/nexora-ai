//! NXR Model Identity System
//! 
//! Defines identity, codename, tier, and metadata for all NXR models

use serde::{Deserialize, Serialize};
use std::fmt::Display;
use uuid::Uuid;

/// NXR Model Series Identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NxrModelId {
    /// NXR-01 ULTRA - Omniscient Reasoning System
    Omnis,
    /// NXR-02 APEX - Code Specialized
    Vortex,
    /// NXR-03 APEX - Emotional Intelligence
    Aether,
    /// NXR-04 PRO - Multimodal Creative
    Spectra,
    /// NXR-05 APEX - Multi-Agent Orchestrator
    Nexum,
    /// NXR-06 ULTRA - Autonomous Decision Maker
    Axiom,
    /// NXR-07 PRO - Cybersecurity Specialist
    Cipher,
    /// NXR-08 EDGE - Ultra Lightweight
    Swift,
    /// NXR-09 CORE - Knowledge Management
    Kronos,
    /// NXR-10 ULTRA - Self-Improving Prototype
    Genesis,
}

impl Display for NxrModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NxrModelId::Omnis => write!(f, "NXR-OMNIS"),
            NxrModelId::Vortex => write!(f, "NXR-VORTEX"),
            NxrModelId::Aether => write!(f, "NXR-ÆTHER"),
            NxrModelId::Spectra => write!(f, "NXR-SPECTRA"),
            NxrModelId::Nexum => write!(f, "NXR-NEXUM"),
            NxrModelId::Axiom => write!(f, "NXR-AXIOM"),
            NxrModelId::Cipher => write!(f, "NXR-CIPHER"),
            NxrModelId::Swift => write!(f, "NXR-SWIFT"),
            NxrModelId::Kronos => write!(f, "NXR-KRONOS"),
            NxrModelId::Genesis => write!(f, "NXR-GENESIS"),
        }
    }
}

impl NxrModelId {
    /// Get model number (1-10)
    pub fn number(&self) -> u8 {
        match self {
            NxrModelId::Omnis => 1,
            NxrModelId::Vortex => 2,
            NxrModelId::Aether => 3,
            NxrModelId::Spectra => 4,
            NxrModelId::Nexum => 5,
            NxrModelId::Axiom => 6,
            NxrModelId::Cipher => 7,
            NxrModelId::Swift => 8,
            NxrModelId::Kronos => 9,
            NxrModelId::Genesis => 10,
        }
    }

    /// Get model codename
    pub fn codename(&self) -> &'static str {
        match self {
            NxrModelId::Omnis => "OMNIS",
            NxrModelId::Vortex => "VORTEX",
            NxrModelId::Aether => "ÆTHER",
            NxrModelId::Spectra => "SPECTRA",
            NxrModelId::Nexum => "NEXUM",
            NxrModelId::Axiom => "AXIOM",
            NxrModelId::Cipher => "CIPHER",
            NxrModelId::Swift => "SWIFT",
            NxrModelId::Kronos => "KRONOS",
            NxrModelId::Genesis => "GENESIS",
        }
    }

    /// Get full model name
    pub fn fullname(&self) -> &'static str {
        match self {
            NxrModelId::Omnis => "Nexus Omniscient Reasoning System",
            NxrModelId::Vortex => "Variable Optimization Recursive Text & Expert eXchange",
            NxrModelId::Aether => "Adaptive Emotional & Holistic Transcendent Empathy Reasoner",
            NxrModelId::Spectra => "Spectral Perception & Encoding for Creative Transcendence & Research Analytics",
            NxrModelId::Nexum => "Networked EXpert Unified Mediator",
            NxrModelId::Axiom => "Autonomous eXpert Intelligence for Operations & Management",
            NxrModelId::Cipher => "Cybersecurity Intelligence & Penetration Hardening Evaluation Responder",
            NxrModelId::Swift => "Sub-millisecond Weighted Inference & Fast Thought",
            NxrModelId::Kronos => "Knowledge Retrieval & Ontological Neural Optimization System",
            NxrModelId::Genesis => "Generative Evolution Network for Emergent Simulation & Intelligence Synthesis",
        }
    }

    /// Get all model IDs in order
    pub fn all() -> Vec<NxrModelId> {
        vec![
            NxrModelId::Omnis,
            NxrModelId::Vortex,
            NxrModelId::Aether,
            NxrModelId::Spectra,
            NxrModelId::Nexum,
            NxrModelId::Axiom,
            NxrModelId::Cipher,
            NxrModelId::Swift,
            NxrModelId::Kronos,
            NxrModelId::Genesis,
        ]
    }
}

/// Model Tier Classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    /// ULTRA - Flagship models with maximum capabilities
    Ultra,
    /// APEX - High-performance specialized models
    Apex,
    /// PRO - Professional-grade models for specific domains
    Pro,
    /// EDGE - Ultra-lightweight models for edge computing
    Edge,
    /// CORE - Foundational models for core infrastructure
    Core,
    /// MASTER - Master-level models with expert capabilities
    Master,
}

impl Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelTier::Ultra => write!(f, "ULTRA"),
            ModelTier::Apex => write!(f, "APEX"),
            ModelTier::Pro => write!(f, "PRO"),
            ModelTier::Edge => write!(f, "EDGE"),
            ModelTier::Core => write!(f, "CORE"),
            ModelTier::Master => write!(f, "MASTER"),
        }
    }
}

impl ModelTier {
    /// Get tier priority (higher = more capable)
    pub fn priority(&self) -> u8 {
        match self {
            ModelTier::Ultra => 6,
            ModelTier::Master => 5,
            ModelTier::Apex => 4,
            ModelTier::Pro => 3,
            ModelTier::Core => 2,
            ModelTier::Edge => 1,
        }
    }

    /// Get tier color for UI display
    pub fn color(&self) -> &'static str {
        match self {
            ModelTier::Ultra => "#ff6b35",
            ModelTier::Master => "#ff1493",
            ModelTier::Apex => "#ffd700",
            ModelTier::Pro => "#00d4ff",
            ModelTier::Edge => "#00ff88",
            ModelTier::Core => "#7b2fff",
        }
    }
}

/// Model Metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    /// Unique identifier
    pub id: NxrModelId,
    /// Tier classification
    pub tier: ModelTier,
    /// Model version
    pub version: String,
    /// Unique UUID
    pub uuid: Uuid,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Model description
    pub description: String,
    /// Parameter count (if known)
    pub parameter_count: Option<u64>,
    /// Context window size
    pub context_window: Option<usize>,
    /// Is this model experimental?
    pub experimental: bool,
    /// Model name
    pub name: String,
    /// Model ID string
    pub model_id: String,
    /// Model capabilities
    pub capabilities: Vec<String>,
    /// Model tags
    pub tags: Vec<String>,
}

impl ModelMeta {
    /// Create new model metadata
    pub fn new(id: NxrModelId, tier: ModelTier, version: String, description: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            tier,
            version,
            uuid: Uuid::new_v4(),
            created_at: now,
            updated_at: now,
            description,
            parameter_count: None,
            context_window: None,
            experimental: false,
            name: id.fullname().to_string(),
            model_id: format!("NXR-{:03}", id.number()),
            capabilities: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Update metadata timestamp
    pub fn touch(&mut self) {
        self.updated_at = chrono::Utc::now();
    }

    /// Set parameter count
    pub fn with_parameters(mut self, count: u64) -> Self {
        self.parameter_count = Some(count);
        self
    }

    /// Set context window
    pub fn with_context_window(mut self, size: usize) -> Self {
        self.context_window = Some(size);
        self
    }

    /// Mark as experimental
    pub fn experimental(mut self) -> Self {
        self.experimental = true;
        self
    }
    
    /// Get reference to self (for compatibility with existing code)
    pub fn meta(&self) -> &Self {
        self
    }
}

impl Default for ModelMeta {
    fn default() -> Self {
        Self::new(
            NxrModelId::Omnis,
            ModelTier::Ultra,
            "1.0.0".to_string(),
            "Default NXR Model".to_string(),
        )
    }
}

/// Get tier for each model
impl NxrModelId {
    pub fn tier(&self) -> ModelTier {
        match self {
            NxrModelId::Omnis => ModelTier::Ultra,
            NxrModelId::Vortex => ModelTier::Apex,
            NxrModelId::Aether => ModelTier::Apex,
            NxrModelId::Spectra => ModelTier::Pro,
            NxrModelId::Nexum => ModelTier::Apex,
            NxrModelId::Axiom => ModelTier::Ultra,
            NxrModelId::Cipher => ModelTier::Pro,
            NxrModelId::Swift => ModelTier::Edge,
            NxrModelId::Kronos => ModelTier::Core,
            NxrModelId::Genesis => ModelTier::Ultra,
        }
    }
}
