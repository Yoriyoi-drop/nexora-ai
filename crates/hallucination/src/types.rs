use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    pub fn as_f32(&self) -> f32 {
        match self {
            RiskLevel::Low => 0.0,
            RiskLevel::Medium => 0.3,
            RiskLevel::High => 0.6,
            RiskLevel::Critical => 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClaimType {
    Fact,
    Number,
    Date,
    Name,
    Quote,
    Reference,
    Opinion,
    Uncertainty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    pub claim_type: ClaimType,
    pub specificity: f32,
    pub is_verified: Option<bool>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreGenCheckResult {
    pub can_proceed: bool,
    pub in_scope: bool,
    pub ambiguity_score: f32,
    pub context_sufficiency: f32,
    pub reason: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InGenCheckResult {
    pub uncertainty_score: f32,
    pub enhanced_prompt: String,
    pub requires_cot: bool,
    pub knowledge_boundary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostGenCheckResult {
    pub internal_consistency: f32,
    pub source_grounding: f32,
    pub high_risk_sentences: Vec<String>,
    pub contradiction_count: usize,
    pub total_claims: usize,
    pub verified_claims: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskScore {
    pub total: f32,
    pub specificity_score: f32,
    pub domain_niche_score: f32,
    pub citation_score: f32,
    pub contradiction_score: f32,
    pub recency_score: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub input: String,
    pub risk_level: RiskLevel,
    pub score: f32,
    pub action_taken: String,
    pub latency_ms: u64,
    pub claims_found: usize,
    pub contradictions: usize,
}
