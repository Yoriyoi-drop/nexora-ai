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

impl Claim {
    pub fn new(text: String, specificity: f32, source: Option<String>) -> Self {
        let claim_type = Self::classify(&text);
        Self {
            text,
            claim_type,
            specificity,
            is_verified: None,
            source,
        }
    }

    pub fn with_verification(mut self, verified: bool) -> Self {
        self.is_verified = Some(verified);
        self
    }

    fn classify(text: &str) -> ClaimType {
        let lower = text.to_lowercase();

        if (text.contains('\'') || text.contains('"'))
            && (lower.contains("said")
                || lower.contains("stated")
                || lower.contains("wrote")
                || lower.contains("according"))
        {
            return ClaimType::Quote;
        }

        if lower.contains("probably")
            || lower.contains("maybe")
            || lower.contains("might")
            || lower.contains("could")
            || lower.contains("uncertain")
            || lower.contains("unclear")
        {
            return ClaimType::Uncertainty;
        }

        if lower.contains("think")
            || lower.contains("believe")
            || lower.contains("feel")
            || lower.contains("opinion")
            || lower.contains("in my view")
        {
            return ClaimType::Opinion;
        }

        if lower.contains("according to")
            || lower.contains("source")
            || lower.contains("reference")
            || lower.contains("cited")
            || lower.contains("reported by")
        {
            return ClaimType::Reference;
        }

        if text.chars().any(|c| c.is_ascii_digit())
            && (text.contains('%')
                || text.contains("percent")
                || text.chars().filter(|c| c.is_ascii_digit()).count() >= 3)
        {
            return ClaimType::Number;
        }

        let month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ];
        if month_names.iter().any(|m| lower.contains(m))
            || text.chars().filter(|&c| c == '/').count() >= 2
        {
            return ClaimType::Date;
        }

        if text.split_whitespace().any(|w| {
            w.len() >= 2
                && w.chars()
                    .all(|c| c.is_ascii_uppercase() && c.is_ascii_alphabetic())
        }) {
            return ClaimType::Name;
        }

        ClaimType::Fact
    }
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
