use regex::Regex;

use crate::ModelTier;

/// Rule-based classifier — tentukan apakah request bisa di-handle tanpa LLM
pub struct RuleEngine {
    rules: Vec<CascadeRule>,
}

pub struct CascadeRule {
    pub name: &'static str,
    pub pattern: Regex,
    pub action: RuleAction,
    pub tier: ModelTier,
}

pub enum RuleAction {
    /// Langsung return response tanpa LLM
    DirectResponse(&'static str),
    /// Route ke tier tertentu
    RouteToTier(ModelTier),
    /// Fallthrough ke tier berikutnya
    Fallthrough,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
        }
    }

    fn default_rules() -> Vec<CascadeRule> {
        vec![
            CascadeRule {
                name: "greeting",
                pattern: Regex::new(r"^(hi|hello|hey|halo|hai)\s*$").unwrap(),
                action: RuleAction::DirectResponse("Hello! How can I help you today?"),
                tier: ModelTier::RuleEngine,
            },
            CascadeRule {
                name: "thanks",
                pattern: Regex::new(r"(thanks|thank you|terima kasih|makasih)").unwrap(),
                action: RuleAction::DirectResponse("You're welcome! Happy to help."),
                tier: ModelTier::RuleEngine,
            },
            CascadeRule {
                name: "goodbye",
                pattern: Regex::new(r"(bye|goodbye|see you|dadah)").unwrap(),
                action: RuleAction::DirectResponse("Goodbye! Have a great day!"),
                tier: ModelTier::RuleEngine,
            },
            CascadeRule {
                name: "yes_no_simple",
                pattern: Regex::new(r"^(yes|no|yep|nope|ya|tidak)$").unwrap(),
                action: RuleAction::RouteToTier(ModelTier::Small),
                tier: ModelTier::RuleEngine,
            },
            CascadeRule {
                name: "math_simple",
                pattern: Regex::new(r"^\d+\s*[\+\-\*\/]\s*\d+$").unwrap(),
                action: RuleAction::RouteToTier(ModelTier::Small),
                tier: ModelTier::RuleEngine,
            },
        ]
    }

    pub fn evaluate(&self, input: &str) -> Option<RuleResult> {
        let lower = input.to_lowercase();
        for rule in &self.rules {
            if rule.pattern.is_match(&lower) {
                return Some(RuleResult {
                    matched_rule: rule.name,
                    tier: rule.tier,
                });
            }
        }
        None
    }
}

pub struct RuleResult {
    pub matched_rule: &'static str,
    pub tier: ModelTier,
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}
