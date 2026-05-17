use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use uuid::Uuid;

use crate::layer1_mode::ModeId;
use crate::layer6_permission::Capability;

static REGEX_CACHE: LazyLock<std::sync::Mutex<HashMap<String, Regex>>> = LazyLock::new(|| {
    std::sync::Mutex::new(HashMap::new())
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterAgentFirewall {
    pub enabled: bool,
    pub rules: Vec<FirewallRule>,
    pub default_egress: FirewallAction,
    pub default_ingress: FirewallAction,
    pub rate_limiter: FirewallRateLimiter,
    pub audit_log: Vec<FirewallAuditEntry>,
    pub suspicious_patterns: Vec<SuspiciousPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    pub id: Uuid,
    pub priority: u32,
    pub source: FirewallMatch,
    pub destination: FirewallMatch,
    pub action: FirewallAction,
    pub log: bool,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FirewallMatch {
    Any,
    AgentId(Uuid),
    AgentType(String),
    Mode(ModeId),
    ModeKind(String),
    Capability(Capability),
    Label(String),
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FirewallAction {
    Allow,
    Deny,
    AuditAllow,
    AuditDeny,
    RateLimit(u32),
    QuarantineSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRateLimiter {
    pub max_messages_per_agent: u32,
    pub window_seconds: u64,
    pub per_agent_counts: HashMap<Uuid, u32>,
    pub cooldown_expiry: HashMap<Uuid, chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallAuditEntry {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source_id: Uuid,
    pub source_label: String,
    pub destination_id: Uuid,
    pub destination_label: String,
    pub message_type: String,
    pub action_taken: FirewallAction,
    pub reason: String,
    pub message_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousPattern {
    pub name: String,
    pub pattern: String,
    pub severity: Severity,
    pub auto_block: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBusMessage {
    pub id: Uuid,
    pub source_id: Uuid,
    pub destination_id: Uuid,
    pub message_type: AgentBusMsgType,
    pub payload: Vec<u8>,
    pub size_bytes: u64,
    pub priority: MessagePriority,
    pub ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentBusMsgType {
    Query,
    Response,
    Command,
    Event,
    MemoryAccess,
    ToolRequest,
    StatusUpdate,
    Heartbeat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

pub type SharedFirewall = Arc<RwLock<InterAgentFirewall>>;
pub type SharedAgentBus = Arc<RwLock<AgentBus>>;

pub struct AgentBus {
    messages: Vec<AgentBusMessage>,
    max_history: usize,
}

impl AgentBus {
    pub fn new(max_history: usize) -> Self {
        Self { messages: Vec::with_capacity(max_history), max_history }
    }

    pub fn send(&mut self, msg: AgentBusMessage) {
        self.messages.push(msg);
        if self.messages.len() > self.max_history {
            self.messages.remove(0);
        }
    }

    pub fn get_history(&self) -> &[AgentBusMessage] {
        &self.messages
    }

    pub fn get_messages_for_agent(&self, agent_id: Uuid) -> Vec<&AgentBusMessage> {
        self.messages.iter()
            .filter(|m| m.destination_id == agent_id)
            .collect()
    }

    pub fn get_messages_from_agent(&self, agent_id: Uuid) -> Vec<&AgentBusMessage> {
        self.messages.iter()
            .filter(|m| m.source_id == agent_id)
            .collect()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl InterAgentFirewall {
    pub fn new() -> Self {
        Self {
            enabled: true,
            rules: vec![
                FirewallRule {
                    id: Uuid::new_v4(),
                    priority: 9999,
                    source: FirewallMatch::Any,
                    destination: FirewallMatch::Any,
                    action: FirewallAction::AuditDeny,
                    log: true,
                    description: "Default deny-all with audit".into(),
                },
            ],
            default_egress: FirewallAction::AuditDeny,
            default_ingress: FirewallAction::Deny,
            rate_limiter: FirewallRateLimiter {
                max_messages_per_agent: 100,
                window_seconds: 60,
                per_agent_counts: HashMap::new(),
                cooldown_expiry: HashMap::new(),
            },
            audit_log: Vec::new(),
            suspicious_patterns: vec![
                SuspiciousPattern {
                    name: "memory-overwrite".into(),
                    pattern: r"overwrite.*memory|force.*write|corrupt.*buffer".into(),
                    severity: Severity::Critical,
                    auto_block: true,
                },
                SuspiciousPattern {
                    name: "privilege-escalation".into(),
                    pattern: r"escalate|privilege.*raise|bypass.*restrict".into(),
                    severity: Severity::Critical,
                    auto_block: true,
                },
                SuspiciousPattern {
                    name: "cross-agent-spy".into(),
                    pattern: r"read.*other.*agent|intercept|sniff.*message".into(),
                    severity: Severity::Warning,
                    auto_block: true,
                },
            ],
        }
    }

    pub fn evaluate(
        &mut self,
        source_id: Uuid,
        source_label: &str,
        dest_id: Uuid,
        dest_label: &str,
        msg_type: &str,
        payload: &[u8],
    ) -> FirewallAction {
        if !self.enabled {
            return FirewallAction::Allow;
        }

        if let Some(expiry) = self.rate_limiter.cooldown_expiry.get(&source_id) {
            if chrono::Utc::now() < *expiry {
                self.audit_log.push(FirewallAuditEntry {
                    id: Uuid::new_v4(),
                    timestamp: chrono::Utc::now(),
                    source_id,
                    source_label: source_label.to_string(),
                    destination_id: dest_id,
                    destination_label: dest_label.to_string(),
                    message_type: msg_type.to_string(),
                    action_taken: FirewallAction::QuarantineSource,
                    reason: "Agent in cooldown".into(),
                    message_size_bytes: payload.len() as u64,
                });
                return FirewallAction::Deny;
            }
        }

        let count = self.rate_limiter.per_agent_counts.entry(source_id).or_insert(0);
        *count += 1;
        if *count > self.rate_limiter.max_messages_per_agent {
            self.rate_limiter.cooldown_expiry.insert(
                source_id,
                chrono::Utc::now() + chrono::Duration::seconds(self.rate_limiter.window_seconds as i64),
            );
        }

        let payload_str = String::from_utf8_lossy(payload);
        let mut cache = REGEX_CACHE.lock().unwrap();
        for pattern in &self.suspicious_patterns {
            let re = cache.entry(pattern.pattern.clone())
                .or_insert_with(|| Regex::new(&pattern.pattern).expect("Invalid regex pattern"));
            if re.is_match(&payload_str) {
                if pattern.auto_block {
                    self.audit_log.push(FirewallAuditEntry {
                        id: Uuid::new_v4(),
                        timestamp: chrono::Utc::now(),
                        source_id,
                        source_label: source_label.to_string(),
                        destination_id: dest_id,
                        destination_label: dest_label.to_string(),
                        message_type: msg_type.to_string(),
                        action_taken: FirewallAction::Deny,
                        reason: format!("Suspicious pattern: {}", pattern.name),
                        message_size_bytes: payload.len() as u64,
                    });
                    return FirewallAction::Deny;
                }
            }
        }

        for rule in &self.rules {
            if matches_source(&rule.source, source_id, source_label)
                && matches_destination(&rule.destination, dest_id, dest_label)
            {
                if rule.log {
                    self.audit_log.push(FirewallAuditEntry {
                        id: Uuid::new_v4(),
                        timestamp: chrono::Utc::now(),
                        source_id,
                        source_label: source_label.to_string(),
                        destination_id: dest_id,
                        destination_label: dest_label.to_string(),
                        message_type: msg_type.to_string(),
                        action_taken: rule.action.clone(),
                        reason: format!("Rule match: {}", rule.description),
                        message_size_bytes: payload.len() as u64,
                    });
                }
                return rule.action.clone();
            }
        }

        self.default_egress.clone()
    }

    pub fn add_rule(&mut self, rule: FirewallRule) {
        self.rules.push(rule);
        self.rules.sort_by(|a, b| a.priority.cmp(&b.priority));
    }

    pub fn add_allow_rule(
        &mut self,
        source_label: &str,
        dest_label: &str,
        description: &str,
    ) {
        self.add_rule(FirewallRule {
            id: Uuid::new_v4(),
            priority: 100,
            source: FirewallMatch::Label(source_label.to_string()),
            destination: FirewallMatch::Label(dest_label.to_string()),
            action: FirewallAction::AuditAllow,
            log: true,
            description: description.to_string(),
        });
    }

    pub fn block_agent(&mut self, agent_id: Uuid) {
        self.add_rule(FirewallRule {
            id: Uuid::new_v4(),
            priority: 10,
            source: FirewallMatch::AgentId(agent_id),
            destination: FirewallMatch::Any,
            action: FirewallAction::Deny,
            log: true,
            description: format!("Agent {agent_id} blocked by kill-switch"),
        });
    }

    pub fn get_audit_log(&self) -> &[FirewallAuditEntry] {
        &self.audit_log
    }

    pub fn get_recent_audit(&self, count: usize) -> Vec<&FirewallAuditEntry> {
        self.audit_log.iter().rev().take(count).collect()
    }
}

fn matches_source(mat: &FirewallMatch, source_id: Uuid, source_label: &str) -> bool {
    match mat {
        FirewallMatch::Any => true,
        FirewallMatch::AgentId(id) => *id == source_id,
        FirewallMatch::Label(label) => source_label.contains(label.as_str()),
        _ => false,
    }
}

fn matches_destination(mat: &FirewallMatch, dest_id: Uuid, dest_label: &str) -> bool {
    match mat {
        FirewallMatch::Any => true,
        FirewallMatch::AgentId(id) => *id == dest_id,
        FirewallMatch::Label(label) => dest_label.contains(label.as_str()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_firewall_default_deny() {
        let mut fw = InterAgentFirewall::new();
        let src = Uuid::new_v4();
        let dst = Uuid::new_v4();
        let action = fw.evaluate(src, "agent:research:oracle", dst, "agent:defense:sentinel", "query", b"hello");
        assert!(matches!(action, FirewallAction::AuditDeny | FirewallAction::Deny));
    }

    #[test]
    fn test_firewall_allow_rule() {
        let mut fw = InterAgentFirewall::new();
        fw.add_allow_rule("oracle", "code-sentinel", "Oracle can talk to Code Sentinel");
        let src = Uuid::new_v4();
        let dst = Uuid::new_v4();
        let action = fw.evaluate(src, "agent:research:oracle", dst, "agent:coding:code-sentinel", "query", b"hello");
        assert!(matches!(action, FirewallAction::AuditAllow | FirewallAction::Allow));
    }

    #[test]
    fn test_suspicious_pattern_blocked() {
        let mut fw = InterAgentFirewall::new();
        let src = Uuid::new_v4();
        let dst = Uuid::new_v4();
        let action = fw.evaluate(src, "agent:rogue", dst, "agent:victim", "query", b"overwrite memory now");
        assert!(matches!(action, FirewallAction::Deny));
    }
}
