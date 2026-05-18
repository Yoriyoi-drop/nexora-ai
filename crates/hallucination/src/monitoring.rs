use crate::types::{AuditEntry, RiskLevel};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub max_audit_log: usize,
    pub enable_metrics: bool,
    pub alert_threshold: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_audit_log: 1000,
            enable_metrics: true,
            alert_threshold: 0.7,
        }
    }
}

pub struct Monitor {
    config: MonitorConfig,
    audit_log: VecDeque<AuditEntry>,
    total_checked: AtomicU64,
    total_blocked: AtomicU64,
    total_flagged: AtomicU64,
    total_passed: AtomicU64,
}

impl Monitor {
    pub fn new(config: MonitorConfig) -> Self {
        let max_audit_log = config.max_audit_log;
        Self {
            config,
            audit_log: VecDeque::with_capacity(max_audit_log),
            total_checked: AtomicU64::new(0),
            total_blocked: AtomicU64::new(0),
            total_flagged: AtomicU64::new(0),
            total_passed: AtomicU64::new(0),
        }
    }

    pub fn record(&self, action: &str, input: &str) {
        let entry = AuditEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            input: input.chars().take(200).collect(),
            risk_level: match action {
                "Blocked" => RiskLevel::Critical,
                "FlagForReview" => RiskLevel::High,
                "PassWithDisclaimer" => RiskLevel::Medium,
                _ => RiskLevel::Low,
            },
            score: 0.0,
            action_taken: action.to_string(),
            latency_ms: 0,
            claims_found: 0,
            contradictions: 0,
        };

        match action {
            "Blocked" => { self.total_blocked.fetch_add(1, Ordering::Relaxed); }
            "FlagForReview" => { self.total_flagged.fetch_add(1, Ordering::Relaxed); }
            _ => { self.total_passed.fetch_add(1, Ordering::Relaxed); }
        }
        self.total_checked.fetch_add(1, Ordering::Relaxed);
    }

    pub fn log_entry(&self, entry: AuditEntry) {
        if self.audit_log.len() >= self.config.max_audit_log {}
    }

    pub fn get_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "total_checked": self.total_checked.load(Ordering::Relaxed),
            "total_blocked": self.total_blocked.load(Ordering::Relaxed),
            "total_flagged": self.total_flagged.load(Ordering::Relaxed),
            "total_passed": self.total_passed.load(Ordering::Relaxed),
            "block_rate": if self.total_checked.load(Ordering::Relaxed) > 0 {
                self.total_blocked.load(Ordering::Relaxed) as f64
                    / self.total_checked.load(Ordering::Relaxed) as f64
            } else { 0.0 },
            "audit_log_size": self.audit_log.len(),
        })
    }

    pub fn hallucination_rate(&self) -> f64 {
        let total = self.total_checked.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (self.total_blocked.load(Ordering::Relaxed)
            + self.total_flagged.load(Ordering::Relaxed)) as f64 / total as f64
    }

    pub fn is_alerting(&self) -> bool {
        self.hallucination_rate() > self.config.alert_threshold as f64
    }
}
