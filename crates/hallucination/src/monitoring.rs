use crate::types::{AuditEntry, RiskLevel};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

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
    audit_log: Mutex<VecDeque<AuditEntry>>,
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
            audit_log: Mutex::new(VecDeque::with_capacity(max_audit_log)),
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
            "Blocked" => {
                self.total_blocked.fetch_add(1, Ordering::Relaxed);
            }
            "FlagForReview" => {
                self.total_flagged.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.total_passed.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.total_checked.fetch_add(1, Ordering::Relaxed);
        self.log_entry(entry);
    }

    pub fn log_entry(&self, entry: AuditEntry) {
        let mut log = self.audit_log.lock().unwrap();
        if log.len() >= self.config.max_audit_log {
            log.pop_front();
        }
        log.push_back(entry);
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
            "audit_log_size": self.audit_log.lock().unwrap().len(),
        })
    }

    pub fn hallucination_rate(&self) -> f64 {
        let total = self.total_checked.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (self.total_blocked.load(Ordering::Relaxed) + self.total_flagged.load(Ordering::Relaxed))
            as f64
            / total as f64
    }

    pub fn is_alerting(&self) -> bool {
        self.hallucination_rate() > self.config.alert_threshold as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> Monitor {
        Monitor::new(MonitorConfig {
            max_audit_log: 10,
            enable_metrics: true,
            alert_threshold: 0.5,
        })
    }

    #[test]
    fn test_initial_stats_zero() {
        let m = monitor();
        let stats = m.get_stats();
        assert_eq!(stats["total_checked"], 0);
        assert_eq!(stats["total_blocked"], 0);
        assert_eq!(stats["total_flagged"], 0);
        assert_eq!(stats["total_passed"], 0);
    }

    #[test]
    fn test_record_blocked() {
        let m = monitor();
        m.record("Blocked", "bad input");
        let stats = m.get_stats();
        assert_eq!(stats["total_checked"], 1);
        assert_eq!(stats["total_blocked"], 1);
        assert_eq!(stats["total_passed"], 0);
    }

    #[test]
    fn test_record_flagged() {
        let m = monitor();
        m.record("FlagForReview", "suspicious input");
        let stats = m.get_stats();
        assert_eq!(stats["total_checked"], 1);
        assert_eq!(stats["total_flagged"], 1);
    }

    #[test]
    fn test_record_passed() {
        let m = monitor();
        m.record("Pass", "good input");
        m.record("PassWithDisclaimer", "ok input");
        let stats = m.get_stats();
        assert_eq!(stats["total_checked"], 2);
        assert_eq!(stats["total_passed"], 2);
    }

    #[test]
    fn test_record_mixed_counts() {
        let m = monitor();
        m.record("Blocked", "a");
        m.record("FlagForReview", "b");
        m.record("Pass", "c");
        m.record("Pass", "d");
        let stats = m.get_stats();
        assert_eq!(stats["total_checked"], 4);
        assert_eq!(stats["total_blocked"], 1);
        assert_eq!(stats["total_flagged"], 1);
        assert_eq!(stats["total_passed"], 2);
    }

    #[test]
    fn test_block_rate() {
        let m = monitor();
        m.record("Blocked", "a");
        m.record("Pass", "b");
        let stats = m.get_stats();
        assert!((stats["block_rate"].as_f64().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_block_rate_zero_when_empty() {
        let m = monitor();
        let stats = m.get_stats();
        assert_eq!(stats["block_rate"], 0.0);
    }

    #[test]
    fn test_hallucination_rate() {
        let m = monitor();
        m.record("Blocked", "a");
        m.record("FlagForReview", "b");
        m.record("Pass", "c");
        assert!((m.hallucination_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hallucination_rate_zero_when_empty() {
        let m = monitor();
        assert_eq!(m.hallucination_rate(), 0.0);
    }

    #[test]
    fn test_is_alerting_when_above_threshold() {
        let m = monitor();
        m.record("Blocked", "a");
        m.record("Blocked", "b");
        // 2 blocked / 2 total = 1.0, threshold = 0.5
        assert!(m.is_alerting());
    }

    #[test]
    fn test_is_alerting_when_below_threshold() {
        let m = monitor();
        m.record("Pass", "a");
        m.record("Pass", "b");
        m.record("Blocked", "c");
        // 1 blocked / 3 total = 0.33, threshold = 0.5
        assert!(!m.is_alerting());
    }

    #[test]
    fn test_record_truncates_long_input() {
        let m = monitor();
        let long_input = "a".repeat(500);
        m.record("Pass", &long_input);
        // If no panic, test passes
        let stats = m.get_stats();
        assert_eq!(stats["total_checked"], 1);
    }

    #[test]
    fn test_audit_log_size_capped() {
        let config = MonitorConfig {
            max_audit_log: 5,
            enable_metrics: true,
            alert_threshold: 0.9,
        };
        let m = Monitor::new(config);
        for i in 0..10 {
            m.record("Pass", &format!("input {}", i));
        }
        let stats = m.get_stats();
        assert_eq!(stats["audit_log_size"], 5);
    }
}
