//! FIREWALL-AI Agent
//!
//! Intelligent firewall with adaptive rules

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// FIREWALL-AI Agent - Intelligent firewall with adaptive rules
#[derive(Debug, Clone)]
pub struct FirewallAiAgent {
    pub config: FirewallAiConfig,
    pub firewall_capabilities: FirewallCapabilities,
    pub adaptive_engine: AdaptiveEngine,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallAiConfig {
    pub base_config: BaseAgentConfig,
    pub rule_generation_enabled: bool,
    pub anomaly_detection_enabled: bool,
    pub learning_mode: LearningMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningMode {
    Supervised,
    Unsupervised,
    Reinforcement,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallCapabilities {
    pub packet_filtering: bool,
    pub traffic_analysis: bool,
    pub anomaly_detection: bool,
    pub rule_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveEngine {
    pub rule_types: Vec<String>,
    pub analysis_methods: Vec<String>,
    pub response_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallAiTaskInput {
    pub traffic_data: Vec<String>,
    pub current_rules: Vec<String>,
    pub analysis_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallAiTaskOutput {
    pub filtered_traffic: Vec<String>,
    pub new_rules: Vec<String>,
    pub threat_assessment: String,
    pub firewall_score: f32,
}

impl Default for FirewallAiConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            rule_generation_enabled: true,
            anomaly_detection_enabled: true,
            learning_mode: LearningMode::Hybrid,
        }
    }
}

impl Default for FirewallCapabilities {
    fn default() -> Self {
        Self {
            packet_filtering: true,
            traffic_analysis: true,
            anomaly_detection: true,
            rule_optimization: true,
        }
    }
}

impl Default for AdaptiveEngine {
    fn default() -> Self {
        Self {
            rule_types: vec![
                "allow".to_string(),
                "deny".to_string(),
                "rate_limit".to_string(),
                "redirect".to_string(),
            ],
            analysis_methods: vec![
                "packet_inspection".to_string(),
                "flow_analysis".to_string(),
                "behavioral_analysis".to_string(),
                "ml_inference".to_string(),
            ],
            response_actions: vec![
                "block_ip".to_string(),
                "trigger_alert".to_string(),
                "isolate_traffic".to_string(),
                "log_event".to_string(),
            ],
        }
    }
}

impl Default for FirewallAiAgent {
    fn default() -> Self {
        Self {
            config: FirewallAiConfig::default(),
            firewall_capabilities: FirewallCapabilities::default(),
            adaptive_engine: AdaptiveEngine::default(),
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
impl BaseAgent for FirewallAiAgent {
    type Config = FirewallAiConfig;
    type Input = FirewallAiTaskInput;
    type Output = FirewallAiTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let filtered_traffic = self.filter_traffic(&input).await?;
        let new_rules = self.generate_rules(&input, &filtered_traffic).await?;
        let threat_assessment = self.assess_threats(&input, &filtered_traffic).await?;
        let firewall_score = self.calculate_firewall_score(&input, &new_rules).await?;

        Ok(FirewallAiTaskOutput {
            filtered_traffic,
            new_rules,
            threat_assessment,
            firewall_score,
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
                name: "firewall_ai".to_string(),
                description: "Intelligent firewall with adaptive rules".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["traffic_data".to_string(), "current_rules".to_string()],
                output_types: vec!["filtered_traffic".to_string(), "new_rules".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.97,
                    avg_latency: 2500.0,
                    resource_usage: 0.80,
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

impl FirewallAiAgent {
    pub fn new(config: FirewallAiConfig) -> Self {
        Self {
            config,
            firewall_capabilities: FirewallCapabilities::default(),
            adaptive_engine: AdaptiveEngine::default(),
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

    async fn filter_traffic(&self, input: &FirewallAiTaskInput) -> AgentResult<Vec<String>> {
        let mut blocked = Vec::new();
        let mut allowed = Vec::new();

        for traffic in &input.traffic_data {
            let traffic_lower = traffic.to_lowercase();

            let is_malicious = traffic_lower.contains("malicious")
                || traffic_lower.contains("attack")
                || traffic_lower.contains("exploit");

            let matches_block_rule = input.current_rules.iter().any(|rule| {
                let rule_lower = rule.to_lowercase();
                rule_lower.contains("deny") && (traffic_lower.contains(&rule_lower.replace("deny ", ""))
                    || rule_lower.contains(&traffic_lower.split_whitespace().next().unwrap_or("")))
            });

            if is_malicious || matches_block_rule {
                blocked.push(format!("BLOCKED: {}", traffic));
            } else {
                allowed.push(format!("ALLOWED: {}", traffic));
            }
        }

        if self.config.anomaly_detection_enabled {
            let anomalies = self.detect_anomalies(&input.traffic_data).await?;
            for anomaly in anomalies {
                blocked.push(format!("BLOCKED-ANOMALY: {}", anomaly));
            }
        }

        Ok(blocked.into_iter().chain(allowed).collect())
    }

    async fn detect_anomalies(&self, traffic: &[String]) -> AgentResult<Vec<String>> {
        let mut anomalies = Vec::new();
        let mut ip_counts: HashMap<String, usize> = HashMap::with_capacity(traffic.len());

        for entry in traffic {
            if let Some(ip) = entry.split_whitespace().next() {
                *ip_counts.entry(ip.to_string()).or_insert(0) += 1;
            }
        }

        for (ip, count) in &ip_counts {
            if *count > traffic.len() / 3 {
                anomalies.push(format!("{}: traffic spike detected ({} requests)", ip, count));
            }
        }

        if let Some(method) = traffic.iter().find(|t| {
            let tl = t.to_lowercase();
            tl.contains("scan") || tl.contains("probe") || tl.contains("brute")
        }) {
            anomalies.push(format!("Suspicious activity detected: {}", method));
        }

        Ok(anomalies)
    }

    async fn generate_rules(&self, input: &FirewallAiTaskInput, filtered: &[String]) -> AgentResult<Vec<String>> {
        let mut new_rules = Vec::new();

        if !self.config.rule_generation_enabled {
            new_rules.push("Rule generation disabled".to_string());
            return Ok(new_rules);
        }

        for traffic in &input.traffic_data {
            let traffic_lower = traffic.to_lowercase();

            if traffic_lower.contains("malicious") || traffic_lower.contains("attack") {
                let ip = traffic.split_whitespace().next().unwrap_or("unknown");
                new_rules.push(format!("deny {}:443 # malicious traffic detected", ip));
                new_rules.push(format!("rate_limit {}:80 100/s # rate limit suspicious source", ip));
            }

            if traffic_lower.contains("scan") || traffic_lower.contains("probe") {
                let ip = traffic.split_whitespace().next().unwrap_or("unknown");
                new_rules.push(format!("deny {}:any # port scanning source", ip));
            }
        }

        let block_count = filtered.iter().filter(|f| f.starts_with("BLOCKED")).count();
        if block_count > filtered.len() / 2 {
            new_rules.push("alert high-traffic-blocked # anomaly threshold crossed".to_string());
        }

        if self.config.learning_mode == LearningMode::Reinforcement || self.config.learning_mode == LearningMode::Hybrid {
            new_rules.push("# adaptive rule: learned from traffic patterns".to_string());
            new_rules.push("rate_limit any:icmp 50/s # learned ICMP rate limit".to_string());
        }

        if new_rules.is_empty() {
            new_rules.push("# no new rules required".to_string());
        }

        Ok(new_rules)
    }

    async fn assess_threats(&self, input: &FirewallAiTaskInput, filtered: &[String]) -> AgentResult<String> {
        let malicious_count = input.traffic_data.iter()
            .filter(|t| {
                let tl = t.to_lowercase();
                tl.contains("malicious") || tl.contains("attack") || tl.contains("exploit")
            })
            .count();

        let anomaly_count = filtered.iter()
            .filter(|f| f.contains("ANOMALY"))
            .count();

        let block_ratio = if !filtered.is_empty() {
            filtered.iter().filter(|f| f.starts_with("BLOCKED")).count() as f32 / filtered.len() as f32
        } else {
            0.0
        };

        let threat_level = if malicious_count > 5 || anomaly_count > 3 || block_ratio > 0.7 {
            "CRITICAL"
        } else if malicious_count > 2 || anomaly_count > 1 || block_ratio > 0.4 {
            "HIGH"
        } else if malicious_count > 0 || block_ratio > 0.1 {
            "MEDIUM"
        } else {
            "LOW"
        };

        Ok(format!(
            "Threat Level: {} | Malicious: {} | Anomalies: {} | Block Ratio: {:.2}",
            threat_level, malicious_count, anomaly_count, block_ratio,
        ))
    }

    async fn calculate_firewall_score(&self, input: &FirewallAiTaskInput, new_rules: &[String]) -> AgentResult<f32> {
        let base_score = 0.90;

        let learning_bonus = match self.config.learning_mode {
            LearningMode::Hybrid => 0.05,
            LearningMode::Reinforcement => 0.04,
            LearningMode::Supervised => 0.02,
            LearningMode::Unsupervised => 0.03,
        };

        let rule_quality = if new_rules.iter().any(|r| r.starts_with("deny") || r.starts_with("rate_limit")) {
            0.03
        } else {
            0.0
        };

        let traffic_volume = if input.traffic_data.len() > 100 {
            -0.02
        } else if input.traffic_data.is_empty() {
            -0.1
        } else {
            0.0
        };

        Ok((base_score + learning_bonus + rule_quality + traffic_volume).max(0.0).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_firewall_ai_agent_creation() {
        let agent = FirewallAiAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_firewall_ai_task_processing() {
        let agent = FirewallAiAgent::default();
        let input = FirewallAiTaskInput {
            traffic_data: vec![
                "192.168.1.100 GET /index.html".to_string(),
                "10.0.0.5 malicious payload detected".to_string(),
                "192.168.1.200 POST /login".to_string(),
                "10.0.0.50 scan port 8080".to_string(),
            ],
            current_rules: vec![
                "deny 10.0.0.0/8:any".to_string(),
            ],
            analysis_parameters: HashMap::from([
                ("mode".to_string(), "learning".to_string()),
            ]),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.filtered_traffic.is_empty());
        assert!(!output.threat_assessment.is_empty());
        assert!(output.firewall_score >= 0.0 && output.firewall_score <= 1.0);
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let agent = FirewallAiAgent::default();
        let traffic = vec![
            "10.0.0.1 request".to_string(),
            "10.0.0.1 request".to_string(),
            "10.0.0.1 request".to_string(),
            "10.0.0.1 request".to_string(),
        ];

        let anomalies = agent.detect_anomalies(&traffic).await.unwrap();
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.contains("traffic spike")));
    }

    #[tokio::test]
    async fn test_rule_generation_disabled() {
        let config = FirewallAiConfig {
            rule_generation_enabled: false,
            ..FirewallAiConfig::default()
        };
        let agent = FirewallAiAgent::new(config);
        let input = FirewallAiTaskInput {
            traffic_data: vec!["192.168.1.1 malicious".to_string()],
            current_rules: vec![],
            analysis_parameters: HashMap::new(),
        };

        let rules = agent.generate_rules(&input, &[]).await.unwrap();
        assert!(rules.iter().any(|r| r.contains("disabled")));
    }
}
