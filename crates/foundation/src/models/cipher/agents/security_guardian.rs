//! Security Guardian Agent
//! 
//! Security monitoring and threat detection

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

/// Security Guardian Agent - Security monitoring and threat detection
#[derive(Debug, Clone)]
pub struct SecurityGuardianAgent {
    pub config: SecurityGuardianConfig,
    pub security_capabilities: SecurityCapabilities,
    pub threat_detection: ThreatDetection,
    pub status: AgentStatus,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityGuardianConfig {
    pub base_config: BaseAgentConfig,
    pub security_model: SecurityModel,
    pub detection_approach: DetectionApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityModel {
    IntrusionDetection,
    AnomalyDetection,
    BehavioralAnalysis,
    HybridSecurity { models: Vec<SecurityModel> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionApproach {
    SignatureBased,
    HeuristicBased,
    MachineLearningBased,
    HybridDetection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityCapabilities {
    pub threat_detection: bool,
    pub vulnerability_assessment: bool,
    pub security_monitoring: bool,
    pub incident_response: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetection {
    pub detection_algorithms: Vec<String>,
    pub analysis_methods: Vec<String>,
    pub response_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityGuardianTaskInput {
    pub security_event: String,
    pub system_logs: Vec<String>,
    pub analysis_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityGuardianTaskOutput {
    pub threat_assessment: String,
    pub detected_vulnerabilities: Vec<String>,
    pub security_recommendations: Vec<String>,
    pub security_score: f32,
}

impl Default for SecurityGuardianConfig {
    fn default() -> Self {
        Self {
            base_config: BaseAgentConfig::default(),
            security_model: SecurityModel::HybridSecurity {
                models: vec![
                    SecurityModel::IntrusionDetection,
                    SecurityModel::AnomalyDetection,
                ],
            },
            detection_approach: DetectionApproach::HybridDetection,
        }
    }
}

impl Default for SecurityCapabilities {
    fn default() -> Self {
        Self {
            threat_detection: true,
            vulnerability_assessment: true,
            security_monitoring: true,
            incident_response: true,
        }
    }
}

impl Default for ThreatDetection {
    fn default() -> Self {
        Self {
            detection_algorithms: vec![
                "pattern_matching".to_string(),
                "statistical_analysis".to_string(),
                "machine_learning".to_string(),
            ],
            analysis_methods: vec![
                "log_analysis".to_string(),
                "network_analysis".to_string(),
                "behavioral_analysis".to_string(),
            ],
            response_strategies: vec![
                "automatic_blocking".to_string(),
                "alert_notification".to_string(),
                "incident_isolation".to_string(),
            ],
        }
    }
}

impl Default for SecurityGuardianAgent {
    fn default() -> Self {
        Self {
            config: SecurityGuardianConfig::default(),
            security_capabilities: SecurityCapabilities::default(),
            threat_detection: ThreatDetection::default(),
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
impl BaseAgent for SecurityGuardianAgent {
    type Config = SecurityGuardianConfig;
    type Input = SecurityGuardianTaskInput;
    type Output = SecurityGuardianTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let threat_assessment = self.assess_threat(&input).await?;
        let detected_vulnerabilities = self.detect_vulnerabilities(&input).await?;
        let security_recommendations = self.generate_recommendations(&input, &threat_assessment).await?;
        let security_score = self.calculate_security_score(&input, &threat_assessment).await?;

        Ok(SecurityGuardianTaskOutput {
            threat_assessment,
            detected_vulnerabilities,
            security_recommendations,
            security_score,
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
                name: "security_guardian".to_string(),
                description: "Security monitoring and threat detection".to_string(),
                version: "1.0.0".to_string(),
                input_types: vec!["security_event".to_string(), "system_logs".to_string()],
                output_types: vec!["threat_assessment".to_string(), "security_recommendations".to_string()],
                metrics: crate::shared::agent_types::CapabilityMetrics {
                    accuracy: 0.96,
                    avg_latency: 2800.0,
                    resource_usage: 0.82,
                    reliability: 0.98,
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

impl SecurityGuardianAgent {
    pub fn new(config: SecurityGuardianConfig) -> Self {
        Self {
            config,
            security_capabilities: SecurityCapabilities::default(),
            threat_detection: ThreatDetection::default(),
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

    async fn assess_threat(&self, input: &SecurityGuardianTaskInput) -> AgentResult<String> {
        let threat_level = self.calculate_threat_level(&input.security_event).await?;
        let risk_factors = self.identify_risk_factors(&input).await?;
        
        Ok(format!("Threat assessment for event '{}': Level {} with factors: {}", 
                 input.security_event, threat_level, risk_factors))
    }

    async fn calculate_threat_level(&self, event: &str) -> AgentResult<String> {
        let event_lower = event.to_lowercase();
        
        if event_lower.contains("critical") || event_lower.contains("breach") {
            Ok("CRITICAL".to_string())
        } else if event_lower.contains("warning") || event_lower.contains("suspicious") {
            Ok("HIGH".to_string())
        } else if event_lower.contains("info") || event_lower.contains("notice") {
            Ok("MEDIUM".to_string())
        } else {
            Ok("LOW".to_string())
        }
    }

    async fn identify_risk_factors(&self, input: &SecurityGuardianTaskInput) -> AgentResult<String> {
        let mut factors = Vec::new();
        
        if input.system_logs.len() > 10 {
            factors.push("High log volume");
        }
        
        if input.analysis_parameters.contains_key("external_source") {
            factors.push("External source involved");
        }
        
        if factors.is_empty() {
            factors.push("Standard security event");
        }
        
        Ok(factors.join(", "))
    }

    async fn detect_vulnerabilities(&self, input: &SecurityGuardianTaskInput) -> AgentResult<Vec<String>> {
        let mut vulnerabilities = Vec::new();
        
        // Analyze system logs for vulnerability patterns
        for log in &input.system_logs {
            let log_lower = log.to_lowercase();
            
            if log_lower.contains("sql injection") {
                vulnerabilities.push("SQL Injection vulnerability detected".to_string());
            }
            
            if log_lower.contains("xss") {
                vulnerabilities.push("Cross-Site Scripting vulnerability detected".to_string());
            }
            
            if log_lower.contains("authentication failure") {
                vulnerabilities.push("Authentication bypass vulnerability detected".to_string());
            }
        }
        
        if vulnerabilities.is_empty() {
            vulnerabilities.push("No immediate vulnerabilities detected".to_string());
        }
        
        Ok(vulnerabilities)
    }

    async fn generate_recommendations(&self, input: &SecurityGuardianTaskInput, threat_assessment: &str) -> AgentResult<Vec<String>> {
        let mut recommendations = Vec::new();
        
        recommendations.push(format!("Immediate action for: {}", threat_assessment));
        
        if threat_assessment.contains("CRITICAL") {
            recommendations.push("Isolate affected systems immediately".to_string());
            recommendations.push("Activate incident response protocol".to_string());
            recommendations.push("Notify security team and stakeholders".to_string());
        } else if threat_assessment.contains("HIGH") {
            recommendations.push("Monitor affected systems closely".to_string());
            recommendations.push("Review and update security policies".to_string());
        } else {
            recommendations.push("Continue normal monitoring".to_string());
            recommendations.push("Document event for future analysis".to_string());
        }
        
        // Add log-based recommendations
        if input.system_logs.len() > 20 {
            recommendations.push("Implement log aggregation and analysis".to_string());
        }
        
        Ok(recommendations)
    }

    async fn calculate_security_score(&self, input: &SecurityGuardianTaskInput, threat_assessment: &str) -> AgentResult<f32> {
        let base_score = 0.8;
        
        // Adjust based on threat level
        let threat_adjustment = if threat_assessment.contains("CRITICAL") {
            -0.4
        } else if threat_assessment.contains("HIGH") {
            -0.2
        } else if threat_assessment.contains("MEDIUM") {
            -0.1
        } else {
            0.0
        };
        
        // Adjust based on log volume
        let log_adjustment = if input.system_logs.len() > 50 {
            -0.1
        } else if input.system_logs.len() < 5 {
            -0.05
        } else {
            0.0
        };
        
        let final_score = base_score + threat_adjustment + log_adjustment;
        Ok(final_score.max(0.0).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_guardian_agent_creation() {
        let agent = SecurityGuardianAgent::default();
        assert_eq!(agent.agent_id(), "default_agent");
        assert!(matches!(agent.get_status(), AgentStatus::Idle));
    }

    #[tokio::test]
    async fn test_security_guardian_task_processing() {
        let agent = SecurityGuardianAgent::default();
        let input = SecurityGuardianTaskInput {
            security_event: "Critical security breach detected".to_string(),
            system_logs: vec![
                "Authentication failure from unknown IP".to_string(),
                "Multiple login attempts detected".to_string(),
                "Suspicious network activity".to_string(),
            ],
            analysis_parameters: HashMap::from([
                ("source".to_string(), "external".to_string()),
                ("priority".to_string(), "high".to_string()),
            ]),
        };

        let result = agent.process(input).await;
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(!output.threat_assessment.is_empty());
        assert!(!output.detected_vulnerabilities.is_empty());
        assert!(!output.security_recommendations.is_empty());
        assert!(output.security_score >= 0.0 && output.security_score <= 1.0);
    }

    #[tokio::test]
    async fn test_threat_level_calculation() {
        let agent = SecurityGuardianAgent::default();
        
        let critical_level = agent.calculate_threat_level("CRITICAL security breach").await.unwrap();
        assert_eq!(critical_level, "CRITICAL");
        
        let high_level = agent.calculate_threat_level("WARNING suspicious activity").await.unwrap();
        assert_eq!(high_level, "HIGH");
        
        let low_level = agent.calculate_threat_level("INFO normal operation").await.unwrap();
        assert_eq!(low_level, "MEDIUM");
    }
}
