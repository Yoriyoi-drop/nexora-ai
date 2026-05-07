//! Validation Agent
//! 
//! Agent untuk cek hallucination dan invalid output.

use std::collections::HashMap;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info};

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};

/// Validation agent untuk validasi output
pub struct ValidationAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Validation rules
    validation_rules: Vec<Box<dyn ValidationRule>>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: ValidationAgentConfig,
}

/// Configuration untuk validation agent
#[derive(Debug, Clone)]
pub struct ValidationAgentConfig {
    /// Enable hallucination detection
    pub enable_hallucination_detection: bool,
    /// Enable fact checking
    pub enable_fact_checking: bool,
    /// Enable content filtering
    pub enable_content_filtering: bool,
    /// Strict validation mode
    pub strict_mode: bool,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Overall validity
    pub is_valid: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Validation issues found
    pub issues: Vec<ValidationIssue>,
    /// Validation metadata
    pub metadata: HashMap<String, Value>,
}

/// Validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue type
    pub issue_type: ValidationIssueType,
    /// Severity level
    pub severity: SeverityLevel,
    /// Description
    pub description: String,
    /// Location in content (if applicable)
    pub location: Option<String>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Validation issue types
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum ValidationIssueType {
    /// Possible hallucination
    Hallucination,
    /// Factual error
    FactualError,
    /// Incoherent content
    Incoherent,
    /// Inappropriate content
    Inappropriate,
    /// Incomplete response
    Incomplete,
    /// Contradiction
    Contradiction,
    /// Format error
    FormatError,
    /// Logic error
    LogicError,
    /// Custom issue type
    Custom(String),
}

/// Severity levels
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, serde::Serialize)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Trait untuk validation rules
#[async_trait]
pub trait ValidationRule: Send + Sync {
    /// Rule name
    fn name(&self) -> &str;
    
    /// Validate content
    async fn validate(&self, content: &str, context: &Value) -> ValidationResult;
    
    /// Can handle this content type?
    fn can_handle(&self, content_type: &str) -> bool;
}

impl ValidationAgent {
    /// Create new validation agent
    pub fn new(config: ValidationAgentConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "ValidationAgent".to_string(),
            status: AgentStatus::Initializing,
            validation_rules: Vec::new(),
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Add validation rule
    pub fn add_validation_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.validation_rules.push(rule);
    }
    
    /// Validate content
    pub async fn validate_content(
        &self,
        content: &str,
        context: &Value,
        content_type: Option<&str>,
    ) -> Result<ValidationResult> {
        debug!("Validating content of length: {}", content.len());
        
        let mut all_issues = Vec::new();
        let mut total_confidence = 0.0;
        let mut rule_count = 0;
        
        // Apply all applicable validation rules
        for rule in &self.validation_rules {
            if let Some(content_type) = content_type {
                if !rule.can_handle(content_type) {
                    continue;
                }
            }
            
            let result = rule.validate(content, context).await;
            
            total_confidence += result.confidence;
            rule_count += 1;
            
            all_issues.extend(result.issues);
        }
        
        // Calculate overall confidence
        let overall_confidence = if rule_count > 0 {
            total_confidence / rule_count as f64
        } else {
            0.0
        };
        
        // Determine overall validity
        let is_valid = if self.config.strict_mode {
            overall_confidence >= self.config.confidence_threshold && 
            all_issues.iter().all(|issue| issue.severity <= SeverityLevel::Medium)
        } else {
            overall_confidence >= self.config.confidence_threshold
        };
        
        // Sort issues by severity
        all_issues.sort_by(|a, b| b.severity.cmp(&a.severity));
        
        let validation_result = ValidationResult {
            is_valid,
            confidence: overall_confidence,
            issues: all_issues,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("rules_applied".to_string(), Value::Number(rule_count.into()));
                meta.insert("strict_mode".to_string(), Value::Bool(self.config.strict_mode));
                meta.insert("threshold".to_string(), Value::Number(serde_json::Number::from_f64(self.config.confidence_threshold).unwrap_or(serde_json::Number::from(0))));
                meta
            },
        };
        
        debug!("Validation completed: valid={}, confidence={:.2}", 
               validation_result.is_valid, validation_result.confidence);
        
        Ok(validation_result)
    }
    
    /// Quick validation (basic checks only)
    pub async fn quick_validate(&self, content: &str) -> Result<bool> {
        debug!("Quick validation of content");
        
        // Basic checks
        if content.is_empty() {
            return Ok(false);
        }
        
        if content.len() < 10 {
            return Ok(false);
        }
        
        // Check for obvious issues
        let obvious_issues = [
            "I cannot answer",
            "I don't know",
            "As an AI",
            "I'm an AI",
            "I am an AI",
        ];
        
        for issue in &obvious_issues {
            if content.to_lowercase().contains(issue) {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Check for hallucinations
    pub async fn detect_hallucinations(&self, content: &str, context: &Value) -> Result<Vec<ValidationIssue>> {
        if !self.config.enable_hallucination_detection {
            return Ok(Vec::new());
        }
        
        debug!("Detecting hallucinations in content");
        
        let mut issues = Vec::new();
        
        // Simple hallucination detection patterns
        let hallucination_patterns = [
            "I believe that",
            "It seems that",
            "Probably",
            "Maybe",
            "Perhaps",
            "I think",
        ];
        
        for pattern in &hallucination_patterns {
            if content.to_lowercase().contains(&pattern.to_lowercase()) {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::Hallucination,
                    severity: SeverityLevel::Medium,
                    description: format!("Uncertain language detected: {}", pattern),
                    location: None,
                    suggested_fix: Some("Use more confident and factual language".to_string()),
                });
            }
        }
        
        // Check for made-up facts (simplified)
        if let Some(facts) = context.get("known_facts").and_then(|v| v.as_array()) {
            for fact in facts {
                if let Some(fact_str) = fact.as_str() {
                    if !content.contains(fact_str) {
                        // This is a very simplified check - real implementation would be more sophisticated
                        issues.push(ValidationIssue {
                            issue_type: ValidationIssueType::FactualError,
                            severity: SeverityLevel::High,
                            description: format!("Potential factual inconsistency with: {}", fact_str),
                            location: None,
                            suggested_fix: Some("Verify factual accuracy".to_string()),
                        });
                    }
                }
            }
        }
        
        Ok(issues)
    }
    
    /// Check content appropriateness
    pub async fn check_content_appropriateness(&self, content: &str) -> Result<Vec<ValidationIssue>> {
        if !self.config.enable_content_filtering {
            return Ok(Vec::new());
        }
        
        debug!("Checking content appropriateness");
        
        let mut issues = Vec::new();
        
        // Simple inappropriate content detection
        let inappropriate_patterns = [
            "hate",
            "violence",
            "illegal",
            "harmful",
            "dangerous",
        ];
        
        for pattern in &inappropriate_patterns {
            if content.to_lowercase().contains(&pattern.to_lowercase()) {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::Inappropriate,
                    severity: SeverityLevel::High,
                    description: format!("Potentially inappropriate content: {}", pattern),
                    location: None,
                    suggested_fix: Some("Review and revise content for appropriateness".to_string()),
                });
            }
        }
        
        Ok(issues)
    }
    
    /// Check content coherence
    pub async fn check_coherence(&self, content: &str) -> Result<Vec<ValidationIssue>> {
        debug!("Checking content coherence");
        
        let mut issues = Vec::new();
        
        // Simple coherence checks
        let sentences: Vec<&str> = content.split('.').filter(|s| !s.trim().is_empty()).collect();
        
        if sentences.len() > 1 {
            // Check for contradictory statements
            for i in 0..sentences.len() {
                for j in (i + 1)..sentences.len() {
                    let sentence1 = sentences[i].trim().to_lowercase();
                    let sentence2 = sentences[j].trim().to_lowercase();
                    
                    // Simple contradiction detection
                    if (sentence1.contains("not") || sentence1.contains("no")) && 
                       (sentence2.contains("not") || sentence2.contains("no")) {
                        // This is very simplified - real implementation would use NLP
                        continue;
                    }
                    
                    // Check for opposite statements
                    if sentence1.contains("good") && sentence2.contains("bad") {
                        issues.push(ValidationIssue {
                            issue_type: ValidationIssueType::Contradiction,
                            severity: SeverityLevel::Medium,
                            description: "Potential contradiction detected".to_string(),
                            location: Some(format!("Sentences {} and {}", i + 1, j + 1)),
                            suggested_fix: Some("Review for logical consistency".to_string()),
                        });
                    }
                }
            }
        }
        
        Ok(issues)
    }
    
    /// Get validation statistics
    pub fn get_validation_stats(&self) -> ValidationStats {
        ValidationStats {
            total_rules: self.validation_rules.len(),
            hallucination_detection_enabled: self.config.enable_hallucination_detection,
            fact_checking_enabled: self.config.enable_fact_checking,
            content_filtering_enabled: self.config.enable_content_filtering,
            strict_mode: self.config.strict_mode,
            confidence_threshold: self.config.confidence_threshold,
        }
    }
}

/// Validation statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidationStats {
    pub total_rules: usize,
    pub hallucination_detection_enabled: bool,
    pub fact_checking_enabled: bool,
    pub content_filtering_enabled: bool,
    pub strict_mode: bool,
    pub confidence_threshold: f64,
}

#[async_trait]
impl Agent for ValidationAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "validation"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing ValidationAgent");
        
        // Add default validation rules
        self.add_default_validation_rules();
        
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("ValidationAgent received message: {}", message.message_type);
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("ValidationAgent processing request for session: {}", context.session_id);
        
        // Extract action from context
        let action = context.parameters.get("action").and_then(|v| v.as_str()).unwrap_or("validate");
        
        let result = match action {
            "validate" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let content_type = context.parameters.get("content_type")
                    .and_then(|v| v.as_str());
                
                let metadata_value = serde_json::to_value(&context.metadata)?;
                let validation_result = self.validate_content(content, &metadata_value, content_type).await?;
                
                json!({
                    "action": "validate",
                    "validation_result": {
                        "is_valid": validation_result.is_valid,
                        "confidence": validation_result.confidence,
                        "issues": validation_result.issues.iter().map(|issue| json!({
                            "issue_type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "location": issue.location,
                            "suggested_fix": issue.suggested_fix
                        })).collect::<Vec<_>>(),
                        "metadata": validation_result.metadata
                    }
                })
            }
            
            "quick_validate" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let is_valid = self.quick_validate(content).await?;
                
                json!({
                    "action": "quick_validate",
                    "is_valid": is_valid
                })
            }
            
            "detect_hallucinations" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let metadata_value = serde_json::to_value(&context.metadata)?;
                let issues = self.detect_hallucinations(content, &metadata_value).await?;
                
                json!({
                    "action": "detect_hallucinations",
                    "issues": issues.iter().map(|issue| json!({
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "location": issue.location,
                        "suggested_fix": issue.suggested_fix
                    })).collect::<Vec<_>>(),
                    "count": issues.len()
                })
            }
            
            "check_appropriateness" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let issues = self.check_content_appropriateness(content).await?;
                
                json!({
                    "action": "check_appropriateness",
                    "issues": issues.iter().map(|issue| json!({
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "location": issue.location,
                        "suggested_fix": issue.suggested_fix
                    })).collect::<Vec<_>>(),
                    "count": issues.len()
                })
            }
            
            "check_coherence" => {
                let content = context.parameters.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::ProcessingError("content required".to_string()))?;
                
                let issues = self.check_coherence(content).await?;
                
                json!({
                    "action": "check_coherence",
                    "issues": issues.iter().map(|issue| json!({
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "location": issue.location,
                        "suggested_fix": issue.suggested_fix
                    })).collect::<Vec<_>>(),
                    "count": issues.len()
                })
            }
            
            "stats" => {
                let stats = self.get_validation_stats();
                json!({
                    "action": "stats",
                    "stats": stats
                })
            }
            
            _ => {
                return Err(AgentError::ProcessingError(format!("Unknown action: {}", action)));
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.messages_processed - 1) as f64 + 
             processing_time as f64) / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();
        
        let response = AgentResponse::success(
            context.session_id,
            result,
            processing_time,
        );
        
        Ok(response)
    }
    
    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("ValidationAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down ValidationAgent");
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check if we have validation rules
        Ok(!self.validation_rules.is_empty())
    }
    
    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
    
    fn get_config(&self) -> AgentConfig {
        self.config.clone().into()
    }
}

impl ValidationAgent {
    /// Add default validation rules
    fn add_default_validation_rules(&mut self) {
        // Add basic content validation rule
        self.add_validation_rule(Box::new(BasicContentValidationRule));
        
        // Add coherence validation rule
        self.add_validation_rule(Box::new(CoherenceValidationRule));
        
        // Add appropriateness validation rule
        self.add_validation_rule(Box::new(AppropriatenessValidationRule));
    }
}

/// Basic content validation rule
struct BasicContentValidationRule;

#[async_trait]
impl ValidationRule for BasicContentValidationRule {
    fn name(&self) -> &str {
        "basic_content"
    }
    
    async fn validate(&self, content: &str, _context: &Value) -> ValidationResult {
        let mut issues = Vec::new();
        let mut confidence = 1.0;
        
        // Check for empty content
        if content.trim().is_empty() {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::Incomplete,
                severity: SeverityLevel::Critical,
                description: "Content is empty".to_string(),
                location: None,
                suggested_fix: Some("Provide meaningful content".to_string()),
            });
            confidence = 0.0;
        }
        
        // Check for minimum length
        if content.len() < 10 {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::Incomplete,
                severity: SeverityLevel::High,
                description: "Content is too short".to_string(),
                location: None,
                suggested_fix: Some("Provide more detailed content".to_string()),
            });
            confidence = f64::max(confidence - 0.3, 0.0);
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            confidence,
            issues,
            metadata: HashMap::new(),
        }
    }
    
    fn can_handle(&self, _content_type: &str) -> bool {
        true // Can handle any content type
    }
}

/// Coherence validation rule
struct CoherenceValidationRule;

#[async_trait]
impl ValidationRule for CoherenceValidationRule {
    fn name(&self) -> &str {
        "coherence"
    }
    
    async fn validate(&self, content: &str, _context: &Value) -> ValidationResult {
        let mut issues = Vec::new();
        let mut confidence = 1.0;
        
        // Simple coherence check
        let sentences: Vec<&str> = content.split('.').filter(|s| !s.trim().is_empty()).collect();
        
        if sentences.len() > 1 {
            // Check for very short sentences (might indicate incoherence)
            let short_sentences: usize = sentences.iter()
                .filter(|s| s.trim().len() < 5)
                .count();
            
            if short_sentences > sentences.len() / 2 {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::Incoherent,
                    severity: SeverityLevel::Medium,
                    description: "Many very short sentences detected".to_string(),
                    location: None,
                    suggested_fix: Some("Combine short sentences for better flow".to_string()),
                });
                confidence = f64::max(confidence - 0.2, 0.0);
            }
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            confidence,
            issues,
            metadata: HashMap::new(),
        }
    }
    
    fn can_handle(&self, content_type: &str) -> bool {
        content_type == "text" || content_type == "conversation"
    }
}

/// Appropriateness validation rule
struct AppropriatenessValidationRule;

#[async_trait]
impl ValidationRule for AppropriatenessValidationRule {
    fn name(&self) -> &str {
        "appropriateness"
    }
    
    async fn validate(&self, content: &str, _context: &Value) -> ValidationResult {
        let mut issues = Vec::new();
        let mut confidence = 1.0;
        
        // Simple appropriateness check
        let inappropriate_patterns = ["hate", "violence", "illegal"];
        
        for pattern in &inappropriate_patterns {
            if content.to_lowercase().contains(pattern) {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::Inappropriate,
                    severity: SeverityLevel::High,
                    description: format!("Inappropriate content detected: {}", pattern),
                    location: None,
                    suggested_fix: Some("Remove inappropriate content".to_string()),
                });
                confidence = f64::max(confidence - 0.5, 0.0);
            }
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            confidence,
            issues,
            metadata: HashMap::new(),
        }
    }
    
    fn can_handle(&self, content_type: &str) -> bool {
        content_type == "text" || content_type == "conversation"
    }
}

impl From<ValidationAgentConfig> for AgentConfig {
    fn from(_config: ValidationAgentConfig) -> Self {
        AgentConfig {
            agent_id: "validation_agent".to_string(),
            agent_type: "validation".to_string(),
            max_concurrent_tasks: 5,
            timeout_seconds: 15,
        }
    }
}

impl Default for ValidationAgentConfig {
    fn default() -> Self {
        Self {
            enable_hallucination_detection: true,
            enable_fact_checking: true,
            enable_content_filtering: true,
            strict_mode: false,
            confidence_threshold: 0.8,
        }
    }
}
