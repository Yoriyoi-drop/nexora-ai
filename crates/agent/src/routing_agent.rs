//! Routing Agent
//! 
//! Agent untuk memilih specialist model berdasarkan intent dan context.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use serde_json::{Value, json};
use tracing::{debug, info, warn};
use rand;

use crate::{
    Agent, AgentError, Result, AgentMessage, AgentResponse, AgentStatus,
    AgentContext, AgentStats, AgentConfig
};
use nexora_model::specialists::{SpecialistModel, SpecialistType, ModelCapability};

/// Routing agent untuk memilih specialist model yang tepat
pub struct RoutingAgent {
    /// Unique ID
    id: Uuid,
    /// Agent name
    name: String,
    /// Current status
    status: AgentStatus,
    /// Available specialist models
    specialist_models: Arc<HashMap<String, Box<dyn SpecialistModel>>>,
    /// Routing rules
    routing_rules: Vec<RoutingRule>,
    /// Statistics
    stats: AgentStats,
    /// Configuration
    config: RoutingAgentConfig,
}

/// Configuration untuk routing agent
#[derive(Debug, Clone)]
pub struct RoutingAgentConfig {
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Enable fallback routing
    pub enable_fallback: bool,
    /// Cache routing decisions (seconds)
    pub cache_duration_seconds: u64,
    /// Maximum routing attempts
    pub max_routing_attempts: u32,
}

/// Routing rule untuk memilih specialist
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Rule ID
    id: String,
    /// Priority (lower = higher priority)
    priority: u8,
    /// Condition untuk match
    condition: RoutingCondition,
    /// Action jika match
    action: RoutingAction,
    /// Rule aktif?
    active: bool,
}

/// Condition untuk routing
#[derive(Debug, Clone)]
pub enum RoutingCondition {
    /// Intent-based routing
    IntentMatch(String),
    /// Regex pattern matching
    PatternMatch(String),
    /// Capability-based routing
    RequiresCapability(ModelCapability),
    /// Content-type based
    ContentType(String),
    /// User-based routing
    UserPreference(Uuid),
    /// Session-based routing
    SessionContext(String),
    /// Complex condition (AND/OR)
    Complex(Box<RoutingCondition>, Box<RoutingCondition>, LogicOperator),
}

/// Logic operator untuk complex conditions
#[derive(Debug, Clone)]
pub enum LogicOperator {
    And,
    Or,
    Not,
}

/// Action untuk routing
#[derive(Debug, Clone)]
pub enum RoutingAction {
    /// Route ke specific specialist
    RouteToSpecialist(String),
    /// Route ke specialist type
    RouteToType(SpecialistType),
    /// Load balance ke multiple specialists
    LoadBalance(Vec<String>),
    /// Fallback chain
    FallbackChain(Vec<String>),
    /// Reject request
    Reject(String),
}

/// Routing decision result
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected specialist(s)
    pub selected_specialists: Vec<String>,
    /// Routing rule yang digunakan
    pub rule_id: Option<String>,
    /// Confidence score
    pub confidence: f64,
    /// Reasoning
    pub reasoning: String,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

impl RoutingAgent {
    /// Create new routing agent
    pub fn new(
        specialist_models: Arc<HashMap<String, Box<dyn SpecialistModel>>>,
        config: RoutingAgentConfig,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "RoutingAgent".to_string(),
            status: AgentStatus::Initializing,
            specialist_models,
            routing_rules: Vec::new(),
            stats: AgentStats::default(),
            config,
        }
    }
    
    /// Add routing rule
    pub fn add_routing_rule(&mut self, rule: RoutingRule) {
        self.routing_rules.push(rule);
        // Sort by priority
        self.routing_rules.sort_by_key(|r| r.priority);
    }
    
    /// Route request ke appropriate specialist
    pub async fn route_request(
        &self,
        context: &AgentContext,
        intent: Option<&str>,
        content: &Value,
    ) -> Result<RoutingDecision> {
        debug!("Routing request for session: {}", context.session_id);
        
        // Evaluate routing rules
        for rule in &self.routing_rules {
            if !rule.active {
                continue;
            }
            
            if self.evaluate_condition(&rule.condition, context, intent, content).await? {
                debug!("Routing rule '{}' matched", rule.id);
                
                let decision = self.execute_action(&rule.action, &rule.id).await?;
                
                return Ok(RoutingDecision {
                    selected_specialists: decision.specialists,
                    rule_id: Some(rule.id.clone()),
                    confidence: decision.confidence,
                    reasoning: decision.reasoning,
                    metadata: decision.metadata,
                });
            }
        }
        
        // Fallback routing
        if self.config.enable_fallback {
            warn!("No routing rule matched, using fallback");
            self.fallback_routing().await
        } else {
            Err(AgentError::ProcessingError("No routing rule matched and fallback disabled".to_string()))
        }
    }
    
    /// Evaluate routing condition
    fn evaluate_condition<'a>(
        &'a self,
        condition: &'a RoutingCondition,
        context: &'a AgentContext,
        intent: Option<&'a str>,
        content: &'a Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
        match condition {
            RoutingCondition::IntentMatch(expected_intent) => {
                if let Some(actual_intent) = intent {
                    Ok(actual_intent == expected_intent)
                } else {
                    Ok(false)
                }
            }
            
            RoutingCondition::PatternMatch(pattern) => {
                // Extract text from content
                let text = self.extract_text_from_content(content);
                
                // Implement regex pattern matching
                match regex::Regex::new(&pattern) {
                    Ok(regex) => {
                        let matches = regex.is_match(&text);
                        debug!("Regex pattern '{}' matches: {}", pattern, matches);
                        Ok(matches)
                    }
                    Err(e) => {
                        warn!("Invalid regex pattern '{}': {}, falling back to simple matching", pattern, e);
                        // Fallback to simple string matching
                        Ok(text.to_lowercase().contains(&pattern.to_lowercase()))
                    }
                }
            }
            
            RoutingCondition::RequiresCapability(capability) => {
                // Check if any specialist has required capability
                for specialist in self.specialist_models.values() {
                    if specialist.has_capability(capability.clone()) {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            
            RoutingCondition::ContentType(expected_type) => {
                if let Some(content_type) = content.get("type").and_then(|v| v.as_str()) {
                    Ok(content_type == expected_type)
                } else {
                    Ok(false)
                }
            }
            
            RoutingCondition::UserPreference(user_id) => {
                Ok(context.user_id == Some(*user_id))
            }
            
            RoutingCondition::SessionContext(key) => {
                Ok(context.session_state.contains_key(key))
            }
            
            RoutingCondition::Complex(left, right, operator) => {
                let left_result = self.evaluate_condition(left, context, intent, content).await?;
                let right_result = self.evaluate_condition(right, context, intent, content).await?;
                
                match operator {
                    LogicOperator::And => Ok(left_result && right_result),
                    LogicOperator::Or => Ok(left_result || right_result),
                    LogicOperator::Not => Ok(!right_result), // NOT only applies to right
                }
            }
        }
        })
    }
    
    /// Execute routing action
    async fn execute_action(&self, action: &RoutingAction, _rule_id: &str) -> Result<RoutingActionResult> {
        match action {
            RoutingAction::RouteToSpecialist(specialist_id) => {
                if self.specialist_models.contains_key(specialist_id) {
                    Ok(RoutingActionResult {
                        specialists: vec![specialist_id.clone()],
                        confidence: 0.9,
                        reasoning: format!("Direct routing to specialist: {}", specialist_id),
                        metadata: HashMap::new(),
                    })
                } else {
                    Err(AgentError::ProcessingError(format!("Specialist {} not found", specialist_id)))
                }
            }
            
            RoutingAction::RouteToType(specialist_type) => {
                let matching_specialists: Vec<String> = self.specialist_models.iter()
                    .filter(|(_, model)| model.specialist_type() == *specialist_type)
                    .map(|(id, _)| id.clone())
                    .collect();
                
                if !matching_specialists.is_empty() {
                    let selected = if self.config.enable_load_balancing && matching_specialists.len() > 1 {
                        // Simple load balancing - select randomly
                        let idx = (rand::random::<usize>() % matching_specialists.len()) as usize;
                        vec![matching_specialists[idx].clone()]
                    } else {
                        vec![matching_specialists[0].clone()]
                    };
                    
                    Ok(RoutingActionResult {
                        specialists: selected,
                        confidence: 0.8,
                        reasoning: format!("Routing to specialist type: {:?}", specialist_type),
                        metadata: HashMap::new(),
                    })
                } else {
                    Err(AgentError::ProcessingError(format!("No specialists of type {:?}", specialist_type)))
                }
            }
            
            RoutingAction::LoadBalance(specialist_ids) => {
                let available_specialists: Vec<String> = specialist_ids.iter()
                    .filter(|id| self.specialist_models.contains_key(*id))
                    .cloned()
                    .collect();
                
                if !available_specialists.is_empty() {
                    let idx = (rand::random::<usize>() % available_specialists.len()) as usize;
                    Ok(RoutingActionResult {
                        specialists: vec![available_specialists[idx].clone()],
                        confidence: 0.7,
                        reasoning: format!("Load balancing across {} specialists", available_specialists.len()),
                        metadata: HashMap::new(),
                    })
                } else {
                    Err(AgentError::ProcessingError("No available specialists for load balancing".to_string()))
                }
            }
            
            RoutingAction::FallbackChain(specialist_ids) => {
                // Return all in chain for fallback logic
                Ok(RoutingActionResult {
                    specialists: specialist_ids.clone(),
                    confidence: 0.6,
                    reasoning: "Fallback chain routing".to_string(),
                    metadata: HashMap::new(),
                })
            }
            
            RoutingAction::Reject(reason) => {
                Err(AgentError::ProcessingError(format!("Request rejected: {}", reason)))
            }
        }
    }
    
    /// Fallback routing logic
    async fn fallback_routing(&self) -> Result<RoutingDecision> {
        // Try to find any available specialist
        if let Some((specialist_id, _)) = self.specialist_models.iter().next() {
            Ok(RoutingDecision {
                selected_specialists: vec![specialist_id.clone()],
                rule_id: None,
                confidence: 0.5,
                reasoning: "Fallback routing - first available specialist".to_string(),
                metadata: HashMap::new(),
            })
        } else {
            Err(AgentError::ProcessingError("No specialists available for fallback routing".to_string()))
        }
    }
    
    /// Extract text from content
    fn extract_text_from_content(&self, content: &Value) -> String {
        if let Some(text) = content.get("text").and_then(|v| v.as_str()) {
            text.to_string()
        } else if let Some(text) = content.get("content").and_then(|v| v.as_str()) {
            text.to_string()
        } else if let Some(text) = content.get("message").and_then(|v| v.as_str()) {
            text.to_string()
        } else {
            // Convert to string as fallback
            content.to_string()
        }
    }
    
    /// Get available specialists by capability
    pub fn get_specialists_by_capability(&self, capability: &ModelCapability) -> Vec<String> {
        self.specialist_models.iter()
            .filter(|(_, model)| model.has_capability(capability.clone()))
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Get routing statistics
    pub fn get_routing_stats(&self) -> RoutingStats {
        RoutingStats {
            total_rules: self.routing_rules.len(),
            active_rules: self.routing_rules.iter().filter(|r| r.active).count(),
            available_specialists: self.specialist_models.len(),
            load_balancing_enabled: self.config.enable_load_balancing,
            fallback_enabled: self.config.enable_fallback,
        }
    }
}

/// Internal routing action result
#[derive(Debug, Clone)]
struct RoutingActionResult {
    specialists: Vec<String>,
    confidence: f64,
    reasoning: String,
    metadata: HashMap<String, Value>,
}

/// Routing statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct RoutingStats {
    pub total_rules: usize,
    pub active_rules: usize,
    pub available_specialists: usize,
    pub load_balancing_enabled: bool,
    pub fallback_enabled: bool,
}

#[async_trait]
impl Agent for RoutingAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> &str {
        "routing"
    }
    
    fn status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn initialize(&mut self, _config: AgentConfig) -> Result<()> {
        info!("Initializing RoutingAgent");
        
        // Add default routing rules
        self.add_default_routing_rules();
        
        self.status = AgentStatus::Ready;
        Ok(())
    }
    
    async fn receive(&mut self, message: AgentMessage) -> Result<()> {
        debug!("RoutingAgent received message: {}", message.message_type);
        Ok(())
    }
    
    async fn process(&mut self, context: AgentContext) -> Result<AgentResponse> {
        let start_time = std::time::Instant::now();
        
        debug!("RoutingAgent processing request for session: {}", context.session_id);
        
        // Extract intent and content from context
        let intent = context.parameters.get("intent").and_then(|v| v.as_str());
        let content = context.parameters.get("content").unwrap_or(&Value::Null);
        
        // Route request
        let routing_decision = self.route_request(&context, intent, content).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Update stats
        self.stats.messages_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.messages_processed - 1) as f64 + 
             processing_time as f64) / self.stats.messages_processed as f64;
        self.stats.last_activity = chrono::Utc::now();
        
        let response = AgentResponse::success(
            context.session_id,
            json!({
                "routing_decision": {
                    "selected_specialists": routing_decision.selected_specialists,
                    "rule_id": routing_decision.rule_id,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                    "metadata": routing_decision.metadata
                },
                "routing_stats": self.get_routing_stats()
            }),
            processing_time,
        );
        
        Ok(response)
    }
    
    async fn respond(&mut self, _response: AgentResponse) -> Result<()> {
        debug!("RoutingAgent sending response");
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down RoutingAgent");
        self.status = AgentStatus::Shutdown;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<bool> {
        // Check if we have any specialists available
        Ok(!self.specialist_models.is_empty())
    }
    
    fn get_stats(&self) -> AgentStats {
        self.stats.clone()
    }
    
    fn get_config(&self) -> AgentConfig {
        self.config.clone().into()
    }
}

impl RoutingAgent {
    /// Add default routing rules
    fn add_default_routing_rules(&mut self) {
        // Rule for text generation
        self.add_routing_rule(RoutingRule {
            id: "text_generation".to_string(),
            priority: 10,
            condition: RoutingCondition::RequiresCapability(ModelCapability::TextGeneration),
            action: RoutingAction::RouteToType(SpecialistType::TextGenerator),
            active: true,
        });
        
        // Rule for analysis
        self.add_routing_rule(RoutingRule {
            id: "analysis".to_string(),
            priority: 20,
            condition: RoutingCondition::IntentMatch("analyze".to_string()),
            action: RoutingAction::RouteToType(SpecialistType::Analyzer),
            active: true,
        });
        
        // Rule for coding
        self.add_routing_rule(RoutingRule {
            id: "coding".to_string(),
            priority: 15,
            condition: RoutingCondition::PatternMatch("code|programming|function".to_string()),
            action: RoutingAction::RouteToType(SpecialistType::CodeGenerator),
            active: true,
        });
        
        // Rule for creative writing
        self.add_routing_rule(RoutingRule {
            id: "creative".to_string(),
            priority: 25,
            condition: RoutingCondition::PatternMatch("write|create|story|poem".to_string()),
            action: RoutingAction::RouteToType(SpecialistType::CreativeWriter),
            active: true,
        });
        
        // Fallback rule
        self.add_routing_rule(RoutingRule {
            id: "fallback".to_string(),
            priority: 100,
            condition: RoutingCondition::ContentType("text".to_string()),
            action: RoutingAction::FallbackChain(vec![
                "general_text".to_string(),
                "basic_analyzer".to_string(),
            ]),
            active: true,
        });
    }
}

impl From<RoutingAgentConfig> for AgentConfig {
    fn from(_config: RoutingAgentConfig) -> Self {
        AgentConfig {
            agent_id: "routing_agent".to_string(),
            agent_type: "routing".to_string(),
            max_concurrent_tasks: 15,
            timeout_seconds: 5,
        }
    }
}

impl Default for RoutingAgentConfig {
    fn default() -> Self {
        Self {
            enable_load_balancing: true,
            enable_fallback: true,
            cache_duration_seconds: 300, // 5 minutes
            max_routing_attempts: 3,
        }
    }
}
