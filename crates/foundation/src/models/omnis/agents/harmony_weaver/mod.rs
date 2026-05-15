//! Harmony Weaver Agent Module
//! 
//! Emotional intelligence and social harmony optimization

pub mod config;
pub mod optimization;
pub mod capabilities;
pub mod types;

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::shared::{
    base_agent::{BaseAgent, BaseAgentConfig},
    agent_types::{AgentStatus, AgentCapability, AgentMetrics, AgentResult},
};

use config::HarmonyWeaverConfig;
use capabilities::{HarmonyWeaverCapabilities, EmotionalIntelligenceCapabilities, SocialHarmonyOptimization, ConflictResolution};
use optimization::{HarmonyOptimizer, InterventionPlanner, ConflictPlanner};
use types::{HarmonyWeaverTaskInput, HarmonyWeaverTaskOutput, EmotionalIntelligenceAnalysis, HarmonyAssessment, InterventionRecommendations, ConflictResolutionStrategies};

/// Harmony Weaver Agent - Emotional intelligence and social harmony optimization
#[derive(Debug, Clone)]
pub struct HarmonyWeaverAgent {
    /// Agent configuration
    pub config: HarmonyWeaverConfig,
    /// Emotional intelligence capabilities
    pub emotional_intelligence_capabilities: EmotionalIntelligenceCapabilities,
    /// Social harmony optimization
    pub social_harmony_optimization: SocialHarmonyOptimization,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
    /// Agent status
    status: AgentStatus,
    /// Agent metrics
    metrics: AgentMetrics,
}

impl HarmonyWeaverAgent {
    /// Create a new Harmony Weaver Agent
    pub fn new(config: HarmonyWeaverConfig) -> Self {
        Self {
            config,
            emotional_intelligence_capabilities: EmotionalIntelligenceCapabilities::default(),
            social_harmony_optimization: SocialHarmonyOptimization::default(),
            conflict_resolution: ConflictResolution::default(),
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

    /// Validate harmony weaver task input
    fn validate_input(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<()> {
        if input.social_context.is_empty() {
            return Err(crate::shared::agent_types::AgentError::InvalidInput(
                "Social context cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }

    /// Assess harmony
    async fn assess_harmony(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<HarmonyAssessment> {
        // Analyze emotional landscape
        let emotional_landscape = self.analyze_emotional_landscape(&input).await?;
        
        // Assess social dynamics
        let social_dynamics = self.assess_social_dynamics(&input).await?;
        
        // Evaluate harmony indicators
        let harmony_indicators = self.evaluate_harmony_indicators(&input).await?;
        
        // Calculate harmony score
        let harmony_score = self.calculate_harmony_score(&emotional_landscape, &social_dynamics, &harmony_indicators);
        
        Ok(HarmonyAssessment {
            emotional_landscape,
            social_dynamics,
            harmony_indicators,
            harmony_score,
            assessment_timestamp: chrono::Utc::now(),
        })
    }

    /// Analyze emotional intelligence
    async fn analyze_emotional_intelligence(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<types::EmotionalIntelligenceAnalysis> {
        // Assess emotional awareness
        let emotional_awareness = self.assess_emotional_awareness(&input).await?;
        
        // Evaluate emotional regulation
        let emotional_regulation = self.evaluate_emotional_regulation(&input).await?;
        
        // Analyze social awareness
        let social_awareness = self.analyze_social_awareness(&input).await?;
        
        // Assess relationship management
        let relationship_management = self.assess_relationship_management(&input).await?;
        
        Ok(types::EmotionalIntelligenceAnalysis {
            emotional_awareness,
            emotional_regulation,
            social_awareness,
            relationship_management,
            overall_ei_score: 0.0, // Will be calculated
            analysis_timestamp: chrono::Utc::now(),
        })
    }

    /// Generate intervention recommendations
    async fn generate_intervention_recommendations(
        &self, 
        input: &HarmonyWeaverTaskInput, 
        harmony_assessment: &HarmonyAssessment, 
        emotional_intelligence_analysis: &types::EmotionalIntelligenceAnalysis
    ) -> AgentResult<InterventionRecommendations> {
        // Identify intervention opportunities
        let intervention_opportunities = self.identify_intervention_opportunities(&harmony_assessment, &emotional_intelligence_analysis).await?;
        
        // Prioritize interventions
        let prioritized_interventions = self.prioritize_interventions(&intervention_opportunities).await?;
        
        // Generate implementation strategies
        let implementation_strategies = self.generate_implementation_strategies(&prioritized_interventions).await?;
        
        Ok(InterventionRecommendations {
            intervention_opportunities,
            prioritized_interventions,
            implementation_strategies,
            expected_outcomes: vec![],
            risk_assessment: vec![],
            recommendation_timestamp: chrono::Utc::now(),
        })
    }

    /// Develop conflict resolution strategies
    async fn develop_conflict_resolution_strategies(
        &self, 
        input: &HarmonyWeaverTaskInput, 
        harmony_assessment: &HarmonyAssessment
    ) -> AgentResult<ConflictResolutionStrategies> {
        // Identify conflicts
        let identified_conflicts = self.identify_conflicts(&input, &harmony_assessment).await?;
        
        // Analyze conflict dynamics
        let conflict_dynamics = self.analyze_conflict_dynamics(&identified_conflicts).await?;
        
        // Develop resolution approaches
        let resolution_approaches = self.develop_resolution_approaches(&conflict_dynamics).await?;
        
        Ok(ConflictResolutionStrategies {
            identified_conflicts,
            conflict_dynamics,
            resolution_approaches,
            success_probability: 0.0,
            implementation_timeline: vec![],
            strategy_timestamp: chrono::Utc::now(),
        })
    }

    /// Create harmony optimization plan
    async fn create_harmony_optimization_plan(
        &self, 
        input: &HarmonyWeaverTaskInput, 
        harmony_assessment: &HarmonyAssessment, 
        intervention_recommendations: &InterventionRecommendations
    ) -> AgentResult<types::HarmonyOptimizationPlan> {
        // Define optimization goals
        let optimization_goals = self.define_optimization_goals(&harmony_assessment).await?;
        
        // Create action plan
        let action_plan = self.create_action_plan(&intervention_recommendations).await?;
        
        // Set success metrics
        let success_metrics = self.define_success_metrics(&optimization_goals).await?;
        
        Ok(types::HarmonyOptimizationPlan {
            optimization_goals,
            action_plan,
            success_metrics,
            timeline: vec![],
            resources_required: vec![],
            plan_timestamp: chrono::Utc::now(),
        })
    }

    // Helper methods with heuristic-based implementations
    async fn analyze_emotional_landscape(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<types::EmotionalLandscape> {
        let text = &input.social_context;
        let positive_words = ["happy", "good", "great", "excellent", "wonderful", "love", "thank", "nice", "pleased", "satisfied"];
        let negative_words = ["angry", "sad", "bad", "terrible", "hate", "frustrated", "upset", "disappointed", "annoyed", "worried"];

        let mut emotions = Vec::new();
        let mut pos_count = 0;
        let mut neg_count = 0;
        let lower = text.to_lowercase();

        for w in positive_words { if lower.contains(w) { pos_count += 1; emotions.push(types::EmotionEntry { emotion: w.to_string(), intensity: 0.7, source_participant: None }); } }
        for w in negative_words { if lower.contains(w) { neg_count += 1; emotions.push(types::EmotionEntry { emotion: w.to_string(), intensity: 0.8, source_participant: None }); } }

        let total = (pos_count + neg_count).max(1) as f32;
        let emotional_intensity = (pos_count as f32 + neg_count as f32) / input.social_context.split_whitespace().count().max(1) as f32;
        let emotional_diversity = if emotions.is_empty() { 0.0 } else { (emotions.len() as f32 / 10.0).min(1.0) };
        let emotional_stability = if pos_count > neg_count { 0.7 + (pos_count as f32 / total) * 0.3 } else { 0.5 - (neg_count as f32 / total) * 0.3 };

        Ok(types::EmotionalLandscape {
            dominant_emotions: emotions,
            emotional_intensity: emotional_intensity.min(1.0),
            emotional_diversity,
            emotional_stability: emotional_stability.max(0.1).min(1.0),
        })
    }

    async fn assess_social_dynamics(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<types::SocialDynamics> {
        let participant_count = input.participants.len().max(1);
        let interaction_count = input.interaction_history.len();

        let group_cohesion = if interaction_count > 0 {
            0.5 + (interaction_count as f32 / (participant_count * 10).max(1) as f32).min(0.5)
        } else {
            0.3
        };

        let communication_patterns: Vec<types::CommunicationPattern> = input.participants.iter().enumerate().map(|(i, p)| {
            types::CommunicationPattern {
                from: p.id.clone(),
                to: input.participants[(i + 1) % participant_count].id.clone(),
                frequency: (interaction_count / participant_count).max(1) as u32,
                average_sentiment: 0.5 + (i as f32 * 0.1),
            }
        }).collect();

        let power_dynamics: Vec<types::PowerDynamic> = input.participants.iter().map(|p| {
            types::PowerDynamic {
                participant_id: p.id.clone(),
                influence_score: 0.3 + (rand::random::<f32>() * 0.4),
                dominance_level: 0.2 + (rand::random::<f32>() * 0.3),
            }
        }).collect();

        let social_network_structure: Vec<types::SocialConnection> = (0..participant_count).flat_map(|i| {
            ((i + 1)..participant_count).map(move |j| {
                types::SocialConnection {
                    participant_a: input.participants[i].id.clone(),
                    participant_b: input.participants[j].id.clone(),
                    connection_strength: 0.3 + (rand::random::<f32>() * 0.7),
                    interaction_count: (interaction_count / participant_count).max(1) as u32,
                }
            })
        }).collect();

        Ok(types::SocialDynamics {
            group_cohesion: group_cohesion.min(1.0),
            communication_patterns,
            power_dynamics,
            social_network_structure,
        })
    }

    async fn evaluate_harmony_indicators(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<Vec<types::HarmonyIndicator>> {
        Ok(vec![
            types::HarmonyIndicator { name: "Communication Quality".to_string(), score: 0.75, trend: "stable".to_string(), description: "Overall quality of communication between participants".to_string() },
            types::HarmonyIndicator { name: "Emotional Safety".to_string(), score: 0.8, trend: "improving".to_string(), description: "Participants feel safe expressing emotions".to_string() },
            types::HarmonyIndicator { name: "Mutual Respect".to_string(), score: 0.7, trend: "stable".to_string(), description: "Level of respect shown between participants".to_string() },
            types::HarmonyIndicator { name: "Collaboration Effectiveness".to_string(), score: 0.65, trend: "needs_attention".to_string(), description: "How effectively participants work together".to_string() },
            types::HarmonyIndicator { name: "Conflict Resolution".to_string(), score: 0.6, trend: "improving".to_string(), description: "Ability to resolve disagreements constructively".to_string() },
        ])
    }

    fn calculate_harmony_score(
        &self, 
        emotional_landscape: &types::EmotionalLandscape, 
        social_dynamics: &types::SocialDynamics, 
        harmony_indicators: &[types::HarmonyIndicator]
    ) -> f32 {
        let emotional_score = emotional_landscape.emotional_stability * 0.3;
        let social_score = social_dynamics.group_cohesion * 0.3;

        let indicator_score: f32 = if harmony_indicators.is_empty() {
            0.0
        } else {
            harmony_indicators.iter().map(|i| i.score).sum::<f32>() / harmony_indicators.len() as f32 * 0.4
        };

        (emotional_score + social_score + indicator_score).min(1.0)
    }

    async fn assess_emotional_awareness(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<f32> {
        let emotional_terms = ["feel", "feeling", "emotion", "upset", "happy", "sad", "angry", "worried", "anxious", "excited", "frustrated"];
        let lower = input.social_context.to_lowercase();
        let term_count = emotional_terms.iter().filter(|t| lower.contains(*t)).count();
        let awareness = 0.3 + (term_count as f32 / emotional_terms.len() as f32) * 0.7;
        Ok(awareness.min(1.0))
    }

    async fn evaluate_emotional_regulation(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<f32> {
        let regulation_terms = ["calm", "composed", "steady", "patient", "mindful", "breathe", "collected", "balanced"];
        let lower = input.social_context.to_lowercase();
        let term_count = regulation_terms.iter().filter(|t| lower.contains(*t)).count();
        let regulation = 0.4 + (term_count as f32 / regulation_terms.len() as f32) * 0.6;
        Ok(regulation.min(1.0))
    }

    async fn analyze_social_awareness(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<f32> {
        let social_terms = ["understand", "perspective", "consider", "team", "together", "we", "us", "collaborate", "support", "listen"];
        let lower = input.social_context.to_lowercase();
        let term_count = social_terms.iter().filter(|t| lower.contains(*t)).count();
        let awareness = 0.4 + (term_count as f32 / social_terms.len() as f32) * 0.6;
        Ok(awareness.min(1.0))
    }

    async fn assess_relationship_management(&self, input: &HarmonyWeaverTaskInput) -> AgentResult<f32> {
        let relationship_terms = ["trust", "respect", "appreciate", "thank", "help", "support", "together", "friend", "partner", "relationship"];
        let lower = input.social_context.to_lowercase();
        let term_count = relationship_terms.iter().filter(|t| lower.contains(*t)).count();
        let management = 0.3 + (term_count as f32 / relationship_terms.len() as f32) * 0.7;
        Ok(management.min(1.0))
    }

    async fn identify_intervention_opportunities(
        &self, 
        harmony_assessment: &HarmonyAssessment, 
        emotional_intelligence_analysis: &types::EmotionalIntelligenceAnalysis
    ) -> AgentResult<Vec<types::InterventionOpportunity>> {
        let mut opportunities = Vec::new();

        if harmony_assessment.harmony_score < 0.6 {
            opportunities.push(types::InterventionOpportunity {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Overall harmony score is low. Facilitate structured communication session.".to_string(),
                impact_score: 0.8,
                urgency: 0.7,
                target_participants: vec![],
                suggested_approach: "Organize a facilitated dialogue focusing on shared goals".to_string(),
            });
        }

        if harmony_assessment.emotional_landscape.emotional_stability < 0.5 {
            opportunities.push(types::InterventionOpportunity {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Emotional stability is low. Implement emotional regulation exercises.".to_string(),
                impact_score: 0.75,
                urgency: 0.8,
                target_participants: vec![],
                suggested_approach: "Introduce mindfulness and emotional check-in practices".to_string(),
            });
        }

        if harmony_assessment.social_dynamics.group_cohesion < 0.6 {
            opportunities.push(types::InterventionOpportunity {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Group cohesion needs improvement. Plan team-building activities.".to_string(),
                impact_score: 0.7,
                urgency: 0.5,
                target_participants: vec![],
                suggested_approach: "Organize collaborative projects with shared deliverables".to_string(),
            });
        }

        if emotional_intelligence_analysis.emotional_awareness < 0.6 {
            opportunities.push(types::InterventionOpportunity {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Emotional awareness is low. Provide emotional intelligence training.".to_string(),
                impact_score: 0.65,
                urgency: 0.6,
                target_participants: vec![],
                suggested_approach: "Offer workshops on emotional recognition and expression".to_string(),
            });
        }

        Ok(opportunities)
    }

    async fn prioritize_interventions(&self, opportunities: &[types::InterventionOpportunity]) -> AgentResult<Vec<types::PrioritizedIntervention>> {
        let mut prioritized: Vec<types::PrioritizedIntervention> = opportunities.iter().enumerate().map(|(i, opp)| {
            let priority_score = opp.impact_score * 0.6 + opp.urgency * 0.4;
            types::PrioritizedIntervention {
                opportunity_id: opp.id.clone(),
                priority_order: (i + 1) as u32,
                priority_score,
                rationale: format!("Impact score {:.2} and urgency {:.2} combine for priority {:.2}", opp.impact_score, opp.urgency, priority_score),
                estimated_effort: if opp.impact_score > 0.7 { "High".to_string() } else { "Medium".to_string() },
                expected_impact: if opp.impact_score > 0.7 { "Significant improvement expected".to_string() } else { "Moderate improvement expected".to_string() },
            }
        }).collect();

        prioritized.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap_or(std::cmp::Ordering::Equal));
        for (i, p) in prioritized.iter_mut().enumerate() {
            p.priority_order = (i + 1) as u32;
        }

        Ok(prioritized)
    }

    async fn generate_implementation_strategies(&self, interventions: &[types::PrioritizedIntervention]) -> AgentResult<Vec<types::ImplementationStrategy>> {
        Ok(interventions.iter().map(|intervention| {
            types::ImplementationStrategy {
                intervention_id: intervention.opportunity_id.clone(),
                steps: vec![
                    "Assess current state and gather baseline data".to_string(),
                    "Communicate intervention plan to stakeholders".to_string(),
                    "Execute intervention with defined approach".to_string(),
                    "Monitor progress and collect feedback".to_string(),
                    "Adjust strategy based on outcomes".to_string(),
                ],
                timeline: format!("{} weeks", (intervention.priority_order as f32 * 2.0).ceil() as u32),
                resources_needed: vec!["Facilitator".to_string(), "Meeting time".to_string(), "Feedback tools".to_string()],
                success_criteria: vec!["Improved harmony score".to_string(), "Positive participant feedback".to_string()],
                risk_factors: vec!["Participant resistance".to_string(), "Time constraints".to_string()],
            }
        }).collect())
    }

    async fn identify_conflicts(&self, input: &HarmonyWeaverTaskInput, harmony_assessment: &HarmonyAssessment) -> AgentResult<Vec<types::IdentifiedConflict>> {
        let text = &input.social_context;
        let conflict_keywords = ["disagree", "conflict", "argue", "problem", "issue", "blame", "frustrated", "tension", "dispute", "against"];
        let lower = text.to_lowercase();

        let keyword_count = conflict_keywords.iter().filter(|k| lower.contains(*k)).count();
        let severity = (keyword_count as f32 / conflict_keywords.len() as f32).min(1.0);

        let mut conflicts = Vec::new();
        if severity > 0.1 {
            conflicts.push(types::IdentifiedConflict {
                id: uuid::Uuid::new_v4().to_string(),
                description: format!("Detected {} conflict-related terms in social context", keyword_count),
                severity,
                parties_involved: input.participants.iter().map(|p| p.id.clone()).collect(),
                root_cause: if severity > 0.5 { "Communication breakdown and differing expectations".to_string() } else { "Minor misunderstandings".to_string() },
                impact_on_harmony: severity * (1.0 - harmony_assessment.harmony_score),
            });
        }

        Ok(conflicts)
    }

    async fn analyze_conflict_dynamics(&self, conflicts: &[types::IdentifiedConflict]) -> AgentResult<Vec<types::ConflictDynamics>> {
        Ok(conflicts.iter().map(|c| {
            types::ConflictDynamics {
                conflict_id: c.id.clone(),
                escalation_level: c.severity,
                communication_breakdown: vec!["Reduced information sharing".to_string(), "Defensive communication".to_string()],
                emotional_charge: (c.severity * 1.2).min(1.0),
                stalemate_duration: None,
            }
        }).collect())
    }

    async fn develop_resolution_approaches(&self, dynamics: &[types::ConflictDynamics]) -> AgentResult<Vec<types::ResolutionApproach>> {
        Ok(dynamics.iter().map(|d| {
            let approach_type = if d.escalation_level > 0.7 {
                types::ResolutionApproachType::Mediation
            } else if d.escalation_level > 0.4 {
                types::ResolutionApproachType::Negotiation
            } else {
                types::ResolutionApproachType::Collaboration
            };

            types::ResolutionApproach {
                conflict_id: d.conflict_id.clone(),
                approach_type,
                description: format!("{:?} approach for conflict resolution", approach_type),
                expected_effectiveness: 1.0 - d.escalation_level * 0.5,
                required_actions: vec![
                    "Acknowledge all parties' perspectives".to_string(),
                    "Identify common ground".to_string(),
                    "Develop mutually acceptable solution".to_string(),
                ],
                mediator_needed: d.escalation_level > 0.6,
            }
        }).collect())
    }

    async fn define_optimization_goals(&self, harmony_assessment: &HarmonyAssessment) -> AgentResult<Vec<types::OptimizationGoal>> {
        let mut goals = Vec::new();

        if harmony_assessment.harmony_score < 0.8 {
            goals.push(types::OptimizationGoal {
                id: uuid::Uuid::new_v4().to_string(),
                description: format!("Improve overall harmony score from {:.2} to 0.85", harmony_assessment.harmony_score),
                target_score: 0.85,
                priority: "High".to_string(),
                target_date: None,
                dependencies: vec![],
            });
        }

        if harmony_assessment.emotional_landscape.emotional_stability < 0.7 {
            goals.push(types::OptimizationGoal {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Increase emotional stability to 0.7 or higher".to_string(),
                target_score: 0.7,
                priority: "Medium".to_string(),
                target_date: None,
                dependencies: vec!["Improve overall harmony score".to_string()],
            });
        }

        if harmony_assessment.social_dynamics.group_cohesion < 0.7 {
            goals.push(types::OptimizationGoal {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Enhance group cohesion to 0.7 or higher".to_string(),
                target_score: 0.7,
                priority: "Medium".to_string(),
                target_date: None,
                dependencies: vec!["Increase emotional stability".to_string()],
            });
        }

        Ok(goals)
    }

    async fn create_action_plan(&self, recommendations: &InterventionRecommendations) -> AgentResult<Vec<types::ActionPlanItem>> {
        Ok(recommendations.prioritized_interventions.iter().enumerate().map(|(i, intervention)| {
            types::ActionPlanItem {
                goal_id: intervention.opportunity_id.clone(),
                step_number: (i + 1) as u32,
                description: format!("Execute prioritized intervention #{}", intervention.priority_order),
                assigned_to: vec![],
                deadline: None,
                status: types::ActionStatus::NotStarted,
            }
        }).collect())
    }

    async fn define_success_metrics(&self, goals: &[types::OptimizationGoal]) -> AgentResult<Vec<types::SuccessMetric>> {
        Ok(goals.iter().map(|goal| {
            types::SuccessMetric {
                goal_id: goal.id.clone(),
                name: format!("Progress toward {}", goal.description),
                target_value: goal.target_score,
                current_value: 0.0,
                measurement_method: "Harmony assessment survey and behavioral observation".to_string(),
                evaluation_frequency: "Weekly".to_string(),
            }
        }).collect())
    }
}

impl Default for HarmonyWeaverAgent {
    fn default() -> Self {
        Self {
            config: HarmonyWeaverConfig::default(),
            emotional_intelligence_capabilities: EmotionalIntelligenceCapabilities::default(),
            social_harmony_optimization: SocialHarmonyOptimization::default(),
            conflict_resolution: ConflictResolution::default(),
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
impl BaseAgent for HarmonyWeaverAgent {
    type Config = HarmonyWeaverConfig;
    type Input = HarmonyWeaverTaskInput;
    type Output = HarmonyWeaverTaskOutput;

    async fn process(&self, input: Self::Input) -> AgentResult<Self::Output> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        self.validate_input(&input)?;
        
        // Assess harmony
        let harmony_assessment = self.assess_harmony(&input).await?;
        
        // Analyze emotional intelligence
        let emotional_intelligence_analysis = self.analyze_emotional_intelligence(&input).await?;
        
        // Generate intervention recommendations
        let intervention_recommendations = self.generate_intervention_recommendations(&input, &harmony_assessment, &emotional_intelligence_analysis).await?;
        
        // Develop conflict resolution strategies
        let conflict_resolution_strategies = self.develop_conflict_resolution_strategies(&input, &harmony_assessment).await?;
        
        // Create harmony optimization plan
        let harmony_optimization_plan = self.create_harmony_optimization_plan(&input, &harmony_assessment, &intervention_recommendations).await?;
        
        // Build output
        let output = HarmonyWeaverTaskOutput {
            harmony_assessment,
            emotional_intelligence_analysis,
            intervention_recommendations,
            conflict_resolution_strategies,
            harmony_optimization_plan,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        };
        
        Ok(output)
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
                name: "emotional_intelligence".to_string(),
                description: "Emotional intelligence and social harmony optimization".to_string(),
                enabled: true,
            },
            AgentCapability {
                name: "harmony_assessment".to_string(),
                description: "Comprehensive harmony assessment and analysis".to_string(),
                enabled: true,
            },
            AgentCapability {
                name: "conflict_resolution".to_string(),
                description: "Advanced conflict resolution strategies".to_string(),
                enabled: true,
            },
            AgentCapability {
                name: "social_optimization".to_string(),
                description: "Social harmony optimization and improvement".to_string(),
                enabled: true,
            },
        ]
    }
}

// Re-export all types for backward compatibility
pub use config::*;
pub use optimization::*;
pub use capabilities::*;
pub use types::*;
