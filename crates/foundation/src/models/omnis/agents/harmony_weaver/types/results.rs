use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionRecommendations {
    pub intervention_opportunities: Vec<InterventionOpportunity>,
    pub prioritized_interventions: Vec<PrioritizedIntervention>,
    pub implementation_strategies: Vec<ImplementationStrategy>,
    pub expected_outcomes: Vec<String>,
    pub risk_assessment: Vec<String>,
    pub recommendation_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionOpportunity {
    pub id: String,
    pub description: String,
    pub impact_score: f32,
    pub urgency: f32,
    pub target_participants: Vec<String>,
    pub suggested_approach: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizedIntervention {
    pub opportunity_id: String,
    pub priority_order: u32,
    pub priority_score: f32,
    pub rationale: String,
    pub estimated_effort: String,
    pub expected_impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationStrategy {
    pub intervention_id: String,
    pub steps: Vec<String>,
    pub timeline: String,
    pub resources_needed: Vec<String>,
    pub success_criteria: Vec<String>,
    pub risk_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionStrategies {
    pub identified_conflicts: Vec<IdentifiedConflict>,
    pub conflict_dynamics: Vec<ConflictDynamics>,
    pub resolution_approaches: Vec<ResolutionApproach>,
    pub success_probability: f32,
    pub implementation_timeline: Vec<String>,
    pub strategy_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedConflict {
    pub id: String,
    pub description: String,
    pub severity: f32,
    pub parties_involved: Vec<String>,
    pub root_cause: String,
    pub impact_on_harmony: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDynamics {
    pub conflict_id: String,
    pub escalation_level: f32,
    pub communication_breakdown: Vec<String>,
    pub emotional_charge: f32,
    pub stalemate_duration: Option<chrono::Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionApproach {
    pub conflict_id: String,
    pub approach_type: ResolutionApproachType,
    pub description: String,
    pub expected_effectiveness: f32,
    pub required_actions: Vec<String>,
    pub mediator_needed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionApproachType {
    Mediation,
    Negotiation,
    Collaboration,
    Accommodation,
    Compromise,
    Avoidance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyOptimizationPlan {
    pub optimization_goals: Vec<OptimizationGoal>,
    pub action_plan: Vec<ActionPlanItem>,
    pub success_metrics: Vec<SuccessMetric>,
    pub timeline: Vec<String>,
    pub resources_required: Vec<String>,
    pub plan_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationGoal {
    pub id: String,
    pub description: String,
    pub target_score: f32,
    pub priority: String,
    pub target_date: Option<chrono::DateTime<chrono::Utc>>,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlanItem {
    pub goal_id: String,
    pub step_number: u32,
    pub description: String,
    pub assigned_to: Vec<String>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub status: ActionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionStatus {
    NotStarted,
    InProgress,
    Completed,
    Blocked,
    Deferred,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetric {
    pub goal_id: String,
    pub name: String,
    pub target_value: f32,
    pub current_value: f32,
    pub measurement_method: String,
    pub evaluation_frequency: String,
}
