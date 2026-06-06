//! Harmony Weaver Input and Output Types

use serde::{Deserialize, Serialize};

/// Harmony Weaver Task Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyWeaverTaskInput {
    /// Social context
    pub social_context: SocialContext,
    /// Group dynamics
    pub group_dynamics: GroupDynamics,
    /// Cultural context
    pub cultural_context: CulturalContext,
    /// Emotional landscape
    pub emotional_landscape: EmotionalLandscape,
    /// Communication patterns
    pub communication_patterns: Vec<CommunicationPattern>,
    /// Conflict indicators
    pub conflict_indicators: Vec<ConflictIndicator>,
    /// Harmony goals
    pub harmony_goals: Vec<HarmonyGoal>,
    /// Context information
    pub context_information: ContextInformation,
    /// Task requirements
    pub task_requirements: TaskRequirements,
}

/// Social Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialContext {
    /// Group size
    pub group_size: u32,
    /// Group composition
    pub group_composition: GroupComposition,
    /// Social structure
    pub social_structure: SocialStructure,
    /// Power dynamics
    pub power_dynamics: PowerDynamics,
    /// Relationship patterns
    pub relationship_patterns: Vec<RelationshipPattern>,
}

impl SocialContext {
    /// Check if the social context is empty
    pub fn is_empty(&self) -> bool {
        self.group_size == 0 && self.relationship_patterns.is_empty()
    }
}

/// Group Composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupComposition {
    /// Demographics
    pub demographics: Demographics,
    /// Cultural diversity
    pub cultural_diversity: f32,
    /// Skill diversity
    pub skill_diversity: f32,
    /// Personality diversity
    pub personality_diversity: f32,
    /// Experience levels
    pub experience_levels: Vec<ExperienceLevel>,
}

/// Demographics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographics {
    /// Age range
    pub age_range: AgeRange,
    /// Gender distribution
    pub gender_distribution: std::collections::HashMap<String, f32>,
    /// Cultural backgrounds
    pub cultural_backgrounds: Vec<String>,
    /// Language diversity
    pub language_diversity: Vec<String>,
    /// Educational backgrounds
    pub educational_backgrounds: Vec<String>,
}

/// Age Range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeRange {
    /// Minimum age
    pub min_age: u32,
    /// Maximum age
    pub max_age: u32,
    /// Average age
    pub average_age: f32,
}

/// Experience Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceLevel {
    /// Level name
    pub level_name: String,
    /// Count
    pub count: u32,
    /// Percentage
    pub percentage: f32,
}

/// Social Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialStructure {
    /// Hierarchy type
    pub hierarchy_type: HierarchyType,
    /// Formal roles
    pub formal_roles: Vec<FormalRole>,
    /// Informal roles
    pub informal_roles: Vec<InformalRole>,
    /// Communication channels
    pub communication_channels: Vec<CommunicationChannel>,
    /// Decision making process
    pub decision_making_process: DecisionMakingProcess,
}

/// Hierarchy Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchyType {
    /// Flat hierarchy
    Flat,
    /// Tall hierarchy
    Tall,
    /// Matrix hierarchy
    Matrix,
    /// Network hierarchy
    Network,
    /// Circular hierarchy
    Circular,
}

/// Formal Role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalRole {
    /// Role title
    pub role_title: String,
    /// Responsibilities
    pub responsibilities: Vec<String>,
    /// Authority level
    pub authority_level: f32,
    /// Decision making power
    pub decision_making_power: f32,
}

/// Informal Role
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformalRole {
    /// Role name
    pub role_name: String,
    /// Role description
    pub role_description: String,
    /// Influence level
    pub influence_level: f32,
    /// Recognition level
    pub recognition_level: f32,
}

/// Communication Channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationChannel {
    /// Channel type
    pub channel_type: ChannelType,
    /// Frequency
    pub frequency: CommunicationFrequency,
    /// Effectiveness
    pub effectiveness: f32,
    /// Participants
    pub participants: Vec<String>,
}

/// Channel Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    /// Formal meetings
    FormalMeetings,
    /// Informal meetings
    InformalMeetings,
    /// Digital communication
    DigitalCommunication,
    /// One-on-one conversations
    OneOnOneConversations,
    /// Group discussions
    GroupDiscussions,
}

/// Communication Frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationFrequency {
    /// Daily
    Daily,
    /// Weekly
    Weekly,
    /// Bi-weekly
    BiWeekly,
    /// Monthly
    Monthly,
    /// As needed
    AsNeeded,
}

/// Decision Making Process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMakingProcess {
    /// Process type
    pub process_type: DecisionProcessType,
    /// Participation level
    pub participation_level: f32,
    /// Consensus requirement
    pub consensus_requirement: f32,
    /// Veto power
    pub veto_power: Vec<String>,
}

/// Decision Process Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionProcessType {
    /// Autocratic
    Autocratic,
    /// Democratic
    Democratic,
    /// Consensus
    Consensus,
    /// Majority vote
    MajorityVote,
    /// Consultative
    Consultative,
}

/// Power Dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDynamics {
    /// Power distribution
    pub power_distribution: PowerDistribution,
    /// Influence patterns
    pub influence_patterns: Vec<InfluencePattern>,
    /// Authority structures
    pub authority_structures: Vec<AuthorityStructure>,
    /// Power struggles
    pub power_struggles: Vec<PowerStruggle>,
}

/// Power Distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDistribution {
    /// Distribution type
    pub distribution_type: PowerDistributionType,
    /// Equality level
    pub equality_level: f32,
    /// Transparency level
    pub transparency_level: f32,
    /// Accountability level
    pub accountability_level: f32,
}

/// PowerDistributionType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerDistributionType {
    /// Centralized
    Centralized,
    /// Decentralized
    Decentralized,
    /// Distributed
    Distributed,
    /// Hierarchical
    Hierarchical,
    /// Networked
    Networked,
}

/// Influence Pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluencePattern {
    /// Pattern type
    pub pattern_type: InfluencePatternType,
    /// Actors involved
    pub actors_involved: Vec<String>,
    /// Influence strength
    pub influence_strength: f32,
    /// Context
    pub context: String,
}

/// InfluencePatternType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfluencePatternType {
    /// Expertise-based
    ExpertiseBased,
    /// Relationship-based
    RelationshipBased,
    /// Position-based
    PositionBased,
    /// Information-based
    InformationBased,
    /// Charisma-based
    CharismaBased,
}

/// Authority Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorityStructure {
    /// Structure type
    pub structure_type: String,
    /// Legitimacy source
    pub legitimacy_source: String,
    /// Scope of authority
    pub scope_of_authority: String,
    /// Limitations
    pub limitations: Vec<String>,
}

/// Power Struggle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerStruggle {
    /// Struggle type
    pub struggle_type: String,
    /// Parties involved
    pub parties_involved: Vec<String>,
    /// Intensity level
    pub intensity_level: f32,
    /// Impact on harmony
    pub impact_on_harmony: f32,
}

/// Relationship Pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipPattern {
    /// Pattern type
    pub pattern_type: RelationshipPatternType,
    /// Actors
    pub actors: Vec<String>,
    /// Quality
    pub quality: f32,
    /// Stability
    pub stability: f32,
    /// Impact on group
    pub impact_on_group: f32,
}

/// RelationshipPatternType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipPatternType {
    /// Collaborative
    Collaborative,
    /// Competitive
    Competitive,
    /// Supportive
    Supportive,
    /// Conflictual
    Conflictual,
    /// Dependent
    Dependent,
    /// Independent
    Independent,
}

/// Group Dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDynamics {
    /// Cohesion level
    pub cohesion_level: f32,
    /// Communication patterns
    pub communication_patterns: Vec<CommunicationPattern>,
    /// Conflict patterns
    pub conflict_patterns: Vec<ConflictPattern>,
    /// Leadership dynamics
    pub leadership_dynamics: LeadershipDynamics,
    /// Norm evolution
    pub norm_evolution: NormEvolution,
}

// Re-export CommunicationPattern from harmony_types
pub use super::harmony_types::CommunicationPattern;
// Re-export SuccessMetric and MeasurementFrequency from config
pub use crate::models::omnis::agents::harmony_weaver::config::strategies::{SuccessMetric, MeasurementFrequency};

/// ConflictPattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictPattern {
    /// Pattern ID
    pub pattern_id: String,
    /// Pattern type
    pub pattern_type: String,
    /// Triggers
    pub triggers: Vec<String>,
    /// Frequency
    pub frequency: f32,
    /// Resolution success rate
    pub resolution_success_rate: f32,
}

/// LeadershipDynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadershipDynamics {
    /// Leadership style
    pub leadership_style: LeadershipStyle,
    /// Leadership effectiveness
    pub leadership_effectiveness: f32,
    /// Succession planning
    pub succession_planning: f32,
    /// Follower satisfaction
    pub follower_satisfaction: f32,
}

/// LeadershipStyle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeadershipStyle {
    /// Autocratic
    Autocratic,
    /// Democratic
    Democratic,
    /// Transformational
    Transformational,
    /// Servant
    Servant,
    /// Laissez-faire
    LaissezFaire,
    /// Situational
    Situational,
}

/// NormEvolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormEvolution {
    /// Current norms
    pub current_norms: Vec<GroupNorm>,
    /// Emerging norms
    pub emerging_norms: Vec<GroupNorm>,
    /// Declining norms
    pub declining_norms: Vec<GroupNorm>,
    /// Norm enforcement level
    pub norm_enforcement_level: f32,
}

/// GroupNorm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupNorm {
    /// Norm description
    pub norm_description: String,
    /// Origin
    pub origin: String,
    /// Acceptance level
    pub acceptance_level: f32,
    /// Enforcement mechanism
    pub enforcement_mechanism: String,
}

/// Cultural Context (re-export from config)
pub use crate::models::omnis::agents::harmony_weaver::config::cultural::CulturalContext;

/// Emotional Landscape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalLandscape {
    /// Dominant emotions
    pub dominant_emotions: Vec<String>,
    /// Emotional intensity
    pub emotional_intensity: f32,
    /// Emotional diversity
    pub emotional_diversity: f32,
    /// Emotional stability
    pub emotional_stability: f32,
    /// Emotional contagion level
    pub emotional_contagion_level: f32,
}

/// Conflict Indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictIndicator {
    /// Indicator type
    pub indicator_type: ConflictIndicatorType,
    /// Severity level
    pub severity_level: f32,
    /// Description
    pub description: String,
    /// Parties involved
    pub parties_involved: Vec<String>,
    /// Impact assessment
    pub impact_assessment: String,
}

/// ConflictIndicatorType
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictIndicatorType {
    /// Communication breakdown
    CommunicationBreakdown,
    /// Value conflict
    ValueConflict,
    /// Resource competition
    ResourceCompetition,
    /// Power struggle
    PowerStruggle,
    /// Personality clash
    PersonalityClash,
    /// Role ambiguity
    RoleAmbiguity,
    /// Goal misalignment
    GoalMisalignment,
}

/// Harmony Goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyGoal {
    /// Goal ID
    pub goal_id: String,
    /// Goal description
    pub goal_description: String,
    /// Goal category
    pub goal_category: HarmonyGoalCategory,
    /// Priority level
    pub priority_level: PriorityLevel,
    /// Target outcome
    pub target_outcome: String,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Timeframe
    pub timeframe: String,
}

/// HarmonyGoalCategory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmonyGoalCategory {
    /// Relationship improvement
    RelationshipImprovement,
    /// Communication enhancement
    CommunicationEnhancement,
    /// Conflict reduction
    ConflictReduction,
    /// Cultural integration
    CulturalIntegration,
    /// Team cohesion
    TeamCohesion,
    /// Trust building
    TrustBuilding,
}

/// PriorityLevel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityLevel {
    /// Critical
    Critical,
    /// High
    High,
    /// Medium
    Medium,
    /// Low
    Low,
}

/// Context Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInformation {
    /// Organizational context
    pub organizational_context: OrganizationalContext,
    /// Environmental context
    pub environmental_context: EnvironmentalContext,
    /// Temporal context
    pub temporal_context: TemporalContext,
    /// External factors
    pub external_factors: Vec<ExternalFactor>,
}

/// OrganizationalContext
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationalContext {
    /// Organization type
    pub organization_type: String,
    /// Organizational culture
    pub organizational_culture: String,
    /// Business objectives
    pub business_objectives: Vec<String>,
    /// Performance metrics
    pub performance_metrics: Vec<String>,
}

/// EnvironmentalContext
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalContext {
    /// Physical environment
    pub physical_environment: String,
    /// Work environment
    pub work_environment: String,
    /// Technological environment
    pub technological_environment: String,
    /// Social environment
    pub social_environment: String,
}

/// TemporalContext
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Time period
    pub time_period: String,
    /// Seasonal factors
    pub seasonal_factors: Vec<String>,
    /// Time pressure level
    pub time_pressure_level: f32,
    /// Deadline proximity
    pub deadline_proximity: f32,
}

/// ExternalFactor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalFactor {
    /// Factor type
    pub factor_type: String,
    /// Description
    pub description: String,
    /// Impact level
    pub impact_level: f32,
    /// Controllability
    pub controllability: f32,
}

/// Task Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    /// Analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Intervention scope
    pub intervention_scope: InterventionScope,
    /// Cultural considerations
    pub cultural_considerations: Vec<String>,
    /// Time constraints
    pub time_constraints: TimeConstraints,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// AnalysisDepth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Surface level
    Surface,
    /// Moderate depth
    Moderate,
    /// Deep analysis
    Deep,
    /// Comprehensive
    Comprehensive,
}

/// InterventionScope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionScope {
    /// Individual level
    Individual,
    /// Team level
    Team,
    /// Department level
    Department,
    /// Organizational level
    Organizational,
    /// System level
    System,
}

/// TimeConstraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Available time
    pub available_time: f32,
    /// Time unit
    pub time_unit: String,
    /// Urgency level
    pub urgency_level: f32,
    /// Flexibility
    pub flexibility: f32,
}

/// ResourceConstraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Budget constraints
    pub budget_constraints: f64,
    /// Personnel constraints
    pub personnel_constraints: u32,
    /// Technology constraints
    pub technology_constraints: Vec<String>,
    /// Authority constraints
    pub authority_constraints: Vec<String>,
}

// Harmony Weaver Task Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyWeaverTaskOutput {
    /// Harmony assessment
    pub harmony_assessment: crate::models::omnis::agents::harmony_weaver::optimization::assessment::HarmonyAssessment,
    /// Emotional intelligence analysis
    pub emotional_intelligence_analysis: EmotionalIntelligenceAnalysis,
    /// Intervention recommendations
    pub intervention_recommendations: crate::models::omnis::agents::harmony_weaver::optimization::interventions::InterventionRecommendations,
    /// Conflict resolution strategies
    pub conflict_resolution_strategies: ConflictResolutionStrategies,
    /// Harmony optimization plan
    pub harmony_optimization_plan: crate::models::omnis::agents::harmony_weaver::optimization::planning::HarmonyOptimizationPlan,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Additional insights
    pub additional_insights: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Emotional Intelligence Analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalIntelligenceAnalysis {
    /// Emotional awareness
    pub emotional_awareness: f32,
    /// Emotional regulation
    pub emotional_regulation: f32,
    /// Social awareness
    pub social_awareness: f32,
    /// Relationship management
    pub relationship_management: f32,
    /// Overall EI score
    pub overall_ei_score: f32,
    /// Analysis timestamp
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

// Conflict Resolution Strategies
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConflictResolutionStrategies {
    /// Identified conflicts
    pub identified_conflicts: Vec<IdentifiedConflict>,
    /// Conflict dynamics
    pub conflict_dynamics: Vec<ConflictDynamics>,
    /// Resolution approaches
    pub resolution_approaches: Vec<ResolutionApproach>,
    /// Success probability
    pub success_probability: f32,
    /// Implementation timeline
    pub implementation_timeline: Vec<TimelineItem>,
    /// Strategy timestamp
    pub strategy_timestamp: chrono::DateTime<chrono::Utc>,
}

/// IdentifiedConflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedConflict {
    /// Conflict ID
    pub conflict_id: String,
    /// Conflict type
    pub conflict_type: String,
    /// Parties involved
    pub parties_involved: Vec<String>,
    /// Severity level
    pub severity_level: f32,
    /// Description
    pub description: String,
    /// Root causes
    pub root_causes: Vec<String>,
    /// Impact assessment
    pub impact_assessment: String,
}

/// ConflictDynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDynamics {
    /// Conflict ID
    pub conflict_id: String,
    /// Dynamic type
    pub dynamic_type: String,
    /// Escalation pattern
    pub escalation_pattern: String,
    /// De-escalation factors
    pub de_escalation_factors: Vec<String>,
    /// Intervention points
    pub intervention_points: Vec<String>,
}

/// ResolutionApproach
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionApproach {
    /// Approach ID
    pub approach_id: String,
    /// Approach name
    pub approach_name: String,
    /// Approach type
    pub approach_type: String,
    /// Description
    pub description: String,
    /// Success probability
    pub success_probability: f32,
    /// Implementation complexity
    pub implementation_complexity: f32,
    /// Resource requirements
    pub resource_requirements: Vec<String>,
    /// Time requirements
    pub time_requirements: f32,
}

/// TimelineItem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineItem {
    /// Item description
    pub item_description: String,
    /// Start date
    pub start_date: chrono::DateTime<chrono::Utc>,
    /// End date
    pub end_date: chrono::DateTime<chrono::Utc>,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Status
    pub status: String,
}
