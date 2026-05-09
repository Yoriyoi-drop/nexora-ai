//! Emotional Intelligence Framework

use serde::{Deserialize, Serialize};

/// Emotional Intelligence Framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalIntelligenceFramework {
    /// Self-awareness capabilities
    pub self_awareness: SelfAwareness,
    /// Self-regulation capabilities
    pub self_regulation: SelfRegulation,
    /// Social awareness capabilities
    pub social_awareness: SocialAwareness,
    /// Relationship management capabilities
    pub relationship_management: RelationshipManagement,
    /// Empathy models
    pub empathy_models: Vec<EmpathyModel>,
}

impl Default for EmotionalIntelligenceFramework {
    fn default() -> Self {
        Self {
            self_awareness: SelfAwareness::default(),
            self_regulation: SelfRegulation::default(),
            social_awareness: SocialAwareness::default(),
            relationship_management: RelationshipManagement::default(),
            empathy_models: vec![
                EmpathyModel::CognitiveEmpathyModel,
                EmpathyModel::AffectiveEmpathyModel,
                EmpathyModel::CompassionateEmpathyModel,
            ],
        }
    }
}

/// Self-Awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAwareness {
    /// Emotional recognition accuracy
    pub emotional_recognition_accuracy: f32,
    /// Emotional vocabulary
    pub emotional_vocabulary: Vec<String>,
    /// Emotional pattern recognition
    pub emotional_pattern_recognition: EmotionalPatternRecognition,
    /// Self-reflection capabilities
    pub self_reflection_capabilities: SelfReflectionCapabilities,
}

impl Default for SelfAwareness {
    fn default() -> Self {
        Self {
            emotional_recognition_accuracy: 0.85,
            emotional_vocabulary: vec![
                "joy".to_string(),
                "sadness".to_string(),
                "anger".to_string(),
                "fear".to_string(),
                "surprise".to_string(),
                "disgust".to_string(),
                "anticipation".to_string(),
                "trust".to_string(),
            ],
            emotional_pattern_recognition: EmotionalPatternRecognition::default(),
            self_reflection_capabilities: SelfReflectionCapabilities::default(),
        }
    }
}

/// Emotional Pattern Recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalPatternRecognition {
    /// Pattern detection algorithms
    pub pattern_detection_algorithms: Vec<String>,
    /// Pattern accuracy
    pub pattern_accuracy: f32,
    /// Learning rate
    pub learning_rate: f32,
}

impl Default for EmotionalPatternRecognition {
    fn default() -> Self {
        Self {
            pattern_detection_algorithms: vec![
                "neural_network_based".to_string(),
                "statistical_analysis".to_string(),
                "temporal_pattern_recognition".to_string(),
            ],
            pattern_accuracy: 0.8,
            learning_rate: 0.1,
        }
    }
}

/// Self Reflection Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfReflectionCapabilities {
    /// Reflection depth
    pub reflection_depth: ReflectionDepth,
    /// Insight generation
    pub insight_generation: InsightGeneration,
    /// Metacognitive awareness
    pub metacognitive_awareness: MetacognitiveAwareness,
}

impl Default for SelfReflectionCapabilities {
    fn default() -> Self {
        Self {
            reflection_depth: ReflectionDepth::Deep,
            insight_generation: InsightGeneration::default(),
            metacognitive_awareness: MetacognitiveAwareness::default(),
        }
    }
}

/// Reflection Depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflectionDepth {
    /// Surface level
    Surface,
    /// Moderate depth
    Moderate,
    /// Deep reflection
    Deep,
    /// Profound reflection
    Profound,
}

/// Insight Generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightGeneration {
    /// Insight quality
    pub insight_quality: f32,
    /// Insight frequency
    pub insight_frequency: f32,
    /// Insight relevance
    pub insight_relevance: f32,
}

impl Default for InsightGeneration {
    fn default() -> Self {
        Self {
            insight_quality: 0.8,
            insight_frequency: 0.7,
            insight_relevance: 0.85,
        }
    }
}

/// Metacognitive Awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveAwareness {
    /// Thinking about thinking
    pub thinking_about_thinking: f32,
    /// Cognitive monitoring
    pub cognitive_monitoring: f32,
    /// Strategic planning
    pub strategic_planning: f32,
}

impl Default for MetacognitiveAwareness {
    fn default() -> Self {
        Self {
            thinking_about_thinking: 0.8,
            cognitive_monitoring: 0.75,
            strategic_planning: 0.7,
        }
    }
}

/// Self-Regulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfRegulation {
    /// Emotional control
    pub emotional_control: EmotionalControl,
    /// Impulse management
    pub impulse_management: ImpulseManagement,
    /// Stress management
    pub stress_management: StressManagement,
    /// Adaptability
    pub adaptability: Adaptability,
}

impl Default for SelfRegulation {
    fn default() -> Self {
        Self {
            emotional_control: EmotionalControl::default(),
            impulse_management: ImpulseManagement::default(),
            stress_management: StressManagement::default(),
            adaptability: Adaptability::default(),
        }
    }
}

/// Emotional Control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalControl {
    /// Emotional regulation strategies
    pub emotional_regulation_strategies: Vec<EmotionalRegulationStrategy>,
    /// Control effectiveness
    pub control_effectiveness: f32,
    /// Response latency
    pub response_latency: f32,
}

impl Default for EmotionalControl {
    fn default() -> Self {
        Self {
            emotional_regulation_strategies: vec![
                EmotionalRegulationStrategy {
                    strategy_name: "cognitive_reappraisal".to_string(),
                    effectiveness: 0.85,
                    usage_frequency: 0.7,
                },
                EmotionalRegulationStrategy {
                    strategy_name: "mindfulness".to_string(),
                    effectiveness: 0.8,
                    usage_frequency: 0.6,
                },
            ],
            control_effectiveness: 0.8,
            response_latency: 0.5,
        }
    }
}

/// Emotional Regulation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalRegulationStrategy {
    /// Strategy name
    pub strategy_name: String,
    /// Effectiveness
    pub effectiveness: f32,
    /// Usage frequency
    pub usage_frequency: f32,
}

/// Impulse Management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseManagement {
    /// Impulse control strength
    pub impulse_control_strength: f32,
    /// Delay of gratification
    pub delay_of_gratification: f32,
    /// Decision making quality
    pub decision_making_quality: f32,
}

impl Default for ImpulseManagement {
    fn default() -> Self {
        Self {
            impulse_control_strength: 0.8,
            delay_of_gratification: 0.75,
            decision_making_quality: 0.85,
        }
    }
}

/// Stress Management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressManagement {
    /// Stress resilience
    pub stress_resilience: f32,
    /// Coping strategies
    pub coping_strategies: Vec<CopingStrategy>,
    /// Recovery speed
    pub recovery_speed: f32,
}

impl Default for StressManagement {
    fn default() -> Self {
        Self {
            stress_resilience: 0.8,
            coping_strategies: vec![
                CopingStrategy {
                    strategy_name: "problem_focused".to_string(),
                    effectiveness: 0.85,
                    applicability: vec!["work_stress".to_string(), "academic_stress".to_string()],
                },
                CopingStrategy {
                    strategy_name: "emotion_focused".to_string(),
                    effectiveness: 0.8,
                    applicability: vec!["relationship_stress".to_string(), "personal_stress".to_string()],
                },
            ],
            recovery_speed: 0.7,
        }
    }
}

/// Coping Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopingStrategy {
    /// Strategy name
    pub strategy_name: String,
    /// Effectiveness
    pub effectiveness: f32,
    /// Applicability
    pub applicability: Vec<String>,
}

/// Adaptability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adaptability {
    /// Cognitive flexibility
    pub cognitive_flexibility: f32,
    /// Behavioral adaptability
    pub behavioral_adaptability: f32,
    /// Emotional adaptability
    pub emotional_adaptability: f32,
}

impl Default for Adaptability {
    fn default() -> Self {
        Self {
            cognitive_flexibility: 0.85,
            behavioral_adaptability: 0.8,
            emotional_adaptability: 0.75,
        }
    }
}

/// Social Awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialAwareness {
    /// Empathy accuracy
    pub empathy_accuracy: f32,
    /// Social cue recognition
    pub social_cue_recognition: SocialCueRecognition,
    /// Cultural intelligence
    pub cultural_intelligence: CulturalIntelligence,
    /// Organizational awareness
    pub organizational_awareness: OrganizationalAwareness,
}

impl Default for SocialAwareness {
    fn default() -> Self {
        Self {
            empathy_accuracy: 0.85,
            social_cue_recognition: SocialCueRecognition::default(),
            cultural_intelligence: CulturalIntelligence::default(),
            organizational_awareness: OrganizationalAwareness::default(),
        }
    }
}

/// Social Cue Recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialCueRecognition {
    /// Non-verbal cue accuracy
    pub non_verbal_cue_accuracy: f32,
    /// Verbal cue accuracy
    pub verbal_cue_accuracy: f32,
    /// Context interpretation
    pub context_interpretation: f32,
}

impl Default for SocialCueRecognition {
    fn default() -> Self {
        Self {
            non_verbal_cue_accuracy: 0.8,
            verbal_cue_accuracy: 0.85,
            context_interpretation: 0.75,
        }
    }
}

/// Cultural Intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalIntelligence {
    /// Cultural knowledge
    pub cultural_knowledge: f32,
    /// Cultural sensitivity
    pub cultural_sensitivity: f32,
    /// Cultural adaptation
    pub cultural_adaptation: f32,
}

impl Default for CulturalIntelligence {
    fn default() -> Self {
        Self {
            cultural_knowledge: 0.8,
            cultural_sensitivity: 0.85,
            cultural_adaptation: 0.75,
        }
    }
}

/// Organizational Awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationalAwareness {
    /// Political awareness
    pub political_awareness: f32,
    /// Social network understanding
    pub social_network_understanding: f32,
    /// Group dynamics
    pub group_dynamics: f32,
}

impl Default for OrganizationalAwareness {
    fn default() -> Self {
        Self {
            political_awareness: 0.7,
            social_network_understanding: 0.8,
            group_dynamics: 0.75,
        }
    }
}

/// Relationship Management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipManagement {
    /// Communication skills
    pub communication_skills: CommunicationSkills,
    /// Conflict resolution
    pub conflict_resolution: ConflictResolution,
    /// Influence and persuasion
    pub influence_and_persuasion: InfluenceAndPersuasion,
    /// Teamwork and collaboration
    pub teamwork_and_collaboration: TeamworkAndCollaboration,
}

impl Default for RelationshipManagement {
    fn default() -> Self {
        Self {
            communication_skills: CommunicationSkills::default(),
            conflict_resolution: ConflictResolution::default(),
            influence_and_persuasion: InfluenceAndPersuasion::default(),
            teamwork_and_collaboration: TeamworkAndCollaboration::default(),
        }
    }
}

/// Communication Skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationSkills {
    /// Active listening
    pub active_listening: f32,
    /// Clarity of expression
    pub clarity_of_expression: f32,
    /// Emotional expression
    pub emotional_expression: f32,
}

impl Default for CommunicationSkills {
    fn default() -> Self {
        Self {
            active_listening: 0.85,
            clarity_of_expression: 0.8,
            emotional_expression: 0.75,
        }
    }
}

/// Conflict Resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    /// Mediation skills
    pub mediation_skills: f32,
    /// Negotiation effectiveness
    pub negotiation_effectiveness: f32,
    /// Win-win orientation
    pub win_win_orientation: f32,
}

impl Default for ConflictResolution {
    fn default() -> Self {
        Self {
            mediation_skills: 0.8,
            negotiation_effectiveness: 0.75,
            win_win_orientation: 0.85,
        }
    }
}

/// Influence and Persuasion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluenceAndPersuasion {
    /// Persuasive communication
    pub persuasive_communication: f32,
    /// Leadership presence
    pub leadership_presence: f32,
    /// Relationship building
    pub relationship_building: f32,
}

impl Default for InfluenceAndPersuasion {
    fn default() -> Self {
        Self {
            persuasive_communication: 0.75,
            leadership_presence: 0.8,
            relationship_building: 0.85,
        }
    }
}

/// Teamwork and Collaboration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamworkAndCollaboration {
    /// Cooperation level
    pub cooperation_level: f32,
    /// Shared understanding
    pub shared_understanding: f32,
    /// Mutual support
    pub mutual_support: f32,
}

impl Default for TeamworkAndCollaboration {
    fn default() -> Self {
        Self {
            cooperation_level: 0.85,
            shared_understanding: 0.8,
            mutual_support: 0.9,
        }
    }
}

/// Response Personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePersonalization {
    /// Personalization level
    pub personalization_level: PersonalizationLevel,
    /// User preferences
    pub user_preferences: UserPreferences,
    /// Context adaptation
    pub context_adaptation: ContextAdaptation,
    /// Learning adaptation
    pub learning_adaptation: LearningAdaptation,
}

impl Default for ResponsePersonalization {
    fn default() -> Self {
        Self {
            personalization_level: PersonalizationLevel::High,
            user_preferences: UserPreferences::default(),
            context_adaptation: ContextAdaptation::default(),
            learning_adaptation: LearningAdaptation::default(),
        }
    }
}

/// Personalization Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersonalizationLevel {
    /// No personalization
    None,
    /// Basic personalization
    Basic,
    /// Moderate personalization
    Moderate,
    /// High personalization
    High,
    /// Adaptive personalization
    Adaptive,
}

/// User Preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Communication style preference
    pub communication_style_preference: CommunicationStylePreference,
    /// Response length preference
    pub response_length_preference: ResponseLengthPreference,
    /// Tone preference
    pub tone_preference: TonePreference,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            communication_style_preference: CommunicationStylePreference::Balanced,
            response_length_preference: ResponseLengthPreference::Medium,
            tone_preference: TonePreference::Supportive,
        }
    }
}

/// Communication Style Preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStylePreference {
    /// Formal style
    Formal,
    /// Informal style
    Informal,
    /// Professional style
    Professional,
    /// Casual style
    Casual,
    /// Balanced style
    Balanced,
}

/// Response Length Preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseLengthPreference {
    /// Short responses
    Short,
    /// Medium responses
    Medium,
    /// Long responses
    Long,
    /// Adaptive responses
    Adaptive,
}

/// Tone Preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TonePreference {
    /// Supportive tone
    Supportive,
    /// Direct tone
    Direct,
    /// Empathetic tone
    Empathetic,
    /// Analytical tone
    Analytical,
    /// Encouraging tone
    Encouraging,
}

/// Context Adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAdaptation {
    /// Situational awareness
    pub situational_awareness: f32,
    /// Cultural adaptation
    pub cultural_adaptation: f32,
    /// Emotional adaptation
    pub emotional_adaptation: f32,
}

impl Default for ContextAdaptation {
    fn default() -> Self {
        Self {
            situational_awareness: 0.85,
            cultural_adaptation: 0.8,
            emotional_adaptation: 0.75,
        }
    }
}

/// Learning Adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningAdaptation {
    /// Feedback incorporation
    pub feedback_incorporation: f32,
    /// Pattern learning
    pub pattern_learning: f32,
    /// Preference evolution
    pub preference_evolution: f32,
}

impl Default for LearningAdaptation {
    fn default() -> Self {
        Self {
            feedback_incorporation: 0.8,
            pattern_learning: 0.75,
            preference_evolution: 0.7,
        }
    }
}

/// Empathy Model (re-export from empathy_model module)
pub use super::empathy_model::EmpathyModel;
