//! Cultural Empathy Settings

use serde::{Deserialize, Serialize};

/// Cultural Empathy Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalEmpathySettings {
    /// Cultural awareness level
    pub cultural_awareness_level: CulturalAwarenessLevel,
    /// Cultural adaptation strategies
    pub cultural_adaptation_strategies: Vec<CulturalAdaptationStrategy>,
    /// Cross-cultural communication
    pub cross_cultural_communication: CrossCulturalCommunication,
    /// Cultural sensitivity
    pub cultural_sensitivity: CulturalSensitivity,
}

impl Default for CulturalEmpathySettings {
    fn default() -> Self {
        Self {
            cultural_awareness_level: CulturalAwarenessLevel::AdvancedAwareness,
            cultural_adaptation_strategies: vec![
                CulturalAdaptationStrategy {
                    strategy_id: "communication_adaptation".to_string(),
                    strategy_name: "Communication Adaptation".to_string(),
                    strategy_description: "Adapt communication style based on cultural context".to_string(),
                    strategy_type: CulturalAdaptationStrategyType::CommunicationAdaptation,
                    strategy_effectiveness: 0.85,
                },
                CulturalAdaptationStrategy {
                    strategy_id: "behavioral_adaptation".to_string(),
                    strategy_name: "Behavioral Adaptation".to_string(),
                    strategy_description: "Adjust behavior according to cultural norms".to_string(),
                    strategy_type: CulturalAdaptationStrategyType::BehavioralAdaptation,
                    strategy_effectiveness: 0.8,
                },
            ],
            cross_cultural_communication: CrossCulturalCommunication::default(),
            cultural_sensitivity: CulturalSensitivity::default(),
        }
    }
}

/// Cultural Awareness Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalAwarenessLevel {
    /// Basic awareness
    BasicAwareness,
    /// Intermediate awareness
    IntermediateAwareness,
    /// Advanced awareness
    AdvancedAwareness,
    /// Expert awareness
    ExpertAwareness,
    /// Master awareness
    MasterAwareness,
}

/// Cultural Adaptation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalAdaptationStrategy {
    /// Strategy ID
    pub strategy_id: String,
    /// Strategy name
    pub strategy_name: String,
    /// Strategy description
    pub strategy_description: String,
    /// Strategy type
    pub strategy_type: CulturalAdaptationStrategyType,
    /// Strategy effectiveness
    pub strategy_effectiveness: f32,
}

/// Cultural Adaptation Strategy Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalAdaptationStrategyType {
    /// Communication adaptation
    CommunicationAdaptation,
    /// Behavioral adaptation
    BehavioralAdaptation,
    /// Emotional adaptation
    EmotionalAdaptation,
    /// Cognitive adaptation
    CognitiveAdaptation,
}

/// Cross Cultural Communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCulturalCommunication {
    /// Communication styles
    pub communication_styles: Vec<CommunicationStyle>,
    /// Non-verbal communication
    pub non_verbal_communication: NonVerbalCommunication,
    /// Language considerations
    pub language_considerations: LanguageConsiderations,
    /// Context sensitivity
    pub context_sensitivity: ContextSensitivity,
}

impl Default for CrossCulturalCommunication {
    fn default() -> Self {
        Self {
            communication_styles: vec![
                CommunicationStyle {
                    style_id: "direct".to_string(),
                    style_name: "Direct Communication".to_string(),
                    culture: "Western".to_string(),
                    directness_level: DirectnessLevel::VeryDirect,
                    context_dependency: ContextDependency::LowContext,
                },
                CommunicationStyle {
                    style_id: "indirect".to_string(),
                    style_name: "Indirect Communication".to_string(),
                    culture: "Eastern".to_string(),
                    directness_level: DirectnessLevel::VeryIndirect,
                    context_dependency: ContextDependency::HighContext,
                },
            ],
            non_verbal_communication: NonVerbalCommunication::default(),
            language_considerations: LanguageConsiderations::default(),
            context_sensitivity: ContextSensitivity::default(),
        }
    }
}

/// Communication Style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    /// Style ID
    pub style_id: String,
    /// Style name
    pub style_name: String,
    /// Culture
    pub culture: String,
    /// Directness level
    pub directness_level: DirectnessLevel,
    /// Context dependency
    pub context_dependency: ContextDependency,
}

/// Directness Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectnessLevel {
    /// Very direct
    VeryDirect,
    /// Direct
    Direct,
    /// Moderate
    Moderate,
    /// Indirect
    Indirect,
    /// Very indirect
    VeryIndirect,
}

/// Context Dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextDependency {
    /// Low context
    LowContext,
    /// Medium context
    MediumContext,
    /// High context
    HighContext,
    /// Very high context
    VeryHighContext,
}

/// Non Verbal Communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonVerbalCommunication {
    /// Body language awareness
    pub body_language_awareness: f32,
    /// Eye contact norms
    pub eye_contact_norms: Vec<EyeContactNorm>,
    /// Personal space considerations
    pub personal_space_considerations: PersonalSpaceConsiderations,
    /// Gesture interpretations
    pub gesture_interpretations: Vec<GestureInterpretation>,
}

impl Default for NonVerbalCommunication {
    fn default() -> Self {
        Self {
            body_language_awareness: 0.8,
            eye_contact_norms: vec![
                EyeContactNorm {
                    culture: "Western".to_string(),
                    norm_type: EyeContactNormType::Direct,
                    appropriateness_level: 0.9,
                },
                EyeContactNorm {
                    culture: "Eastern".to_string(),
                    norm_type: EyeContactNormType::Indirect,
                    appropriateness_level: 0.85,
                },
            ],
            personal_space_considerations: PersonalSpaceConsiderations::default(),
            gesture_interpretations: vec![],
        }
    }
}

/// Eye Contact Norm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeContactNorm {
    /// Culture
    pub culture: String,
    /// Norm type
    pub norm_type: EyeContactNormType,
    /// Appropriateness level
    pub appropriateness_level: f32,
}

/// Eye Contact Norm Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EyeContactNormType {
    /// Direct eye contact
    Direct,
    /// Indirect eye contact
    Indirect,
    /// Minimal eye contact
    Minimal,
    /// Variable eye contact
    Variable,
}

/// Personal Space Considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalSpaceConsiderations {
    /// Personal distance
    pub personal_distance: f32,
    /// Social distance
    pub social_distance: f32,
    /// Cultural variations
    pub cultural_variations: Vec<CulturalSpaceVariation>,
}

impl Default for PersonalSpaceConsiderations {
    fn default() -> Self {
        Self {
            personal_distance: 1.0,
            social_distance: 2.0,
            cultural_variations: vec![
                CulturalSpaceVariation {
                    culture: "North American".to_string(),
                    personal_distance: 0.8,
                    social_distance: 1.5,
                },
                CulturalSpaceVariation {
                    culture: "Japanese".to_string(),
                    personal_distance: 1.2,
                    social_distance: 2.5,
                },
            ],
        }
    }
}

/// Cultural Space Variation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalSpaceVariation {
    /// Culture
    pub culture: String,
    /// Personal distance
    pub personal_distance: f32,
    /// Social distance
    pub social_distance: f32,
}

/// Gesture Interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureInterpretation {
    /// Gesture
    pub gesture: String,
    /// Culture
    pub culture: String,
    /// Meaning
    pub meaning: String,
    /// Context
    pub context: String,
}

/// Language Considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConsiderations {
    /// Language proficiency
    pub language_proficiency: Vec<LanguageProficiency>,
    /// Cultural nuances
    pub cultural_nuances: Vec<CulturalNuance>,
    /// Idiomatic expressions
    pub idiomatic_expressions: Vec<IdiomaticExpression>,
    /// Translation considerations
    pub translation_considerations: TranslationConsiderations,
}

impl Default for LanguageConsiderations {
    fn default() -> Self {
        Self {
            language_proficiency: vec![
                LanguageProficiency {
                    language: "English".to_string(),
                    proficiency_level: ProficiencyLevel::Native,
                    cultural_context: "Western".to_string(),
                },
                LanguageProficiency {
                    language: "Spanish".to_string(),
                    proficiency_level: ProficiencyLevel::Advanced,
                    cultural_context: "Latin American".to_string(),
                },
            ],
            cultural_nuances: vec![],
            idiomatic_expressions: vec![],
            translation_considerations: TranslationConsiderations::default(),
        }
    }
}

/// Language Proficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageProficiency {
    /// Language
    pub language: String,
    /// Proficiency level
    pub proficiency_level: ProficiencyLevel,
    /// Cultural context
    pub cultural_context: String,
}

/// Proficiency Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProficiencyLevel {
    /// Basic proficiency
    Basic,
    /// Intermediate proficiency
    Intermediate,
    /// Advanced proficiency
    Advanced,
    /// Native proficiency
    Native,
    /// Bilingual proficiency
    Bilingual,
}

/// Cultural Nuance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalNuance {
    /// Nuance type
    pub nuance_type: String,
    /// Culture
    pub culture: String,
    /// Description
    pub description: String,
    /// Examples
    pub examples: Vec<String>,
}

/// Idiomatic Expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdiomaticExpression {
    /// Expression
    pub expression: String,
    /// Language
    pub language: String,
    /// Literal meaning
    pub literal_meaning: String,
    /// Figurative meaning
    pub figurative_meaning: String,
    /// Cultural context
    pub cultural_context: String,
}

/// Translation Considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConsiderations {
    /// Direct translation accuracy
    pub direct_translation_accuracy: f32,
    /// Cultural adaptation needed
    pub cultural_adaptation_needed: bool,
    /// Context preservation priority
    pub context_preservation_priority: f32,
}

impl Default for TranslationConsiderations {
    fn default() -> Self {
        Self {
            direct_translation_accuracy: 0.7,
            cultural_adaptation_needed: true,
            context_preservation_priority: 0.9,
        }
    }
}

/// Context Sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSensitivity {
    /// Context awareness level
    pub context_awareness_level: f32,
    /// Situational adaptation
    pub situational_adaptation: Vec<SituationalAdaptation>,
    /// Environmental factors
    pub environmental_factors: Vec<EnvironmentalFactor>,
}

impl Default for ContextSensitivity {
    fn default() -> Self {
        Self {
            context_awareness_level: 0.85,
            situational_adaptation: vec![
                SituationalAdaptation {
                    situation: "formal_meeting".to_string(),
                    adaptation_strategy: "use_formal_language".to_string(),
                    effectiveness: 0.9,
                },
                SituationalAdaptation {
                    situation: "casual_conversation".to_string(),
                    adaptation_strategy: "use_informal_language".to_string(),
                    effectiveness: 0.85,
                },
            ],
            environmental_factors: vec![],
        }
    }
}

/// Situational Adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationalAdaptation {
    /// Situation
    pub situation: String,
    /// Adaptation strategy
    pub adaptation_strategy: String,
    /// Effectiveness
    pub effectiveness: f32,
}

/// Environmental Factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalFactor {
    /// Factor type
    pub factor_type: String,
    /// Impact level
    pub impact_level: f32,
    /// Adaptation needed
    pub adaptation_needed: String,
}

/// Cultural Sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalSensitivity {
    /// Sensitivity level
    pub sensitivity_level: SensitivityLevel,
    /// Cultural awareness
    pub cultural_awareness: Vec<CulturalAwareness>,
    /// Bias detection
    pub bias_detection: BiasDetection,
    /// Inclusive language
    pub inclusive_language: InclusiveLanguage,
}

impl Default for CulturalSensitivity {
    fn default() -> Self {
        Self {
            sensitivity_level: SensitivityLevel::High,
            cultural_awareness: vec![
                CulturalAwareness {
                    culture: "Western".to_string(),
                    awareness_level: 0.9,
                    key_considerations: vec![
                        "Individualism".to_string(),
                        "Direct communication".to_string(),
                        "Time consciousness".to_string(),
                    ],
                },
                CulturalAwareness {
                    culture: "Eastern".to_string(),
                    awareness_level: 0.85,
                    key_considerations: vec![
                        "Collectivism".to_string(),
                        "Indirect communication".to_string(),
                        "Relationship focus".to_string(),
                    ],
                },
            ],
            bias_detection: BiasDetection::default(),
            inclusive_language: InclusiveLanguage::default(),
        }
    }
}

/// Sensitivity Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensitivityLevel {
    /// Low sensitivity
    Low,
    /// Medium sensitivity
    Medium,
    /// High sensitivity
    High,
    /// Very high sensitivity
    VeryHigh,
    /// Expert sensitivity
    Expert,
}

/// Cultural Awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalAwareness {
    /// Culture
    pub culture: String,
    /// Awareness level
    pub awareness_level: f32,
    /// Key considerations
    pub key_considerations: Vec<String>,
}

/// Bias Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasDetection {
    /// Bias detection algorithms
    pub bias_detection_algorithms: Vec<String>,
    /// Sensitivity threshold
    pub sensitivity_threshold: f32,
    /// Correction strategies
    pub correction_strategies: Vec<String>,
}

impl Default for BiasDetection {
    fn default() -> Self {
        Self {
            bias_detection_algorithms: vec![
                "cultural_bias_detector".to_string(),
                "gender_bias_detector".to_string(),
                "age_bias_detector".to_string(),
            ],
            sensitivity_threshold: 0.7,
            correction_strategies: vec![
                "language_neutralization".to_string(),
                "perspective_balancing".to_string(),
                "context_reframing".to_string(),
            ],
        }
    }
}

/// Inclusive Language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InclusiveLanguage {
    /// Inclusive language guidelines
    pub inclusive_language_guidelines: Vec<String>,
    /// Preferred terminology
    pub preferred_terminology: Vec<TerminologyPreference>,
    /// Avoided terms
    pub avoided_terms: Vec<AvoidedTerm>,
}

impl Default for InclusiveLanguage {
    fn default() -> Self {
        Self {
            inclusive_language_guidelines: vec![
                "Use gender-neutral language".to_string(),
                "Avoid cultural stereotypes".to_string(),
                "Respect individual preferences".to_string(),
            ],
            preferred_terminology: vec![
                TerminologyPreference {
                    term: "partner".to_string(),
                    alternatives: vec!["spouse".to_string(), "significant_other".to_string()],
                    context: "relationship".to_string(),
                },
            ],
            avoided_terms: vec![
                AvoidedTerm {
                    term: "manpower".to_string(),
                    reason: "Gender-specific".to_string(),
                    alternatives: vec!["workforce".to_string(), "personnel".to_string()],
                },
            ],
        }
    }
}

/// Terminology Preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminologyPreference {
    /// Term
    pub term: String,
    /// Alternatives
    pub alternatives: Vec<String>,
    /// Context
    pub context: String,
}

/// Avoided Term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvoidedTerm {
    /// Term
    pub term: String,
    /// Reason
    pub reason: String,
    /// Alternatives
    pub alternatives: Vec<String>,
}
