//! NXR-ÆTHER Configuration
//! 
//! Model-specific configuration for NXR-ÆTHER

use serde::{Deserialize, Serialize};
use crate::shared::{model_config::NxrModelConfig, deeplearning_integration::DeepLearningConfig};

/// NXR-ÆTHER Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Emotional analysis configuration
    pub emotional: EmotionalConfig,
    /// Psychological configuration
    pub psychological: PsychologicalConfig,
    /// Empathy configuration
    pub empathy: EmpathyConfig,
    /// Cultural configuration
    pub cultural: CulturalConfig,
    /// Deep learning configuration
    pub deep_learning: DeepLearningConfig,
}

/// Emotional Analysis Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalConfig {
    /// Empathy depth
    pub empathy_depth: u8,
    /// Emotional sensitivity
    pub emotional_sensitivity: f32,
    /// Enable tone analysis
    pub enable_tone_analysis: bool,
    /// Emotional granularity
    pub emotional_granularity: EmotionalGranularity,
    /// Context awareness level
    pub context_awareness: ContextAwarenessLevel,
    /// Emotional model type
    pub emotional_model: EmotionalModelType,
}

/// Emotional Granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionalGranularity {
    /// Basic emotions only
    Basic,
    /// Basic + complex emotions
    Standard,
    /// Full emotional spectrum
    Comprehensive,
    /// Ultra-fine emotional granularity
    UltraFine,
}

/// Context Awareness Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextAwarenessLevel {
    /// No context awareness
    None,
    /// Local context only
    Local,
    /// Global context
    Global,
    /// Multi-dimensional context
    MultiDimensional,
}

/// Emotional Model Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionalModelType {
    /// Rule-based model
    RuleBased,
    /// Neural network model
    NeuralNetwork,
    /// Hybrid model
    Hybrid { neural_weight: f32, rule_weight: f32 },
    /// Ensemble model
    Ensemble { models: Vec<EmotionalModelType> },
}

/// Psychological Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychologicalConfig {
    /// Enable profiling
    pub enable_profiling: bool,
    /// Analysis depth
    pub analysis_depth: u8,
    /// Cultural sensitivity
    pub cultural_sensitivity: f32,
    /// Psychological framework
    pub psychological_framework: PsychologicalFramework,
    /// Assessment methods
    pub assessment_methods: Vec<AssessmentMethod>,
    /// Privacy level
    pub privacy_level: PrivacyLevel,
}

/// Psychological Framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PsychologicalFramework {
    /// Cognitive Behavioral Therapy (CBT)
    CBT,
    /// Psychodynamic approach
    Psychodynamic,
    /// Humanistic approach
    Humanistic,
    /// Positive psychology
    PositivePsychology,
    /// Integrative approach
    Integrative { frameworks: Vec<PsychologicalFramework> },
}

/// Assessment Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentMethod {
    /// Behavioral analysis
    Behavioral,
    /// Linguistic analysis
    Linguistic,
    /// Sentiment analysis
    Sentiment,
    /// Pattern recognition
    PatternRecognition,
    /// Psychological testing
    PsychologicalTesting,
    /// Clinical assessment
    Clinical,
}

/// Privacy Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// No privacy protection
    None,
    /// Basic privacy protection
    Basic,
    /// Enhanced privacy protection
    Enhanced,
    /// Maximum privacy protection
    Maximum,
}

/// Empathy Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathyConfig {
    /// Empathy weight in reasoning
    pub empathy_weight: f32,
    /// Empathy types
    pub empathy_types: Vec<EmpathyType>,
    /// Empathy response style
    pub response_style: EmpathyResponseStyle,
    /// Compassion level
    pub compassion_level: CompassionLevel,
    /// Support generation
    pub support_generation: SupportGeneration,
}

/// Empathy Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmpathyType {
    /// Cognitive empathy
    Cognitive,
    /// Emotional empathy
    Emotional,
    /// Compassionate empathy
    Compassionate,
    /// Somatic empathy
    Somatic,
    /// Spiritual empathy
    Spiritual,
}

/// Empathy Response Style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmpathyResponseStyle {
    /// Direct and supportive
    DirectSupportive,
    /// Gentle and nurturing
    GentleNurturing,
    /// Professional and clinical
    ProfessionalClinical,
    /// Warm and personal
    WarmPersonal,
    /// Adaptive to context
    Adaptive,
}

/// Compassion Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompassionLevel {
    /// Low compassion
    Low,
    /// Medium compassion
    Medium,
    /// High compassion
    High,
    /// Maximum compassion
    Maximum,
}

/// Support Generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportGeneration {
    /// Enable support generation
    pub enable_support: bool,
    /// Support types
    pub support_types: Vec<SupportType>,
    /// Support customization
    pub customization: SupportCustomization,
    /// Support validation
    pub validation: SupportValidation,
}

/// Support Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportType {
    /// Emotional support
    Emotional,
    /// Practical advice
    PracticalAdvice,
    /// Resource recommendation
    ResourceRecommendation,
    /// Referral suggestion
    ReferralSuggestion,
    /// Coping strategies
    CopingStrategies,
    /// Validation and affirmation
    Validation,
}

/// Support Customization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportCustomization {
    /// No customization
    None,
    /// Basic customization
    Basic,
    /// Advanced customization
    Advanced,
    /// Personalized customization
    Personalized,
}

/// Support Validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportValidation {
    /// No validation
    None,
    /// Ethical validation
    Ethical,
    /// Clinical validation
    Clinical,
    /// Multi-level validation
    MultiLevel,
}

/// Cultural Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalConfig {
    /// Cultural adaptation mode
    pub adaptation_mode: CulturalAdaptationMode,
    /// Supported cultures
    pub supported_cultures: Vec<CulturalContext>,
    /// Cultural sensitivity level
    pub sensitivity_level: CulturalSensitivityLevel,
    /// Cross-cultural awareness
    pub cross_cultural_awareness: bool,
    /// Cultural learning mode
    pub learning_mode: CulturalLearningMode,
}

/// Cultural Adaptation Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalAdaptationMode {
    /// No adaptation
    None,
    /// Basic adaptation
    Basic,
    /// Deep adaptation
    Deep,
    /// Dynamic adaptation
    Dynamic,
    /// Learning adaptation
    Learning,
}

/// Cultural Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalContext {
    /// Culture name
    pub name: String,
    /// Cultural values
    pub values: Vec<String>,
    /// Communication style
    pub communication_style: CommunicationStyle,
    /// Emotional expression norms
    pub emotional_norms: EmotionalExpressionNorms,
    /// Support preferences
    pub support_preferences: Vec<String>,
}

/// Communication Style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStyle {
    /// Direct communication
    Direct,
    /// Indirect communication
    Indirect,
    /// High-context communication
    HighContext,
    /// Low-context communication
    LowContext,
    /// Formal communication
    Formal,
    /// Informal communication
    Informal,
}

/// Emotional Expression Norms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalExpressionNorms {
    /// Openness to emotional expression
    pub openness: f32,
    /// Preferred emotional intensity
    pub preferred_intensity: f32,
    /// Taboo emotions
    pub taboo_emotions: Vec<String>,
    /// Celebrated emotions
    pub celebrated_emotions: Vec<String>,
}

/// Cultural Sensitivity Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalSensitivityLevel {
    /// Low sensitivity
    Low,
    /// Medium sensitivity
    Medium,
    /// High sensitivity
    High,
    /// Maximum sensitivity
    Maximum,
}

/// Cultural Learning Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CulturalLearningMode {
    /// No learning
    None,
    /// Static learning
    Static,
    /// Adaptive learning
    Adaptive,
    /// Continuous learning
    Continuous,
}

impl Default for AetherConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Aether),
            emotional: EmotionalConfig::default(),
            psychological: PsychologicalConfig::default(),
            empathy: EmpathyConfig::default(),
            cultural: CulturalConfig::default(),
        }
    }
}

impl Default for EmotionalConfig {
    fn default() -> Self {
        Self {
            empathy_depth: 8,
            emotional_sensitivity: 0.9,
            enable_tone_analysis: true,
            emotional_granularity: EmotionalGranularity::Comprehensive,
            context_awareness: ContextAwarenessLevel::MultiDimensional,
            emotional_model: EmotionalModelType::Hybrid {
                neural_weight: 0.7,
                rule_weight: 0.3,
            },
        }
    }
}

impl Default for PsychologicalConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            analysis_depth: 6,
            cultural_sensitivity: 0.85,
            psychological_framework: PsychologicalFramework::Integrative {
                frameworks: vec![
                    PsychologicalFramework::CBT,
                    PsychologicalFramework::Humanistic,
                    PsychologicalFramework::PositivePsychology,
                ],
            },
            assessment_methods: vec![
                AssessmentMethod::Linguistic,
                AssessmentMethod::Sentiment,
                AssessmentMethod::PatternRecognition,
            ],
            privacy_level: PrivacyLevel::Enhanced,
        }
    }
}

impl Default for EmpathyConfig {
    fn default() -> Self {
        Self {
            empathy_weight: 0.3,
            empathy_types: vec![
                EmpathyType::Cognitive,
                EmpathyType::Emotional,
                EmpathyType::Compassionate,
            ],
            response_style: EmpathyResponseStyle::Adaptive,
            compassion_level: CompassionLevel::High,
            support_generation: SupportGeneration {
                enable_support: true,
                support_types: vec![
                    SupportType::Emotional,
                    SupportType::Validation,
                    SupportType::CopingStrategies,
                ],
                customization: SupportCustomization::Personalized,
                validation: SupportValidation::Ethical,
            },
        }
    }
}

impl Default for CulturalConfig {
    fn default() -> Self {
        Self {
            adaptation_mode: CulturalAdaptationMode::Dynamic,
            supported_cultures: vec![
                CulturalContext {
                    name: "Western".to_string(),
                    values: vec!["individualism".to_string(), "directness".to_string()],
                    communication_style: CommunicationStyle::Direct,
                    emotional_norms: EmotionalExpressionNorms {
                        openness: 0.8,
                        preferred_intensity: 0.7,
                        taboo_emotions: vec!["excessive anger".to_string()],
                        celebrated_emotions: vec!["happiness".to_string(), "excitement".to_string()],
                    },
                    support_preferences: vec!["direct advice".to_string(), "practical solutions".to_string()],
                },
                CulturalContext {
                    name: "Eastern".to_string(),
                    values: vec!["collectivism".to_string(), "harmony".to_string()],
                    communication_style: CommunicationStyle::Indirect,
                    emotional_norms: EmotionalExpressionNorms {
                        openness: 0.5,
                        preferred_intensity: 0.4,
                        taboo_emotions: vec!["public anger".to_string(), "direct confrontation".to_string()],
                        celebrated_emotions: vec!["respect".to_string(), "gratitude".to_string()],
                    },
                    support_preferences: vec!["indirect guidance".to_string(), "group harmony".to_string()],
                },
            ],
            sensitivity_level: CulturalSensitivityLevel::High,
            cross_cultural_awareness: true,
            learning_mode: CulturalLearningMode::Adaptive,
        }
    }
}

impl AetherConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate emotional configuration
        if self.emotional.empathy_depth == 0 {
            return Err("empathy_depth must be > 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.emotional.emotional_sensitivity) {
            return Err("emotional_sensitivity must be between 0.0 and 1.0".to_string());
        }

        // Validate psychological configuration
        if self.psychological.analysis_depth == 0 {
            return Err("analysis_depth must be > 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.psychological.cultural_sensitivity) {
            return Err("cultural_sensitivity must be between 0.0 and 1.0".to_string());
        }

        // Validate empathy configuration
        if !(0.0..=1.0).contains(&self.empathy.empathy_weight) {
            return Err("empathy_weight must be between 0.0 and 1.0".to_string());
        }

        if self.empathy.empathy_types.is_empty() {
            return Err("At least one empathy type must be specified".to_string());
        }

        // Validate cultural configuration
        if self.cultural.supported_cultures.is_empty() {
            return Err("At least one supported culture must be specified".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific component
    pub fn get_component_config(&self, component: &str) -> Option<serde_json::Value> {
        match component {
            "emotional" => Some(serde_json::to_value(&self.emotional).unwrap_or_default()),
            "psychological" => Some(serde_json::to_value(&self.psychological).unwrap_or_default()),
            "empathy" => Some(serde_json::to_value(&self.empathy).unwrap_or_default()),
            "cultural" => Some(serde_json::to_value(&self.cultural).unwrap_or_default()),
            _ => None,
        }
    }

    /// Update component configuration
    pub fn update_component_config<T>(&mut self, component: String, config: T) -> Result<(), serde_json::Error>
    where
        T: Serialize,
    {
        let json_value = serde_json::to_value(config)?;
        
        match component.as_str() {
            "emotional" => {
                self.emotional = serde_json::from_value(json_value)?;
            }
            "psychological" => {
                self.psychological = serde_json::from_value(json_value)?;
            }
            "empathy" => {
                self.empathy = serde_json::from_value(json_value)?;
            }
            "cultural" => {
                self.cultural = serde_json::from_value(json_value)?;
            }
            _ => {
                return Err(serde_json::Error::syntax(serde_json::error::ErrorCode::ExpectedColon, 0, 0));
            }
        }

        Ok(())
    }

    /// Get empathy types as strings
    pub fn get_empathy_types(&self) -> Vec<String> {
        self.empathy.empathy_types.iter().map(|t| format!("{:?}", t)).collect()
    }

    /// Get supported cultures as strings
    pub fn get_supported_cultures(&self) -> Vec<String> {
        self.cultural.supported_cultures.iter().map(|c| c.name.clone()).collect()
    }

    /// Check if tone analysis is enabled
    pub fn is_tone_analysis_enabled(&self) -> bool {
        self.emotional.enable_tone_analysis
    }

    /// Check if profiling is enabled
    pub fn is_profiling_enabled(&self) -> bool {
        self.psychological.enable_profiling
    }

    /// Check if support generation is enabled
    pub fn is_support_generation_enabled(&self) -> bool {
        self.empathy.support_generation.enable_support
    }

    /// Get empathy depth
    pub fn get_empathy_depth(&self) -> u8 {
        self.emotional.empathy_depth
    }

    /// Set empathy depth
    pub fn set_empathy_depth(&mut self, depth: u8) {
        self.emotional.empathy_depth = depth;
    }

    /// Get emotional sensitivity
    pub fn get_emotional_sensitivity(&self) -> f32 {
        self.emotional.emotional_sensitivity
    }

    /// Set emotional sensitivity
    pub fn set_emotional_sensitivity(&mut self, sensitivity: f32) {
        self.emotional.emotional_sensitivity = sensitivity;
    }

    /// Get empathy weight
    pub fn get_empathy_weight(&self) -> f32 {
        self.empathy.empathy_weight
    }

    /// Set empathy weight
    pub fn set_empathy_weight(&mut self, weight: f32) {
        self.empathy.empathy_weight = weight;
    }

    /// Get cultural sensitivity
    pub fn get_cultural_sensitivity(&self) -> f32 {
        self.psychological.cultural_sensitivity
    }

    /// Set cultural sensitivity
    pub fn set_cultural_sensitivity(&mut self, sensitivity: f32) {
        self.psychological.cultural_sensitivity = sensitivity;
    }

    /// Add supported culture
    pub fn add_supported_culture(&mut self, culture: CulturalContext) {
        self.cultural.supported_cultures.push(culture);
    }

    /// Remove supported culture
    pub fn remove_supported_culture(&mut self, culture_name: &str) -> bool {
        let original_len = self.cultural.supported_cultures.len();
        self.cultural.supported_cultures.retain(|c| c.name != culture_name);
        self.cultural.supported_cultures.len() < original_len
    }

    /// Get support types as strings
    pub fn get_support_types(&self) -> Vec<String> {
        self.empathy.support_generation.support_types.iter().map(|t| format!("{:?}", t)).collect()
    }

    /// Add support type
    pub fn add_support_type(&mut self, support_type: SupportType) {
        self.empathy.support_generation.support_types.push(support_type);
    }

    /// Remove support type
    pub fn remove_support_type(&mut self, support_type: &SupportType) -> bool {
        let original_len = self.empathy.support_generation.support_types.len();
        self.empathy.support_generation.support_types.retain(|t| t != support_type);
        self.empathy.support_generation.support_types.len() < original_len
    }

    /// Get assessment methods as strings
    pub fn get_assessment_methods(&self) -> Vec<String> {
        self.psychological.assessment_methods.iter().map(|m| format!("{:?}", m)).collect()
    }

    /// Add assessment method
    pub fn add_assessment_method(&mut self, method: AssessmentMethod) {
        self.psychological.assessment_methods.push(method);
    }

    /// Remove assessment method
    pub fn remove_assessment_method(&mut self, method: &AssessmentMethod) -> bool {
        let original_len = self.psychological.assessment_methods.len();
        self.psychological.assessment_methods.retain(|m| m != method);
        self.psychological.assessment_methods.len() < original_len
    }

    /// Get privacy level
    pub fn get_privacy_level(&self) -> &PrivacyLevel {
        &self.psychological.privacy_level
    }

    /// Set privacy level
    pub fn set_privacy_level(&mut self, level: PrivacyLevel) {
        self.psychological.privacy_level = level;
    }

    /// Get emotional granularity
    pub fn get_emotional_granularity(&self) -> &EmotionalGranularity {
        &self.emotional.emotional_granularity
    }

    /// Set emotional granularity
    pub fn set_emotional_granularity(&mut self, granularity: EmotionalGranularity) {
        self.emotional.emotional_granularity = granularity;
    }

    /// Get context awareness level
    pub fn get_context_awareness_level(&self) -> &ContextAwarenessLevel {
        &self.emotional.context_awareness
    }

    /// Set context awareness level
    pub fn set_context_awareness_level(&mut self, level: ContextAwarenessLevel) {
        self.emotional.context_awareness = level;
    }

    /// Get emotional model type
    pub fn get_emotional_model_type(&self) -> &EmotionalModelType {
        &self.emotional.emotional_model
    }

    /// Set emotional model type
    pub fn set_emotional_model_type(&mut self, model_type: EmotionalModelType) {
        self.emotional.emotional_model = model_type;
    }

    /// Get psychological framework
    pub fn get_psychological_framework(&self) -> &PsychologicalFramework {
        &self.psychological.psychological_framework
    }

    /// Set psychological framework
    pub fn set_psychological_framework(&mut self, framework: PsychologicalFramework) {
        self.psychological.psychological_framework = framework;
    }

    /// Get empathy response style
    pub fn get_empathy_response_style(&self) -> &EmpathyResponseStyle {
        &self.empathy.response_style
    }

    /// Set empathy response style
    pub fn set_empathy_response_style(&mut self, style: EmpathyResponseStyle) {
        self.empathy.response_style = style;
    }

    /// Get compassion level
    pub fn get_compassion_level(&self) -> &CompassionLevel {
        &self.empathy.compassion_level
    }

    /// Set compassion level
    pub fn set_compassion_level(&mut self, level: CompassionLevel) {
        self.empathy.compassion_level = level;
    }

    /// Get cultural adaptation mode
    pub fn get_cultural_adaptation_mode(&self) -> &CulturalAdaptationMode {
        &self.cultural.adaptation_mode
    }

    /// Set cultural adaptation mode
    pub fn set_cultural_adaptation_mode(&mut self, mode: CulturalAdaptationMode) {
        self.cultural.adaptation_mode = mode;
    }

    /// Get cultural sensitivity level
    pub fn get_cultural_sensitivity_level(&self) -> &CulturalSensitivityLevel {
        &self.cultural.sensitivity_level
    }

    /// Set cultural sensitivity level
    pub fn set_cultural_sensitivity_level(&mut self, level: CulturalSensitivityLevel) {
        self.cultural.sensitivity_level = level;
    }

    /// Check if cross-cultural awareness is enabled
    pub fn is_cross_cultural_awareness_enabled(&self) -> bool {
        self.cultural.cross_cultural_awareness
    }

    /// Set cross-cultural awareness
    pub fn set_cross_cultural_awareness(&mut self, enabled: bool) {
        self.cultural.cross_cultural_awareness = enabled;
    }

    /// Get cultural learning mode
    pub fn get_cultural_learning_mode(&self) -> &CulturalLearningMode {
        &self.cultural.learning_mode
    }

    /// Set cultural learning mode
    pub fn set_cultural_learning_mode(&mut self, mode: CulturalLearningMode) {
        self.cultural.learning_mode = mode;
    }

    /// Get support customization
    pub fn get_support_customization(&self) -> &SupportCustomization {
        &self.empathy.support_generation.customization
    }

    /// Set support customization
    pub fn set_support_customization(&mut self, customization: SupportCustomization) {
        self.empathy.support_generation.customization = customization;
    }

    /// Get support validation
    pub fn get_support_validation(&self) -> &SupportValidation {
        &self.empathy.support_generation.validation
    }

    /// Set support validation
    pub fn set_support_validation(&mut self, validation: SupportValidation) {
        self.empathy.support_generation.validation = validation;
    }

    /// Create configuration for specific culture
    pub fn create_culture_specific_config(&self, culture_name: &str) -> Option<AetherConfig> {
        let culture = self.cultural.supported_cultures.iter().find(|c| c.name == culture_name)?;
        
        let mut culture_config = self.clone();
        
        // Adjust empathy weight based on cultural preferences
        culture_config.empathy.empathy_weight = match culture.communication_style {
            CommunicationStyle::Direct => 0.4,
            CommunicationStyle::Indirect => 0.2,
            CommunicationStyle::HighContext => 0.3,
            CommunicationStyle::LowContext => 0.4,
            CommunicationStyle::Formal => 0.3,
            CommunicationStyle::Informal => 0.4,
        };
        
        // Adjust emotional sensitivity based on cultural norms
        culture_config.emotional.emotional_sensitivity = culture.emotional_norms.openness;
        
        Some(culture_config)
    }

    /// Validate cultural context
    pub fn validate_cultural_context(&self, context: &str) -> Result<(), String> {
        if self.cultural.supported_cultures.iter().any(|c| c.name == context) {
            Ok(())
        } else {
            Err(format!("Unsupported cultural context: {}", context))
        }
    }

    /// Get recommended empathy types for culture
    pub fn get_recommended_empathy_types_for_culture(&self, culture_name: &str) -> Vec<EmpathyType> {
        if let Some(culture) = self.cultural.supported_cultures.iter().find(|c| c.name == culture_name) {
            match culture.communication_style {
                CommunicationStyle::Direct => vec![
                    EmpathyType::Cognitive,
                    EmpathyType::Compassionate,
                ],
                CommunicationStyle::Indirect => vec![
                    EmpathyType::Emotional,
                    EmpathyType::Compassionate,
                ],
                CommunicationStyle::HighContext => vec![
                    EmpathyType::Emotional,
                    EmpathyType::Compassionate,
                    EmpathyType::Spiritual,
                ],
                CommunicationStyle::LowContext => vec![
                    EmpathyType::Cognitive,
                    EmpathyType::Compassionate,
                ],
                CommunicationStyle::Formal => vec![
                    EmpathyType::Cognitive,
                    EmpathyType::Compassionate,
                ],
                CommunicationStyle::Informal => vec![
                    EmpathyType::Emotional,
                    EmpathyType::Compassionate,
                ],
            }
        } else {
            self.empathy.empathy_types.clone()
        }
    }

    /// Get recommended support types for culture
    pub fn get_recommended_support_types_for_culture(&self, culture_name: &str) -> Vec<SupportType> {
        if let Some(culture) = self.cultural.supported_cultures.iter().find(|c| c.name == culture_name) {
            culture.support_preferences.iter().filter_map(|pref| {
                match pref.as_str() {
                    "direct advice" => Some(SupportType::PracticalAdvice),
                    "practical solutions" => Some(SupportType::PracticalAdvice),
                    "indirect guidance" => Some(SupportType::Emotional),
                    "group harmony" => Some(SupportType::Validation),
                    _ => None,
                }
            }).collect()
        } else {
            self.empathy.support_generation.support_types.clone()
        }
    }
}
