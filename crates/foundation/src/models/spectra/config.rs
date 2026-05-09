//! NXR-SPECTRA Configuration
//! 
//! Model-specific configuration for NXR-SPECTRA

use serde::{Deserialize, Serialize};
use crate::shared::model_config::NxrModelConfig;

/// NXR-SPECTRA Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectraConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Creative configuration
    pub creative: CreativeConfig,
    /// Multimodal configuration
    pub multimodal: MultimodalConfig,
    /// Style configuration
    pub style: StyleConfig,
    /// Innovation configuration
    pub innovation: InnovationConfig,
}

/// Creative Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeConfig {
    /// Creativity level
    pub creativity_level: CreativityLevel,
    /// Originality weight
    pub originality_weight: f32,
    /// Enable cross-modal creativity
    pub enable_cross_modal_creativity: bool,
    /// Creative granularity
    pub creative_granularity: CreativeGranularity,
    /// Innovation threshold
    pub innovation_threshold: f32,
    /// Creative model type
    pub creative_model: CreativeModelType,
}

/// Creativity Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativityLevel {
    /// Conservative creativity
    Conservative,
    /// Moderate creativity
    Moderate,
    /// High creativity
    High,
    /// Maximum creativity
    Maximum,
    /// Transcendent creativity
    Transcendent,
}

/// Creative Granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativeGranularity {
    /// Coarse granularity
    Coarse,
    /// Medium granularity
    Medium,
    /// Fine granularity
    Fine,
    /// Ultra-fine granularity
    UltraFine,
}

/// Creative Model Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativeModelType {
    /// Generative model
    Generative,
    /// Transformative model
    Transformative,
    /// Hybrid model
    Hybrid { generative_weight: f32, transformative_weight: f32 },
    /// Ensemble model
    Ensemble { models: Vec<CreativeModelType> },
}

/// Multimodal Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    /// Supported modalities
    pub supported_modalities: Vec<Modality>,
    /// Cross-modal attention
    pub cross_modal_attention: bool,
    /// Modality fusion strategy
    pub modality_fusion: ModalityFusionStrategy,
    /// Multimodal creativity
    pub multimodal_creativity: bool,
    /// Modality weights
    pub modality_weights: HashMap<String, f32>,
}

/// Modality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Modality {
    /// Text modality
    Text,
    /// Image modality
    Image,
    /// Audio modality
    Audio,
    /// Video modality
    Video,
    /// 3D modality
    ThreeD,
    /// Interactive modality
    Interactive,
}

/// Modality Fusion Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityFusionStrategy {
    /// Early fusion
    Early,
    /// Late fusion
    Late,
    /// Hierarchical fusion
    Hierarchical,
    /// Attention-based fusion
    AttentionBased,
    /// Adaptive fusion
    Adaptive,
}

/// Style Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConfig {
    /// Style adaptation mode
    pub style_adaptation_mode: StyleAdaptationMode,
    /// Supported styles
    pub supported_styles: Vec<ArtisticStyle>,
    /// Style learning capability
    pub style_learning: bool,
    /// Style synthesis depth
    pub style_synthesis_depth: u8,
    /// Cross-style creativity
    pub cross_style_creativity: bool,
}

/// Style Adaptation Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StyleAdaptationMode {
    /// No adaptation
    None,
    /// Basic adaptation
    Basic,
    /// Advanced adaptation
    Advanced,
    /// Deep adaptation
    Deep,
    /// Learning adaptation
    Learning,
}

/// Artistic Style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtisticStyle {
    /// Style name
    pub name: String,
    /// Style category
    pub category: StyleCategory,
    /// Style characteristics
    pub characteristics: Vec<String>,
    /// Style parameters
    pub parameters: StyleParameters,
    /// Historical context
    pub historical_context: Option<String>,
}

/// Style Category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StyleCategory {
    /// Classical styles
    Classical,
    /// Modern styles
    Modern,
    /// Contemporary styles
    Contemporary,
    /// Digital styles
    Digital,
    /// Experimental styles
    Experimental,
    /// Cultural styles
    Cultural,
}

/// Style Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleParameters {
    /// Color palette
    pub color_palette: Vec<String>,
    /// Brush stroke style
    pub brush_stroke_style: String,
    /// Composition style
    pub composition_style: String,
    /// Texture characteristics
    pub texture_characteristics: Vec<String>,
    /// Mood characteristics
    pub mood_characteristics: Vec<String>,
}

/// Innovation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationConfig {
    /// Innovation mode
    pub innovation_mode: InnovationMode,
    /// Novelty threshold
    pub novelty_threshold: f32,
    /// Enable concept generation
    pub enable_concept_generation: bool,
    /// Innovation domains
    pub innovation_domains: Vec<InnovationDomain>,
    /// Creative constraints
    pub creative_constraints: Vec<CreativeConstraint>,
}

/// Innovation Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InnovationMode {
    /// Incremental innovation
    Incremental,
    /// Radical innovation
    Radical,
    /// Disruptive innovation
    Disruptive,
    /// Transformative innovation
    Transformative,
    /// Adaptive innovation
    Adaptive,
}

/// Innovation Domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InnovationDomain {
    /// Visual innovation
    Visual,
    /// Audio innovation
    Audio,
    /// Text innovation
    Text,
    /// Interactive innovation
    Interactive,
    /// Conceptual innovation
    Conceptual,
    /// Technical innovation
    Technical,
}

/// Creative Constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint parameters
    pub parameters: HashMap<String, f32>,
    /// Constraint flexibility
    pub flexibility: f32,
}

/// Constraint Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Style constraint
    Style,
    /// Medium constraint
    Medium,
    /// Theme constraint
    Theme,
    /// Technical constraint
    Technical,
    /// Cultural constraint
    Cultural,
}

impl Default for SpectraConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Spectra),
            creative: CreativeConfig::default(),
            multimodal: MultimodalConfig::default(),
            style: StyleConfig::default(),
            innovation: InnovationConfig::default(),
        }
    }
}

impl Default for CreativeConfig {
    fn default() -> Self {
        Self {
            creativity_level: CreativityLevel::High,
            originality_weight: 0.8,
            enable_cross_modal_creativity: true,
            creative_granularity: CreativeGranularity::Fine,
            innovation_threshold: 0.7,
            creative_model: CreativeModelType::Hybrid {
                generative_weight: 0.6,
                transformative_weight: 0.4,
            },
        }
    }
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        let mut modality_weights = HashMap::new();
        modality_weights.insert("text".to_string(), 0.3);
        modality_weights.insert("image".to_string(), 0.3);
        modality_weights.insert("audio".to_string(), 0.2);
        modality_weights.insert("video".to_string(), 0.2);

        Self {
            supported_modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
            ],
            cross_modal_attention: true,
            modality_fusion: ModalityFusionStrategy::AttentionBased,
            multimodal_creativity: true,
            modality_weights,
        }
    }
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            style_adaptation_mode: StyleAdaptationMode::Advanced,
            supported_styles: vec![
                ArtisticStyle {
                    name: "Impressionism".to_string(),
                    category: StyleCategory::Classical,
                    characteristics: vec!["light and color".to_string(), "visible brush strokes".to_string()],
                    parameters: StyleParameters {
                        color_palette: vec!["vibrant".to_string(), "natural".to_string()],
                        brush_stroke_style: "visible".to_string(),
                        composition_style: "balanced".to_string(),
                        texture_characteristics: vec!["textured".to_string()],
                        mood_characteristics: vec!["serene".to_string(), "dynamic".to_string()],
                    },
                    historical_context: Some("19th century France".to_string()),
                },
                ArtisticStyle {
                    name: "Abstract Expressionism".to_string(),
                    category: StyleCategory::Modern,
                    characteristics: vec!["abstract".to_string(), "emotional".to_string(), "gestural".to_string()],
                    parameters: StyleParameters {
                        color_palette: vec!["bold".to_string(), "contrasting".to_string()],
                        brush_stroke_style: "gestural".to_string(),
                        composition_style: "dynamic".to_string(),
                        texture_characteristics: vec!["thick".to_string(), "layered".to_string()],
                        mood_characteristics: vec!["intense".to_string(), "expressive".to_string()],
                    },
                    historical_context: Some("1940s-1950s America".to_string()),
                },
            ],
            style_learning: true,
            style_synthesis_depth: 8,
            cross_style_creativity: true,
        }
    }
}

impl Default for InnovationConfig {
    fn default() -> Self {
        Self {
            innovation_mode: InnovationMode::Transformative,
            novelty_threshold: 0.75,
            enable_concept_generation: true,
            innovation_domains: vec![
                InnovationDomain::Visual,
                InnovationDomain::Audio,
                InnovationDomain::Text,
                InnovationDomain::Conceptual,
            ],
            creative_constraints: vec![
                CreativeConstraint {
                    name: "aesthetic_balance".to_string(),
                    constraint_type: ConstraintType::Style,
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("balance".to_string(), 0.8);
                        params.insert("harmony".to_string(), 0.7);
                        params
                    },
                    flexibility: 0.3,
                },
            ],
        }
    }
}

impl SpectraConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate creative configuration
        if !(0.0..=1.0).contains(&self.creative.originality_weight) {
            return Err("originality_weight must be between 0.0 and 1.0".to_string());
        }

        if !(0.0..=1.0).contains(&self.creative.innovation_threshold) {
            return Err("innovation_threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate multimodal configuration
        if self.multimodal.supported_modalities.is_empty() {
            return Err("At least one modality must be supported".to_string());
        }

        // Validate style configuration
        if self.style.style_synthesis_depth == 0 {
            return Err("style_synthesis_depth must be > 0".to_string());
        }

        // Validate innovation configuration
        if !(0.0..=1.0).contains(&self.innovation.novelty_threshold) {
            return Err("novelty_threshold must be between 0.0 and 1.0".to_string());
        }

        if self.innovation.innovation_domains.is_empty() {
            return Err("At least one innovation domain must be specified".to_string());
        }

        Ok(())
    }

    /// Get configuration for specific component
    pub fn get_component_config(&self, component: &str) -> Option<serde_json::Value> {
        match component {
            "creative" => Some(serde_json::to_value(&self.creative).unwrap_or_default()),
            "multimodal" => Some(serde_json::to_value(&self.multimodal).unwrap_or_default()),
            "style" => Some(serde_json::to_value(&self.style).unwrap_or_default()),
            "innovation" => Some(serde_json::to_value(&self.innovation).unwrap_or_default()),
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
            "creative" => {
                self.creative = serde_json::from_value(json_value)?;
            }
            "multimodal" => {
                self.multimodal = serde_json::from_value(json_value)?;
            }
            "style" => {
                self.style = serde_json::from_value(json_value)?;
            }
            "innovation" => {
                self.innovation = serde_json::from_value(json_value)?;
            }
            _ => {
                return Err(serde_json::Error::syntax(serde_json::error::ErrorCode::ExpectedColon, 0, 0));
            }
        }

        Ok(())
    }

    /// Get supported modalities as strings
    pub fn get_supported_modalities(&self) -> Vec<String> {
        self.multimodal.supported_modalities.iter().map(|m| format!("{:?}", m)).collect()
    }

    /// Get supported styles as strings
    pub fn get_supported_styles(&self) -> Vec<String> {
        self.style.supported_styles.iter().map(|s| s.name.clone()).collect()
    }

    /// Get innovation domains as strings
    pub fn get_innovation_domains(&self) -> Vec<String> {
        self.innovation.innovation_domains.iter().map(|d| format!("{:?}", d)).collect()
    }

    /// Check if cross-modal creativity is enabled
    pub fn is_cross_modal_creativity_enabled(&self) -> bool {
        self.creative.enable_cross_modal_creativity
    }

    /// Check if style learning is enabled
    pub fn is_style_learning_enabled(&self) -> bool {
        self.style.style_learning
    }

    /// Check if concept generation is enabled
    pub fn is_concept_generation_enabled(&self) -> bool {
        self.innovation.enable_concept_generation
    }

    /// Get creativity level
    pub fn get_creativity_level(&self) -> &CreativityLevel {
        &self.creative.creativity_level
    }

    /// Set creativity level
    pub fn set_creativity_level(&mut self, level: CreativityLevel) {
        self.creative.creativity_level = level;
    }

    /// Get originality weight
    pub fn get_originality_weight(&self) -> f32 {
        self.creative.originality_weight
    }

    /// Set originality weight
    pub fn set_originality_weight(&mut self, weight: f32) {
        self.creative.originality_weight = weight;
    }

    /// Get innovation threshold
    pub fn get_innovation_threshold(&self) -> f32 {
        self.creative.innovation_threshold
    }

    /// Set innovation threshold
    pub fn set_innovation_threshold(&mut self, threshold: f32) {
        self.creative.innovation_threshold = threshold;
    }

    /// Get novelty threshold
    pub fn get_novelty_threshold(&self) -> f32 {
        self.innovation.novelty_threshold
    }

    /// Set novelty threshold
    pub fn set_novelty_threshold(&mut self, threshold: f32) {
        self.innovation.novelty_threshold = threshold;
    }

    /// Get style synthesis depth
    pub fn get_style_synthesis_depth(&self) -> u8 {
        self.style.style_synthesis_depth
    }

    /// Set style synthesis depth
    pub fn set_style_synthesis_depth(&mut self, depth: u8) {
        self.style.style_synthesis_depth = depth;
    }

    /// Add supported modality
    pub fn add_supported_modality(&mut self, modality: Modality) {
        if !self.multimodal.supported_modalities.contains(&modality) {
            self.multimodal.supported_modalities.push(modality);
        }
    }

    /// Remove supported modality
    pub fn remove_supported_modality(&mut self, modality: &Modality) -> bool {
        let original_len = self.multimodal.supported_modalities.len();
        self.multimodal.supported_modalities.retain(|m| m != modality);
        self.multimodal.supported_modalities.len() < original_len
    }

    /// Add supported style
    pub fn add_supported_style(&mut self, style: ArtisticStyle) {
        self.style.supported_styles.push(style);
    }

    /// Remove supported style
    pub fn remove_supported_style(&mut self, style_name: &str) -> bool {
        let original_len = self.style.supported_styles.len();
        self.style.supported_styles.retain(|s| s.name != style_name);
        self.style.supported_styles.len() < original_len
    }

    /// Add innovation domain
    pub fn add_innovation_domain(&mut self, domain: InnovationDomain) {
        if !self.innovation.innovation_domains.contains(&domain) {
            self.innovation.innovation_domains.push(domain);
        }
    }

    /// Remove innovation domain
    pub fn remove_innovation_domain(&mut self, domain: &InnovationDomain) -> bool {
        let original_len = self.innovation.innovation_domains.len();
        self.innovation.innovation_domains.retain(|d| d != domain);
        self.innovation.innovation_domains.len() < original_len
    }

    /// Get modality weight
    pub fn get_modality_weight(&self, modality: &str) -> f32 {
        self.multimodal.modality_weights.get(modality).copied().unwrap_or(0.0)
    }

    /// Set modality weight
    pub fn set_modality_weight(&mut self, modality: String, weight: f32) {
        self.multimodal.modality_weights.insert(modality, weight);
    }

    /// Add creative constraint
    pub fn add_creative_constraint(&mut self, constraint: CreativeConstraint) {
        self.innovation.creative_constraints.push(constraint);
    }

    /// Remove creative constraint
    pub fn remove_creative_constraint(&mut self, constraint_name: &str) -> bool {
        let original_len = self.innovation.creative_constraints.len();
        self.innovation.creative_constraints.retain(|c| c.name != constraint_name);
        self.innovation.creative_constraints.len() < original_len
    }

    /// Get creative constraints as strings
    pub fn get_creative_constraints(&self) -> Vec<String> {
        self.innovation.creative_constraints.iter().map(|c| c.name.clone()).collect()
    }

    /// Validate modality
    pub fn validate_modality(&self, modality: &str) -> Result<(), String> {
        let modality_enum = match modality {
            "text" => Modality::Text,
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            "video" => Modality::Video,
            "3d" => Modality::ThreeD,
            "interactive" => Modality::Interactive,
            _ => return Err(format!("Unsupported modality: {}", modality)),
        };

        if !self.multimodal.supported_modalities.contains(&modality_enum) {
            return Err(format!("Modality {} is not supported", modality));
        }

        Ok(())
    }

    /// Validate style
    pub fn validate_style(&self, style_name: &str) -> Result<(), String> {
        if !self.style.supported_styles.iter().any(|s| s.name == style_name) {
            return Err(format!("Style {} is not supported", style_name));
        }

        Ok(())
    }

    /// Get style by name
    pub fn get_style_by_name(&self, style_name: &str) -> Option<&ArtisticStyle> {
        self.style.supported_styles.iter().find(|s| s.name == style_name)
    }

    /// Create configuration for specific style
    pub fn create_style_specific_config(&self, style_name: &str) -> Option<SpectraConfig> {
        let style = self.get_style_by_name(style_name)?;
        
        let mut style_config = self.clone();
        
        // Adjust creativity level based on style
        style_config.creative.creativity_level = match style.category {
            StyleCategory::Experimental => CreativityLevel::Transcendent,
            StyleCategory::Contemporary => CreativityLevel::Maximum,
            StyleCategory::Modern => CreativityLevel::High,
            StyleCategory::Digital => CreativityLevel::High,
            StyleCategory::Classical => CreativityLevel::Moderate,
            StyleCategory::Cultural => CreativityLevel::Moderate,
        };
        
        // Adjust innovation threshold based on style characteristics
        if style.characteristics.contains(&"innovative".to_string()) {
            style_config.creative.innovation_threshold = 0.8;
        }
        
        Some(style_config)
    }

    /// Get recommended modality weights for task
    pub fn get_recommended_modality_weights_for_task(&self, task: &str) -> HashMap<String, f32> {
        let mut weights = self.multimodal.modality_weights.clone();
        
        // Adjust weights based on task type
        if task.contains("visual") || task.contains("image") {
            weights.insert("image".to_string(), 0.5);
            weights.insert("text".to_string(), 0.2);
            weights.insert("audio".to_string(), 0.1);
            weights.insert("video".to_string(), 0.2);
        } else if task.contains("audio") || task.contains("music") {
            weights.insert("audio".to_string(), 0.5);
            weights.insert("text".to_string(), 0.3);
            weights.insert("image".to_string(), 0.1);
            weights.insert("video".to_string(), 0.1);
        } else if task.contains("multimedia") || task.contains("video") {
            weights.insert("video".to_string(), 0.4);
            weights.insert("image".to_string(), 0.2);
            weights.insert("audio".to_string(), 0.2);
            weights.insert("text".to_string(), 0.2);
        }
        
        weights
    }

    /// Get recommended style for task
    pub fn get_recommended_style_for_task(&self, task: &str) -> Option<String> {
        if task.contains("abstract") || task.contains("modern") {
            Some("Abstract Expressionism".to_string())
        } else if task.contains("classical") || task.contains("traditional") {
            Some("Impressionism".to_string())
        } else if task.contains("digital") || task.contains("contemporary") {
            Some("Digital Art".to_string())
        } else {
            None
        }
    }

    /// Get innovation strategy for domain
    pub fn get_innovation_strategy_for_domain(&self, domain: &InnovationDomain) -> InnovationMode {
        match domain {
            InnovationDomain::Visual => InnovationMode::Transformative,
            InnovationDomain::Audio => InnovationMode::Radical,
            InnovationDomain::Text => InnovationMode::Incremental,
            InnovationDomain::Interactive => InnovationMode::Disruptive,
            InnovationDomain::Conceptual => InnovationMode::Transformative,
            InnovationDomain::Technical => InnovationMode::Adaptive,
        }
    }

    /// Calculate creative potential score
    pub fn calculate_creative_potential_score(&self) -> f32 {
        let creativity_level_score = match self.creative.creativity_level {
            CreativityLevel::Conservative => 0.2,
            CreativityLevel::Moderate => 0.4,
            CreativityLevel::High => 0.6,
            CreativityLevel::Maximum => 0.8,
            CreativityLevel::Transcendent => 1.0,
        };
        
        let cross_modal_bonus = if self.creative.enable_cross_modal_creativity { 0.1 } else { 0.0 };
        let style_learning_bonus = if self.style.style_learning { 0.1 } else { 0.0 };
        let concept_generation_bonus = if self.innovation.enable_concept_generation { 0.1 } else { 0.0 };
        
        (creativity_level_score + cross_modal_bonus + style_learning_bonus + concept_generation_bonus).min(1.0)
    }

    /// Get configuration summary
    pub fn get_configuration_summary(&self) -> ConfigurationSummary {
        ConfigurationSummary {
            creativity_level: format!("{:?}", self.creative.creativity_level),
            supported_modalities: self.get_supported_modalities(),
            supported_styles: self.get_supported_styles(),
            innovation_domains: self.get_innovation_domains(),
            creative_potential_score: self.calculate_creative_potential_score(),
            cross_modal_creativity: self.creative.enable_cross_modal_creativity,
            style_learning: self.style.style_learning,
            concept_generation: self.innovation.enable_concept_generation,
        }
    }
}

/// Configuration summary
#[derive(Debug, Clone)]
pub struct ConfigurationSummary {
    pub creativity_level: String,
    pub supported_modalities: Vec<String>,
    pub supported_styles: Vec<String>,
    pub innovation_domains: Vec<String>,
    pub creative_potential_score: f32,
    pub cross_modal_creativity: bool,
    pub style_learning: bool,
    pub concept_generation: bool,
}
