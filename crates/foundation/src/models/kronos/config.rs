//! NXR-KRONOS Configuration
//! 
//! Model-specific configuration for NXR-KRONOS

use serde::{Deserialize, Serialize};
use crate::shared::{model_config::NxrModelConfig, deeplearning_integration::DeepLearningConfig, gnac_integration::GnacIntegrationConfig};

/// NXR-KRONOS Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KronosConfig {
    /// Base configuration
    pub base: NxrModelConfig,
    /// Indexing configuration
    pub indexing: IndexingConfig,
    /// Semantic search configuration
    pub semantic_search: SemanticSearchConfig,
    /// Knowledge graph configuration
    pub knowledge_graph: KnowledgeGraphConfig,
    /// Agent configuration
    pub agents: AgentConfig,
    /// Deep learning configuration
    pub deep_learning: DeepLearningConfig,
    /// GNAC integration configuration
    pub gnac: GnacIntegrationConfig,
}

/// Indexing Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Indexing strategy
    pub indexing_strategy: IndexingStrategy,
    /// Indexing depth
    pub indexing_depth: IndexingDepth,
    /// Index format
    pub index_format: IndexFormat,
}

/// IndexingStrategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStrategy {
    /// Full-text indexing
    FullText,
    /// Semantic indexing
    Semantic,
    /// Hybrid indexing
    Hybrid,
    /// Custom indexing
    Custom { strategy: String },
}

/// IndexingDepth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingDepth {
    /// Shallow indexing
    Shallow,
    /// Medium indexing
    Medium,
    /// Deep indexing
    Deep,
    /// Comprehensive indexing
    Comprehensive,
}

/// IndexFormat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexFormat {
    /// Inverted index
    Inverted,
    /// Vector index
    Vector,
    /// Graph index
    Graph,
    /// Hybrid index
    Hybrid,
}

/// Semantic Search Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchConfig {
    /// Search algorithm
    pub search_algorithm: SearchAlgorithm,
    /// Embedding model
    pub embedding_model: EmbeddingModel,
    /// Similarity metric
    pub similarity_metric: SimilarityMetric,
}

/// SearchAlgorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchAlgorithm {
    /// Vector search
    VectorSearch,
    /// Hybrid search
    HybridSearch,
    /// Graph search
    GraphSearch,
    /// Neural search
    NeuralSearch,
}

/// EmbeddingModel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModel {
    /// BERT-based
    BERT,
    /// RoBERTa-based
    RoBERTa,
    /// Custom model
    Custom { model_name: String },
}

/// SimilarityMetric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    DotProduct,
    /// Jaccard similarity
    Jaccard,
}

/// Knowledge Graph Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphConfig {
    /// Graph structure
    pub graph_structure: GraphStructure,
    /// Entity extraction
    pub entity_extraction: EntityExtraction,
    /// Relation extraction
    pub relation_extraction: RelationExtraction,
}

/// GraphStructure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphStructure {
    /// Directed graph
    Directed,
    /// Undirected graph
    Undirected,
    /// Heterogeneous graph
    Heterogeneous,
    /// Temporal graph
    Temporal,
}

/// EntityExtraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtraction {
    /// Extraction method
    pub extraction_method: ExtractionMethod,
    /// Entity types
    pub entity_types: Vec<String>,
}

/// ExtractionMethod
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionMethod {
    /// Rule-based extraction
    RuleBased,
    /// Machine learning extraction
    MachineLearning,
    /// Hybrid extraction
    Hybrid,
}

/// RelationExtraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationExtraction {
    /// Extraction method
    pub extraction_method: ExtractionMethod,
    /// Relation types
    pub relation_types: Vec<String>,
}

/// Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// INDEX-BUILDER configuration
    pub index_builder: IndexBuilderConfig,
    /// SEMANTIC-SEARCH configuration
    pub semantic_search: SemanticSearchAgentConfig,
    /// KNOWLEDGE-GRAPH configuration
    pub knowledge_graph: KnowledgeGraphAgentConfig,
    /// SYNTHESIZER configuration
    pub synthesizer: SynthesizerConfig,
}

/// IndexBuilderConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexBuilderConfig {
    /// Indexing mode
    pub indexing_mode: IndexingMode,
    /// Update frequency
    pub update_frequency: UpdateFrequency,
    /// Batch size
    pub batch_size: u32,
}

/// IndexingMode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingMode {
    /// Real-time indexing
    RealTime,
    /// Batch indexing
    Batch,
    /// Incremental indexing
    Incremental,
}

/// UpdateFrequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateFrequency {
    /// Manual updates
    Manual,
    /// Periodic updates
    Periodic { interval_hours: u32 },
    /// Event-driven updates
    EventDriven,
}

/// SemanticSearchAgentConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchAgentConfig {
    /// Search depth
    pub search_depth: SearchDepth,
    /// Result limit
    pub result_limit: u32,
    /// Reranking enabled
    pub reranking_enabled: bool,
}

/// SearchDepth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchDepth {
    /// Shallow search
    Shallow,
    /// Deep search
    Deep,
    /// Exhaustive search
    Exhaustive,
}

/// KnowledgeGraphAgentConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphAgentConfig {
    /// Graph construction
    pub graph_construction: GraphConstruction,
    /// Graph querying
    pub graph_querying: GraphQuerying,
    /// Graph updates
    pub graph_updates: GraphUpdates,
}

/// GraphConstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphConstruction {
    /// Automatic construction
    Automatic,
    /// Manual construction
    Manual,
    /// Hybrid construction
    Hybrid,
}

/// GraphQuerying
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuerying {
    /// Query language
    pub query_language: QueryLanguage,
    /// Query optimization
    pub query_optimization: bool,
}

/// QueryLanguage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryLanguage {
    /// SPARQL
    SPARQL,
    /// Cypher
    Cypher,
    /// Gremlin
    Gremlin,
    /// Custom language
    Custom { language: String },
}

/// GraphUpdates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphUpdates {
    /// Update strategy
    pub update_strategy: UpdateStrategy,
    /// Update frequency
    pub update_frequency: UpdateFrequency,
}

/// SynthesizerConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizerConfig {
    /// Synthesis method
    pub synthesis_method: SynthesisMethod,
    /// Abstraction level
    pub abstraction_level: AbstractionLevel,
    /// Output format
    pub output_format: OutputFormat,
}

/// SynthesisMethod
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisMethod {
    /// Abstractive synthesis
    Abstractive,
    /// Extractive synthesis
    Extractive,
    /// Hybrid synthesis
    Hybrid,
}

/// AbstractionLevel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbstractionLevel {
    /// Low abstraction
    Low,
    /// Medium abstraction
    Medium,
    /// High abstraction
    High,
}

/// OutputFormat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Summary format
    Summary,
    /// Report format
    Report,
    /// Knowledge base format
    KnowledgeBase,
}

impl Default for KronosConfig {
    fn default() -> Self {
        Self {
            base: NxrModelConfig::for_model(crate::shared::model_identity::NxrModelId::Kronos),
            indexing: IndexingConfig::default(),
            semantic_search: SemanticSearchConfig::default(),
            knowledge_graph: KnowledgeGraphConfig::default(),
            agents: AgentConfig::default(),
            deep_learning: DeepLearningConfig::star_x(),
            gnac: GnacIntegrationConfig::default(),
        }
    }
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            indexing_strategy: IndexingStrategy::Hybrid,
            indexing_depth: IndexingDepth::Deep,
            index_format: IndexFormat::Hybrid,
        }
    }
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            search_algorithm: SearchAlgorithm::HybridSearch,
            embedding_model: EmbeddingModel::BERT,
            similarity_metric: SimilarityMetric::Cosine,
        }
    }
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            graph_structure: GraphStructure::Heterogeneous,
            entity_extraction: EntityExtraction {
                extraction_method: ExtractionMethod::Hybrid,
                entity_types: vec![
                    "person".to_string(),
                    "organization".to_string(),
                    "location".to_string(),
                    "concept".to_string(),
                ],
            },
            relation_extraction: RelationExtraction {
                extraction_method: ExtractionMethod::Hybrid,
                relation_types: vec![
                    "belongs_to".to_string(),
                    "located_in".to_string(),
                    "related_to".to_string(),
                ],
            },
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            index_builder: IndexBuilderConfig::default(),
            semantic_search: SemanticSearchAgentConfig::default(),
            knowledge_graph: KnowledgeGraphAgentConfig::default(),
            synthesizer: SynthesizerConfig::default(),
        }
    }
}

impl Default for IndexBuilderConfig {
    fn default() -> Self {
        Self {
            indexing_mode: IndexingMode::Incremental,
            update_frequency: UpdateFrequency::Periodic { interval_hours: 24 },
            batch_size: 1000,
        }
    }
}

impl Default for SemanticSearchAgentConfig {
    fn default() -> Self {
        Self {
            search_depth: SearchDepth::Deep,
            result_limit: 100,
            reranking_enabled: true,
        }
    }
}

impl Default for KnowledgeGraphAgentConfig {
    fn default() -> Self {
        Self {
            graph_construction: GraphConstruction::Hybrid,
            graph_querying: GraphQuerying {
                query_language: QueryLanguage::Cypher,
                query_optimization: true,
            },
            graph_updates: GraphUpdates {
                update_strategy: UpdateStrategy::EventDriven,
                update_frequency: UpdateFrequency::EventDriven,
            },
        }
    }
}

impl Default for SynthesizerConfig {
    fn default() -> Self {
        Self {
            synthesis_method: SynthesisMethod::Hybrid,
            abstraction_level: AbstractionLevel::Medium,
            output_format: OutputFormat::Summary,
        }
    }
}

impl KronosConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate base configuration
        self.base.validate()?;

        // Validate indexing configuration
        if matches!(self.indexing.indexing_depth, IndexingDepth::Shallow) {
            return Err("Indexing depth cannot be shallow".to_string());
        }

        // Validate semantic search configuration
        // Note: result_limit validation removed as field doesn't exist in SemanticSearchConfig

        // Validate knowledge graph configuration
        if self.knowledge_graph.entity_extraction.entity_types.is_empty() {
            return Err("At least one entity type required".to_string());
        }

        // Validate deep learning configuration
        self.deep_learning.validate()?;

        Ok(())
    }

    /// Get configuration for specific agent
    pub fn get_agent_config(&self, agent_name: &str) -> Option<serde_json::Value> {
        match agent_name {
            "index_builder" => Some(serde_json::to_value(&self.agents.index_builder).unwrap_or_default()),
            "semantic_search" => Some(serde_json::to_value(&self.agents.semantic_search).unwrap_or_default()),
            "knowledge_graph" => Some(serde_json::to_value(&self.agents.knowledge_graph).unwrap_or_default()),
            "synthesizer" => Some(serde_json::to_value(&self.agents.synthesizer).unwrap_or_default()),
            _ => None,
        }
    }
}
