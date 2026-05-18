use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    pub id: Uuid,
    pub text: String,
    pub token_ids: Option<Vec<u32>>,
    pub metadata: HashMap<String, String>,
    pub source: SourceInfo,
    pub stats: SampleStats,
    pub domains: Vec<Domain>,
    pub score: Option<f64>,
    pub curriculum_level: Option<CurriculumLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub name: String,
    pub url: Option<String>,
    pub trust_score: f64,
    pub category: SourceCategory,
    pub fetch_timestamp: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceCategory {
    Wikipedia,
    Arxiv,
    GitHub,
    WebCrawl,
    CommonCrawl,
    Books,
    Academic,
    Forum,
    SocialMedia,
    SEOFarm,
    Synthetic,
    Telemetry,
    Other,
    Documentation,
    News,
    StackOverflow,
    Reddit,
    YouTube,
    Patents,
    Government,
    Medical,
    Legal,
    Education,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Code,
    Reasoning,
    Memory,
    Architecture,
    Planning,
    Science,
    Math,
    Creative,
    Instruction,
    Conversation,
    Knowledge,
    General,
}

impl Domain {
    pub fn curriculum_level(&self) -> CurriculumLevel {
        match self {
            Domain::Conversation | Domain::Instruction => CurriculumLevel::BasicGrammar,
            Domain::Knowledge | Domain::Creative | Domain::General => CurriculumLevel::BasicInstruction,
            Domain::Code | Domain::Memory | Domain::Math => CurriculumLevel::MediumReasoning,
            Domain::Science | Domain::Architecture => CurriculumLevel::ChainOfThought,
            Domain::Reasoning => CurriculumLevel::AgenticPlanning,
            Domain::Planning => CurriculumLevel::MultiHopLogic,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SampleStats {
    pub char_count: usize,
    pub word_count: usize,
    pub token_count: usize,
    pub line_count: usize,
    pub entropy: f64,
    pub perplexity: f64,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CurriculumLevel {
    BasicGrammar = 1,
    BasicInstruction = 2,
    MediumReasoning = 3,
    ChainOfThought = 4,
    AgenticPlanning = 5,
    MultiHopLogic = 6,
}

#[derive(Debug, Clone)]
pub struct FilterConfig {
    pub name: String,
    pub enabled: bool,
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct FilterResult {
    pub passed: bool,
    pub sample_id: Uuid,
    pub filter_name: String,
    pub reason: Option<String>,
    pub score_delta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FilterAction {
    Accept,
    Reject,
    Reroute(Domain),
    Flag,
}

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_wait_ms: u64,
    pub prefetch_count: usize,
    pub enable_dynamic: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_wait_ms: 100,
            prefetch_count: 4,
            enable_dynamic: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub samples_in: u64,
    pub samples_accepted: u64,
    pub samples_rejected: u64,
    pub samples_rerouted: u64,
    pub total_latency_ms: u64,
    pub filter_breakdown: HashMap<String, FilterMetric>,
}

#[derive(Debug, Clone)]
pub struct FilterMetric {
    pub processed: u64,
    pub passed: u64,
    pub rejected: u64,
    pub avg_latency_us: f64,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            samples_in: 0,
            samples_accepted: 0,
            samples_rejected: 0,
            samples_rerouted: 0,
            total_latency_ms: 0,
            filter_breakdown: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScoreMap(pub HashMap<String, f64>);

impl Default for TrustScoreMap {
    fn default() -> Self {
        let mut m = HashMap::new();
        m.insert("wikipedia.org".to_string(), 0.95);
        m.insert("arxiv.org".to_string(), 0.97);
        m.insert("github.com".to_string(), 0.85);
        m.insert("en.wikipedia.org".to_string(), 0.95);
        m.insert("stackoverflow.com".to_string(), 0.80);
        m.insert("reddit.com".to_string(), 0.55);
        m.insert("twitter.com".to_string(), 0.40);
        m.insert("x.com".to_string(), 0.40);
        m.insert("medium.com".to_string(), 0.60);
        m.insert("blogspot.com".to_string(), 0.45);
        m.insert("youtube.com".to_string(), 0.50);
        m.insert("patents.google.com".to_string(), 0.90);
        m.insert("uspto.gov".to_string(), 0.92);
        m.insert("pubmed.ncbi.nlm.nih.gov".to_string(), 0.97);
        m.insert("ncbi.nlm.nih.gov".to_string(), 0.95);
        m.insert("clinicaltrials.gov".to_string(), 0.93);
        m.insert("data.gov".to_string(), 0.85);
        m.insert("europa.eu".to_string(), 0.88);
        m.insert("who.int".to_string(), 0.92);
        m.insert("courts.gov".to_string(), 0.90);
        m.insert("law.cornell.edu".to_string(), 0.88);
        m.insert("education.gov".to_string(), 0.85);
        m.insert("khanacademy.org".to_string(), 0.88);
        m.insert("coursera.org".to_string(), 0.82);
        m.insert("edx.org".to_string(), 0.84);
        m.insert("news.ycombinator.com".to_string(), 0.65);
        m.insert("reuters.com".to_string(), 0.78);
        m.insert("apnews.com".to_string(), 0.80);
        m.insert("bbc.com".to_string(), 0.82);
        m.insert("nature.com".to_string(), 0.96);
        Self(m)
    }
}
