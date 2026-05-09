//! Tipe data fundamental untuk Nexora Core
//! 
//! Module ini berisi semua tipe data dasar yang dimigrasi dari C ke Rust
//! dengan ownership dan type safety yang lebih baik.

use serde::{Deserialize, Serialize};
use crate::error::{CoreError, CoreResult};

// ==================== Input Types ====================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputType {
    Text = 0,
    Command = 1,
    Query = 2,
    Data = 3,
    Internal = 4,
}

impl Default for InputType {
    fn default() -> Self {
        InputType::Text
    }
}

// ==================== Intent Types ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentType {
    Unknown = 0,
    Coding = 1,        // Programming, code generation
    Memory = 2,        // Memory operations
    Debugging = 3,     // Debug, error analysis
    Planning = 4,      // Planning, organization
    Reasoning = 5,     // Logic, reasoning
    Ranking = 6,       // Ranking, evaluation
    Retrieval = 7,     // Information retrieval
    Validation = 8,    // Validation, checking
    Personality = 9,   // Personality, style
    Optimization = 10, // Optimization, improvement
}

impl Default for IntentType {
    fn default() -> Self {
        IntentType::Unknown
    }
}

impl IntentType {
    pub const COUNT: usize = 11;
    
    pub fn name(&self) -> &'static str {
        match self {
            IntentType::Unknown => "Unknown",
            IntentType::Coding => "Coding",
            IntentType::Memory => "Memory",
            IntentType::Debugging => "Debugging",
            IntentType::Planning => "Planning",
            IntentType::Reasoning => "Reasoning",
            IntentType::Ranking => "Ranking",
            IntentType::Retrieval => "Retrieval",
            IntentType::Validation => "Validation",
            IntentType::Personality => "Personality",
            IntentType::Optimization => "Optimization",
        }
    }
}

// ==================== Model IDs ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelId {
    Controller = 0,     // Core Controller (ini)
    Memory = 1,         // Memory specialist
    Coding = 2,         // Coding specialist
    Logic = 3,          // Logic specialist
    Planner = 4,         // Planning specialist
    Ranking = 5,         // Ranking specialist
    Retrieval = 6,       // Retrieval specialist
    Validator = 7,      // Validation specialist
    Personality = 8,    // Personality specialist
    Optimizer = 9,      // Optimization specialist
}

impl Default for ModelId {
    fn default() -> Self {
        ModelId::Controller
    }
}

impl ModelId {
    pub fn name(&self) -> &'static str {
        match self {
            ModelId::Controller => "Core Controller",
            ModelId::Memory => "Memory Specialist",
            ModelId::Coding => "Coding Specialist",
            ModelId::Logic => "Logic Specialist",
            ModelId::Planner => "Planning Specialist",
            ModelId::Ranking => "Ranking Specialist",
            ModelId::Retrieval => "Retrieval Specialist",
            ModelId::Validator => "Validation Specialist",
            ModelId::Personality => "Personality Specialist",
            ModelId::Optimizer => "Optimization Specialist",
        }
    }
}

// ==================== Memory Layers ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayer {
    Short = 0,    // Current request
    Session = 1,  // Active conversation
    Long = 2,     // Persistent knowledge
    Knowledge = 3, // Database internal
}

impl Default for MemoryLayer {
    fn default() -> Self {
        MemoryLayer::Short
    }
}

// ==================== Input Data ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputData {
    pub raw_input: String,        // Input asli dari user
    pub input_type: InputType,    // Tipe input
    pub timestamp: u64,           // Waktu input diterima
    pub metadata: Option<String>, // Metadata tambahan
    pub input_length: usize,      // Panjang input
    pub is_valid: bool,           // Validasi input
}

impl InputData {
    pub fn new(input: String, input_type: InputType) -> Self {
        let input_length = input.len();
        Self {
            raw_input: input,
            input_type,
            timestamp: Self::current_timestamp_ms(),
            metadata: None,
            input_length,
            is_valid: false, // Will be validated separately
        }
    }
    
    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    pub fn validate(&mut self) -> bool {
        self.is_valid = self.validate_input();
        self.is_valid
    }
    
    fn validate_input(&self) -> bool {
        // Basic validation
        if self.raw_input.is_empty() || self.input_length > 10000 {
            return false;
        }
        
        // Check for control characters (except newline and tab)
        if self.raw_input.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
            return false;
        }
        
        // Type-specific validation
        match self.input_type {
            InputType::Text => self.validate_text(),
            InputType::Command => self.validate_command(),
            InputType::Query => self.validate_query(),
            InputType::Data => self.validate_data(),
            InputType::Internal => true, // Internal input - always valid
        }
    }
    
    fn validate_text(&self) -> bool {
        // Text should contain printable characters (including Unicode)
        // Allow letters, numbers, punctuation, whitespace, and common Unicode characters
        self.raw_input.chars().all(|c| {
            c.is_alphanumeric() || 
            c.is_whitespace() || 
            c.is_ascii_punctuation() || 
            // Common symbols and Unicode characters
            "@#$%^&*()_+-=[]{}|;':\",./<>?~`".contains(c) ||
            // Allow common Unicode ranges
            (c as u32 >= 0x00A0 && c as u32 <= 0x00FF) || // Latin-1 supplement
            (c as u32 >= 0x0400 && c as u32 <= 0x04FF)    // Cyrillic
        })
    }
    
    fn validate_command(&self) -> bool {
        // Command should start with specific Indonesian command prefixes
        let input_lower = self.raw_input.to_lowercase();
        let command_prefixes = [
            "buat ", "simpan ", "cari ", "analisis ", "jelaskan ",
            "tambah ", "hapus ", "ubah ", "perbarui ", "tampilkan ",
            "hitung ", "proses ", "eksekusi ", "jalankan ", "mulai ",
            "stop ", "berhenti ", "reset ", "clear ", "kosongkan "
        ];
        
        command_prefixes.iter().any(|prefix| input_lower.starts_with(prefix))
    }
    
    fn validate_query(&self) -> bool {
        // Query validation - more comprehensive
        let input_lower = self.raw_input.to_lowercase();
        
        // Question words in Indonesian and English
        let question_words = [
            "apa", "bagaimana", "mengapa", "kapan", "dimana", "siapa", 
            "berapa", "adakah", "apakah", "bisakah", "bagaimanakah",
            "what", "how", "why", "when", "where", "who", "which",
            "can", "could", "would", "should", "is", "are", "do", "does"
        ];
        
        let has_question_word = question_words.iter().any(|word| {
            input_lower.contains(word) || input_lower.starts_with(word)
        });
        
        let has_question_mark = self.raw_input.contains('?');
        let is_long_enough = self.input_length > 10;
        
        // Valid if it has question words, question mark, or is reasonably long
        has_question_word || has_question_mark || is_long_enough
    }
    
    fn validate_data(&self) -> bool {
        // Data validation - check if it looks like structured data
        let has_json = self.raw_input.contains('{') && self.raw_input.contains('}');
        let has_array = self.raw_input.contains('[') && self.raw_input.contains(']');
        let has_key_value = self.raw_input.contains('=') || self.raw_input.contains(':');
        let has_csv = self.raw_input.split(',').count() > 1;
        let has_xml = self.raw_input.contains('<') && self.raw_input.contains('>');
        
        has_json || has_array || has_key_value || has_csv || has_xml
    }
    
    fn current_timestamp_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ==================== Intent Detection ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentScore {
    pub intent_type: IntentType,
    pub confidence: f32,  // 0.0 - 1.0
}

impl IntentScore {
    pub fn new(intent_type: IntentType, confidence: f32) -> Self {
        Self {
            intent_type,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    pub intents: Vec<IntentScore>,
    pub primary_intent: IntentType,
    pub intent_reasoning: String,
    pub is_multi_intent: bool,
}

impl IntentResult {
    pub fn new() -> Self {
        Self {
            intents: Vec::new(),
            primary_intent: IntentType::Unknown,
            intent_reasoning: String::new(),
            is_multi_intent: false,
        }
    }
    
    pub fn add_intent(&mut self, intent_type: IntentType, confidence: f32) {
        self.intents.push(IntentScore::new(intent_type, confidence));
        self.is_multi_intent = self.intents.len() > 1;
        
        // Update primary intent if this has higher confidence
        if let Some(highest) = self.intents.iter().max_by(|a, b| {
            a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.primary_intent = highest.intent_type;
        }
    }
    
    pub fn get_confidence(&self, intent_type: IntentType) -> f32 {
        self.intents
            .iter()
            .find(|score| score.intent_type == intent_type)
            .map(|score| score.confidence)
            .unwrap_or(0.0)
    }
}

// ==================== Context Information ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo {
    pub previous_context: Option<String>,
    pub current_context: String,
    pub active_model: ModelId,
    pub active_task: Option<String>,
    pub has_memory: bool,
    pub context_relevance: f32,  // 0.0 - 1.0
}

impl ContextInfo {
    pub fn new(current_context: String, active_model: ModelId) -> Self {
        Self {
            previous_context: None,
            current_context,
            active_model,
            active_task: None,
            has_memory: false,
            context_relevance: 0.8,
        }
    }
    
    pub fn with_previous_context(mut self, previous: String) -> Self {
        self.previous_context = Some(previous);
        self
    }
    
    pub fn update_context(&mut self, new_context: String) {
        self.previous_context = Some(std::mem::take(&mut self.current_context));
        self.current_context = new_context;
        self.context_relevance = 1.0; // Fresh context is fully relevant
    }
}

// ==================== Configuration ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    pub max_concurrent_tasks: usize,
    pub intent_threshold: f32,      // Lower = more sensitive
    pub routing_threshold: f32,     // Higher = more selective
    pub fusion_threshold: f32,
    pub enable_multi_model: bool,
    pub enable_memory_management: bool,
    pub context_ttl_ms: u64,        // Context TTL in milliseconds
    pub context_cache_size: usize,  // Maximum context cache size
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            intent_threshold: 0.5,   // More sensitive
            routing_threshold: 0.65, // More selective
            fusion_threshold: 0.7,
            enable_multi_model: true,
            enable_memory_management: true,
            context_ttl_ms: 300000,  // 5 minutes
            context_cache_size: 1000,
        }
    }
}

// ==================== Task Distribution ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecution {
    pub task_id: String,
    pub assigned_model: ModelId,
    pub task_description: String,
    pub task_input: String,
    pub is_completed: bool,
    pub was_successful: bool,
    pub task_output: Option<String>,
    pub start_time: u64,
    pub end_time: u64,
    pub retry_count: u32,
}

// ==================== Response Fusion ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    pub model_responses: Vec<String>,
    pub response_sources: Vec<ModelId>,
    pub response_count: usize,
    pub fused_response: String,
    pub fusion_confidence: f32,
    pub has_conflicts: bool,
    pub conflict_descriptions: Vec<String>,
}

// ==================== Statistics ====================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControllerStats {
    pub total_requests_processed: u64,
    pub successful_routings: u64,
    pub failed_routings: u64,
    pub memory_accesses: u64,
    pub avg_processing_time_ms: f32,
}

impl ControllerStats {
    pub fn success_rate(&self) -> f32 {
        if self.successful_routings + self.failed_routings == 0 {
            0.0
        } else {
            self.successful_routings as f32 / (self.successful_routings + self.failed_routings) as f32
        }
    }
}

// ==================== Controller State ====================

#[derive(Debug, Clone)]
pub struct ControllerState {
    pub current_input: Option<InputData>,
    pub detected_intent: Option<IntentResult>,
    pub context: Option<ContextInfo>,
    pub is_processing: bool,
    pub processing_start_time: u64,
    pub active_task_count: usize,
    pub last_active_model: ModelId,
    pub stats: ControllerStats,
}

// ==================== Routing Decision ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub target_model: ModelId,
    pub routed_query: String,
    pub routing_confidence: f32,
    pub routing_reasoning: String,
    pub requires_multi_model: bool,
    pub secondary_models: Vec<ModelId>,
}

// ==================== Specialist Model Trait ====================

#[async_trait::async_trait]
pub trait SpecialistModel: Send + Sync {
    async fn process(&self, input: &str, context: &ContextInfo) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
    fn model_id(&self) -> ModelId;
    fn can_handle(&self, intent: IntentType) -> bool;
}

// ==================== LRU Context Cache ====================

#[derive(Debug, Clone)]
pub struct LruContextCache {
    cache: std::collections::HashMap<String, (ContextInfo, u64)>, // (context, expiry_time)
    max_size: usize,
}

impl LruContextCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
        }
    }
    
    pub fn get(&mut self, key: &str) -> Option<ContextInfo> {
        let now = Self::current_timestamp_ms();
        if let Some((context, expiry)) = self.cache.get(key) {
            if now < *expiry {
                Some(context.clone())
            } else {
                self.cache.remove(key);
                None
            }
        } else {
            None
        }
    }
    
    pub fn put(&mut self, key: String, context: &ContextInfo, ttl_ms: u64) {
        let now = Self::current_timestamp_ms();
        let expiry = now + ttl_ms;
        
        // Remove expired entries if cache is full
        if self.cache.len() >= self.max_size {
            self.cleanup_expired();
        }
        
        // Still full? Remove oldest entry
        if self.cache.len() >= self.max_size {
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            }
        }
        
        self.cache.insert(key, (context.clone(), expiry));
    }
    
    pub fn len(&self) -> usize {
        self.cache.len()
    }
    
    pub fn cleanup_expired(&mut self) {
        let now = Self::current_timestamp_ms();
        self.cache.retain(|_, (_, expiry)| now < *expiry);
    }
    
    fn current_timestamp_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ==================== Controller Core Functions ====================

pub struct ControllerCore;

impl ControllerCore {
    pub fn current_timestamp_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    pub fn generate_context_key(input_data: &InputData) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        input_data.raw_input.hash(&mut hasher);
        input_data.input_type.hash(&mut hasher);
        format!("ctx_{:x}", hasher.finish())
    }
    
    pub fn map_intent_to_model(intent: IntentType) -> ModelId {
        match intent {
            IntentType::Coding => ModelId::Coding,
            IntentType::Memory => ModelId::Memory,
            IntentType::Debugging => ModelId::Logic,
            IntentType::Planning => ModelId::Planner,
            IntentType::Reasoning => ModelId::Logic,
            IntentType::Ranking => ModelId::Ranking,
            IntentType::Retrieval => ModelId::Retrieval,
            IntentType::Validation => ModelId::Validator,
            IntentType::Personality => ModelId::Personality,
            IntentType::Optimization => ModelId::Optimizer,
            IntentType::Unknown => ModelId::Controller,
        }
    }
    
    pub fn calculate_routing_confidence(intent: IntentType, context: &ContextInfo) -> f32 {
        let base_confidence = match intent {
            IntentType::Coding => 0.9,
            IntentType::Memory => 0.85,
            IntentType::Debugging => 0.8,
            IntentType::Planning => 0.75,
            IntentType::Reasoning => 0.8,
            IntentType::Ranking => 0.7,
            IntentType::Retrieval => 0.85,
            IntentType::Validation => 0.75,
            IntentType::Personality => 0.7,
            IntentType::Optimization => 0.8,
            IntentType::Unknown => 0.3,
        };
        
        // Adjust based on context relevance
        (base_confidence * context.context_relevance).min(1.0)
    }
    
    pub fn get_secondary_models(intents: &[IntentScore]) -> Vec<ModelId> {
        intents
            .iter()
            .filter(|score| score.confidence > 0.5)
            .map(|score| Self::map_intent_to_model(score.intent_type))
            .collect()
    }
    
    pub fn find_alternative_model(
        specialist_models: &std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<String, Box<dyn SpecialistModel>>>>,
        target_model: ModelId,
        intent: &IntentResult,
    ) -> Option<ModelId> {
        let models = specialist_models.read();
        
        // Try to find a model that can handle the primary intent
        for model in models.values() {
            if model.model_id() != target_model && model.can_handle(intent.primary_intent) {
                return Some(model.model_id());
            }
        }
        
        // Fallback to controller
        Some(ModelId::Controller)
    }
    
    pub async fn route_to_alternative_model(
        model: ModelId,
        intent: &IntentResult,
        context: &ContextInfo,
    ) -> CoreResult<RoutingDecision> {
        Ok(RoutingDecision {
            target_model: model,
            routed_query: context.current_context.clone(),
            routing_confidence: 0.6, // Lower confidence for alternative
            routing_reasoning: format!(
                "Alternative routing to {} due to primary model unavailability",
                model.name()
            ),
            requires_multi_model: false,
            secondary_models: Vec::new(),
        })
    }
    
    pub async fn execute_task(
        routing: &RoutingDecision,
        original_input: &str,
        specialist_models: &std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<String, Box<dyn SpecialistModel>>>>,
    ) -> CoreResult<String> {
        // For now, return a simple response
        // In a real implementation, this would call the actual specialist model
        Ok(format!(
            "Model Processing Result from {:?} for input: {}",
            routing.target_model, original_input
        ))
    }
}

// ==================== Controller Metrics ====================

#[derive(Debug, Default)]
pub struct ControllerMetrics {
    pub total_requests: std::sync::atomic::AtomicU64,
    pub successful_requests: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub cache_misses: std::sync::atomic::AtomicU64,
    pub model_switches: std::sync::atomic::AtomicU64,
    pub avg_response_time_ms: std::sync::atomic::AtomicU64,
}
