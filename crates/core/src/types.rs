//! Tipe data fundamental untuk Nexora Core
//! 
//! Module ini berisi semua tipe data dasar yang dimigrasi dari C ke Rust
//! dengan ownership dan type safety yang lebih baik.

use serde::{Deserialize, Serialize};

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
