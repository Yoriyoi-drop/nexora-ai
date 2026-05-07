//! Core type definitions for Nexora-AI

use chrono::{DateTime, Utc};

/// Input type for request routing
#[derive(Debug, Clone, PartialEq)]
pub enum InputType {
    Command,
    Query,
    Code,
    Data,
    Text,
}

/// Generation type for text generation
#[derive(Debug, Clone, PartialEq)]
pub enum GenerationType {
    Code,
    Question,
    LongForm,
    Short,
}

/// Prompt analysis for generation strategy
#[derive(Debug, Clone)]
pub struct PromptAnalysis {
    pub word_count: usize,
    pub sentence_count: usize,
    pub question_count: usize,
    pub code_blocks: usize,
    pub complexity_score: f64,
    pub generation_type: GenerationType,
}

/// Chat intent for message analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ChatIntent {
    Greeting,
    Question,
    Command,
    Casual,
    Code,
    System,
}

/// Sentiment analysis
#[derive(Debug, Clone, PartialEq)]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

/// Message urgency level
#[derive(Debug, Clone, PartialEq)]
pub enum Urgency {
    High,
    Medium,
    Low,
}

/// Chat message analysis
#[derive(Debug, Clone)]
pub struct ChatMessageAnalysis {
    pub intent: ChatIntent,
    pub sentiment: Sentiment,
    pub word_count: usize,
    pub has_code: bool,
    pub urgency: Urgency,
}

/// Conversation context
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub conversation_id: String,
    pub turn_count: usize,
    pub last_activity: DateTime<Utc>,
    pub topics: Vec<String>,
    pub user_preferences: UserPreferences,
}

/// User preferences for chat
#[derive(Debug, Clone, Default)]
pub struct UserPreferences {
    pub response_style: ResponseStyle,
    pub verbosity: VerbosityLevel,
    pub code_assistance: bool,
}

/// Response style preference
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseStyle {
    Formal,
    Casual,
    Technical,
    Friendly,
}

impl Default for ResponseStyle {
    fn default() -> Self {
        ResponseStyle::Friendly
    }
}

/// Verbosity level
#[derive(Debug, Clone, PartialEq)]
pub enum VerbosityLevel {
    Concise,
    Normal,
    Detailed,
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        VerbosityLevel::Normal
    }
}

/// Code analysis result
#[derive(Debug, Clone)]
pub struct CodeAnalysis {
    pub language: String,
    pub line_count: usize,
    pub character_count: usize,
    pub functions: Vec<FunctionInfo>,
    pub classes: Vec<ClassInfo>,
    pub imports: Vec<ImportInfo>,
    pub complexity: ComplexityMetrics,
    pub issues: Vec<CodeIssue>,
    pub patterns: Vec<PatternInfo>,
    pub metrics: CodeMetrics,
}

/// Function information
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub line_number: usize,
    pub parameters: String,
    pub return_type: Option<String>,
    pub visibility: String,
}

/// Class/Struct information
#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub name: String,
    pub line_number: usize,
    pub type_name: String,
    pub visibility: Option<String>,
    pub inheritance: Option<String>,
}

/// Import information
#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub module: String,
    pub line_number: usize,
    pub import_type: String,
}

/// Complexity metrics
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: u32,
    pub nested_loops: u32,
    pub conditionals: u32,
    pub functions: u32,
    pub total_lines: u32,
    pub code_lines: u32,
    pub comment_lines: u32,
    pub comment_ratio: f64,
}

/// Code issue
#[derive(Debug, Clone)]
pub struct CodeIssue {
    pub line_number: usize,
    pub severity: IssueSeverity,
    pub message: String,
    pub suggestion: String,
}

/// Issue severity
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Design pattern information
#[derive(Debug, Clone)]
pub struct PatternInfo {
    pub name: String,
    pub confidence: f64,
    pub description: String,
}

/// General code metrics
#[derive(Debug, Clone)]
pub struct CodeMetrics {
    pub total_lines: u32,
    pub empty_lines: u32,
    pub comment_lines: u32,
    pub code_lines: u32,
}

/// System information structure
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemInfo {
    pub version: String,
    pub uptime: u64,
    pub components: ComponentStatus,
    pub memory_stats: MemoryStats,
    pub active_models: Vec<String>,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub last_updated: DateTime<Utc>,
    pub process_count: u64,
    pub thread_count: u64,
    pub load_average: (f64, f64, f64),
}

/// Component health status
#[derive(Debug, Clone, serde::Serialize)]
pub struct ComponentStatus {
    pub core: String,
    pub models: String,
    pub memory: String,
    pub inference: String,
    pub agent: String,
    pub api: String,
}

/// Memory statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryStats {
    pub total_memory: u64,
    pub used_memory: u64,
    pub available_memory: u64,
    pub cache_size: u64,
}

/// Health check result
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub performance_score: f64,
    pub component_health: std::collections::HashMap<String, bool>,
    pub core_status: String,
    pub tokenizer_status: String,
    pub models_status: String,
    pub memory_status: String,
    pub total_operations: u64,
    pub average_response_time: f64,
    pub error_rate: f64,
    pub last_check: DateTime<Utc>,
    pub uptime_seconds: u64,
    pub active_connections: u64,
}
