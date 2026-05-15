//! API request/response types

use serde::{Deserialize, Serialize};

/// Process request types
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessRequest {
    pub input: String,
    pub context: Option<String>,
    pub options: Option<ProcessOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessOptions {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessResponse {
    pub input: String,
    pub response: String,
    pub tokens_used: Option<usize>,
    pub processing_time_ms: Option<u64>,
    pub confidence: Option<f32>,
}

/// Generate request types
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop_sequences: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub prompt: String,
    pub generated: String,
    pub tokens_used: Option<usize>,
    pub processing_time_ms: Option<u64>,
    pub finish_reason: Option<String>,
}

/// Chat request types
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub conversation_id: Option<String>,
    pub context: Option<String>,
    pub options: Option<ChatOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatOptions {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub include_history: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: String,
    pub response: String,
    pub conversation_id: Option<String>,
    pub tokens_used: Option<usize>,
    pub processing_time_ms: Option<u64>,
    pub history: Option<Vec<ChatMessage>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

/// Code analysis request types
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyzeCodeRequest {
    pub code: String,
    pub language: String,
    pub options: Option<AnalysisOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisOptions {
    pub include_suggestions: Option<bool>,
    pub include_metrics: Option<bool>,
    pub include_security_check: Option<bool>,
    pub include_performance_analysis: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyzeCodeResponse {
    pub code: String,
    pub language: String,
    pub analysis: CodeAnalysis,
    pub suggestions: Option<Vec<CodeSuggestion>>,
    pub metrics: Option<CodeMetrics>,
    pub security_issues: Option<Vec<SecurityIssue>>,
    pub performance_issues: Option<Vec<PerformanceIssue>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeAnalysis {
    pub summary: String,
    pub complexity: Option<String>,
    pub quality_score: Option<f32>,
    pub issues: Vec<String>,
    pub strengths: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeSuggestion {
    pub type_: String,
    pub description: String,
    pub code_snippet: Option<String>,
    pub line_number: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeMetrics {
    pub lines_of_code: usize,
    pub cyclomatic_complexity: Option<usize>,
    pub maintainability_index: Option<f32>,
    pub technical_debt: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub severity: String,
    pub description: String,
    pub line_number: Option<usize>,
    pub recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceIssue {
    pub severity: String,
    pub description: String,
    pub line_number: Option<usize>,
    pub recommendation: String,
    pub estimated_impact: Option<String>,
}

/// Code generation request types
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateCodeRequest {
    pub description: String,
    pub language: String,
    pub options: Option<CodeGenerationOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeGenerationOptions {
    pub include_comments: Option<bool>,
    pub include_tests: Option<bool>,
    pub style: Option<String>,
    pub framework: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateCodeResponse {
    pub description: String,
    pub language: String,
    pub generated_code: String,
    pub explanation: Option<String>,
    pub tokens_used: Option<usize>,
    pub processing_time_ms: Option<u64>,
}
