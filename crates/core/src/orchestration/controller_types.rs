//! Controller Types - Structs dan traits untuk Core Controller

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{CoreError, CoreResult};
use crate::types::{ModelId, ContextInfo, ControllerConfig, ControllerStats, InputData, IntentResult};

/// Specialist model trait untuk dynamic model registration
pub trait SpecialistModel: Send + Sync {
    fn process(&self, input: &str, context: &ContextInfo) -> Pin<Box<dyn Future<Output = CoreResult<String>> + Send + '_>>;
    fn model_type(&self) -> ModelId;
    fn capabilities(&self) -> Vec<String>;
    fn confidence_threshold(&self) -> f32;
}

/// Internal state dari CoreController
#[derive(Debug)]
pub struct ControllerState {
    /// Komponen utama Model 1
    pub current_input: Option<InputData>,
    pub detected_intent: Option<IntentResult>,
    pub context: Option<ContextInfo>,
    
    /// State management
    pub is_processing: bool,
    pub processing_start_time: u64,
    pub active_task_count: usize,
    pub last_active_model: ModelId,
    
    /// Statistics
    pub stats: ControllerStats,
}

/// LRU Cache untuk context management
#[derive(Debug)]
pub struct LruContextCache {
    pub cache: HashMap<String, ContextEntry>,
    pub access_order: VecDeque<String>,
    pub max_size: usize,
    pub cleanup_threshold: usize,
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub context: ContextInfo,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: usize,
    pub ttl_ms: u64,
}

/// Performance metrics untuk controller
#[derive(Debug, Default)]
pub struct ControllerMetrics {
    pub total_requests: AtomicUsize,
    pub successful_requests: AtomicUsize,
    pub failed_requests: AtomicUsize,
    pub avg_response_time_ms: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub cache_misses: AtomicUsize,
    pub model_switches: AtomicUsize,
}

/// Routing Decision struct
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub target_model: ModelId,
    pub routed_query: String,
    pub routing_confidence: f32,
    pub routing_reasoning: String,
    pub requires_multi_model: bool,
    pub secondary_models: Vec<ModelId>,
}
