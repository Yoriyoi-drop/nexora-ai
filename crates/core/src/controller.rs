//! Core Controller - Pusat lalu lintas untuk 10 model spesialis Nexora
//! 
//! Fungsi: menerima input, deteksi intent, routing, memory management
//! Migrasi dari C ke Rust dengan ownership dan thread safety


use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::SystemTime;

use crate::types::{ModelId, ContextInfo, ControllerConfig, InputType, IntentType, IntentResult, ControllerStats, InputData, ControllerState, RoutingDecision, SpecialistModel, LruContextCache, ControllerCore, ControllerMetrics};
use crate::input::InputReceiver;
use crate::intent::IntentDetector;
use crate::context::ContextAnalyzer;
use crate::error::{CoreError, CoreResult};
use parking_lot::RwLock;

use tracing::{debug, info, warn, instrument};


// Re-exports


/// Core Controller - Main struct untuk mengelola seluruh sistem
pub struct CoreController {
    /// State management dengan thread-safe access
    state: Arc<RwLock<ControllerState>>,
    /// Configuration
    config: ControllerConfig,
    /// Komponen-komponen sistem
    input_receiver: InputReceiver,
    intent_detector: IntentDetector,
    context_analyzer: ContextAnalyzer,
    /// Specialist models registry
    specialist_models: Arc<RwLock<HashMap<String, Box<dyn SpecialistModel>>>>,
    /// Context management dengan LRU cache
    context_cache: Arc<RwLock<LruContextCache>>,
    /// Performance metrics
    metrics: Arc<ControllerMetrics>,
}

impl CoreController {
    /// Create new CoreController dengan konfigurasi default
    pub fn new() -> Self {
        Self::with_config(ControllerConfig::default())
    }
    
    /// Create new CoreController dengan konfigurasi kustom
    pub fn with_config(config: ControllerConfig) -> Self {
        let state = ControllerState {
            current_input: None,
            detected_intent: None,
            context: None,
            is_processing: false,
            processing_start_time: 0,
            active_task_count: 0,
            last_active_model: ModelId::Controller,
            stats: ControllerStats::default(),
        };
        
        let context_cache = LruContextCache::new(config.context_cache_size);
        
        Self {
            state: Arc::new(RwLock::new(state)),
            config: config.clone(),
            input_receiver: InputReceiver::new(),
            intent_detector: IntentDetector::new().with_threshold(config.intent_threshold),
            context_analyzer: ContextAnalyzer::new().with_memory(config.enable_memory_management),
            specialist_models: Arc::new(RwLock::new(HashMap::new())),
            context_cache: Arc::new(RwLock::new(context_cache)),
            metrics: Arc::new(ControllerMetrics::default()),
        }
    }
    
    /// Process user request melalui complete pipeline
    #[instrument(skip(self))]
    pub async fn process_request(&self, input: &str, input_type: InputType) -> CoreResult<String> {
        let start_time = SystemTime::now();
        
        // Update metrics
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        
        info!("Processing request: type={:?}, length={}", input_type, input.len());
        
        // 1. Validate input
        let input_data = self.input_receiver.receive_input(input, input_type).await?;
        
        // 2. Update state
        {
            let mut state = self.state.write();
            state.current_input = Some(input_data.clone());
            state.is_processing = true;
            state.processing_start_time = ControllerCore::current_timestamp_ms();
            state.stats.total_requests_processed += 1;
        }
        
        // 3. Generate context key untuk caching
        let context_key = ControllerCore::generate_context_key(&input_data);
        
        // 4. Check context cache & parallelize context analysis with intent detection
        let (context_info, intent_result) = if let Some(cached_context) = self.get_cached_context(&context_key) {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            let intent_result = self.intent_detector.detect_intent(&input_data).await?;
            (cached_context, intent_result)
        } else {
            self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
            let (context, intent_result) = tokio::try_join!(
                self.context_analyzer.analyze_context(&input_data, ModelId::Controller),
                self.intent_detector.detect_intent(&input_data),
            )?;
            self.cache_context(context_key, &context);
            (context, intent_result)
        };
        
        // 6. Route to appropriate model
        let routing_decision = self.route_request(&intent_result, &context_info).await?;
        
        // 7. Execute task dengan specialist model
        let result = self.execute_task_internal(&routing_decision, input).await?;
        
        // 8. Update statistics dan metrics
        {
            let mut state = self.state.write();
            state.detected_intent = Some(intent_result);
            state.context = Some(context_info);
            
            // Track model switches
            if state.last_active_model != routing_decision.target_model {
                state.last_active_model = routing_decision.target_model;
                self.metrics.model_switches.fetch_add(1, Ordering::Relaxed);
            }
            
            state.is_processing = false;
            
            // Update processing time
            if let Ok(duration) = start_time.elapsed() {
                let processing_time = duration.as_millis() as usize;
                let total_requests = state.stats.total_requests_processed;
                if total_requests > 0 {
                    state.stats.avg_processing_time_ms = 
                        (state.stats.avg_processing_time_ms * (total_requests - 1) as f32 + processing_time as f32) / total_requests as f32;
                }
                
                // Update metrics
                self.metrics.avg_response_time_ms.store(
                    (self.metrics.avg_response_time_ms.load(Ordering::Relaxed) + processing_time as u64) / 2, 
                    Ordering::Relaxed
                );
            }
            
            state.stats.successful_routings += 1;
        }
        
        self.metrics.successful_requests.fetch_add(1, Ordering::Relaxed);
        info!("Request processed successfully in {:?}ms", start_time.elapsed().unwrap_or_default().as_millis());
        Ok(result)
    }
    
    /// Route request ke model yang sesuai
    async fn route_request(&self, intent: &IntentResult, context: &ContextInfo) -> CoreResult<RoutingDecision> {
        debug!("Routing request: primary_intent={:?}", intent.primary_intent);
        
        let target_model = ControllerCore::map_intent_to_model(intent.primary_intent);
        
        // Check model availability dengan specialist models registry
        if !self.is_model_available(target_model) {
            // Try to find alternative model from specialist models
            let alternative = self.find_alternative_model(target_model, intent);
            if let Some(alt_model) = alternative {
                warn!("Primary model {:?} not available, using alternative: {:?}", target_model, alt_model);
                return self.route_to_alternative_model(alt_model, intent, context).await;
            } else {
                return Err(CoreError::ModelNotAvailable { model_id: target_model as u8 });
            }
        }
        
        let routing_confidence = ControllerCore::calculate_routing_confidence(intent.primary_intent, context);
        
        if routing_confidence < self.config.routing_threshold {
            return Err(CoreError::Routing(format!(
                "Routing confidence {:.2} below threshold {:.2}", 
                routing_confidence, self.config.routing_threshold
            )));
        }
        
        let decision = RoutingDecision {
            target_model,
            routed_query: context.current_context.clone(),
            routing_confidence,
            routing_reasoning: format!(
                "Routed to {} based on intent {} with confidence {:.2}",
                target_model.name(),
                intent.primary_intent.name(),
                routing_confidence
            ),
            requires_multi_model: intent.is_multi_intent && self.config.enable_multi_model,
            secondary_models: if intent.is_multi_intent {
                ControllerCore::get_secondary_models(&intent.intents)
            } else {
                Vec::new()
            },
        };
        
        debug!("Routing decision: model={:?}, confidence={:.2}", decision.target_model, decision.routing_confidence);
        Ok(decision)
    }
    
    /// Check apakah model tersedia
    pub fn is_model_available(&self, model_id: ModelId) -> bool {
        if model_id == ModelId::Controller {
            return true;
        }
        let models = self.specialist_models.read();
        models.contains_key(model_id.name())
    }
    
    /// Get current context count (for testing)
    pub fn get_context_count(&self) -> usize {
        let cache = self.context_cache.read();
        cache.len()
    }
    
    /// Cleanup expired contexts (for testing)
    pub fn cleanup_expired_contexts(&self) {
        let mut cache = self.context_cache.write();
        cache.cleanup_expired();
        info!("Cleaned up contexts, remaining: {}", cache.len());
    }
    
    /// Register specialist model (for testing)
    pub fn register_specialist_model(&mut self, name: &str, model: Box<dyn SpecialistModel>) {
        let mut models = self.specialist_models.write();
        models.insert(name.to_string(), model);
        info!("Registered specialist model: {}", name);
    }
    
    /// Get cached context
    fn get_cached_context(&self, key: &str) -> Option<ContextInfo> {
        let mut cache = self.context_cache.write();
        cache.get(key)
    }
    
    /// Cache context dengan LRU policy
    fn cache_context(&self, key: String, context: &ContextInfo) {
        let mut cache = self.context_cache.write();
        cache.put(key, context, self.config.context_ttl_ms);
    }
    
    /// Find alternative model dari specialist models
    fn find_alternative_model(&self, target_model: ModelId, intent: &IntentResult) -> Option<ModelId> {
        ControllerCore::find_alternative_model(&self.specialist_models, target_model, intent)
    }
    
    /// Route to alternative model
    async fn route_to_alternative_model(&self, model: ModelId, intent: &IntentResult, context: &ContextInfo) -> CoreResult<RoutingDecision> {
        ControllerCore::route_to_alternative_model(model, intent, context).await
    }
    
    /// Execute task (public for testing)
    pub async fn execute_task(&self, routing: &RoutingDecision, original_input: &str) -> CoreResult<String> {
        self.execute_task_internal(routing, original_input).await
    }
    
    /// Internal execute task
    async fn execute_task_internal(&self, routing: &RoutingDecision, original_input: &str) -> CoreResult<String> {
        ControllerCore::execute_task(routing, original_input, &self.specialist_models).await
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> ControllerStats {
        self.state.read().stats.clone()
    }
    
    /// Reset controller state
    pub fn reset(&self) {
        let mut state = self.state.write();
        state.current_input = None;
        state.detected_intent = None;
        state.context = None;
        state.is_processing = false;
        state.processing_start_time = 0;
        state.active_task_count = 0;
        state.last_active_model = ModelId::Controller;
        
        info!("Controller state reset");
    }
    
    /// Check if controller is currently processing
    pub fn is_processing(&self) -> bool {
        self.state.read().is_processing
    }
    
    /// Get active task count
    pub fn active_task_count(&self) -> usize {
        self.state.read().active_task_count
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: ControllerConfig) {
        self.config = config.clone();
        self.intent_detector = IntentDetector::new().with_threshold(config.intent_threshold);
        self.context_analyzer = ContextAnalyzer::new().with_memory(config.enable_memory_management);
        
        info!("Configuration updated");
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &ControllerConfig {
        &self.config
    }
    
    fn current_timestamp_ms() -> u64 {
        ControllerCore::current_timestamp_ms()
    }
    
    /// Detect intent from input text - public interface
    pub async fn detect_intent(&self, input: &str) -> CoreResult<IntentResult> {
        let input_data = InputData::new(input.to_string(), InputType::Text);
        self.intent_detector.detect_intent(&input_data).await
    }
    
}

impl Default for CoreController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_process_request() {
        let mut config = ControllerConfig::default();
        config.routing_threshold = 0.3;
        let controller = CoreController::with_config(config);
        
        let result = controller.process_request("buat fungsi rust", InputType::Text).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(response.contains("Model Processing Result"));
    }
    
    #[tokio::test]
    async fn test_intent_mapping() {
        // Test coding intent
        assert_eq!(ControllerCore::map_intent_to_model(IntentType::Coding), ModelId::Coding);
        
        // Test memory intent
        assert_eq!(ControllerCore::map_intent_to_model(IntentType::Memory), ModelId::Memory);
        
        // Test unknown intent
        assert_eq!(ControllerCore::map_intent_to_model(IntentType::Unknown), ModelId::Controller);
    }
    
    #[test]
    fn test_routing_confidence() {
        let context = ContextInfo::new("test context".to_string(), ModelId::Controller);
        
        let confidence = ControllerCore::calculate_routing_confidence(IntentType::Coding, &context);
        assert!(confidence > 0.5);
        assert!(confidence <= 1.0);
    }
    
    #[test]
    fn test_controller_state() {
        let controller = CoreController::new();
        
        assert!(!controller.is_processing());
        assert_eq!(controller.active_task_count(), 0);
        
        let stats = controller.get_stats();
        assert_eq!(stats.total_requests_processed, 0);
    }
    
    #[test]
    fn test_reset() {
        let controller = CoreController::new();
        controller.reset();
        
        assert!(!controller.is_processing());
        assert_eq!(controller.active_task_count(), 0);
    }
}
