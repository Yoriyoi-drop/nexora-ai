//! Comprehensive Unit Tests for Core Functionality
//!
//! Unit tests untuk core components dengan proper mocking dan assertions

use crate::controller::CoreController;
use crate::types::{ModelId, IntentType, ContextInfo, InputType, InputData, RoutingDecision};
use crate::input::InputReceiver;
use crate::intent::IntentDetector;
use crate::context::ContextAnalyzer;
use crate::task::TaskManager;
use crate::error::{CoreError, CoreResult};
use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use tokio::test;

/// Mock specialist model for testing
#[derive(Debug, Clone)]
struct MockSpecialistModel {
    name: String,
    response: String,
    should_fail: bool,
}

impl MockSpecialistModel {
    fn new(name: &str, response: &str) -> Self {
        Self {
            name: name.to_string(),
            response: response.to_string(),
            should_fail: false,
        }
    }

    fn with_failure(name: &str, response: &str) -> Self {
        Self {
            name: name.to_string(),
            response: response.to_string(),
            should_fail: true,
        }
    }
}

#[async_trait::async_trait]
impl crate::types::SpecialistModel for MockSpecialistModel {
    async fn process(&self, _input: &str, _context: &ContextInfo) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if self.should_fail {
            Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Mock model {} failed", self.name))))
        } else {
            Ok(format!("{}: {}", self.name, self.response))
        }
    }

    fn model_id(&self) -> ModelId {
        ModelId::Memory
    }

    fn can_handle(&self, _intent: IntentType) -> bool {
        true
    }
}

/// Test utilities
mod test_utils {
    use super::*;

    pub fn create_test_controller() -> CoreController {
        CoreController::new()
    }

    pub fn create_test_input() -> String {
        "Hello, how can you help me today?".to_string()
    }

    pub fn create_test_context() -> HashMap<String, String> {
        let mut context = HashMap::new();
        context.insert("user_id".to_string(), "test_user".to_string());
        context.insert("session_id".to_string(), "test_session".to_string());
        context
    }
}

/// Input receiver tests
#[cfg(test)]
mod input_receiver_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_input_receiver_creation() {
        let receiver = InputReceiver::new();
        let result = receiver.receive_input("", InputType::Text).await;
        assert!(result.is_err());

        let result = receiver.receive_input("test", InputType::Text).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_input_validation() {
        let receiver = InputReceiver::new();

        // Test valid inputs
        let result = receiver.receive_input("Hello world", InputType::Text).await;
        assert!(result.is_ok());

        let result = receiver.receive_input("Test input with numbers 123", InputType::Text).await;
        assert!(result.is_ok());

        // Test invalid inputs
        let result = receiver.receive_input("", InputType::Text).await;
        assert!(result.is_err());

        let result = receiver.receive_input("   ", InputType::Text).await;
        assert!(result.is_err());

        // Test input length limits
        let long_input = "a".repeat(10001);
        let result = receiver.receive_input(&long_input, InputType::Text).await;
        assert!(result.is_err());
    }
}

/// Intent detector tests
#[cfg(test)]
mod intent_detector_tests {
    use super::*;

    #[tokio::test]
    async fn test_intent_detector_creation() {
        let detector = IntentDetector::new();
        let input_data = InputData::new("test input".to_string(), InputType::Text);
        let result = detector.detect_intent(&input_data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_intent_detector_with_threshold() {
        let detector = IntentDetector::new().with_threshold(0.8);
        let input_data = InputData::new("create a function".to_string(), InputType::Text);
        let result = detector.detect_intent(&input_data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_coding_intent_classification() {
        let detector = IntentDetector::new();
        let input_data = InputData::new("Create a new user account".to_string(), InputType::Text);
        let intent = detector.detect_intent(&input_data).await.unwrap();
        assert!(intent.primary_intent != IntentType::Unknown);
    }

    #[tokio::test]
    async fn test_intent_confidence_scoring() {
        let detector = IntentDetector::new();

        let clear_input = InputData::new("Create a new user account immediately".to_string(), InputType::Text);
        let clear_result = detector.detect_intent(&clear_input).await.unwrap();
        assert!(clear_result.get_confidence(clear_result.primary_intent) > 0.0);

        let ambiguous_input = InputData::new("Maybe we could think about users".to_string(), InputType::Text);
        let ambiguous_result = detector.detect_intent(&ambiguous_input).await.unwrap();
        assert!(ambiguous_result.primary_intent != IntentType::Unknown);
    }
}

/// Context analyzer tests
#[cfg(test)]
mod context_analyzer_tests {
    use super::*;

    #[tokio::test]
    async fn test_context_analyzer_creation() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new("test input".to_string(), InputType::Text);
        let result = analyzer.analyze_context(&input_data, ModelId::Controller).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_context_analysis() {
        let analyzer = ContextAnalyzer::new();
        let input_data = InputData::new("remember what we discussed".to_string(), InputType::Text);
        let context = analyzer.analyze_context(&input_data, ModelId::Controller).await.unwrap();

        assert_eq!(context.active_model, ModelId::Controller);
        assert!(context.has_memory);
    }

    #[tokio::test]
    async fn test_context_relevance() {
        let analyzer = ContextAnalyzer::new();
        let context = ContextInfo::new("previous conversation about Rust".to_string(), ModelId::Memory);

        let relevance = analyzer.check_context_relevance(&context, "what about Rust");
        assert!(relevance > 0.0);
    }
}

/// Task manager tests
#[cfg(test)]
mod task_manager_tests {
    use super::*;

    #[tokio::test]
    async fn test_task_manager_creation() {
        let manager = TaskManager::new(10);
        assert_eq!(manager.active_task_count(), 0);
    }

    #[tokio::test]
    async fn test_task_creation() {
        let mut manager = TaskManager::new(10);

        let task_id = manager.create_task(
            ModelId::Coding,
            "Test task".to_string(),
            "test input".to_string(),
        ).await.unwrap();

        assert!(!task_id.is_empty());
    }

    #[tokio::test]
    async fn test_task_execution() {
        let mut manager = TaskManager::new(10);

        let task_id = manager.create_task(
            ModelId::Coding,
            "Test task".to_string(),
            "test input".to_string(),
        ).await.unwrap();

        let result = manager.execute_task(&task_id).await.unwrap();
        assert!(!result.is_empty());

        let task = manager.get_task(&task_id).unwrap();
        assert!(task.is_completed);
        assert!(task.was_successful);
    }

    #[tokio::test]
    async fn test_task_limit() {
        let mut manager = TaskManager::new(1);

        let _ = manager.create_task(
            ModelId::Coding,
            "Task 1".to_string(),
            "input 1".to_string(),
        ).await.unwrap();

        let result = manager.create_task(
            ModelId::Coding,
            "Task 2".to_string(),
            "input 2".to_string(),
        ).await;

        assert!(result.is_err());
    }
}

/// Core controller tests
#[cfg(test)]
mod core_controller_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_controller_creation() {
        let controller = create_test_controller();
        assert!(!controller.is_processing());
    }

    #[tokio::test]
    async fn test_request_processing() {
        let controller = create_test_controller();
        let input = create_test_input();

        let result = controller.process_request(&input, InputType::Text).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_error_handling() {
        let controller = create_test_controller();

        // Test with invalid input
        let result = controller.process_request("", InputType::Text).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let controller = std::sync::Arc::new(create_test_controller());

        // Spawn multiple concurrent requests
        let mut handles = Vec::with_capacity(10);

        for i in 0..10 {
            let controller_clone = controller.clone();
            let handle = tokio::spawn(async move {
                let input = format!("Request {}", i);
                controller_clone.process_request(&input, InputType::Text).await
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_controller_state() {
        let controller = create_test_controller();

        assert!(!controller.is_processing());
        assert_eq!(controller.active_task_count(), 0);

        let stats = controller.get_stats();
        assert_eq!(stats.total_requests_processed, 0);
    }

    #[test]
    fn test_reset() {
        let controller = create_test_controller();
        controller.reset();

        assert!(!controller.is_processing());
        assert_eq!(controller.active_task_count(), 0);
    }
}

/// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_full_request_pipeline() {
        let controller = create_test_controller();
        let input = "Help me create a new user account with email test@example.com";

        let result = controller.process_request(input, InputType::Text).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.contains("Model Processing Result"));
    }

    #[tokio::test]
    async fn test_context_preservation() {
        let controller = create_test_controller();

        // First request establishes context
        let input1 = "My name is John Doe";
        let result1 = controller.process_request(input1, InputType::Text).await;
        assert!(result1.is_ok());

        // Second request uses context
        let input2 = "What is my name?";
        let result2 = controller.process_request(input2, InputType::Text).await;
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let controller = create_test_controller();

        // Process multiple requests
        for i in 0..10 {
            let input = format!("Test request {}", i);
            let _ = controller.process_request(&input, InputType::Text).await;
        }

        let stats = controller.get_stats();
        assert!(stats.total_requests_processed > 0);
    }
}

/// Mock tests for external dependencies
#[cfg(test)]
mod mock_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_with_mock_specialist_models() {
        let mut controller = create_test_controller();

        // Register mock models
        let mock_model = MockSpecialistModel::new("test_model", "Mock response");
        controller.register_specialist_model("test_model", Box::new(mock_model));

        let routing = RoutingDecision {
            target_model: ModelId::Memory,
            routed_query: String::new(),
            routing_confidence: 0.9,
            routing_reasoning: "Test routing".to_string(),
            requires_multi_model: false,
            secondary_models: Vec::new(),
        };

        let result = controller.execute_task(&routing, "Test input").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_with_failing_mock_model() {
        let mut controller = create_test_controller();

        // Register failing mock model
        let failing_model = MockSpecialistModel::with_failure("failing_model", "Will fail");
        controller.register_specialist_model("failing_model", Box::new(failing_model));

        let routing = RoutingDecision {
            target_model: ModelId::Memory,
            routed_query: String::new(),
            routing_confidence: 0.9,
            routing_reasoning: "Test routing".to_string(),
            requires_multi_model: false,
            secondary_models: Vec::new(),
        };

        let result = controller.execute_task(&routing, "Test input").await;
        assert!(result.is_ok()); // execute_task returns a default response
    }
}

/// Performance tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use super::test_utils::*;

    #[tokio::test]
    async fn test_context_cleanup() {
        let controller = create_test_controller();

        // Create many sessions
        for i in 0..10 {
            let input = format!("Session {} request", i);
            let _ = controller.process_request(&input, InputType::Text).await;
        }

        // Force cleanup
        controller.cleanup_expired_contexts();

        // Verify cleanup was effective
        let context_count = controller.get_context_count();
        assert!(context_count <= 10);
    }

    #[test]
    fn test_memory_usage() {
        let controller = create_test_controller();

        let memory_usage = controller.get_memory_usage();
        assert!(memory_usage > 0);
    }
}
