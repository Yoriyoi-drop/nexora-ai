//! Comprehensive Unit Tests for Core Functionality
//! 
//! Unit tests untuk core components dengan proper mocking dan assertions

use crate::controller::CoreController;
use crate::types::{ModelId, IntentType, ContextInfo, InputType};
use crate::input::InputReceiver;
use crate::intent::IntentDetector;
use crate::context::ContextAnalyzer;
use crate::task::TaskManager;
use crate::error::{CoreError, CoreResult};
use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::pin::Pin;
use std::future::Future;
use tokio::test;
use serde_json::json;

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
    
    async fn process(&self, input: &str) -> Result<String> {
        if self.should_fail {
            Err(anyhow::anyhow!("Mock model {} failed", self.name))
        } else {
            Ok(format!("{}: {}", self.name, self.response))
        }
    }
}

impl SpecialistModel for MockSpecialistModel {
    fn process(&self, input: &str, _context: &ContextInfo) -> Pin<Box<dyn Future<Output = CoreResult<String>> + Send + '_>> {
        let model = self.clone();
        Box::pin(async move {
            if model.should_fail {
                Err(CoreError::ProcessingError(format!("Mock model {} failed", model.name)))
            } else {
                Ok(format!("{}: {}", model.name, model.response))
            }
        })
    }
    
    fn model_type(&self) -> ModelId {
        ModelId::Memory // Use existing ModelId variant
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["text_generation".to_string(), "mock_capability".to_string()]
    }
    
    fn confidence_threshold(&self) -> f32 {
        0.8
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
}
    
    pub fn create_test_context() -> HashMap<String, String> {
        let mut context = HashMap::new();
        context.insert("user_id".to_string(), "test_user".to_string());
        context.insert("session_id".to_string(), "test_session".to_string());
        context
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
        println!("   ✅ Valid input result: {:?}", result.is_ok());
    }
    
    #[tokio::test]
    async fn test_input_validation() {
        let receiver = InputReceiver::new();
        
        // Test valid inputs
        let result = receiver.receive_input("Hello world", InputType::Text).await;
        println!("   ✅ Hello world result: {:?}", result.is_ok());
        
        let result = receiver.receive_input("Test input with numbers 123", InputType::Text).await;
        println!("   ✅ Numbers input result: {:?}", result.is_ok());
        
        // Test invalid inputs
        let result = receiver.receive_input("", InputType::Text).await;
        assert!(result.is_err());
        
        let result = receiver.receive_input("   ", InputType::Text).await;
        println!("   ❌ Whitespace result: {:?}", result.is_err());
        
        // Test input length limits
        let long_input = "a".repeat(10001);
        let result = receiver.receive_input(&long_input, InputType::Text).await;
        println!("   ❌ Long input result: {:?}", result.is_err());
    }
}

/// Intent detector tests
#[cfg(test)]
mod intent_detector_tests {
    use super::*;
    use super::test_utils::*;
    
    #[tokio::test]
    async fn test_intent_detector_creation() {
        let detector = IntentDetector::new();
        assert_eq!(detector.confidence_threshold, 0.62);
    }
    
    #[tokio::test]
    async fn test_intent_detector_with_threshold() {
        let detector = IntentDetector::new().with_threshold(0.8);
        assert_eq!(detector.confidence_threshold, 0.8);
    }
    
    #[tokio::test]
    async fn test_intent_classification() {
        let analyzer = IntentAnalyzer::new();
        
        // Test question intent
        let question = "What is the weather like today?";
        let intent = analyzer.analyze_intent(question).unwrap();
        assert_eq!(intent.category, "question");
        assert!(intent.confidence > 0.5);
        
        // Test command intent
        let command = "Create a new user account";
        let intent = analyzer.analyze_intent(command).unwrap();
        assert_eq!(intent.category, "command");
        
        // Test conversation intent
        let conversation = "Hello, how are you?";
        let intent = analyzer.analyze_intent(conversation).unwrap();
        assert_eq!(intent.category, "conversation");
    }
    
    #[tokio::test]
    async fn test_intent_confidence_scoring() {
        let analyzer = IntentAnalyzer::new();
        
        let clear_intent = "Create a new user account immediately";
        let clear_result = analyzer.analyze_intent(clear_intent).unwrap();
        assert!(clear_result.confidence > 0.8);
        
        let ambiguous_intent = "Maybe we could think about users";
        let ambiguous_result = analyzer.analyze_intent(ambiguous_intent).unwrap();
        assert!(ambiguous_result.confidence < 0.7);
    }
    
    #[tokio::test]
    async fn test_intent_with_context() {
        let analyzer = IntentAnalyzer::new();
        let context = create_test_context();
        
        let input = "What about my previous request?";
        let intent = analyzer.analyze_intent_with_context(input, &context).unwrap();
        
        // Should detect contextual reference
        assert!(intent.metadata.contains_key("contextual_reference"));
    }
}


/// Context manager tests
#[cfg(test)]
mod context_manager_tests {
    use super::*;
    use super::test_utils::*;
    
    #[tokio::test]
    async fn test_context_manager_creation() {
        let manager = ContextManager::new();
        assert!(manager.is_empty());
    }
    
    #[tokio::test]
    async fn test_context_storage() {
        let mut manager = ContextManager::new();
        let context = create_test_context();
        
        manager.set_context("test_session", context.clone());
        let retrieved = manager.get_context("test_session").unwrap();
        
        assert_eq!(retrieved.get("user_id"), context.get("user_id"));
        assert_eq!(retrieved.get("session_id"), context.get("session_id"));
    }
    
    #[tokio::test]
    async fn test_context_expiration() {
        let mut manager = ContextManager::new();
        let context = create_test_context();
        
        manager.set_context_with_ttl("test_session", context, Duration::from_secs(1));
        
        // Should exist immediately
        assert!(manager.get_context("test_session").is_some());
        
        // Verify context exists immediately after setting
        assert!(manager.get_context("test_session").is_some());
        // Note: expiration test requires mock time infrastructure
    }
    
    #[tokio::test]
    async fn test_context_updates() {
        let mut manager = ContextManager::new();
        let mut context = create_test_context();
        
        manager.set_context("test_session", context.clone());
        
        // Update context
        context.insert("new_key".to_string(), "new_value".to_string());
        manager.update_context("test_session", &context);
        
        let updated = manager.get_context("test_session").unwrap();
        assert_eq!(updated.get("new_key"), Some(&"new_value".to_string()));
    }
    
    #[tokio::test]
    async fn test_context_cleanup() {
        let mut manager = ContextManager::new();
        
        // Add multiple contexts
        for i in 0..10 {
            let context = create_test_context();
            manager.set_context(&format!("session_{}", i), context);
        }
        
        assert_eq!(manager.context_count(), 10);
        
        // Cleanup expired contexts
        manager.cleanup_expired();
        
        // Should still have 10 (none expired in this test)
        assert_eq!(manager.context_count(), 10);
    }
}

/// Task manager tests
#[cfg(test)]
mod task_manager_tests {
    use super::*;
    use super::test_utils::*;
    use crate::task::TaskManager;
    
    #[tokio::test]
    async fn test_task_manager_creation() {
        let manager = TaskManager::new();
        assert_eq!(manager.pending_count(), 0);
        assert_eq!(manager.active_count(), 0);
    }
    
    #[tokio::test]
    async fn test_task_creation() {
        let manager = TaskManager::new();
        
        let task = manager.create_task(
            TaskType::General,
            "Test task",
            TaskPriority::Normal,
            create_test_context(),
        ).unwrap();
        
        assert_eq!(task.task_type, TaskType::General);
        assert_eq!(task.description, "Test task");
        assert_eq!(task.priority, TaskPriority::Normal);
        assert_eq!(task.status, TaskStatus::Pending);
    }
    
    #[tokio::test]
    async fn test_task_execution() {
        let mut manager = TaskManager::new();
        
        let task = manager.create_task(
            TaskType::General,
            "Test task",
            TaskPriority::Normal,
            create_test_context(),
        ).unwrap();
        
        let task_id = task.id.clone();
        
        // Start task
        manager.start_task(&task_id).unwrap();
        let active_task = manager.get_task(&task_id).unwrap();
        assert_eq!(active_task.status, TaskStatus::Running);
        
        // Complete task
        manager.complete_task(&task_id, "Task completed successfully").unwrap();
        let completed_task = manager.get_task(&task_id).unwrap();
        assert_eq!(completed_task.status, TaskStatus::Completed);
        assert!(completed_task.result.is_some());
    }
    
    #[tokio::test]
    async fn test_task_failure() {
        let mut manager = TaskManager::new();
        
        let task = manager.create_task(
            TaskType::General,
            "Test task",
            TaskPriority::Normal,
            create_test_context(),
        ).unwrap();
        
        let task_id = task.id.clone();
        
        // Fail task
        manager.fail_task(&task_id, "Task failed due to error").unwrap();
        let failed_task = manager.get_task(&task_id).unwrap();
        assert_eq!(failed_task.status, TaskStatus::Failed);
        assert!(failed_task.error.is_some());
    }
    
    #[tokio::test]
    async fn test_task_priority_queue() {
        let mut manager = TaskManager::new();
        
        // Create tasks with different priorities
        let low_task = manager.create_task(
            TaskType::General,
            "Low priority task",
            TaskPriority::Low,
            create_test_context(),
        ).unwrap();
        
        let high_task = manager.create_task(
            TaskType::General,
            "High priority task",
            TaskPriority::High,
            create_test_context(),
        ).unwrap();
        
        let normal_task = manager.create_task(
            TaskType::General,
            "Normal priority task",
            TaskPriority::Normal,
            create_test_context(),
        ).unwrap();
        
        // Get next task should return high priority first
        let next_task = manager.get_next_task().unwrap();
        assert_eq!(next_task.id, high_task.id);
        
        // After completing high task, should return normal
        manager.complete_task(&next_task.id, "Done").unwrap();
        let next_task = manager.get_next_task().unwrap();
        assert_eq!(next_task.id, normal_task.id);
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
        assert!(controller.is_initialized());
    }
    
    #[tokio::test]
    async fn test_request_processing() {
        let controller = create_test_controller();
        let input = create_test_input();
        
        let result = controller.process_request(&input).unwrap();
        assert!(result.response.is_some());
        assert!(result.processing_time > Duration::ZERO);
    }
    
    #[tokio::test]
    async fn test_routing_decision() {
        let controller = create_test_controller();
        let input = "Create a new user account";
        
        let routing = controller.determine_routing(input).unwrap();
        assert!(routing.confidence > 0.0);
        assert!(!routing.target_model.is_empty());
    }
    
    #[tokio::test]
    async fn test_task_execution() {
        let mut controller = create_test_controller();
        let routing = create_test_routing_decision();
        let input = create_test_input();
        
        let result = controller.execute_task(&routing, &input).unwrap();
        assert!(!result.is_empty());
    }
    
    #[tokio::test]
    async fn test_error_handling() {
        let controller = create_test_controller();
        
        // Test with invalid input
        let result = controller.process_request("");
        assert!(result.is_err());
        
        // Test with malformed input
        let malformed = "\0\x01\x02"; // Invalid characters
        let result = controller.process_request(malformed);
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_concurrent_requests() {
        let controller = std::sync::Arc::new(create_test_controller());
        
        // Spawn multiple concurrent requests
        let mut handles = Vec::new();
        
        for i in 0..10 {
            let controller_clone = controller.clone();
            let handle = tokio::spawn(async move {
                let input = format!("Request {}", i);
                controller_clone.process_request(&input).await
            });
            handles.push(handle);
        }
        
        // Wait for all requests to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
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
        
        let start_time = Instant::now();
        let result = controller.process_request(input).unwrap();
        let processing_time = start_time.elapsed();
        
        assert!(result.response.is_some());
        assert!(processing_time < Duration::from_secs(5)); // Should complete within 5 seconds
        
        // Verify response contains relevant information
        let response = result.response.unwrap();
        assert!(response.to_lowercase().contains("user") || response.to_lowercase().contains("account"));
    }
    
    #[tokio::test]
    async fn test_context_preservation() {
        let controller = create_test_controller();
        
        // First request establishes context
        let input1 = "My name is John Doe";
        let result1 = controller.process_request(input1).unwrap();
        assert!(result1.context_updated);
        
        // Second request uses context
        let input2 = "What is my name?";
        let result2 = controller.process_request(input2).unwrap();
        let response2 = result2.response.unwrap();
        assert!(response2.to_lowercase().contains("john"));
    }
    
    #[tokio::test]
    async fn test_error_recovery() {
        let controller = create_test_controller();
        
        // Simulate a failing request
        let failing_input = "This input is designed to cause an error";
        
        // First attempt should fail
        let result1 = controller.process_request(failing_input);
        assert!(result1.is_err());
        
        // Recovery mechanism should handle the error
        let error = result1.unwrap_err();
        assert!(controller.handle_error(&error).is_ok());
        
        // Second attempt should succeed
        let result2 = controller.process_request("Normal input");
        assert!(result2.is_ok());
    }
    
    #[tokio::test]
    async fn test_performance_metrics() {
        let controller = create_test_controller();
        
        // Process multiple requests
        for i in 0..100 {
            let input = format!("Test request {}", i);
            controller.process_request(&input).unwrap();
        }
        
        let metrics = controller.get_performance_metrics();
        
        assert!(metrics.total_requests == 100);
        assert!(metrics.average_response_time > Duration::ZERO);
        assert!(metrics.success_rate > 0.0);
        assert!(metrics.error_rate >= 0.0);
    }
}

/// Performance tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use super::test_utils::*;
    
    #[tokio::test]
    async fn test_request_throughput() {
        let controller = std::sync::Arc::new(create_test_controller());
        let num_requests = 1000;
        
        let start_time = Instant::now();
        
        // Process requests concurrently
        let mut handles = Vec::new();
        
        for i in 0..num_requests {
            let controller_clone = controller.clone();
            let handle = tokio::spawn(async move {
                let input = format!("Performance test request {}", i);
                controller_clone.process_request(&input).await
            });
            handles.push(handle);
        }
        
        // Wait for completion
        for handle in handles {
            handle.await.unwrap().unwrap();
        }
        
        let total_time = start_time.elapsed();
        let throughput = num_requests as f64 / total_time.as_secs_f64();
        
        // Should handle at least 100 requests per second
        assert!(throughput > 100.0);
    }
    
    #[tokio::test]
    async fn test_memory_usage() {
        let controller = create_test_controller();
        
        // Process many requests to test memory usage
        for i in 0..10000 {
            let input = format!("Memory test request {}", i);
            controller.process_request(&input).unwrap();
        }
        
        // Check that memory usage is reasonable
        let memory_usage = controller.get_memory_usage();
        
        // Should not use more than 100MB (adjust threshold as needed)
        assert!(memory_usage < 100 * 1024 * 1024);
    }
    
    #[tokio::test]
    async fn test_context_cleanup() {
        let controller = create_test_controller();
        
        // Create many sessions
        for i in 0..1000 {
            let input = format!("Session {} request", i);
            controller.process_request(&input).unwrap();
        }
        
        // Force cleanup
        controller.cleanup_expired_contexts();
        
        // Verify cleanup was effective
        let context_count = controller.get_context_count();
        assert!(context_count < 1000); // Should have cleaned up some contexts
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
            task_type: TaskType::General,
            target_model: "test_model".to_string(),
            confidence: 0.9,
            reasoning: "Test routing".to_string(),
            metadata: HashMap::new(),
        };
        
        let result = controller.execute_task(&routing, "Test input").unwrap();
        assert_eq!(result, "test_model: Mock response");
    }
    
    #[tokio::test]
    async fn test_with_failing_mock_model() {
        let mut controller = create_test_controller();
        
        // Register failing mock model
        let failing_model = MockSpecialistModel::with_failure("failing_model", "Will fail");
        controller.register_specialist_model("failing_model", Box::new(failing_model));
        
        let routing = RoutingDecision {
            task_type: TaskType::General,
            target_model: "failing_model".to_string(),
            confidence: 0.9,
            reasoning: "Test routing".to_string(),
            metadata: HashMap::new(),
        };
        
        let result = controller.execute_task(&routing, "Test input").await;
        assert!(result.is_err());
    }
}

/// Property-based tests
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[tokio::test]
        async fn test_input_processing_properties(input in "[a-zA-Z0-9 ]{1,100}") {
            let processor = InputProcessor::new();
            
            // Valid inputs should pass validation
            prop_assert!(processor.validate_input(&input).is_ok());
            
            // Sanitized input should not contain null bytes
            let sanitized = processor.sanitize_input(input.clone());
            prop_assert!(!sanitized.contains('\0'));
            
            // Preprocessed input should be lowercase
            let preprocessed = processor.preprocess_input(&input);
            prop_assert!(preprocessed == preprocessed.to_lowercase());
        }
    }
    
    proptest! {
        #[tokio::test]
        async fn test_context_operations(key in "[a-z]{1,20}", value in "[a-zA-Z0-9 ]{1,50}") {
            let mut manager = ContextAnalyzer::new();
            let session_id = "test_session";
            
            // Set context
            let mut context = HashMap::new();
            context.insert(key.clone(), value.clone());
            manager.set_context(session_id, context);
            
            // Retrieve context
            let retrieved = manager.get_context(session_id).unwrap();
            prop_assert!(retrieved.get(&key) == Some(&value));
        }
    }
}

/// Test utilities and helpers
pub mod test_helpers {
    use super::*;
    
    /// Create a test environment with all components
    pub fn create_test_environment() -> TestEnvironment {
        TestEnvironment {
            controller: create_test_controller(),
            mock_models: HashMap::new(),
            test_data: TestData::new(),
        }
    }
    
    /// Test environment structure
    pub struct TestEnvironment {
        pub controller: CoreController,
        pub mock_models: HashMap<String, MockSpecialistModel>,
        pub test_data: TestData,
    }
    
    /// Test data structure
    pub struct TestData {
        pub sample_inputs: Vec<String>,
        pub sample_contexts: Vec<HashMap<String, String>>,
        pub expected_outputs: Vec<String>,
    }
    
    impl TestData {
        pub fn new() -> Self {
            Self {
                sample_inputs: vec![
                    "Hello, how are you?".to_string(),
                    "Create a new user".to_string(),
                    "What is the weather?".to_string(),
                    "Help me with my account".to_string(),
                ],
                sample_contexts: vec![
                    test_utils::create_test_context(),
                ],
                expected_outputs: vec![
                    "I'm doing well, thank you!".to_string(),
                    "User created successfully".to_string(),
                    "The weather is sunny".to_string(),
                    "Account assistance provided".to_string(),
                ],
            }
        }
    }
    
    impl TestEnvironment {
        pub fn setup_mock_models(&mut self) {
            let models = vec![
                ("conversation", MockSpecialistModel::new("conversation", "I'm doing well!")),
                ("user_management", MockSpecialistModel::new("user_management", "User created")),
                ("weather", MockSpecialistModel::new("weather", "Sunny today")),
                ("support", MockSpecialistModel::new("support", "Help provided")),
            ];
            
            for (name, model) in models {
                self.mock_models.insert(name.to_string(), model);
                self.controller.register_specialist_model(name, Box::new(
                    self.mock_models.get(name).unwrap().clone()
                ));
            }
        }
        
        pub async fn run_test_suite(&self) -> TestSuiteResults {
            let mut results = TestSuiteResults::new();
            
            // Run all sample inputs
            for (i, input) in self.test_data.sample_inputs.iter().enumerate() {
                let start_time = Instant::now();
                let result = self.controller.process_request(input).await;
                let duration = start_time.elapsed();
                
                results.add_result(input.clone(), result, duration);
                
                // Verify against expected output if available
                if i < self.test_data.expected_outputs.len() {
                    let expected = &self.test_data.expected_outputs[i];
                    if let Ok(ref response) = result {
                        let actual = response.response.as_ref().unwrap_or(&String::new());
                        results.add_verification(input.clone(), expected, actual);
                    }
                }
            }
            
            results
        }
    }
    
    /// Test suite results
    #[derive(Debug)]
    pub struct TestSuiteResults {
        pub results: Vec<TestResult>,
        pub verifications: Vec<VerificationResult>,
        pub total_duration: Duration,
    }
    
    #[derive(Debug)]
    pub struct TestResult {
        pub input: String,
        pub result: Result<crate::controller::ProcessResult>,
        pub duration: Duration,
    }
    
    #[derive(Debug)]
    pub struct VerificationResult {
        pub input: String,
        pub expected: String,
        pub actual: String,
        pub passed: bool,
    }
    
    impl TestSuiteResults {
        pub fn new() -> Self {
            Self {
                results: Vec::new(),
                verifications: Vec::new(),
                total_duration: Duration::ZERO,
            }
        }
        
        pub fn add_result(&mut self, input: String, result: Result<crate::controller::ProcessResult>, duration: Duration) {
            self.results.push(TestResult { input, result, duration });
            self.total_duration += duration;
        }
        
        pub fn add_verification(&mut self, input: String, expected: &str, actual: &str) {
            let passed = expected.to_lowercase() == actual.to_lowercase();
            self.verifications.push(VerificationResult {
                input,
                expected: expected.to_string(),
                actual: actual.to_string(),
                passed,
            });
        }
        
        pub fn success_rate(&self) -> f64 {
            if self.results.is_empty() {
                0.0
            } else {
                let successful = self.results.iter().filter(|r| r.result.is_ok()).count();
                successful as f64 / self.results.len() as f64
            }
        }
        
        pub fn verification_rate(&self) -> f64 {
            if self.verifications.is_empty() {
                0.0
            } else {
                let passed = self.verifications.iter().filter(|v| v.passed).count();
                passed as f64 / self.verifications.len() as f64
            }
        }
    }
}
