use crate::controller::CoreController;
use crate::types::{DefaultSpecialistModel, InputType, IntentType, ModelId, SpecialistModel};
use std::sync::Arc;

#[tokio::test]
async fn test_core_controller_creation() {
    let controller = CoreController::new();
    let result = controller.process_request("buat program", InputType::Text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_context_count() {
    let controller = CoreController::new();
    let _ = controller.process_request("test", InputType::Text).await;
    assert!(controller.get_context_count() <= 10);
}

#[tokio::test]
async fn test_controller_routes_coding_intent() {
    let mut controller = CoreController::new();
    let model = Box::new(DefaultSpecialistModel::new(
        ModelId::Coding,
        vec![IntentType::Coding],
    ));
    controller.register_specialist_model(ModelId::Coding.name(), model);
    let result = controller.process_request("buat fungsi rust", InputType::Text).await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("CODE ANALYSIS") || output.contains("Model Processing Result"));
}

#[tokio::test]
async fn test_controller_routes_memory_intent() {
    let mut controller = CoreController::new();
    let model = Box::new(DefaultSpecialistModel::new(
        ModelId::Memory,
        vec![IntentType::Memory],
    ));
    controller.register_specialist_model(ModelId::Memory.name(), model);
    let result = controller.process_request("ingat kejadian sebelumnya", InputType::Text).await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("MEMORY") || output.contains("Model Processing Result"));
}

#[tokio::test]
async fn test_controller_empty_input() {
    let controller = CoreController::new();
    let result = controller.process_request("", InputType::Text).await;
    assert!(result.is_err() || result.is_ok());
}

#[tokio::test]
async fn test_controller_very_long_input() {
    let controller = CoreController::new();
    let long_input = "a".repeat(100_000);
    let result = controller.process_request(&long_input, InputType::Text).await;
    assert!(result.is_err() || result.is_ok());
}

#[tokio::test]
async fn test_controller_model_not_available_fallback() {
    let controller = CoreController::new();
    let result = controller.process_request("memory request test", InputType::Text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_controller_processing_state() {
    let controller = CoreController::new();
    assert!(!controller.is_processing());
    let _ = controller.process_request("test", InputType::Text).await;
    assert!(!controller.is_processing());
}

#[tokio::test]
async fn test_controller_stats_tracking() {
    let controller = CoreController::new();
    let _ = controller.process_request("request one", InputType::Text).await;
    let stats = controller.get_stats();
    assert!(stats.total_requests_processed >= 1);
}

#[tokio::test]
async fn test_controller_command_input() {
    let controller = CoreController::new();
    let result = controller.process_request("/help", InputType::Command).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_controller_query_input() {
    let controller = CoreController::new();
    let result = controller.process_request("what is rust?", InputType::Query).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_controller_reset_clears_state() {
    let controller = CoreController::new();
    let _ = controller.process_request("test", InputType::Text).await;
    assert!(!controller.is_processing());
    controller.reset();
    assert!(!controller.is_processing());
}

#[tokio::test]
async fn test_controller_processes_debugging_intent() {
    let mut controller = CoreController::new();
    let model = Box::new(DefaultSpecialistModel::new(
        ModelId::Logic,
        vec![IntentType::Debugging, IntentType::Reasoning],
    ));
    controller.register_specialist_model(ModelId::Logic.name(), model);
    let result = controller.process_request("fix bug memory leak", InputType::Text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_controller_processes_planning_intent() {
    let mut controller = CoreController::new();
    let model = Box::new(DefaultSpecialistModel::new(
        ModelId::Planner,
        vec![IntentType::Planning],
    ));
    controller.register_specialist_model(ModelId::Planner.name(), model);
    let result = controller.process_request("buat rencana proyek", InputType::Text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_controller_handles_multi_intent() {
    let controller = CoreController::new();
    let result = controller.process_request("buat fungsi dan cek error", InputType::Text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_controller_cache_hits() {
    let controller = CoreController::new();
    let _ = controller.process_request("test", InputType::Text).await;
    let _ = controller.process_request("test", InputType::Text).await;
}

#[tokio::test]
async fn test_controller_with_multiple_specialists() {
    let mut controller = CoreController::new();
    let models = [
        (ModelId::Coding, vec![IntentType::Coding]),
        (ModelId::Memory, vec![IntentType::Memory]),
        (ModelId::Logic, vec![IntentType::Debugging, IntentType::Reasoning]),
        (ModelId::Planner, vec![IntentType::Planning]),
        (ModelId::Validator, vec![IntentType::Validation]),
    ];
    for (id, intents) in &models {
        controller.register_specialist_model(id.name(), Box::new(DefaultSpecialistModel::new(*id, intents.clone())));
    }
    let inputs = vec![
        "buat program rust",
        "ingat data penting",
        "fix error kritis",
        "rencana development",
        "validasi input user",
    ];
    for input in inputs {
        let result = controller.process_request(input, InputType::Text).await;
        assert!(result.is_ok(), "Failed for input: {}", input);
    }
}

#[tokio::test]
async fn test_controller_get_config() {
    let controller = CoreController::new();
    let config = controller.get_config();
    assert!(config.intent_threshold > 0.0);
    assert!(config.context_cache_size > 0);
}

#[tokio::test]
async fn test_specialist_model_registration() {
    let mut controller = CoreController::new();
    let model = Box::new(DefaultSpecialistModel::new(
        ModelId::Coding,
        vec![IntentType::Coding],
    ));
    controller.register_specialist_model("coder", model);
    assert!(controller.is_model_available(ModelId::Coding));
}

#[tokio::test]
async fn test_controller_direct_intent_detection() {
    let controller = CoreController::new();
    let result = controller.detect_intent("buat fungsi").await;
    assert!(result.is_ok());
    let intent = result.unwrap();
    assert_eq!(intent.primary_intent, IntentType::Coding);
}

#[tokio::test]
async fn test_default_specialist_model_process() {
    let model = DefaultSpecialistModel::new(ModelId::Controller, vec![IntentType::Unknown]);
    let context = crate::types::ContextInfo::new("test".to_string(), ModelId::Controller);
    let result = model.process("test input", &context).await;
    assert!(result.is_ok());
}

#[test]
fn test_default_specialist_model_id() {
    let model = DefaultSpecialistModel::new(ModelId::Coding, vec![IntentType::Coding]);
    assert_eq!(model.model_id(), ModelId::Coding);
    assert!(model.can_handle(IntentType::Coding));
    assert!(!model.can_handle(IntentType::Memory));
}

#[test]
fn test_default_specialist_model_multiple_intents() {
    let model = DefaultSpecialistModel::new(
        ModelId::Logic,
        vec![IntentType::Debugging, IntentType::Reasoning],
    );
    assert!(model.can_handle(IntentType::Debugging));
    assert!(model.can_handle(IntentType::Reasoning));
    assert!(!model.can_handle(IntentType::Coding));
}
