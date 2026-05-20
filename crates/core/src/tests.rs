use crate::controller::CoreController;
use crate::types::InputType;

#[tokio::test]
async fn test_core_controller_creation() {
    let controller = CoreController::new();
    let result = controller.process_request("hello", InputType::Text).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_context_count() {
    let controller = CoreController::new();
    let _ = controller.process_request("test", InputType::Text).await;
    assert!(controller.get_context_count() <= 10);
}
