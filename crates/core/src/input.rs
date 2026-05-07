//! Input receiver dan validation untuk Nexora Core

use crate::types::{InputData, InputType};
use crate::error::{CoreError, CoreResult};
use tracing::{debug, info, warn};

/// Input receiver untuk menerima dan memvalidasi input dari user
pub struct InputReceiver {
    validation_enabled: bool,
}

impl InputReceiver {
    pub fn new() -> Self {
        Self {
            validation_enabled: true,
        }
    }
    
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validation_enabled = enabled;
        self
    }
    
    /// Receive dan validate input
    pub async fn receive_input(&self, input: &str, input_type: InputType) -> CoreResult<InputData> {
        debug!("Receiving input: type={:?}, length={}", input_type, input.len());
        
        // Create input data
        let mut input_data = InputData::new(input.to_string(), input_type.clone());
        
        // Validate if enabled
        if self.validation_enabled {
            if !input_data.validate() {
                warn!("Input validation failed for: {}", input);
                return Err(CoreError::InputValidation(format!(
                    "Input validation failed for type: {:?}", input_type
                )));
            }
        }
        
        info!("Input received and validated successfully");
        Ok(input_data)
    }
    
    /// Batch receive multiple inputs
    pub async fn receive_batch(&self, inputs: &[(String, InputType)]) -> CoreResult<Vec<InputData>> {
        let mut results = Vec::with_capacity(inputs.len());
        
        for (input, input_type) in inputs {
            let input_data = self.receive_input(input, input_type.clone()).await?;
            results.push(input_data);
        }
        
        Ok(results)
    }
}

impl Default for InputReceiver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_text_input_validation() {
        let receiver = InputReceiver::new();
        
        // Valid text input
        let result = receiver.receive_input("Hello world", InputType::Text).await;
        assert!(result.is_ok());
        
        // Invalid text input (too long)
        let long_input = "a".repeat(10001);
        let result = receiver.receive_input(&long_input, InputType::Text).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_command_input_validation() {
        let receiver = InputReceiver::new();
        
        // Valid command
        let result = receiver.receive_input("buat program", InputType::Command).await;
        assert!(result.is_ok());
        
        // Invalid command
        let result = receiver.receive_input("random text", InputType::Command).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_query_input_validation() {
        let receiver = InputReceiver::new();
        
        // Valid query with question mark
        let result = receiver.receive_input("Apa ini?", InputType::Query).await;
        assert!(result.is_ok());
        
        // Valid query with "apa"
        let result = receiver.receive_input("apa yang terjadi", InputType::Query).await;
        assert!(result.is_ok());
        
        // Invalid query (too short and no question words)
        let result = receiver.receive_input("test", InputType::Query).await;
        assert!(result.is_err());
    }
}
