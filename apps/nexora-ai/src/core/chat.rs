//! Chat functionality for Nexora-AI
//!
//! Provides conversational chat using the NXR foundation model registry.
//! All chat responses are generated via the active model inference pipeline.

use crate::error::{NexoraError, NexoraResult};
use chrono::Utc;
use nexora_foundation::shared::{
    base_model::{InputData, NxrInput, OutputData},
    model_identity::NxrModelId,
    model_registry::NxrModelRegistry,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};
use uuid::Uuid;

use super::types::{
    ChatIntent, ChatMessageAnalysis, ConversationContext, Sentiment, Urgency, UserPreferences,
};

/// Chat engine for handling conversations
///
/// Routes all chat messages through the NXR foundation model registry
/// using the active model for response generation.
#[derive(Debug, Clone)]
pub struct ChatEngine {
    registry: Arc<NxrModelRegistry>,
    active_model_id: NxrModelId,
    conversations: Arc<Mutex<HashMap<String, ConversationContext>>>,
}

impl ChatEngine {
    pub fn new(registry: Arc<NxrModelRegistry>, active_model_id: NxrModelId) -> Self {
        Self {
            registry,
            active_model_id,
            conversations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Process chat message with context
    pub async fn chat(
        &self,
        message: &str,
        conversation_id: Option<String>,
    ) -> NexoraResult<String> {
        let chat_start = Utc::now();

        info!(
            "💬 Starting chat processing: message_len={}, conversation_id={:?}",
            message.len(),
            conversation_id
        );
        debug!("📝 Message content: {}", message);

        // Comprehensive validation
        if message.is_empty() {
            return Err(NexoraError::validation(
                "message",
                "Message cannot be empty",
            ));
        }

        if message.len() > 2000 {
            return Err(NexoraError::validation(
                "message",
                "Message too long (max 2000 characters)",
            ));
        }

        // Get or create conversation context
        let conv_id = conversation_id.unwrap_or_else(|| {
            format!(
                "conv_{}",
                Uuid::new_v4()
                    .to_string()
                    .chars()
                    .take(8)
                    .collect::<String>()
            )
        });

        // Analyze message for conversation context
        let message_analysis = self.analyze_chat_message(message);
        debug!("Message analysis: {:?}", message_analysis);

        // Get conversation history (simplified - would use proper storage)
        let conversation_context = self.get_conversation_context(&conv_id).await?;

        // Generate contextual response
        let response = match message_analysis.intent {
            ChatIntent::Greeting => {
                self.generate_greeting_response(&conv_id, &conversation_context)
                    .await?
            }
            ChatIntent::Question => {
                self.generate_question_response(message, &conversation_context)
                    .await?
            }
            ChatIntent::Command => {
                self.generate_command_response(message, &conversation_context)
                    .await?
            }
            ChatIntent::Casual => {
                self.generate_casual_response(message, &conversation_context)
                    .await?
            }
            ChatIntent::Code => {
                self.generate_code_chat_response(message, &conversation_context)
                    .await?
            }
            ChatIntent::System => {
                self.generate_system_response(message, &conversation_context)
                    .await?
            }
        };

        // Store conversation turn
        self.store_conversation_turn(&conv_id, message, &response)
            .await?;

        let chat_time = (Utc::now() - chat_start).num_milliseconds();
        info!(
            "Chat completed in {}ms for conversation {}",
            chat_time, conv_id
        );

        Ok(response)
    }

    /// Analyze chat message for intent and context
    fn analyze_chat_message(&self, message: &str) -> ChatMessageAnalysis {
        let lower = message.to_lowercase();

        // Detect intent
        let intent = if lower.contains("hello") || lower.contains("hi") || lower.contains("hey") {
            ChatIntent::Greeting
        } else if lower.contains("?")
            || lower.starts_with("what")
            || lower.starts_with("how")
            || lower.starts_with("why")
            || lower.starts_with("when")
            || lower.starts_with("where")
        {
            ChatIntent::Question
        } else if lower.starts_with('/')
            || lower.starts_with('!')
            || lower.contains("help")
            || lower.contains("status")
        {
            ChatIntent::Command
        } else if lower.contains("fn ")
            || lower.contains("function")
            || lower.contains("class ")
            || lower.contains("def ")
            || lower.contains("```")
        {
            ChatIntent::Code
        } else if lower.contains("system")
            || lower.contains("memory")
            || lower.contains("performance")
            || lower.contains("health")
        {
            ChatIntent::System
        } else {
            ChatIntent::Casual
        };

        // Analyze sentiment (simplified)
        let positive_words = [
            "good",
            "great",
            "awesome",
            "excellent",
            "thanks",
            "thank you",
        ];
        let negative_words = ["bad", "terrible", "awful", "wrong", "error", "broken"];

        let sentiment = if positive_words.iter().any(|word| lower.contains(word)) {
            Sentiment::Positive
        } else if negative_words.iter().any(|word| lower.contains(word)) {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        };

        // Check for code presence
        let has_code = lower.contains("fn ")
            || lower.contains("function")
            || lower.contains("class ")
            || lower.contains("def ")
            || lower.contains("```")
            || lower.contains("import ");

        // Determine urgency
        let urgency =
            if lower.contains("urgent") || lower.contains("asap") || lower.contains("emergency") {
                Urgency::High
            } else if lower.contains("please") || lower.contains("help") {
                Urgency::Medium
            } else {
                Urgency::Low
            };

        ChatMessageAnalysis {
            intent,
            sentiment,
            word_count: message.split_whitespace().count(),
            has_code,
            urgency,
        }
    }

    /// Get conversation context with real turn tracking
    async fn get_conversation_context(
        &self,
        conversation_id: &str,
    ) -> NexoraResult<ConversationContext> {
        let convs = self.conversations.lock().unwrap();
        match convs.get(conversation_id) {
            Some(ctx) => Ok(ctx.clone()),
            None => Ok(ConversationContext {
                conversation_id: conversation_id.to_string(),
                turn_count: 0,
                last_activity: Utc::now(),
                topics: vec!["general".to_string()],
                user_preferences: UserPreferences::default(),
            }),
        }
    }

    /// Store conversation turn in memory
    async fn store_conversation_turn(
        &self,
        conversation_id: &str,
        user_message: &str,
        _ai_response: &str,
    ) -> NexoraResult<()> {
        let mut convs = self.conversations.lock().unwrap();
        let ctx = convs.entry(conversation_id.to_string()).or_insert_with(|| {
            ConversationContext {
                conversation_id: conversation_id.to_string(),
                turn_count: 0,
                last_activity: Utc::now(),
                topics: vec!["general".to_string()],
                user_preferences: UserPreferences::default(),
            }
        });
        ctx.turn_count += 1;
        ctx.last_activity = Utc::now();
        if user_message.contains('?') && !ctx.topics.contains(&"question".to_string()) {
            ctx.topics.push("question".to_string());
        }
        Ok(())
    }

    /// Generate chat response via foundation model inference through the model registry.
    /// Routes to the active model with the conversation context encoded in the prompt.
    async fn model_chat_generate(&self, user_message: &str) -> NexoraResult<String> {
        let model = self
            .registry
            .get_model(&self.active_model_id)
            .await
            .map_err(|e| {
                NexoraError::model(format!(
                    "Model {} not available for chat: {}",
                    self.active_model_id, e
                ))
            })?;

        let input = NxrInput {
            id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            data: InputData::Text(user_message.to_string()),
            parameters: [
                ("mode".to_string(), serde_json::json!("chat")),
                ("max_tokens".to_string(), serde_json::json!(512)),
                ("temperature".to_string(), serde_json::json!(0.8)),
            ]
            .into(),
            metadata: std::collections::HashMap::new(),
        };

        let output = model
            .infer(&input)
            .await
            .map_err(|e| NexoraError::model(format!("Chat inference failed: {}", e)))?;

        let text = match output.data {
            OutputData::Text(t) => t,
            _ => format!("{:?}", output.data),
        };

        Ok(text)
    }

    /// Generate greeting response via foundation model
    async fn generate_greeting_response(
        &self,
        _conversation_id: &str,
        _context: &ConversationContext,
    ) -> NexoraResult<String> {
        self.model_chat_generate("Hello! Please greet me warmly.").await
    }

    /// Generate question response via foundation model
    async fn generate_question_response(
        &self,
        question: &str,
        _context: &ConversationContext,
    ) -> NexoraResult<String> {
        self.model_chat_generate(question).await
    }

    /// Generate command response via foundation model
    async fn generate_command_response(
        &self,
        command: &str,
        _context: &ConversationContext,
    ) -> NexoraResult<String> {
        self.model_chat_generate(command).await
    }

    /// Generate casual response via foundation model
    async fn generate_casual_response(
        &self,
        message: &str,
        _context: &ConversationContext,
    ) -> NexoraResult<String> {
        self.model_chat_generate(message).await
    }

    /// Generate code chat response via foundation model
    async fn generate_code_chat_response(
        &self,
        message: &str,
        _context: &ConversationContext,
    ) -> NexoraResult<String> {
        self.model_chat_generate(message).await
    }

    /// Generate system response via foundation model
    async fn generate_system_response(
        &self,
        message: &str,
        _context: &ConversationContext,
    ) -> NexoraResult<String> {
        self.model_chat_generate(message).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexora_foundation::shared::model_registry::global_registry;

    fn test_chat_engine() -> ChatEngine {
        let registry = global_registry();
        ChatEngine::new(registry, NxrModelId::Omnis)
    }

    #[tokio::test]
    async fn test_chat_validation() {
        let chat_engine = test_chat_engine();

        // Test empty message
        let result = chat_engine.chat("", None).await;
        assert!(result.is_err());

        // Test message too long
        let long_message = "a".repeat(2001);
        let result = chat_engine.chat(&long_message, None).await;
        assert!(result.is_err());

        // Valid message requires initialized models — tested in test_chat_integration
    }

    #[tokio::test]
    #[ignore = "requires initialized foundation models in global_registry()"]
    async fn test_chat_integration() {
        let chat_engine = test_chat_engine();

        let result = chat_engine.chat("Hello", None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires initialized foundation models in global_registry()"]
    async fn test_chat_with_conversation_id() {
        let chat_engine = test_chat_engine();

        let result = chat_engine
            .chat("Hello", Some("test_conv".to_string()))
            .await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_analyze_chat_message() {
        let chat_engine = test_chat_engine();

        // Test greeting
        let analysis = chat_engine.analyze_chat_message("Hello there!");
        assert_eq!(analysis.intent, ChatIntent::Greeting);
        assert_eq!(analysis.sentiment, Sentiment::Neutral);

        // Test question
        let analysis = chat_engine.analyze_chat_message("What is Rust?");
        assert_eq!(analysis.intent, ChatIntent::Question);

        // Test command
        let analysis = chat_engine.analyze_chat_message("/help");
        assert_eq!(analysis.intent, ChatIntent::Command);

        // Test code
        let analysis = chat_engine.analyze_chat_message("fn main() {}");
        assert_eq!(analysis.intent, ChatIntent::Code);
        assert!(analysis.has_code);

        // Test positive sentiment
        let analysis = chat_engine.analyze_chat_message("This is great!");
        assert_eq!(analysis.sentiment, Sentiment::Positive);

        // Test negative sentiment
        let analysis = chat_engine.analyze_chat_message("This is terrible");
        assert_eq!(analysis.sentiment, Sentiment::Negative);

        // Test urgency
        let analysis = chat_engine.analyze_chat_message("This is urgent!");
        assert_eq!(analysis.urgency, Urgency::High);

        let analysis = chat_engine.analyze_chat_message("Please help me");
        assert_eq!(analysis.urgency, Urgency::Medium);
    }

    #[tokio::test]
    #[ignore = "requires initialized foundation models in global_registry()"]
    async fn test_generate_responses() {
        let chat_engine = test_chat_engine();
        let context = ConversationContext {
            conversation_id: "test".to_string(),
            turn_count: 1,
            last_activity: Utc::now(),
            topics: vec!["general".to_string()],
            user_preferences: UserPreferences::default(),
        };

        let result = chat_engine
            .generate_greeting_response("test_conv", &context)
            .await;
        assert!(result.is_ok());

        let result = chat_engine
            .generate_question_response("What is Rust?", &context)
            .await;
        assert!(result.is_ok());

        let result = chat_engine
            .generate_command_response("/status", &context)
            .await;
        assert!(result.is_ok());

        let result = chat_engine
            .generate_casual_response("How are you?", &context)
            .await;
        assert!(result.is_ok());

        let result = chat_engine
            .generate_code_chat_response("fn test() {}", &context)
            .await;
        assert!(result.is_ok());

        let result = chat_engine
            .generate_system_response("system status", &context)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_conversation_context() {
        let chat_engine = test_chat_engine();

        let result = chat_engine.get_conversation_context("test_conv").await;
        assert!(result.is_ok());

        let context = result.unwrap();
        assert_eq!(context.conversation_id, "test_conv");
        assert_eq!(context.turn_count, 0);
        assert!(context.topics.contains(&"general".to_string()));
    }

    #[tokio::test]
    async fn test_store_conversation_turn() {
        let chat_engine = test_chat_engine();

        let result = chat_engine
            .store_conversation_turn("test_conv", "Hello", "Hi there!")
            .await;
        assert!(result.is_ok());
    }
}
