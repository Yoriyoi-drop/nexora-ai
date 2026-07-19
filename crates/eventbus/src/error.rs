use thiserror::Error;

pub type Result<T> = std::result::Result<T, BusError>;

#[derive(Debug, Error)]
pub enum BusError {
    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    #[error("Subscriber not found: {0}")]
    SubscriberNotFound(String),

    #[error("Publisher not found: {0}")]
    PublisherNotFound(String),

    #[error("Queue full: capacity={capacity}")]
    QueueFull { capacity: usize },

    #[error("Message expired")]
    MessageExpired,

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}
