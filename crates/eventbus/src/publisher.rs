use async_trait::async_trait;
use uuid::Uuid;

use crate::event::{Event, EventPriority, TopicName};
use crate::Result;

pub type PublisherId = Uuid;

#[async_trait]
pub trait Publisher: Send + Sync {
    fn id(&self) -> PublisherId;
    fn name(&self) -> &str;

    async fn publish(&self, topic: TopicName, payload: serde_json::Value) -> Result<Event>;

    async fn publish_with_priority(
        &self,
        topic: TopicName,
        payload: serde_json::Value,
        priority: EventPriority,
    ) -> Result<Event>;

    async fn publish_raw(&self, event: Event) -> Result<()>;
}

pub struct BasicPublisher {
    id: PublisherId,
    name: String,
}

impl BasicPublisher {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
        }
    }
}

#[async_trait]
impl Publisher for BasicPublisher {
    fn id(&self) -> PublisherId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn publish(&self, topic: TopicName, payload: serde_json::Value) -> Result<Event> {
        let event = Event::new(topic, payload).with_source(self.name.clone());
        Ok(event)
    }

    async fn publish_with_priority(
        &self,
        topic: TopicName,
        payload: serde_json::Value,
        priority: EventPriority,
    ) -> Result<Event> {
        let event = Event::new(topic, payload)
            .with_priority(priority)
            .with_source(self.name.clone());
        Ok(event)
    }

    async fn publish_raw(&self, _event: Event) -> Result<()> {
        Ok(())
    }
}
