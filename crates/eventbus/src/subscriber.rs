use async_trait::async_trait;
use uuid::Uuid;

use crate::event::Event;
use crate::Result;

pub type SubscriberId = Uuid;

#[async_trait]
pub trait Subscriber: Send + Sync {
    fn id(&self) -> SubscriberId;
    fn name(&self) -> &str;

    async fn on_event(&self, event: &Event) -> Result<()>;

    fn topics(&self) -> Vec<String>;
}

pub struct CallbackSubscriber {
    id: SubscriberId,
    name: String,
    topics: Vec<String>,
    callback: Box<dyn Fn(&Event) -> Result<()> + Send + Sync>,
}

impl CallbackSubscriber {
    pub fn new(
        name: impl Into<String>,
        topics: Vec<String>,
        callback: Box<dyn Fn(&Event) -> Result<()> + Send + Sync>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            topics,
            callback,
        }
    }
}

#[async_trait]
impl Subscriber for CallbackSubscriber {
    fn id(&self) -> SubscriberId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn on_event(&self, event: &Event) -> Result<()> {
        (self.callback)(event)
    }

    fn topics(&self) -> Vec<String> {
        self.topics.clone()
    }
}
