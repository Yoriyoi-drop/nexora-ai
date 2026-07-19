use dashmap::DashSet;
use std::sync::Arc;
use uuid::Uuid;

pub type SubscriberId = Uuid;
pub type PublisherId = Uuid;

#[derive(Debug, Clone)]
pub struct TopicConfig {
    pub name: String,
    pub max_subscribers: usize,
    pub persistent: bool,
    pub replay_limit: usize,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            max_subscribers: 1024,
            persistent: false,
            replay_limit: 100,
        }
    }
}

pub struct Topic {
    pub config: TopicConfig,
    subscribers: Arc<DashSet<(SubscriberId, String)>>,
    #[allow(dead_code)]
    created_at: chrono::DateTime<chrono::Utc>,
}

impl Topic {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: TopicConfig {
                name: name.into(),
                ..Default::default()
            },
            subscribers: Arc::new(DashSet::new()),
            created_at: chrono::Utc::now(),
        }
    }

    pub fn with_config(config: TopicConfig) -> Self {
        Self {
            subscribers: Arc::new(DashSet::new()),
            config,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn subscribe(&self, id: SubscriberId, name: String) -> bool {
        if self.subscribers.len() >= self.config.max_subscribers {
            return false;
        }
        self.subscribers.insert((id, name))
    }

    pub fn unsubscribe(&self, id: &SubscriberId) -> bool {
        self.subscribers.retain(|entry| &entry.0 != id);
        true
    }

    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    pub fn subscriber_ids(&self) -> Vec<SubscriberId> {
        self.subscribers
            .iter()
            .map(|entry| entry.0)
            .collect()
    }
}
