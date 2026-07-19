use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, warn};

use crate::error::Result;
use crate::event::{topics, Event, EventPriority, TopicName};
use crate::message_queue::MessageQueue;
use crate::publisher::{BasicPublisher, Publisher, PublisherId};
use crate::subscriber::{Subscriber, SubscriberId};
use crate::topic::{Topic, TopicConfig};

pub struct EventBus {
    topics: Arc<DashMap<String, Arc<RwLock<Topic>>>>,
    subscribers: Arc<DashMap<SubscriberId, Box<dyn Subscriber>>>,
    queues: Arc<DashMap<SubscriberId, MessageQueue>>,
    publishers: Arc<DashMap<PublisherId, Box<dyn Publisher>>>,
    broadcast_tx: broadcast::Sender<Event>,
    shutdown: Arc<tokio::sync::Notify>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(16384);
        let bus = Self {
            topics: Arc::new(DashMap::new()),
            subscribers: Arc::new(DashMap::new()),
            queues: Arc::new(DashMap::new()),
            publishers: Arc::new(DashMap::new()),
            broadcast_tx: tx,
            shutdown: Arc::new(tokio::sync::Notify::new()),
        };
        bus.init_system_topics();
        bus
    }

    pub fn with_capacity(cap: usize) -> Self {
        let (tx, _) = broadcast::channel(cap);
        let bus = Self {
            topics: Arc::new(DashMap::new()),
            subscribers: Arc::new(DashMap::new()),
            queues: Arc::new(DashMap::new()),
            publishers: Arc::new(DashMap::new()),
            broadcast_tx: tx,
            shutdown: Arc::new(tokio::sync::Notify::new()),
        };
        bus.init_system_topics();
        bus
    }

    fn init_system_topics(&self) {
        let system_topics = [
            topics::SCHEDULER_TASK_SUBMITTED,
            topics::SCHEDULER_TASK_COMPLETED,
            topics::SCHEDULER_TASK_FAILED,
            topics::GPU_KERNEL_QUEUED,
            topics::GPU_KERNEL_COMPLETED,
            topics::MEMORY_POOL_LOW,
            topics::MEMORY_POOL_CRITICAL,
            topics::AGENT_SPAWNED,
            topics::AGENT_STOPPED,
            topics::AGENT_SCALED,
            topics::CACHE_HIT,
            topics::CACHE_MISS,
            topics::COST_OPTIMIZER_ROUTE,
            topics::SYSTEM_STARTUP,
            topics::SYSTEM_SHUTDOWN,
            topics::SYSTEM_ERROR,
            topics::OBSERVABILITY_METRIC,
        ];
        for t in &system_topics {
            self.topics
                .insert(t.to_string(), Arc::new(RwLock::new(Topic::new(*t))));
        }
    }

    pub fn create_topic(&self, name: impl Into<String>) {
        let name: String = name.into();
        self.topics
            .entry(name.clone())
            .or_insert_with(|| Arc::new(RwLock::new(Topic::new(name))));
    }

    pub fn create_topic_with_config(&self, config: TopicConfig) {
        let name = config.name.clone();
        self.topics
            .entry(name)
            .or_insert_with(|| Arc::new(RwLock::new(Topic::with_config(config))));
    }

    pub fn register_publisher(&self, publisher: Box<dyn Publisher>) {
        let id = publisher.id();
        self.publishers.insert(id, publisher);
    }

    pub fn create_publisher(&self, name: impl Into<String>) -> PublisherId {
        let publisher = Box::new(BasicPublisher::new(name));
        let id = publisher.id();
        self.publishers.insert(id, publisher);
        id
    }

    pub fn subscribe(&self, subscriber: Box<dyn Subscriber>) -> Result<SubscriberId> {
        let id = subscriber.id();
        let topics = subscriber.topics().clone();
        let queue = MessageQueue::bounded(4096, 100);
        self.queues.insert(id, queue);
        for topic_name in &topics {
            let topic = self
                .topics
                .get(topic_name)
                .map(|t| t.value().clone())
                .unwrap_or_else(|| {
                    let t = Arc::new(RwLock::new(Topic::new(topic_name.clone())));
                    self.topics.insert(topic_name.clone(), t.clone());
                    t
                });
            topic.write().subscribe(id, subscriber.name().to_string());
        }
        self.subscribers.insert(id, subscriber);
        Ok(id)
    }

    pub fn subscribe_to(
        &self,
        subscriber: Box<dyn Subscriber>,
        topic: impl Into<String>,
    ) -> Result<SubscriberId> {
        let topic_name: String = topic.into();
        let id = subscriber.id();
        let queue = MessageQueue::bounded(4096, 100);
        self.queues.insert(id, queue);
        let t = self
            .topics
            .entry(topic_name.clone())
            .or_insert_with(|| Arc::new(RwLock::new(Topic::new(topic_name))));
        t.write().subscribe(id, subscriber.name().to_string());
        self.subscribers.insert(id, subscriber);
        Ok(id)
    }

    pub fn unsubscribe(&self, id: &SubscriberId) {
        self.subscribers.remove(id);
        self.queues.remove(id);
        for topic in self.topics.iter() {
            topic.write().unsubscribe(id);
        }
    }

    pub async fn publish(&self, event: Event) -> Result<()> {
        if event.is_expired() {
            return Ok(());
        }
        let topic_name = event.topic.clone();
        let _ = self.broadcast_tx.send(event.clone());
        if let Some(topic) = self.topics.get(&topic_name) {
            let subscriber_ids = topic.read().subscriber_ids();
            for sid in subscriber_ids {
                if let Some(queue) = self.queues.get(&sid) {
                    if let Err(_event) = queue.push(event.clone()) {
                        warn!("subscriber queue full, dropping event: sid={}", sid);
                    }
                }
            }
        }
        debug!("published event: topic={}, id={}", topic_name, event.id);
        Ok(())
    }

    pub async fn publish_raw(
        &self,
        publisher_id: &PublisherId,
        topic: TopicName,
        payload: serde_json::Value,
    ) -> Result<Event> {
        let publisher = self
            .publishers
            .get(publisher_id)
            .ok_or_else(|| crate::error::BusError::PublisherNotFound(publisher_id.to_string()))?;
        let event = publisher.publish(topic, payload).await?;
        self.publish(event.clone()).await?;
        Ok(event)
    }

    pub fn subscribe_to_broadcast(&self) -> broadcast::Receiver<Event> {
        self.broadcast_tx.subscribe()
    }

    pub fn next_event(&self, sid: &SubscriberId) -> Option<Event> {
        self.queues.get(sid).and_then(|q| q.pop())
    }

    pub async fn next_event_blocking(&self, sid: &SubscriberId) -> Option<Event> {
        loop {
            if let Some(event) = self.next_event(sid) {
                return Some(event);
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }
    }

    pub fn subscriber_queue_len(&self, sid: &SubscriberId) -> Option<usize> {
        self.queues.get(sid).map(|q| q.len())
    }

    pub fn topic_subscriber_count(&self, topic: &str) -> usize {
        self.topics
            .get(topic)
            .map(|t| t.read().subscriber_count())
            .unwrap_or(0)
    }

    pub fn topic_count(&self) -> usize {
        self.topics.len()
    }

    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    pub fn publisher_count(&self) -> usize {
        self.publishers.len()
    }

    pub fn emit_system_event(&self, topic: &str, message: &str) {
        let event = Event::new(topic.to_string(), serde_json::json!({"message": message}))
            .with_source("system")
            .with_priority(EventPriority::Normal);
        let _ = self.publish_blocking(event);
    }

    fn publish_blocking(&self, event: Event) -> Result<()> {
        if event.is_expired() {
            return Ok(());
        }
        let topic_name = event.topic.clone();
        let _ = self.broadcast_tx.send(event.clone());
        if let Some(topic) = self.topics.get(&topic_name) {
            let subscriber_ids = topic.read().subscriber_ids();
            for sid in subscriber_ids {
                if let Some(queue) = self.queues.get(&sid) {
                    if let Err(_event) = queue.push(event.clone()) {
                        warn!("subscriber queue full, dropping event: sid={}", sid);
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    pub fn stats(&self) -> BusStats {
        BusStats {
            topics: self.topics.len(),
            subscribers: self.subscribers.len(),
            publishers: self.publishers.len(),
            total_queue_depth: self.queues.iter().map(|e| e.len()).sum(),
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BusStats {
    pub topics: usize,
    pub subscribers: usize,
    pub publishers: usize,
    pub total_queue_depth: usize,
}


