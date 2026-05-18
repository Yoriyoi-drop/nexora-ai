//! Communication Module
//! 
//! Agent-to-agent messaging system untuk multi-stage reasoning.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, broadcast};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::warn;
use tracing::debug;

use crate::{AgentError, Result, MessagePriority};

/// Inter-agent message dengan routing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterAgentMessage {
    /// Unique message ID
    pub message_id: Uuid,
    /// Source agent ID
    pub source_agent_id: Uuid,
    /// Destination agent ID (None untuk broadcast)
    pub destination_agent_id: Option<Uuid>,
    /// Message type
    pub message_type: String,
    /// Payload
    pub payload: serde_json::Value,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Priority
    pub priority: MessagePriority,
    /// Reply-to message ID (jika ini adalah reply)
    pub reply_to: Option<Uuid>,
    /// TTL (time to live) dalam hops
    pub ttl: Option<u32>,
}

/// Message routing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Direct ke destination
    Direct,
    /// Broadcast ke semua agent
    Broadcast,
    /// Topic-based routing
    Topic(String),
    /// Role-based routing
    Role(String),
    /// Load-balanced ke agents dengan role tertentu
    LoadBalanced(String),
}

/// Message delivery status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Pending delivery
    Pending,
    /// Successfully delivered
    Delivered,
    /// Delivery failed
    Failed(String),
    /// Message expired
    Expired,
    /// Message rejected
    Rejected,
}

/// Message tracking information
#[derive(Debug, Clone)]
pub struct MessageTracking {
    /// Message
    pub message: InterAgentMessage,
    /// Delivery status per destination
    pub delivery_status: HashMap<Uuid, DeliveryStatus>,
    /// Send attempts
    pub send_attempts: u32,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last attempt timestamp
    pub last_attempt_at: Option<DateTime<Utc>>,
}

/// Message bus untuk inter-agent communication
pub struct MessageBus {
    /// Channel untuk incoming messages
    _message_tx: mpsc::UnboundedSender<Arc<InterAgentMessage>>,
    /// Receiver untuk incoming messages
    _message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<Arc<InterAgentMessage>>>>>,
    /// Per-agent message channels (bounded, buffer=16)
    subscribers: Arc<RwLock<HashMap<Uuid, mpsc::Sender<Arc<InterAgentMessage>>>>>,
    /// Topic subscribers (topic -> vec of agent IDs)
    topic_subscribers: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
    /// Role-based subscribers (role -> vec of agent IDs)
    role_subscribers: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
    /// Message tracking
    message_tracking: Arc<RwLock<HashMap<Uuid, MessageTracking>>>,
    /// Broadcast channel untuk events
    event_tx: broadcast::Sender<MessageBusEvent>,
    /// Message queue untuk pending messages (VecDeque for O(1) front removal)
    message_queue: Arc<RwLock<VecDeque<Arc<InterAgentMessage>>>>,
}

/// Events dari message bus
#[derive(Debug, Clone)]
pub enum MessageBusEvent {
    /// Message received
    MessageReceived(InterAgentMessage),
    /// Message delivered
    MessageDelivered { message_id: Uuid, agent_id: Uuid },
    /// Message failed
    MessageFailed { message_id: Uuid, agent_id: Uuid, error: String },
    /// Agent subscribed
    AgentSubscribed { agent_id: Uuid },
    /// Agent unsubscribed
    AgentUnsubscribed { agent_id: Uuid },
}

impl MessageBus {
    /// Create new message bus
    pub fn new() -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (event_tx, _) = broadcast::channel(1000);
        
        Self {
            _message_tx: message_tx,
            _message_rx: Arc::new(RwLock::new(Some(message_rx))),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            topic_subscribers: Arc::new(RwLock::new(HashMap::new())),
            role_subscribers: Arc::new(RwLock::new(HashMap::new())),
            message_tracking: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            message_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
    
    /// Subscribe agent ke message bus
    pub async fn subscribe(&self, agent_id: Uuid) -> Result<mpsc::Receiver<Arc<InterAgentMessage>>> {
        debug!("Agent {} subscribing to message bus", agent_id);
        
        let (tx, rx) = mpsc::channel(16);
        
        // Register subscriber
        {
            let mut subscribers = self.subscribers.write().await;
            subscribers.insert(agent_id, tx);
        }
        
        // Emit event
        let event = MessageBusEvent::AgentSubscribed { agent_id };
        if self.event_tx.send(event).is_err() {
            warn!("Message bus event receiver dropped");
        }
        
        // Process any queued messages
        self.process_queued_messages(agent_id).await?;
        
        Ok(rx)
    }
    
    /// Unsubscribe agent dari message bus
    pub async fn unsubscribe(&self, agent_id: Uuid) -> Result<()> {
        debug!("Agent {} unsubscribing from message bus", agent_id);
        
        // Remove subscriber
        {
            let mut subscribers = self.subscribers.write().await;
            subscribers.remove(&agent_id);
        }
        
        // Remove from topic subscriptions
        {
            let mut topic_subscribers = self.topic_subscribers.write().await;
            for subscribers in topic_subscribers.values_mut() {
                subscribers.retain(|&id| id != agent_id);
            }
        }
        
        // Remove from role subscriptions
        {
            let mut role_subscribers = self.role_subscribers.write().await;
            for subscribers in role_subscribers.values_mut() {
                subscribers.retain(|&id| id != agent_id);
            }
        }
        
        // Emit event
        let event = MessageBusEvent::AgentUnsubscribed { agent_id };
        if self.event_tx.send(event).is_err() {
            warn!("Message bus event receiver dropped");
        }
        
        Ok(())
    }
    
    /// Subscribe agent ke topic
    pub async fn subscribe_to_topic(&self, agent_id: Uuid, topic: String) -> Result<()> {
        debug!("Agent {} subscribing to topic: {}", agent_id, topic);
        
        let mut topic_subscribers = self.topic_subscribers.write().await;
        topic_subscribers.entry(topic).or_insert_with(Vec::new).push(agent_id);
        
        Ok(())
    }
    
    /// Subscribe agent ke role
    pub async fn subscribe_to_role(&self, agent_id: Uuid, role: String) -> Result<()> {
        debug!("Agent {} subscribing to role: {}", agent_id, role);
        
        let mut role_subscribers = self.role_subscribers.write().await;
        role_subscribers.entry(role).or_insert_with(Vec::new).push(agent_id);
        
        Ok(())
    }
    
    /// Send message ke agent tertentu
    pub async fn send_message(&self, message: InterAgentMessage) -> Result<()> {
        let message = Arc::new(message);
        debug!("Sending message {} from {} to {:?}", 
               message.message_id, message.source_agent_id, message.destination_agent_id);
        
        // Track message
        self.track_message((*message).clone()).await?;
        
        // Route message based on destination
        match message.destination_agent_id {
            Some(dest_id) => {
                self.send_to_agent(dest_id, message).await?;
            }
            None => {
                // Broadcast atau topic-based
                self.route_message(message).await?;
            }
        }
        
        Ok(())
    }
    
    /// Send reply ke message
    pub async fn send_reply(&self, original_message_id: Uuid, reply: InterAgentMessage) -> Result<()> {
        debug!("Sending reply to message {}", original_message_id);
        
        // Get original message for routing
        let original_message = {
            let tracking = self.message_tracking.read().await;
            tracking.get(&original_message_id)
                .map(|t| t.message.clone())
        };
        
        if let Some(original) = original_message {
            // Set reply-to
            let mut reply = reply;
            reply.reply_to = Some(original_message_id);
            
            // Send to original sender
            reply.destination_agent_id = Some(original.source_agent_id);
            
            self.send_message(reply).await
        } else {
            Err(AgentError::CommunicationError(
                format!("Original message {} not found", original_message_id)
            ))
        }
    }
    
    /// Get event subscriber
    pub fn get_event_subscriber(&self) -> broadcast::Receiver<MessageBusEvent> {
        self.event_tx.subscribe()
    }
    
    /// Get message tracking info
    pub async fn get_message_tracking(&self, message_id: Uuid) -> Result<Option<MessageTracking>> {
        let tracking = self.message_tracking.read().await;
        Ok(tracking.get(&message_id).cloned())
    }
    
    /// Get pending messages count
    #[allow(clippy::let_and_return)]
    pub async fn get_pending_count(&self) -> usize {
        let queue = self.message_queue.read().await;
        let len = queue.len();
        len
    }
    
    /// Cleanup old messages
    pub async fn cleanup_old_messages(&self, max_age_hours: u64) -> Result<usize> {
        let now = Utc::now();
        let mut removed_count = 0;
        
        {
            let mut tracking = self.message_tracking.write().await;
            tracking.retain(|message_id, tracking| {
                let age_hours = (now - tracking.created_at).num_hours() as u64;
                let should_keep = age_hours <= max_age_hours;
                
                if !should_keep {
                    debug!("Cleaning up old message tracking: {}", message_id);
                    removed_count += 1;
                }
                
                should_keep
            });
        }
        
        Ok(removed_count)
    }
    
    /// Send message ke specific agent
    async fn send_to_agent(&self, agent_id: Uuid, message: Arc<InterAgentMessage>) -> Result<()> {
        // Clone sender under lock, drop lock, then send (avoids holding lock across await)
        let sender = {
            let subscribers = self.subscribers.read().await;
            subscribers.get(&agent_id).cloned()
        };
        
        if let Some(sender) = sender {
            if let Err(_) = sender.send(message.clone()).await {
                // Agent tidak bisa menerima message, queue dulu
                self.queue_message(message.clone()).await?;
                
                // Update tracking
                self.update_delivery_status(message.message_id, agent_id, 
                                           DeliveryStatus::Failed("Agent not receiving".to_string())).await?;
            } else {
                // Update tracking
                self.update_delivery_status(message.message_id, agent_id, 
                                           DeliveryStatus::Delivered).await?;
                
                // Emit event
                let event = MessageBusEvent::MessageDelivered { 
                    message_id: message.message_id, 
                    agent_id 
                };
                if self.event_tx.send(event).is_err() {
            warn!("Message bus event receiver dropped");
        }
            }
        } else {
            // Agent tidak subscribed, queue dulu
            self.queue_message(message).await?;
        }
        
        Ok(())
    }
    
    /// Route message berdasarkan strategy (uses Arc sharing — cheap clone)
    async fn route_message(&self, message: Arc<InterAgentMessage>) -> Result<()> {
        // Check untuk topic-based routing
        if let Some(topic) = message.metadata.get("topic") {
            if let Some(topic_str) = topic.as_str() {
                let topic_subscribers = self.topic_subscribers.read().await;
                if let Some(subscribers) = topic_subscribers.get(topic_str) {
                    for &agent_id in subscribers {
                        if Some(agent_id) != message.destination_agent_id {
                            self.send_to_agent(agent_id, message.clone()).await?;
                        }
                    }
                    return Ok(());
                }
            }
        }
        
        // Check untuk role-based routing
        if let Some(role) = message.metadata.get("role") {
            if let Some(role_str) = role.as_str() {
                let role_subscribers = self.role_subscribers.read().await;
                if let Some(subscribers) = role_subscribers.get(role_str) {
                    for &agent_id in subscribers {
                        if Some(agent_id) != message.destination_agent_id {
                            self.send_to_agent(agent_id, message.clone()).await?;
                        }
                    }
                    return Ok(());
                }
            }
        }
        
        // Default: broadcast ke semua subscribers
        let subscribers = self.subscribers.read().await;
        for (&agent_id, _) in subscribers.iter() {
            if Some(agent_id) != message.destination_agent_id && 
               agent_id != message.source_agent_id {
                self.send_to_agent(agent_id, message.clone()).await?;
            }
        }
        
        Ok(())
    }
    
    /// Queue message untuk nanti
    async fn queue_message(&self, message: Arc<InterAgentMessage>) -> Result<()> {
        let mut queue = self.message_queue.write().await;
        queue.push_back(message);
        Ok(())
    }
    
    /// Process queued messages untuk agent (VecDeque front-pop for O(1) removal)
    async fn process_queued_messages(&self, agent_id: Uuid) -> Result<()> {
        let mut queue = self.message_queue.write().await;
        let mut processed = 0usize;
        let mut i = 0;
        
        while i < queue.len() {
            let should_deliver = {
                let msg = &queue[i];
                msg.destination_agent_id == Some(agent_id) || 
                msg.destination_agent_id.is_none()
            };
            
            if should_deliver {
                if let Some(message) = queue.remove(i) {
                    self.send_to_agent(agent_id, message).await?;
                    processed += 1;
                }
            } else {
                i += 1;
            }
        }
        
        Ok(())
    }
    
    /// Track message
    async fn track_message(&self, message: InterAgentMessage) -> Result<()> {
        let tracking = MessageTracking {
            delivery_status: HashMap::new(),
            send_attempts: 0,
            created_at: Utc::now(),
            last_attempt_at: None,
            message,
        };
        
        let mut message_tracking = self.message_tracking.write().await;
        message_tracking.insert(tracking.message.message_id, tracking);
        
        Ok(())
    }
    
    /// Update delivery status
    async fn update_delivery_status(&self, message_id: Uuid, agent_id: Uuid, 
                                  status: DeliveryStatus) -> Result<()> {
        let mut message_tracking = self.message_tracking.write().await;
        if let Some(tracking) = message_tracking.get_mut(&message_id) {
            tracking.delivery_status.insert(agent_id, status);
            tracking.last_attempt_at = Some(Utc::now());
        }
        
        Ok(())
    }
}

impl InterAgentMessage {
    /// Create new message
    pub fn new(
        source_agent_id: Uuid,
        message_type: String,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            message_id: Uuid::new_v4(),
            source_agent_id,
            destination_agent_id: None,
            message_type,
            payload,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            priority: MessagePriority::Normal,
            reply_to: None,
            ttl: Some(10), // Default TTL 10 hops
        }
    }
    
    /// Set destination
    pub fn to(mut self, destination_agent_id: Uuid) -> Self {
        self.destination_agent_id = Some(destination_agent_id);
        self
    }
    
    /// Set priority
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set TTL
    pub fn with_ttl(mut self, ttl: u32) -> Self {
        self.ttl = Some(ttl);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Set as reply to
    pub fn reply_to(mut self, original_message_id: Uuid) -> Self {
        self.reply_to = Some(original_message_id);
        self
    }
}
