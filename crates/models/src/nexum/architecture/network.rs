//! Communication network and message routing

/// Communication Network
#[derive(Debug, Clone)]
pub struct CommunicationNetwork {
    /// Network topology
    pub topology: NetworkTopology,
    /// Communication protocol
    pub protocol: CommunicationProtocol,
    /// Message router
    pub message_router: MessageRouter,
    /// Network monitor
    pub network_monitor: NetworkMonitor,
    /// Communication metrics
    pub metrics: CommunicationMetrics,
}

/// Network Topology
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    /// Star topology
    Star,
    /// Mesh topology
    Mesh,
    /// Ring topology
    Ring,
    /// Tree topology
    Tree,
    /// Hybrid topology
    Hybrid,
    /// Dynamic topology
    Dynamic,
}

/// Communication Protocol
#[derive(Debug, Clone)]
pub enum CommunicationProtocol {
    /// Synchronous communication
    Synchronous,
    /// Asynchronous communication
    Asynchronous,
    /// Event-driven communication
    EventDriven,
    /// Message queue communication
    MessageQueue,
    /// Publish-subscribe communication
    PubSub,
    /// Hybrid communication
    Hybrid { sync_weight: f32, async_weight: f32 },
}

/// Message Router
#[derive(Debug, Clone)]
pub struct MessageRouter {
    /// Routing algorithm
    pub routing_algorithm: RoutingAlgorithm,
    /// Message queue
    pub message_queue: MessageQueue,
    /// Routing history
    pub routing_history: Vec<MessageRoutingRecord>,
    /// Router metrics
    pub metrics: MessageRouterMetrics,
}

/// Routing Algorithm
#[derive(Debug, Clone)]
pub enum RoutingAlgorithm {
    /// Direct routing
    Direct,
    /// Broadcast routing
    Broadcast,
    /// Multicast routing
    Multicast,
    /// Adaptive routing
    Adaptive,
    /// Load-balanced routing
    LoadBalanced,
    /// Priority routing
    Priority,
}

/// Message Queue
#[derive(Debug, Clone)]
pub struct MessageQueue {
    /// Queue type
    pub queue_type: MessageQueueType,
    /// Queue capacity
    pub capacity: usize,
    /// Queue length
    pub length: usize,
    /// Queue metrics
    pub metrics: MessageQueueMetrics,
}

/// Message Queue Type
#[derive(Debug, Clone)]
pub enum MessageQueueType {
    /// FIFO queue
    FIFO,
    /// Priority queue
    Priority,
    /// Fair queue
    Fair,
    /// Deadline queue
    Deadline,
}

/// Message Queue Metrics
#[derive(Debug, Clone)]
pub struct MessageQueueMetrics {
    /// Queue utilization
    pub utilization: f32,
    /// Average wait time
    pub avg_wait_time_ms: f64,
    /// Throughput
    pub throughput: f32,
    /// Drop rate
    pub drop_rate: f32,
}

/// Message Routing Record
#[derive(Debug, Clone)]
pub struct MessageRoutingRecord {
    /// Routing ID
    pub id: String,
    /// Source agent
    pub source_agent: String,
    /// Destination agent
    pub destination_agent: String,
    /// Message type
    pub message_type: String,
    /// Routing time
    pub routing_time_ms: u64,
    /// Routing success
    pub success: bool,
}

/// Message Router Metrics
#[derive(Debug, Clone)]
pub struct MessageRouterMetrics {
    /// Routing accuracy
    pub routing_accuracy: f32,
    /// Average routing time
    pub avg_routing_time_ms: f64,
    /// Message throughput
    pub message_throughput: f32,
    /// Network efficiency
    pub network_efficiency: f32,
}

/// Network Monitor
#[derive(Debug, Clone)]
pub struct NetworkMonitor {
    /// Monitoring frequency
    pub monitoring_frequency_ms: u64,
    /// Network health
    pub network_health: NetworkHealth,
    /// Performance metrics
    pub performance_metrics: NetworkPerformanceMetrics,
    /// Monitor metrics
    pub metrics: NetworkMonitorMetrics,
}

/// Network Health
#[derive(Debug, Clone)]
pub struct NetworkHealth {
    /// Overall health score
    pub overall_score: f32,
    /// Connectivity status
    pub connectivity_status: ConnectivityStatus,
    /// Latency status
    pub latency_status: LatencyStatus,
    /// Throughput status
    pub throughput_status: ThroughputStatus,
}

/// Connectivity Status
#[derive(Debug, Clone)]
pub enum ConnectivityStatus {
    /// Excellent
    Excellent,
    /// Good
    Good,
    /// Fair
    Fair,
    /// Poor
    Poor,
    /// Failed
    Failed,
}

/// Latency Status
#[derive(Debug, Clone)]
pub enum LatencyStatus {
    /// Excellent
    Excellent,
    /// Good
    Good,
    /// Fair
    Fair,
    /// Poor
    Poor,
    /// Critical
    Critical,
}

/// Throughput Status
#[derive(Debug, Clone)]
pub enum ThroughputStatus {
    /// Excellent
    Excellent,
    /// Good
    Good,
    /// Fair
    Fair,
    /// Poor
    Poor,
    /// Insufficient
    Insufficient,
}

/// Network Performance Metrics
#[derive(Debug, Clone)]
pub struct NetworkPerformanceMetrics {
    /// Average latency
    pub avg_latency_ms: f64,
    /// Throughput
    pub throughput: f32,
    /// Packet loss rate
    pub packet_loss_rate: f32,
    /// Jitter
    pub jitter_ms: f64,
}

/// Network Monitor Metrics
#[derive(Debug, Clone)]
pub struct NetworkMonitorMetrics {
    /// Monitoring accuracy
    pub monitoring_accuracy: f32,
    /// Anomaly detection rate
    pub anomaly_detection_rate: f32,
    /// Response time
    pub response_time_ms: f64,
    /// Coverage
    pub coverage: f32,
}

/// Communication Metrics
#[derive(Debug, Clone)]
pub struct CommunicationMetrics {
    /// Communication efficiency
    pub communication_efficiency: f32,
    /// Message delivery rate
    pub message_delivery_rate: f32,
    /// Average message latency
    pub avg_message_latency_ms: f64,
    /// Network utilization
    pub network_utilization: f32,
}

/// Message
#[derive(Debug, Clone)]
pub struct Message {
    pub id: String,
    pub sender: String,
    pub message_type: String,
    pub content: String,
    pub priority: MessagePriority,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Message Priority
#[derive(Debug, Clone)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Message Routing
#[derive(Debug, Clone)]
pub struct MessageRouting {
    pub routing_result: RoutingResult,
    pub routing_time_ms: u64,
    pub routing_success: bool,
}

impl MessageRouting {
    pub fn new() -> Self {
        Self {
            routing_result: RoutingResult::new(),
            routing_time_ms: 0,
            routing_success: false,
        }
    }
}

/// Routing Result
#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub successful_recipients: Vec<String>,
    pub failed_recipients: Vec<String>,
}

impl RoutingResult {
    pub fn new() -> Self {
        Self {
            successful_recipients: Vec::new(),
            failed_recipients: Vec::new(),
        }
    }
}

use super::super::config as nexum_config;

impl From<nexum_config::CommunicationProtocol> for CommunicationProtocol {
    fn from(c: nexum_config::CommunicationProtocol) -> Self {
        match c {
            nexum_config::CommunicationProtocol::Synchronous => Self::Synchronous,
            nexum_config::CommunicationProtocol::Asynchronous => Self::Asynchronous,
            nexum_config::CommunicationProtocol::EventDriven => Self::EventDriven,
            nexum_config::CommunicationProtocol::MessageQueue => Self::MessageQueue,
            nexum_config::CommunicationProtocol::PubSub => Self::PubSub,
            nexum_config::CommunicationProtocol::Hybrid { sync_weight, async_weight } => Self::Hybrid { sync_weight, async_weight },
        }
    }
}
