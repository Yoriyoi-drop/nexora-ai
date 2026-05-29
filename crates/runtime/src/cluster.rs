use chrono::{DateTime, Utc};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum NodeState {
    Alive,
    Suspect,
    Dead,
    Leaving,
}

impl NodeState {
    pub fn is_available(&self) -> bool {
        matches!(self, NodeState::Alive)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct NodeLoad {
    pub active_requests: u32,
    pub queue_depth: u32,
    pub gpu_util_pct: f32,
    pub memory_used_mb: u64,
    pub tokens_per_sec: f64,
    pub kv_fragmentation_pct: f32,
}

impl Default for NodeLoad {
    fn default() -> Self {
        Self {
            active_requests: 0,
            queue_depth: 0,
            gpu_util_pct: 0.0,
            memory_used_mb: 0,
            tokens_per_sec: 0.0,
            kv_fragmentation_pct: 0.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeCapabilities {
    pub models: Vec<String>,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub supports_streaming: bool,
    pub supports_cuda: bool,
    pub supports_paged_cache: bool,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            max_batch_size: 32,
            max_seq_len: 2048,
            supports_streaming: true,
            supports_cuda: false,
            supports_paged_cache: true,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub node_id: Uuid,
    pub address: String,
    pub state: NodeState,
    pub load: NodeLoad,
    pub capabilities: NodeCapabilities,
    pub started_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub version: String,
}

impl NodeInfo {
    pub fn new(node_id: Uuid, address: String) -> Self {
        Self {
            node_id,
            address,
            state: NodeState::Alive,
            load: NodeLoad::default(),
            capabilities: NodeCapabilities::default(),
            started_at: Utc::now(),
            last_heartbeat: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn is_stale(&self, timeout: Duration) -> bool {
        let elapsed = Utc::now() - self.last_heartbeat;
        elapsed.num_milliseconds() as u64 > timeout.as_millis() as u64
    }

    pub fn load_score(&self) -> f64 {
        let active = self.load.active_requests as f64;
        let queue = self.load.queue_depth as f64;
        let gpu = (100.0 - self.load.gpu_util_pct) as f64 / 100.0;
        let mem = self.load.memory_used_mb as f64 / 1024.0;
        let tok = (self.load.tokens_per_sec / 1000.0).min(10.0);
        active * 0.3 + queue * 0.3 - gpu * 0.2 + mem * 0.1 - tok * 0.1
    }
}

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub node_id: Uuid,
    pub listen_address: String,
    pub gossip_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub node_eviction_timeout_ms: u64,
    pub seed_nodes: Vec<String>,
    pub max_concurrent_remote: usize,
    pub request_timeout_ms: u64,
    pub shared_secret: Option<String>,
    pub tls_enabled: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            listen_address: "127.0.0.1:0".to_string(),
            gossip_interval_ms: 1000,
            heartbeat_timeout_ms: 5000,
            node_eviction_timeout_ms: 30000,
            seed_nodes: Vec::new(),
            max_concurrent_remote: 64,
            request_timeout_ms: 30000,
            shared_secret: None,
            tls_enabled: false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GossipState {
    pub node_id: Uuid,
    pub load: NodeLoad,
    pub incarnation: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GossipMessage {
    pub sender_id: Uuid,
    pub sender_addr: String,
    pub state: GossipState,
    pub known_nodes: Vec<(Uuid, NodeState, u64)>,
}

pub struct NodeRegistry {
    config: ClusterConfig,
    local_info: RwLock<NodeInfo>,
    nodes: RwLock<HashMap<Uuid, NodeInfo>>,
    incarnation: RwLock<u64>,
    dynamic_load_fn: RwLock<Option<Arc<dyn Fn() -> NodeLoad + Send + Sync>>>,
}

impl NodeRegistry {
    pub fn new(config: ClusterConfig) -> Self {
        let local = NodeInfo::new(config.node_id, config.listen_address.clone());
        Self {
            local_info: RwLock::new(local),
            nodes: RwLock::new(HashMap::new()),
            config,
            incarnation: RwLock::new(0),
            dynamic_load_fn: RwLock::new(None),
        }
    }

    pub fn set_dynamic_load_fn(&self, f: Option<Arc<dyn Fn() -> NodeLoad + Send + Sync>>) {
        if let Ok(mut guard) = self.dynamic_load_fn.try_write() {
            *guard = f;
        }
    }

    pub async fn local_info(&self) -> NodeInfo {
        self.local_info.read().await.clone()
    }

    pub async fn update_local_load(&self, load: NodeLoad) {
        let mut info = self.local_info.write().await;
        info.load = load;
        info.last_heartbeat = Utc::now();
    }

    pub async fn refresh_dynamic_load(&self) {
        if let Ok(guard) = self.dynamic_load_fn.try_read() {
            if let Some(ref f) = *guard {
                let load = f();
                self.update_local_load(load).await;
            }
        }
    }

    pub async fn register_node(&self, info: NodeInfo) {
        let mut nodes = self.nodes.write().await;
        nodes.insert(info.node_id, info);
    }

    pub async fn update_node_state(&self, node_id: Uuid, state: NodeState) {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.state = state;
            node.last_heartbeat = Utc::now();
        }
    }

    pub async fn update_node_load(&self, node_id: Uuid, load: NodeLoad) {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(&node_id) {
            node.load = load;
            node.last_heartbeat = Utc::now();
        }
    }

    pub async fn get_node(&self, node_id: &Uuid) -> Option<NodeInfo> {
        self.nodes.read().await.get(node_id).cloned()
    }

    pub async fn alive_nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.read().await;
        nodes
            .values()
            .filter(|n| n.state.is_available())
            .cloned()
            .collect()
    }

    pub async fn all_nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.read().await;
        nodes.values().cloned().collect()
    }

    pub async fn best_node_for_model(&self, model_id: &str, exclude: &[Uuid]) -> Option<NodeInfo> {
        let alive = self.alive_nodes().await;
        let exclude_set: std::collections::HashSet<Uuid> = exclude.iter().copied().collect();
        let candidates: Vec<NodeInfo> = alive
            .into_iter()
            .filter(|n| {
                n.node_id != self.config.node_id
                    && !exclude_set.contains(&n.node_id)
                    && n.capabilities.models.iter().any(|m| m == model_id)
            })
            .collect();
        candidates
            .into_iter()
            .min_by(|a, b| a.load_score().partial_cmp(&b.load_score()).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub async fn least_loaded_node(&self, exclude: &[Uuid]) -> Option<NodeInfo> {
        let alive = self.alive_nodes().await;
        let exclude_set: std::collections::HashSet<Uuid> = exclude.iter().copied().collect();
        let candidates: Vec<NodeInfo> = alive
            .into_iter()
            .filter(|n| n.node_id != self.config.node_id && !exclude_set.contains(&n.node_id))
            .collect();
        candidates
            .into_iter()
            .min_by(|a, b| a.load_score().partial_cmp(&b.load_score()).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub async fn random_alive_node(&self) -> Option<NodeInfo> {
        let alive = self.alive_nodes().await;
        alive.choose(&mut rand::thread_rng()).cloned()
    }

    pub async fn mark_stale_nodes(&self) -> Vec<Uuid> {
        let timeout = Duration::from_millis(self.config.heartbeat_timeout_ms * 3);
        let mut nodes = self.nodes.write().await;
        let mut stale = Vec::new();
        for node in nodes.values_mut() {
            if node.state == NodeState::Alive && node.is_stale(timeout) {
                node.state = NodeState::Suspect;
                stale.push(node.node_id);
            } else if node.state == NodeState::Suspect && node.is_stale(Duration::from_millis(self.config.node_eviction_timeout_ms)) {
                node.state = NodeState::Dead;
            }
        }
        stale
    }

    pub async fn evict_dead_nodes(&self) -> usize {
        let mut nodes = self.nodes.write().await;
        let before = nodes.len();
        nodes.retain(|_, n| n.state != NodeState::Dead);
        before - nodes.len()
    }

    pub async fn inc_incarnation(&self) -> u64 {
        let mut inc = self.incarnation.write().await;
        *inc += 1;
        *inc
    }

    pub async fn incarnation(&self) -> u64 {
        *self.incarnation.read().await
    }

    pub async fn known_nodes_snapshot(&self) -> Vec<(Uuid, NodeState, u64)> {
        let nodes = self.nodes.read().await;
        nodes
            .values()
            .map(|n| (n.node_id, n.state, 0))
            .collect()
    }

    pub async fn merge_gossip(&self, message: &GossipMessage) {
        let mut nodes = self.nodes.write().await;
        for (remote_id, remote_state, _incarnation) in &message.known_nodes {
            if *remote_id == self.config.node_id {
                continue;
            }
            match nodes.get(remote_id) {
                None => {
                    let info = NodeInfo::new(*remote_id, String::new());
                    nodes.insert(*remote_id, info);
                }
                Some(existing) => {
                    if *remote_state == NodeState::Dead
                        && existing.state != NodeState::Dead
                    {
                        if let Some(node) = nodes.get_mut(remote_id) {
                            node.state = NodeState::Suspect;
                        }
                    }
                }
            }
        }
        if message.sender_id != self.config.node_id {
            match nodes.get_mut(&message.sender_id) {
                Some(existing) => {
                    existing.load = message.state.load.clone();
                    existing.last_heartbeat = Utc::now();
                    existing.state = NodeState::Alive;
                    existing.address = message.sender_addr.clone();
                }
                None => {
                    let mut info = NodeInfo::new(message.sender_id, message.sender_addr.clone());
                    info.load = message.state.load.clone();
                    info.last_heartbeat = Utc::now();
                    nodes.insert(message.sender_id, info);
                }
            }
        }
    }
}
