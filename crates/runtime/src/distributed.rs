use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

use crate::cluster::{ClusterConfig, NodeInfo, NodeLoad, NodeRegistry, NodeState};
use crate::scheduler::{RequestScheduler, SchedulingStrategy};
use crate::scheduler_trait::Scheduler;
use crate::{InferenceRequest, InferenceResponse, Result, SchedulerStats};

#[derive(Debug, Clone, PartialEq)]
pub enum RoutingStrategy {
    LeastLoaded,
    ModelAffinity,
    Random,
    RoundRobin,
    CacheLocality,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::LeastLoaded
    }
}

#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub cluster: ClusterConfig,
    pub routing: RoutingStrategy,
    pub local_scheduling_strategy: SchedulingStrategy,
    pub max_local_concurrent: usize,
    pub enable_remote: bool,
    pub health_check_interval_ms: u64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            cluster: ClusterConfig::default(),
            routing: RoutingStrategy::default(),
            local_scheduling_strategy: SchedulingStrategy::Priority,
            max_local_concurrent: 16,
            enable_remote: true,
            health_check_interval_ms: 2000,
        }
    }
}

struct RemoteClient {
    client: reqwest::Client,
    base_url: String,
    #[allow(dead_code)]
    node_id: Uuid,
}

impl RemoteClient {
    fn new(address: &str, node_id: Uuid) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            base_url: format!("http://{}/inference", address),
            node_id,
        }
    }

    async fn submit(
        &self,
        request: InferenceRequest,
        timeout_ms: u64,
    ) -> Result<mpsc::Receiver<InferenceResponse>> {
        let url = format!("{}/submit", self.base_url);
        let (tx, rx) = mpsc::channel(1);
        let tx_clone = tx.clone();
        let client = self.client.clone();

        tokio::spawn(async move {
            match client
                .post(&url)
                .json(&request)
                .timeout(Duration::from_millis(timeout_ms))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    match resp.json::<InferenceResponse>().await {
                        Ok(response) => {
                            let _ = tx_clone.send(response).await;
                        }
                        Err(e) => {
                            warn!("Failed to decode remote response: {}", e);
                        }
                    }
                }
                Ok(resp) => {
                    warn!("Remote submit returned {}", resp.status());
                }
                Err(e) => {
                    warn!("Remote submit failed: {}", e);
                }
            }
        });

        Ok(rx)
    }

    async fn health(&self) -> Result<NodeLoad> {
        let url = format!("{}/health", self.base_url);
        let resp = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Health check failed: {}", e))?;
        let load = resp
            .json::<NodeLoad>()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to decode health: {}", e))?;
        Ok(load)
    }
}

pub struct DistributedScheduler {
    local: Arc<RequestScheduler>,
    registry: Arc<NodeRegistry>,
    config: DistributedConfig,
    remote_clients: RwLock<Vec<RemoteClient>>,
    rr_index: RwLock<usize>,
    shutdown: AtomicBool,
    enabled: AtomicBool,
}

impl DistributedScheduler {
    pub fn new(
        local: Arc<RequestScheduler>,
        registry: Arc<NodeRegistry>,
        config: DistributedConfig,
    ) -> Self {
        let enabled = config.enable_remote;
        Self {
            local,
            registry,
            config,
            remote_clients: RwLock::new(Vec::new()),
            rr_index: RwLock::new(0),
            shutdown: AtomicBool::new(false),
            enabled: AtomicBool::new(enabled),
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing distributed scheduler (node_id={})", self.config.cluster.node_id);
        self.local.initialize().await?;
        self.sync_remote_clients().await;
        self.enabled.store(self.config.enable_remote, Ordering::Relaxed);
        Ok(())
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn registry(&self) -> &Arc<NodeRegistry> {
        &self.registry
    }

    pub async fn submit_request(
        &self,
        request: InferenceRequest,
        response_tx: mpsc::Sender<InferenceResponse>,
    ) -> Result<()> {
        if !self.is_enabled() {
            return self
                .local
                .submit_request(request, response_tx)
                .await;
        }

        let model_id = &request.model_id;
        let local_load = self.registry.local_info().await.load;

        let remote_candidates = self.select_remote_nodes(model_id).await;

        let local_score = local_load.active_requests as f64 + local_load.queue_depth as f64 * 0.5;
        let should_route_remote = !remote_candidates.is_empty()
            && remote_candidates[0].load_score() < local_score;

        if should_route_remote {
            let peer = &remote_candidates[0];
            self.route_to_remote(peer, request, response_tx).await
        } else {
            self.local.submit_request(request, response_tx).await
        }
    }

    async fn select_remote_nodes(&self, model_id: &str) -> Vec<NodeInfo> {
        let alive = self.registry.alive_nodes().await;
        let mut candidates: Vec<NodeInfo> = alive
            .into_iter()
            .filter(|n| n.node_id != self.config.cluster.node_id)
            .collect();

        match self.config.routing {
            RoutingStrategy::ModelAffinity => {
                candidates.retain(|n| n.capabilities.models.iter().any(|m| m == model_id));
                candidates.sort_by(|a, b| {
                    a.load_score()
                        .partial_cmp(&b.load_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            RoutingStrategy::LeastLoaded => {
                candidates.sort_by(|a, b| {
                    a.load_score()
                        .partial_cmp(&b.load_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            RoutingStrategy::Random => {
                use rand::seq::SliceRandom;
                candidates.shuffle(&mut rand::thread_rng());
            }
            RoutingStrategy::RoundRobin => {
                let mut idx = self.rr_index.write().await;
                if !candidates.is_empty() {
                    *idx = (*idx + 1) % candidates.len();
                    let rotated = candidates.split_off(*idx);
                    candidates = [rotated, candidates].concat();
                }
            }
            RoutingStrategy::CacheLocality => {
                candidates.sort_by(|a, b| {
                    let a_cache_score = (100.0 - a.load.kv_fragmentation_pct) as f64;
                    let b_cache_score = (100.0 - b.load.kv_fragmentation_pct) as f64;
                    b_cache_score
                        .partial_cmp(&a_cache_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        candidates
    }

    async fn route_to_remote(
        &self,
        peer: &NodeInfo,
        request: InferenceRequest,
        response_tx: mpsc::Sender<InferenceResponse>,
    ) -> Result<()> {
        let client = RemoteClient::new(&peer.address, peer.node_id);
        let timeout = self.config.cluster.request_timeout_ms;

        match client.submit(request, timeout).await {
            Ok(mut rx) => {
                let tx = response_tx.clone();
                tokio::spawn(async move {
                    while let Some(resp) = rx.recv().await {
                        if tx.send(resp).await.is_err() {
                            break;
                        }
                    }
                });
                Ok(())
            }
            Err(e) => {
                warn!("Remote routing to {} failed: {}", peer.address, e);
                self.registry
                    .update_node_state(peer.node_id, NodeState::Suspect)
                    .await;
                Err(e)
            }
        }
    }

    async fn sync_remote_clients(&self) {
        let alive = self.registry.alive_nodes().await;
        let mut clients = self.remote_clients.write().await;
        clients.clear();
        for node in &alive {
            if node.node_id != self.config.cluster.node_id {
                clients.push(RemoteClient::new(&node.address, node.node_id));
            }
        }
    }

    pub async fn health_check_loop(self: Arc<Self>) {
        let interval = Duration::from_millis(self.config.health_check_interval_ms);
        loop {
            tokio::time::sleep(interval).await;
            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            let alive = self.registry.alive_nodes().await;
            for node in alive {
                if node.node_id == self.config.cluster.node_id {
                    continue;
                }
                let client = RemoteClient::new(&node.address, node.node_id);
                match client.health().await {
                    Ok(load) => {
                        self.registry.update_node_load(node.node_id, load).await;
                    }
                    Err(_) => {
                        self.registry
                            .update_node_state(node.node_id, NodeState::Suspect)
                            .await;
                    }
                }
            }

            self.sync_remote_clients().await;
        }
    }
}

#[async_trait]
impl Scheduler for DistributedScheduler {
    async fn submit(
        &self,
        request_id: Uuid,
        response_tx: mpsc::Sender<InferenceResponse>,
    ) -> Result<()> {
        let request = InferenceRequest {
            model_id: String::new(),
            inputs: vec![],
            parameters: std::collections::HashMap::new(),
            request_id: Some(request_id.to_string()),
            input_tokens: vec![],
            target_tokens: None,
            priority: 50,
            metadata: std::collections::HashMap::new(),
        };
        self.submit_request(request, response_tx).await
    }

    async fn cancel(&self, request_id: Uuid) -> Result<bool> {
        self.local.cancel_request(request_id).await
    }

    async fn status(
        &self,
        request_id: Uuid,
    ) -> Result<Option<crate::RequestStatus>> {
        self.local.get_request_status(request_id).await
    }

    async fn stats(&self) -> SchedulerStats {
        self.local.get_stats().await
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down distributed scheduler");
        self.shutdown.store(true, Ordering::Relaxed);
        self.local.shutdown().await
    }
}
