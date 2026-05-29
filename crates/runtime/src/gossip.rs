use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::debug;

use crate::cluster::{GossipMessage, GossipState, NodeInfo, NodeRegistry};

#[derive(Debug, Clone)]
pub struct GossipConfig {
    pub push_pull_interval_ms: u64,
    pub fanout: usize,
    pub max_packet_size: usize,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            push_pull_interval_ms: 1000,
            fanout: 2,
            max_packet_size: 65536,
        }
    }
}

enum GossipRound {
    Push,
    Pull,
}

pub struct GossipProtocol {
    registry: Arc<NodeRegistry>,
    config: GossipConfig,
    client: reqwest::Client,
    round: Mutex<GossipRound>,
}

impl GossipProtocol {
    pub fn new(registry: Arc<NodeRegistry>) -> Self {
        Self {
            registry,
            config: GossipConfig::default(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            round: Mutex::new(GossipRound::Push),
        }
    }

    pub fn with_config(mut self, config: GossipConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub async fn start(self: Arc<Self>) {
        let interval = Duration::from_millis(self.config.push_pull_interval_ms);
        loop {
            tokio::time::sleep(interval).await;
            if let Err(e) = self.run_round().await {
                debug!("Gossip round failed: {}", e);
            }
        }
    }

    async fn run_round(&self) -> Result<(), anyhow::Error> {
        self.registry.refresh_dynamic_load().await;
        self.registry.mark_stale_nodes().await;
        self.registry.evict_dead_nodes().await;

        let peer = match self.registry.random_alive_node().await {
            Some(node) => node,
            None => return Ok(()),
        };

        let message = self.build_gossip_message().await;

        let is_push = {
            let mut round = self.round.lock().unwrap();
            let current = std::mem::replace(&mut *round, GossipRound::Pull);
            matches!(current, GossipRound::Push)
        };

        if is_push {
            self.push_to(peer, message).await?;
        } else {
            if let Some(response) = self.pull_from(peer).await? {
                self.registry.merge_gossip(&response).await;
            }
            *self.round.lock().unwrap() = GossipRound::Push;
        }

        Ok(())
    }

    async fn build_gossip_message(&self) -> GossipMessage {
        let local = self.registry.local_info().await;
        let inc = self.registry.incarnation().await;
        let known = self.registry.known_nodes_snapshot().await;

        GossipMessage {
            sender_id: local.node_id,
            sender_addr: local.address.clone(),
            state: GossipState {
                node_id: local.node_id,
                load: local.load,
                incarnation: inc,
            },
            known_nodes: known,
        }
    }

    async fn push_to(
        &self,
        peer: NodeInfo,
        message: GossipMessage,
    ) -> Result<(), anyhow::Error> {
        let url = format!("http://{}/cluster/gossip/push", peer.address);
        let resp = self
            .client
            .post(&url)
            .json(&message)
            .timeout(Duration::from_secs(3))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                if let Ok(remote_msg) = r.json::<GossipMessage>().await {
                    self.registry.merge_gossip(&remote_msg).await;
                }
                Ok(())
            }
            Ok(r) => {
                debug!("Gossip push to {} returned {}", peer.address, r.status());
                self.registry
                    .update_node_state(peer.node_id, crate::cluster::NodeState::Suspect)
                    .await;
                Ok(())
            }
            Err(e) => {
                debug!("Gossip push to {} failed: {}", peer.address, e);
                self.registry
                    .update_node_state(peer.node_id, crate::cluster::NodeState::Suspect)
                    .await;
                Ok(())
            }
        }
    }

    async fn pull_from(
        &self,
        peer: NodeInfo,
    ) -> Result<Option<GossipMessage>, anyhow::Error> {
        let url = format!("http://{}/cluster/gossip/pull", peer.address);
        let local_msg = self.build_gossip_message().await;

        let resp = self
            .client
            .post(&url)
            .json(&local_msg)
            .timeout(Duration::from_secs(3))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                match r.json::<GossipMessage>().await {
                    Ok(remote_msg) => Ok(Some(remote_msg)),
                    Err(e) => {
                        debug!("Failed to decode gossip pull response: {}", e);
                        Ok(None)
                    }
                }
            }
            Ok(r) => {
                debug!("Gossip pull from {} returned {}", peer.address, r.status());
                self.registry
                    .update_node_state(peer.node_id, crate::cluster::NodeState::Suspect)
                    .await;
                Ok(None)
            }
            Err(e) => {
                debug!("Gossip pull from {} failed: {}", peer.address, e);
                self.registry
                    .update_node_state(peer.node_id, crate::cluster::NodeState::Suspect)
                    .await;
                Ok(None)
            }
        }
    }

    pub async fn handle_push(
        &self,
        message: GossipMessage,
    ) -> Result<GossipMessage, anyhow::Error> {
        self.registry.merge_gossip(&message).await;
        Ok(self.build_gossip_message().await)
    }

    pub async fn handle_pull(
        &self,
        message: GossipMessage,
    ) -> Result<GossipMessage, anyhow::Error> {
        self.registry.merge_gossip(&message).await;
        Ok(self.build_gossip_message().await)
    }
}
