use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use nexora_runtime::cluster::{NodeInfo, NodeLoad, NodeRegistry};

use crate::{GeneratedToken, InferenceError, InferenceRequest, InferenceResponse};

pub struct DistributedRouter {
    registry: Arc<NodeRegistry>,
    client: reqwest::Client,
    local_node_id: Uuid,
    shared_secret: Option<String>,
    tls_enabled: bool,
}

impl DistributedRouter {
    pub fn new(registry: Arc<NodeRegistry>, local_node_id: Uuid) -> Self {
        Self::new_with_auth(registry, local_node_id, None, false)
    }

    pub fn new_with_auth(
        registry: Arc<NodeRegistry>,
        local_node_id: Uuid,
        shared_secret: Option<String>,
        tls_enabled: bool,
    ) -> Self {
        Self::register_load_fn(&registry);
        let mut client_builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(30));
        if let Some(ref secret) = shared_secret {
            let mut headers = reqwest::header::HeaderMap::new();
            if let Ok(val) = reqwest::header::HeaderValue::try_from(secret.as_str()) {
                headers.insert("x-nexora-auth", val);
            }
            client_builder = client_builder.default_headers(headers);
        }
        Self {
            registry,
            client: client_builder.build().unwrap_or_default(),
            local_node_id,
            shared_secret,
            tls_enabled,
        }
    }

    fn register_load_fn(registry: &Arc<NodeRegistry>) {
        let _reg = Arc::clone(registry);
        registry.set_dynamic_load_fn(Some(Arc::new(move || {
            use crate::inference_trait::{
                KV_CACHE_EXTERNAL_FRAG, KV_CACHE_INTERNAL_FRAG, KV_CACHE_TOTAL_BLOCKS,
                KV_CACHE_USED_BLOCKS, SCHEDULER_QUEUE_DEPTH,
            };
            use std::sync::atomic::Ordering;

            let _used = KV_CACHE_USED_BLOCKS.load(Ordering::Relaxed);
            let _total = KV_CACHE_TOTAL_BLOCKS.load(Ordering::Relaxed);
            let int_frag = KV_CACHE_INTERNAL_FRAG.load(Ordering::Relaxed) as f64 / 1_000_000.0;
            let ext_frag = KV_CACHE_EXTERNAL_FRAG.load(Ordering::Relaxed) as f64 / 1_000_000.0;

            NodeLoad {
                active_requests: 0,
                queue_depth: SCHEDULER_QUEUE_DEPTH.load(Ordering::Relaxed).max(0) as u32,
                gpu_util_pct: 0.0,
                memory_used_mb: 0,
                tokens_per_sec: 0.0,
                kv_fragmentation_pct: ((int_frag + ext_frag) * 50.0) as f32,
            }
        })));
    }

    pub fn registry(&self) -> &Arc<NodeRegistry> {
        &self.registry
    }

    pub async fn best_remote_node(&self) -> Option<NodeInfo> {
        let alive = self.registry.alive_nodes().await;
        let candidates: Vec<NodeInfo> = alive
            .into_iter()
            .filter(|n| n.node_id != self.local_node_id)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let local_load = self.registry.local_info().await.load;
        let local_score = local_load.active_requests as f64 + local_load.queue_depth as f64 * 0.5;

        candidates
            .into_iter()
            .filter(|n| n.load_score() < local_score)
            .min_by(|a, b| {
                a.load_score()
                    .partial_cmp(&b.load_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn rpc_scheme(&self) -> &'static str {
        if self.tls_enabled { "https" } else { "http" }
    }

    pub async fn route_remote(
        &self,
        node: &NodeInfo,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let url = format!("{}://{}/generate", self.rpc_scheme(), node.address);
        let resp = self
            .client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| {
                InferenceError::InternalError(format!("Remote request failed: {}", e))
            })?;

        if !resp.status().is_success() {
            return Err(InferenceError::InternalError(format!(
                "Remote node returned {}",
                resp.status()
            )));
        }

        resp.json::<InferenceResponse>()
            .await
            .map_err(|e| {
                InferenceError::InternalError(format!(
                    "Failed to decode remote response: {}",
                    e
                ))
            })
    }

    pub async fn route_remote_streaming(
        &self,
        node: &NodeInfo,
        request: &InferenceRequest,
    ) -> Result<mpsc::Receiver<Arc<GeneratedToken>>, InferenceError> {
        let url = format!("{}://{}/generate/stream", self.rpc_scheme(), node.address);
        let resp = self
            .client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| {
                InferenceError::InternalError(format!("Remote streaming failed: {}", e))
            })?;

        if !resp.status().is_success() {
            return Err(InferenceError::InternalError(format!(
                "Remote node returned {}",
                resp.status()
            )));
        }

        let (tx, rx) = mpsc::channel(64);
        let stream = resp.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::new();
            use futures::StreamExt;
            tokio::pin!(stream);

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(_) => break,
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                loop {
                    match buffer.find('\n') {
                        Some(nl) => {
                            let line = buffer[..nl].trim().to_string();
                            buffer = buffer[nl + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    return;
                                }
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                                    let text = val
                                        .get("token")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let pos = val
                                        .get("position")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as usize;

                                    let token = Arc::new(GeneratedToken {
                                        token_id: 0,
                                        token_text: text.into(),
                                        log_prob: 0.0,
                                        position: pos,
                                        metadata: Default::default(),
                                    });

                                    if tx.send(token).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                        None => break,
                    }
                }
            }
        });

        Ok(rx)
    }
}
