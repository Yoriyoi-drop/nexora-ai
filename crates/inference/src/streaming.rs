use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;
use tracing::debug;

use crate::GeneratedToken;

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub stream_id: Uuid,
    pub token_count: usize,
    pub is_active: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

struct ActiveStream {
    sender: mpsc::Sender<GeneratedToken>,
    token_count: usize,
    created_at: Instant,
    last_token_at: Instant,
}

pub struct StreamingEngine {
    active_streams: Arc<RwLock<HashMap<Uuid, ActiveStream>>>,
    max_concurrent_streams: usize,
    stream_timeout: Duration,
}

impl StreamingEngine {
    pub fn new() -> Self {
        Self {
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_streams: 100,
            stream_timeout: Duration::from_secs(300),
        }
    }

    pub fn with_max_streams(mut self, max: usize) -> Self {
        self.max_concurrent_streams = max;
        self
    }

    pub fn with_stream_timeout(mut self, timeout: Duration) -> Self {
        self.stream_timeout = timeout;
        self
    }

    pub async fn initialize(&self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    pub async fn create_stream(
        &self,
    ) -> Result<(Uuid, mpsc::Receiver<GeneratedToken>), anyhow::Error> {
        let stream_id = Uuid::new_v4();

        {
            let mut streams = self.active_streams.write().await;
            if streams.len() >= self.max_concurrent_streams {
                self.evict_stale_locked(&mut streams);
            }
            if streams.len() >= self.max_concurrent_streams {
                return Err(anyhow::anyhow!(
                    "Max concurrent streams reached ({})",
                    self.max_concurrent_streams
                ));
            }
        }

        let (tx, rx) = mpsc::channel(64);
        {
            let mut streams = self.active_streams.write().await;
            streams.insert(
                stream_id,
                ActiveStream {
                    sender: tx,
                    token_count: 0,
                    created_at: Instant::now(),
                    last_token_at: Instant::now(),
                },
            );
        }

        Ok((stream_id, rx))
    }

    pub async fn send_token(
        &self,
        stream_id: Uuid,
        token: GeneratedToken,
    ) -> Result<(), anyhow::Error> {
        let streams = self.active_streams.read().await;
        match streams.get(&stream_id) {
            Some(active) => {
                active
                    .sender
                    .send(token)
                    .await
                    .map_err(|_| anyhow::anyhow!("Stream {} receiver dropped", stream_id))?;
                drop(streams);
                // update token_count outside read lock
                let mut streams = self.active_streams.write().await;
                if let Some(entry) = streams.get_mut(&stream_id) {
                    entry.token_count += 1;
                    entry.last_token_at = Instant::now();
                }
                Ok(())
            }
            None => Err(anyhow::anyhow!("Stream {} not found", stream_id)),
        }
    }

    pub async fn push_tokens(
        &self,
        stream_id: Uuid,
        tokens: Vec<GeneratedToken>,
        is_last: bool,
    ) -> Result<(), anyhow::Error> {
        for token in tokens {
            self.send_token(stream_id, token).await?;
        }
        if is_last {
            self.close_stream(stream_id).await;
        }
        Ok(())
    }

    pub async fn close_stream(&self, stream_id: Uuid) {
        let mut streams = self.active_streams.write().await;
        streams.remove(&stream_id);
    }

    pub async fn cancel_stream(&self, stream_id: Uuid) -> Result<bool, anyhow::Error> {
        let mut streams = self.active_streams.write().await;
        Ok(streams.remove(&stream_id).is_some())
    }

    pub async fn active_stream_count(&self) -> usize {
        self.active_streams.read().await.len()
    }

    pub async fn get_stream_status(&self, stream_id: Uuid) -> Option<StreamInfo> {
        let streams = self.active_streams.read().await;
        streams.get(&stream_id).map(|s| StreamInfo {
            stream_id,
            token_count: s.token_count,
            is_active: true,
            created_at: chrono::DateTime::from(
                std::time::SystemTime::now()
                    - std::time::SystemTime::UNIX_EPOCH
                          .elapsed()
                          .unwrap_or_default()
                          .saturating_sub(s.created_at.elapsed()),
            ),
        })
    }

    pub async fn evict_stale(&self) -> usize {
        let mut streams = self.active_streams.write().await;
        self.evict_stale_locked(&mut streams)
    }

    fn evict_stale_locked(&self, streams: &mut HashMap<Uuid, ActiveStream>) -> usize {
        let before = streams.len();
        streams.retain(|_, s| s.last_token_at.elapsed() < self.stream_timeout);
        let evicted = before - streams.len();
        if evicted > 0 {
            debug!("Evicted {} stale streams", evicted);
        }
        evicted
    }

    pub async fn shutdown(&self) -> Result<(), anyhow::Error> {
        let mut streams = self.active_streams.write().await;
        streams.clear();
        Ok(())
    }
}

impl Default for StreamingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_close_stream() {
        let engine = StreamingEngine::new();
        let (stream_id, _rx) = engine.create_stream().await.unwrap();
        assert_eq!(engine.active_stream_count().await, 1);
        engine.close_stream(stream_id).await;
        assert_eq!(engine.active_stream_count().await, 0);
    }

    #[tokio::test]
    async fn test_send_and_receive_token() {
        let engine = StreamingEngine::new();
        let (stream_id, mut rx) = engine.create_stream().await.unwrap();

        let token = GeneratedToken::new(1, "hello".to_string(), -0.5, 0);
        engine.send_token(stream_id, token).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.token_id, 1);
        assert_eq!(&*received.token_text, "hello");

        engine.close_stream(stream_id).await;
    }

    #[tokio::test]
    async fn test_cancel_stream() {
        let engine = StreamingEngine::new();
        let (stream_id, _rx) = engine.create_stream().await.unwrap();
        assert!(engine.cancel_stream(stream_id).await.unwrap());
        assert!(!engine.cancel_stream(stream_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_max_concurrent_streams() {
        let engine = StreamingEngine::new().with_max_streams(2);
        let (sid1, _) = engine.create_stream().await.unwrap();
        let (sid2, _) = engine.create_stream().await.unwrap();
        let result = engine.create_stream().await;
        assert!(result.is_err());
        engine.close_stream(sid1).await;
        engine.close_stream(sid2).await;
    }

    #[tokio::test]
    async fn test_stream_status() {
        let engine = StreamingEngine::new();
        let (stream_id, _rx) = engine.create_stream().await.unwrap();
        let status = engine.get_stream_status(stream_id).await;
        assert!(status.is_some());
        assert_eq!(status.unwrap().stream_id, stream_id);
        engine.close_stream(stream_id).await;
        assert!(engine.get_stream_status(stream_id).await.is_none());
    }
}
