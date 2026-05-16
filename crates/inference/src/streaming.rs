use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use tokio::sync::{mpsc, RwLock};

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub stream_id: Uuid,
    pub token_count: usize,
    pub is_active: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct ActiveStream {
    sender: mpsc::UnboundedSender<crate::GeneratedToken>,
    token_count: usize,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
pub struct StreamingEngine {
    active_streams: Arc<RwLock<HashMap<Uuid, ActiveStream>>>,
    max_concurrent_streams: usize,
}

impl StreamingEngine {
    pub fn new() -> Self {
        Self {
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_streams: 100,
        }
    }

    pub fn with_max_streams(mut self, max: usize) -> Self {
        self.max_concurrent_streams = max;
        self
    }

    pub async fn initialize(&mut self) -> Result<(), anyhow::Error> {
        Ok(())
    }

    pub async fn submit_request(&self, request: crate::InferenceRequest) -> Result<mpsc::UnboundedReceiver<crate::GeneratedToken>, anyhow::Error> {
        let stream_info = self.create_stream(&request).await?;
        let (tx, rx) = mpsc::unbounded_channel();

        {
            let mut streams = self.active_streams.write().await;
            if streams.len() >= self.max_concurrent_streams {
                return Err(anyhow::anyhow!("Max concurrent streams reached ({})", self.max_concurrent_streams));
            }
            streams.insert(stream_info.stream_id, ActiveStream {
                sender: tx,
                token_count: 0,
                created_at: chrono::Utc::now(),
            });
        }

        let stream_id = stream_info.stream_id;
        let streams = self.active_streams.clone();
        tokio::spawn(async move {
            let tokens = generate_tokens_from_prompt(&request);
            for token in tokens {
                let streams_read = streams.read().await;
                if let Some(active) = streams_read.get(&stream_id) {
                    let _ = active.sender.send(token);
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            let mut streams_write = streams.write().await;
            streams_write.remove(&stream_id);
        });

        Ok(rx)
    }

    pub async fn create_stream(&self, _request: &crate::InferenceRequest) -> Result<StreamInfo, anyhow::Error> {
        let stream_id = Uuid::new_v4();
        Ok(StreamInfo {
            stream_id,
            token_count: 0,
            is_active: true,
            created_at: chrono::Utc::now(),
        })
    }

    pub async fn get_stream_status(&self, stream_id: Uuid) -> Result<Option<StreamInfo>, anyhow::Error> {
        let streams = self.active_streams.read().await;
        Ok(streams.get(&stream_id).map(|s| StreamInfo {
            stream_id,
            token_count: s.token_count,
            is_active: true,
            created_at: s.created_at,
        }))
    }

    pub async fn send_token(&self, stream_id: Uuid, token: crate::GeneratedToken, is_last: bool) -> Result<(), anyhow::Error> {
        let streams = self.active_streams.read().await;
        if let Some(active) = streams.get(&stream_id) {
            active.sender.send(token).map_err(|e| anyhow::anyhow!("Failed to send token: {}", e))?;
            if is_last {
                drop(active);
                drop(streams);
                let mut streams = self.active_streams.write().await;
                streams.remove(&stream_id);
            }
        }
        Ok(())
    }

    pub async fn cancel_stream(&self, stream_id: Uuid) -> Result<bool, anyhow::Error> {
        let mut streams = self.active_streams.write().await;
        Ok(streams.remove(&stream_id).is_some())
    }

    pub async fn active_stream_count(&self) -> usize {
        self.active_streams.read().await.len()
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

fn generate_tokens_from_prompt(request: &crate::InferenceRequest) -> Vec<crate::GeneratedToken> {
    let text = &request.prompt;

    let pieces: Vec<String> = match nexora_tokenizer::pretokenizer::pretokenize(text) {
        Ok(pt) => pt.pieces.iter()
            .filter(|p| {
                match p.piece_type {
                    nexora_tokenizer::pretokenizer::PieceType::Whitespace => false,
                    _ => true,
                }
            })
            .map(|p| p.text.clone())
            .collect(),
        Err(_) => {
            text.chars().map(|c| c.to_string()).collect()
        }
    };

    if pieces.is_empty() {
        return Vec::new();
    }

    let n = pieces.len() as f64;
    let base: f64 = n.ln();

    pieces.iter().enumerate().map(|(i, piece)| {
        let raw_lp = -(base * (i as f64 + 1.0) / n.max(1.0));
        crate::GeneratedToken::new(
            i as u32,
            piece.to_string(),
            raw_lp as f32,
            i,
        )
    }).collect()
}
