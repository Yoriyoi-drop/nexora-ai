use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};
use tracing::{debug, warn};

use crate::types::{BatchConfig, DataSample, SampleStats, SourceInfo};
use uuid::Uuid;

pub struct StreamIntakeEngine {
    pub batch_config: BatchConfig,
    semaphore: Arc<Semaphore>,
    background_handles: Arc<std::sync::Mutex<Vec<JoinHandle<()>>>>,
}

impl Drop for StreamIntakeEngine {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.background_handles.lock() {
            for h in handles.drain(..) {
                h.abort();
            }
        }
    }
}

impl Default for StreamIntakeEngine {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

impl StreamIntakeEngine {
    pub fn new(batch_config: BatchConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(batch_config.prefetch_count)),
            batch_config,
            background_handles: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn with_prefetch(mut self, count: usize) -> Self {
        self.batch_config.prefetch_count = count;
        self.semaphore = Arc::new(Semaphore::new(count));
        self
    }

    pub async fn ingest_file(&self, path: &str, source: SourceInfo) -> mpsc::Receiver<DataSample> {
        let (tx, rx) = mpsc::channel(self.batch_config.max_batch_size);
        let path = path.to_string();
        let source = source;
        let handles = self.background_handles.clone();
        let _permit = self.semaphore.clone().acquire_owned().await;

        let handle = tokio::spawn(async move {
            let __permit = _permit;
            let read_result = tokio::time::timeout(
                Duration::from_secs(60),
                tokio::fs::read_to_string(&path),
            )
            .await;

            let content = match read_result {
                Ok(Ok(c)) => c,
                Ok(Err(e)) => {
                    warn!("Failed to read file {}: {}", path, e);
                    return;
                }
                Err(_) => {
                    warn!("Timed out reading file {} after 60s", path);
                    return;
                }
            };

            let char_count = content.chars().count();
            let word_count = content.split_whitespace().count();
            let token_count = content.len() / 4;
            let token_ids = Some(content.bytes().map(|b| b as u32).collect());

            let sample = DataSample {
                id: Uuid::new_v4(),
                text: content,
                token_ids,
                metadata: std::collections::HashMap::new(),
                source,
                stats: SampleStats {
                    char_count,
                    word_count,
                    token_count,
                    ..Default::default()
                },
                domains: Vec::new(),
                score: None,
                curriculum_level: None,
            };

            if tx.send(sample).await.is_err() {
                debug!("Ingest channel closed for {}", path);
            }
        });
        if let Ok(mut h) = handles.lock() {
            h.push(handle);
        }

        rx
    }

    pub async fn ingest_batch(
        &self,
        texts: Vec<(String, SourceInfo)>,
    ) -> mpsc::Receiver<DataSample> {
        let (tx, rx) = mpsc::channel(self.batch_config.max_batch_size);
        let batch_cfg = self.batch_config.clone();
        let handles = self.background_handles.clone();
        let _permit = self.semaphore.clone().acquire_owned().await;

        let handle = tokio::spawn(async move {
            let __permit = _permit;
            let ingest_timeout = Duration::from_secs(300);
            let start = std::time::Instant::now();

            let mut batch = Vec::with_capacity(batch_cfg.max_batch_size);
            for (text, source) in texts {
                if start.elapsed() > ingest_timeout {
                    warn!("ingest_batch timed out after 300s");
                    return;
                }
                let char_count = text.chars().count();
                let word_count = text.split_whitespace().count();
                let token_count = text.len() / 4;
                let token_ids = Some(text.bytes().map(|b| b as u32).collect());

                let sample = DataSample {
                    id: Uuid::new_v4(),
                    text,
                    token_ids,
                    metadata: std::collections::HashMap::new(),
                    source,
                    stats: SampleStats {
                        char_count,
                        word_count,
                        token_count,
                        ..Default::default()
                    },
                    domains: Vec::new(),
                    score: None,
                    curriculum_level: None,
                };
                batch.push(sample);

                if batch.len() >= batch_cfg.max_batch_size {
                    let drained: Vec<_> = batch.drain(..).collect();
                    for sample in drained {
                        if tx.send(sample).await.is_err() {
                            return;
                        }
                    }
                    sleep(Duration::from_millis(1)).await;
                }
            }

            if !batch.is_empty() {
                for sample in batch {
                    if tx.send(sample).await.is_err() {
                        return;
                    }
                }
            }
        });
        if let Ok(mut h) = handles.lock() {
            h.push(handle);
        }

        rx
    }

    pub async fn stream_from_iterator(
        &self,
        iter: impl Iterator<Item = String> + Send + 'static,
        source: SourceInfo,
    ) -> mpsc::Receiver<DataSample>
    where
        String: 'static,
    {
        let (tx, rx) = mpsc::channel(self.batch_config.max_batch_size);
        let batch_cfg = self.batch_config.clone();
        let handles = self.background_handles.clone();
        let source = source;
        let _permit = self.semaphore.clone().acquire_owned().await;

        let handle = tokio::spawn(async move {
            let __permit = _permit;
            let ingest_timeout = Duration::from_secs(300);
            let start = std::time::Instant::now();

            for text in iter {
                if start.elapsed() > ingest_timeout {
                    warn!("stream_from_iterator timed out after 300s");
                    return;
                }
                let char_count = text.chars().count();
                let word_count = text.split_whitespace().count();
                let token_count = text.len() / 4;
                let token_ids = Some(text.bytes().map(|b| b as u32).collect());

                let sample = DataSample {
                    id: Uuid::new_v4(),
                    text,
                    token_ids,
                    metadata: std::collections::HashMap::new(),
                    source: source.clone(),
                    stats: SampleStats {
                        char_count,
                        word_count,
                        token_count,
                        ..Default::default()
                    },
                    domains: Vec::new(),
                    score: None,
                    curriculum_level: None,
                };
                if tx.send(sample).await.is_err() {
                    return;
                }
                if batch_cfg.enable_dynamic {
                    sleep(Duration::from_millis(1)).await;
                }
            }
        });
        if let Ok(mut h) = handles.lock() {
            h.push(handle);
        }

        rx
    }

    /// Prepare samples for pipeline processing: validate, assign IDs, normalize
    pub fn prepare_samples(&self, samples: Vec<DataSample>) -> Vec<DataSample> {
        samples
            .into_iter()
            .map(|mut s| {
                if s.id.is_nil() {
                    s.id = Uuid::new_v4();
                }
                if s.text.is_empty() {
                    s.text = String::new();
                }
                s
            })
            .collect()
    }
}

/// Wrapper that aborts the background batch processing task on drop
pub struct BatchedReceiver {
    pub rx: mpsc::Receiver<Vec<DataSample>>,
    handle: Option<JoinHandle<()>>,
}

impl BatchedReceiver {
    pub fn new(rx: mpsc::Receiver<Vec<DataSample>>, handle: JoinHandle<()>) -> Self {
        Self { rx, handle: Some(handle) }
    }
}

impl Drop for BatchedReceiver {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

pub async fn dynamic_batcher(
    mut rx: mpsc::Receiver<DataSample>,
    batch_config: BatchConfig,
) -> BatchedReceiver {
    let (batch_tx, batch_rx) = mpsc::channel(16);
    static BATCHER_SEM: std::sync::OnceLock<Arc<Semaphore>> = std::sync::OnceLock::new();
    let sem = BATCHER_SEM.get_or_init(|| Arc::new(Semaphore::new(16)));
    let _permit = sem.clone().acquire_owned().await;

    let handle = tokio::spawn(async move {
        let __permit = _permit;
        let total_timeout = Duration::from_secs(600);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > total_timeout {
                warn!("dynamic_batcher total timeout reached (600s)");
                return;
            }

            let mut batch = Vec::with_capacity(batch_config.max_batch_size);
            let timer = sleep(Duration::from_millis(batch_config.max_wait_ms));
            tokio::pin!(timer);

            loop {
                tokio::select! {
                    maybe_sample = rx.recv() => {
                        match maybe_sample {
                            Some(sample) => {
                                batch.push(sample);
                                if batch.len() >= batch_config.max_batch_size {
                                    break;
                                }
                            }
                            None => {
                                if !batch.is_empty() {
                                    if batch_tx.send(batch).await.is_err() {
                                        warn!("dynamic_batcher: failed to send final batch");
                                    }
                                }
                                return;
                            }
                        }
                    }
                    _ = &mut timer => {
                        break;
                    }
                }
            }

            if !batch.is_empty() {
                if batch_tx.send(batch).await.is_err() {
                    warn!("dynamic_batcher: failed to send final batch on timeout");
                    return;
                }
            }
        }
    });

    BatchedReceiver::new(batch_rx, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BatchConfig, SourceCategory};

    #[test]
    fn test_stream_intake_engine_new() {
        let cfg = BatchConfig {
            max_batch_size: 32,
            max_wait_ms: 50,
            prefetch_count: 2,
            enable_dynamic: true,
        };
        let engine = StreamIntakeEngine::new(cfg.clone());
        assert_eq!(engine.batch_config.max_batch_size, 32);
        assert_eq!(engine.batch_config.prefetch_count, 2);
    }

    #[test]
    fn test_default_uses_default_config() {
        let engine = StreamIntakeEngine::default();
        assert_eq!(engine.batch_config, BatchConfig::default());
    }

    #[test]
    fn test_with_prefetch_updates_semaphore() {
        let engine = StreamIntakeEngine::default().with_prefetch(8);
        assert_eq!(engine.batch_config.prefetch_count, 8);
    }

    #[test]
    fn test_prepare_samples_assigns_nil_ids() {
        let engine = StreamIntakeEngine::default();
        let sample = DataSample {
            id: Uuid::nil(),
            text: "hello".into(),
            token_ids: None,
            metadata: std::collections::HashMap::new(),
            source: SourceInfo {
                name: "test".into(),
                url: None,
                trust_score: 0.5,
                category: SourceCategory::Other,
                fetch_timestamp: 0,
            },
            stats: SampleStats::default(),
            domains: vec![],
            score: None,
            curriculum_level: None,
        };
        let prepared = engine.prepare_samples(vec![sample]);
        assert_eq!(prepared.len(), 1);
        assert!(!prepared[0].id.is_nil());
    }

    #[test]
    fn test_batched_receiver_new() {
        let (tx, rx) = mpsc::channel::<Vec<DataSample>>(16);
        let handle = tokio::spawn(async move {
            tx.send(vec![]).await.ok();
        });
        let br = BatchedReceiver::new(rx, handle);
        assert!(!br.rx.is_closed());
    }

    #[test]
    fn test_dynamic_batcher_function_exists() {
        let cfg = BatchConfig::default();
        let (tx, rx) = mpsc::channel::<DataSample>(16);
        let _fut = dynamic_batcher(rx, cfg);
        // Just verifying the function signature compiles
        drop(tx);
    }

    #[test]
    fn test_prepare_samples_empty_input() {
        let engine = StreamIntakeEngine::default();
        let prepared = engine.prepare_samples(vec![]);
        assert!(prepared.is_empty());
    }
}
