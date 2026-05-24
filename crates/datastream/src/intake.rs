use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::{sleep, Duration};
use tracing::{debug, warn};

use crate::types::{BatchConfig, DataSample, SampleStats, SourceInfo};
use uuid::Uuid;

pub struct StreamIntakeEngine {
    pub batch_config: BatchConfig,
    semaphore: Arc<Semaphore>,
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
        let semaphore = self.semaphore.clone();

        tokio::spawn(async move {
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

            let sample = DataSample {
                id: Uuid::new_v4(),
                text: content,
                token_ids: None,
                metadata: std::collections::HashMap::new(),
                source: source.clone(),
                stats: SampleStats::default(),
                domains: Vec::new(),
                score: None,
                curriculum_level: None,
            };

            let _permit = semaphore.acquire().await;
            if tx.send(sample).await.is_err() {
                debug!("Ingest channel closed for {}", path);
            }
        });

        rx
    }

    pub async fn ingest_batch(
        &self,
        texts: Vec<(String, SourceInfo)>,
    ) -> mpsc::Receiver<DataSample> {
        let (tx, rx) = mpsc::channel(self.batch_config.max_batch_size);
        let semaphore = self.semaphore.clone();
        let batch_cfg = self.batch_config.clone();

        tokio::spawn(async move {
            let ingest_timeout = Duration::from_secs(300);
            let start = std::time::Instant::now();

            let mut batch = Vec::with_capacity(batch_cfg.max_batch_size);
            for (text, source) in texts {
                if start.elapsed() > ingest_timeout {
                    warn!("ingest_batch timed out after 300s");
                    return;
                }
                let sample = DataSample {
                    id: Uuid::new_v4(),
                    text,
                    token_ids: None,
                    metadata: std::collections::HashMap::new(),
                    source,
                    stats: SampleStats::default(),
                    domains: Vec::new(),
                    score: None,
                    curriculum_level: None,
                };
                batch.push(sample);

                if batch.len() >= batch_cfg.max_batch_size {
                    let _permit = semaphore.acquire().await;
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
                let _permit = semaphore.acquire().await;
                for sample in batch {
                    if tx.send(sample).await.is_err() {
                        return;
                    }
                }
            }
        });

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
        let semaphore = self.semaphore.clone();
        let batch_cfg = self.batch_config.clone();

        tokio::spawn(async move {
            let ingest_timeout = Duration::from_secs(300);
            let start = std::time::Instant::now();

            for text in iter {
                if start.elapsed() > ingest_timeout {
                    warn!("stream_from_iterator timed out after 300s");
                    return;
                }
                let _permit = semaphore.acquire().await;
                let sample = DataSample {
                    id: Uuid::new_v4(),
                    text,
                    token_ids: None,
                    metadata: std::collections::HashMap::new(),
                    source: source.clone(),
                    stats: SampleStats::default(),
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

pub async fn dynamic_batcher(
    mut rx: mpsc::Receiver<DataSample>,
    batch_config: BatchConfig,
) -> mpsc::Receiver<Vec<DataSample>> {
    let (batch_tx, batch_rx) = mpsc::channel(16);

    tokio::spawn(async move {
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

    batch_rx
}
