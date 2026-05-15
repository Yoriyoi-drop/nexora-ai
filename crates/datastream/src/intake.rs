use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::{sleep, Duration};
use futures::stream::StreamExt;
use tracing::{info, debug, warn};

use crate::types::{DataSample, BatchConfig, SourceInfo, SampleStats, SourceCategory};
use uuid::Uuid;

pub struct StreamIntakeEngine {
    pub batch_config: BatchConfig,
    semaphore: Arc<Semaphore>,
    source_reputation: std::collections::HashMap<String, f64>,
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
            source_reputation: std::collections::HashMap::new(),
        }
    }

    pub fn with_prefetch(mut self, count: usize) -> Self {
        self.batch_config.prefetch_count = count;
        self.semaphore = Arc::new(Semaphore::new(count));
        self
    }

    pub async fn ingest_file(
        &self,
        path: &str,
        source: SourceInfo,
    ) -> mpsc::Receiver<DataSample> {
        let (tx, rx) = mpsc::channel(self.batch_config.max_batch_size);
        let path = path.to_string();
        let semaphore = self.semaphore.clone();

        tokio::spawn(async move {
            let content = match tokio::fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(e) => {
                    warn!("Failed to read file {}: {}", path, e);
                    return;
                }
            };

            let sample = DataSample {
                id: Uuid::new_v4(),
                text: content,
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
            let mut batch = Vec::with_capacity(batch_cfg.max_batch_size);
            for (text, source) in texts {
                let sample = DataSample {
                    id: Uuid::new_v4(),
                    text,
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
            for text in iter {
                let _permit = semaphore.acquire().await;
                let sample = DataSample {
                    id: Uuid::new_v4(),
                    text,
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
}

pub async fn dynamic_batcher(
    mut rx: mpsc::Receiver<DataSample>,
    batch_config: BatchConfig,
) -> mpsc::Receiver<Vec<DataSample>> {
    let (batch_tx, batch_rx) = mpsc::channel(16);

    tokio::spawn(async move {
        loop {
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
                                    let _ = batch_tx.send(batch).await;
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
                    return;
                }
            }
        }
    });

    batch_rx
}
