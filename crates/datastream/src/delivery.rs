use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tracing::{debug, info};

use crate::types::{DataSample, BatchConfig};

pub struct TrainingDeliveryLayer {
    pub batch_config: BatchConfig,
    pub max_queue_depth: usize,
    pub output_format: OutputFormat,
    pub enable_gpu_direct: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    JsonLines,
    Arrow,
    TensorRecords,
    RawText,
}

impl Default for TrainingDeliveryLayer {
    fn default() -> Self {
        Self {
            batch_config: BatchConfig::default(),
            max_queue_depth: 1024,
            output_format: OutputFormat::JsonLines,
            enable_gpu_direct: false,
        }
    }
}

impl TrainingDeliveryLayer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn with_gpu_direct(mut self, enable: bool) -> Self {
        self.enable_gpu_direct = enable;
        self
    }

    pub async fn deliver(
        &self,
        mut rx: mpsc::Receiver<Vec<DataSample>>,
        output_path: &str,
    ) -> Result<u64, anyhow::Error> {
        let output_path = output_path.to_string();
        let mut total = 0u64;

        while let Some(batch) = rx.recv().await {
            let batch_size = batch.len();
            match self.output_format {
                OutputFormat::JsonLines => {
                    self.write_jsonlines(&batch, &output_path, total).await?;
                }
                OutputFormat::RawText => {
                    self.write_raw_text(&batch, &output_path).await?;
                }
                _ => {
                    self.write_jsonlines(&batch, &output_path, total).await?;
                }
            }
            total += batch_size as u64;
            debug!("Delivered {} samples (total: {})", batch_size, total);
        }

        info!("Training delivery complete: {} samples to {}", total, output_path);
        Ok(total)
    }

    async fn write_jsonlines(
        &self,
        batch: &[DataSample],
        output_path: &str,
        offset: u64,
    ) -> Result<(), anyhow::Error> {
        let path = if offset == 0 {
            output_path.to_string()
        } else {
            format!("{}.part{}", output_path, offset / self.batch_config.max_batch_size as u64)
        };

        let mut content = String::with_capacity(batch.len() * 1024);
        for sample in batch {
            if let Ok(line) = serde_json::to_string(sample) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    async fn write_raw_text(
        &self,
        batch: &[DataSample],
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        let mut content = String::with_capacity(batch.len() * 1024);
        for sample in batch {
            content.push_str(&sample.text);
            content.push_str("\n---NEXORA_SEPARATOR---\n");
        }
        tokio::fs::write(output_path, content).await?;
        Ok(())
    }

    pub fn create_async_iterator(
        &self,
        rx: mpsc::Receiver<Vec<DataSample>>,
    ) -> impl futures::Stream<Item = Vec<DataSample>> {
        tokio_stream::wrappers::ReceiverStream::new(rx)
    }

    pub fn zero_copy_batch(&self, samples: Vec<DataSample>) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(samples.len() * 1024);
        for sample in &samples {
            if let Ok(json) = serde_json::to_vec(sample) {
                let len = json.len() as u32;
                buffer.extend_from_slice(&len.to_le_bytes());
                buffer.extend_from_slice(&json);
            }
        }
        buffer
    }

    pub async fn distributed_push(
        &self,
        mut rx: mpsc::Receiver<Vec<DataSample>>,
        endpoints: Vec<String>,
    ) -> Result<u64, anyhow::Error> {
        let mut total = 0u64;
        let client = reqwest::Client::new();

        while let Some(batch) = rx.recv().await {
            let payload = self.zero_copy_batch(batch);
            for endpoint in &endpoints {
                match client
                    .post(endpoint)
                    .header("content-type", "application/octet-stream")
                    .body(payload.clone())
                    .send()
                    .await
                {
                    Ok(resp) => {
                        if !resp.status().is_success() {
                            debug!("Push to {} returned status {}", endpoint, resp.status());
                        }
                    }
                    Err(e) => {
                        debug!("Failed to push to {}: {}", endpoint, e);
                    }
                }
            }
            total += 1;
            if total % 100 == 0 {
                sleep(Duration::from_millis(1)).await;
            }
        }

        Ok(total)
    }
}
