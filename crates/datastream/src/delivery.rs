use std::path::PathBuf;

use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tracing::{debug, info};

use crate::types::{BatchConfig, DataSample};
use nexora_common::retry::RetryConfig;

pub struct TrainingDeliveryLayer {
    pub batch_config: BatchConfig,
    pub output_format: OutputFormat,
    pub output_dir: PathBuf,
}

/// Output format for training data delivery.
///
/// - `JsonLines`: Newline-delimited JSON (default, always supported)
/// - `Arrow`: Apache Arrow IPC format (requires `arrow` feature)
/// - `TensorRecords`: Binary length-prefixed records (always supported)
/// - `RawText`: Raw text with separators (always supported)
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
            output_format: OutputFormat::JsonLines,
            output_dir: PathBuf::from("output"),
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

    pub fn with_output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = dir;
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
                OutputFormat::Arrow => {
                    self.write_arrow(&batch, &output_path, total).await?;
                }
                OutputFormat::TensorRecords => {
                    self.write_tensor_records(&batch, &output_path, total).await?;
                }
            }
            total += batch_size as u64;
            debug!("Delivered {} samples (total: {})", batch_size, total);
        }

        info!(
            "Training delivery complete: {} samples to {}",
            total, output_path
        );
        Ok(total)
    }

    async fn write_jsonlines(
        &self,
        batch: &[DataSample],
        output_path: &str,
        offset: u64,
    ) -> Result<(), anyhow::Error> {
        use tokio::io::AsyncWriteExt;
        let path = if offset == 0 {
            output_path.to_string()
        } else {
            format!(
                "{}.part{}",
                output_path,
                offset / self.batch_config.max_batch_size as u64
            )
        };

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        let mut content = String::with_capacity(batch.len() * 1024);
        for sample in batch {
            if let Ok(line) = serde_json::to_string(sample) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        file.write_all(content.as_bytes()).await?;
        Ok(())
    }

    async fn write_raw_text(
        &self,
        batch: &[DataSample],
        output_path: &str,
    ) -> Result<(), anyhow::Error> {
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .await?;
        let mut content = String::with_capacity(batch.len() * 1024);
        for sample in batch {
            content.push_str(&sample.text);
            content.push_str("\n---NEXORA_SEPARATOR---\n");
        }
        file.write_all(content.as_bytes()).await?;
        Ok(())
    }

    /// Write samples in Apache Arrow IPC format (requires `arrow` feature)
    #[cfg(feature = "arrow")]
    async fn write_arrow(
        &self,
        batch: &[DataSample],
        output_path: &str,
        offset: u64,
    ) -> Result<(), anyhow::Error> {
        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::FileWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("source_name", DataType::Utf8, false),
            Field::new("source_url", DataType::Utf8, true),
            Field::new("source_trust_score", DataType::Float64, false),
            Field::new("source_category", DataType::Utf8, false),
            Field::new("score", DataType::Float64, true),
        ]));

        let path = if offset == 0 {
            output_path.to_string()
        } else {
            format!(
                "{}.part{}",
                output_path,
                offset / self.batch_config.max_batch_size as u64
            )
        };

        let id_array = StringArray::from(
            batch.iter().map(|s| s.id.to_string()).collect::<Vec<_>>(),
        );
        let text_array = StringArray::from(
            batch.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        );
        let source_name_array = StringArray::from(
            batch.iter().map(|s| s.source.name.as_str()).collect::<Vec<_>>(),
        );
        let source_url_array = StringArray::from(
            batch.iter().map(|s| s.source.url.as_deref().unwrap_or("")).collect::<Vec<_>>(),
        );
        let trust_score_array = Float64Array::from(
            batch.iter().map(|s| s.source.trust_score).collect::<Vec<_>>(),
        );
        let source_category_array = StringArray::from(
            batch.iter().map(|s| format!("{:?}", s.source.category)).collect::<Vec<_>>(),
        );
        let score_array = Float64Array::from(
            batch.iter().map(|s| s.score.unwrap_or(f64::NAN)).collect::<Vec<_>>(),
        );

        let record_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(text_array),
                Arc::new(source_name_array),
                Arc::new(source_url_array),
                Arc::new(trust_score_array),
                Arc::new(source_category_array),
                Arc::new(score_array),
            ],
        )?;

        let file = std::fs::File::create(&path)?;
        let mut writer = FileWriter::try_new(file, &record_batch.schema())?;
        writer.write(&record_batch)?;
        writer.finish()?;
        Ok(())
    }

    /// Write samples in Apache Arrow IPC format — `arrow` feature required
    #[cfg(not(feature = "arrow"))]
    async fn write_arrow(
        &self,
        _batch: &[DataSample],
        _output_path: &str,
        _offset: u64,
    ) -> Result<(), anyhow::Error> {
        Err(anyhow::anyhow!(
            "Arrow output format requires the 'arrow' feature to be enabled"
        ))
    }

    /// Write samples as length-prefixed binary records (TensorRecords format)
    async fn write_tensor_records(
        &self,
        batch: &[DataSample],
        output_path: &str,
        offset: u64,
    ) -> Result<(), anyhow::Error> {
        use tokio::io::AsyncWriteExt;
        let path = if offset == 0 {
            output_path.to_string()
        } else {
            format!(
                "{}.part{}",
                output_path,
                offset / self.batch_config.max_batch_size as u64
            )
        };

        let data = self.zero_copy_batch(batch);
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(&data).await?;
        Ok(())
    }

    fn zero_copy_batch(&self, samples: &[DataSample]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(samples.len() * 1024);
        for sample in samples {
            if let Ok(json) = serde_json::to_vec(sample) {
                let len = json.len() as u32;
                buffer.extend_from_slice(&len.to_le_bytes());
                buffer.extend_from_slice(&json);
            }
        }
        buffer
    }

    /// Deliver a batch of accepted samples to disk (used by Pipeline::run)
    pub async fn deliver_batch(&self, samples: &[DataSample]) -> Result<u64, anyhow::Error> {
        let count = samples.len() as u64;
        if samples.is_empty() {
            return Ok(0);
        }
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let filename = format!("data_{}.json", timestamp);
        tokio::fs::create_dir_all(&self.output_dir).await?;
        let path = self.output_dir.join(&filename);

        let mut content = String::with_capacity(samples.len() * 1024);
        for sample in samples {
            if let Ok(line) = serde_json::to_string(sample) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        tokio::fs::write(&path, content.as_bytes()).await?;

        info!("Delivered batch of {} samples to {}", count, path.display());
        Ok(count)
    }

    pub async fn distributed_push(
        &self,
        mut rx: mpsc::Receiver<Vec<DataSample>>,
        endpoints: Vec<String>,
    ) -> Result<u64, anyhow::Error> {
        let mut total = 0u64;
        let retry_config = RetryConfig::new(3, 500);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        while let Some(batch) = rx.recv().await {
            let payload = self.zero_copy_batch(&batch);
            for endpoint in &endpoints {
                let ep = endpoint.clone();
                let payload = payload.clone();
                let client = client.clone();
                let _ = retry_config
                    .retry(|| {
                        let ep = ep.clone();
                        let payload = payload.clone();
                        let client = client.clone();
                        async move {
                            let resp = client
                                .post(&ep)
                                .header("content-type", "application/octet-stream")
                                .body(payload)
                                .send()
                                .await
                                .map_err(|e| format!("request failed: {}", e))?;
                            if !resp.status().is_success() {
                                return Err(format!(
                                    "push to {} returned status {}",
                                    ep,
                                    resp.status()
                                ));
                            }
                            Ok::<_, String>(())
                        }
                    })
                    .await;
            }
            total += 1;
            if total % 100 == 0 {
                sleep(Duration::from_millis(1)).await;
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSample, SampleStats, SourceInfo, SourceCategory};
    use uuid::Uuid;

    fn sample(text: &str) -> DataSample {
        DataSample {
            id: Uuid::new_v4(),
            text: text.into(),
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
        }
    }

    #[test]
    fn test_default_format() {
        let layer = TrainingDeliveryLayer::default();
        assert_eq!(layer.output_format, OutputFormat::JsonLines);
        assert_eq!(layer.output_dir, PathBuf::from("output"));
    }

    #[test]
    fn test_with_format() {
        let layer = TrainingDeliveryLayer::new()
            .with_format(OutputFormat::RawText);
        assert_eq!(layer.output_format, OutputFormat::RawText);
    }

    #[test]
    fn test_with_output_dir() {
        let layer = TrainingDeliveryLayer::new()
            .with_output_dir(PathBuf::from("/tmp/out"));
        assert_eq!(layer.output_dir, PathBuf::from("/tmp/out"));
    }

    #[test]
    fn test_output_format_debug_and_clone() {
        let a = OutputFormat::Arrow;
        let b = a;
        assert_eq!(a, b);
        assert_eq!(format!("{:?}", OutputFormat::JsonLines), "JsonLines");
    }

    #[test]
    fn test_zero_copy_batch() {
        let layer = TrainingDeliveryLayer::default();
        let samples = vec![sample("hello")];
        let data = layer.zero_copy_batch(&samples);
        assert!(!data.is_empty());

        // First 4 bytes are the length prefix
        let len = u32::from_le_bytes(data[0..4].try_into().unwrap());
        assert!(len > 0);
    }

    #[test]
    fn test_zero_copy_batch_empty() {
        let layer = TrainingDeliveryLayer::default();
        let data = layer.zero_copy_batch(&[]);
        assert!(data.is_empty());
    }

    #[tokio::test]
    async fn test_deliver_batch_empty_returns_zero() {
        let layer = TrainingDeliveryLayer::default();
        let result = layer.deliver_batch(&[]).await;
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_output_format_partial_eq() {
        assert_eq!(OutputFormat::JsonLines, OutputFormat::JsonLines);
        assert_ne!(OutputFormat::JsonLines, OutputFormat::TensorRecords);
    }
}
