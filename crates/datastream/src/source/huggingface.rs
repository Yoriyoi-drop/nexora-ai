use async_trait::async_trait;
use std::collections::HashMap;
use tracing::info;

use crate::source::SourceProvider;
use crate::types::{DataSample, SampleStats, SourceCategory};
use nexora_common::retry::RetryConfig;
use uuid::Uuid;

fn client() -> Option<reqwest::Client> {
    match reqwest::Client::builder()
        .user_agent("Nexora-DataStream/1.0")
        .timeout(std::time::Duration::from_secs(60))
        .build()
    {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!("Failed to build HTTP client: {}", e);
            None
        }
    }
}

async fn fetch_with_retry(
    client: &reqwest::Client,
    url: &str,
) -> Result<reqwest::Response, String> {
    let config = RetryConfig::new(3, 500);
    let url = url.to_string();
    config
        .retry(|| async {
            let resp = client
                .get(&url)
                .send()
                .await
                .map_err(|e| format!("Request failed: {}", e))?;

            let status = resp.status();

            if status.is_success() {
                return Ok(resp);
            }

            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(format!("404 Not Found: {}", url));
            }

            if status == reqwest::StatusCode::NOT_IMPLEMENTED {
                return Err(format!("501 Not Implemented: {}", url));
            }

            if status.is_server_error() {
                return Err(format!("Server error {}: {}", status, url));
            }

            Err(format!("HTTP {}: {}", status, url))
        })
        .await
}

async fn fetch_json(client: &reqwest::Client, url: &str) -> Result<serde_json::Value, String> {
    let resp = fetch_with_retry(client, url).await?;
    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| format!("JSON parse failed: {}", e))
}

/// Auto-detect available configs untuk dataset dari HF Datasets Server API.
/// Contoh: wikitext → ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]
async fn resolve_config_names(client: &reqwest::Client, dataset: &str) -> Vec<String> {
    let url = format!(
        "https://datasets-server.huggingface.co/configs?dataset={}",
        dataset
    );
    match fetch_json(client, &url).await {
        Ok(json) => json["configs"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|c| c["config"]["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        Err(e) => {
            tracing::warn!("HF configs fetch failed for '{}': {}", dataset, e);
            vec![]
        }
    }
}

/// Provider yang fetch dataset dari Hugging Face via Datasets Server API
///
/// API: https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}
///
/// Dataset populer: wikitext, tiny_shakespeare, dair-ai/emotion, ag_news, imdb, cnn_dailymail
pub struct HuggingFaceDatasetProvider {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub max_samples: usize,
}

impl HuggingFaceDatasetProvider {
    pub fn new(dataset: &str, max_samples: usize) -> Self {
        Self {
            dataset: dataset.to_string(),
            config: "default".to_string(),
            split: "train".to_string(),
            max_samples,
        }
    }

    pub fn with_split(mut self, split: &str) -> Self {
        self.split = split.to_string();
        self
    }

    pub fn with_config(mut self, config: &str) -> Self {
        self.config = config.to_string();
        self
    }

    /// Resolve dataset config secara otomatis.
    /// Jika config "default" tidak ditemukan, cari config pertama yang tersedia.
    pub async fn resolve_config(&mut self) {
        let http = match client() {
            Some(c) => c,
            None => return,
        };

        let configs = resolve_config_names(&http, &self.dataset).await;
        if configs.is_empty() {
            return;
        }

        if configs.contains(&self.config) {
            return;
        }

        if let Some(first) = configs.first() {
            tracing::info!(
                "HF: auto-resolved config for '{}': '{}' (was '{}')",
                self.dataset,
                first,
                self.config
            );
            self.config = first.clone();
        }
    }
}

#[async_trait]
impl SourceProvider for HuggingFaceDatasetProvider {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn url(&self) -> &str {
        "https://huggingface.co/datasets"
    }

    fn category(&self) -> SourceCategory {
        SourceCategory::HuggingFace
    }

    fn default_trust_score(&self) -> f64 {
        0.90
    }

    fn description(&self) -> &str {
        "Hugging Face Datasets via Datasets Server API"
    }

    fn sample_data(&self) -> Vec<String> {
        vec![]
    }

    async fn fetch_samples(&self) -> Vec<DataSample> {
        let http = match client() {
            Some(c) => c,
            None => return vec![],
        };
        let source = self.source_info();
        let mut samples: Vec<DataSample> = Vec::new();
        let limit = 100.min(self.max_samples);
        let mut offset: usize = 0;
        let mut current_config = self.config.clone();

        loop {
            if samples.len() >= self.max_samples {
                break;
            }
            let remaining = self.max_samples - samples.len();
            let length = limit.min(remaining);

            let url = format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
                self.dataset, current_config, self.split, offset, length
            );

            let json: serde_json::Value = match fetch_json(&http, &url).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("HF request failed at offset {}: {}", offset, e);

                    let configs = resolve_config_names(&http, &self.dataset).await;
                    if let Some(first) = configs.first() {
                        if *first != current_config {
                            tracing::info!(
                                "HF fallback config '{}' -> '{}'",
                                current_config,
                                first
                            );
                            current_config = first.clone();
                            continue;
                        }
                    }

                    break;
                }
            };

            let rows = match json["rows"].as_array() {
                Some(arr) if !arr.is_empty() => arr,
                _ => break,
            };

            let features: Vec<String> = json["features"]
                .as_array()
                .map(|f| {
                    f.iter()
                        .filter_map(|col| col["name"].as_str().map(|s| s.to_string()))
                        .filter(|name| {
                            !name.starts_with('_')
                                && name != "id"
                                && name != "idx"
                                && name != "label"
                        })
                        .collect()
                })
                .unwrap_or_else(|| vec!["text".to_string()]);

            for row in rows {
                let row_data = &row["row"];
                let text: String = features
                    .iter()
                    .filter_map(|f| row_data[f].as_str())
                    .collect::<Vec<&str>>()
                    .join(" ");

                if text.len() < 10 {
                    continue;
                }

                let metadata: HashMap<String, String> = row_data
                    .as_object()
                    .map(|obj| {
                        obj.iter()
                            .filter(|(k, _)| !features.contains(k))
                            .map(|(k, v)| {
                                let val = v.as_str().map(|s| s.to_string())
                                    .or_else(|| v.as_i64().map(|i| i.to_string()))
                                    .or_else(|| v.as_f64().map(|f| f.to_string()))
                                    .unwrap_or_default();
                                (k.clone(), val)
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                samples.push(DataSample {
                    id: Uuid::new_v4(),
                    text,
                    token_ids: None,
                    metadata,
                    source: source.clone(),
                    stats: SampleStats::default(),
                    domains: vec![],
                    score: None,
                    curriculum_level: None,
                });

                if samples.len() >= self.max_samples {
                    break;
                }
            }

            offset += length;
        }

        info!(
            "Fetched {} samples from HuggingFace dataset '{}' (config={}, split={})",
            samples.len(),
            self.dataset,
            current_config,
            self.split
        );
        samples
    }
}
