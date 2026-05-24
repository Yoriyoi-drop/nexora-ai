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
        .timeout(std::time::Duration::from_secs(30))
        .build()
    {
        Ok(client) => Some(client),
        Err(e) => {
            tracing::warn!("Failed to build HTTP client: {}", e);
            None
        }
    }
}

async fn fetch_with_retry(client: &reqwest::Client, url: &str) -> Result<reqwest::Response, String> {
    let config = RetryConfig::new(3, 200);
    let url = url.to_string();
    config
        .retry(|| async {
            client
                .get(&url)
                .send()
                .await
                .map_err(|e| format!("Request failed: {}", e))
                .and_then(|r| r.error_for_status().map_err(|e| format!("HTTP error: {}", e)))
        })
        .await
}

async fn fetch_json_with_retry(client: &reqwest::Client, url: &str) -> Result<serde_json::Value, String> {
    let resp = fetch_with_retry(client, url).await?;
    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| format!("JSON parse failed: {}", e))
}

/// Fetch HackerNews top stories via Firebase API (no auth required).
pub struct HackerNewsProvider;

#[async_trait]
impl SourceProvider for HackerNewsProvider {
    fn name(&self) -> &str {
        "hackernews"
    }
    fn url(&self) -> &str {
        "https://news.ycombinator.com"
    }
    fn category(&self) -> SourceCategory {
        SourceCategory::News
    }
    fn default_trust_score(&self) -> f64 {
        0.70
    }
    fn description(&self) -> &str {
        "Hacker News top stories and discussions"
    }
    fn sample_data(&self) -> Vec<String> {
        vec![]
    }

    async fn fetch_samples(&self) -> Vec<DataSample> {
        let client = match client() {
            Some(c) => c,
            None => return vec![],
        };
        let source = self.source_info();

        let ids: Vec<u64> = match fetch_json_with_retry(&client, "https://hacker-news.firebaseio.com/v0/topstories.json").await {
            Ok(serde_json::Value::Array(arr)) => arr.iter().filter_map(|v| v.as_u64()).collect(),
            Ok(_) => return vec![],
            Err(e) => {
                tracing::warn!("HN: request failed after retries: {}", e);
                return vec![];
            }
        };

        let mut samples = Vec::with_capacity(ids.len().min(100));
        for &id in ids.iter().take(100) {
            let url = format!("https://hacker-news.firebaseio.com/v0/item/{}.json", id);
            let item: serde_json::Value = match fetch_json_with_retry(&client, &url).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("HN: item {} failed after retries: {}", id, e);
                    continue;
                }
            };

            let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
            let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let by = item
                .get("by")
                .and_then(|b| b.as_str())
                .unwrap_or("anonymous");
            let content = if text.is_empty() {
                format!("[HN] {} (by {})", title, by)
            } else {
                format!("[HN] {} - {} (by {})", title, text, by)
            };
            samples.push(DataSample {
                id: Uuid::new_v4(),
                text: content,
                token_ids: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("hn_id".into(), id.to_string());
                    m.insert("by".into(), by.to_string());
                    m
                },
                source: source.clone(),
                stats: SampleStats::default(),
                domains: vec![],
                score: None,
                curriculum_level: None,
            });
        }

        info!("Fetched {} samples from HackerNews", samples.len());
        samples
    }
}

/// Fetch recent Wikipedia articles via the Wikimedia API (no auth required).
pub struct WikipediaProvider;

#[async_trait]
impl SourceProvider for WikipediaProvider {
    fn name(&self) -> &str {
        "wikipedia"
    }
    fn url(&self) -> &str {
        "https://en.wikipedia.org"
    }
    fn category(&self) -> SourceCategory {
        SourceCategory::Wikipedia
    }
    fn default_trust_score(&self) -> f64 {
        0.95
    }
    fn description(&self) -> &str {
        "Wikipedia featured and random articles"
    }
    fn sample_data(&self) -> Vec<String> {
        vec![]
    }

    async fn fetch_samples(&self) -> Vec<DataSample> {
        let client = match client() {
            Some(c) => c,
            None => return vec![],
        };
        let source = self.source_info();
        let mut samples = Vec::new();

        for _ in 0..10 {
            let url = "https://en.wikipedia.org/w/api.php?action=query&list=random&rnlimit=1&rnnamespace=0&format=json".to_string();
            let json: serde_json::Value = match fetch_json_with_retry(&client, &url).await {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(pages) = json["query"]["random"].as_array() {
                for page in pages {
                    if let Some(title) = page["title"].as_str() {
                        if let Some(extract) = fetch_wikipedia_extract(&client, title).await {
                            samples.push(DataSample {
                                id: Uuid::new_v4(),
                                text: format!("[Wikipedia] {} - {}", title, extract),
                                token_ids: None,
                                metadata: {
                                    let mut m = HashMap::new();
                                    m.insert("title".into(), title.to_string());
                                    m
                                },
                                source: source.clone(),
                                stats: SampleStats::default(),
                                domains: vec![],
                                score: None,
                                curriculum_level: None,
                            });
                        }
                    }
                }
            }
        }

        info!("Fetched {} samples from Wikipedia", samples.len());
        samples
    }
}

async fn fetch_wikipedia_extract(client: &reqwest::Client, title: &str) -> Option<String> {
    let url = format!(
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={}&format=json",
        urlencoding(title)
    );
    let json: serde_json::Value = match fetch_json_with_retry(client, &url).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("Wikipedia extract fetch failed: {}", e);
            return None;
        }
    };

    let pages = json["query"]["pages"].as_object()?;
    let extract = pages.values().next()?.get("extract")?.as_str()?.to_string();
    Some(extract)
}

fn urlencoding(s: &str) -> String {
    s.chars().map(|c| match c {
        ' ' => "%20".to_string(),
        '&' => "%26".to_string(),
        '?' => "%3F".to_string(),
        '=' => "%3D".to_string(),
        '#' => "%23".to_string(),
        '%' => "%25".to_string(),
        _ => c.to_string(),
    }).collect()
}

/// Fetch recent Reddit posts from technology subreddits (no auth required).
pub struct RedditProvider {
    pub subreddits: Vec<String>,
    pub limit: usize,
}

impl Default for RedditProvider {
    fn default() -> Self {
        Self {
            subreddits: vec![
                "technology".into(),
                "MachineLearning".into(),
                "rust".into(),
                "programming".into(),
                "artificial".into(),
                "science".into(),
            ],
            limit: 25,
        }
    }
}

#[async_trait]
impl SourceProvider for RedditProvider {
    fn name(&self) -> &str {
        "reddit"
    }
    fn url(&self) -> &str {
        "https://reddit.com"
    }
    fn category(&self) -> SourceCategory {
        SourceCategory::Reddit
    }
    fn default_trust_score(&self) -> f64 {
        0.55
    }
    fn description(&self) -> &str {
        "Reddit discussions from technology and science subreddits"
    }
    fn sample_data(&self) -> Vec<String> {
        vec![]
    }

    async fn fetch_samples(&self) -> Vec<DataSample> {
        let client = match reqwest::Client::builder()
            .user_agent("Nexora-DataStream/1.0 (by /u/nexora)")
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to build Reddit HTTP client: {}", e);
                return vec![];
            }
        };
        let source = self.source_info();
        let mut samples = Vec::new();

        for sub in &self.subreddits {
            let url = format!(
                "https://www.reddit.com/r/{}/top.json?limit={}",
                sub, self.limit
            );
            let json: serde_json::Value = match fetch_json_with_retry(&client, &url).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Reddit r/{}: request failed after retries: {}", sub, e);
                    continue;
                }
            };

            if let Some(children) = json["data"]["children"].as_array() {
                for child in children {
                    let data = &child["data"];
                    let title = data["title"].as_str().unwrap_or("");
                    let selftext = data["selftext"].as_str().unwrap_or("");
                    let author = data["author"].as_str().unwrap_or("anonymous");
                    let subreddit = data["subreddit"].as_str().unwrap_or(sub);
                    let score = data["score"].as_i64().unwrap_or(0);
                    let excerpt = if selftext.is_empty() || selftext.len() < 20 {
                        String::new()
                    } else if selftext.len() > 500 {
                        let end = selftext.char_indices().map(|(i, _)| i).nth(500).unwrap_or(selftext.len());
                        format!(" - {}", &selftext[..end])
                    } else {
                        format!(" - {}", selftext)
                    };
                    let content = format!(
                        "[r/{}] {} {} (by {}, +{})",
                        subreddit, title, excerpt, author, score
                    );
                    samples.push(DataSample {
                        id: Uuid::new_v4(),
                        text: content,
                        token_ids: None,
                        metadata: {
                            let mut m = HashMap::new();
                            m.insert("subreddit".into(), subreddit.to_string());
                            m.insert("author".into(), author.to_string());
                            m.insert("score".into(), score.to_string());
                            m
                        },
                        source: source.clone(),
                        stats: SampleStats::default(),
                        domains: vec![],
                        score: None,
                        curriculum_level: None,
                    });
                }
            }
        }

        info!("Fetched {} samples from Reddit", samples.len());
        samples
    }
}

pub fn build_registry() -> super::SourceRegistry {
    let mut reg = super::SourceRegistry::new();
    reg.register(Box::new(HackerNewsProvider));
    reg.register(Box::new(WikipediaProvider));
    reg.register(Box::new(RedditProvider::default()));
    reg
}
