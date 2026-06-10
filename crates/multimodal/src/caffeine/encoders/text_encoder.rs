use crate::caffeine::error::Result;
use crate::caffeine::types::*;
use ndarray::{Array2, ArrayD};
use rand::Rng;

/// FNV-1a 32-bit hash — deterministic across runs and processes.
fn fnv1a_hash(input: &str) -> u32 {
    const FNV_OFFSET_BASIS: u32 = 2166136261;
    const FNV_PRIME: u32 = 16777619;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in input.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

fn xavier_init(rows: usize, cols: usize) -> Result<Array2<f32>> {
    let scale = (2.0 / (rows as f32 + cols as f32)).sqrt();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
        .collect();
    Ok(Array2::from_shape_vec((rows, cols), data)?)
}

/// Learned token embedding table with Xavier init
struct TokenEmbedding {
    weight: Array2<f32>,
}

impl TokenEmbedding {
    fn new(vocab_size: usize, embed_dim: usize) -> Result<Self> {
        Ok(Self {
            weight: xavier_init(vocab_size, embed_dim)?,
        })
    }

    fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        let seq_len = token_ids.len();
        let embed_dim = self.weight.shape()[1];
        let mut output = Array2::zeros((seq_len, embed_dim));
        let vocab_size = self.weight.shape()[0];
        for (i, &id) in token_ids.iter().enumerate() {
            let id = id.min(vocab_size - 1);
            for d in 0..embed_dim {
                output[[i, d]] = self.weight[[id, d]];
            }
        }
        output
    }
}

/// 2-layer MLP with GELU activation for text FFN
struct TextFFN {
    pub(crate) fc1: Array2<f32>,
    pub(crate) fc2: Array2<f32>,
}

impl TextFFN {
    fn new(embed_dim: usize, hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            fc1: xavier_init(embed_dim, hidden_dim)?,
            fc2: xavier_init(hidden_dim, embed_dim)?,
        })
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = x.dot(&self.fc1);
        let gelu_h = h.mapv(gelu);
        gelu_h.dot(&self.fc2)
    }
}

/// Text encoder based on BERT
pub struct TextEncoder {
    config: crate::caffeine::config::TextEncoderConfig,
    model_loaded: bool,
    vocab_size: usize,
    max_position_embeddings: usize,
    token_embedding: TokenEmbedding,
    ffn_layers: Vec<TextFFN>,
    /// Q/K/V projection weight: [embed_dim, embed_dim]
    qkv_proj: Array2<f32>,
}

impl TextEncoder {
    /// Create new text encoder
    pub fn new(config: crate::caffeine::config::TextEncoderConfig) -> Result<Self> {
        let embed_dim = config.output_dim;
        let hidden_dim = embed_dim * 4;
        let ffn_layers = (0..6)
            .map(|_| TextFFN::new(embed_dim, hidden_dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vocab_size: config.vocab_size,
            max_position_embeddings: 512,
            model_loaded: false,
            token_embedding: TokenEmbedding::new(config.vocab_size, embed_dim)?,
            ffn_layers,
            qkv_proj: xavier_init(embed_dim, embed_dim)?,
            config,
        })
    }

    /// Load model weights
    pub fn load_model(&mut self) -> Result<()> {
        self.model_loaded = true;
        Ok(())
    }

    /// Encode text input
    pub fn encode(&mut self, input: &TextInput) -> Result<ArrayD<f32>> {
        if !self.model_loaded {
            self.load_model()?;
        }

        let tokens = self.tokenize(&input.text)?;
        let token_ids = self.tokens_to_ids(&tokens)?;
        let encoded = self.encode_tokens(&token_ids)?;

        Ok(encoded)
    }

    /// Tokenize text
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        tokens.push("[CLS]".to_string());

        for word in text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
            if !clean_word.is_empty() {
                tokens.push(clean_word.to_lowercase());
            }
        }

        tokens.push("[SEP]".to_string());

        if tokens.len() > self.max_position_embeddings {
            tokens.truncate(self.max_position_embeddings - 1);
            tokens.push("[SEP]".to_string());
        }

        Ok(tokens)
    }

    /// Convert tokens to IDs
    fn tokens_to_ids(&self, tokens: &[String]) -> Result<Vec<usize>> {
        let mut ids = Vec::new();
        for token in tokens {
            ids.push(self.token_to_id(token)?);
        }
        Ok(ids)
    }

    fn token_to_id(&self, token: &str) -> Result<usize> {
        let hash = fnv1a_hash(token);
        let id = (hash as usize) % self.vocab_size;
        Ok(id)
    }

    /// Encode token IDs with BERT layers
    fn encode_tokens(&self, token_ids: &[usize]) -> Result<ArrayD<f32>> {
        let seq_len = token_ids.len();
        let embed_dim = self.config.output_dim;
        let num_heads = 8;
        let head_dim = embed_dim / num_heads;

        // Learned token embedding [seq_len, embed_dim]
        let mut hidden = self.token_embedding.forward(token_ids);

        // Add sinusoidal position encoding
        for pos in 0..seq_len {
            for d in 0..embed_dim {
                let pos_val = if d % 2 == 0 {
                    (pos as f32 / 10000.0_f32.powf(d as f32 / embed_dim as f32)).sin()
                } else {
                    (pos as f32 / 10000.0_f32.powf(d as f32 / embed_dim as f32)).cos()
                };
                hidden[[pos, d]] += pos_val;
            }
        }

        // Apply 6 transformer layers
        for layer_idx in 0..6 {
            // Multi-head attention with pre-norm
            let normed = layer_norm(&hidden, embed_dim);
            let attn_out = self.multi_head_attention(&normed, seq_len, embed_dim, num_heads, head_dim)?;
            hidden = &hidden + &attn_out;

            // FFN with pre-norm
            let normed = layer_norm(&hidden, embed_dim);
            let ff_out = self.ffn_layers[layer_idx].forward(&normed);
            hidden = &hidden + &ff_out;
        }

        let shape = vec![1, seq_len, embed_dim];
        Ok(ArrayD::from_shape_vec(shape, hidden.into_raw_vec_and_offset().0)?)
    }

    /// Multi-head self-attention
    fn multi_head_attention(
        &self,
        hidden: &Array2<f32>,
        seq_len: usize,
        embed_dim: usize,
        num_heads: usize,
        _head_dim: usize,
    ) -> Result<Array2<f32>> {
        let scale = (embed_dim as f32 / num_heads as f32).sqrt().recip();

        // Learned Q/K/V projection: [embed_dim, embed_dim]
        let proj_weight = self.qkv_proj.view();
        let q = hidden.dot(&proj_weight);
        let k = hidden.dot(&proj_weight);
        let v = hidden.dot(&proj_weight);

        let head_size = embed_dim / num_heads;
        let mut context = Array2::zeros((seq_len, embed_dim));

        for h in 0..num_heads {
            let h_start = h * head_size;
            let h_end = h_start + head_size;

            let mut scores = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0;
                    for d in h_start..h_end {
                        dot += q[[i, d]] * k[[j, d]];
                    }
                    scores[[i, j]] = dot * scale;
                }
            }

            // Causal mask
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
            }

            let weights = softmax_2d(&scores);

            for i in 0..seq_len {
                for d in h_start..h_end {
                    let mut val = 0.0;
                    for j in 0..seq_len {
                        val += weights[[i, j]] * v[[j, d]];
                    }
                    context[[i, d]] = val;
                }
            }
        }

        Ok(context)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get configuration
    pub fn config(&self) -> &crate::caffeine::config::TextEncoderConfig {
        &self.config
    }

    /// Collect all trainable weights for checkpoint
    pub(crate) fn collect_weights(&self) -> Vec<(String, ndarray::ArrayD<f32>)> {
        let mut weights = Vec::new();
        weights.push(("text_encoder.token_embed.weight".to_string(), self.token_embedding.weight.clone().into_dyn()));
        weights.push(("text_encoder.qkv_proj".to_string(), self.qkv_proj.clone().into_dyn()));
        for (i, ffn) in self.ffn_layers.iter().enumerate() {
            weights.push((format!("text_encoder.ffn_{}.fc1", i), ffn.fc1.clone().into_dyn()));
            weights.push((format!("text_encoder.ffn_{}.fc2", i), ffn.fc2.clone().into_dyn()));
        }
        weights
    }
}

fn layer_norm(x: &Array2<f32>, _dim: usize) -> Array2<f32> {
    let mut output = x.clone();
    for mut row in output.rows_mut() {
        let len = row.len() as f32;
        let sum: f32 = row.iter().sum();
        let mean = sum / len;
        let var_sum: f32 = row.iter().map(|v| (*v - mean).powi(2)).sum();
        let std = (var_sum / len + 1e-6).sqrt();
        for val in row.iter_mut() {
            *val = (*val - mean) / std;
        }
    }
    output
}

fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = x.dim();
    let mut output = Array2::zeros((rows, cols));
    for i in 0..rows {
        let mut max = f32::NEG_INFINITY;
        for j in 0..cols {
            if x[[i, j]] > max {
                max = x[[i, j]];
            }
        }
        let mut sum = 0.0;
        for j in 0..cols {
            let val = (x[[i, j]] - max).exp();
            output[[i, j]] = val;
            sum += val;
        }
        if sum > 0.0 {
            for j in 0..cols {
                output[[i, j]] /= sum;
            }
        }
    }
    output
}

/// Multi-lingual text processor
#[allow(dead_code)]
pub struct MultiLingualProcessor {
    supported_languages: Vec<String>,
    language_detectors: std::collections::HashMap<String, f32>,
}

impl MultiLingualProcessor {
    /// Create new multi-lingual processor
    pub fn new() -> Self {
        let supported_languages = vec![
            "en".to_string(),
            "id".to_string(),
            "zh".to_string(),
            "es".to_string(),
            "fr".to_string(),
            "de".to_string(),
            "ja".to_string(),
            "ko".to_string(),
            "ar".to_string(),
        ];

        let mut language_detectors = std::collections::HashMap::new();
        for lang in &supported_languages {
            language_detectors.insert(lang.clone(), 0.0);
        }

        Self {
            supported_languages,
            language_detectors,
        }
    }

    /// Detect language of text
    pub fn detect_language(&mut self, text: &str) -> Result<String> {
        let id_score =
            text.chars().filter(|c| *c >= 'a' && *c <= 'z').count() as f32 / text.len() as f32;
        let zh_score = text
            .chars()
            .filter(|c| (*c as u32) >= 0x4E00 && (*c as u32) <= 0x9FFF)
            .count() as f32
            / text.len() as f32;

        self.language_detectors.insert("id".to_string(), id_score);
        self.language_detectors.insert("zh".to_string(), zh_score);

        let mut best_lang = "en".to_string();
        let mut best_score = 0.0;

        for (lang, score) in &self.language_detectors {
            if *score > best_score {
                best_score = *score;
                best_lang = lang.clone();
            }
        }

        Ok(best_lang)
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> &[String] {
        &self.supported_languages
    }
}
