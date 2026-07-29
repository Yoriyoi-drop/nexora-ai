use crate::multimodal::error::Result;
use ndarray::ArrayD;
use rand::Rng;

fn softmax_flat(scores: &mut [f32]) {
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; num_kv];
    for i in 0..num_q {
        for j in 0..num_kv {
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = s * scale;
        }
        softmax_flat(&mut scores);
        for d in 0..head_dim {
            let mut o = 0.0f32;
            for j in 0..num_kv {
                o += scores[j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = o;
        }
    }
}

pub struct QuerySet {
    queries: Vec<QueryToken>,
    hidden_dim: usize,
    query_type: String,
    attention_weights: Option<ArrayD<f32>>,
    _wq: Vec<Vec<f32>>,
    _wk: Vec<Vec<f32>>,
    _wv: Vec<Vec<f32>>,
}

impl QuerySet {
    pub fn new(num_queries: usize, hidden_dim: usize, query_type: String) -> Result<Self> {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        let mut queries = Vec::new();
        for i in 0..num_queries {
            queries.push(QueryToken::new(i, hidden_dim)?);
        }

        let wq = (0..num_queries)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        let wk = (0..num_queries)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        let wv = (0..num_queries)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();

        Ok(Self {
            queries,
            hidden_dim,
            query_type,
            attention_weights: None,
            _wq: wq,
            _wk: wk,
            _wv: wv,
        })
    }

    pub fn forward(&mut self, inputs: &[ArrayD<f32>]) -> Result<ArrayD<f32>> {
        if inputs.is_empty() {
            return Err(crate::multimodal::error::CaffeineError::qformer(
                "No inputs provided for query processing",
            ));
        }

        let mut query_embeddings = vec![0.0f32; self.queries.len() * self.hidden_dim];
        for (i, query) in self.queries.iter().enumerate() {
            for d in 0..self.hidden_dim {
                query_embeddings[i * self.hidden_dim + d] = query.embedding[d];
            }
        }

        for input in inputs {
            self.process_input(input, &mut query_embeddings)?;
        }

        let attended_queries = self.apply_query_attention(&query_embeddings)?;

        let weights = self.compute_query_attention_weights(&query_embeddings)?;
        self.attention_weights = Some(weights);

        let shape = vec![1, self.queries.len(), self.hidden_dim];
        Ok(ArrayD::from_shape_vec(shape, attended_queries)?)
    }

    fn process_input(&self, input: &ArrayD<f32>, query_embeddings: &mut [f32]) -> Result<()> {
        let input_shape = input.shape();
        let input_dim = input_shape.iter().product::<usize>();
        let num_input_tokens = if self.hidden_dim > 0 { input_dim / self.hidden_dim } else { 0 };

        if num_input_tokens == 0 {
            return Ok(());
        }

        let num_queries = self.queries.len();
        let head_dim = self.hidden_dim;

        let mut q = vec![0.0f32; num_queries * head_dim];
        for i in 0..num_queries {
            for d in 0..head_dim {
                q[i * head_dim + d] = query_embeddings[i * head_dim + d];
            }
        }

        let mut k = vec![0.0f32; num_input_tokens * head_dim];
        let mut v = vec![0.0f32; num_input_tokens * head_dim];
        for i in 0..num_input_tokens {
            for d in 0..head_dim {
                let idx = i * head_dim + d;
                k[i * head_dim + d] = input.get([idx]).copied().unwrap_or(0.0);
                v[i * head_dim + d] = input.get([idx]).copied().unwrap_or(0.0);
            }
        }

        let mut out = vec![0.0f32; num_queries * head_dim];
        scaled_dot_product_attention(&q, &k, &v, num_queries, num_input_tokens, head_dim, &mut out);

        for i in 0..num_queries {
            for d in 0..head_dim {
                query_embeddings[i * head_dim + d] += out[i * head_dim + d];
            }
        }

        Ok(())
    }

    fn apply_query_attention(&self, query_embeddings: &[f32]) -> Result<Vec<f32>> {
        let num_queries = self.queries.len();
        let head_dim = self.hidden_dim;
        let mut attended = vec![0.0f32; query_embeddings.len()];

        scaled_dot_product_attention(
            query_embeddings,
            query_embeddings,
            query_embeddings,
            num_queries,
            num_queries,
            head_dim,
            &mut attended,
        );

        Ok(attended)
    }

    fn compute_query_attention_weights(&self, query_embeddings: &[f32]) -> Result<ArrayD<f32>> {
        let num_queries = self.queries.len();
        let mut attention_weights = vec![0.0f32; num_queries * num_queries];

        for i in 0..num_queries {
            for j in 0..num_queries {
                let mut similarity = 0.0f32;
                for d in 0..self.hidden_dim {
                    let idx_i = i * self.hidden_dim + d;
                    let idx_j = j * self.hidden_dim + d;
                    if idx_i < query_embeddings.len() && idx_j < query_embeddings.len() {
                        similarity += query_embeddings[idx_i] * query_embeddings[idx_j];
                    }
                }
                attention_weights[i * num_queries + j] = similarity;
            }
        }

        let shape = vec![num_queries, num_queries];
        Ok(ArrayD::from_shape_vec(shape, attention_weights)?)
    }

    pub fn get_embeddings(&self) -> Vec<Vec<f32>> {
        self.queries.iter().map(|q| q.embedding.clone()).collect()
    }

    pub fn query_type(&self) -> &str {
        &self.query_type
    }

    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }
}

#[derive(Debug, Clone)]
pub struct QueryToken {
    _id: usize,
    embedding: Vec<f32>,
    position_encoding: Vec<f32>,
}

impl QueryToken {
    pub fn new(id: usize, hidden_dim: usize) -> Result<Self> {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        let embedding: Vec<f32> = (0..hidden_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
            .collect();

        let mut position_encoding = vec![0.0f32; hidden_dim];
        for d in 0..hidden_dim {
            if d % 2 == 0 {
                position_encoding[d] =
                    (id as f32 / 10000.0_f32.powf(d as f32 / hidden_dim as f32)).sin();
            } else {
                position_encoding[d] =
                    (id as f32 / 10000.0_f32.powf(d as f32 / hidden_dim as f32)).cos();
            }
        }

        Ok(Self {
            _id: id,
            embedding,
            position_encoding,
        })
    }

    pub fn update_embedding(&mut self, new_embedding: Vec<f32>) -> Result<()> {
        if new_embedding.len() != self.embedding.len() {
            return Err(crate::multimodal::error::CaffeineError::qformer(
                "Embedding dimension mismatch",
            ));
        }
        self.embedding = new_embedding;
        Ok(())
    }

    pub fn get_positional_embedding(&self) -> Vec<f32> {
        self.embedding
            .iter()
            .zip(self.position_encoding.iter())
            .map(|(e, p)| e + p)
            .collect()
    }
}

pub struct QueryProcessor {
    hidden_dim: usize,
    num_heads: usize,
    _dropout_rate: f32,
    _wq: Vec<Vec<f32>>,
    _wk: Vec<Vec<f32>>,
    _wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
}

impl QueryProcessor {
    pub fn new(hidden_dim: usize, num_heads: usize, _dropout_rate: f32) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        let wq = (0..hidden_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        let wk = (0..hidden_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        let wv = (0..hidden_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        let wo = (0..hidden_dim)
            .map(|_| (0..hidden_dim).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect())
            .collect();

        Self {
            hidden_dim,
            num_heads,
            _dropout_rate,
            _wq: wq,
            _wk: wk,
            _wv: wv,
            wo,
        }
    }

    pub fn process_queries(&self, queries: &[f32], num_queries: usize) -> Result<Vec<f32>> {
        let head_dim = self.hidden_dim / self.num_heads;
        let mut q = vec![0.0f32; num_queries * self.hidden_dim];
        let mut k = vec![0.0f32; num_queries * self.hidden_dim];
        let mut v = vec![0.0f32; num_queries * self.hidden_dim];

        for i in 0..num_queries {
            for d in 0..self.hidden_dim {
                q[i * self.hidden_dim + d] = queries[i * self.hidden_dim + d];
                k[i * self.hidden_dim + d] = queries[i * self.hidden_dim + d];
                v[i * self.hidden_dim + d] = queries[i * self.hidden_dim + d];
            }
        }

        let mut out = vec![0.0f32; num_queries * self.hidden_dim];

        for h in 0..self.num_heads {
            let hd = h * head_dim;
            let mut hq = vec![0.0f32; num_queries * head_dim];
            let mut hk = vec![0.0f32; num_queries * head_dim];
            let mut hv = vec![0.0f32; num_queries * head_dim];

            for i in 0..num_queries {
                for d in 0..head_dim {
                    hq[i * head_dim + d] = q[i * self.hidden_dim + hd + d];
                    hk[i * head_dim + d] = k[i * self.hidden_dim + hd + d];
                    hv[i * head_dim + d] = v[i * self.hidden_dim + hd + d];
                }
            }

            let mut hout = vec![0.0f32; num_queries * head_dim];
            scaled_dot_product_attention(&hq, &hk, &hv, num_queries, num_queries, head_dim, &mut hout);

            for i in 0..num_queries {
                for d in 0..head_dim {
                    out[i * self.hidden_dim + hd + d] = hout[i * head_dim + d];
                }
            }
        }

        let mut output = vec![0.0f32; queries.len()];
        for i in 0..num_queries {
            for d in 0..self.hidden_dim {
                let mut s = 0.0f32;
                for kk in 0..self.hidden_dim {
                    s += out[i * self.hidden_dim + kk] * self.wo[kk][d];
                }
                output[i * self.hidden_dim + d] = s;
            }
        }

        Ok(output)
    }

    pub fn layer_norm(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        let mean = inputs.iter().sum::<f32>() / inputs.len() as f32;
        let variance = inputs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / inputs.len() as f32;
        let std_dev = variance.sqrt();
        if std_dev == 0.0 {
            return Ok(inputs.to_vec());
        }
        Ok(inputs.iter().map(|x| (x - mean) / std_dev).collect())
    }
}
