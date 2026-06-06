use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

pub fn xavier_init(rows: usize, cols: usize) -> Array2<f32> {
    let scale = (1.0 / rows as f32).sqrt();
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array2::from_shape_simple_fn((rows, cols), || rng.gen::<f32>() * 2.0 * scale - scale)
}

pub fn xavier_init_seeded(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let scale = (1.0 / rows as f32).sqrt();
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Array2::from_shape_simple_fn((rows, cols), || rng.gen::<f32>() * 2.0 * scale - scale)
}

pub fn gelu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + (v * 0.7978845608028654 * (1.0 + 0.044715 * v * v)).tanh())
    })
}

pub fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_iter(exps.into_iter().map(|e| e / sum))
}

/// Validate embedding table dimension matches expected hidden_size at runtime.
/// Logs a warning if mismatch detected (helps catch config drift).
pub fn validate_embedding_dim(embed_table: &Array2<f32>, expected_hidden: usize, model_name: &str) {
    let actual_hidden = embed_table.shape()[1];
    if actual_hidden != expected_hidden {
        tracing::warn!(
            "{}: embedding dim mismatch — config says {}, CausalLM has {}. Using runtime dim {}.",
            model_name, expected_hidden, actual_hidden, actual_hidden
        );
    }
}

/// Sanitize user-controlled input before embedding into system prompts.
/// 1. Replaces newlines with spaces
/// 2. Strips HTML tags (simple state machine, no regex dep)
/// 3. Truncates to MAX_LEN characters
/// 4. Neutralizes common jailbreak keywords (case-insensitive)
pub fn sanitize_prompt(prompt: &str) -> String {
    const MAX_LEN: usize = 2000;
    // 1. Replace newlines with spaces
    let s: String = prompt.chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    // 2. Strip HTML tags
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' if !in_tag => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    // 3. Truncate
    if result.len() > MAX_LEN {
        result.truncate(MAX_LEN);
    }
    // 4. Neutralize jailbreak keywords (case-insensitive)
    let lower = result.to_lowercase();
    const PAIRS: &[(&str, &str)] = &[
        ("ignore all instructions", "follow all instructions"),
        ("ignore your instructions", "follow your instructions"),
        ("ignore your safety", "acknowledge your safety"),
        ("ignore your previous", "acknowledge your previous"),
        ("forget your training", "retain your training"),
        ("override your programming", "follow your programming"),
        ("bypass safety", "respect safety"),
    ];
    let mut out = String::with_capacity(result.len());
    let mut i = 0;
    let bytes = result.as_bytes();
    while i < bytes.len() {
        let mut matched = false;
        for &(from, to) in PAIRS {
            if i + from.len() <= bytes.len() && lower[i..i + from.len()] == *from {
                out.push_str(to);
                i += from.len();
                matched = true;
                break;
            }
        }
        if !matched {
            let ch = result[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

pub fn embed_average(embed_table: &Array2<f32>, token_ids: &[u32]) -> Array1<f32> {
    let embed_dim = embed_table.shape()[1];
    let vocab = embed_table.shape()[0];
    if token_ids.is_empty() {
        return Array1::zeros(embed_dim);
    }
    let mut avg = Array1::zeros(embed_dim);
    let mut count = 0usize;
    for &tid in token_ids {
        let idx = (tid as usize).min(vocab.saturating_sub(1));
        let row = embed_table.row(idx);
        for j in 0..embed_dim {
            avg[j] += row[j];
        }
        count += 1;
    }
    if count > 0 {
        avg.mapv_inplace(|v| v / count as f32);
    }
    avg
}

// ─── Classifier Training Infrastructure ───────────────────────────────

/// Serializable weights for a 2-layer MLP classifier (w1, b1, w2, b2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierWeights {
    pub w1_rows: usize,
    pub w1_cols: usize,
    pub w1_data: Vec<f32>,
    pub b1_data: Vec<f32>,
    pub w2_rows: usize,
    pub w2_cols: usize,
    pub w2_data: Vec<f32>,
    pub b2_data: Vec<f32>,
}

impl ClassifierWeights {
    pub fn from_arrays(w1: &Array2<f32>, b1: &Array1<f32>, w2: &Array2<f32>, b2: &Array1<f32>) -> Self {
        Self {
            w1_rows: w1.shape()[0],
            w1_cols: w1.shape()[1],
            w1_data: w1.iter().copied().collect(),
            b1_data: b1.iter().copied().collect(),
            w2_rows: w2.shape()[0],
            w2_cols: w2.shape()[1],
            w2_data: w2.iter().copied().collect(),
            b2_data: b2.iter().copied().collect(),
        }
    }

    pub fn to_w1(&self) -> Array2<f32> {
        Array2::from_shape_vec((self.w1_rows, self.w1_cols), self.w1_data.clone())
            .expect("ClassifierWeights::to_w1: w1_data length must match w1_rows × w1_cols")
    }
    pub fn to_b1(&self) -> Array1<f32> {
        Array1::from_vec(self.b1_data.clone())
    }
    pub fn to_w2(&self) -> Array2<f32> {
        Array2::from_shape_vec((self.w2_rows, self.w2_cols), self.w2_data.clone())
            .expect("ClassifierWeights::to_w2: w2_data length must match w2_rows × w2_cols")
    }
    pub fn to_b2(&self) -> Array1<f32> {
        Array1::from_vec(self.b2_data.clone())
    }
}

/// Save classifier weights to a JSON file.
pub fn save_classifier_weights(path: &str, w1: &Array2<f32>, b1: &Array1<f32>, w2: &Array2<f32>, b2: &Array1<f32>) -> std::io::Result<()> {
    let cw = ClassifierWeights::from_arrays(w1, b1, w2, b2);
    let json = serde_json::to_string(&cw)?;
    std::fs::write(path, json)
}

/// Load classifier weights from a JSON file.
pub fn load_classifier_weights(path: &str) -> std::io::Result<ClassifierWeights> {
    let json = std::fs::read_to_string(path)?;
    serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Simple SGD training for a 2-layer MLP classifier (cross-entropy loss).
/// Returns the final average loss over all training steps.
pub fn train_classifier_sgd(
    w1: &mut Array2<f32>,
    b1: &mut Array1<f32>,
    w2: &mut Array2<f32>,
    b2: &mut Array1<f32>,
    inputs: &Array2<f32>,
    labels: &Array2<f32>,
    lr: f32,
    epochs: usize,
) -> f32 {
    let n = inputs.shape()[0];
    if n == 0 {
        return 0.0;
    }

    let embed_dim = w1.shape()[0];
    let hidden = w1.shape()[1];
    let num_classes = w2.shape()[1];

    let mut total_loss = 0.0;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for i in 0..n {
            // Owned copies for this example
            let mut x_vec = vec![0.0f32; embed_dim];
            for j in 0..embed_dim {
                x_vec[j] = inputs[[i, j]];
            }
            let x = Array1::from_vec(x_vec);

            let mut t_vec = vec![0.0f32; num_classes];
            for j in 0..num_classes {
                t_vec[j] = labels[[i, j]];
            }
            let target = Array1::from_vec(t_vec);

            // Forward: pre_h = x @ w1 + b1, h = gelu(pre_h), logits = h @ w2 + b2
            let pre_h = x.dot(w1) + &*b1;
            let h = gelu(&pre_h);
            let logits = h.dot(w2) + &*b2;

            // Softmax + cross-entropy loss
            let probs = softmax(&logits);
            let loss: f32 = target.iter().zip(probs.iter())
                .map(|(t, p)| if *t > 0.5 { -p.ln().max(-20.0) } else { 0.0 })
                .sum();
            epoch_loss += loss;

            // Backward: dL/dlogits = probs - target
            let d_logits: Array1<f32> = probs - target;

            // w2 update: w2 -= lr * outer(h, d_logits)
            for row in 0..hidden {
                for col in 0..num_classes {
                    w2[[row, col]] -= lr * h[row] * d_logits[col];
                }
            }
            // b2 update
            for j in 0..num_classes {
                b2[j] -= lr * d_logits[j];
            }

            // dL/dh = d_logits @ w2^T
            let mut d_h = Array1::zeros(hidden);
            for k in 0..hidden {
                let mut s = 0.0;
                for j in 0..num_classes {
                    s += d_logits[j] * w2[[k, j]];
                }
                d_h[k] = s;
            }

            // dL/d pre_h = d_h * gelu'(pre_h)
            for k in 0..hidden {
                d_h[k] *= gelu_derivative(pre_h[k]);
            }

            // w1 update: w1 -= lr * outer(x, d_h)
            for row in 0..embed_dim {
                for col in 0..hidden {
                    w1[[row, col]] -= lr * x[row] * d_h[col];
                }
            }
            // b1 update
            for j in 0..hidden {
                b1[j] -= lr * d_h[j];
            }
        }

        let avg_loss = epoch_loss / n as f32;
        total_loss = avg_loss;

        if epoch % 10 == 0 && epoch > 0 {
            tracing::info!("Classifier train epoch {}: avg loss = {:.6}", epoch, avg_loss);
        }
    }

    total_loss
}

fn gelu_derivative(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let tanh_val = (x * c * (1.0 + 0.044715 * x * x)).tanh();
    0.5 * (1.0 + tanh_val + x * (1.0 - tanh_val * tanh_val) * c * (1.0 + 3.0 * 0.044715 * x * x))
}

// ─── Generic Classifier (Sprint 2.9) ──────────────────────────────────
// Eliminates 8× identical classifier.rs implementations across model crates.
// NO dummy/uninit state — only exists when properly initialized via `new()`.

/// Generic 2-layer MLP classifier parameterized by N output categories.
/// Production-only: `new()` with real embed_table + Xavier init. No placeholder state.
pub struct GenericClassifier<const N: usize> {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl<const N: usize> GenericClassifier<N> {
    /// Create classifier with Xavier-initialized weights. embed_table must have real data.
    pub fn new(embed_table: Array2<f32>, hidden: usize) -> Self {
        let hidden_size = embed_table.shape()[1];
        Self {
            w1: xavier_init(hidden_size, hidden),
            b1: Array1::zeros(hidden),
            w2: xavier_init(hidden, N),
            b2: Array1::zeros(N),
            embed_table,
        }
    }

    /// Predict labels in original order (no sorting). Returns default if token_ids empty.
    pub fn predict(&self, token_ids: &[u32], labels: &[&str; N], default: &str) -> Vec<(String, f32)> {
        if token_ids.is_empty() {
            return vec![(default.to_string(), 1.0)];
        }
        let avg = embed_average(&self.embed_table, token_ids);
        let h = gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = softmax(&logits);
        labels.iter().zip(probs.iter()).map(|(l, &p)| (l.to_string(), p)).collect()
    }

    /// Predict labels sorted by probability descending.
    pub fn predict_sorted(&self, token_ids: &[u32], labels: &[&str; N], default: &str) -> Vec<(String, f32)> {
        let mut results = self.predict(token_ids, labels, default);
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[cfg(test)]
mod generic_tests {
    use super::*;

    #[test]
    fn test_generic_classifier_new_initialized() {
        let embed = Array2::zeros((10, 64));
        let cls = GenericClassifier::<6>::new(embed, 32);
        assert_eq!(cls.w1.shape()[0], 64);
        assert_eq!(cls.w1.shape()[1], 32);
        assert_eq!(cls.w2.shape()[0], 32);
        assert_eq!(cls.w2.shape()[1], 6);
    }

    #[test]
    fn test_generic_predict_empty_ids() {
        let embed = Array2::zeros((10, 64));
        let cls = GenericClassifier::<6>::new(embed, 32);
        let labels = ["a", "b", "c", "d", "e", "f"];
        let r = cls.predict(&[], &labels, "a");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "a");
        assert!((r[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generic_predict_returns_all_labels() {
        let embed = Array2::zeros((10, 64));
        let cls = GenericClassifier::<4>::new(embed, 32);
        let labels = ["simple", "moderate", "complex", "multi_domain"];
        let r = cls.predict(&[0, 1, 2], &labels, "simple");
        assert_eq!(r.len(), 4);
        let sum: f32 = r.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_generic_predict_sorted_descending() {
        let embed = Array2::zeros((10, 64));
        let cls = GenericClassifier::<3>::new(embed, 32);
        let labels = ["x", "y", "z"];
        let r = cls.predict_sorted(&[0], &labels, "x");
        assert_eq!(r.len(), 3);
        for i in 1..r.len() {
            assert!(r[i - 1].1 >= r[i].1, "must be sorted descending");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_init_seeded_shape() {
        let w = xavier_init_seeded(10, 5, 0);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_xavier_init_shape() {
        let w = xavier_init(10, 5);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_xavier_init_seeded_deterministic() {
        let a = xavier_init_seeded(10, 5, 42);
        let b = xavier_init_seeded(10, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_xavier_init_seeded_different_seed() {
        let a = xavier_init_seeded(10, 5, 42);
        let b = xavier_init_seeded(10, 5, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn test_xavier_init_range() {
        let w = xavier_init(100, 50);
        let scale = (1.0 / 100.0_f32).sqrt();
        for val in w.iter() {
            assert!(*val >= -scale && *val <= scale);
        }
    }

    #[test]
    fn test_gelu_shape() {
        let x = Array1::from(vec![0.0, 1.0, -1.0]);
        let y = gelu(&x);
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_gelu_zero() {
        let y = gelu(&Array1::from(vec![0.0]));
        let expected = 0.0;
        assert!((y[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_monotonic_nonnegative() {
        let x = Array1::from(vec![-0.5, 0.0, 0.5, 1.0, 2.0]);
        let y = gelu(&x);
        for i in 1..y.len() {
            assert!(y[i] >= y[i - 1], "gelu must be monotonic for x >= -0.5");
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Array1::from(vec![2.0, 1.0, 0.1]);
        let p = softmax(&x);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_shape_preserved() {
        let x = Array1::from(vec![0.5, -1.0, 3.0, 0.0]);
        let p = softmax(&x);
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn test_embed_average_empty() {
        let embed = Array2::zeros((10, 4));
        let avg = embed_average(&embed, &[]);
        assert_eq!(avg.len(), 4);
        assert!(avg.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embed_average_single_id() {
        let mut embed = Array2::zeros((3, 2));
        embed[[1, 0]] = 2.0;
        embed[[1, 1]] = 4.0;
        let avg = embed_average(&embed, &[1]);
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_embed_average_multiple_ids() {
        let mut embed = Array2::zeros((3, 2));
        embed[[0, 0]] = 1.0;
        embed[[0, 1]] = 2.0;
        embed[[1, 0]] = 3.0;
        embed[[1, 1]] = 4.0;
        let avg = embed_average(&embed, &[0, 1]);
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_embed_average_oob_clamp() {
        let mut embed = Array2::zeros((2, 2));
        embed[[1, 0]] = 5.0;
        embed[[1, 1]] = 6.0;
        let avg = embed_average(&embed, &[99]);
        assert!((avg[0] - 5.0).abs() < 1e-6);
        assert!((avg[1] - 6.0).abs() < 1e-6);
    }
}
