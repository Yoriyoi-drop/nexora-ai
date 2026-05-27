use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const REASONING_TYPES: [&str; 6] = [
    "deductive",
    "inductive",
    "abductive",
    "analogical",
    "causal",
    "analytical",
];

const HIDDEN: usize = 32;

static CLASSIFIER: OnceLock<ReasoningClassifier> = OnceLock::new();

pub struct ReasoningClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl ReasoningClassifier {
    pub fn global() -> &'static Self {
        CLASSIFIER.get_or_init(|| Self {
            embed_table: Array2::zeros((1, 1)),
            w1: Array2::zeros((1, 1)),
            b1: Array1::zeros(1),
            w2: Array2::zeros((1, 1)),
            b2: Array1::zeros(1),
        })
    }

    pub fn init(embed_table: Array2<f32>) {
        let hidden_size = embed_table.shape()[1];
        let _ = CLASSIFIER.set(Self {
            w1: rand_init(hidden_size, HIDDEN),
            b1: Array1::zeros(HIDDEN),
            w2: rand_init(HIDDEN, REASONING_TYPES.len()),
            b2: Array1::zeros(REASONING_TYPES.len()),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    fn embed_average(&self, token_ids: &[u32]) -> Array1<f32> {
        let embed_dim = self.embed_table.shape()[1];
        let vocab = self.embed_table.shape()[0];
        if token_ids.is_empty() {
            return Array1::zeros(embed_dim);
        }
        let mut avg = Array1::zeros(embed_dim);
        let mut count = 0usize;
        for &tid in token_ids {
            let idx = (tid as usize).min(vocab.saturating_sub(1));
            let row = self.embed_table.row(idx);
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

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("analytical".to_string(), 1.0)];
        }

        let h = gelu(&(self.embed_average(token_ids).dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = softmax(&logits);

        let mut results: Vec<_> = REASONING_TYPES
            .iter()
            .zip(probs.iter())
            .map(|(c, p)| ((*c).to_string(), *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

fn rand_init(rows: usize, cols: usize) -> Array2<f32> {
    let scale = (1.0 / rows as f32).sqrt();
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array2::from_shape_simple_fn((rows, cols), || rng.gen::<f32>() * 2.0 * scale - scale)
}

fn gelu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 0.5 * v * (1.0 + (v * 0.7978845608028654 * (1.0 + 0.044715 * v * v)).tanh()))
}

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_iter(exps.into_iter().map(|e| e / sum))
}

const REASONING_PROMPTS: &[(&str, &str)] = &[
    ("deductive", "Apply deductive reasoning from general principles to specific conclusions."),
    ("inductive", "Use inductive reasoning to derive general patterns from specific observations."),
    ("abductive", "Apply abductive reasoning to infer the most likely explanation."),
    ("analogical", "Use analogical reasoning by comparing with known similar cases."),
    ("causal", "Apply causal reasoning to identify cause-effect relationships."),
    ("analytical", "Use systematic analytical reasoning breaking down the problem into components."),
];

pub fn reasoning_prompt(reasoning_type: &str) -> &'static str {
    REASONING_PROMPTS
        .iter()
        .find(|(r, _)| *r == reasoning_type)
        .map(|(_, p)| *p)
        .unwrap_or("Use systematic analytical reasoning breaking down the problem into components.")
}

pub fn detect_reasoning_type(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let classifier = CLASSIFIER.get().unwrap_or_else(ReasoningClassifier::global);
    if token_ids.is_empty() || !classifier.is_initialized() {
        return vec![("analytical".to_string(), 1.0)];
    }
    classifier.predict(token_ids)
}
