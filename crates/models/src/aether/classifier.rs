use ndarray::{Array1, Array2};
use std::sync::OnceLock;

const NUM_EMOTIONS: usize = 8;
const HIDDEN: usize = 64;

static CLASSIFIER: OnceLock<EmotionClassifier> = OnceLock::new();

pub struct EmotionClassifier {
    embed_table: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl EmotionClassifier {
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
            w2: rand_init(HIDDEN, NUM_EMOTIONS),
            b2: Array1::zeros(NUM_EMOTIONS),
            embed_table,
        });
    }

    pub fn is_initialized(&self) -> bool {
        self.embed_table.shape()[0] > 1
    }

    pub fn predict(&self, token_ids: &[u32]) -> Vec<(String, f32)> {
        if token_ids.is_empty() || !self.is_initialized() {
            return vec![("neutral".to_string(), 1.0)];
        }

        let embed_dim = self.embed_table.shape()[1];
        let vocab = self.embed_table.shape()[0];

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

        let h = gelu(&(avg.dot(&self.w1) + &self.b1));
        let logits = h.dot(&self.w2) + &self.b2;
        let probs = softmax(&logits);

        EMOTION_LABELS.iter().zip(probs.iter()).map(|(label, &p)| (label.to_string(), p)).collect()
    }
}

fn rand_init(rows: usize, cols: usize) -> Array2<f32> {
    let scale = (1.0 / rows as f32).sqrt();
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Array2::from_shape_simple_fn((rows, cols), || rng.gen::<f32>() * 2.0 * scale - scale)
}

fn gelu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + (v * 0.7978845608028654 * (1.0 + 0.044715 * v * v)).tanh())
    })
}

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    Array1::from_iter(exps.into_iter().map(|e| e / sum))
}

const EMOTION_LABELS: [&str; NUM_EMOTIONS] = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "neutral",
];

pub fn detect_emotions(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let clf = CLASSIFIER.get().unwrap_or_else(EmotionClassifier::global);
    if token_ids.is_empty() {
        return vec![("neutral".to_string(), 1.0)];
    }
    if !clf.is_initialized() {
        return vec![("neutral".to_string(), 1.0)];
    }
    clf.predict(token_ids)
}
