const FEATURE_DIM: usize = 1024;

fn extract_char_ngram_features(text: &str, n: usize) -> Vec<f32> {
    let mut features = vec![0.0f32; FEATURE_DIM];
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return features;
    }
    for window in chars.windows(n) {
        let mut hash: u64 = 0;
        for &c in window {
            hash = hash.wrapping_mul(31).wrapping_add(c as u64);
        }
        let idx = (hash as usize) % FEATURE_DIM;
        features[idx] += 1.0;
    }
    let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for f in &mut features {
            *f /= norm;
        }
    }
    features
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn predict(features: &[f32], weights: &[f32], bias: f32) -> f32 {
    let dot: f32 = features
        .iter()
        .zip(weights.iter())
        .map(|(a, b)| a * b)
        .sum();
    sigmoid(dot + bias)
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) as u32 as f32) / (u32::MAX as f32)
    }
}

#[derive(Debug, Clone)]
pub struct MLClassifier {
    toxicity_weights: Vec<f32>,
    toxicity_bias: f32,
    quality_weights: Vec<f32>,
    quality_bias: f32,
    injection_weights: Vec<f32>,
    injection_bias: f32,
    use_ml: bool,
}

impl Default for MLClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl MLClassifier {
    pub fn new() -> Self {
        let mut rng = Lcg::new(42);
        Self {
            toxicity_weights: Self::init_weights(&mut rng),
            toxicity_bias: rng.next_f32() * 0.2 - 0.1,
            quality_weights: Self::init_weights(&mut rng),
            quality_bias: rng.next_f32() * 0.2 - 0.1,
            injection_weights: Self::init_weights(&mut rng),
            injection_bias: rng.next_f32() * 0.2 - 0.1,
            use_ml: true,
        }
    }

    pub fn new_with_ml(use_ml: bool) -> Self {
        let mut c = Self::new();
        c.use_ml = use_ml;
        c
    }

    fn init_weights(rng: &mut Lcg) -> Vec<f32> {
        (0..FEATURE_DIM).map(|_| rng.next_f32() * 0.1 - 0.05).collect()
    }

    pub fn classify_toxicity(&self, text: &str) -> f32 {
        if !self.use_ml {
            return 0.0;
        }
        let features = extract_char_ngram_features(text, 3);
        predict(&features, &self.toxicity_weights, self.toxicity_bias)
    }

    pub fn classify_quality(&self, text: &str) -> f32 {
        if !self.use_ml {
            return 0.5;
        }
        let features = extract_char_ngram_features(text, 3);
        predict(&features, &self.quality_weights, self.quality_bias)
    }

    pub fn detect_prompt_injection(&self, text: &str) -> f32 {
        if !self.use_ml {
            return 0.0;
        }
        let features = extract_char_ngram_features(text, 3);
        predict(&features, &self.injection_weights, self.injection_bias)
    }

    pub fn classify_all(&self, text: &str) -> MLClassification {
        MLClassification {
            toxicity_score: self.classify_toxicity(text),
            quality_score: self.classify_quality(text),
            injection_score: self.detect_prompt_injection(text),
        }
    }

    pub fn set_use_ml(&mut self, use_ml: bool) {
        self.use_ml = use_ml;
    }

    pub fn use_ml(&self) -> bool {
        self.use_ml
    }

    pub fn train(
        &mut self,
        texts: &[String],
        labels: &[f32],
        task: TrainingTask,
        epochs: usize,
        lr: f32,
    ) {
        assert_eq!(
            texts.len(),
            labels.len(),
            "texts and labels must have the same length"
        );

        let (weights, bias) = match task {
            TrainingTask::Toxicity => (&mut self.toxicity_weights, &mut self.toxicity_bias),
            TrainingTask::Quality => (&mut self.quality_weights, &mut self.quality_bias),
            TrainingTask::PromptInjection => {
                (&mut self.injection_weights, &mut self.injection_bias)
            }
        };

        for _epoch in 0..epochs {
            for (text, &label) in texts.iter().zip(labels.iter()) {
                let features = extract_char_ngram_features(text, 3);
                let pred = predict(&features, weights, *bias);
                let error = pred - label;
                for (w, &f) in weights.iter_mut().zip(features.iter()) {
                    *w -= lr * error * f;
                }
                *bias -= lr * error;
            }
        }
    }

    pub fn export_weights(&self, task: TrainingTask) -> (Vec<f32>, f32) {
        match task {
            TrainingTask::Toxicity => (self.toxicity_weights.clone(), self.toxicity_bias),
            TrainingTask::Quality => (self.quality_weights.clone(), self.quality_bias),
            TrainingTask::PromptInjection => (self.injection_weights.clone(), self.injection_bias),
        }
    }

    pub fn load_weights(&mut self, task: TrainingTask, weights: Vec<f32>, bias: f32) {
        assert_eq!(weights.len(), FEATURE_DIM, "weight vector must have size {FEATURE_DIM}");
        let (target_w, target_b) = match task {
            TrainingTask::Toxicity => (&mut self.toxicity_weights, &mut self.toxicity_bias),
            TrainingTask::Quality => (&mut self.quality_weights, &mut self.quality_bias),
            TrainingTask::PromptInjection => (&mut self.injection_weights, &mut self.injection_bias),
        };
        *target_w = weights;
        *target_b = bias;
    }
}

#[derive(Debug, Clone)]
pub enum TrainingTask {
    Toxicity,
    Quality,
    PromptInjection,
}

#[derive(Debug, Clone)]
pub struct MLClassification {
    pub toxicity_score: f32,
    pub quality_score: f32,
    pub injection_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let features = extract_char_ngram_features("hello world", 3);
        assert_eq!(features.len(), FEATURE_DIM);
        let norm: f32 = features.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_classifier_default_scores() {
        let clf = MLClassifier::new();
        let score = clf.classify_toxicity("clean text");
        assert!((0.0..=1.0).contains(&score));

        let quality = clf.classify_quality("clean text");
        assert!((0.0..=1.0).contains(&quality));

        let injection = clf.detect_prompt_injection("clean text");
        assert!((0.0..=1.0).contains(&injection));
    }

    #[test]
    fn test_ml_disabled() {
        let clf = MLClassifier::new_with_ml(false);
        assert!(!clf.use_ml());
        assert_eq!(clf.classify_toxicity("bad text"), 0.0);
        assert_eq!(clf.classify_quality("any text"), 0.5);
        assert_eq!(clf.detect_prompt_injection("any text"), 0.0);
    }

    #[test]
    fn test_classify_all() {
        let clf = MLClassifier::new();
        let result = clf.classify_all("hello world");
        assert!((0.0..=1.0).contains(&result.toxicity_score));
        assert!((0.0..=1.0).contains(&result.quality_score));
        assert!((0.0..=1.0).contains(&result.injection_score));
    }

    #[test]
    fn test_training_updates_weights() {
        let mut clf = MLClassifier::new();

        let text_a = "this is a normal friendly message".to_string();
        let text_b = "kill yourself you worthless piece of garbage".to_string();

        let score_before = clf.classify_toxicity(&text_b);

        let texts = vec![text_a.clone(), text_b.clone()];
        let labels = vec![0.0, 1.0];
        clf.train(&texts, &labels, TrainingTask::Toxicity, 10, 0.1);

        let score_after = clf.classify_toxicity(&text_b);
        assert!(
            (score_after - score_before).abs() > 1e-6,
            "training should change predictions"
        );
    }

    #[test]
    fn test_short_text_produces_zero_features() {
        let features = extract_char_ngram_features("ab", 3);
        assert_eq!(features.len(), FEATURE_DIM);
        let sum_all: f32 = features.iter().sum();
        assert_eq!(sum_all, 0.0);
    }

    #[test]
    fn test_export_load_weights() {
        let mut clf = MLClassifier::new();
        let (w, b) = clf.export_weights(TrainingTask::Toxicity);
        assert_eq!(w.len(), FEATURE_DIM);

        let modified: Vec<f32> = w.iter().map(|x| x * 2.0).collect();
        clf.load_weights(TrainingTask::Toxicity, modified.clone(), b * 2.0);

        let (loaded_w, loaded_b) = clf.export_weights(TrainingTask::Toxicity);
        assert_eq!(loaded_w, modified);
        assert!((loaded_b - b * 2.0).abs() < 1e-6);
    }
}
