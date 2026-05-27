use nexora_has_moe_ffn::Router;
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

pub const DOMAINS: [&str; 7] = [
    "math",
    "science",
    "code",
    "creative",
    "reasoning",
    "factual",
    "general",
];

static ROUTER: OnceLock<OmnisMoERouter> = OnceLock::new();

pub struct OmnisMoERouter {
    moe_router: Router,
    embed_table: Array2<f32>,
}

impl OmnisMoERouter {
    pub fn global() -> &'static Self {
        ROUTER.get_or_init(|| {
            Self {
                moe_router: Router::new(768, DOMAINS.len(), 2),
                embed_table: Array2::zeros((1, 768)),
            }
        })
    }

    pub fn init(embed_table: Array2<f32>) {
        let hidden_size = embed_table.shape()[1];
        ROUTER.set(Self {
            moe_router: Router::new(hidden_size, DOMAINS.len(), 2),
            embed_table,
        }).ok();
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
            return vec![("general".to_string(), 1.0)];
        }

        let avg = self.embed_average(token_ids);
        let input = avg.clone().into_shape((1, avg.len())).unwrap();
        let probs = self.moe_router.forward(&input);

        let mut results: Vec<_> = DOMAINS
            .iter()
            .enumerate()
            .map(|(i, d)| ((*d).to_string(), probs[[0, i]]))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

const EXPERT_PROMPTS: &[(&str, &str)] = &[
    ("math", "You are a mathematics expert. Solve step-by-step with clear reasoning."),
    ("science", "You are a science expert. Provide accurate, evidence-based explanations."),
    ("code", "You are a code expert. Write clean, idiomatic, well-documented code."),
    ("creative", "You are a creative writing expert. Be imaginative and engaging."),
    ("reasoning", "You are a logical reasoning expert. Analyze systematically from first principles."),
    ("factual", "You are a factual knowledge expert. Provide precise, well-sourced information."),
    ("general", "You are a general-purpose assistant. Respond helpfully and accurately."),
];

pub fn domain_system_prompt(domain: &str) -> &'static str {
    EXPERT_PROMPTS
        .iter()
        .find(|(d, _)| *d == domain)
        .map(|(_, p)| *p)
        .unwrap_or("You are a general-purpose assistant. Respond helpfully and accurately.")
}

pub fn detect_domains(text: &str, token_ids: &[u32]) -> Vec<(String, f32)> {
    let router = ROUTER.get().unwrap_or_else(OmnisMoERouter::global);
    if token_ids.is_empty() || !router.is_initialized() {
        return vec![("general".to_string(), 1.0)];
    }
    router.predict(token_ids)
}
